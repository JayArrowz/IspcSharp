using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// The unsigned widening multiply: <see cref="Spmd.MultiplyHighUnsigned"/> in kernel source,
/// <see cref="VInt.MultiplyHighUnsigned"/> in the vector companion, <c>vpmuludq</c> underneath.
///
/// The pairing that matters is (MultiplyHighUnsigned, operator *) — high and low halves of the
/// same 32x32 → 64 product — because that is what a counter-based RNG round needs and what
/// otherwise forces the generator to route through 64-bit lanes.
/// </summary>
public static partial class MulHiKernels
{
    /// <summary>Philox4x32's word-0 multiplier, a value with the top bit set.</summary>
    public const int M0 = unchecked((int)0xD2511F53);

    [Spmd]
    public static void MulHi(int[] a, int[] b, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = Spmd.MultiplyHighUnsigned(a[i], b[i]);
        }
    }

    /// <summary>
    /// Both halves of one widening multiply by a constant, the Philox round shape.
    /// </summary>
    [Spmd]
    public static void WideMulByConst(int[] a, int[] hi, int[] lo, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            hi[i] = Spmd.MultiplyHighUnsigned(a[i], M0);
            lo[i] = a[i] * M0;
        }
    }
}

public class MultiplyHighTests
{
    /// <summary>Values chosen to straddle the sign bit, where unsigned and signed disagree.</summary>
    private static readonly int[] Edge =
    [
        0, 1, 2, 0x7FFFFFFF, unchecked((int)0x80000000), unchecked((int)0xFFFFFFFF),
        unchecked((int)0xD2511F53), unchecked((int)0xCD9E8D57), unchecked((int)0x9E3779B9),
        -1, -2, int.MinValue, 65535, 65536, 0x00010001,
    ];

    private static int Reference(int a, int b) => unchecked((int)(((ulong)(uint)a * (uint)b) >> 32));

    [Fact]
    public void Scalar_MatchesWideningReference()
    {
        foreach (int a in Edge)
            foreach (int b in Edge)
                Assert.Equal(Reference(a, b), Spmd.MultiplyHighUnsigned(a, b));
    }

    [Fact]
    public void Scalar_IsUnsigned_NotSignExtended()
    {
        // -1 as unsigned is 0xFFFFFFFF; 0xFFFFFFFF^2 = 0xFFFFFFFE00000001, high word 0xFFFFFFFE.
        Assert.Equal(unchecked((int)0xFFFFFFFE), Spmd.MultiplyHighUnsigned(-1, -1));
        // A signed multiply-high would give 0 here instead.
        Assert.NotEqual(0, Spmd.MultiplyHighUnsigned(-1, -1));
    }

    [Fact]
    public void Vector_MatchesScalar_OnEdgeValues()
    {
        int w = VInt.LaneCount;
        foreach (int b in Edge)
        {
            for (int off = 0; off < Edge.Length; off += w)
            {
                int[] lhs = new int[w];
                for (int l = 0; l < w; l++)
                    lhs[l] = Edge[(off + l) % Edge.Length];

                VInt got = VInt.MultiplyHighUnsigned(VInt.Load(lhs, 0), new VInt(b));
                for (int l = 0; l < w; l++)
                    Assert.Equal(Reference(lhs[l], b), got.GetLane(l));
            }
        }
    }

    [Fact]
    public void Vector_MatchesScalar_OnRandomValues()
    {
        var rnd = new Random(20260816);
        int w = VInt.LaneCount;
        int[] a = new int[w];
        int[] b = new int[w];

        for (int t = 0; t < 20000; t++)
        {
            for (int l = 0; l < w; l++)
            {
                a[l] = rnd.Next(int.MinValue, int.MaxValue);
                b[l] = rnd.Next(int.MinValue, int.MaxValue);
            }

            VInt got = VInt.MultiplyHighUnsigned(VInt.Load(a, 0), VInt.Load(b, 0));
            for (int l = 0; l < w; l++)
                Assert.Equal(Reference(a[l], b[l]), got.GetLane(l));
        }
    }

    /// <summary>
    /// Every lane must get its own product. vpmuludq works on even lanes only, so a
    /// recombination bug shows up as odd lanes holding their neighbour's result — which a
    /// uniform-input test would not catch.
    /// </summary>
    [Fact]
    public void Vector_EachLaneIsIndependent()
    {
        int w = VInt.LaneCount;
        int[] a = new int[w];
        int[] b = new int[w];
        for (int l = 0; l < w; l++)
        {
            a[l] = unchecked((int)(0x9E3779B9u * (uint)(l + 1)));
            b[l] = unchecked((int)(0xD2511F53u ^ (uint)(l * 0x01010101)));
        }

        VInt got = VInt.MultiplyHighUnsigned(VInt.Load(a, 0), VInt.Load(b, 0));
        for (int l = 0; l < w; l++)
            Assert.Equal(Reference(a[l], b[l]), got.GetLane(l));
    }

    [Theory]
    [InlineData(64)]     // exact multiple of any lane count
    [InlineData(67)]     // leaves a scalar tail
    [InlineData(3)]      // shorter than the gang: tail only
    public void Kernel_MatchesScalar(int count)
    {
        var rnd = new Random(7);
        int[] a = new int[count];
        int[] b = new int[count];
        for (int i = 0; i < count; i++)
        {
            a[i] = rnd.Next(int.MinValue, int.MaxValue);
            b[i] = rnd.Next(int.MinValue, int.MaxValue);
        }

        int[] output = new int[count];
        MulHiKernels.MulHi_Simd(a, b, output, count);

        for (int i = 0; i < count; i++)
            Assert.Equal(Reference(a[i], b[i]), output[i]);
    }

    /// <summary>
    /// High and low halves together must reconstruct the exact 64-bit product — the property
    /// a Philox round depends on.
    /// </summary>
    [Theory]
    [InlineData(64)]
    [InlineData(67)]
    public void Kernel_WideningMultiply_ReconstructsFullProduct(int count)
    {
        var rnd = new Random(11);
        int[] a = new int[count];
        for (int i = 0; i < count; i++)
            a[i] = rnd.Next(int.MinValue, int.MaxValue);

        int[] hi = new int[count];
        int[] lo = new int[count];
        MulHiKernels.WideMulByConst_Simd(a, hi, lo, count);

        for (int i = 0; i < count; i++)
        {
            ulong expected = (ulong)(uint)a[i] * unchecked((uint)MulHiKernels.M0);
            ulong actual = ((ulong)(uint)hi[i] << 32) | (uint)lo[i];
            Assert.Equal(expected, actual);
        }
    }
}

/// <summary>
/// <see cref="VMask.ToBitmask"/>: the bridge from an execution mask to an integer whose bits
/// are lane positions. Every active lane must map to exactly its own bit.
/// </summary>
public class MaskBitmaskTests
{
    private static VMask FromPattern(uint pattern)
    {
        int w = VMask.LaneCount;
        int[] v = new int[w];
        for (int l = 0; l < w; l++)
            v[l] = (pattern & (1u << l)) != 0 ? 1 : 0;
        return VInt.Load(v, 0) > VInt.Zero;
    }

    [Fact]
    public void ToBitmask_RoundTripsEveryPattern()
    {
        int w = VMask.LaneCount;
        for (uint p = 0; p < (1u << w); p++)
            Assert.Equal((ulong)p, FromPattern(p).ToBitmask());
    }

    [Fact]
    public void ToBitmask_MatchesIsLaneActive()
    {
        int w = VMask.LaneCount;
        for (uint p = 0; p < (1u << w); p++)
        {
            VMask m = FromPattern(p);
            ulong bits = m.ToBitmask();
            for (int l = 0; l < w; l++)
                Assert.Equal(m.IsLaneActive(l), (bits & (1UL << l)) != 0);
        }
    }

    [Fact]
    public void ToBitmask_EdgeMasks()
    {
        int w = VMask.LaneCount;
        ulong all = w == 64 ? ulong.MaxValue : (1UL << w) - 1;
        Assert.Equal(all, VMask.All.ToBitmask());
        Assert.Equal(0UL, VMask.None.ToBitmask());
        for (int n = 0; n <= w; n++)
            Assert.Equal(n == 64 ? ulong.MaxValue : (1UL << n) - 1, VMask.FirstN(n).ToBitmask());
    }

    /// <summary>CountActive is now derived from ToBitmask; it must still agree lane by lane.</summary>
    [Fact]
    public void CountActive_MatchesPopCount()
    {
        int w = VMask.LaneCount;
        for (uint p = 0; p < (1u << w); p++)
            Assert.Equal(System.Numerics.BitOperations.PopCount(p), FromPattern(p).CountActive());
    }
}

/// <summary>
/// <c>ToBitmask</c> across the whole mask family. VMaskB is the one that needs all 64 bits: a
/// 512-bit gang is 64 byte lanes today, which is why the return type is <c>ulong</c> and not
/// <c>uint</c>.
/// </summary>
public class MaskBitmaskFamilyTests
{
    private static ulong Expected(int lanes, Func<int, bool> active)
    {
        ulong m = 0;
        for (int l = 0; l < lanes; l++)
            if (active(l)) m |= 1UL << l;
        return m;
    }

    [Fact]
    public void VMaskB_ToBitmask_MatchesLanes()
    {
        int w = VMaskB.LaneCount;
        byte[] v = new byte[w];
        for (int l = 0; l < w; l++) v[l] = (byte)(l % 3 == 0 ? 200 : 1);
        VMaskB m = VByte.Load(v) > new VByte(100);

        ulong bits = m.ToBitmask();
        Assert.Equal(Expected(w, l => l % 3 == 0), bits);
        for (int l = 0; l < w; l++)
            Assert.Equal(m.IsLaneActive(l), (bits & (1UL << l)) != 0);
        Assert.Equal(m.CountActive(), System.Numerics.BitOperations.PopCount(bits));
    }

    [Fact]
    public void VMaskS_ToBitmask_MatchesLanes()
    {
        int w = VMaskS.LaneCount;
        short[] v = new short[w];
        for (int l = 0; l < w; l++) v[l] = (short)(l % 4 == 1 ? 900 : -5);
        VMaskS m = VShort.Load(v) > new VShort(100);

        ulong bits = m.ToBitmask();
        Assert.Equal(Expected(w, l => l % 4 == 1), bits);
        for (int l = 0; l < w; l++)
            Assert.Equal(m.IsLaneActive(l), (bits & (1UL << l)) != 0);
        Assert.Equal(m.CountActive(), System.Numerics.BitOperations.PopCount(bits));
    }

    [Fact]
    public void VMaskD_ToBitmask_MatchesLanes()
    {
        int w = VMaskD.LaneCount;
        double[] v = new double[w];
        for (int l = 0; l < w; l++) v[l] = l % 2 == 0 ? 5.0 : -5.0;
        VMaskD m = VDouble.Load(v) > new VDouble(0.0);

        ulong bits = m.ToBitmask();
        Assert.Equal(Expected(w, l => l % 2 == 0), bits);
        for (int l = 0; l < w; l++)
            Assert.Equal(m.IsLaneActive(l), (bits & (1UL << l)) != 0);
    }

    [Fact]
    public void AllAndNone_AreSaturatedAndEmpty()
    {
        Assert.Equal(Expected(VMask.LaneCount, _ => true), VMask.All.ToBitmask());
        Assert.Equal(Expected(VMaskB.LaneCount, _ => true), VMaskB.All.ToBitmask());
        Assert.Equal(Expected(VMaskS.LaneCount, _ => true), VMaskS.All.ToBitmask());
        Assert.Equal(Expected(VMaskD.LaneCount, _ => true), VMaskD.All.ToBitmask());
        Assert.Equal(0UL, VMask.None.ToBitmask());
        Assert.Equal(0UL, VMaskB.None.ToBitmask());
        Assert.Equal(0UL, VMaskS.None.ToBitmask());
        Assert.Equal(0UL, VMaskD.None.ToBitmask());
    }

    /// <summary>Walking the mask by bit must visit exactly the active lanes, in order.</summary>
    [Fact]
    public void Bitmask_DrivesLaneWalk()
    {
        int w = VMaskB.LaneCount;
        byte[] v = new byte[w];
        for (int l = 0; l < w; l++) v[l] = (byte)(l % 5 == 2 ? 200 : 1);
        VMaskB m = VByte.Load(v) > new VByte(100);

        var walked = new System.Collections.Generic.List<int>();
        for (ulong bits = m.ToBitmask(); bits != 0; bits &= bits - 1)
            walked.Add(System.Numerics.BitOperations.TrailingZeroCount(bits));

        var expected = new System.Collections.Generic.List<int>();
        for (int l = 0; l < w; l++) if (m.IsLaneActive(l)) expected.Add(l);
        Assert.Equal(expected, walked);
    }
}
