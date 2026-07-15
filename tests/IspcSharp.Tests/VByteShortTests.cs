using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for the 8/16-bit varying types: VByte, VShort, VMaskB, VMaskS.
/// These exercise load/store, wrapping vs saturating arithmetic, unsigned byte
/// comparisons, shifts, widen/narrow bridges, reductions, and cross-lane shuffles.
/// </summary>
public class VByteShortTests
{
    private static byte[] MakeBytes(int n, Func<int, byte> gen)
    {
        byte[] a = new byte[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    private static short[] MakeShorts(int n, Func<int, short> gen)
    {
        short[] a = new short[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    [Fact]
    public void VByte_LoadStore_RoundTrips()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(i * 3));
        byte[] result = new byte[w];

        VByte.Load(data, 0).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void VByte_LoadMasked_HandlesTail()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(i + 1));
        byte[] result = new byte[w];

        VMaskB mask = VMaskB.FirstN(3);
        VByte.LoadMasked(data, 0, mask, fallback: 99).Store(result, 0);

        Assert.Equal(1, result[0]);
        Assert.Equal(2, result[1]);
        Assert.Equal(3, result[2]);
        for (int i = 3; i < w; i++)
            Assert.Equal(99, result[i]);
    }

    [Fact]
    public void VByte_StoreMasked_OnlyWritesActiveLanes()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(i + 10));
        byte[] result = MakeBytes(w, _ => 200);

        VMaskB mask = VMaskB.FirstN(2);
        VByte.Load(data, 0).StoreMasked(result, 0, mask);

        Assert.Equal(10, result[0]);
        Assert.Equal(11, result[1]);
        for (int i = 2; i < w; i++)
            Assert.Equal(200, result[i]);
    }

    [Fact]
    public void VByte_Add_WrapsLikeScalarCast()
    {
        int w = VByte.LaneCount;
        byte[] result = new byte[w];

        (new VByte(200) + new VByte(100)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(unchecked((byte)(200 + 100)), result[i]); // 44
    }

    [Fact]
    public void VByte_Multiply_WrapsPerLane()
    {
        int w = VByte.LaneCount;
        byte[] a = MakeBytes(w, i => (byte)(i + 17));
        byte[] result = new byte[w];

        (VByte.Load(a, 0) * 7).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(unchecked((byte)((i + 17) * 7)), result[i]);
    }

    [Fact]
    public void VByte_AddSaturate_ClampsTo255()
    {
        int w = VByte.LaneCount;
        byte[] a = MakeBytes(w, i => i % 2 == 0 ? (byte)200 : (byte)10);
        byte[] result = new byte[w];

        VByte.AddSaturate(VByte.Load(a, 0), new VByte(100)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(i % 2 == 0 ? (byte)255 : (byte)110, result[i]);
    }

    [Fact]
    public void VByte_SubtractSaturate_ClampsToZero()
    {
        int w = VByte.LaneCount;
        byte[] a = MakeBytes(w, i => i % 2 == 0 ? (byte)10 : (byte)200);
        byte[] result = new byte[w];

        VByte.SubtractSaturate(VByte.Load(a, 0), new VByte(50)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(i % 2 == 0 ? (byte)0 : (byte)150, result[i]);
    }

    [Fact]
    public void VByte_Average_RoundsUp()
    {
        int w = VByte.LaneCount;
        byte[] a = MakeBytes(w, i => (byte)i);
        byte[] b = MakeBytes(w, i => (byte)(i + 1));
        byte[] result = new byte[w];

        VByte.Average(VByte.Load(a, 0), VByte.Load(b, 0)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal((byte)((i + i + 1 + 1) >> 1), result[i]); // rounds up
    }

    [Fact]
    public void VByte_Comparisons_AreUnsigned()
    {
        // 0xC8 (200) vs 0x64 (100): a signed-byte compare would call 200 "negative" and
        // get this wrong; unsigned lanes must report 200 > 100.
        VMaskB gt = new VByte(200) > new VByte(100);
        VMaskB lt = new VByte(100) < new VByte(200);
        VMaskB wrongWay = new VByte(100) > new VByte(200);

        Assert.True(gt.AllActive());
        Assert.True(lt.AllActive());
        Assert.True(wrongWay.NoneActive());
    }

    [Fact]
    public void VByte_MinMax_PerLane()
    {
        int w = VByte.LaneCount;
        byte[] a = MakeBytes(w, i => (byte)i);
        byte[] b = MakeBytes(w, i => (byte)(w - i));
        byte[] mn = new byte[w];
        byte[] mx = new byte[w];

        VByte va = VByte.Load(a, 0);
        VByte vb = VByte.Load(b, 0);
        VByte.Min(va, vb).Store(mn, 0);
        VByte.Max(va, vb).Store(mx, 0);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal(Math.Min(a[i], b[i]), mn[i]);
            Assert.Equal(Math.Max(a[i], b[i]), mx[i]);
        }
    }

    [Fact]
    public void VByte_Shifts()
    {
        int w = VByte.LaneCount;
        byte[] a = MakeBytes(w, i => (byte)(i + 128)); // high bit set in some lanes
        byte[] left = new byte[w];
        byte[] right = new byte[w];

        (VByte.Load(a, 0) << 1).Store(left, 0);
        (VByte.Load(a, 0) >> 2).Store(right, 0);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal(unchecked((byte)(a[i] << 1)), left[i]);
            Assert.Equal((byte)(a[i] >> 2), right[i]); // logical: high bit never smears
        }
    }

    [Fact]
    public void VByte_Select_Blend()
    {
        int w = VByte.LaneCount;
        VMaskB mask = VMaskB.FirstN(w / 2);

        VByte sel = VByte.Select(mask, new VByte(11), new VByte(22));
        VByte blended = new VByte(22).Blend(mask, new VByte(11));

        for (int l = 0; l < w; l++)
        {
            byte expected = l < w / 2 ? (byte)11 : (byte)22;
            Assert.Equal(expected, sel.GetLane(l));
            Assert.Equal(expected, blended.GetLane(l));
        }
    }

    [Fact]
    public void VByte_WidenNarrow_RoundTrips()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(255 - i));

        var (lo, hi) = VByte.Widen(VByte.Load(data, 0));

        // Widened halves hold the original values zero-extended.
        for (int l = 0; l < VShort.LaneCount; l++)
        {
            Assert.Equal(data[l], lo.GetLane(l));
            Assert.Equal(data[l + VShort.LaneCount], hi.GetLane(l));
        }

        byte[] result = new byte[w];
        VByte.Narrow(lo, hi).Store(result, 0);
        for (int i = 0; i < w; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void VByte_Narrow_TruncatesToLow8Bits()
    {
        byte[] result = new byte[VByte.LaneCount];
        VByte.Narrow(new VShort(0x1FF), new VShort(0x102)).Store(result, 0);

        for (int i = 0; i < VByte.LaneCount / 2; i++)
            Assert.Equal(0xFF, result[i]);
        for (int i = VByte.LaneCount / 2; i < VByte.LaneCount; i++)
            Assert.Equal(0x02, result[i]);
    }

    [Fact]
    public void VByte_ProgramIndex_IsZeroThroughWidth()
    {
        VByte idx = VByte.ProgramIndex;
        for (int l = 0; l < VByte.LaneCount; l++)
            Assert.Equal(l, idx.GetLane(l));
    }

    [Fact]
    public void VByte_DivideRemainder_PerLane()
    {
        int w = VByte.LaneCount;
        byte[] a = MakeBytes(w, i => (byte)(i + 100));
        byte[] q = new byte[w];
        byte[] r = new byte[w];

        VByte va = VByte.Load(a, 0);
        (va / new VByte(7)).Store(q, 0);
        (va % new VByte(7)).Store(r, 0);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal((byte)(a[i] / 7), q[i]);
            Assert.Equal((byte)(a[i] % 7), r[i]);
        }
    }

    // ---------- VShort ----------

    [Fact]
    public void VShort_LoadStore_RoundTrips()
    {
        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)(i * 100 - 500));
        short[] result = new short[w];

        VShort.Load(data, 0).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void VShort_LoadMasked_StoreMasked()
    {
        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)(i + 1));
        short[] loaded = new short[w];
        short[] stored = MakeShorts(w, _ => -1);

        VMaskS mask = VMaskS.FirstN(3);
        VShort.LoadMasked(data, 0, mask, fallback: -99).Store(loaded, 0);
        VShort.Load(data, 0).StoreMasked(stored, 0, mask);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal(i < 3 ? (short)(i + 1) : (short)-99, loaded[i]);
            Assert.Equal(i < 3 ? (short)(i + 1) : (short)-1, stored[i]);
        }
    }

    [Fact]
    public void VShort_Arithmetic_WrapsLikeScalarCast()
    {
        int w = VShort.LaneCount;
        short[] result = new short[w];

        (new VShort(30000) + new VShort(30000)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(unchecked((short)60000), result[i]); // -5536
    }

    [Fact]
    public void VShort_AddSaturate_ClampsToShortRange()
    {
        int w = VShort.LaneCount;
        short[] a = MakeShorts(w, i => i % 2 == 0 ? (short)30000 : (short)100);
        short[] result = new short[w];

        VShort.AddSaturate(VShort.Load(a, 0), new VShort(30000)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(i % 2 == 0 ? short.MaxValue : (short)30100, result[i]);
    }

    [Fact]
    public void VShort_SubtractSaturate_ClampsToShortRange()
    {
        int w = VShort.LaneCount;
        short[] a = MakeShorts(w, i => i % 2 == 0 ? (short)-30000 : (short)100);
        short[] result = new short[w];

        VShort.SubtractSaturate(VShort.Load(a, 0), new VShort(30000)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(i % 2 == 0 ? short.MinValue : (short)-29900, result[i]);
    }

    [Fact]
    public void VShort_MultiplyHigh_MatchesScalarReference()
    {
        int w = VShort.LaneCount;
        short[] a = MakeShorts(w, i => (short)(i * 1000 - 3000));
        short[] b = MakeShorts(w, i => (short)(i * 700 + 123));
        short[] result = new short[w];

        VShort.MultiplyHigh(VShort.Load(a, 0), VShort.Load(b, 0)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal((short)((a[i] * b[i]) >> 16), result[i]);
    }

    [Fact]
    public void VShort_ArithmeticShift_ExtendsSign()
    {
        int w = VShort.LaneCount;
        short[] result = new short[w];
        short[] logical = new short[w];

        (new VShort(-256) >> 4).Store(result, 0);
        VShort.ShiftRightLogical(new VShort(-256), 4).Store(logical, 0);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal((short)-16, result[i]);                        // sign extends
            Assert.Equal((short)((-256 & 0xFFFF) >> 4), logical[i]); // zero fills
        }
    }

    [Fact]
    public void VShort_MinMaxAbs()
    {
        int w = VShort.LaneCount;
        short[] a = MakeShorts(w, i => (short)(i - (w / 2)));
        short[] mn = new short[w];
        short[] mx = new short[w];
        short[] abs = new short[w];

        VShort va = VShort.Load(a, 0);
        VShort.Min(va, VShort.Zero).Store(mn, 0);
        VShort.Max(va, VShort.Zero).Store(mx, 0);
        VShort.Abs(va).Store(abs, 0);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal(Math.Min(a[i], (short)0), mn[i]);
            Assert.Equal(Math.Max(a[i], (short)0), mx[i]);
            Assert.Equal(Math.Abs(a[i]), abs[i]);
        }
    }

    [Fact]
    public void VShort_Comparisons_AndSelect()
    {
        int w = VShort.LaneCount;
        short[] a = MakeShorts(w, i => (short)(i - 2));

        VShort va = VShort.Load(a, 0);
        VMaskS negative = va < VShort.Zero;
        VShort clamped = VShort.Select(negative, VShort.Zero, va);

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(a[l] < 0, negative.IsLaneActive(l));
            Assert.Equal(Math.Max(a[l], (short)0), clamped.GetLane(l));
        }
    }

    [Fact]
    public void VShort_WidenNarrow_RoundTrips()
    {
        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)(i * 1000 - 7000));

        var (lo, hi) = VShort.Widen(VShort.Load(data, 0));

        for (int l = 0; l < VInt.LaneCount; l++)
        {
            Assert.Equal(data[l], lo.GetLane(l));
            Assert.Equal(data[l + VInt.LaneCount], hi.GetLane(l));
        }

        short[] result = new short[w];
        VShort.Narrow(lo, hi).Store(result, 0);
        for (int i = 0; i < w; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void VShort_ProgramIndex_IsZeroThroughWidth()
    {
        VShort idx = VShort.ProgramIndex;
        for (int l = 0; l < VShort.LaneCount; l++)
            Assert.Equal(l, idx.GetLane(l));
    }

    [Fact]
    public void VShort_DivideRemainder_PerLane()
    {
        int w = VShort.LaneCount;
        short[] a = MakeShorts(w, i => (short)(i * 37 - 100));
        short[] q = new short[w];
        short[] r = new short[w];

        VShort va = VShort.Load(a, 0);
        (va / new VShort(9)).Store(q, 0);
        (va % new VShort(9)).Store(r, 0);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal((short)(a[i] / 9), q[i]);
            Assert.Equal((short)(a[i] % 9), r[i]);
        }
    }
        
    [Fact]
    public void VMaskB_BasicOperations()
    {
        int w = VMaskB.LaneCount;
        Assert.Equal(VByte.LaneCount, w);

        Assert.True(VMaskB.All.AllActive());
        Assert.True(VMaskB.None.NoneActive());

        VMaskB firstHalf = VMaskB.FirstN(w / 2);
        Assert.Equal(w / 2, firstHalf.CountActive());
        Assert.True(firstHalf.Any());
        Assert.False(firstHalf.AllActive());

        VMaskB inverted = !firstHalf;
        for (int l = 0; l < w; l++)
            Assert.Equal(l >= w / 2, inverted.IsLaneActive(l));

        Assert.Equal(VMaskB.None, firstHalf & inverted);
        Assert.Equal(VMaskB.All, firstHalf | inverted);
        Assert.Equal(VMaskB.All, firstHalf ^ inverted);
        Assert.Equal(firstHalf, VMaskB.AndNot(VMaskB.All, inverted));
    }

    [Fact]
    public void VMaskS_BasicOperations()
    {
        int w = VMaskS.LaneCount;
        Assert.Equal(VShort.LaneCount, w);

        Assert.True(VMaskS.All.AllActive());
        Assert.True(VMaskS.None.NoneActive());

        VMaskS firstHalf = VMaskS.FirstN(w / 2);
        Assert.Equal(w / 2, firstHalf.CountActive());

        VMaskS inverted = ~firstHalf;
        Assert.Equal(VMaskS.None, firstHalf & inverted);
        Assert.Equal(VMaskS.All, firstHalf | inverted);
    }

    [Fact]
    public void VMaskB_WidenNarrow_RoundTrips()
    {
        int w = VMaskB.LaneCount;
        VMaskB original = VMaskB.FirstN(w / 2 + 1);

        var (lo, hi) = VMaskB.Widen(original);

        for (int l = 0; l < VMaskS.LaneCount; l++)
        {
            Assert.Equal(original.IsLaneActive(l), lo.IsLaneActive(l));
            Assert.Equal(original.IsLaneActive(l + VMaskS.LaneCount), hi.IsLaneActive(l));
        }

        // Widened lanes must be full -1/0 so ConditionalSelect works on them.
        VShort sel = VShort.Select(lo, new VShort(5), new VShort(9));
        for (int l = 0; l < VShort.LaneCount; l++)
            Assert.Equal(lo.IsLaneActive(l) ? (short)5 : (short)9, sel.GetLane(l));

        Assert.Equal(original, VMaskB.Narrow(lo, hi));
    }

    [Fact]
    public void VMaskS_WidenNarrow_RoundTrips()
    {
        int w = VMaskS.LaneCount;
        VMaskS original = VMaskS.FirstN(w / 2 + 1);

        var (lo, hi) = VMaskS.Widen(original);

        for (int l = 0; l < VMask.LaneCount; l++)
        {
            Assert.Equal(original.IsLaneActive(l), lo.IsLaneActive(l));
            Assert.Equal(original.IsLaneActive(l + VMask.LaneCount), hi.IsLaneActive(l));
        }

        Assert.Equal(original, VMaskS.Narrow(lo, hi));
    }

    [Fact]
    public void Reduce_Add_VByte_WidensBeyondByteRange()
    {
        // Every lane 255: the sum must be 255 * LaneCount, provably wider than a byte.
        int sum = Reduce.Add(new VByte(255));
        Assert.Equal(255 * VByte.LaneCount, sum);
    }

    [Fact]
    public void Reduce_Add_VByte_MatchesScalarSum()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(i * 7 + 3));
        int expected = 0;
        for (int i = 0; i < w; i++)
            expected += data[i];

        Assert.Equal(expected, Reduce.Add(VByte.Load(data, 0)));
    }

    [Fact]
    public void Reduce_Add_VByte_Masked()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(i + 1));
        int expected = 0;
        for (int i = 0; i < 3; i++)
            expected += data[i];

        Assert.Equal(expected, Reduce.Add(VByte.Load(data, 0), VMaskB.FirstN(3)));
    }

    [Fact]
    public void Reduce_MinMax_VByte()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)((i * 31 + 7) % 251));
        byte mn = byte.MaxValue, mx = byte.MinValue;
        for (int i = 0; i < w; i++)
        {
            mn = Math.Min(mn, data[i]);
            mx = Math.Max(mx, data[i]);
        }

        VByte v = VByte.Load(data, 0);
        Assert.Equal(mn, Reduce.Min(v));
        Assert.Equal(mx, Reduce.Max(v));

        // Masked: restrict to the first two lanes.
        VMaskB first2 = VMaskB.FirstN(2);
        Assert.Equal(Math.Min(data[0], data[1]), Reduce.Min(v, first2));
        Assert.Equal(Math.Max(data[0], data[1]), Reduce.Max(v, first2));
    }

    [Fact]
    public void Reduce_Add_VShort_HandlesNegativesAndOverflow()
    {
        // Every lane 30000: an unwidened short accumulator would overflow immediately.
        Assert.Equal(30000 * VShort.LaneCount, Reduce.Add(new VShort(30000)));

        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)(i * 500 - 2000));
        int expected = 0;
        for (int i = 0; i < w; i++)
            expected += data[i];

        Assert.Equal(expected, Reduce.Add(VShort.Load(data, 0)));
    }

    [Fact]
    public void Reduce_Add_VShort_Masked()
    {
        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)(i - 3));
        int expected = 0;
        for (int i = 0; i < 4; i++)
            expected += data[i];

        Assert.Equal(expected, Reduce.Add(VShort.Load(data, 0), VMaskS.FirstN(4)));
    }

    [Fact]
    public void Reduce_MinMax_VShort()
    {
        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)((i * 313 + 17) % 1000 - 500));
        short mn = short.MaxValue, mx = short.MinValue;
        for (int i = 0; i < w; i++)
        {
            mn = Math.Min(mn, data[i]);
            mx = Math.Max(mx, data[i]);
        }

        VShort v = VShort.Load(data, 0);
        Assert.Equal(mn, Reduce.Min(v));
        Assert.Equal(mx, Reduce.Max(v));

        VMaskS first2 = VMaskS.FirstN(2);
        Assert.Equal(Math.Min(data[0], data[1]), Reduce.Min(v, first2));
        Assert.Equal(Math.Max(data[0], data[1]), Reduce.Max(v, first2));
    }

    [Fact]
    public void VInt_LoadZeroExtend_WidensBytes()
    {
        int w = VInt.LaneCount;
        byte[] data = MakeBytes(w + 3, i => (byte)(250 - i)); // includes values > 127 (sign trap)
        int[] result = new int[w];

        VInt.LoadZeroExtend(data, 3).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(data[i + 3], result[i]); // zero-extended, never negative
    }

    [Fact]
    public void VInt_LoadSignExtend_WidensShorts()
    {
        int w = VInt.LaneCount;
        short[] data = MakeShorts(w + 2, i => (short)((i * 5000) - 20000)); // negatives included
        int[] result = new int[w];

        VInt.LoadSignExtend(data, 2).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(data[i + 2], result[i]); // sign-extended
    }

    [Fact]
    public void VInt_StoreNarrow_TruncatesLikeCast()
    {
        int w = VInt.LaneCount;
        int[] values = new int[w];
        for (int i = 0; i < w; i++)
            values[i] = (i * 1000) - 3000 + ((i % 2) * 0x1FF00); // mix of negative, >255, >65535

        VInt v = VInt.Load(values, 0);

        byte[] bytes = new byte[w + 1];
        v.StoreNarrow(bytes, 1);
        for (int i = 0; i < w; i++)
            Assert.Equal((byte)values[i], bytes[i + 1]);

        short[] shorts = new short[w + 1];
        v.StoreNarrow(shorts, 1);
        for (int i = 0; i < w; i++)
            Assert.Equal((short)values[i], shorts[i + 1]);
    }

    [Fact]
    public void VInt_NarrowBridges_RoundTrip()
    {
        int w = VInt.LaneCount;
        byte[] bytes = MakeBytes(w, i => (byte)(i * 16 + 7));
        short[] shorts = MakeShorts(w, i => (short)((i * 999) - 4000));

        byte[] bOut = new byte[w];
        VInt.LoadZeroExtend(bytes, 0).StoreNarrow(bOut, 0);
        Assert.Equal(bytes, bOut);

        short[] sOut = new short[w];
        VInt.LoadSignExtend(shorts, 0).StoreNarrow(sOut, 0);
        Assert.Equal(shorts, sOut);
    }

    [Fact]
    public void Lanes_Shuffle_VByte_Reverses()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(i + 1));
        byte[] rev = MakeBytes(w, i => (byte)(w - 1 - i));

        VByte shuffled = Lanes.Shuffle(VByte.Load(data, 0), VByte.Load(rev, 0));

        for (int l = 0; l < w; l++)
            Assert.Equal(data[w - 1 - l], shuffled.GetLane(l));
    }

    [Fact]
    public void Lanes_Shuffle_VShort_Reverses()
    {
        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)(i * 11 - 40));
        short[] rev = MakeShorts(w, i => (short)(w - 1 - i));

        VShort shuffled = Lanes.Shuffle(VShort.Load(data, 0), VShort.Load(rev, 0));

        for (int l = 0; l < w; l++)
            Assert.Equal(data[w - 1 - l], shuffled.GetLane(l));
    }

    [Fact]
    public void Lanes_Rotate_VByte_WrapsBothDirections()
    {
        int w = VByte.LaneCount;
        byte[] data = MakeBytes(w, i => (byte)(i + 1));
        VByte v = VByte.Load(data, 0);

        VByte forward = Lanes.Rotate(v, 1);
        VByte backward = Lanes.Rotate(v, -1);

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(data[(l + 1) % w], forward.GetLane(l));
            Assert.Equal(data[(l - 1 + w) % w], backward.GetLane(l));
        }
    }

    [Fact]
    public void Lanes_Rotate_VShort_WrapsBothDirections()
    {
        int w = VShort.LaneCount;
        short[] data = MakeShorts(w, i => (short)(i + 1));
        VShort v = VShort.Load(data, 0);

        VShort forward = Lanes.Rotate(v, 1);
        VShort backward = Lanes.Rotate(v, -1);

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(data[(l + 1) % w], forward.GetLane(l));
            Assert.Equal(data[(l - 1 + w) % w], backward.GetLane(l));
        }
    }

    [Fact]
    public void Lanes_Broadcast_ByteAndShort()
    {
        int wb = VByte.LaneCount;
        byte[] bytes = MakeBytes(wb, i => (byte)(i + 5));
        VByte vb = Lanes.Broadcast(VByte.Load(bytes, 0), 2);
        for (int l = 0; l < wb; l++)
            Assert.Equal(bytes[2], vb.GetLane(l));

        int ws = VShort.LaneCount;
        short[] shorts = MakeShorts(ws, i => (short)(i - 3));
        VShort vs = Lanes.Broadcast(VShort.Load(shorts, 0), 1);
        for (int l = 0; l < ws; l++)
            Assert.Equal(shorts[1], vs.GetLane(l));
    }

    [Fact]
    public void Lanes_ShiftLanes_EdgesBecomeZero()
    {
        int wb = VByte.LaneCount;
        byte[] bytes = MakeBytes(wb, i => (byte)(i + 1));
        VByte vb = VByte.Load(bytes, 0);

        VByte bUp = Lanes.ShiftLanes(vb, 1);
        VByte bDown = Lanes.ShiftLanes(vb, -1);
        for (int l = 0; l < wb; l++)
        {
            Assert.Equal(l + 1 < wb ? bytes[l + 1] : (byte)0, bUp.GetLane(l));
            Assert.Equal(l - 1 >= 0 ? bytes[l - 1] : (byte)0, bDown.GetLane(l));
        }

        int ws = VShort.LaneCount;
        short[] shorts = MakeShorts(ws, i => (short)(i + 1));
        VShort vs = VShort.Load(shorts, 0);

        VShort sUp = Lanes.ShiftLanes(vs, 1);
        VShort sDown = Lanes.ShiftLanes(vs, -1);
        for (int l = 0; l < ws; l++)
        {
            Assert.Equal(l + 1 < ws ? shorts[l + 1] : (short)0, sUp.GetLane(l));
            Assert.Equal(l - 1 >= 0 ? shorts[l - 1] : (short)0, sDown.GetLane(l));
        }
    }
}
