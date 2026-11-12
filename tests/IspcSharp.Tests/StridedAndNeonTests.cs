using System;
using System.Linq;
using Xunit;

namespace IspcSharp.Tests;

[SpmdStruct]
public struct Pair2
{
    public float X;
    public float Y;
}

[SpmdStruct]
public struct Quad4
{
    public float A;
    public float B;
    public float C;
    public float D;
}

[SpmdStruct]
public struct IntPair
{
    public int Lo;
    public int Hi;
}

/// <summary>
/// Kernels over Array-of-Structs buffers. <c>buf[i].field</c> is a strided access, lowered to
/// <c>Memory.LoadStrided</c>/<c>StoreStrided</c>: a de-interleave where the platform has one
/// (NEON ld2/ld3/ld4, Vector.Unzip on .NET 11), a gather/scatter elsewhere. Same results either way.
/// </summary>
public static partial class StridedKernels
{
    [Spmd(GenerateParallel = false)]
    public static void SwapScale(Pair2[] pts, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = pts[i].X;
            float y = pts[i].Y;
            pts[i].X = y * 2f;
            pts[i].Y = x + 1f;
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void MaskedFieldStore(Pair2[] pts, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            if (pts[i].X > 0f)
                pts[i].Y = -pts[i].Y;
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void WholeStruct(Pair2[] src, Pair2[] dst, float k, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            Pair2 p = src[i];
            dst[i] = new Pair2 { X = p.X + (p.Y * k), Y = p.X - p.Y };
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void NeighbourField(Pair2[] pts, float[] o, int count)
    {
        foreach (int i in Spmd.Range(0, count - 1))
        {
            o[i] = pts[i + 1].X - pts[i].Y;
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void QuadMix(Quad4[] q, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float a = q[i].A;
            float d = q[i].D;
            q[i].D = a + (q[i].B * q[i].C);
            q[i].A = d - q[i].C;
            if (q[i].B < 0f)
                q[i].C = 0f;
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void IntPairMix(IntPair[] p, int[] o, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            o[i] = (p[i].Hi << 4) ^ p[i].Lo;
            p[i].Lo = p[i].Lo + p[i].Hi;
        }
    }
}

public class StridedAccessTests
{
    private static int N => (5 * Spmd.LaneCount) + 3;

    private static Pair2[] Pairs(int n) =>
        [.. Enumerable.Range(0, n).Select(i => new Pair2 { X = ((i * 7) % 11) - 5f, Y = (i * 0.5f) + 1f })];

    [Fact]
    public void TwoFieldStruct_ReadAndWriteBothFields()
    {
        int n = N;
        Pair2[] expected = Pairs(n), actual = Pairs(n);
        StridedKernels.SwapScale(expected, n);
        StridedKernels.SwapScale_Simd(actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TwoFieldStruct_MaskedStoreLeavesOtherLanesAndFieldsAlone()
    {
        int n = N;
        Pair2[] expected = Pairs(n), actual = Pairs(n);
        StridedKernels.MaskedFieldStore(expected, n);
        StridedKernels.MaskedFieldStore_Simd(actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TwoFieldStruct_WholeStructReadAndWrite()
    {
        int n = N;
        Pair2[] src = Pairs(n);
        Pair2[] expected = new Pair2[n], actual = new Pair2[n];
        StridedKernels.WholeStruct(src, expected, 0.25f, n);
        StridedKernels.WholeStruct_Simd(src, actual, 0.25f, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TwoFieldStruct_OffsetIndex()
    {
        int n = N;
        Pair2[] pts = Pairs(n);
        float[] expected = new float[n], actual = new float[n];
        StridedKernels.NeighbourField(pts, expected, n);
        StridedKernels.NeighbourField_Simd(pts, actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void FourFieldStruct_MatchesScalar()
    {
        int n = N;
        Quad4[] Make() => [.. Enumerable.Range(0, n).Select(i => new Quad4 { A = i, B = (i % 3) - 1f, C = i * 0.25f, D = 100f - i })];
        Quad4[] expected = Make(), actual = Make();
        StridedKernels.QuadMix(expected, n);
        StridedKernels.QuadMix_Simd(actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void IntFields_MatchScalar()
    {
        int n = N;
        IntPair[] Make() => [.. Enumerable.Range(0, n).Select(i => new IntPair { Lo = (i * 31) - 40, Hi = i ^ 0x55 })];
        IntPair[] expected = Make(), actual = Make();
        int[] eo = new int[n], ao = new int[n];
        StridedKernels.IntPairMix(expected, eo, n);
        StridedKernels.IntPairMix_Simd(actual, ao, n);
        Assert.Equal(eo, ao);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void LoadStrided_And_StoreStrided_MatchTheirDefinition(int stride)
    {
        int w = VFloat.LaneCount;
        int structs = w + 3;
        float[] f = [.. Enumerable.Range(0, structs * stride).Select(i => i + 0.5f)];
        int[] ints = [.. Enumerable.Range(0, structs * stride).Select(i => (i * 13) - 7)];
        int wd = VDouble.LaneCount;
        double[] d = [.. Enumerable.Range(0, (wd + 3) * stride).Select(i => i + 0.25)];
        long[] l = [.. Enumerable.Range(0, (wd + 3) * stride).Select(i => ((long)i << 33) + i)];

        for (int baseIndex = 0; baseIndex <= 3; baseIndex++)
        {
            for (int field = 0; field < stride; field++)
            {
                VFloat vf = Memory.LoadStrided(f, baseIndex, stride, field);
                VInt vi = Memory.LoadStrided(ints, baseIndex, stride, field);
                for (int lane = 0; lane < w; lane++)
                {
                    Assert.Equal(f[((baseIndex + lane) * stride) + field], vf.GetLane(lane));
                    Assert.Equal(ints[((baseIndex + lane) * stride) + field], vi.GetLane(lane));
                }

                VDouble vd = Memory.LoadStrided(d, baseIndex, stride, field);
                VLong vl = Memory.LoadStrided(l, baseIndex, stride, field);
                for (int lane = 0; lane < wd; lane++)
                {
                    Assert.Equal(d[((baseIndex + lane) * stride) + field], vd.GetLane(lane));
                    Assert.Equal(l[((baseIndex + lane) * stride) + field], vl.GetLane(lane));
                }

                // Stores: only the addressed field of the active lanes changes.
                foreach (VMask mask in new[] { VMask.All, VMask.FirstN(1), VMask.FirstN(w - 1), !VMask.FirstN(2), VMask.None })
                {
                    float[] fe = (float[])f.Clone(), fa = (float[])f.Clone();
                    int[] ie = (int[])ints.Clone(), ia = (int[])ints.Clone();
                    VFloat newF = VInt.ProgramIndex.ToFloat() - 1000f;
                    VInt newI = VInt.ProgramIndex - 5000;
                    for (int lane = 0; lane < w; lane++)
                    {
                        if (!mask.IsLaneActive(lane))
                            continue;
                        fe[((baseIndex + lane) * stride) + field] = newF.GetLane(lane);
                        ie[((baseIndex + lane) * stride) + field] = newI.GetLane(lane);
                    }

                    Memory.StoreStrided(fa, baseIndex, stride, field, newF, mask);
                    Memory.StoreStrided(ia, baseIndex, stride, field, newI, mask);
                    Assert.Equal(fe, fa);
                    Assert.Equal(ie, ia);
                }

                foreach (VMaskD mask in new[] { VMaskD.All, VMaskD.FirstN(1), VMaskD.None })
                {
                    double[] de = (double[])d.Clone(), da = (double[])d.Clone();
                    long[] le = (long[])l.Clone(), la = (long[])l.Clone();
                    VDouble newD = VDouble.ProgramIndex - 1000d;
                    VLong newL = VLong.ProgramIndex - 5000L;
                    for (int lane = 0; lane < wd; lane++)
                    {
                        if (!mask.IsLaneActive(lane))
                            continue;
                        de[((baseIndex + lane) * stride) + field] = newD.GetLane(lane);
                        le[((baseIndex + lane) * stride) + field] = newL.GetLane(lane);
                    }

                    Memory.StoreStrided(da, baseIndex, stride, field, newD, mask);
                    Memory.StoreStrided(la, baseIndex, stride, field, newL, mask);
                    Assert.Equal(de, da);
                    Assert.Equal(le, la);
                }
            }
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void LoadStrided_OutOfRange_Throws(int stride)
    {
        int w = VFloat.LaneCount;
        float[] f = new float[w * stride];
        _ = Memory.LoadStrided(f, 0, stride, stride - 1);                       // exactly fits
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.LoadStrided(f, 1, stride, 0));
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.LoadStrided(f, -1, stride, 0));
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.LoadStrided(f, int.MaxValue / 2, stride, 0));
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.StoreStrided(f, 1, stride, 0, VFloat.One, VMask.All));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(101)]
    public void SoaFloat2_Interleave_RoundTripsAtAnyLength(int n)
    {
        float[] xy = [.. Enumerable.Range(0, 2 * n).Select(i => i * 1.5f)];
        var soa = SoaFloat2.FromInterleaved(xy);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(xy[2 * i], soa.X[i]);
            Assert.Equal(xy[(2 * i) + 1], soa.Y[i]);
        }

        float[] back = new float[2 * n];
        soa.ToInterleaved(back);
        Assert.Equal(xy, back);
    }
}

/// <summary>
/// Lane operations that have a dedicated NEON path, checked lane by lane against the scalar
/// definition. On x86 the same tests cover the SSE/AVX paths.
/// </summary>
public class LaneOperationReferenceTests
{
    private static readonly short[] ShortEdges = [0, 1, -1, 2, -2, 255, 256, -256, 12345, -12345, short.MaxValue, short.MinValue, short.MaxValue - 1, short.MinValue + 1];
    private static readonly int[] IntEdges = [0, 1, -1, 2, 65535, 65536, -65536, 0x7FFFFFFF, unchecked((int)0x80000000), unchecked((int)0xFFFF0000), 0x12345678, unchecked((int)0xDEADBEEF)];

    [Fact]
    public void VShort_MultiplyHigh_AllEdgePairs()
    {
        var rnd = new Random(11);
        int w = VShort.LaneCount;
        int count = ((ShortEdges.Length * ShortEdges.Length) + 500 + w - 1) / w * w;
        short[] a = new short[count], b = new short[count];
        int p = 0;
        foreach (short x in ShortEdges)
        {
            foreach (short y in ShortEdges)
            {
                a[p] = x;
                b[p++] = y;
            }
        }

        for (; p < count; p++)
        {
            a[p] = (short)rnd.Next(short.MinValue, short.MaxValue + 1);
            b[p] = (short)rnd.Next(short.MinValue, short.MaxValue + 1);
        }

        for (int i = 0; i < count; i += w)
        {
            VShort r = VShort.MultiplyHigh(VShort.Load(a, i), VShort.Load(b, i));
            for (int lane = 0; lane < w; lane++)
                Assert.Equal((short)((a[i + lane] * b[i + lane]) >> 16), r.GetLane(lane));
        }
    }

    [Fact]
    public void VInt_MultiplyHighUnsigned_AllEdgePairs()
    {
        var rnd = new Random(12);
        int w = VInt.LaneCount;
        int count = ((IntEdges.Length * IntEdges.Length) + 500 + w - 1) / w * w;
        int[] a = new int[count], b = new int[count];
        int p = 0;
        foreach (int x in IntEdges)
        {
            foreach (int y in IntEdges)
            {
                a[p] = x;
                b[p++] = y;
            }
        }

        for (; p < count; p++)
        {
            a[p] = rnd.Next(int.MinValue, int.MaxValue);
            b[p] = rnd.Next(int.MinValue, int.MaxValue);
        }

        for (int i = 0; i < count; i += w)
        {
            VInt r = VInt.MultiplyHighUnsigned(VInt.Load(a, i), VInt.Load(b, i));
            for (int lane = 0; lane < w; lane++)
                Assert.Equal(Spmd.MultiplyHighUnsigned(a[i + lane], b[i + lane]), r.GetLane(lane));
        }
    }

    [Fact]
    public void VInt_VariableShifts_UseLowFiveBitsOfEachCount()
    {
        int w = VInt.LaneCount;
        int[] counts = [0, 1, 2, 7, 15, 16, 30, 31, 32, 33, 63, 64, -1, -31, -32, 1000];
        int total = (IntEdges.Length * counts.Length + w - 1) / w * w;
        int[] x = new int[total], c = new int[total];
        for (int i = 0; i < total; i++)
        {
            x[i] = IntEdges[i / counts.Length % IntEdges.Length];
            c[i] = counts[i % counts.Length];
        }

        for (int i = 0; i < total; i += w)
        {
            VInt vx = VInt.Load(x, i), vc = VInt.Load(c, i);
            VInt left = VInt.ShiftLeftVariable(vx, vc);
            VInt arithmetic = VInt.ShiftRightArithmeticVariable(vx, vc);
            VInt logical = VInt.ShiftRightLogicalVariable(vx, vc);
            for (int lane = 0; lane < w; lane++)
            {
                int xv = x[i + lane], cv = c[i + lane];
                Assert.Equal(xv << cv, left.GetLane(lane));
                Assert.Equal(xv >> cv, arithmetic.GetLane(lane));
                Assert.Equal(xv >>> cv, logical.GetLane(lane));
            }
        }
    }

    [Fact]
    public void ReduceAdd_NarrowLanes_WidenBeforeSumming()
    {
        var rnd = new Random(13);
        byte[] bytes = new byte[VByte.LaneCount];
        short[] shorts = new short[VShort.LaneCount];

        Assert.Equal(255 * VByte.LaneCount, Reduce.Add(new VByte(255)));
        Assert.Equal(0, Reduce.Add(VByte.Zero));
        Assert.Equal(short.MinValue * VShort.LaneCount, Reduce.Add(new VShort(short.MinValue)));
        Assert.Equal(short.MaxValue * VShort.LaneCount, Reduce.Add(new VShort(short.MaxValue)));

        for (int trial = 0; trial < 50; trial++)
        {
            rnd.NextBytes(bytes);
            for (int i = 0; i < shorts.Length; i++)
                shorts[i] = (short)rnd.Next(short.MinValue, short.MaxValue + 1);
            Assert.Equal(bytes.Sum(v => (int)v), Reduce.Add(VByte.Load(bytes)));
            Assert.Equal(shorts.Sum(v => (int)v), Reduce.Add(VShort.Load(shorts)));
        }
    }

    [Fact]
    public void StoreNarrow_TruncatesLikeACast()
    {
        int w = VInt.LaneCount;
        int[] src = [.. Enumerable.Range(0, w).Select(i => IntEdges[i % IntEdges.Length] + (i * 300))];
        byte[] b = new byte[w + 2];
        short[] s = new short[w + 2];
        VInt.Load(src).StoreNarrow(b, 1);
        VInt.Load(src).StoreNarrow(s, 1);
        for (int lane = 0; lane < w; lane++)
        {
            Assert.Equal((byte)src[lane], b[lane + 1]);
            Assert.Equal((short)src[lane], s[lane + 1]);
        }

        Assert.Equal(0, b[0]);
        Assert.Equal(0, b[^1]);
        Assert.Equal(0, s[0]);
        Assert.Equal(0, s[^1]);
    }
}
