using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Constants a kernel reads from outside its own body: <c>const</c> fields and
/// <c>static readonly</c> ones. Both are uniform for the whole run, so a use inside a
/// vectorized expression is a broadcast, exactly like a uniform parameter.
/// </summary>
public static class ConstHolder
{
    public const float Gain = 2.5f;
    public const int Shift = 3;
    public const double Rate = 0.125;
    public static readonly float Bias = 0.75f;
}

/// <summary>
/// Constants declared in one part of a partial class and read from a kernel in another part,
/// the case a purely syntactic scan of the containing declaration would miss.
/// </summary>
public static partial class ConstKernels
{
    private const float Scale = 1.1920929e-7f;   // 2^-23
    private const float Offset = 5.9604645e-8f;  // 2^-24
    private const float TwoPi = 6.2831853f;
}

[SpmdStruct]
public struct UnitPair
{
    public float A;
    public float B;
}

/// <summary>
/// Field names that collide with constant names in the calling class. Naming a field in an
/// object initializer is not a value read, so it must not shadow or poison the constant.
/// </summary>
[SpmdStruct]
public struct Word2
{
    public int W0;
    public int W1;
}

/// <summary>
/// A helper library in its own class: kernels reach these by qualified name, and the constants
/// they read live here too.
/// </summary>
public static partial class ConstLib
{
    private const int Rounds = 3;
    private const float Weight = 0.25f;

    [SpmdFunction]
    public static float Blend(float a, float b) => (a * Weight) + (b * (1f - Weight));

    [SpmdFunction]
    public static Word2 Step(int x) => new Word2 { W0 = x + Rounds, W1 = x * Rounds };
}

public static partial class ConstKernels
{
    private const int LowBits = 9;
    private const int ByteMask = 0xFF;
    private const long Mix = 0x5555555555555555L;
    private const byte NarrowBias = 7;
    private const int FixedCount = 1000;

    // Same names as Word2's fields, on purpose: the two must not be confused.
    private const int W0 = unchecked((int)0x9E3779B9);
    private const int W1 = unchecked((int)0xBB67AE85);
    private static readonly int Stride = 2;

    /// <summary>Bare constant in a float lane expression (the ISPC003 case this adds).</summary>
    [Spmd]
    public static void ScaleByConst(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (input[i] * Scale) + Offset;
        }
    }

    /// <summary>Constant qualified by its declaring type, and one from another class.</summary>
    [Spmd]
    public static void QualifiedConsts(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (input[i] * ConstKernels.TwoPi) + ConstHolder.Gain;
        }
    }

    /// <summary>A <c>static readonly</c> field: uniform at runtime, so it broadcasts too.</summary>
    [Spmd]
    public static void StaticReadonlyBias(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = input[i] + ConstHolder.Bias;
        }
    }

    /// <summary>
    /// Int constants as a uniform shift count and a mask: both stay integer lane work
    /// (the shift count is emitted as a scalar, not broadcast).
    /// </summary>
    [Spmd]
    public static void BitsByConst(int[] input, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (input[i] >>> LowBits) & ByteMask;
        }
    }

    /// <summary>A byte constant widens to int lanes, matching C#'s own promotion.</summary>
    [Spmd]
    public static void NarrowConst(byte[] input, byte[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (byte)Math.Min(input[i] + NarrowBias, 255);
        }
    }

    /// <summary>A constant in a condition and in both arms of a masked branch.</summary>
    [Spmd]
    public static void ThresholdByConst(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (x > Scale)
                x = x * TwoPi;
            else
                x = x + Offset;
            output[i] = x;
        }
    }

    /// <summary>A local shadowing a constant wins, exactly as in the scalar method.</summary>
    [Spmd]
    public static void ShadowedConst(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float Scale = 10f;
            output[i] = input[i] * Scale;
        }
    }

    /// <summary>
    /// A constant used before the loop, inside it, and in the trailing scalar statement, so
    /// the companion's scalar tail has to resolve it as well as the vector body.
    /// </summary>
    [Spmd]
    public static float ConstWeightedSum(float[] input, int count)
    {
        float sum = 0f;
        foreach (int i in Spmd.Range(count))
        {
            sum += input[i] * ConstHolder.Gain;
        }

        return sum + Offset;
    }

    /// <summary>A constant as a uniform offset in an index expression stays contiguous.</summary>
    [Spmd]
    public static void ConstIndexOffset(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = input[i + Stride];
        }
    }

    /// <summary>A constant as the loop bound: it lands in the range expression, not the gang.</summary>
    [Spmd]
    public static void FixedRange(float[] input, float[] output)
    {
        foreach (int i in Spmd.Range(FixedCount))
        {
            output[i] = input[i] * TwoPi;
        }
    }

    /// <summary>Double kernel: a double constant broadcasts into the 64-bit float gang.</summary>
    [Spmd]
    public static void DoubleConst(double[] input, double[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = input[i] * ConstHolder.Rate;
        }
    }

    /// <summary>
    /// Float kernel reading a double constant: the expression widens to a full-gang-width
    /// VDouble2 pair, then narrows back on the store.
    /// </summary>
    [Spmd]
    public static void DoubleConstInFloatKernel(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (float)(input[i] * ConstHolder.Rate);
        }
    }

    /// <summary>
    /// Int kernel reading a long constant: the multiply widens into a full-gang-width
    /// VLong2 pair, so it can't overflow the way a 32-bit product would.
    /// </summary>
    [Spmd]
    public static void LongConstInIntKernel(int[] input, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (int)((input[i] * Mix) >> 32);
        }
    }

    /// <summary>Long kernel: a long constant broadcasts into the 64-bit integer gang.</summary>
    [Spmd]
    public static void LongConst(long[] input, long[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            long x = input[i] ^ (input[i] >> LowBits);
            output[i] = x * Mix;
        }
    }

    /// <summary>
    /// A constant whose name matches a [SpmdStruct] field. The 'W0 =' in the object initializer
    /// designates the field, so it must neither resolve to the constant nor stop 'k + W0' from
    /// doing so.
    /// </summary>
    [SpmdFunction]
    public static Word2 BumpWords(int a, int b)
    {
        Word2 w = new Word2 { W0 = a, W1 = b };
        int k0 = w.W0 + W0;
        int k1 = w.W1 + W1;
        return new Word2 { W0 = k0, W1 = k1 };
    }

    [Spmd]
    public static void ConstNameMatchesField(int[] a, int[] outW0, int[] outW1, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            Word2 w = BumpWords(a[i], a[i] + 1);
            outW0[i] = w.W0;
            outW1[i] = w.W1;
        }
    }

    /// <summary>
    /// Helpers in another class, called by qualified name: one scalar-returning, one
    /// struct-returning, each reading constants private to its own class.
    /// </summary>
    [Spmd]
    public static void CrossClassHelpers(float[] a, float[] b, float[] o, int[] steps, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            o[i] = ConstLib.Blend(a[i], b[i]);
            Word2 w = ConstLib.Step(i);
            steps[i] = w.W0 + w.W1;
        }
    }

    /// <summary>Constants inside a [SpmdFunction] helper, including a struct-returning one.</summary>
    [SpmdFunction]
    public static float Unit(int bits)
        => ((bits >>> LowBits) * Scale) + Offset;

    [SpmdFunction]
    public static UnitPair UnitPairOf(int bitsA, int bitsB)
        => new UnitPair { A = Unit(bitsA) * TwoPi, B = Unit(bitsB) * ConstHolder.Gain };

    [Spmd]
    public static void UnitsFromBits(int[] bits, float[] outA, float[] outB, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            UnitPair p = UnitPairOf(bits[i], bits[i] + 1);
            outA[i] = p.A;
            outB[i] = p.B;
        }
    }
}

public class ConstantKernelTests
{
    private const int N = (8192 * 2) + 3;

    private static float[] Rand(int n, Random r)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = (float)((r.NextDouble() * 2) - 1);
        return a;
    }

    private static float UnitScalar(int bits)
        => ((bits >>> 9) * 1.1920929e-7f) + 5.9604645e-8f;

    [Fact]
    public void ScaleByConst_MatchesScalar()
    {
        float[] a = Rand(N, new Random(1)), o = new float[N];
        ConstKernels.ScaleByConst_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal((a[i] * 1.1920929e-7f) + 5.9604645e-8f, o[i], 9);
    }

    [Fact]
    public void QualifiedConsts_MatchScalar()
    {
        float[] a = Rand(N, new Random(2)), o = new float[N];
        ConstKernels.QualifiedConsts_Simd(a, o, N);
        for (int i = 0; i < N; i++)
        {
            // 'x * TwoPi + Gain' fuses into an FMA, so compare with a tolerance, not bit-exact.
            float expected = (a[i] * 6.2831853f) + 2.5f;
            Assert.True(MathF.Abs(o[i] - expected) <= 1e-6f * (1f + MathF.Abs(expected)),
                $"i={i}: {o[i]} vs {expected}");
        }
    }

    [Fact]
    public void StaticReadonly_Broadcasts()
    {
        float[] a = Rand(N, new Random(3)), o = new float[N];
        ConstKernels.StaticReadonlyBias_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] + ConstHolder.Bias, o[i], 5);
    }

    [Fact]
    public void IntConsts_AsShiftCountAndMask()
    {
        int[] a = new int[N], o = new int[N];
        Random r = new Random(4);
        for (int i = 0; i < N; i++)
            a[i] = r.Next(int.MinValue, int.MaxValue);
        ConstKernels.BitsByConst_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal((a[i] >>> 9) & 0xFF, o[i]);
    }

    [Fact]
    public void ByteConst_WidensToIntLanes()
    {
        byte[] a = new byte[N], o = new byte[N];
        Random r = new Random(5);
        for (int i = 0; i < N; i++)
            a[i] = (byte)r.Next(256);
        ConstKernels.NarrowConst_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal((byte)Math.Min(a[i] + 7, 255), o[i]);
    }

    [Fact]
    public void ConstInMaskedBranch_MatchesScalar()
    {
        float[] a = Rand(N, new Random(6)), o = new float[N];
        ConstKernels.ThresholdByConst_Simd(a, o, N);
        for (int i = 0; i < N; i++)
        {
            float expected = a[i] > 1.1920929e-7f ? a[i] * 6.2831853f : a[i] + 5.9604645e-8f;
            Assert.Equal(expected, o[i], 4);
        }
    }

    [Fact]
    public void LocalShadowsConst()
    {
        float[] a = Rand(N, new Random(7)), o = new float[N];
        ConstKernels.ShadowedConst_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] * 10f, o[i], 4);
    }

    [Fact]
    public void ConstInReductionAndTail_MatchesScalar()
    {
        float[] a = Rand(N, new Random(8));
        float expected = 0f;
        for (int i = 0; i < N; i++)
            expected += a[i] * 2.5f;
        expected += 5.9604645e-8f;

        float actual = ConstKernels.ConstWeightedSum_Simd(a, N);
        Assert.True(MathF.Abs(actual - expected) <= 1e-3f * (1f + MathF.Abs(expected)),
            $"{actual} vs {expected}");
    }

    [Fact]
    public void StaticReadonlyIndexOffset_StaysContiguous()
    {
        float[] a = Rand(N + 8, new Random(9)), o = new float[N];
        ConstKernels.ConstIndexOffset_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i + 2], o[i], 6);
    }

    [Fact]
    public void ConstAsRangeBound_Iterates()
    {
        float[] a = Rand(1024, new Random(15)), o = new float[1024];
        ConstKernels.FixedRange_Simd(a, o);
        for (int i = 0; i < 1000; i++)
            Assert.Equal(a[i] * 6.2831853f, o[i], 4);
        for (int i = 1000; i < 1024; i++)
            Assert.Equal(0f, o[i]);   // past the constant bound, untouched
    }

    [Fact]
    public void DoubleConst_InDoubleKernel()
    {
        double[] a = new double[N], o = new double[N];
        Random r = new Random(10);
        for (int i = 0; i < N; i++)
            a[i] = (r.NextDouble() * 2) - 1;
        ConstKernels.DoubleConst_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] * 0.125, o[i], 12);
    }

    [Fact]
    public void DoubleConst_InFloatKernel_WidensToPairs()
    {
        float[] a = Rand(N, new Random(11)), o = new float[N];
        ConstKernels.DoubleConstInFloatKernel_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal((float)(a[i] * 0.125), o[i], 6);
    }

    [Fact]
    public void LongConst_InIntKernel_WidensToPairs()
    {
        int[] a = new int[N], o = new int[N];
        Random r = new Random(16);
        for (int i = 0; i < N; i++)
            a[i] = r.Next(int.MinValue, int.MaxValue);
        ConstKernels.LongConstInIntKernel_Simd(a, o, N);
        for (int i = 0; i < N; i++)
            Assert.Equal((int)(unchecked(a[i] * 0x5555555555555555L) >> 32), o[i]);
    }

    [Fact]
    public void LongConst_InLongKernel()
    {
        long[] a = new long[N], o = new long[N];
        Random r = new Random(12);
        for (int i = 0; i < N; i++)
            a[i] = ((long)r.Next() << 32) | (uint)r.Next();
        ConstKernels.LongConst_Simd(a, o, N);
        for (int i = 0; i < N; i++)
        {
            long x = a[i] ^ (a[i] >> 9);
            Assert.Equal(unchecked(x * 0x5555555555555555L), o[i]);
        }
    }

    [Fact]
    public void ConstsInSpmdFunction_MatchScalar()
    {
        int[] bits = new int[N];
        Random r = new Random(13);
        for (int i = 0; i < N; i++)
            bits[i] = r.Next(int.MinValue, int.MaxValue);

        float[] outA = new float[N], outB = new float[N];
        ConstKernels.UnitsFromBits_Simd(bits, outA, outB, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(UnitScalar(bits[i]) * 6.2831853f, outA[i], 5);
            Assert.Equal(UnitScalar(bits[i] + 1) * 2.5f, outB[i], 5);
        }
    }

    [Fact]
    public void ConstNameMatchingStructField_ResolvesBoth()
    {
        int[] a = new int[N], w0 = new int[N], w1 = new int[N];
        Random r = new Random(17);
        for (int i = 0; i < N; i++)
            a[i] = r.Next(int.MinValue, int.MaxValue - 1);

        ConstKernels.ConstNameMatchesField_Simd(a, w0, w1, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(unchecked(a[i] + (int)0x9E3779B9), w0[i]);
            Assert.Equal(unchecked(a[i] + 1 + (int)0xBB67AE85), w1[i]);
        }
    }

    [Fact]
    public void CrossClassHelpers_ResolveByQualifiedName()
    {
        float[] a = Rand(N, new Random(18)), b = Rand(N, new Random(19)), o = new float[N];
        int[] steps = new int[N];

        ConstKernels.CrossClassHelpers_Simd(a, b, o, steps, N);
        for (int i = 0; i < N; i++)
        {
            float expected = (a[i] * 0.25f) + (b[i] * 0.75f);
            Assert.True(MathF.Abs(o[i] - expected) <= 1e-6f * (1f + MathF.Abs(expected)),
                $"i={i}: {o[i]} vs {expected}");
            Assert.Equal((i + 3) + (i * 3), steps[i]);
        }
    }

    [Fact]
    public void ParallelVariant_AgreesWithSerial()
    {
        float[] a = Rand(N, new Random(14)), serial = new float[N], parallel = new float[N];
        ConstKernels.ScaleByConst_Simd(a, serial, N);
        ConstKernels.ScaleByConst_ParallelSimd(a, parallel, N);
        Assert.Equal(serial, parallel);
    }
}
