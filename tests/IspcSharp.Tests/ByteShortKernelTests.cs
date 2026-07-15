using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// [Spmd] kernels over byte[] and short[] buffers, ISPC-style narrow types: the gang width
/// stays the full int-gang width, loads widen (vpmovzxbd/vpmovsxwd) into int lanes, compute
/// follows C#'s own byte/short-to-int promotion, and stores truncate back (StoreNarrow).
/// Narrow buffers mix freely with int/float buffers in the same kernel.
/// </summary>
public static partial class ByteShortKernels
{
    /// <summary>
    /// Byte image brighten with clamp: widening load, uniform byte param, int-lane
    /// Math.Min, narrowing cast + store.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="amount"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Brighten(byte[] input, byte[] output, byte amount, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (byte)Math.Min(input[i] + amount, 255);
        }
    }

    /// <summary>
    /// Divergent if/else over byte lanes: masked narrowing stores.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="cutoff"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void ThresholdBytes(byte[] input, byte[] output, int cutoff, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            if (input[i] > cutoff)
                output[i] = 255;
            else
                output[i] = 0;
        }
    }

    /// <summary>
    /// Mixed byte[] + int[] buffers in one kernel: int reduction over byte data.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="result"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void SumBytes(byte[] input, int[] result, int count)
    {
        int sum = 0;
        foreach (int i in Spmd.Range(count))
        {
            sum += input[i];
        }

        result[0] = sum;
    }

    /// <summary>
    /// Mixed byte[] + float[] buffers: normalize pixels to [0, 1].
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Normalize(byte[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = input[i] * (1f / 255f);
        }
    }

    /// <summary>
    /// Byte lane local with wrapping cast arithmetic.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void LocalWrap(byte[] input, byte[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            byte x = input[i];
            x = (byte)((x * 5) + 200);
            output[i] = x;
        }
    }

    /// <summary>
    /// Compound store into a byte buffer (C#'s implicit narrowing on 'buf[i] += x').
    /// </summary>
    /// <param name="output"></param>
    /// <param name="delta"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void AddInPlace(byte[] output, byte delta, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] += delta;
        }
    }

    /// <summary>
    /// Q15 fixed-point multiply on short lanes: widening loads, int multiply, shift, narrow.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Q15Mul(short[] a, short[] b, short[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (short)((a[i] * b[i]) >> 15);
        }
    }

    /// <summary>
    /// Short truncation parity: values overflow short range and must wrap like the scalar cast.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void WrapCast(short[] input, short[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (short)((input[i] * 3) + 1);
        }
    }

    /// <summary>
    /// Min/max reductions with int accumulators over short data.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="result"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void MinMaxShorts(short[] input, int[] result, int count)
    {
        int mn = int.MaxValue;
        int mx = int.MinValue;
        foreach (int i in Spmd.Range(count))
        {
            mn = Math.Min(mn, input[i]);
            mx = Math.Max(mx, input[i]);
        }

        result[0] = mn;
        result[1] = mx;
    }

    /// <summary>
    /// All three widths in one kernel: byte weights, short samples, float output.
    /// </summary>
    /// <param name="samples"></param>
    /// <param name="weights"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void WeightedMix(short[] samples, byte[] weights, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = samples[i] * (weights[i] * (1f / 255f));
        }
    }
}

public class ByteShortKernelTests
{
    private static byte[] Bytes(int n, Func<int, byte> gen)
    {
        byte[] a = new byte[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    private static short[] Shorts(int n, Func<int, short> gen)
    {
        short[] a = new short[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    // Deliberately not a multiple of the gang width, so the scalar tail runs too.
    private static int OddCount => (VInt.LaneCount * 3) + 3;

    [Fact]
    public void Brighten_MatchesScalar_AndClamps()
    {
        int n = OddCount;
        byte[] input = Bytes(n, i => (byte)(i * 17));
        byte[] expected = new byte[n];
        byte[] actual = new byte[n];
        ByteShortKernels.Brighten(input, expected, 90, n);

        ByteShortKernels.Brighten_Simd(input, actual, 90, n);

        Assert.Equal(expected, actual);
        Assert.Contains(actual, v => v == 255); // clamp actually engaged
    }

    [Fact]
    public void Brighten_ParallelSimd_MatchesScalar()
    {
        int n = OddCount;
        byte[] input = Bytes(n, i => (byte)(255 - (i % 256)));
        byte[] expected = new byte[n];
        byte[] actual = new byte[n];
        ByteShortKernels.Brighten(input, expected, 40, n);

        ByteShortKernels.Brighten_ParallelSimd(input, actual, 40, n, minChunkSize: VInt.LaneCount);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ThresholdBytes_MaskedNarrowStores_MatchScalar()
    {
        int n = OddCount;
        byte[] input = Bytes(n, i => (byte)((i * 31) % 256));
        byte[] expected = new byte[n];
        byte[] actual = new byte[n];
        ByteShortKernels.ThresholdBytes(input, expected, 128, n);

        ByteShortKernels.ThresholdBytes_Simd(input, actual, 128, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void SumBytes_IntReductionOverBytes()
    {
        int n = OddCount;
        byte[] input = Bytes(n, i => (byte)((i * 7) + 200)); // sums far past byte range
        int[] expected = new int[1];
        int[] actual = new int[1];
        ByteShortKernels.SumBytes(input, expected, n);

        ByteShortKernels.SumBytes_Simd(input, actual, n);

        Assert.Equal(expected[0], actual[0]);

        int[] parallel = new int[1];
        ByteShortKernels.SumBytes_ParallelSimd(input, parallel, n, minChunkSize: VInt.LaneCount);
        Assert.Equal(expected[0], parallel[0]);
    }

    [Fact]
    public void Normalize_ByteToFloat()
    {
        int n = OddCount;
        byte[] input = Bytes(n, i => (byte)(i % 256));
        float[] expected = new float[n];
        float[] actual = new float[n];
        ByteShortKernels.Normalize(input, expected, n);

        ByteShortKernels.Normalize_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void LocalWrap_ByteLocalWrapsLikeScalar()
    {
        int n = OddCount;
        byte[] input = Bytes(n, i => (byte)(i * 13));
        byte[] expected = new byte[n];
        byte[] actual = new byte[n];
        ByteShortKernels.LocalWrap(input, expected, n);

        ByteShortKernels.LocalWrap_Simd(input, actual, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void AddInPlace_CompoundNarrowingStore()
    {
        int n = OddCount;
        byte[] expected = Bytes(n, i => (byte)(i * 11));
        byte[] actual = (byte[])expected.Clone();
        ByteShortKernels.AddInPlace(expected, 200, n); // wraps past 255 in many lanes

        ByteShortKernels.AddInPlace_Simd(actual, 200, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Q15Mul_MatchesScalar()
    {
        int n = OddCount;
        short[] a = Shorts(n, i => (short)((i * 1103) - 9000));
        short[] b = Shorts(n, i => (short)((i * 797) + 123));
        short[] expected = new short[n];
        short[] actual = new short[n];
        ByteShortKernels.Q15Mul(a, b, expected, n);

        ByteShortKernels.Q15Mul_Simd(a, b, actual, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void WrapCast_ShortTruncationParity()
    {
        int n = OddCount;
        short[] input = Shorts(n, i => (short)(15000 + (i * 500))); // *3 overflows short
        short[] expected = new short[n];
        short[] actual = new short[n];
        ByteShortKernels.WrapCast(input, expected, n);

        ByteShortKernels.WrapCast_Simd(input, actual, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void MinMaxShorts_Reductions()
    {
        int n = OddCount;
        short[] input = Shorts(n, i => (short)(((i * 313) % 2000) - 1000));
        int[] expected = new int[2];
        int[] actual = new int[2];
        ByteShortKernels.MinMaxShorts(input, expected, n);

        ByteShortKernels.MinMaxShorts_Simd(input, actual, n);

        Assert.Equal(expected[0], actual[0]);
        Assert.Equal(expected[1], actual[1]);
    }

    [Fact]
    public void WeightedMix_ByteShortFloatInOneKernel()
    {
        int n = OddCount;
        short[] samples = Shorts(n, i => (short)((i * 601) - 4000));
        byte[] weights = Bytes(n, i => (byte)((i * 37) % 256));
        float[] expected = new float[n];
        float[] actual = new float[n];
        ByteShortKernels.WeightedMix(samples, weights, expected, n);

        ByteShortKernels.WeightedMix_Simd(samples, weights, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }
}
