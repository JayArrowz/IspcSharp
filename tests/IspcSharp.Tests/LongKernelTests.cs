using System;
using System.Linq;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// [Spmd] kernels over 64-bit integer (long) lanes, long[], long[,], and long locals.
/// A kernel with long buffers runs as a 64-bit gang (VLong/VMaskD), same lane count as double.
/// </summary>
public static partial class LongKernels
{
    /// <summary>
    /// Arithmetic + uniform long param.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="factor"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Scale(long[] input, long[] output, long factor, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = (input[i] * factor) + 7;
        }
    }

    /// <summary>
    /// Reduction: sum (long accumulator over long[]).
    /// </summary>
    /// <param name="input"></param>
    /// <param name="result"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Sum(long[] input, long[] result, int count)
    {
        long sum = 0;
        foreach (int i in Spmd.Range(count))
        {
            sum += input[i];
        }

        result[0] = sum;
    }

    /// <summary>
    /// Reduction: max (long accumulator over long[]).
    /// </summary>
    /// <param name="input"></param>
    /// <param name="result"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void MaxK(long[] input, long[] result, int count)
    {
        long mx = long.MinValue;
        foreach (int i in Spmd.Range(count))
        {
            mx = Math.Max(mx, input[i]);
        }

        result[0] = mx;
    }

    /// <summary>
    /// Bit-manipulation hash: xor-shift + multiply (uniform shift counts, hex long literal).
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Hash(long[] input, long[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            long x = input[i];
            x ^= x >> 33;
            x *= 0x5555555555555555L;
            x ^= x >> 29;
            output[i] = x;
        }
    }

    /// <summary>
    /// Integer divide + remainder on long lanes.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="q"></param>
    /// <param name="r"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void DivMod(long[] a, long[] b, long[] q, long[] r, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            q[i] = a[i] / b[i];
            r[i] = a[i] % b[i];
        }
    }

    /// <summary>
    /// Gather with a long index; masked (only where in range).
    /// </summary>
    /// <param name="table"></param>
    /// <param name="indices"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void GatherIndexed(long[] table, long[] indices, long[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            long j = indices[i];
            output[i] = table[j] * 2;
        }
    }

    /// <summary>
    /// Scatter with a long index.
    /// </summary>
    /// <param name="values"></param>
    /// <param name="indices"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void ScatterIndexed(long[] values, long[] indices, long[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            long j = indices[i];
            output[j] = values[i] * 3;
        }
    }

    /// <summary>
    /// Cross-gang-width: a 64-bit (long) accumulator inside a 32-bit int/float kernel.
    /// These kernels have NO long buffer (they return the scalar, like the double PreciseSum),
    /// so they run as 32-bit int/float gangs; the long locals/accumulators widen into VLong2
    /// pairs, the integer mirror of VDouble2.
    /// Exact long sum over int[] data, the headline VLong2 case. A plain int accumulator would
    /// overflow past ~2 billion; the long accumulator does not.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    [Spmd]
    public static long WideSum(int[] input, int count)
    {
        long sum = 0;
        foreach (int i in Spmd.Range(count))
        {
            sum += input[i];
        }

        return sum;
    }

    /// <summary>
    /// Exact sum of products over int[], each product needs 64 bits, so it is widened before
    /// accumulating ('long p = (long)a[i] * b[i];' is an overflow-free multiply).
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    [Spmd]
    public static long WideDot(int[] a, int[] b, int count)
    {
        long sum = 0;
        foreach (int i in Spmd.Range(count))
        {
            long p = (long)a[i] * b[i];
            sum += p;
        }

        return sum;
    }

    /// <summary>
    /// Reduce max in a long accumulator over int[] data.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    [Spmd]
    public static long WideMax(int[] input, int count)
    {
        long mx = long.MinValue;
        foreach (int i in Spmd.Range(count))
        {
            mx = Math.Max(mx, input[i]);
        }

        return mx;
    }

    /// <summary>
    /// Long local as a 64-bit intermediate, narrowed back to the int[] output, the integer
    /// analog of DoubleIntermediate. '(long)a[i] * b[i]' can't overflow before the >> 16.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void WideMul(int[] a, int[] b, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            long p = (long)a[i] * b[i];
            output[i] = (int)(p >> 16);
        }
    }

    /// <summary>
    /// 2-D long matrix multiplication (both a[row,k] contiguous and b[k,col] gather).
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="c"></param>
    /// <param name="n"></param>
    /// <param name="m"></param>
    /// <param name="p"></param>
    [Spmd]
    public static void MatMul(long[,] a, long[,] b, long[,] c, int n, int m, int p)
    {
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < p; col++)
            {
                long sum = 0;
                foreach (int k in Spmd.Range(m))
                {
                    sum += a[row, k] * b[k, col];
                }

                c[row, col] = sum;
            }
        }
    }
}

public class LongKernelTests
{
    private static long[] MakeLong(int n, Func<int, long> gen)
    {
        long[] a = new long[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    private const int TailCount = (8192 * 3) + 5;   // not a multiple of any lane count

    [Fact]
    public void Scale_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(1);
        long[] input = MakeLong(n, _ => rng.NextInt64(-1_000_000_000L, 1_000_000_000L));
        long[] expected = new long[n];
        long[] actual = new long[n];
        long factor = 1_000_003L;

        for (int i = 0; i < n; i++)
            expected[i] = (input[i] * factor) + 7;

        LongKernels.Scale_Simd(input, actual, factor, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void Scale_ParallelSimd_MatchesScalar()
    {
        int n = 1_000_003;
        Random rng = new Random(1);
        long[] input = MakeLong(n, _ => rng.NextInt64(-1_000_000_000L, 1_000_000_000L));
        long[] expected = new long[n];
        long[] actual = new long[n];
        long factor = 1_000_003L;

        for (int i = 0; i < n; i++)
            expected[i] = (input[i] * factor) + 7;

        LongKernels.Scale_ParallelSimd(input, actual, factor, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void Sum_Simd_MatchesScalar_Exactly()
    {
        int n = TailCount;
        Random rng = new Random(2);
        long[] input = MakeLong(n, _ => rng.NextInt64(-1_000_000L, 1_000_000L));
        long[] result = new long[1];

        long expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];

        LongKernels.Sum_Simd(input, result, n);

        // Integer sums are exact regardless of reassociation.
        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void Sum_ParallelSimd_MatchesScalar_Exactly()
    {
        int n = 1_000_003;
        Random rng = new Random(2);
        long[] input = MakeLong(n, _ => rng.NextInt64(-1_000_000L, 1_000_000L));
        long[] result = new long[1];

        long expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];

        LongKernels.Sum_ParallelSimd(input, result, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void MaxK_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(3);
        long[] input = MakeLong(n, _ => rng.NextInt64(long.MinValue / 2, long.MaxValue / 2));
        long[] result = new long[1];

        long expected = long.MinValue;
        for (int i = 0; i < n; i++)
            expected = Math.Max(expected, input[i]);

        LongKernels.MaxK_Simd(input, result, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void Hash_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(4);
        long[] input = MakeLong(n, _ => rng.NextInt64());
        long[] expected = new long[n];
        long[] actual = new long[n];

        for (int i = 0; i < n; i++)
        {
            long x = input[i];
            x ^= x >> 33;
            x *= 0x5555555555555555L;
            x ^= x >> 29;
            expected[i] = x;
        }

        LongKernels.Hash_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void DivMod_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(5);
        long[] a = MakeLong(n, _ => rng.NextInt64(-1_000_000_000L, 1_000_000_000L));
        long[] b = MakeLong(n, _ => rng.NextInt64(1, 1000) * (rng.Next(2) == 0 ? 1 : -1));  // nonzero
        long[] q = new long[n];
        long[] r = new long[n];

        LongKernels.DivMod_Simd(a, b, q, r, n);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(a[i] / b[i], q[i]);
            Assert.Equal(a[i] % b[i], r[i]);
        }
    }

    [Fact]
    public void GatherIndexed_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(6);
        long[] table = MakeLong(256, i => i * 1_000_000L);
        long[] indices = MakeLong(n, _ => rng.Next(0, 256));
        long[] expected = new long[n];
        long[] actual = new long[n];

        for (int i = 0; i < n; i++)
            expected[i] = table[indices[i]] * 2;

        LongKernels.GatherIndexed_Simd(table, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void ScatterIndexed_Simd_MatchesScalar()
    {
        int n = 10_007;
        Random rng = new Random(7);
        long[] values = MakeLong(n, i => i * 100L);
        int[] perm = [.. Enumerable.Range(0, n).OrderBy(_ => rng.Next())];
        long[] indices = MakeLong(n, i => perm[i]);
        long[] expected = new long[n];
        long[] actual = new long[n];

        for (int i = 0; i < n; i++)
            expected[indices[i]] = values[i] * 3;

        LongKernels.ScatterIndexed_Simd(values, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void MatMul_Simd_MatchesScalar_Exactly()
    {
        int n = 31, m = 53, p = 17;
        Random rng = new Random(8);
        long[,] a = new long[n, m];
        long[,] b = new long[m, p];
        for (int r = 0; r < n; r++)
        {
            for (int k = 0; k < m; k++)
                a[r, k] = rng.Next(-50, 50);
        }

        for (int k = 0; k < m; k++)
        {
            for (int col = 0; col < p; col++)
                b[k, col] = rng.Next(-50, 50);
        }

        long[,] expected = new long[n, p];
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < p; col++)
            {
                long sum = 0;
                for (int k = 0; k < m; k++)
                    sum += a[row, k] * b[k, col];
                expected[row, col] = sum;
            }
        }

        long[,] actual = new long[n, p];
        LongKernels.MatMul_Simd(a, b, actual, n, m, p);

        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < p; col++)
                Assert.Equal(expected[row, col], actual[row, col]);
        }
    }

    private static int[] MakeInt(int n, Func<int, int> gen)
    {
        int[] a = new int[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    [Fact]
    public void WideSum_Simd_MatchesScalar_Exactly()
    {
        int n = TailCount;
        Random rng = new Random(20);
        // Values chosen so the true sum overflows a 32-bit int (~n * 1e6 ≈ 2.4e10 > int.MaxValue),
        // proving the accumulation really happens in 64 bits.
        int[] input = MakeInt(n, _ => rng.Next(500_000, 1_000_000));

        long expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];
        Assert.True(expected > int.MaxValue);   // sanity: an int accumulator would have overflowed

        long actual = LongKernels.WideSum_Simd(input, n);

        Assert.Equal(expected, actual);   // integer sums are exact regardless of reassociation
    }

    [Fact]
    public void WideSum_ParallelSimd_MatchesScalar_Exactly()
    {
        int n = 1_000_003;
        Random rng = new Random(20);
        int[] input = MakeInt(n, _ => rng.Next(500_000, 1_000_000));

        long expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];
        Assert.True(expected > int.MaxValue);

        long actual = LongKernels.WideSum_ParallelSimd(input, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void WideDot_Simd_MatchesScalar_Exactly()
    {
        int n = TailCount;
        Random rng = new Random(21);
        // Each product is up to ~1e6 * 1e6 = 1e12, far past 32-bit range, so the widening
        // multiply and 64-bit accumulate both matter.
        int[] a = MakeInt(n, _ => rng.Next(-1_000_000, 1_000_000));
        int[] b = MakeInt(n, _ => rng.Next(-1_000_000, 1_000_000));

        long expected = 0;
        for (int i = 0; i < n; i++)
            expected += (long)a[i] * b[i];

        long actual = LongKernels.WideDot_Simd(a, b, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void WideMax_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(22);
        int[] input = MakeInt(n, _ => rng.Next(int.MinValue, int.MaxValue));

        long expected = long.MinValue;
        for (int i = 0; i < n; i++)
            expected = Math.Max(expected, input[i]);

        long actual = LongKernels.WideMax_Simd(input, n);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void WideMul_Simd_MatchesScalar_Exactly()
    {
        int n = TailCount;
        Random rng = new Random(23);
        int[] a = MakeInt(n, _ => rng.Next(-2_000_000, 2_000_000));
        int[] b = MakeInt(n, _ => rng.Next(-2_000_000, 2_000_000));
        int[] expected = new int[n];
        int[] actual = new int[n];

        for (int i = 0; i < n; i++)
            expected[i] = (int)(((long)a[i] * b[i]) >> 16);

        LongKernels.WideMul_Simd(a, b, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }
}
