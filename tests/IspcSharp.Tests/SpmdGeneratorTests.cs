using System;
using System.Linq;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Kernels annotated with [Spmd] for the source generator to vectorize.
/// The generator emits ClampScale_Simd, ClampScale_ParallelSimd, etc.
/// at compile time. These tests verify the generated code produces
/// identical results to the scalar reference.
/// </summary>
public static partial class TestKernels
{
    [Spmd]
    public static void ClampScale(float[] input, float[] output, float scale, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (x < 0f)
                x = 0f;
            else if (x > 1f)
                x = 1f;
            output[i] = x * scale;
        }
    }

    [Spmd]
    public static void SumOfSquares(float[] input, float[] result, int count)
    {
        float sum = 0f;
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            sum += x * x;
        }

        result[0] = sum;
    }

    [Spmd]
    public static void NewtonSqrt(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float guess = x;
            float err = 1f;
            while (err > 0.0001f)
            {
                float next = 0.5f * (guess + (x / guess));
                err = MathF.Abs(next - guess);
                guess = next;
            }

            output[i] = guess;
        }
    }

    [Spmd]
    public static void Tonemap(float[] input, float[] output, float exposure, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i] * exposure;
            output[i] = 1f - MathF.Exp(-x);
        }
    }

    [Spmd]
    public static void Threshold(float[] input, float[] output, float lo, float hi, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            output[i] = x < lo ? 0f : x > hi ? 1f : (x - lo) / (hi - lo);
        }
    }

    [Spmd]
    public static void AbsTernary(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            output[i] = x < 0f ? -x : x;
        }
    }

    [Spmd]
    public static void IntScale(int[] input, int[] output, int scale, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int x = input[i];
            output[i] = x * scale;
        }
    }

    [Spmd]
    public static void Accumulate(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            x += 1f;
            x *= 2f;
            output[i] = x;
        }
    }

    [Spmd]
    public static void RangeOffset(float[] input, float[] output, int start, int end)
    {
        foreach (int i in Spmd.Range(start, end))
        {
            output[i] = input[i] * 3f;
        }
    }

    [Spmd]
    public static void SqrtKernel(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = MathF.Sqrt(input[i]);
        }
    }

    [Spmd]
    public static void GatherScatter(float[] input, int[] indices, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int idx = indices[i];
            output[i] = input[idx] * 2f;
        }
    }

    [Spmd]
    public static void GatherAdd(float[] data, int[] indices, float[] output, float addend, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = data[indices[i]] + addend;
        }
    }

    [Spmd]
    public static void ScatterKernel(float[] input, int[] indices, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[indices[i]] = input[i] * 3f;
        }
    }

    [Spmd]
    public static void GatherScatterRoundtrip(float[] table, int[] indices, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[indices[i]] = table[indices[i]] + 1f;
        }
    }

    [Spmd]
    public static void UniformForLoop(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float sum = 0f;
            for (int j = 0; j < 4; j++)
            {
                sum += x;
            }

            output[i] = sum;
        }
    }

    [Spmd]
    public static void UniformForBreak(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float sum = 0f;
            for (int j = 0; j < 100; j++)
            {
                if (sum > 10f)
                    break;
                sum += x;
            }

            output[i] = sum;
        }
    }

    [Spmd]
    public static void UniformForContinue(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float sum = 0f;
            for (int j = 0; j < 10; j++)
            {
                if (j == 5)
                    continue;
                sum += x;
            }

            output[i] = sum;
        }
    }

    [Spmd]
    public static void WhileBreak(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float sum = 0f;
            float j = 0f;
            while (j < 100f)
            {
                if (sum > 10f)
                    break;
                sum += x;
                j += 1f;
            }

            output[i] = sum;
        }
    }

    [Spmd]
    public static void WhileContinue(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float sum = 0f;
            float j = 0f;
            while (j < 10f)
            {
                j += 1f;
                if (j == 5f)
                    continue;
                sum += x;
            }

            output[i] = sum;
        }
    }

    [Spmd]
    public static void BitwiseAnd(int[] input, int[] output, int mask, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int x = input[i];
            output[i] = x & mask;
        }
    }

    [Spmd]
    public static void BitwiseAndCheck(int[] flags, int[] output, int flag, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int f = flags[i];
            output[i] = (f & flag) == 0 ? 1 : 0;
        }
    }

    [Spmd]
    public static void BitwiseOr(int[] input, int[] output, int mask, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int x = input[i];
            output[i] = x | mask;
        }
    }

    [Spmd]
    public static void BitwiseXor(int[] input, int[] output, int mask, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int x = input[i];
            output[i] = x ^ mask;
        }
    }

    [Spmd]
    public static void ForeachContinue(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (x < 0f)
                continue;
            output[i] = x * 2f;
        }
    }

    [Spmd]
    public static void LocalAfterContinue(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (x < 0f)
                continue;
            float y = x * 3f;
            output[i] = y + 1f;
        }
    }

    [Spmd]
    public static void LocalAfterContinueInBlock(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (x < 0f)
            {
                continue;
            }

            float y = x * 3f;
            output[i] = y + 1f;
        }
    }

    [Spmd]
    public static void MultipleContinuesWithLocals(float[] input, int[] flags, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (x < 0f)
                continue;
            int flag = flags[i];
            if ((flag & 0x1) != 0)
                continue;
            float y = x * 2f;
            output[i] = y;
        }
    }

    [Spmd]
    public static void MinReduce(float[] input, float[] result, int count)
    {
        float mn = float.MaxValue;
        foreach (int i in Spmd.Range(count))
        {
            mn = MathF.Min(mn, input[i]);
        }

        result[0] = mn;
    }

    [Spmd]
    public static void MaxReduce(float[] input, float[] result, int count)
    {
        float mx = float.MinValue;
        foreach (int i in Spmd.Range(count))
        {
            mx = MathF.Max(mx, input[i]);
        }

        result[0] = mx;
    }

    [Spmd]
    public static void IntMaxReduce(int[] input, int[] result, int count)
    {
        int mx = int.MinValue;
        foreach (int i in Spmd.Range(count))
        {
            mx = Math.Max(mx, input[i]);
        }

        result[0] = mx;
    }

    [Spmd]
    public static void SumAndMin(float[] input, float[] result, int count)
    {
        float sum = 0f;
        float mn = float.MaxValue;
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            sum += x;
            mn = MathF.Min(mn, x);
        }

        result[0] = sum;
        result[1] = mn;
    }

    [Spmd]
    public static void MaskedMinReduce(float[] input, float[] result, float cutoff, int count)
    {
        float mn = float.MaxValue;
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (x > cutoff)
                mn = MathF.Min(mn, x);
        }

        result[0] = mn;
    }

    [Spmd]
    public static void CoherentClamp(float[] input, float[] output, float cutoff, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            if (Spmd.Coherent(x > cutoff))
            {
                output[i] = MathF.Sqrt(x - cutoff) + 1f;
            }
            else
            {
                output[i] = x * 0.5f;
            }
        }
    }

    [Spmd]
    public static void DoubleClampScale(double[] input, double[] output, double scale, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            double x = input[i];
            if (x < 0.0)
                x = 0.0;
            else if (x > 1.0)
                x = 1.0;
            output[i] = x * scale;
        }
    }

    [Spmd]
    public static void DoubleTonemap(double[] input, double[] output, double exposure, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            double x = input[i] * exposure;
            output[i] = 1.0 - Math.Exp(-x);
        }
    }

    [Spmd]
    public static void DoubleSumOfSquares(double[] input, double[] result, int count)
    {
        double sum = 0.0;
        foreach (int i in Spmd.Range(count))
        {
            double x = input[i];
            sum += x * x;
        }

        result[0] = sum;
    }

    [Spmd]
    public static void Scale2D(float[] input, float[] output, float scale, int width, int height)
    {
        foreach (var (x, y) in Spmd.Range2D(width, height))
        {
            float v = input[(y * width) + x];
            output[(y * width) + x] = (v * scale) + y;
        }
    }

    [Spmd]
    public static void Sum2D(float[] input, float[] result, int width, int height)
    {
        float sum = 0f;
        foreach (var (x, y) in Spmd.Range2D(width, height))
        {
            sum += input[(y * width) + x];
        }

        result[0] = sum;
    }

    [Spmd]
    public static void OffsetRead(float[] input, float[] output, int offset, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = input[i + offset] * 2f;
        }
    }

    [Spmd]
    public static void CountHalvings(float[] input, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            int steps = 0;
            while (x > 1f)
            {
                x *= 0.5f;
                steps++;
            }

            output[i] = steps;
        }
    }

    [Spmd]
    public static void MandelbrotIters(int[] iters, float minX, float minY, float dx, float dy, int maxIter, int width, int height)
    {
        foreach (var (x, y) in Spmd.Range2D(width, height))
        {
            float cx = minX + (x * dx);
            float cy = minY + (y * dy);
            float zx = 0f;
            float zy = 0f;
            int i = 0;
            while ((zx * zx) + (zy * zy) < 4f && i < maxIter)
            {
                float n = (zx * zx) - (zy * zy) + cx;
                zy = (2f * zx * zy) + cy;
                zx = n;
                i++;
            }

            iters[(y * width) + x] = i;
        }
    }

    [Spmd]
    public static void TiledScale(float[] input, float[] output, float scale, int width, int height)
    {
        foreach (var (x, y) in Spmd.Range2DTiled(width, height))
        {
            output[(y * width) + x] = (input[(y * width) + x] * scale) + y;
        }
    }

    [Spmd]
    public static void TiledSum(float[] input, float[] result, int width, int height)
    {
        float sum = 0f;
        foreach (var (x, y) in Spmd.Range2DTiled(width, height, 32, 16))
        {
            sum += input[(y * width) + x];
        }

        result[0] = sum;
    }

    [Spmd]
    public static float SumReturn(float[] input, int count)
    {
        float sum = 0f;
        foreach (int i in Spmd.Range(count))
        {
            sum += input[i];
        }

        return sum;
    }

    [Spmd]
    public static int MaxReturn(int[] input, int count)
    {
        int mx = int.MinValue;
        foreach (int i in Spmd.Range(count))
        {
            mx = Math.Max(mx, input[i]);
        }

        return mx;
    }

    [Spmd]
    public static void IntDivMod(int[] a, int[] b, int[] q, int[] r, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            q[i] = a[i] / b[i];
            r[i] = a[i] % b[i];
        }
    }

    [Spmd]
    public static void MaskedDiv(int[] a, int[] b, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int d = b[i];
            if (d != 0)
                output[i] = a[i] / d;
        }
    }

    [Spmd]
    public static void BitOps(int[] input, int[] output, int shift, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int x = input[i];
            output[i] = ((x << shift) >> 2) ^ (~x & 0xFF);
        }
    }

    [Spmd]
    public static void DoubleMath(double[] input, double[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            double x = input[i];
            output[i] = Math.Sin(x) + (Math.Sqrt(x + 2.0) * Math.Log(x + 3.0)) - Math.Atan(x);
        }
    }

    [Spmd]
    public static void DoubleGather(double[] table, double[] indices, double[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            double j = indices[i];
            output[i] = table[(int)j] * 2.0;
        }
    }

    [Spmd]
    public static void DoubleScatter(double[] values, double[] indices, double[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[(int)indices[i]] = values[i] * 3.0;
        }
    }

    [Spmd]
    public static void VarShift(int[] input, int[] counts, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int x = input[i];
            int c = counts[i];
            output[i] = (x << c) ^ (x >> c) ^ (x >>> c);
        }
    }

    [Spmd]
    public static void DoubleIntermediate(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            double d = ((double)x * 1.0000000001d) + 0.5d;
            output[i] = (float)Math.Sqrt(d);
        }
    }

    [Spmd]
    public static double PreciseSum(float[] input, int count)
    {
        double sum = 0d;
        foreach (int i in Spmd.Range(count))
        {
            sum += input[i];
        }

        return sum;
    }

    // Affine index locals: 'int a = i + lo' → contiguous load, propagating through 'int b = a + 3'.
    [Spmd]
    public static void AffineIndexed(float[] input, float[] output, int lo, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int a = i + lo;      // affine, offset lo
            int b = a + 3;       // affine via propagation, offset lo + 3
            output[i] = input[a] + input[b];
        }
    }

    // Non-affine index local (stride 2) must still work, via gather.
    [Spmd]
    public static void StridedIndexed(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int s = i * 2;       // stride 2 → not contiguous → gather
            output[i] = input[s];
        }
    }

    [Spmd]
    public static void ReturnInLoop(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float acc = 0f;
            float j = 0f;
            while (j < 50f)
            {
                acc += x;
                if (acc > 10f)
                    return;              // this lane is done: skips the write below
                j += 1f;
            }

            output[i] = acc;
        }
    }
}

/// <summary>
/// Tests that the [Spmd] source generator produces correct vectorized output.
/// Each test compares the generated _Simd (and _ParallelSimd where applicable)
/// method against the scalar reference, with element counts that exercise
/// the masked tail path.
/// </summary>
public class SpmdGeneratorTests
{
    private static float[] MakeInput(int n, Func<int, float> gen)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    private static int[] MakeIntInput(int n, Func<int, int> gen)
    {
        int[] a = new int[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    // Use a count that deliberately hits the tail (not a multiple of LaneCount).
    private const int TailCount = 1_000_003;

    [Fact]
    public void ClampScale_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(42).NextDouble() * 2) - 0.5));
        float[] expected = new float[n];
        float[] actual = new float[n];

        // Scalar reference
        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            if (x < 0f)
                x = 0f;
            else if (x > 1f)
                x = 1f;
            expected[i] = x * 2.5f;
        }

        TestKernels.ClampScale_Simd(input, actual, 2.5f, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void ClampScale_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(42).NextDouble() * 2) - 0.5));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            if (x < 0f)
                x = 0f;
            else if (x > 1f)
                x = 1f;
            expected[i] = x * 2.5f;
        }

        TestKernels.ClampScale_ParallelSimd(input, actual, 2.5f, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void SumOfSquares_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(99).NextDouble() * 10));
        float[] resultSimd = new float[1];
        float[] resultScalar = new float[1];

        // Scalar reference
        float sum = 0f;
        for (int i = 0; i < n; i++)
            sum += input[i] * input[i];
        resultScalar[0] = sum;

        TestKernels.SumOfSquares_Simd(input, resultSimd, n);

        // Reductions accumulate in float with different summation order between
        // scalar (sequential) and SIMD (horizontal reduce per gang). For ~1M elements
        // with values up to 10, the sum is ~33M; float precision means ~0.3% drift.
        float ratio = resultSimd[0] / resultScalar[0];
        Assert.True(Math.Abs(ratio - 1f) < 0.01f,
            $"SumOfSquares: scalar={resultScalar[0]}, simd={resultSimd[0]}, ratio={ratio}");
    }

    [Fact]
    public void NewtonSqrt_Simd_MatchesMathSqrt()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(7).NextDouble() * 4) + 0.01));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = MathF.Sqrt(input[i]);

        TestKernels.NewtonSqrt_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 3);
    }

    [Fact]
    public void NewtonSqrt_ParallelSimd_MatchesMathSqrt()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(7).NextDouble() * 4) + 0.01));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = MathF.Sqrt(input[i]);

        TestKernels.NewtonSqrt_ParallelSimd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 3);
    }

    [Fact]
    public void Tonemap_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(33).NextDouble() * 4));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = 1f - MathF.Exp(-input[i] * 1.5f);

        TestKernels.Tonemap_Simd(input, actual, 1.5f, n);

        // VectorMath.Exp is a polynomial approximation.
        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 4);
    }

    [Fact]
    public void Threshold_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(55).NextDouble() * 2));
        float[] expected = new float[n];
        float[] actual = new float[n];
        float lo = 0.3f, hi = 1.5f;

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            expected[i] = x < lo ? 0f : x > hi ? 1f : (x - lo) / (hi - lo);
        }

        TestKernels.Threshold_Simd(input, actual, lo, hi, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void AbsTernary_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(88).NextDouble() * 4) - 2));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[i] < 0f ? -input[i] : input[i];

        TestKernels.AbsTernary_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void IntScale_Simd_MatchesScalar()
    {
        int n = TailCount;
        int[] input = MakeIntInput(n, i => i - (n / 2));
        int[] expected = new int[n];
        int[] actual = new int[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[i] * 3;

        TestKernels.IntScale_Simd(input, actual, 3, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void Accumulate_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(12).NextDouble() * 5));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            x += 1f;
            x *= 2f;
            expected[i] = x;
        }

        TestKernels.Accumulate_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void RangeOffset_Simd_MatchesScalar()
    {
        int n = 10_000;
        int start = 100, end = start + n;
        float[] input = MakeInput(end, i => i * 0.5f);
        float[] expected = new float[end];
        float[] actual = new float[end];

        for (int i = start; i < end; i++)
            expected[i] = input[i] * 3f;

        TestKernels.RangeOffset_Simd(input, actual, start, end);

        for (int i = start; i < end; i++)
            Assert.Equal(expected[i], actual[i]);
        // Before start should be untouched (0).
        for (int i = 0; i < start; i++)
            Assert.Equal(0f, actual[i]);
    }

    [Fact]
    public void SqrtKernel_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(66).NextDouble() * 9) + 0.1));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = MathF.Sqrt(input[i]);

        TestKernels.SqrtKernel_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 3);
    }

    [Fact]
    public void ClampScale_Simd_SmallCount()
    {
        int n = 3; // less than LaneCount
        float[] input = [-0.5f, 0.5f, 1.5f];
        float[] expected = [0f, 0.5f * 2f, 1f * 2f];
        float[] actual = new float[n];

        TestKernels.ClampScale_Simd(input, actual, 2f, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void ClampScale_Simd_ExactMultiple()
    {
        int n = Spmd.LaneCount * 4;
        float[] input = MakeInput(n, i => (float)((new Random(1).NextDouble() * 2) - 0.5));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            if (x < 0f)
                x = 0f;
            else if (x > 1f)
                x = 1f;
            expected[i] = x * 2f;
        }

        TestKernels.ClampScale_Simd(input, actual, 2f, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void GatherScatter_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => i * 1.5f);
        int[] indices = MakeIntInput(n, i => i * 7 % n); // pseudo-random indices
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[indices[i]] * 2f;

        TestKernels.GatherScatter_Simd(input, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void GatherScatter_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => i * 1.5f);
        int[] indices = MakeIntInput(n, i => i * 7 % n);
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[indices[i]] * 2f;

        TestKernels.GatherScatter_ParallelSimd(input, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void GatherAdd_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] data = MakeInput(n, i => (float)(new Random(42).NextDouble() * 100));
        int[] indices = MakeIntInput(n, i => ((i * 13) + 3) % n);
        float[] expected = new float[n];
        float[] actual = new float[n];
        float addend = 5.5f;

        for (int i = 0; i < n; i++)
            expected[i] = data[indices[i]] + addend;

        TestKernels.GatherAdd_Simd(data, indices, actual, addend, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void ScatterKernel_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(99).NextDouble() * 10));
        // Use unique indices to avoid race conditions in scatter
        int[] indices = MakeIntInput(n, i => i);
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[indices[i]] = input[i] * 3f;

        TestKernels.ScatterKernel_Simd(input, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void GatherScatterRoundtrip_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] table = MakeInput(n, i => (float)(new Random(77).NextDouble() * 50));
        int[] indices = MakeIntInput(n, i => ((i * 17) + 5) % n);
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[indices[i]] = table[indices[i]] + 1f;

        TestKernels.GatherScatterRoundtrip_Simd(table, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void UniformForLoop_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(3).NextDouble() * 3));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[i] * 4; // 4 iterations of sum += x

        TestKernels.UniformForLoop_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 3);
    }

    [Fact]
    public void UniformForBreak_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(5).NextDouble() * 3));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            float sum = 0f;
            for (int j = 0; j < 100; j++)
            {
                if (sum > 10f)
                    break;
                sum += x;
            }

            expected[i] = sum;
        }

        TestKernels.UniformForBreak_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 2);
    }

    [Fact]
    public void UniformForContinue_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(7).NextDouble() * 3));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            float sum = 0f;
            for (int j = 0; j < 10; j++)
            {
                if (j == 5)
                    continue;
                sum += x;
            }

            expected[i] = sum; // 9 iterations (skipped j==5)
        }

        TestKernels.UniformForContinue_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 2);
    }

    [Fact]
    public void WhileBreak_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(9).NextDouble() * 3));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            float sum = 0f;
            float j = 0f;
            while (j < 100f)
            {
                if (sum > 10f)
                    break;
                sum += x;
                j += 1f;
            }

            expected[i] = sum;
        }

        TestKernels.WhileBreak_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 2);
    }

    [Fact]
    public void WhileContinue_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(11).NextDouble() * 3));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            float sum = 0f;
            float j = 0f;
            while (j < 10f)
            {
                j += 1f;
                if (j == 5f)
                    continue;
                sum += x;
            }

            expected[i] = sum; // 9 iterations (skipped j==5)
        }

        TestKernels.WhileContinue_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 2);
    }

    [Fact]
    public void BitwiseAnd_Simd_MatchesScalar()
    {
        int n = TailCount;
        int[] input = MakeIntInput(n, i => (i * 3) + 7);
        int[] expected = new int[n];
        int[] actual = new int[n];
        int mask = 0xFF;

        for (int i = 0; i < n; i++)
            expected[i] = input[i] & mask;

        TestKernels.BitwiseAnd_Simd(input, actual, mask, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void BitwiseAndCheck_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(42);
        int[] flags = MakeIntInput(n, _ => rng.Next(256));
        int[] expected = new int[n];
        int[] actual = new int[n];
        int flag = 0x10; // BLOCK_SW

        for (int i = 0; i < n; i++)
            expected[i] = (flags[i] & flag) == 0 ? 1 : 0;

        TestKernels.BitwiseAndCheck_Simd(flags, actual, flag, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void BitwiseOr_Simd_MatchesScalar()
    {
        int n = TailCount;
        int[] input = MakeIntInput(n, i => i * 2);
        int[] expected = new int[n];
        int[] actual = new int[n];
        int mask = 0x100;

        for (int i = 0; i < n; i++)
            expected[i] = input[i] | mask;

        TestKernels.BitwiseOr_Simd(input, actual, mask, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void BitwiseXor_Simd_MatchesScalar()
    {
        int n = TailCount;
        int[] input = MakeIntInput(n, i => i ^ 0xAA);
        int[] expected = new int[n];
        int[] actual = new int[n];
        int mask = 0xFF;

        for (int i = 0; i < n; i++)
            expected[i] = input[i] ^ mask;

        TestKernels.BitwiseXor_Simd(input, actual, mask, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void ForeachContinue_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(22).NextDouble() * 4) - 2));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            if (x < 0f)
                continue;
            expected[i] = x * 2f;
        }

        TestKernels.ForeachContinue_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void LocalAfterContinue_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(33).NextDouble() * 4) - 2));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            if (x < 0f)
                continue;
            float y = x * 3f;
            expected[i] = y + 1f;
        }

        TestKernels.LocalAfterContinue_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void LocalAfterContinueInBlock_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(44).NextDouble() * 4) - 2));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            if (x < 0f)
            {
                continue;
            }

            float y = x * 3f;
            expected[i] = y + 1f;
        }

        TestKernels.LocalAfterContinueInBlock_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void MultipleContinuesWithLocals_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(55).NextDouble() * 4) - 2));
        Random rng = new Random(66);
        int[] flags = MakeIntInput(n, _ => rng.Next(256));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            if (x < 0f)
                continue;
            int flag = flags[i];
            if ((flag & 0x1) != 0)
                continue;
            float y = x * 2f;
            expected[i] = y;
        }

        TestKernels.MultipleContinuesWithLocals_Simd(input, flags, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void MinReduce_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(11).NextDouble() * 100) - 50));
        float[] result = new float[1];

        float expected = float.MaxValue;
        for (int i = 0; i < n; i++)
            expected = MathF.Min(expected, input[i]);

        TestKernels.MinReduce_Simd(input, result, n);

        // Min/max reductions are exact, no floating-point reassociation drift.
        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void MinReduce_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(11).NextDouble() * 100) - 50));
        float[] result = new float[1];

        float expected = float.MaxValue;
        for (int i = 0; i < n; i++)
            expected = MathF.Min(expected, input[i]);

        TestKernels.MinReduce_ParallelSimd(input, result, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void MaxReduce_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(12).NextDouble() * 100) - 50));
        float[] result = new float[1];

        float expected = float.MinValue;
        for (int i = 0; i < n; i++)
            expected = MathF.Max(expected, input[i]);

        TestKernels.MaxReduce_Simd(input, result, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void MaxReduce_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(12).NextDouble() * 100) - 50));
        float[] result = new float[1];

        float expected = float.MinValue;
        for (int i = 0; i < n; i++)
            expected = MathF.Max(expected, input[i]);

        TestKernels.MaxReduce_ParallelSimd(input, result, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void IntMaxReduce_Simd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(13);
        int[] input = MakeIntInput(n, _ => rng.Next(-1000000, 1000000));
        int[] result = new int[1];

        int expected = int.MinValue;
        for (int i = 0; i < n; i++)
            expected = Math.Max(expected, input[i]);

        TestKernels.IntMaxReduce_Simd(input, result, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void IntMaxReduce_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        Random rng = new Random(13);
        int[] input = MakeIntInput(n, _ => rng.Next(-1000000, 1000000));
        int[] result = new int[1];

        int expected = int.MinValue;
        for (int i = 0; i < n; i++)
            expected = Math.Max(expected, input[i]);

        TestKernels.IntMaxReduce_ParallelSimd(input, result, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void SumOfSquares_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(99).NextDouble() * 10));
        float[] result = new float[1];

        float expected = 0f;
        for (int i = 0; i < n; i++)
            expected += input[i] * input[i];

        TestKernels.SumOfSquares_ParallelSimd(input, result, n);

        // Chunked + horizontal summation reassociates float adds; allow small drift.
        float ratio = result[0] / expected;
        Assert.True(Math.Abs(ratio - 1f) < 0.01f,
            $"SumOfSquares parallel: scalar={expected}, simd={result[0]}, ratio={ratio}");
    }

    [Fact]
    public void SumAndMin_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(21).NextDouble() * 4) - 2));
        float[] result = new float[2];

        float expectedSum = 0f, expectedMin = float.MaxValue;
        for (int i = 0; i < n; i++)
        {
            expectedSum += input[i];
            expectedMin = MathF.Min(expectedMin, input[i]);
        }

        TestKernels.SumAndMin_Simd(input, result, n);

        float ratio = result[0] / expectedSum;
        Assert.True(Math.Abs(ratio - 1f) < 0.01f,
            $"SumAndMin sum: scalar={expectedSum}, simd={result[0]}");
        Assert.Equal(expectedMin, result[1]);
    }

    [Fact]
    public void SumAndMin_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)((new Random(21).NextDouble() * 4) - 2));
        float[] result = new float[2];

        float expectedSum = 0f, expectedMin = float.MaxValue;
        for (int i = 0; i < n; i++)
        {
            expectedSum += input[i];
            expectedMin = MathF.Min(expectedMin, input[i]);
        }

        TestKernels.SumAndMin_ParallelSimd(input, result, n);

        float ratio = result[0] / expectedSum;
        Assert.True(Math.Abs(ratio - 1f) < 0.01f,
            $"SumAndMin parallel sum: scalar={expectedSum}, simd={result[0]}");
        Assert.Equal(expectedMin, result[1]);
    }

    [Fact]
    public void MaskedMinReduce_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(31).NextDouble() * 10));
        float[] result = new float[1];
        float cutoff = 5f;

        float expected = float.MaxValue;
        for (int i = 0; i < n; i++)
        {
            if (input[i] > cutoff)
                expected = MathF.Min(expected, input[i]);
        }

        TestKernels.MaskedMinReduce_Simd(input, result, cutoff, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void MaskedMinReduce_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(31).NextDouble() * 10));
        float[] result = new float[1];
        float cutoff = 5f;

        float expected = float.MaxValue;
        for (int i = 0; i < n; i++)
        {
            if (input[i] > cutoff)
                expected = MathF.Min(expected, input[i]);
        }

        TestKernels.MaskedMinReduce_ParallelSimd(input, result, cutoff, n);

        Assert.Equal(expected, result[0]);
    }

    [Fact]
    public void CoherentClamp_Simd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(41).NextDouble() * 4));
        float[] expected = new float[n];
        float[] actual = new float[n];
        float cutoff = 2f;

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            expected[i] = x > cutoff ? MathF.Sqrt(x - cutoff) + 1f : x * 0.5f;
        }

        TestKernels.CoherentClamp_Simd(input, actual, cutoff, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 4);
    }

    [Fact]
    public void CoherentClamp_Simd_BranchNeverTaken_MatchesScalar()
    {
        // Every gang skips the then-branch entirely (the cif fast path).
        int n = (Spmd.LaneCount * 5) + 3;
        float[] input = MakeInput(n, i => i * 0.01f);           // all well below cutoff
        float[] expected = new float[n];
        float[] actual = new float[n];
        float cutoff = 100f;

        for (int i = 0; i < n; i++)
            expected[i] = input[i] * 0.5f;

        TestKernels.CoherentClamp_Simd(input, actual, cutoff, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void CoherentClamp_ParallelSimd_MatchesScalar()
    {
        int n = TailCount;
        float[] input = MakeInput(n, i => (float)(new Random(41).NextDouble() * 4));
        float[] expected = new float[n];
        float[] actual = new float[n];
        float cutoff = 2f;

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            expected[i] = x > cutoff ? MathF.Sqrt(x - cutoff) + 1f : x * 0.5f;
        }

        TestKernels.CoherentClamp_ParallelSimd(input, actual, cutoff, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i], 4);
    }

    private static double[] MakeDoubleInput(int n, Func<int, double> gen)
    {
        double[] a = new double[n];
        for (int i = 0; i < n; i++)
            a[i] = gen(i);
        return a;
    }

    [Fact]
    public void DoubleClampScale_Simd_MatchesScalar()
    {
        int n = 100_003;   // not a multiple of the double lane count → exercises the tail
        Random rng = new Random(71);
        double[] input = MakeDoubleInput(n, _ => (rng.NextDouble() * 2) - 0.5);
        double[] expected = new double[n];
        double[] actual = new double[n];

        for (int i = 0; i < n; i++)
        {
            double x = input[i];
            if (x < 0.0)
                x = 0.0;
            else if (x > 1.0)
                x = 1.0;
            expected[i] = x * 2.5;
        }

        TestKernels.DoubleClampScale_Simd(input, actual, 2.5, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void DoubleClampScale_ParallelSimd_MatchesScalar()
    {
        int n = 1_000_003;
        Random rng = new Random(71);
        double[] input = MakeDoubleInput(n, _ => (rng.NextDouble() * 2) - 0.5);
        double[] expected = new double[n];
        double[] actual = new double[n];

        for (int i = 0; i < n; i++)
        {
            double x = input[i];
            if (x < 0.0)
                x = 0.0;
            else if (x > 1.0)
                x = 1.0;
            expected[i] = x * 2.5;
        }

        TestKernels.DoubleClampScale_ParallelSimd(input, actual, 2.5, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void DoubleTonemap_Simd_MatchesScalar()
    {
        int n = 100_003;
        Random rng = new Random(72);
        double[] input = MakeDoubleInput(n, _ => rng.NextDouble() * 4);
        double[] expected = new double[n];
        double[] actual = new double[n];

        for (int i = 0; i < n; i++)
            expected[i] = 1.0 - Math.Exp(-input[i] * 1.5);

        TestKernels.DoubleTonemap_Simd(input, actual, 1.5, n);

        // VectorMath.Exp(VDouble) is an approximation (~1e-13 relative).
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12 * Math.Max(1.0, Math.Abs(expected[i])),
                $"i={i}: expected {expected[i]:G17}, got {actual[i]:G17}");
        }
    }

    [Fact]
    public void DoubleSumOfSquares_Simd_MatchesScalar()
    {
        int n = 100_003;
        Random rng = new Random(73);
        double[] input = MakeDoubleInput(n, _ => rng.NextDouble() * 10);
        double[] result = new double[1];

        double expected = 0.0;
        for (int i = 0; i < n; i++)
            expected += input[i] * input[i];

        TestKernels.DoubleSumOfSquares_Simd(input, result, n);

        Assert.True(Math.Abs((result[0] / expected) - 1.0) < 1e-10,
            $"scalar={expected:G17}, simd={result[0]:G17}");
    }

    [Fact]
    public void DoubleSumOfSquares_ParallelSimd_MatchesScalar()
    {
        int n = 1_000_003;
        Random rng = new Random(73);
        double[] input = MakeDoubleInput(n, _ => rng.NextDouble() * 10);
        double[] result = new double[1];

        double expected = 0.0;
        for (int i = 0; i < n; i++)
            expected += input[i] * input[i];

        TestKernels.DoubleSumOfSquares_ParallelSimd(input, result, n);

        Assert.True(Math.Abs((result[0] / expected) - 1.0) < 1e-10,
            $"scalar={expected:G17}, simd={result[0]:G17}");
    }

    [Fact]
    public void Scale2D_Simd_MatchesScalar()
    {
        int width = 1021, height = 37;   // width not a multiple of LaneCount → row tails
        float[] input = MakeInput(width * height, i => (float)((new Random(81).NextDouble() * 4) - 2));
        float[] expected = new float[width * height];
        float[] actual = new float[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float v = input[(y * width) + x];
                expected[(y * width) + x] = (v * 1.5f) + y;
            }
        }

        TestKernels.Scale2D_Simd(input, actual, 1.5f, width, height);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void Scale2D_ParallelSimd_MatchesScalar()
    {
        int width = 2053, height = 129;
        float[] input = MakeInput(width * height, i => (float)((new Random(81).NextDouble() * 4) - 2));
        float[] expected = new float[width * height];
        float[] actual = new float[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float v = input[(y * width) + x];
                expected[(y * width) + x] = (v * 1.5f) + y;
            }
        }

        TestKernels.Scale2D_ParallelSimd(input, actual, 1.5f, width, height);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void Sum2D_Simd_MatchesScalar()
    {
        int width = 517, height = 23;
        float[] input = MakeInput(width * height, i => (float)(new Random(82).NextDouble() * 2));
        float[] result = new float[1];

        float expected = 0f;
        for (int i = 0; i < width * height; i++)
            expected += input[i];

        TestKernels.Sum2D_Simd(input, result, width, height);

        Assert.True(Math.Abs((result[0] / expected) - 1f) < 0.01f,
            $"Sum2D: scalar={expected}, simd={result[0]}");
    }

    [Fact]
    public void Sum2D_ParallelSimd_MatchesScalar()
    {
        int width = 2053, height = 129;
        float[] input = MakeInput(width * height, i => (float)(new Random(82).NextDouble() * 2));
        float[] result = new float[1];

        float expected = 0f;
        for (int i = 0; i < width * height; i++)
            expected += input[i];

        TestKernels.Sum2D_ParallelSimd(input, result, width, height);

        Assert.True(Math.Abs((result[0] / expected) - 1f) < 0.01f,
            $"Sum2D parallel: scalar={expected}, simd={result[0]}");
    }

    [Fact]
    public void OffsetRead_Simd_MatchesScalar()
    {
        int n = 100_003, offset = 7;
        float[] input = MakeInput(n + offset, i => i * 0.25f);
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[i + offset] * 2f;

        TestKernels.OffsetRead_Simd(input, actual, offset, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void CountHalvings_Simd_MatchesScalar()
    {
        int n = 100_003;
        float[] input = MakeInput(n, i => (float)(new Random(91).NextDouble() * 100));
        int[] expected = new int[n];
        int[] actual = new int[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            int steps = 0;
            while (x > 1f)
            { x *= 0.5f; steps++; }

            expected[i] = steps;
        }

        TestKernels.CountHalvings_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void MandelbrotIters_Simd_MatchesScalar_Exactly()
    {
        int w = 131, h = 53, maxIter = 64;   // width not a multiple of LaneCount → row tails
        float minX = -2f, minY = -1.25f, dx = 2.5f / w, dy = 2.5f / h;
        int[] expected = new int[w * h];
        int[] actual = new int[w * h];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float cx = minX + (x * dx), cy = minY + (y * dy);
                float zx = 0, zy = 0;
                int i = 0;
                while ((zx * zx) + (zy * zy) < 4f && i < maxIter)
                {
                    float n2 = (zx * zx) - (zy * zy) + cx;
                    zy = (2f * zx * zy) + cy;
                    zx = n2;
                    i++;
                }

                expected[(y * w) + x] = i;
            }
        }

        TestKernels.MandelbrotIters_Simd(actual, minX, minY, dx, dy, maxIter, w, h);

        // Lane arithmetic is the same float sequence as scalar → counts match exactly.
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void MandelbrotIters_ParallelSimd_MatchesSerial()
    {
        int w = 257, h = 129, maxIter = 64;
        float minX = -2f, minY = -1.25f, dx = 2.5f / w, dy = 2.5f / h;
        int[] serial = new int[w * h];
        int[] parallel = new int[w * h];

        TestKernels.MandelbrotIters_Simd(serial, minX, minY, dx, dy, maxIter, w, h);
        TestKernels.MandelbrotIters_ParallelSimd(parallel, minX, minY, dx, dy, maxIter, w, h, minChunkSize: 1024);

        for (int i = 0; i < serial.Length; i++)
            Assert.Equal(serial[i], parallel[i]);
    }

    [Fact]
    public void TiledScale_Simd_MatchesScalar()
    {
        int width = 203, height = 71;   // not multiples of the 64x64 default tile
        float[] input = MakeInput(width * height, i => (float)((new Random(101).NextDouble() * 4) - 2));
        float[] expected = new float[width * height];
        float[] actual = new float[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
                expected[(y * width) + x] = (input[(y * width) + x] * 1.5f) + y;
        }

        TestKernels.TiledScale_Simd(input, actual, 1.5f, width, height);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void TiledScale_ParallelSimd_MatchesScalar()
    {
        int width = 1021, height = 257;
        float[] input = MakeInput(width * height, i => (float)((new Random(101).NextDouble() * 4) - 2));
        float[] expected = new float[width * height];
        float[] actual = new float[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
                expected[(y * width) + x] = (input[(y * width) + x] * 1.5f) + y;
        }

        TestKernels.TiledScale_ParallelSimd(input, actual, 1.5f, width, height);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void TiledSum_Simd_MatchesScalar()
    {
        int width = 203, height = 71;   // explicit 32x16 tiles, ragged edges
        float[] input = MakeInput(width * height, i => (float)(new Random(102).NextDouble() * 2));
        float[] result = new float[1];

        float expected = 0f;
        for (int i = 0; i < width * height; i++)
            expected += input[i];

        TestKernels.TiledSum_Simd(input, result, width, height);

        Assert.True(Math.Abs((result[0] / expected) - 1f) < 0.01f,
            $"TiledSum: scalar={expected}, simd={result[0]}");
    }

    [Fact]
    public void TiledSum_ParallelSimd_MatchesScalar()
    {
        int width = 1021, height = 257;
        float[] input = MakeInput(width * height, i => (float)(new Random(102).NextDouble() * 2));
        float[] result = new float[1];

        float expected = 0f;
        for (int i = 0; i < width * height; i++)
            expected += input[i];

        TestKernels.TiledSum_ParallelSimd(input, result, width, height);

        Assert.True(Math.Abs((result[0] / expected) - 1f) < 0.01f,
            $"TiledSum parallel: scalar={expected}, simd={result[0]}");
    }

    [Fact]
    public void SumReturn_Simd_ReturnsReduction()
    {
        int n = TailCount;
        Random rng = new Random(111);
        float[] input = MakeInput(n, _ => (float)((rng.NextDouble() * 2) - 1));

        // Double-precision reference: for ~1M floats the sequential *float* sum itself
        // drifts more than the SIMD tree sum does, so compare both against the truth.
        double expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];

        float actual = TestKernels.SumReturn_Simd(input, n);

        Assert.True(Math.Abs(actual - expected) < Math.Max(1.0, Math.Abs(expected)) * 0.01,
            $"SumReturn: reference={expected}, simd={actual}");
    }

    [Fact]
    public void SumReturn_ParallelSimd_ReturnsReduction()
    {
        int n = TailCount;
        Random rng = new Random(111);
        float[] input = MakeInput(n, _ => (float)((rng.NextDouble() * 2) - 1));

        double expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];

        float actual = TestKernels.SumReturn_ParallelSimd(input, n);

        Assert.True(Math.Abs(actual - expected) < Math.Max(1.0, Math.Abs(expected)) * 0.01,
            $"SumReturn parallel: reference={expected}, simd={actual}");
    }

    [Fact]
    public void MaxReturn_SimdAndParallel_ReturnExactMax()
    {
        int n = TailCount;
        Random rng = new Random(112);
        int[] input = MakeIntInput(n, _ => rng.Next(-1000000, 1000000));

        int expected = int.MinValue;
        for (int i = 0; i < n; i++)
            expected = Math.Max(expected, input[i]);

        Assert.Equal(expected, TestKernels.MaxReturn_Simd(input, n));
        Assert.Equal(expected, TestKernels.MaxReturn_ParallelSimd(input, n));
    }

    [Fact]
    public void IntDivMod_Simd_MatchesScalar()
    {
        int n = 100_003;
        Random rng = new Random(121);
        int[] a = MakeIntInput(n, _ => rng.Next(-100000, 100000));
        int[] b = MakeIntInput(n, _ => rng.Next(1, 100) * (rng.Next(2) == 0 ? 1 : -1));  // nonzero
        int[] q = new int[n];
        int[] r = new int[n];

        TestKernels.IntDivMod_Simd(a, b, q, r, n);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(a[i] / b[i], q[i]);
            Assert.Equal(a[i] % b[i], r[i]);
        }
    }

    [Fact]
    public void MaskedDiv_Simd_ZeroDivisorsOutsideMask_DoNotThrow()
    {
        int n = 100_003;
        Random rng = new Random(122);
        int[] a = MakeIntInput(n, _ => rng.Next(-100000, 100000));
        // Roughly half the divisors are zero, the kernel only divides where b != 0.
        int[] b = MakeIntInput(n, _ => rng.Next(2) == 0 ? 0 : rng.Next(1, 50));
        int[] expected = new int[n];
        int[] actual = new int[n];

        for (int i = 0; i < n; i++)
        {
            if (b[i] != 0)
                expected[i] = a[i] / b[i];
        }

        TestKernels.MaskedDiv_Simd(a, b, actual, n);   // must not throw

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void BitOps_Simd_MatchesScalar()
    {
        int n = 100_003;
        Random rng = new Random(131);
        int[] input = MakeIntInput(n, _ => rng.Next(int.MinValue, int.MaxValue));
        int[] expected = new int[n];
        int[] actual = new int[n];
        int shift = 3;

        for (int i = 0; i < n; i++)
        {
            int x = input[i];
            expected[i] = ((x << shift) >> 2) ^ (~x & 0xFF);
        }

        TestKernels.BitOps_Simd(input, actual, shift, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void DoubleMath_Simd_MatchesScalar()
    {
        int n = 100_003;
        Random rng = new Random(141);
        double[] input = MakeDoubleInput(n, _ => (rng.NextDouble() * 4) - 1);
        double[] expected = new double[n];
        double[] actual = new double[n];

        for (int i = 0; i < n; i++)
        {
            double x = input[i];
            expected[i] = Math.Sin(x) + (Math.Sqrt(x + 2.0) * Math.Log(x + 3.0)) - Math.Atan(x);
        }

        TestKernels.DoubleMath_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12 * Math.Max(1.0, Math.Abs(expected[i])),
                $"i={i}: expected {expected[i]:G17}, got {actual[i]:G17}");
        }
    }

    [Fact]
    public void DoubleGather_Simd_MatchesScalar()
    {
        int n = 100_003;
        Random rng = new Random(151);
        double[] table = MakeDoubleInput(256, i => i * 1.5);
        double[] indices = MakeDoubleInput(n, _ => rng.Next(0, 256));
        double[] expected = new double[n];
        double[] actual = new double[n];

        for (int i = 0; i < n; i++)
            expected[i] = table[(int)indices[i]] * 2.0;

        TestKernels.DoubleGather_Simd(table, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void DoubleScatter_Simd_MatchesScalar()
    {
        int n = 10_007;
        Random rng = new Random(152);
        double[] values = MakeDoubleInput(n, i => i * 0.25);
        // Unique index per element (a permutation) so scalar/SIMD write order can't differ.
        int[] perm = [.. Enumerable.Range(0, n).OrderBy(_ => rng.Next())];
        double[] indices = MakeDoubleInput(n, i => perm[i]);
        double[] expected = new double[n];
        double[] actual = new double[n];

        for (int i = 0; i < n; i++)
            expected[(int)indices[i]] = values[i] * 3.0;

        TestKernels.DoubleScatter_Simd(values, indices, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void VarShift_Simd_MatchesScalar()
    {
        int n = 100_003;
        Random rng = new Random(161);
        int[] input = MakeIntInput(n, _ => rng.Next(int.MinValue, int.MaxValue));
        // Counts include values >= 32 to exercise the C# low-5-bits semantics.
        int[] counts = MakeIntInput(n, _ => rng.Next(0, 40));
        int[] expected = new int[n];
        int[] actual = new int[n];

        for (int i = 0; i < n; i++)
        {
            int x = input[i];
            int c = counts[i];
            expected[i] = (x << c) ^ (x >> c) ^ (x >>> c);
        }

        TestKernels.VarShift_Simd(input, counts, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void DoubleIntermediate_Simd_MatchesScalar_Exactly()
    {
        int n = 100_003;
        Random rng = new Random(162);
        float[] input = MakeInput(n, _ => (float)(rng.NextDouble() * 100));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            double d = ((double)x * 1.0000000001d) + 0.5d;
            expected[i] = (float)Math.Sqrt(d);
        }

        TestKernels.DoubleIntermediate_Simd(input, actual, n);

        // Widen, hardware double sqrt, and narrow are all exact, bit-identical results.
        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void PreciseSum_Simd_AccumulatesInDouble()
    {
        int n = TailCount;
        Random rng = new Random(163);
        float[] input = MakeInput(n, _ => (float)((rng.NextDouble() * 2) - 1));

        double expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];

        double actual = TestKernels.PreciseSum_Simd(input, n);

        // Double accumulation: only reassociation-order differences remain (~1e-12),
        // far tighter than any float accumulator could achieve.
        Assert.True(Math.Abs(actual - expected) < Math.Max(1.0, Math.Abs(expected)) * 1e-9,
            $"PreciseSum: reference={expected:G17}, simd={actual:G17}");
    }

    [Fact]
    public void PreciseSum_ParallelSimd_AccumulatesInDouble()
    {
        int n = TailCount;
        Random rng = new Random(163);
        float[] input = MakeInput(n, _ => (float)((rng.NextDouble() * 2) - 1));

        double expected = 0;
        for (int i = 0; i < n; i++)
            expected += input[i];

        double actual = TestKernels.PreciseSum_ParallelSimd(input, n);

        Assert.True(Math.Abs(actual - expected) < Math.Max(1.0, Math.Abs(expected)) * 1e-9,
            $"PreciseSum parallel: reference={expected:G17}, simd={actual:G17}");
    }

    [Fact]
    public void AffineIndexed_Simd_MatchesScalar()
    {
        int n = 100_003, lo = 5;
        float[] input = MakeInput(n + lo + 3, i => i * 0.5f);
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[i + lo] + input[i + lo + 3];

        TestKernels.AffineIndexed_Simd(input, actual, lo, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void AffineIndexed_ParallelSimd_MatchesScalar()
    {
        int n = 1_000_003, lo = 5;
        float[] input = MakeInput(n + lo + 3, i => i * 0.5f);
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[i + lo] + input[i + lo + 3];

        TestKernels.AffineIndexed_ParallelSimd(input, actual, lo, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void StridedIndexed_Simd_StillCorrectViaGather()
    {
        int n = 100_003;
        float[] input = MakeInput(n * 2, i => i * 0.25f);
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
            expected[i] = input[i * 2];

        TestKernels.StridedIndexed_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void ReturnInLoop_Simd_RetiresLanesPerElement()
    {
        // Per-lane semantics: each element is simulated independently; a 'return' skips
        // that element's trailing write but other elements keep processing. (This is ISPC's
        // return, NOT scalar C#'s method-exit, see the README's limitations section.)
        int n = 100_003;   // not a multiple of LaneCount → exercises the goto-based tail too
        Random rng = new Random(171);
        float[] input = MakeInput(n, _ => (float)((rng.NextDouble() * 0.6) - 0.1));
        float[] expected = new float[n];
        float[] actual = new float[n];

        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            float acc = 0f;
            bool returned = false;
            for (float j = 0f; j < 50f; j += 1f)
            {
                acc += x;
                if (acc > 10f)
                { returned = true; break; }
            }

            if (!returned)
                expected[i] = acc;
        }

        TestKernels.ReturnInLoop_Simd(input, actual, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    [Fact]
    public void ReturnInLoop_ParallelSimd_MatchesSerial()
    {
        int n = 1_000_003;
        Random rng = new Random(171);
        float[] input = MakeInput(n, _ => (float)((rng.NextDouble() * 0.6) - 0.1));
        float[] serial = new float[n];
        float[] parallel = new float[n];

        TestKernels.ReturnInLoop_Simd(input, serial, n);
        TestKernels.ReturnInLoop_ParallelSimd(input, parallel, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(serial[i], parallel[i]);
    }
}
