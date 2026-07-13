using System;
using IspcSharp;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Kernels where the Spmd.Range loop is nested inside uniform control flow (the "scaffold"
/// shape) and kernels over 2-D arrays (float[,]/int[,]/double[,]).
/// </summary>
public static partial class ScaffoldKernels
{
    /// <summary>
    /// Matrix multiply: c = a·b. Nested uniform row/col loops around a reduction foreach.
    /// a[row,k] is contiguous along k; b[k,col] is a strided gather; c[row,col] is a uniform store.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="c"></param>
    /// <param name="n"></param>
    /// <param name="m"></param>
    /// <param name="p"></param>
    [Spmd]
    public static void MatMulF(float[,] a, float[,] b, float[,] c, int n, int m, int p)
    {
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < p; col++)
            {
                float sum = 0f;
                foreach (var k in Spmd.Range(m))
                {
                    sum += a[row, k] * b[k, col];
                }
                c[row, col] = sum;
            }
        }
    }

    [Spmd]
    public static void MatMulI(int[,] a, int[,] b, int[,] c, int n, int m, int p)
    {
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < p; col++)
            {
                int sum = 0;
                foreach (var k in Spmd.Range(m))
                {
                    sum += a[row, k] * b[k, col];
                }
                c[row, col] = sum;
            }
        }
    }

    [Spmd]
    public static void MatMulD(double[,] a, double[,] b, double[,] c, int n, int m, int p)
    {
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < p; col++)
            {
                double sum = 0d;
                foreach (var k in Spmd.Range(m))
                {
                    sum += a[row, k] * b[k, col];
                }
                c[row, col] = sum;
            }
        }
    }

    /// <summary>
    /// Matrix multiply against a pre-transposed b, both a[row,k] and bT[col,k] are contiguous.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="bT"></param>
    /// <param name="c"></param>
    /// <param name="n"></param>
    /// <param name="m"></param>
    /// <param name="p"></param>
    [Spmd]
    public static void MatMulT(float[,] a, float[,] bT, float[,] c, int n, int m, int p)
    {
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < p; col++)
            {
                float sum = 0f;
                foreach (var k in Spmd.Range(m))
                {
                    sum += a[row, k] * bT[col, k];
                }
                c[row, col] = sum;
            }
        }
    }

    [Spmd]
    public static void AddRowBias(float[,] input, float[] bias, float[,] output, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            foreach (var c in Spmd.Range(cols))
            {
                output[r, c] = input[r, c] + bias[r];
            }
        }
    }

    [Spmd]
    public static void ScaleBlocks(float[] data, float[] scales, int blocks, int blockSize)
    {
        for (int bi = 0; bi < blocks; bi++)
        {
            float s = scales[bi];
            int baseIdx = bi * blockSize;
            foreach (var t in Spmd.Range(blockSize))
            {
                data[baseIdx + t] = data[baseIdx + t] * s;
            }
        }
    }

    [Spmd]
    public static void Fft(float[] real, float[] imag, int n)
    {
        // Bit reversal (all uniform, passes through as scalar C#).
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0)
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if (i < j)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        // FFT stages.
        for (int len = 2; len <= n; len <<= 1)
        {
            float angle = -2.0f * MathF.PI / len;
            int half = len / 2;
            for (int i = 0; i < n; i += len)
            {
                foreach (var k in Spmd.Range(half))
                {
                    float wReal = MathF.Cos(k * angle);
                    float wImag = MathF.Sin(k * angle);

                    int even = i + k;
                    int odd = i + k + half;

                    float oddReal = real[odd] * wReal - imag[odd] * wImag;
                    float oddImag = real[odd] * wImag + imag[odd] * wReal;

                    float evenReal = real[even];
                    float evenImag = imag[even];

                    real[even] = evenReal + oddReal;
                    imag[even] = evenImag + oddImag;

                    real[odd] = evenReal - oddReal;
                    imag[odd] = evenImag - oddImag;
                }
            }
        }
    }
}

public class ScaffoldAndMatrixTests
{
    private static float[,] RandMatrixF(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r, c] = (float)(rng.NextDouble() * 2 - 1);
        return m;
    }

    private static int[,] RandMatrixI(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new int[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r, c] = rng.Next(-50, 50);
        return m;
    }

    private static double[,] RandMatrixD(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new double[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r, c] = rng.NextDouble() * 2 - 1;
        return m;
    }

    [Fact]
    public void MatMulF_Simd_MatchesScalar()
    {
        int n = 37, m = 71, p = 29;   // m (the vectorized dim) is not a multiple of LaneCount
        var a = RandMatrixF(n, m, 1);
        var b = RandMatrixF(m, p, 2);
        var expected = new float[n, p];
        var actual = new float[n, p];

        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
            {
                float sum = 0f;
                for (int k = 0; k < m; k++) sum += a[row, k] * b[k, col];
                expected[row, col] = sum;
            }

        ScaffoldKernels.MatMulF_Simd(a, b, actual, n, m, p);

        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
                Assert.True(Math.Abs(expected[row, col] - actual[row, col]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[row, col])),
                    $"[{row},{col}]: expected {expected[row, col]}, got {actual[row, col]}");
    }

    [Fact]
    public void MatMulI_Simd_MatchesScalar_Exactly()
    {
        int n = 31, m = 53, p = 17;
        var a = RandMatrixI(n, m, 3);
        var b = RandMatrixI(m, p, 4);
        var expected = new int[n, p];
        var actual = new int[n, p];

        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
            {
                int sum = 0;
                for (int k = 0; k < m; k++) sum += a[row, k] * b[k, col];
                expected[row, col] = sum;
            }

        ScaffoldKernels.MatMulI_Simd(a, b, actual, n, m, p);

        // Integer sums are exact regardless of reassociation.
        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
                Assert.Equal(expected[row, col], actual[row, col]);
    }

    [Fact]
    public void MatMulT_Simd_MatchesScalar_AndPlainMatMul()
    {
        int n = 37, m = 71, p = 29;
        var a = RandMatrixF(n, m, 1);
        var b = RandMatrixF(m, p, 2);

        // Reference triple loop.
        var expected = new float[n, p];
        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
            {
                float sum = 0f;
                for (int k = 0; k < m; k++) sum += a[row, k] * b[k, col];
                expected[row, col] = sum;
            }

        // Transpose b, then run the transposed matmul.
        var bT = Memory.Transposed(b);   // [p, m]
        var actual = new float[n, p];
        ScaffoldKernels.MatMulT_Simd(a, bT, actual, n, m, p);

        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
                Assert.True(Math.Abs(expected[row, col] - actual[row, col]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[row, col])),
                    $"[{row},{col}]: expected {expected[row, col]}, got {actual[row, col]}");
    }

    [Fact]
    public void MatMulD_Simd_MatchesScalar()
    {
        int n = 23, m = 61, p = 19;
        var a = RandMatrixD(n, m, 5);
        var b = RandMatrixD(m, p, 6);
        var expected = new double[n, p];
        var actual = new double[n, p];

        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
            {
                double sum = 0d;
                for (int k = 0; k < m; k++) sum += a[row, k] * b[k, col];
                expected[row, col] = sum;
            }

        ScaffoldKernels.MatMulD_Simd(a, b, actual, n, m, p);

        for (int row = 0; row < n; row++)
            for (int col = 0; col < p; col++)
                Assert.True(Math.Abs(expected[row, col] - actual[row, col]) <= 1e-10 * Math.Max(1.0, Math.Abs(expected[row, col])),
                    $"[{row},{col}]: expected {expected[row, col]:G17}, got {actual[row, col]:G17}");
    }

    [Fact]
    public void AddRowBias_Simd_MatchesScalar()
    {
        int rows = 40, cols = 101;
        var input = RandMatrixF(rows, cols, 7);
        var bias = new float[rows];
        var rng = new Random(8);
        for (int r = 0; r < rows; r++) bias[r] = (float)(rng.NextDouble() * 10);
        var expected = new float[rows, cols];
        var actual = new float[rows, cols];

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                expected[r, c] = input[r, c] + bias[r];

        ScaffoldKernels.AddRowBias_Simd(input, bias, actual, rows, cols);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.Equal(expected[r, c], actual[r, c]);
    }

    [Fact]
    public void ScaleBlocks_Simd_MatchesScalar()
    {
        int blocks = 50, blockSize = 103;
        int n = blocks * blockSize;
        var rng = new Random(9);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 4 - 2);
        var scales = new float[blocks];
        for (int b = 0; b < blocks; b++) scales[b] = (float)(rng.NextDouble() * 3);

        var expected = (float[])data.Clone();
        for (int b = 0; b < blocks; b++)
            for (int t = 0; t < blockSize; t++)
                expected[b * blockSize + t] *= scales[b];

        ScaffoldKernels.ScaleBlocks_Simd(data, scales, blocks, blockSize);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], data[i], 4);
    }

    /// <summary>
    /// Scalar reference using the exact same per-lane-twiddle algorithm the kernel vectorizes,
    /// so results match to the Cos/Sin approximation tolerance.
    /// </summary>
    /// <param name="real"></param>
    /// <param name="imag"></param>
    /// <param name="n"></param>
    private static void FftReference(float[] real, float[] imag, int n)
    {
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
            j ^= bit;
            if (i < j)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            float angle = -2.0f * MathF.PI / len;
            int half = len / 2;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < half; k++)
                {
                    float wReal = MathF.Cos(k * angle);
                    float wImag = MathF.Sin(k * angle);
                    int even = i + k, odd = i + k + half;
                    float oddReal = real[odd] * wReal - imag[odd] * wImag;
                    float oddImag = real[odd] * wImag + imag[odd] * wReal;
                    float evenReal = real[even], evenImag = imag[even];
                    real[even] = evenReal + oddReal;
                    imag[even] = evenImag + oddImag;
                    real[odd] = evenReal - oddReal;
                    imag[odd] = evenImag - oddImag;
                }
        }
    }

    [Fact]
    public void Fft_Simd_MatchesScalarReference()
    {
        int n = 1024;   // power of two
        var rng = new Random(11);
        var real0 = new float[n];
        var imag0 = new float[n];
        for (int i = 0; i < n; i++) { real0[i] = (float)(rng.NextDouble() * 2 - 1); imag0[i] = 0f; }

        var realRef = (float[])real0.Clone();
        var imagRef = (float[])imag0.Clone();
        FftReference(realRef, imagRef, n);

        var realSimd = (float[])real0.Clone();
        var imagSimd = (float[])imag0.Clone();
        ScaffoldKernels.Fft_Simd(realSimd, imagSimd, n);

        // VectorMath.Cos/Sin are ~1e-6 approximations; error compounds across log2(n)=10 stages.
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(realRef[i] - realSimd[i]) < 1e-2f,
                $"real[{i}]: ref {realRef[i]}, simd {realSimd[i]}");
            Assert.True(Math.Abs(imagRef[i] - imagSimd[i]) < 1e-2f,
                $"imag[{i}]: ref {imagRef[i]}, simd {imagSimd[i]}");
        }
    }
}
