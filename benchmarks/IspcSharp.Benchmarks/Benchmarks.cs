using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace IspcSharp.Benchmarks;

public static class Program
{
    public static void Main(string[] args)
    {
        PrintEnvironmentBanner();
        _ = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
    }

    /// <summary>
    /// The SIMD width and ISA the JIT actually chose, printed so a benchmark result is unambiguous
    /// about the lane count it ran at. A `Vector&lt;float&gt;.Count` of 16 means 512-bit vectors: on
    /// many CPUs sustained wide-vector FP triggers frequency down-clocking, which can make a
    /// compute-bound SIMD kernel LOSE to a full-clock scalar loop (e.g. a transcendental vs. glibc).
    /// If you see that, re-run with the environment variable <c>DOTNET_PreferredVectorBitWidth=256</c>
    /// to force 256-bit vectors and compare.
    /// </summary>
    private static void PrintEnvironmentBanner()
    {
        Console.WriteLine("==== IspcSharp benchmark environment ====");
        Console.WriteLine($"OS/Arch          : {System.Runtime.InteropServices.RuntimeInformation.OSDescription.Trim()} / {System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"Vector<float>    : Count={System.Numerics.Vector<float>.Count} ({System.Numerics.Vector<float>.Count * 32}-bit), HwAccelerated={System.Numerics.Vector.IsHardwareAccelerated}");
        Console.WriteLine($"Vector<double>   : Count={System.Numerics.Vector<double>.Count}");
        Console.WriteLine($"Vector<long>     : Count={System.Numerics.Vector<long>.Count}");
#if NET8_0_OR_GREATER
        bool x = System.Runtime.Intrinsics.X86.Avx2.IsSupported;
        Console.WriteLine($"x86 ISA          : SSE2={System.Runtime.Intrinsics.X86.Sse2.IsSupported} AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} " +
                          $"AVX-512F={System.Runtime.Intrinsics.X86.Avx512F.IsSupported} AVX-512DQ={System.Runtime.Intrinsics.X86.Avx512DQ.IsSupported} FMA={System.Runtime.Intrinsics.X86.Fma.IsSupported}");
        Console.WriteLine($"ARM ISA          : AdvSimd(NEON)={System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported}");
#endif
        string? pref = Environment.GetEnvironmentVariable("DOTNET_PreferredVectorBitWidth");
        Console.WriteLine($"PreferredVectorBitWidth env : {(string.IsNullOrEmpty(pref) ? "(unset, JIT default)" : pref)}");
        if (System.Numerics.Vector<float>.Count == 16)
            Console.WriteLine("NOTE: 512-bit vectors in use. If compute-bound SIMD benches lose to scalar, retry with DOTNET_PreferredVectorBitWidth=256 (down-clock check).");
        Console.WriteLine("=========================================");
        Console.WriteLine();
    }
}

/// <summary>
/// [Spmd]-generated kernels. Every benchmark below runs these generated
/// _Simd / _ParallelSimd companions rather than hand-written runtime-API
/// lambdas: the generator emits full-speed contiguous loads/stores in the
/// main loop with a scalar tail, keeps all state in registers (no closures),
/// and has no per-gang delegate calls, the fast patterns, by construction.
/// </summary>
public static partial class GeneratedKernels
{
    /// <summary>
    /// Branchless saturate + scale, best case for SIMD.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="scale"></param>
    /// <param name="count"></param>
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

    /// <summary>
    /// Reduction: sum of squares.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="result"></param>
    /// <param name="count"></param>
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

    /// <summary>
    /// Reduction over two arrays: dot product.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="result"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void DotProduct(float[] a, float[] b, float[] result, int count)
    {
        float sum = 0f;
        foreach (int i in Spmd.Range(count))
        {
            sum += a[i] * b[i];
        }

        result[0] = sum;
    }

    /// <summary>
    /// Cross-gang-width 64-bit accumulator (VLong2): exact long sum over int[] data. No long
    /// buffer, so this stays a 32-bit int gang; the accumulator widens into VLong2 pairs.
    /// A plain int accumulator would overflow, this is the integer mirror of a double PreciseSum.
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
    /// Element-wise two-input add, "matrix/vector add" (c = a + b). Memory-bound: two reads,
    /// one write per element, so it stiffens the memory wall harder than the 1R1W copy below.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="c"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void VectorAdd(float[] a, float[] b, float[] c, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            c[i] = a[i] + b[i];
        }
    }

    /// <summary>
    /// BLAS level-1 AXPY: y = alpha*x + y (in-place). One FMA per element over two live streams —
    /// the canonical "is it memory- or compute-bound?" microbenchmark.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="alpha"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Saxpy(float[] x, float[] y, float alpha, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            y[i] = (alpha * x[i]) + y[i];
        }
    }

    /// <summary>
    /// Integer-lane compute: an int32 bit-mix hash (no floats). Exercises VInt multiply, xor, and
    /// uniform shifts, the 32-bit integer throughput story the float kernels don't cover.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void IntMix(int[] input, int[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            int x = input[i];
            x ^= x >> 16;
            x *= 0x45d9f3b;
            x ^= x >> 16;
            x *= 0x45d9f3b;
            x ^= x >> 16;
            output[i] = x;
        }
    }

    /// <summary>
    /// Native 64-bit long gang (long[] buffers ⇒ VLong/VMaskD): a splitmix-style bit hash. Half the
    /// lane count of the 32-bit kernels; shows where the 64-bit-integer subset lands on real work.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void LongHash(long[] input, long[] output, int count)
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
    /// Double-precision gang (double[] buffers ⇒ VDouble/VMaskD): the double tonemap. Half the lane
    /// count of the float version, plus the higher-order double Exp polynomial, the 64-bit float cost.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="exposure"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void DoubleTonemap(double[] input, double[] output, double exposure, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = 1.0 - Math.Exp(-input[i] * exposure);
        }
    }

    /// <summary>
    /// Cross-gang-width VDouble2: a double accumulator over float[] data (an exact, drift-free sum).
    /// The floating mirror of WideSum, stays a 32-bit gang while accumulating in double precision.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="count"></param>
    /// <returns></returns>
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

    /// <summary>
    /// Reduce-max (not add), the min/max reduction path. Horizontal max after a vector-max fold.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    [Spmd]
    public static float ArrayMax(float[] input, int count)
    {
        float mx = float.NegativeInfinity;
        foreach (int i in Spmd.Range(count))
        {
            mx = MathF.Max(mx, input[i]);
        }

        return mx;
    }

    /// <summary>
    /// Non-contiguous read: out[i] = table[indices[i]], a hardware/emulated gather. Deliberately
    /// the slow-memory case the ISPC100 analyzer warns about; benched so the cost is visible.
    /// </summary>
    /// <param name="table"></param>
    /// <param name="indices"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void GatherRead(float[] table, int[] indices, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = table[indices[i]];
        }
    }

    /// <summary>
    /// Trig-heavy (Sin·Cos), a different transcendental mix from the Exp tonemap, both range-reduced
    /// polynomials. Common in signal generation / rotation kernels.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void SinCos(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            output[i] = MathF.Sin(x) * MathF.Cos(x);
        }
    }

    /// <summary>
    /// Transcendental-heavy: tonemap 1 - e^-x.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="exposure"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void Tonemap(float[] input, float[] output, float exposure, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = 1f - MathF.Exp(-input[i] * exposure);
        }
    }

    /// <summary>
    /// Memory-bandwidth-bound control: copy with add.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void AddOne(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            output[i] = input[i] + 1f;
        }
    }

    /// <summary>
    /// Divergent while with per-lane trip counts.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void NewtonSqrt(float[] input, float[] output, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = input[i];
            float guess = x;
            float err = 1f;
            while (err > 0.0001f)         // per-lane trip counts, masked automatically
            {
                float next = 0.5f * (guess + (x / guess));
                err = MathF.Abs(next - guess);
                guess = next;
            }

            output[i] = guess;
        }
    }

    /// <summary>
    /// Worst realistic case: Mandelbrot. 2D domain (x-gangs along rows, y uniform),
    /// divergent while with per-pixel trip counts. iters[y*width + x] is recognized
    /// as an affine contiguous store, no scatter.
    /// </summary>
    /// <param name="iters"></param>
    /// <param name="minX"></param>
    /// <param name="minY"></param>
    /// <param name="dx"></param>
    /// <param name="dy"></param>
    /// <param name="maxIter"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    [Spmd]
    public static void Mandelbrot(int[] iters, float minX, float minY, float dx, float dy, int maxIter, int width, int height)
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

    /// <summary>
    /// Iterative radix-2 FFT. The uniform scaffolding, bit-reversal (with `j ^= bit`, tuple
    /// swaps), `MathF.PI`, the `for (i += len)` stage loop, passes through verbatim; only the
    /// inner butterfly loop is a `Spmd.Range` that gets vectorized. The classic twiddle
    /// *recurrence* (`w *= wStep`) is a serial dependency no SPMD compiler can vectorize, so the
    /// twiddles are recomputed per lane as cos/sin(k·angle), the standard vectorizable rewrite.
    /// </summary>
    /// <param name="real"></param>
    /// <param name="imag"></param>
    [Spmd]
    public static void Transform(float[] real, float[] imag)
    {
        int n = real.Length;

        // Bit reversal, all uniform, emitted as scalar C#.
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
                foreach (int k in Spmd.Range(half))     // vectorized butterfly
                {
                    float wReal = MathF.Cos(k * angle);
                    float wImag = MathF.Sin(k * angle);

                    int even = i + k;
                    int odd = i + k + half;

                    float oddReal = (real[odd] * wReal) - (imag[odd] * wImag);
                    float oddImag = (real[odd] * wImag) + (imag[odd] * wReal);

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

    /// <summary>
    /// One FFT group (a slice of a stage). The scaffold FFT and 2-D matmul are generated as
    /// _Simd only (nested/uniform control flow and 2-D arrays skip _ParallelSimd, ISPC005),
    /// so the "parallel" benchmark parallelizes the OUTER loop over these simple-shape helper
    /// kernels, exactly as the ISPC005 diagnostic recommends. Groups within a stage touch
    /// disjoint [start, start+2*half) ranges, so they run on separate threads safely.
    /// </summary>
    /// <param name="real"></param>
    /// <param name="imag"></param>
    /// <param name="start"></param>
    /// <param name="half"></param>
    /// <param name="angle"></param>
    [Spmd]
    public static void FftGroup(float[] real, float[] imag, int start, int half, float angle)
    {
        foreach (int k in Spmd.Range(half))
        {
            float wReal = MathF.Cos(k * angle);
            float wImag = MathF.Sin(k * angle);

            int even = start + k;
            int odd = start + k + half;

            float oddReal = (real[odd] * wReal) - (imag[odd] * wImag);
            float oddImag = (real[odd] * wImag) + (imag[odd] * wReal);

            float evenReal = real[even];
            float evenImag = imag[even];

            real[even] = evenReal + oddReal;
            imag[even] = evenImag + oddImag;

            real[odd] = evenReal - oddReal;
            imag[odd] = evenImag - oddImag;
        }
    }

    /// <summary>
    /// Matmul against a PRE-TRANSPOSED b (bT[col,k] == b[k,col]). Now BOTH a[row,k] and bT[col,k]
    /// are contiguous loads, no gather. This is the "transpose b" fix for the column-access gather.
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
                foreach (int k in Spmd.Range(m))
                {
                    sum += a[row, k] * bT[col, k];
                }

                c[row, col] = sum;
            }
        }
    }

    /// <summary>
    /// One row of the transposed matmul (for the parallel variant).
    /// </summary>
    /// <param name="a"></param>
    /// <param name="bT"></param>
    /// <param name="c"></param>
    /// <param name="row"></param>
    /// <param name="m"></param>
    /// <param name="p"></param>
    [Spmd]
    public static void MatMulTRow(float[,] a, float[,] bT, float[,] c, int row, int m, int p)
    {
        for (int col = 0; col < p; col++)
        {
            float sum = 0f;
            foreach (int k in Spmd.Range(m))
            {
                sum += a[row, k] * bT[col, k];
            }

            c[row, col] = sum;
        }
    }
}

/// <summary>
/// Branchless compute-bound kernel: saturate + scale. The best case for SIMD —
/// expect speedup ≈ lane count on a single thread.
/// </summary>
[MemoryDiagnoser]
public class ClampScaleBench
{
    [Params(1 << 16, 1 << 22)]
    public int N;

    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new float[N];
        _output = new float[N];
        for (int i = 0; i < N; i++)
            _input[i] = (float)((rng.NextDouble() * 2) - 0.5);
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
        {
            float x = _input[i];
            if (x < 0f)
                x = 0f;
            else if (x > 1f)
                x = 1f;
            _output[i] = x * 2.5f;
        }
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.ClampScale_Simd(_input, _output, 2.5f, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.ClampScale_ParallelSimd(_input, _output, 2.5f, N);
}

/// <summary>
/// Reduction: sum of squares, including the thread-local-partials _ParallelSimd.
/// </summary>
[MemoryDiagnoser]
public class GeneratedReductionBench
{
    [Params(1 << 20)]
    public int N;

    private float[] _input = null!;
    private float[] _result = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new float[N];
        _result = new float[1];
        for (int i = 0; i < N; i++)
            _input[i] = (float)rng.NextDouble();
    }

    [Benchmark(Baseline = true)]
    public float Scalar()
    {
        float sum = 0f;
        for (int i = 0; i < N; i++)
            sum += _input[i] * _input[i];
        return sum;
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.SumOfSquares_Simd(_input, _result, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.SumOfSquares_ParallelSimd(_input, _result, N);
}

/// <summary>
/// Reduction over two streams: dot product (generated reduction kernel).
/// </summary>
[MemoryDiagnoser]
public class DotProductBench
{
    [Params(1 << 20)]
    public int N;

    private float[] _a = null!;
    private float[] _b = null!;
    private float[] _result = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _a = new float[N];
        _b = new float[N];
        _result = new float[1];
        for (int i = 0; i < N; i++)
        { _a[i] = (float)rng.NextDouble(); _b[i] = (float)rng.NextDouble(); }
    }

    [Benchmark(Baseline = true)]
    public float Scalar()
    {
        float sum = 0f;
        for (int i = 0; i < N; i++)
            sum += _a[i] * _b[i];
        return sum;
    }

    [Benchmark]
    public float GeneratedSimd()
    {
        GeneratedKernels.DotProduct_Simd(_a, _b, _result, N);
        return _result[0];
    }

    [Benchmark]
    public float GeneratedParallelSimd()
    {
        GeneratedKernels.DotProduct_ParallelSimd(_a, _b, _result, N);
        return _result[0];
    }
}

/// <summary>
/// Cross-gang-width 64-bit reduction: exact long sum over int[] data (the VLong2 case). The
/// accumulator is a full-int-gang-width long pair, so the SIMD variant still processes a whole
/// int gang per step, expect a solid speedup over the scalar long sum, plus the thread-local
/// partials _ParallelSimd. Values are large enough that an int accumulator would overflow.
/// </summary>
[MemoryDiagnoser]
public class WideSumBench
{
    [Params(1 << 20)]
    public int N;

    private int[] _input = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new int[N];
        for (int i = 0; i < N; i++)
            _input[i] = rng.Next(500_000, 1_000_000);
    }

    [Benchmark(Baseline = true)]
    public long Scalar()
    {
        long sum = 0;
        for (int i = 0; i < N; i++)
            sum += _input[i];
        return sum;
    }

    [Benchmark]
    public long GeneratedSimd() => GeneratedKernels.WideSum_Simd(_input, N);

    [Benchmark]
    public long GeneratedParallelSimd() => GeneratedKernels.WideSum_ParallelSimd(_input, N);
}

/// <summary>
/// Transcendental-heavy kernel (tonemap: 1 - e^-x). Tests VectorMath.Exp vs MathF.Exp —
/// typically the biggest wins because scalar transcendentals are expensive.
/// </summary>
[MemoryDiagnoser]
public class TonemapBench
{
    [Params(1 << 20)]
    public int N;

    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new float[N];
        _output = new float[N];
        for (int i = 0; i < N; i++)
            _input[i] = (float)rng.NextDouble() * 4f;
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
            _output[i] = 1f - MathF.Exp(-_input[i] * 1.5f);
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.Tonemap_Simd(_input, _output, 1.5f, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.Tonemap_ParallelSimd(_input, _output, 1.5f, N);
}

/// <summary>
/// Generated divergent-loop kernel: Newton-Raphson sqrt with per-lane trip counts.
/// </summary>
[MemoryDiagnoser]
public class GeneratedNewtonSqrtBench
{
    [Params(1 << 20)]
    public int N;

    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new float[N];
        _output = new float[N];
        for (int i = 0; i < N; i++)
            _input[i] = (float)((rng.NextDouble() * 4) + 0.01);
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
        {
            float x = _input[i], guess = x, err = 1f;
            while (err > 0.0001f)
            {
                float next = 0.5f * (guess + (x / guess));
                err = MathF.Abs(next - guess);
                guess = next;
            }

            _output[i] = guess;
        }
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.NewtonSqrt_Simd(_input, _output, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.NewtonSqrt_ParallelSimd(_input, _output, N);
}

/// <summary>
/// Divergent-loop kernel: Mandelbrot via the generated 2D kernel. Worst realistic
/// case for SPMD, per-lane trip counts mean lanes idle behind the slowest lane in
/// each gang. Expect gains well below lane count but still a solid multiple, plus
/// near-linear core scaling. Zero allocations: no closures, no delegates.
/// </summary>
[MemoryDiagnoser]
public class MandelbrotBench
{
    private const int W = 512, H = 512, MaxIter = 256;
    private int[] _iters = null!;

    [GlobalSetup]
    public void Setup() => _iters = new int[W * H];

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        float dx = 2.5f / W, dy = 2.5f / H;
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                float cx = -2f + (x * dx), cy = -1.25f + (y * dy);
                float zx = 0, zy = 0;
                int i = 0;
                while ((zx * zx) + (zy * zy) < 4f && i < MaxIter)
                {
                    float n = (zx * zx) - (zy * zy) + cx;
                    zy = (2f * zx * zy) + cy;
                    zx = n;
                    i++;
                }

                _iters[(y * W) + x] = i;
            }
        }
    }

    [Benchmark]
    public void GeneratedSimd()
        => GeneratedKernels.Mandelbrot_Simd(_iters, -2f, -1.25f, 2.5f / W, 2.5f / H, MaxIter, W, H);

    [Benchmark]
    public void GeneratedParallelSimd()
        => GeneratedKernels.Mandelbrot_ParallelSimd(_iters, -2f, -1.25f, 2.5f / W, 2.5f / H, MaxIter, W, H);
}

/// <summary>
/// Memory-bandwidth-bound control: plain copy-with-add. Included deliberately to show
/// where SIMD does NOT help much, if your kernel looks like this, fix data movement,
/// not vectorization. With the generated kernel the SIMD variant should be ≈1.0×
/// (bounded by memory, not instructions), not slower.
/// </summary>
[MemoryDiagnoser]
public class BandwidthBoundBench
{
    [Params(1 << 24)]
    public int N;

    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        _input = new float[N];
        _output = new float[N];
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
            _output[i] = _input[i] + 1f;
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.AddOne_Simd(_input, _output, N);
}

/// <summary>
/// Iterative radix-2 FFT, the "scaffold" shape: uniform bit-reversal and stage loops around a
/// vectorized butterfly. The generator emits <c>Transform_Simd</c> only (nested control flow ⇒
/// no <c>_ParallelSimd</c>), so the parallel variant distributes each stage's independent groups
/// across cores over the <c>FftGroup_Simd</c> helper. All three run the identical per-lane-twiddle
/// algorithm, so this isolates the vectorization/threading win. The transform is in place, so each
/// invocation reloads pristine input first (a cheap O(n) copy charged equally to all three).
/// </summary>
[MemoryDiagnoser]
public class FftBench
{
    [Params(1 << 16)]
    public int N;

    private float[] _realSrc = null!;
    private float[] _imagSrc = null!;
    private float[] _real = null!;
    private float[] _imag = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _realSrc = new float[N];
        _imagSrc = new float[N];
        for (int i = 0; i < N; i++)
        { _realSrc[i] = (float)((rng.NextDouble() * 2) - 1); _imagSrc[i] = 0f; }

        _real = new float[N];
        _imag = new float[N];
    }

    private void Reload()
    {
        Array.Copy(_realSrc, _real, N);
        Array.Copy(_imagSrc, _imag, N);
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        Reload();
        ScalarFft(_real, _imag, N);
    }

    [Benchmark]
    public void GeneratedSimd()
    {
        Reload();
        GeneratedKernels.Transform_Simd(_real, _imag);
    }

    [Benchmark]
    public void GeneratedParallelSimd()
    {
        Reload();
        ParallelFft(_real, _imag, N);
    }

    // Same algorithm as the [Spmd] kernel, run scalar, the fair single-thread baseline.
    private static void ScalarFft(float[] real, float[] imag, int n)
    {
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0)
            { j ^= bit; bit >>= 1; }

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
            {
                for (int k = 0; k < half; k++)
                {
                    float wReal = MathF.Cos(k * angle);
                    float wImag = MathF.Sin(k * angle);
                    int even = i + k, odd = i + k + half;
                    float oddReal = (real[odd] * wReal) - (imag[odd] * wImag);
                    float oddImag = (real[odd] * wImag) + (imag[odd] * wReal);
                    float evenReal = real[even], evenImag = imag[even];
                    real[even] = evenReal + oddReal;
                    imag[even] = evenImag + oddImag;
                    real[odd] = evenReal - oddReal;
                    imag[odd] = evenImag - oddImag;
                }
            }
        }
    }

    // Bit-reversal + stages serial; each stage's groups run in parallel over FftGroup_Simd.
    private static void ParallelFft(float[] real, float[] imag, int n)
    {
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0)
            { j ^= bit; bit >>= 1; }

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
            int numGroups = n / len;
            _ = System.Threading.Tasks.Parallel.For(0, numGroups, g =>
                GeneratedKernels.FftGroup_Simd(real, imag, g * len, half, angle));
        }
    }
}

/// <summary>
/// Matrix multiply over 2-D arrays. The generator emits <c>MatMul_Simd</c> only (a <c>float[,]</c>
/// flat-span view can't cross threads ⇒ no <c>_ParallelSimd</c>), so the parallel variant runs one
/// generated row kernel (<c>MatMulRow_Simd</c>) per row across all cores. Vectorization is along the
/// inner dimension: <c>a[row,k]</c> is a contiguous load, <c>b[k,col]</c> a strided gather, so this
/// is a gather-bound kernel (the classic case for transposing b or using SoA).
/// </summary>
[MemoryDiagnoser]
public class MatMulBench
{
    [Params(256)]
    public int N;   // square N×N · N×N

    private float[,] _a = null!;
    private float[,] _b = null!;
    private float[,] _bT = null!;
    private float[,] _c = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _a = new float[N, N];
        _b = new float[N, N];
        _bT = new float[N, N];
        _c = new float[N, N];
        for (int r = 0; r < N; r++)
        {
            for (int col = 0; col < N; col++)
            {
                _a[r, col] = (float)((rng.NextDouble() * 2) - 1);
                _b[r, col] = (float)((rng.NextDouble() * 2) - 1);
            }
        }
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int row = 0; row < N; row++)
        {
            for (int col = 0; col < N; col++)
            {
                float sum = 0f;
                for (int k = 0; k < N; k++)
                    sum += _a[row, k] * _b[k, col];
                _c[row, col] = sum;
            }
        }
    }

    [Benchmark]
    public void GeneratedSimd()
    {
        Memory.Transpose(_b, _bT);
        GeneratedKernels.MatMulT_Simd(_a, _bT, _c, N, N, N);
    }

    [Benchmark]
    public void GeneratedParallelSimd()
    {
        Memory.Transpose(_b, _bT);
        _ = System.Threading.Tasks.Parallel.For(0, N, row =>
            GeneratedKernels.MatMulTRow_Simd(_a, _bT, _c, row, N, N));
    }
}

/// <summary>
/// Element-wise two-input add (c = a + b), the "matrix/vector add". Memory-bound (2 reads + 1
/// write per element): expect ≈1× SIMD (bandwidth, not instructions, is the limit) and the real
/// win from _ParallelSimd spreading the traffic across memory controllers.
/// </summary>
[MemoryDiagnoser]
public class VectorAddBench
{
    [Params(1 << 24)]
    public int N;

    private float[] _a = null!;
    private float[] _b = null!;
    private float[] _c = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _a = new float[N];
        _b = new float[N];
        _c = new float[N];
        for (int i = 0; i < N; i++)
        { _a[i] = (float)rng.NextDouble(); _b[i] = (float)rng.NextDouble(); }
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
            _c[i] = _a[i] + _b[i];
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.VectorAdd_Simd(_a, _b, _c, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.VectorAdd_ParallelSimd(_a, _b, _c, N);
}

/// <summary>
/// BLAS L1 AXPY: y = alpha*x + y (in-place, one FMA per element). The textbook memory-vs-compute
/// boundary case, enough arithmetic to vectorize, but two live streams keep it near bandwidth.
/// </summary>
[MemoryDiagnoser]
public class SaxpyBench
{
    [Params(1 << 24)]
    public int N;

    private float[] _x = null!;
    private float[] _y = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _x = new float[N];
        _y = new float[N];
        for (int i = 0; i < N; i++)
        { _x[i] = (float)rng.NextDouble(); _y[i] = (float)rng.NextDouble(); }
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
            _y[i] = (2.5f * _x[i]) + _y[i];
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.Saxpy_Simd(_x, _y, 2.5f, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.Saxpy_ParallelSimd(_x, _y, 2.5f, N);
}

/// <summary>
/// Integer-lane compute: an int32 bit-mix hash (multiply + xor + shift, no floats). Compute-bound
/// on VInt, expect close to lane-count speedup, the 32-bit-integer analog of ClampScale.
/// </summary>
[MemoryDiagnoser]
public class IntMixBench
{
    [Params(1 << 22)]
    public int N;

    private int[] _input = null!;
    private int[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new int[N];
        _output = new int[N];
        for (int i = 0; i < N; i++)
            _input[i] = rng.Next();
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
        {
            int x = _input[i];
            x ^= x >> 16;
            x *= 0x45d9f3b;
            x ^= x >> 16;
            x *= 0x45d9f3b;
            x ^= x >> 16;
            _output[i] = x;
        }
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.IntMix_Simd(_input, _output, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.IntMix_ParallelSimd(_input, _output, N);
}

/// <summary>
/// Native 64-bit long gang (long[] buffers ⇒ VLong/VMaskD): a splitmix-style bit hash. Half the
/// lane count of the 32-bit kernels, and 64-bit packed multiply needs AVX-512DQ, so this shows
/// the realistic long-lane speedup on additive/bitwise/multiply work.
/// </summary>
[MemoryDiagnoser]
public class LongHashBench
{
    [Params(1 << 22)]
    public int N;

    private long[] _input = null!;
    private long[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new long[N];
        _output = new long[N];
        for (int i = 0; i < N; i++)
            _input[i] = rng.NextInt64();
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
        {
            long x = _input[i];
            x ^= x >> 33;
            x *= 0x5555555555555555L;
            x ^= x >> 29;
            _output[i] = x;
        }
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.LongHash_Simd(_input, _output, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.LongHash_ParallelSimd(_input, _output, N);
}

/// <summary>
/// Double-precision gang (double[] buffers ⇒ VDouble/VMaskD): the double tonemap 1 - e^-x. Half the
/// lane count of the float tonemap plus a longer double Exp polynomial, the 64-bit floating cost.
/// </summary>
[MemoryDiagnoser]
public class DoubleTonemapBench
{
    [Params(1 << 20)]
    public int N;

    private double[] _input = null!;
    private double[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new double[N];
        _output = new double[N];
        for (int i = 0; i < N; i++)
            _input[i] = rng.NextDouble() * 4.0;
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
            _output[i] = 1.0 - Math.Exp(-_input[i] * 1.5);
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.DoubleTonemap_Simd(_input, _output, 1.5, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.DoubleTonemap_ParallelSimd(_input, _output, 1.5, N);
}

/// <summary>
/// Cross-gang-width VDouble2: a double accumulator over float[] data (exact, drift-free sum). Stays
/// a 32-bit gang but accumulates in double, the floating mirror of WideSum. Compare its accuracy
/// and speed to a naive float reduction.
/// </summary>
[MemoryDiagnoser]
public class PreciseSumBench
{
    [Params(1 << 20)]
    public int N;

    private float[] _input = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new float[N];
        for (int i = 0; i < N; i++)
            _input[i] = (float)((rng.NextDouble() * 2) - 1);
    }

    [Benchmark(Baseline = true)]
    public double Scalar()
    {
        double sum = 0;
        for (int i = 0; i < N; i++)
            sum += _input[i];
        return sum;
    }

    [Benchmark]
    public double GeneratedSimd() => GeneratedKernels.PreciseSum_Simd(_input, N);

    [Benchmark]
    public double GeneratedParallelSimd() => GeneratedKernels.PreciseSum_ParallelSimd(_input, N);
}

/// <summary>
/// Reduce-max (not add): horizontal max after a vector-max fold, the min/max reduction path,
/// distinct from the sum reductions. Bandwidth-bound read with a trivial per-element op.
/// </summary>
[MemoryDiagnoser]
public class ArrayMaxBench
{
    [Params(1 << 22)]
    public int N;

    private float[] _input = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new float[N];
        for (int i = 0; i < N; i++)
            _input[i] = (float)((rng.NextDouble() * 1000) - 500);
    }

    [Benchmark(Baseline = true)]
    public float Scalar()
    {
        float mx = float.NegativeInfinity;
        for (int i = 0; i < N; i++)
            mx = MathF.Max(mx, _input[i]);
        return mx;
    }

    [Benchmark]
    public float GeneratedSimd() => GeneratedKernels.ArrayMax_Simd(_input, N);

    [Benchmark]
    public float GeneratedParallelSimd() => GeneratedKernels.ArrayMax_ParallelSimd(_input, N);
}

/// <summary>
/// Non-contiguous gather: out[i] = table[indices[i]]. AVX2 hardware gather (net8+) or an emulated
/// per-lane path, deliberately the slow-memory case the ISPC100 analyzer flags, benched so the
/// gap versus a contiguous load is visible. Random indices defeat the cache on purpose.
/// </summary>
[MemoryDiagnoser]
public class GatherReadBench
{
    [Params(1 << 20)]
    public int N;

    private float[] _table = null!;
    private int[] _indices = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _table = new float[N];
        _indices = new int[N];
        _output = new float[N];
        for (int i = 0; i < N; i++)
        { _table[i] = (float)rng.NextDouble(); _indices[i] = rng.Next(0, N); }
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
            _output[i] = _table[_indices[i]];
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.GatherRead_Simd(_table, _indices, _output, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.GatherRead_ParallelSimd(_table, _indices, _output, N);
}

/// <summary>
/// Trig-heavy: Sin·Cos per element. A different transcendental mix from the Exp tonemap, both are
/// range-reduced polynomial approximations, and both are where scalar Math is slowest, so expect a
/// large SIMD win.
/// </summary>
[MemoryDiagnoser]
public class SinCosBench
{
    [Params(1 << 20)]
    public int N;

    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        Random rng = new Random(42);
        _input = new float[N];
        _output = new float[N];
        for (int i = 0; i < N; i++)
            _input[i] = (float)((rng.NextDouble() * 20) - 10);
    }

    [Benchmark(Baseline = true)]
    public void Scalar()
    {
        for (int i = 0; i < N; i++)
        {
            float x = _input[i];
            _output[i] = MathF.Sin(x) * MathF.Cos(x);
        }
    }

    [Benchmark]
    public void GeneratedSimd() => GeneratedKernels.SinCos_Simd(_input, _output, N);

    [Benchmark]
    public void GeneratedParallelSimd() => GeneratedKernels.SinCos_ParallelSimd(_input, _output, N);
}
