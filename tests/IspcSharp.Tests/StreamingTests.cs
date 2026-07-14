using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// [Spmd(Streaming = true)] emits non-temporal stores + a fence. The runtime falls back to an
/// ordinary store for unaligned buffers, so results are always correct regardless of alignment.
/// </summary>
public static partial class StreamingKernels
{
    [Spmd(Streaming = true)]
    public static void VectorAdd(float[] a, float[] b, float[] c, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            c[i] = a[i] + b[i];
        }
    }

    [Spmd(Streaming = true)]
    public static void Saxpy(float[] x, float[] y, float alpha, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            y[i] = (alpha * x[i]) + y[i];
        }
    }
}

public class StreamingTests
{
    private const int N = (8192 * 2) + 5;   // not a lane multiple → exercises the scalar tail

    [Fact]
    public void VectorAdd_Streaming_MatchesScalar()
    {
        Random r = new Random(1);
        float[] a = new float[N];
        float[] b = new float[N];
        float[] c = new float[N];
        for (int i = 0; i < N; i++)
        { a[i] = (float)r.NextDouble(); b[i] = (float)r.NextDouble(); }

        StreamingKernels.VectorAdd_Simd(a, b, c, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] + b[i], c[i], 5);

        StreamingKernels.VectorAdd_ParallelSimd(a, b, c, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] + b[i], c[i], 5);
    }

    [Fact]
    public void Saxpy_Streaming_MatchesScalar()
    {
        Random r = new Random(2);
        float[] x = new float[N];
        float[] y = new float[N];
        float[] expected = new float[N];
        float alpha = 2.5f;
        for (int i = 0; i < N; i++)
        { x[i] = (float)r.NextDouble(); y[i] = (float)r.NextDouble(); expected[i] = (alpha * x[i]) + y[i]; }

        StreamingKernels.Saxpy_Simd(x, y, alpha, N);
        // 'alpha*x + y' fuses into an FMA (single rounding) → ~1 ULP vs the naive scalar reference.
        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(y[i] - expected[i]) <= 1e-5f * (1f + MathF.Abs(expected[i])), $"i={i}");
    }
}
