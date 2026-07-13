using System;
using IspcSharp;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>Blittable [SpmdStruct] values and [SpmdFunction] helpers inside [Spmd] kernels.</summary>
[SpmdStruct]
public struct Complex
{
    public float Re;
    public float Im;
}

[SpmdStruct]
public struct Vec3
{
    public float X;
    public float Y;
    public float Z;
}

/// <summary>
/// Mixed-typed fields (float + int), companion has a VFloat and a VInt field.
/// </summary>
[SpmdStruct]
public struct Hit
{
    public float T;
    public int Id;
}

public static partial class StructKernels
{
    /// <summary>
    /// Helper with primitive args/return.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    [SpmdFunction]
    public static float Lerp(float a, float b, float t) => a + (b - a) * t;

    /// <summary>
    /// Helper that calls another helper (function composition).
    /// </summary>
    /// <param name="d"></param>
    /// <returns></returns>
    [SpmdFunction]
    public static float Attenuate(float d) => 1f / (1f + d * d);

    [SpmdFunction]
    public static float Falloff(float d, float k) => Attenuate(d) * k;

    // Helper taking and returning a struct.
    [SpmdFunction]
    public static Complex CMul(Complex a, Complex b)
        => new Complex { Re = a.Re * b.Re - a.Im * b.Im, Im = a.Re * b.Im + a.Im * b.Re };

    /// <summary>
    /// Helper composing another helper + a struct return.
    /// </summary>
    /// <param name="v"></param>
    /// <param name="s"></param>
    /// <returns></returns>
    [SpmdFunction]
    public static Vec3 Scale(Vec3 v, float s)
        => new Vec3 { X = v.X * s, Y = v.Y * s, Z = v.Z * s };

    /// <summary>
    /// Kernel: primitive helper.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="o"></param>
    /// <param name="t"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void LerpArrays(float[] a, float[] b, float[] o, float t, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            o[i] = Lerp(a[i], b[i], t);
        }
    }

    /// <summary>
    /// Kernel: struct locals + struct-returning helper.
    /// </summary>
    /// <param name="ar"></param>
    /// <param name="ai"></param>
    /// <param name="br"></param>
    /// <param name="bi"></param>
    /// <param name="or"></param>
    /// <param name="oi"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void ComplexMul(float[] ar, float[] ai, float[] br, float[] bi, float[] or, float[] oi, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            Complex a = new Complex { Re = ar[i], Im = ai[i] };
            Complex b = new Complex { Re = br[i], Im = bi[i] };
            Complex c = CMul(a, b);
            or[i] = c.Re;
            oi[i] = c.Im;
        }
    }

    /// <summary>
    /// Helper returning a struct via a ternary → per-field masked Select (mixed float/int fields).
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    [SpmdFunction]
    public static Hit Closer(Hit a, Hit b) => a.T < b.T ? a : b;

    /// <summary>
    /// Kernel: keep the nearest hit per element across two candidate streams.
    /// </summary>
    /// <param name="ta"></param>
    /// <param name="ida"></param>
    /// <param name="tb"></param>
    /// <param name="idb"></param>
    /// <param name="outT"></param>
    /// <param name="outId"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void NearestHit(float[] ta, int[] ida, float[] tb, int[] idb, float[] outT, int[] outId, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            Hit a = new Hit { T = ta[i], Id = ida[i] };
            Hit b = new Hit { T = tb[i], Id = idb[i] };
            Hit best = Closer(a, b);
            outT[i] = best.T;
            outId[i] = best.Id;
        }
    }

    /// <summary>
    /// Kernel: helper that composes another helper.
    /// </summary>
    /// <param name="d"></param>
    /// <param name="o"></param>
    /// <param name="k"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void FalloffArray(float[] d, float[] o, float k, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            o[i] = Falloff(d[i], k);
        }
    }

    /// <summary>
    /// Kernel: AoS struct buffer, whole-struct read + whole-struct write (strided gather/scatter).
    /// </summary>
    /// <param name="pts"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void NormalizeBuffer(Vec3[] pts, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            Vec3 v = pts[i];
            float len = MathF.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
            float inv = 1f / len;
            pts[i] = new Vec3 { X = v.X * inv, Y = v.Y * inv, Z = v.Z * inv };
        }
    }

    /// <summary>
    /// Kernel: AoS struct buffer, single-field read + write (buf[i].field).
    /// </summary>
    /// <param name="pts"></param>
    /// <param name="dx"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void TranslateX(Vec3[] pts, float dx, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            pts[i].X = pts[i].X + dx;
        }
    }

    /// <summary>
    /// Kernel: struct local mutated field-by-field.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="z"></param>
    /// <param name="s"></param>
    /// <param name="count"></param>
    [Spmd]
    public static void ScaleVectors(float[] x, float[] y, float[] z, float s, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            Vec3 v = new Vec3 { X = x[i], Y = y[i], Z = z[i] };
            v = Scale(v, s);
            v.X = v.X + 1f;                 // field write
            x[i] = v.X;
            y[i] = v.Y;
            z[i] = v.Z;
        }
    }
}

public class StructAndFunctionTests
{
    private static float[] Rand(int n, Random r)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1);
        return a;
    }

    private const int N = 8192 * 2 + 3;

    [Fact]
    public void LerpArrays_MatchesScalar()
    {
        var r = new Random(1);
        float[] a = Rand(N, r), b = Rand(N, r), o = new float[N];
        float t = 0.3f;
        StructKernels.LerpArrays_Simd(a, b, o, t, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] + (b[i] - a[i]) * t, o[i], 3);
    }

    [Fact]
    public void NearestHit_StructTernary_SelectsPerLane()
    {
        var r = new Random(7);
        float[] ta = Rand(N, r), tb = Rand(N, r);
        var ida = new int[N]; var idb = new int[N];
        for (int i = 0; i < N; i++) { ida[i] = i; idb[i] = i + 1_000_000; }
        var outT = new float[N]; var outId = new int[N];
        StructKernels.NearestHit_Simd(ta, ida, tb, idb, outT, outId, N);
        for (int i = 0; i < N; i++)
        {
            bool aWins = ta[i] < tb[i];
            Assert.Equal(aWins ? ta[i] : tb[i], outT[i], 3);
            Assert.Equal(aWins ? ida[i] : idb[i], outId[i]);
        }
    }

    [Fact]
    public void FalloffArray_ComposesHelpers()
    {
        var r = new Random(6);
        float[] d = Rand(N, r), o = new float[N];
        float k = 4f;
        StructKernels.FalloffArray_Simd(d, o, k, N);
        for (int i = 0; i < N; i++)
        {
            // Relative tolerance, the generated kernel fuses '1 + d*d' into an FMA (single rounding),
            // so it differs from the naive scalar reference by ~1 ULP (decimal-place rounding is fragile).
            float expected = 1f / (1f + d[i] * d[i]) * k;
            Assert.True(MathF.Abs(o[i] - expected) <= 1e-4f * (1f + MathF.Abs(expected)),
                $"i={i}: expected={expected}, actual={o[i]}");
        }
    }

    [Fact]
    public void ComplexMul_MatchesScalar()
    {
        var r = new Random(2);
        float[] ar = Rand(N, r), ai = Rand(N, r), br = Rand(N, r), bi = Rand(N, r);
        float[] or = new float[N], oi = new float[N];
        StructKernels.ComplexMul_Simd(ar, ai, br, bi, or, oi, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(ar[i] * br[i] - ai[i] * bi[i], or[i], 3);
            Assert.Equal(ar[i] * bi[i] + ai[i] * br[i], oi[i], 3);
        }
    }

    [Fact]
    public void NormalizeBuffer_MatchesScalar()
    {
        var r = new Random(4);
        var pts = new Vec3[N];
        for (int i = 0; i < N; i++) pts[i] = new Vec3 { X = (float)(r.NextDouble() + 0.1), Y = (float)(r.NextDouble() + 0.1), Z = (float)(r.NextDouble() + 0.1) };
        var expected = new Vec3[N];
        for (int i = 0; i < N; i++)
        {
            float len = MathF.Sqrt(pts[i].X * pts[i].X + pts[i].Y * pts[i].Y + pts[i].Z * pts[i].Z);
            float inv = 1f / len;
            expected[i] = new Vec3 { X = pts[i].X * inv, Y = pts[i].Y * inv, Z = pts[i].Z * inv };
        }
        StructKernels.NormalizeBuffer_Simd(pts, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(expected[i].X, pts[i].X, 3);
            Assert.Equal(expected[i].Y, pts[i].Y, 3);
            Assert.Equal(expected[i].Z, pts[i].Z, 3);
        }
    }

    [Fact]
    public void TranslateX_MatchesScalar()
    {
        var r = new Random(5);
        var pts = new Vec3[N];
        for (int i = 0; i < N; i++) pts[i] = new Vec3 { X = (float)r.NextDouble(), Y = (float)r.NextDouble(), Z = (float)r.NextDouble() };
        var expX = new float[N];
        float dx = 3.5f;
        for (int i = 0; i < N; i++) expX[i] = pts[i].X + dx;
        StructKernels.TranslateX_Simd(pts, dx, N);
        for (int i = 0; i < N; i++) Assert.Equal(expX[i], pts[i].X, 3);
    }

    [Fact]
    public void ScaleVectors_MatchesScalar()
    {
        var r = new Random(3);
        float[] x = Rand(N, r), y = Rand(N, r), z = Rand(N, r);
        var (ex, ey, ez) = (new float[N], new float[N], new float[N]);
        float s = 2.5f;
        for (int i = 0; i < N; i++) { ex[i] = x[i] * s + 1f; ey[i] = y[i] * s; ez[i] = z[i] * s; }
        StructKernels.ScaleVectors_Simd(x, y, z, s, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(ex[i], x[i], 3);
            Assert.Equal(ey[i], y[i], 3);
            Assert.Equal(ez[i], z[i], 3);
        }
    }
}
