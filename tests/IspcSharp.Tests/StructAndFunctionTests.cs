using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Blittable [SpmdStruct] values and [SpmdFunction] helpers inside [Spmd] kernels.
/// </summary>
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

/// <summary>
/// Struct built from reduction accumulators and returned from a kernel (ISPC export-style).
/// </summary>
[SpmdStruct]
public struct Stats
{
    public float Sum;
    public float Min;
    public float Max;
}

/// <summary>
/// ISPC-style fixed-size array members ([SpmdArray(N)]), including differently-typed arrays.
/// </summary>
[SpmdStruct]
public struct Poly
{
    [SpmdArray(3)] public float[] Coef;   // 3 float gangs held SoA in registers
    [SpmdArray(2)] public int[] Tag;      // 2 int gangs (different element type)
    public float Bias;                    // plain scalar field alongside the arrays
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
    public static float Lerp(float a, float b, float t) => a + ((b - a) * t);

    /// <summary>
    /// Helper that calls another helper (function composition).
    /// </summary>
    /// <param name="d"></param>
    /// <returns></returns>
    [SpmdFunction]
    public static float Attenuate(float d) => 1f / (1f + (d * d));

    [SpmdFunction]
    public static float Falloff(float d, float k) => Attenuate(d) * k;

    // Helper taking and returning a struct.
    [SpmdFunction]
    public static Complex CMul(Complex a, Complex b)
        => new Complex
        { Re = (a.Re * b.Re) - (a.Im * b.Im), Im = (a.Re * b.Im) + (a.Im * b.Re) };

    /// <summary>
    /// Helper composing another helper + a struct return.
    /// </summary>
    /// <param name="v"></param>
    /// <param name="s"></param>
    /// <returns></returns>
    [SpmdFunction]
    public static Vec3 Scale(Vec3 v, float s)
        => new Vec3
        { X = v.X * s, Y = v.Y * s, Z = v.Z * s };

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
        foreach (int i in Spmd.Range(count))
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
        foreach (int i in Spmd.Range(count))
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
        foreach (int i in Spmd.Range(count))
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
        foreach (int i in Spmd.Range(count))
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
        foreach (int i in Spmd.Range(count))
        {
            Vec3 v = pts[i];
            float len = MathF.Sqrt((v.X * v.X) + (v.Y * v.Y) + (v.Z * v.Z));
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
        foreach (int i in Spmd.Range(count))
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
        foreach (int i in Spmd.Range(count))
        {
            Vec3 v = new Vec3 { X = x[i], Y = y[i], Z = z[i] };
            v = Scale(v, s);
            v.X = v.X + 1f;                 // field write
            x[i] = v.X;
            y[i] = v.Y;
            z[i] = v.Z;
        }
    }

    /// <summary>
    /// Kernel: mixed-field struct buffer (Hit { float T; int Id; }[]), whole-struct read + write.
    /// The float and int fields are gathered/scattered through their own correctly-typed flat views.
    /// </summary>
    [Spmd]
    public static void BumpHits(Hit[] hits, float dt, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            Hit h = hits[i];            // whole-struct gather (float T + int Id)
            h.T = h.T + dt;             // float field arithmetic
            h.Id = h.Id + 1;            // int field arithmetic
            hits[i] = h;                // whole-struct scatter
        }
    }

    /// <summary>
    /// Kernel: mixed-field struct buffer, single-field access on each differently-typed field.
    /// </summary>
    [Spmd]
    public static void RetagHits(Hit[] hits, int delta, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            hits[i].Id = hits[i].Id + delta;    // int field gather + scatter
            hits[i].T = hits[i].T * 2f;         // float field gather + scatter
        }
    }

    /// <summary>
    /// Kernel returning a [SpmdStruct] built from three reduction accumulators after the loop
    /// (sum / min / max), the ISPC 'export a struct' pattern.
    /// </summary>
    [Spmd]
    public static Stats Summarize(float[] a, int count)
    {
        float sum = 0f;
        float mn = float.MaxValue;
        float mx = float.NegativeInfinity;
        foreach (int i in Spmd.Range(count))
        {
            sum += a[i];
            mn = Math.Min(mn, a[i]);
            mx = Math.Max(mx, a[i]);
        }

        return new Stats { Sum = sum, Min = mn, Max = mx };
    }

    /// <summary>
    /// Helper reading a struct's fixed-size array members by constant index.
    /// </summary>
    [SpmdFunction]
    public static float EvalPoly(Poly p, float x)
        => p.Coef[0] + (p.Coef[1] * x) + (p.Coef[2] * (x * x)) + p.Bias;

    /// <summary>
    /// Kernel: build a struct with array members (array + sized-empty initializers), pass it to a
    /// helper that indexes the array members, ISPC's struct-with-array-members feature.
    /// </summary>
    [Spmd]
    public static void ApplyPoly(float[] c0, float[] c1, float[] c2, float[] bias, float[] x, float[] o, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            Poly p = new Poly { Coef = new float[] { c0[i], c1[i], c2[i] }, Tag = new int[2], Bias = bias[i] };
            o[i] = EvalPoly(p, x[i]);
        }
    }

    /// <summary>
    /// Kernel: allocate a struct with array members, write each element (float and int arrays), read back.
    /// </summary>
    [Spmd]
    public static void PolyTags(float[] x, float[] o, int[] tag, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            Poly p = new Poly { Coef = new float[3], Tag = new int[2], Bias = 0f };
            p.Coef[0] = x[i];
            p.Coef[1] = x[i] * 2f;
            p.Tag[0] = i;
            p.Tag[1] = i * 2;
            o[i] = p.Coef[0] + p.Coef[1];
            tag[i] = p.Tag[0] + p.Tag[1];
        }
    }

    /// <summary>
    /// Uniform [SpmdStruct] kernel parameter (ISPC's 'uniform struct' export arg): scalar
    /// field, literal-indexed array members (float AND int), all broadcast per gang.
    /// </summary>
    [Spmd]
    public static void EvalUniformPoly(float[] xs, float[] ys, Poly p, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            float x = xs[i];
            ys[i] = (p.Coef[0] + (p.Coef[1] * x) + (p.Coef[2] * (x * x)) + p.Bias) * p.Tag[0];
        }
    }

    /// <summary>
    /// Uniform struct array member indexed by a RUNTIME uniform value — legal on a uniform
    /// param (it's an ordinary scalar array), unlike varying struct locals which need a
    /// compile-time literal.
    /// </summary>
    [Spmd]
    public static void ScaleBySelectedCoef(float[] xs, float[] ys, Poly p, int idx, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            ys[i] = xs[i] * p.Coef[idx];
        }
    }

    /// <summary>
    /// Uniform→varying struct broadcast: assigning the uniform param to a struct local
    /// splats every field gang (ISPC's implicit uniform→varying struct conversion).
    /// </summary>
    [Spmd]
    public static void PolyThroughLocal(float[] xs, float[] ys, Poly p, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            Poly q = p;
            ys[i] = (q.Coef[1] * xs[i]) + q.Bias;
        }
    }

    /// <summary>
    /// Uniform struct passed straight into a [SpmdFunction] helper's struct parameter.
    /// </summary>
    [Spmd]
    public static void MulByUniformComplex(float[] re, float[] im, float[] outRe, float[] outIm, Complex w, int count)
    {
        foreach (int i in Spmd.Range(count))
        {
            Complex v = new Complex { Re = re[i], Im = im[i] };
            Complex r = CMul(v, w);
            outRe[i] = r.Re;
            outIm[i] = r.Im;
        }
    }
}

public class StructAndFunctionTests
{
    private static float[] Rand(int n, Random r)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = (float)((r.NextDouble() * 2) - 1);
        return a;
    }

    private const int N = (8192 * 2) + 3;

    [Fact]
    public void LerpArrays_MatchesScalar()
    {
        Random r = new Random(1);
        float[] a = Rand(N, r), b = Rand(N, r), o = new float[N];
        float t = 0.3f;
        StructKernels.LerpArrays_Simd(a, b, o, t, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] + ((b[i] - a[i]) * t), o[i], 3);
    }

    [Fact]
    public void NearestHit_StructTernary_SelectsPerLane()
    {
        Random r = new Random(7);
        float[] ta = Rand(N, r), tb = Rand(N, r);
        int[] ida = new int[N];
        int[] idb = new int[N];
        for (int i = 0; i < N; i++)
        { ida[i] = i; idb[i] = i + 1_000_000; }

        float[] outT = new float[N];
        int[] outId = new int[N];
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
        Random r = new Random(6);
        float[] d = Rand(N, r), o = new float[N];
        float k = 4f;
        StructKernels.FalloffArray_Simd(d, o, k, N);
        for (int i = 0; i < N; i++)
        {
            // Relative tolerance, the generated kernel fuses '1 + d*d' into an FMA (single rounding),
            // so it differs from the naive scalar reference by ~1 ULP (decimal-place rounding is fragile).
            float expected = 1f / (1f + (d[i] * d[i])) * k;
            Assert.True(MathF.Abs(o[i] - expected) <= 1e-4f * (1f + MathF.Abs(expected)),
                $"i={i}: expected={expected}, actual={o[i]}");
        }
    }

    [Fact]
    public void ComplexMul_MatchesScalar()
    {
        Random r = new Random(2);
        float[] ar = Rand(N, r), ai = Rand(N, r), br = Rand(N, r), bi = Rand(N, r);
        float[] or = new float[N], oi = new float[N];
        StructKernels.ComplexMul_Simd(ar, ai, br, bi, or, oi, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal((ar[i] * br[i]) - (ai[i] * bi[i]), or[i], 3);
            Assert.Equal((ar[i] * bi[i]) + (ai[i] * br[i]), oi[i], 3);
        }
    }

    [Fact]
    public void NormalizeBuffer_MatchesScalar()
    {
        Random r = new Random(4);
        Vec3[] pts = new Vec3[N];
        for (int i = 0; i < N; i++)
            pts[i] = new Vec3 { X = (float)(r.NextDouble() + 0.1), Y = (float)(r.NextDouble() + 0.1), Z = (float)(r.NextDouble() + 0.1) };
        Vec3[] expected = new Vec3[N];
        for (int i = 0; i < N; i++)
        {
            float len = MathF.Sqrt((pts[i].X * pts[i].X) + (pts[i].Y * pts[i].Y) + (pts[i].Z * pts[i].Z));
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
        Random r = new Random(5);
        Vec3[] pts = new Vec3[N];
        for (int i = 0; i < N; i++)
            pts[i] = new Vec3 { X = (float)r.NextDouble(), Y = (float)r.NextDouble(), Z = (float)r.NextDouble() };
        float[] expX = new float[N];
        float dx = 3.5f;
        for (int i = 0; i < N; i++)
            expX[i] = pts[i].X + dx;
        StructKernels.TranslateX_Simd(pts, dx, N);
        for (int i = 0; i < N; i++)
            Assert.Equal(expX[i], pts[i].X, 3);
    }

    [Fact]
    public void ScaleVectors_MatchesScalar()
    {
        Random r = new Random(3);
        float[] x = Rand(N, r), y = Rand(N, r), z = Rand(N, r);
        (float[]? ex, float[]? ey, float[]? ez) = (new float[N], new float[N], new float[N]);
        float s = 2.5f;
        for (int i = 0; i < N; i++)
        { ex[i] = (x[i] * s) + 1f; ey[i] = y[i] * s; ez[i] = z[i] * s; }

        StructKernels.ScaleVectors_Simd(x, y, z, s, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(ex[i], x[i], 3);
            Assert.Equal(ey[i], y[i], 3);
            Assert.Equal(ez[i], z[i], 3);
        }
    }

    [Fact]
    public void BumpHits_MixedFieldBuffer_MatchesScalar()
    {
        Random r = new Random(11);
        Hit[] hits = new Hit[N];
        for (int i = 0; i < N; i++)
            hits[i] = new Hit { T = (float)(r.NextDouble() * 10), Id = i };
        (float[]? expT, int[]? expId) = (new float[N], new int[N]);
        float dt = 1.25f;
        for (int i = 0; i < N; i++)
        { expT[i] = hits[i].T + dt; expId[i] = hits[i].Id + 1; }

        StructKernels.BumpHits_Simd(hits, dt, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(expT[i], hits[i].T, 3);
            Assert.Equal(expId[i], hits[i].Id);
        }
    }

    [Fact]
    public void RetagHits_MixedFieldBuffer_SingleFieldAccess()
    {
        Random r = new Random(12);
        Hit[] hits = new Hit[N];
        for (int i = 0; i < N; i++)
            hits[i] = new Hit { T = (float)((r.NextDouble() * 4) - 2), Id = i * 3 };
        (float[]? expT, int[]? expId) = (new float[N], new int[N]);
        int delta = 100;
        for (int i = 0; i < N; i++)
        { expId[i] = hits[i].Id + delta; expT[i] = hits[i].T * 2f; }

        StructKernels.RetagHits_Simd(hits, delta, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(expId[i], hits[i].Id);
            Assert.Equal(expT[i], hits[i].T, 3);
        }
    }

    [Fact]
    public void Summarize_ReturnsStruct_MatchesScalar()
    {
        Random r = new Random(13);
        float[] a = Rand(N, r);
        float sum = 0f, mn = float.MaxValue, mx = float.NegativeInfinity;
        for (int i = 0; i < N; i++)
        { sum += a[i]; mn = MathF.Min(mn, a[i]); mx = MathF.Max(mx, a[i]); }

        Stats s = StructKernels.Summarize_Simd(a, N);
        Assert.Equal(sum, s.Sum, 2);
        Assert.Equal(mn, s.Min, 4);
        Assert.Equal(mx, s.Max, 4);

        // Parallel variant reassociates the sum across chunks, so compare with a relative tolerance;
        // min/max are order-independent and stay exact.
        Stats p = StructKernels.Summarize_ParallelSimd(a, N);
        Assert.True(MathF.Abs(p.Sum - sum) <= 1e-3f * (1f + MathF.Abs(sum)), $"sum: {p.Sum} vs {sum}");
        Assert.Equal(mn, p.Min, 4);
        Assert.Equal(mx, p.Max, 4);
    }

    [Fact]
    public void ApplyPoly_ArrayMembers_MatchesScalar()
    {
        Random r = new Random(21);
        float[] c0 = Rand(N, r), c1 = Rand(N, r), c2 = Rand(N, r), bias = Rand(N, r), x = Rand(N, r), o = new float[N];
        StructKernels.ApplyPoly_Simd(c0, c1, c2, bias, x, o, N);
        for (int i = 0; i < N; i++)
        {
            // Helper fuses coef*x terms into FMAs (single rounding), so use a relative tolerance.
            float e = c0[i] + (c1[i] * x[i]) + (c2[i] * (x[i] * x[i])) + bias[i];
            Assert.True(MathF.Abs(o[i] - e) <= 1e-4f * (1f + MathF.Abs(e)), $"i={i}: {o[i]} vs {e}");
        }
    }

    [Fact]
    public void PolyTags_ArrayMemberWrites_MatchesScalar()
    {
        Random r = new Random(22);
        float[] x = Rand(N, r), o = new float[N];
        int[] tag = new int[N];
        StructKernels.PolyTags_Simd(x, o, tag, N);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(x[i] + (x[i] * 2f), o[i], 3);
            Assert.Equal(i + (i * 2), tag[i]);
        }
    }

    private static Poly MakePoly() => new Poly
    {
        Coef = [0.5f, -1.25f, 2.75f],
        Tag = [3, 7],
        Bias = 0.125f,
    };

    [Fact]
    public void EvalUniformPoly_UniformStructArg_MatchesScalar()
    {
        Random r = new Random(23);
        float[] xs = Rand(N, r);
        float[] expected = new float[N];
        float[] actual = new float[N];
        Poly p = MakePoly();
        StructKernels.EvalUniformPoly(xs, expected, p, N);

        StructKernels.EvalUniformPoly_Simd(xs, actual, p, N);

        for (int i = 0; i < N; i++)
        {
            // coef*x terms fuse into FMAs (single rounding), so compare with a relative tolerance.
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-4f * (1f + MathF.Abs(expected[i])),
                $"i={i}: {actual[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void EvalUniformPoly_ParallelSimd_MatchesScalar()
    {
        Random r = new Random(24);
        float[] xs = Rand(N, r);
        float[] expected = new float[N];
        float[] actual = new float[N];
        Poly p = MakePoly();
        StructKernels.EvalUniformPoly(xs, expected, p, N);

        StructKernels.EvalUniformPoly_ParallelSimd(xs, actual, p, N, minChunkSize: 64);

        for (int i = 0; i < N; i++)
        {
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-4f * (1f + MathF.Abs(expected[i])),
                $"i={i}: {actual[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void ScaleBySelectedCoef_RuntimeUniformIndex()
    {
        Random r = new Random(25);
        float[] xs = Rand(N, r);
        float[] actual = new float[N];
        Poly p = MakePoly();

        for (int idx = 0; idx < 3; idx++)
        {
            StructKernels.ScaleBySelectedCoef_Simd(xs, actual, p, idx, N);
            for (int i = 0; i < N; i++)
                Assert.Equal(xs[i] * p.Coef[idx], actual[i]);
        }
    }

    [Fact]
    public void PolyThroughLocal_UniformToVaryingBroadcast()
    {
        Random r = new Random(26);
        float[] xs = Rand(N, r);
        float[] expected = new float[N];
        float[] actual = new float[N];
        Poly p = MakePoly();
        StructKernels.PolyThroughLocal(xs, expected, p, N);

        StructKernels.PolyThroughLocal_Simd(xs, actual, p, N);

        for (int i = 0; i < N; i++)
        {
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-4f * (1f + MathF.Abs(expected[i])),
                $"i={i}: {actual[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void MulByUniformComplex_StructArgToHelper()
    {
        Random r = new Random(27);
        float[] re = Rand(N, r), im = Rand(N, r);
        float[] expRe = new float[N], expIm = new float[N];
        float[] actRe = new float[N], actIm = new float[N];
        Complex w = new Complex { Re = 0.6f, Im = -0.8f };
        StructKernels.MulByUniformComplex(re, im, expRe, expIm, w, N);

        StructKernels.MulByUniformComplex_Simd(re, im, actRe, actIm, w, N);

        for (int i = 0; i < N; i++)
        {
            Assert.True(MathF.Abs(actRe[i] - expRe[i]) <= 1e-4f * (1f + MathF.Abs(expRe[i])), $"re i={i}");
            Assert.True(MathF.Abs(actIm[i] - expIm[i]) <= 1e-4f * (1f + MathF.Abs(expIm[i])), $"im i={i}");
        }
    }
}
