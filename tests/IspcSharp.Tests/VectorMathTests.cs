using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for the vectorized math library. Transcendentals are polynomial
/// approximations (~1e-6), so we test against MathF with appropriate tolerance.
/// </summary>
public class VectorMathTests
{
    private const float ApproxTolerance = 1e-4f;

    private static VFloat Range(float start, float step, int n)
    {
        float[] data = new float[n];
        for (int i = 0; i < n; i++)
            data[i] = start + (i * step);
        return VFloat.Load(data, 0);
    }

    private static void AssertApprox(VFloat actual, VFloat expected, float tol)
    {
        int w = VFloat.LaneCount;
        for (int l = 0; l < w; l++)
        {
            float a = actual.GetLane(l);
            float e = expected.GetLane(l);
            float allowed = tol * Math.Max(1f, Math.Abs(e));
            Assert.True(Math.Abs(a - e) <= allowed,
                $"Lane {l}: expected {e:G9}, got {a:G9} (diff {Math.Abs(a - e):G9}, tol {allowed:G9})");
        }
    }

    [Fact]
    public void Sqrt_MatchesMathF()
    {
        VFloat x = Range(0.5f, 1.5f, VFloat.LaneCount);
        VFloat result = VectorMath.Sqrt(x);

        for (int l = 0; l < VFloat.LaneCount; l++)
        {
            float expected = MathF.Sqrt(x.GetLane(l));
            Assert.Equal(expected, result.GetLane(l), 3);
        }
    }

    [Fact]
    public void Abs_MatchesMathF()
    {
        VFloat x = Range(-4f, 1f, VFloat.LaneCount);
        VFloat result = VectorMath.Abs(x);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(MathF.Abs(x.GetLane(l)), result.GetLane(l), 3);
    }

    [Fact]
    public void MinMax_MatchMathF()
    {
        VFloat a = Range(1f, 1f, VFloat.LaneCount);
        VFloat b = Range(10f, -1f, VFloat.LaneCount);

        VFloat min = VectorMath.Min(a, b);
        VFloat max = VectorMath.Max(a, b);

        for (int l = 0; l < VFloat.LaneCount; l++)
        {
            Assert.Equal(MathF.Min(a.GetLane(l), b.GetLane(l)), min.GetLane(l), 3);
            Assert.Equal(MathF.Max(a.GetLane(l), b.GetLane(l)), max.GetLane(l), 3);
        }
    }

    [Fact]
    public void Clamp_MatchesMathF()
    {
        VFloat x = Range(-2f, 1f, VFloat.LaneCount);
        VFloat lo = new VFloat(0f);
        VFloat hi = new VFloat(1f);

        VFloat result = VectorMath.Clamp(x, lo, hi);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(Math.Clamp(x.GetLane(l), 0f, 1f), result.GetLane(l), 3);
    }

    [Fact]
    public void Lerp_MatchesFormula()
    {
        VFloat a = Range(0f, 1f, VFloat.LaneCount);
        VFloat b = Range(10f, 1f, VFloat.LaneCount);
        VFloat t = new VFloat(0.5f);

        VFloat result = VectorMath.Lerp(a, b, t);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(a.GetLane(l) + ((b.GetLane(l) - a.GetLane(l)) * 0.5f), result.GetLane(l), 3);
    }

    [Fact]
    public void Exp_MatchesMathF()
    {
        VFloat x = Range(-3f, 0.5f, VFloat.LaneCount);
        VFloat result = VectorMath.Exp(x);
        AssertApprox(result, BuildExpected(MathF.Exp, x), ApproxTolerance);
    }

    [Fact]
    public void Log_MatchesMathF()
    {
        VFloat x = Range(0.1f, 1f, VFloat.LaneCount);
        VFloat result = VectorMath.Log(x);
        AssertApprox(result, BuildExpected(MathF.Log, x), ApproxTolerance);
    }

    [Fact]
    public void Sin_MatchesMathF()
    {
        VFloat x = Range(-3f, 0.5f, VFloat.LaneCount);
        VFloat result = VectorMath.Sin(x);
        AssertApprox(result, BuildExpected(MathF.Sin, x), ApproxTolerance);
    }

    [Fact]
    public void Cos_MatchesMathF()
    {
        VFloat x = Range(-3f, 0.5f, VFloat.LaneCount);
        VFloat result = VectorMath.Cos(x);
        AssertApprox(result, BuildExpected(MathF.Cos, x), ApproxTolerance);
    }

    [Fact]
    public void Tan_MatchesMathF()
    {
        VFloat x = Range(-1f, 0.2f, VFloat.LaneCount);
        VFloat result = VectorMath.Tan(x);
        AssertApprox(result, BuildExpected(MathF.Tan, x), ApproxTolerance);
    }

    [Fact]
    public void Tanh_MatchesMathF()
    {
        VFloat x = Range(-3f, 0.5f, VFloat.LaneCount);
        VFloat result = VectorMath.Tanh(x);
        AssertApprox(result, BuildExpected(MathF.Tanh, x), ApproxTolerance);
    }

    [Fact]
    public void Sigmoid_MatchesFormula()
    {
        VFloat x = Range(-4f, 1f, VFloat.LaneCount);
        VFloat result = VectorMath.Sigmoid(x);

        for (int l = 0; l < VFloat.LaneCount; l++)
        {
            float expected = 1f / (1f + MathF.Exp(-x.GetLane(l)));
            Assert.Equal(expected, result.GetLane(l), 4);
        }
    }

    [Fact]
    public void Pow_MatchesMathF()
    {
        VFloat x = Range(0.5f, 0.5f, VFloat.LaneCount);
        VFloat y = new VFloat(2.5f);
        VFloat result = VectorMath.Pow(x, y);

        for (int l = 0; l < VFloat.LaneCount; l++)
        {
            float expected = MathF.Pow(x.GetLane(l), 2.5f);
            float allowed = ApproxTolerance * Math.Max(1f, Math.Abs(expected));
            Assert.True(Math.Abs(result.GetLane(l) - expected) <= allowed,
                $"Lane {l}: expected {expected}, got {result.GetLane(l)}");
        }
    }

    [Fact]
    public void Floor_MatchesMathF()
    {
        VFloat x = Range(-3.7f, 0.9f, VFloat.LaneCount);
        VFloat result = VectorMath.Floor(x);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(MathF.Floor(x.GetLane(l)), result.GetLane(l), 3);
    }

    [Fact]
    public void Round_MatchesMathF()
    {
        VFloat x = Range(-3.4f, 0.8f, VFloat.LaneCount);
        VFloat result = VectorMath.Round(x);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(MathF.Round(x.GetLane(l)), result.GetLane(l), 3);
    }

    [Fact]
    public void Rcp_MatchesDivision()
    {
        VFloat x = Range(1f, 1f, VFloat.LaneCount);
        VFloat result = VectorMath.Rcp(x);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(1f / x.GetLane(l), result.GetLane(l), 3);
    }

    [Fact]
    public void Rsqrt_MatchesFormula()
    {
        VFloat x = Range(0.5f, 1f, VFloat.LaneCount);
        VFloat result = VectorMath.Rsqrt(x);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(1f / MathF.Sqrt(x.GetLane(l)), result.GetLane(l), 3);
    }

    [Fact]
    public void MulAdd_ComputesABPlusC()
    {
        VFloat a = Range(1f, 1f, VFloat.LaneCount);
        VFloat b = Range(2f, 1f, VFloat.LaneCount);
        VFloat c = Range(0.5f, 0.5f, VFloat.LaneCount);

        VFloat result = VFloat.MulAdd(a, b, c);

        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal((a.GetLane(l) * b.GetLane(l)) + c.GetLane(l), result.GetLane(l), 3);
    }

    [Fact]
    public void Atan_MatchesMathF()
    {
        VFloat x = Range(-8f, 2.3f, VFloat.LaneCount);
        AssertApprox(VectorMath.Atan(x), BuildExpected(MathF.Atan, x), ApproxTolerance);
    }

    [Fact]
    public void Atan2_AllQuadrants_MatchesMathF()
    {
        // One point per quadrant plus the axes.
        (float y, float x)[] cases =
        [
            (1f, 2f), (1f, -2f), (-1f, -2f), (-1f, 2f),
            (1f, 0f), (-1f, 0f), (0f, 2f), (0f, -2f),
        ];
        foreach ((float y, float x) in cases)
        {
            VFloat r = VectorMath.Atan2(new VFloat(y), new VFloat(x));
            float expected = MathF.Atan2(y, x);
            Assert.True(Math.Abs(r.GetLane(0) - expected) < 1e-4f,
                $"atan2({y}, {x}): expected {expected}, got {r.GetLane(0)}");
        }
    }

    [Fact]
    public void Atan2_Origin_ReturnsZero()
    {
        VFloat r = VectorMath.Atan2(VFloat.Zero, VFloat.Zero);
        Assert.Equal(0f, r.GetLane(0));
    }

    [Fact]
    public void Asin_Acos_MatchMathF()
    {
        VFloat x = Range(-0.95f, 0.27f, VFloat.LaneCount);
        AssertApprox(VectorMath.Asin(x), BuildExpected(MathF.Asin, x), ApproxTolerance);
        AssertApprox(VectorMath.Acos(x), BuildExpected(MathF.Acos, x), ApproxTolerance);
    }

    [Fact]
    public void Asin_Acos_Endpoints()
    {
        Assert.True(Math.Abs(VectorMath.Acos(new VFloat(1f)).GetLane(0)) < 1e-4f);
        Assert.True(Math.Abs(VectorMath.Acos(new VFloat(-1f)).GetLane(0) - MathF.PI) < 1e-4f);
        Assert.True(Math.Abs(VectorMath.Asin(new VFloat(1f)).GetLane(0) - (MathF.PI / 2)) < 1e-4f);
    }

    [Fact]
    public void Cbrt_MatchesMathF_IncludingNegatives()
    {
        VFloat x = Range(-27f, 9.5f, VFloat.LaneCount);
        AssertApprox(VectorMath.Cbrt(x), BuildExpected(MathF.Cbrt, x), ApproxTolerance);
        Assert.Equal(0f, VectorMath.Cbrt(VFloat.Zero).GetLane(0));
    }

    [Fact]
    public void Hypot_MatchesReference()
    {
        VFloat x = Range(-3f, 1.7f, VFloat.LaneCount);
        VFloat y = Range(4f, -0.9f, VFloat.LaneCount);
        VFloat r = VectorMath.Hypot(x, y);
        for (int l = 0; l < VFloat.LaneCount; l++)
        {
            float expected = MathF.Sqrt((x.GetLane(l) * x.GetLane(l)) + (y.GetLane(l) * y.GetLane(l)));
            Assert.True(Math.Abs(r.GetLane(l) - expected) <= 1e-4f * Math.Max(1f, expected),
                $"Lane {l}: expected {expected}, got {r.GetLane(l)}");
        }

        Assert.Equal(0f, VectorMath.Hypot(VFloat.Zero, VFloat.Zero).GetLane(0));
    }
    private static VFloat BuildExpected(Func<float, float> fn, VFloat input)
    {
        int w = VFloat.LaneCount;
        float[] data = new float[w];
        for (int l = 0; l < w; l++)
            data[l] = fn(input.GetLane(l));
        return VFloat.Load(data, 0);
    }
}
