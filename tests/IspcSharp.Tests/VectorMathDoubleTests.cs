using System;
using IspcSharp;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for the double-precision (VDouble) transcendentals in VectorMath.
/// These are polynomial/rational approximations targeting near-double-ULP accuracy
/// (~1e-14 relative over their stated ranges), tested against System.Math.
/// </summary>
public class VectorMathDoubleTests
{
    private const double Tolerance = 1e-13;

    /// <summary>
    /// Sin/Cos accumulate range-reduction error; still far tighter than the float versions.
    /// </summary>
    private const double TrigTolerance = 1e-12;

    private static VDouble Range(double start, double step, int n)
    {
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = start + i * step;
        return VDouble.Load(data, 0);
    }

    private static void AssertApprox(VDouble actual, VDouble expected, double tol)
    {
        int w = VDouble.LaneCount;
        for (int l = 0; l < w; l++)
        {
            double a = actual.GetLane(l);
            double e = expected.GetLane(l);
            double allowed = tol * Math.Max(1d, Math.Abs(e));
            Assert.True(Math.Abs(a - e) <= allowed,
                $"Lane {l}: expected {e:G17}, got {a:G17} (diff {Math.Abs(a - e):G17}, tol {allowed:G17})");
        }
    }

    private static VDouble Map(VDouble x, Func<double, double> f)
    {
        var data = new double[VDouble.LaneCount];
        for (int l = 0; l < VDouble.LaneCount; l++) data[l] = f(x.GetLane(l));
        return VDouble.Load(data, 0);
    }

    [Fact]
    public void Sqrt_MatchesMath()
    {
        var x = Range(0.5, 1.5, VDouble.LaneCount);
        AssertApprox(VectorMath.Sqrt(x), Map(x, Math.Sqrt), Tolerance);
    }

    [Fact]
    public void MulAdd_MatchesMulPlusAdd()
    {
        var a = Range(1.1, 0.7, VDouble.LaneCount);
        var b = Range(-2.3, 0.9, VDouble.LaneCount);
        var c = Range(0.5, -0.4, VDouble.LaneCount);
        var r = VDouble.MulAdd(a, b, c);

        for (int l = 0; l < VDouble.LaneCount; l++)
        {
            double expected = Math.FusedMultiplyAdd(a.GetLane(l), b.GetLane(l), c.GetLane(l));
            double plain = a.GetLane(l) * b.GetLane(l) + c.GetLane(l);
            // Result must match either the fused or the two-rounding computation exactly.
            Assert.True(r.GetLane(l) == expected || r.GetLane(l) == plain,
                $"Lane {l}: got {r.GetLane(l):G17}, expected {expected:G17} (fused) or {plain:G17} (mul+add)");
        }
    }

    [Fact]
    public void Floor_MatchesMath()
    {
        var x = Range(-3.7, 0.9, VDouble.LaneCount);
        AssertApprox(VectorMath.Floor(x), Map(x, Math.Floor), 0d);
    }

    [Fact]
    public void Round_MatchesMidpointAwayFromZero()
    {
        var x = Range(-2.5, 1.25, VDouble.LaneCount);
        AssertApprox(VectorMath.Round(x), Map(x, v => Math.Round(v, MidpointRounding.AwayFromZero)), 0d);
    }

    [Fact]
    public void Clamp_And_Lerp_Work()
    {
        var x = Range(-2.0, 1.0, VDouble.LaneCount);
        var clamped = VectorMath.Clamp(x, new VDouble(-1d), new VDouble(1d));
        for (int l = 0; l < VDouble.LaneCount; l++)
            Assert.Equal(Math.Clamp(x.GetLane(l), -1d, 1d), clamped.GetLane(l));

        var lerped = VectorMath.Lerp(new VDouble(2d), new VDouble(4d), new VDouble(0.25));
        for (int l = 0; l < VDouble.LaneCount; l++)
            Assert.Equal(2.5, lerped.GetLane(l));
    }

    [Fact]
    public void Exp_MatchesMath_MidRange()
    {
        var x = Range(-5.0, 1.3, VDouble.LaneCount);
        AssertApprox(VectorMath.Exp(x), Map(x, Math.Exp), Tolerance);
    }

    [Fact]
    public void Exp_MatchesMath_LargeRange()
    {
        var x = Range(-700.0, 175.0, VDouble.LaneCount);
        AssertApprox(VectorMath.Exp(x), Map(x, Math.Exp), Tolerance);
    }

    [Fact]
    public void Exp_Zero_IsOne()
    {
        var r = VectorMath.Exp(VDouble.Zero);
        for (int l = 0; l < VDouble.LaneCount; l++)
            Assert.Equal(1d, r.GetLane(l));
    }

    [Fact]
    public void Log_MatchesMath()
    {
        var x = Range(0.1, 2.7, VDouble.LaneCount);
        AssertApprox(VectorMath.Log(x), Map(x, Math.Log), Tolerance);
    }

    [Fact]
    public void Log_MatchesMath_WideRange()
    {
        var x = Range(1e-8, 1e8, VDouble.LaneCount);
        AssertApprox(VectorMath.Log(x), Map(x, Math.Log), Tolerance);
    }

    [Fact]
    public void Log_Exp_Roundtrip()
    {
        var x = Range(0.5, 0.8, VDouble.LaneCount);
        AssertApprox(VectorMath.Log(VectorMath.Exp(x)), x, Tolerance);
    }

    [Fact]
    public void Pow_MatchesMath()
    {
        var x = Range(0.5, 1.1, VDouble.LaneCount);
        var y = new VDouble(2.5);
        AssertApprox(VectorMath.Pow(x, y), Map(x, v => Math.Pow(v, 2.5)), Tolerance);
    }

    [Fact]
    public void Sin_MatchesMath_SmallRange()
    {
        var x = Range(-3.0, 0.9, VDouble.LaneCount);
        AssertApprox(VectorMath.Sin(x), Map(x, Math.Sin), TrigTolerance);
    }

    [Fact]
    public void Sin_MatchesMath_ModerateRange()
    {
        var x = Range(-50.0, 17.7, VDouble.LaneCount);
        AssertApprox(VectorMath.Sin(x), Map(x, Math.Sin), TrigTolerance);
    }

    [Fact]
    public void Cos_MatchesMath()
    {
        var x = Range(-3.0, 1.1, VDouble.LaneCount);
        AssertApprox(VectorMath.Cos(x), Map(x, Math.Cos), TrigTolerance);
    }

    [Fact]
    public void Tan_MatchesMath()
    {
        // Stay away from odd multiples of pi/2 where tan blows up.
        var x = Range(-1.2, 0.4, VDouble.LaneCount);
        AssertApprox(VectorMath.Tan(x), Map(x, Math.Tan), TrigTolerance);
    }

    [Fact]
    public void SinCos_PythagoreanIdentity()
    {
        var x = Range(-10.0, 3.3, VDouble.LaneCount);
        var s = VectorMath.Sin(x);
        var c = VectorMath.Cos(x);
        var one = s * s + c * c;
        for (int l = 0; l < VDouble.LaneCount; l++)
            Assert.True(Math.Abs(one.GetLane(l) - 1d) < 1e-12,
                $"Lane {l}: sin²+cos² = {one.GetLane(l):G17}");
    }

    [Fact]
    public void Tanh_MatchesMath()
    {
        var x = Range(-4.0, 1.3, VDouble.LaneCount);
        AssertApprox(VectorMath.Tanh(x), Map(x, Math.Tanh), Tolerance);
    }

    [Fact]
    public void Tanh_Saturates()
    {
        var r = VectorMath.Tanh(new VDouble(50d));
        for (int l = 0; l < VDouble.LaneCount; l++)
            Assert.Equal(1d, r.GetLane(l), 12);
    }

    [Fact]
    public void Sigmoid_MatchesReference()
    {
        var x = Range(-6.0, 1.7, VDouble.LaneCount);
        AssertApprox(VectorMath.Sigmoid(x), Map(x, v => 1d / (1d + Math.Exp(-v))), Tolerance);
    }

    [Fact]
    public void Rcp_Rsqrt_MatchReference()
    {
        var x = Range(0.5, 1.4, VDouble.LaneCount);
        AssertApprox(VectorMath.Rcp(x), Map(x, v => 1d / v), Tolerance);
        AssertApprox(VectorMath.Rsqrt(x), Map(x, v => 1d / Math.Sqrt(v)), Tolerance);
    }

    [Fact]
    public void Atan_MatchesMath()
    {
        var x = Range(-8.0, 2.3, VDouble.LaneCount);
        AssertApprox(VectorMath.Atan(x), Map(x, Math.Atan), Tolerance);
    }

    [Fact]
    public void Atan2_AllQuadrants_MatchesMath()
    {
        (double y, double x)[] cases =
        {
            (1, 2), (1, -2), (-1, -2), (-1, 2),
            (1, 0), (-1, 0), (0, 2), (0, -2),
        };
        foreach (var (y, x) in cases)
        {
            double r = VectorMath.Atan2(new VDouble(y), new VDouble(x)).GetLane(0);
            double expected = Math.Atan2(y, x);
            Assert.True(Math.Abs(r - expected) < 1e-13,
                $"atan2({y}, {x}): expected {expected:G17}, got {r:G17}");
        }
        Assert.Equal(0d, VectorMath.Atan2(VDouble.Zero, VDouble.Zero).GetLane(0));
    }

    [Fact]
    public void Asin_Acos_MatchMath()
    {
        var x = Range(-0.95, 0.27, VDouble.LaneCount);
        AssertApprox(VectorMath.Asin(x), Map(x, Math.Asin), 1e-12);
        AssertApprox(VectorMath.Acos(x), Map(x, Math.Acos), 1e-12);
    }

    [Fact]
    public void Cbrt_MatchesMath_IncludingNegatives()
    {
        var x = Range(-27.0, 9.5, VDouble.LaneCount);
        AssertApprox(VectorMath.Cbrt(x), Map(x, Math.Cbrt), Tolerance);
        Assert.Equal(0d, VectorMath.Cbrt(VDouble.Zero).GetLane(0));
    }

    [Fact]
    public void Hypot_MatchesReference()
    {
        var x = Range(-3.0, 1.7, VDouble.LaneCount);
        var y = Range(4.0, -0.9, VDouble.LaneCount);
        var r = VectorMath.Hypot(x, y);
        for (int l = 0; l < VDouble.LaneCount; l++)
        {
            double expected = Math.Sqrt(x.GetLane(l) * x.GetLane(l) + y.GetLane(l) * y.GetLane(l));
            Assert.True(Math.Abs(r.GetLane(l) - expected) <= 1e-13 * Math.Max(1d, expected),
                $"Lane {l}: expected {expected:G17}, got {r.GetLane(l):G17}");
        }
        Assert.Equal(0d, VectorMath.Hypot(VDouble.Zero, VDouble.Zero).GetLane(0));
    }
}
