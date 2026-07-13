using System;
using IspcSharp;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for SoA layout helpers (SoaFloat2, SoaFloat3) and
/// double-precision lanes (VDouble, VMaskD).
/// </summary>
public class SoaAndDoubleTests
{
    [Fact]
    public void SoaFloat2_LoadGang_RoundTrips()
    {
        int w = VFloat.LaneCount;
        int n = w * 2;
        var soa = new SoaFloat2(n);
        for (int i = 0; i < n; i++) { soa.X[i] = i; soa.Y[i] = i * 10; }

        var (x, y) = soa.LoadGang(0);
        soa.StoreGang(w, x, y);

        for (int i = 0; i < w; i++)
        {
            Assert.Equal(i, soa.X[w + i]);
            Assert.Equal(i * 10, soa.Y[w + i]);
        }
    }

    [Fact]
    public void SoaFloat2_StoreGangMasked_OnlyWritesActive()
    {
        int w = VFloat.LaneCount;
        int n = w;
        var soa = new SoaFloat2(n);
        for (int i = 0; i < n; i++) { soa.X[i] = -1; soa.Y[i] = -1; }

        var x = new VFloat(42f);
        var y = new VFloat(99f);
        var mask = VMask.FirstN(2);
        soa.StoreGangMasked(0, x, y, mask);

        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(42f, soa.X[i]);
            Assert.Equal(99f, soa.Y[i]);
        }
        for (int i = 2; i < w; i++)
        {
            Assert.Equal(-1, soa.X[i]);
            Assert.Equal(-1, soa.Y[i]);
        }
    }

    [Fact]
    public void SoaFloat3_LoadGang_RoundTrips()
    {
        int w = VFloat.LaneCount;
        int n = w;
        var soa = new SoaFloat3(n);
        for (int i = 0; i < n; i++) { soa.X[i] = i; soa.Y[i] = i * 2; soa.Z[i] = i * 3; }

        var (x, y, z) = soa.LoadGang(0);

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(l, x.GetLane(l));
            Assert.Equal(l * 2, y.GetLane(l));
            Assert.Equal(l * 3, z.GetLane(l));
        }
    }

    [Fact]
    public void SoaFloat3_Dot_ProductPerLane()
    {
        int w = VFloat.LaneCount;
        var ax = new float[w]; var ay = new float[w]; var az = new float[w];
        var bx = new float[w]; var by = new float[w]; var bz = new float[w];
        for (int i = 0; i < w; i++)
        {
            ax[i] = i; ay[i] = i + 1; az[i] = i + 2;
            bx[i] = 2; by[i] = 3; bz[i] = 4;
        }

        var a = (VFloat.Load(ax, 0), VFloat.Load(ay, 0), VFloat.Load(az, 0));
        var b = (VFloat.Load(bx, 0), VFloat.Load(by, 0), VFloat.Load(bz, 0));
        var dot = SoaFloat3.Dot(a, b);

        for (int l = 0; l < w; l++)
            Assert.Equal(l * 2 + (l + 1) * 3 + (l + 2) * 4, dot.GetLane(l));
    }

    [Fact]
    public void VDouble_LoadStore_RoundTrips()
    {
        int w = VDouble.LaneCount;
        var data = new double[w];
        var result = new double[w];
        for (int i = 0; i < w; i++) data[i] = i * 1.5;

        VDouble.Load(data, 0).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void VDouble_Arithmetic_BasicOps()
    {
        int w = VDouble.LaneCount;
        var a = new double[w];
        var b = new double[w];
        var result = new double[w];
        for (int i = 0; i < w; i++) { a[i] = i + 1; b[i] = i * 2; }

        var va = VDouble.Load(a, 0);
        var vb = VDouble.Load(b, 0);
        (va + vb).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal((i + 1) + i * 2, result[i]);
    }

    [Fact]
    public void VDouble_Sqrt_MatchesMath()
    {
        int w = VDouble.LaneCount;
        var data = new double[w];
        var result = new double[w];
        for (int i = 0; i < w; i++) data[i] = (i + 1) * 2.5;

        var v = VDouble.Load(data, 0);
        VDouble.Sqrt(v).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(Math.Sqrt(data[i]), result[i], 10);
    }

    [Fact]
    public void VDouble_AbsMinMax_MatchMath()
    {
        int w = VDouble.LaneCount;
        var a = new double[w];
        var b = new double[w];
        for (int i = 0; i < w; i++) { a[i] = -(i + 1); b[i] = i + 1; }

        var va = VDouble.Load(a, 0);
        var vb = VDouble.Load(b, 0);

        var abs = VDouble.Abs(va);
        var min = VDouble.Min(va, vb);
        var max = VDouble.Max(va, vb);

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(Math.Abs(a[l]), abs.GetLane(l), 10);
            Assert.Equal(Math.Min(a[l], b[l]), min.GetLane(l), 10);
            Assert.Equal(Math.Max(a[l], b[l]), max.GetLane(l), 10);
        }
    }

    [Fact]
    public void VDouble_ReduceAdd_SumsAllLanes()
    {
        int w = VDouble.LaneCount;
        var data = new double[w];
        for (int i = 0; i < w; i++) data[i] = i + 1;

        var v = VDouble.Load(data, 0);
        double result = VDouble.ReduceAdd(v);

        double expected = 0;
        for (int i = 0; i < w; i++) expected += i + 1;
        Assert.Equal(expected, result, 6);
    }

    [Fact]
    public void VDouble_ReduceAdd_Masked_SumsOnlyActive()
    {
        int w = VDouble.LaneCount;
        var data = new double[w];
        for (int i = 0; i < w; i++) data[i] = i + 1;

        var v = VDouble.Load(data, 0);
        // Lane-count-aware: double gangs can be as narrow as 2 lanes (no SIMD acceleration).
        int active = Math.Min(3, w);
        var mask = VMaskD.FirstN(active);
        double result = VDouble.ReduceAdd(v, mask);

        double expected = 0;
        for (int i = 0; i < active; i++) expected += i + 1;
        Assert.Equal(expected, result, 6);
    }

    [Fact]
    public void VDouble_LoadMasked_HandlesTail()
    {
        int w = VDouble.LaneCount;
        var data = new double[w];
        for (int i = 0; i < w; i++) data[i] = i + 1;

        var mask = VMaskD.FirstN(2);
        var result = VDouble.LoadMasked(data, 0, mask, fallback: -99.0);

        Assert.Equal(1.0, result.GetLane(0));
        Assert.Equal(2.0, result.GetLane(1));
        for (int l = 2; l < w; l++)
            Assert.Equal(-99.0, result.GetLane(l));
    }

    [Fact]
    public void VDouble_StoreMasked_OnlyWritesActiveLanes()
    {
        int w = VDouble.LaneCount;
        var result = new double[w];
        for (int i = 0; i < w; i++) result[i] = -1;

        var v = new VDouble(42.0);
        int active = Math.Min(3, w);
        var mask = VMaskD.FirstN(active);
        v.StoreMasked(result, 0, mask);

        for (int i = 0; i < active; i++)
            Assert.Equal(42.0, result[i]);
        for (int i = active; i < w; i++)
            Assert.Equal(-1.0, result[i]);
    }

    [Fact]
    public void VDouble_Select_PicksPerLane()
    {
        int w = VDouble.LaneCount;
        var a = new VDouble(100.0);
        var b = new VDouble(200.0);
        var mask = VMaskD.FirstN(w / 2);

        var result = VDouble.Select(mask, a, b);

        for (int l = 0; l < w; l++)
            Assert.Equal(l < w / 2 ? 100.0 : 200.0, result.GetLane(l));
    }

    [Fact]
    public void VDouble_Comparisons_ProduceMasks()
    {
        int w = VDouble.LaneCount;
        var a = new double[w];
        var b = new double[w];
        for (int i = 0; i < w; i++) { a[i] = i; b[i] = w - i; }

        var va = VDouble.Load(a, 0);
        var vb = VDouble.Load(b, 0);
        var lt = va < vb;

        for (int l = 0; l < w; l++)
            Assert.Equal(l < w - l, lt.IsLaneActive(l));
    }

    [Fact]
    public void VMaskD_All_HasEveryLaneActive()
    {
        var m = VMaskD.All;
        int w = VMaskD.LaneCount;

        for (int l = 0; l < w; l++)
            Assert.True(m.IsLaneActive(l));
        Assert.True(m.Any());
    }

    [Fact]
    public void VMaskD_None_HasNoLanesActive()
    {
        var m = VMaskD.None;
        Assert.True(m.NoneActive());
        Assert.False(m.Any());
    }

    [Fact]
    public void VMaskD_FirstN_ActivatesFirstNLanes()
    {
        int w = VMaskD.LaneCount;
        int n = w / 2;
        var m = VMaskD.FirstN(n);

        for (int l = 0; l < w; l++)
            Assert.Equal(l < n, m.IsLaneActive(l));
    }

    [Fact]
    public void VMaskD_AndOrNot()
    {
        int w = VMaskD.LaneCount;
        var a = VMaskD.All;
        var b = VMaskD.FirstN(w / 2);

        var and = a & b;
        var not = !b;

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(l < w / 2, and.IsLaneActive(l));
            Assert.Equal(l >= w / 2, not.IsLaneActive(l));
        }
    }
}
