using System;
using IspcSharp;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for the core varying types: VFloat, VInt, VMask.
/// These exercise load/store, arithmetic, comparisons, select/blend,
/// and lane conversions, the primitives every kernel builds on.
/// </summary>
public class VaryingTypesTests
{
    private static float[] MakeData(int n, Func<int, float> gen)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = gen(i);
        return a;
    }

    [Fact]
    public void VFloat_LoadStore_RoundTrips()
    {
        int n = VFloat.LaneCount;
        var data = MakeData(n, i => i * 2.5f);
        var result = new float[n];

        VFloat.Load(data, 0).Store(result, 0);

        for (int i = 0; i < n; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void VFloat_LoadMasked_HandlesTail()
    {
        int w = VFloat.LaneCount;
        var data = MakeData(w, i => i + 1);
        var result = new float[w];

        // Only first 3 lanes active; rest should be fallback.
        var mask = VMask.FirstN(3);
        VFloat.LoadMasked(data, 0, mask, fallback: -99f).Store(result, 0);

        Assert.Equal(1f, result[0]);
        Assert.Equal(2f, result[1]);
        Assert.Equal(3f, result[2]);
        for (int i = 3; i < w; i++)
            Assert.Equal(-99f, result[i]);
    }

    [Fact]
    public void VFloat_StoreMasked_OnlyWritesActiveLanes()
    {
        int w = VFloat.LaneCount;
        var data = MakeData(w, i => (i + 1) * 10f);
        var result = MakeData(w, _ => -1f); // pre-filled with -1

        var mask = VMask.FirstN(2);
        VFloat.Load(data, 0).StoreMasked(result, 0, mask);

        Assert.Equal(10f, result[0]);
        Assert.Equal(20f, result[1]);
        for (int i = 2; i < w; i++)
            Assert.Equal(-1f, result[i]); // untouched
    }

    [Fact]
    public void VFloat_Arithmetic_BasicOps()
    {
        int w = VFloat.LaneCount;
        var a = MakeData(w, i => i + 1);
        var b = MakeData(w, i => (i + 1) * 2);
        var result = new float[w];

        var va = VFloat.Load(a, 0);
        var vb = VFloat.Load(b, 0);
        (va + vb).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal((i + 1) + (i + 1) * 2, result[i]);
    }

    [Fact]
    public void VFloat_SubtractMultiplyDivide()
    {
        int w = VFloat.LaneCount;
        var a = MakeData(w, i => (i + 2) * 3);
        var b = MakeData(w, i => i + 2);
        var result = new float[w];

        var va = VFloat.Load(a, 0);
        var vb = VFloat.Load(b, 0);
        (va / vb).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(3f, result[i]);
    }

    [Fact]
    public void VFloat_UnaryNegation()
    {
        int w = VFloat.LaneCount;
        var data = MakeData(w, i => i + 1);
        var result = new float[w];

        (-VFloat.Load(data, 0)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(-(i + 1), result[i]);
    }

    [Fact]
    public void VFloat_ImplicitBroadcastFromScalar()
    {
        int w = VFloat.LaneCount;
        var result = new float[w];

        VFloat v = 42f; // implicit broadcast
        v.Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(42f, result[i]);
    }

    [Fact]
    public void VFloat_MixedUniformVaryingArithmetic()
    {
        int w = VFloat.LaneCount;
        var data = MakeData(w, i => i + 1);
        var result = new float[w];

        var v = VFloat.Load(data, 0);
        (v * 2f + 1f).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal((i + 1) * 2f + 1f, result[i]);
    }

    [Fact]
    public void VFloat_Comparisons_ProduceMasks()
    {
        int w = VFloat.LaneCount;
        var a = MakeData(w, i => i);
        var b = MakeData(w, i => w - i);

        var va = VFloat.Load(a, 0);
        var vb = VFloat.Load(b, 0);
        var lt = va < vb;

        for (int l = 0; l < w; l++)
        {
            bool expected = l < w - l;
            Assert.Equal(expected, lt.IsLaneActive(l));
        }
    }

    [Fact]
    public void VFloat_Select_PicksPerLane()
    {
        int w = VFloat.LaneCount;
        var a = MakeData(w, i => 100f + i);
        var b = MakeData(w, i => 200f + i);
        var result = new float[w];

        var va = VFloat.Load(a, 0);
        var vb = VFloat.Load(b, 0);
        var mask = VMask.FirstN(w / 2);
        VFloat.Select(mask, va, vb).Store(result, 0);

        for (int i = 0; i < w; i++)
        {
            float expected = i < w / 2 ? 100f + i : 200f + i;
            Assert.Equal(expected, result[i]);
        }
    }

    [Fact]
    public void VFloat_Blend_MasksAssignment()
    {
        int w = VFloat.LaneCount;
        var original = MakeData(w, i => i + 1);
        var updated = MakeData(w, i => 999f);

        var v = VFloat.Load(original, 0);
        var u = VFloat.Load(updated, 0);
        var mask = VMask.FirstN(3);
        var blended = v.Blend(mask, u);

        for (int l = 0; l < w; l++)
        {
            float expected = l < 3 ? 999f : l + 1;
            Assert.Equal(expected, blended.GetLane(l));
        }
    }

    [Fact]
    public void VFloat_EqNeq()
    {
        int w = VFloat.LaneCount;
        var a = MakeData(w, i => i);
        var b = MakeData(w, i => i % 2);

        var va = VFloat.Load(a, 0);
        var vb = VFloat.Load(b, 0);
        var eq = VFloat.Eq(va, vb);
        var neq = VFloat.Neq(va, vb);

        // a[l] = l, b[l] = l % 2. Equal when l == l % 2, i.e. l < 2.
        for (int l = 0; l < w; l++)
        {
            Assert.Equal(l < 2, eq.IsLaneActive(l));
            Assert.Equal(l >= 2, neq.IsLaneActive(l));
        }
    }

    [Fact]
    public void VFloat_GetLane()
    {
        int w = VFloat.LaneCount;
        var data = MakeData(w, i => (i + 1) * 7);
        var v = VFloat.Load(data, 0);

        for (int l = 0; l < w; l++)
            Assert.Equal((l + 1) * 7, v.GetLane(l));
    }

    [Fact]
    public void VInt_LoadStore_RoundTrips()
    {
        int w = VInt.LaneCount;
        var data = new int[w];
        for (int i = 0; i < w; i++) data[i] = i * 3;
        var result = new int[w];

        VInt.Load(data, 0).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void VInt_Arithmetic()
    {
        int w = VInt.LaneCount;
        var a = new int[w];
        var b = new int[w];
        for (int i = 0; i < w; i++) { a[i] = i + 1; b[i] = i * 2; }
        var result = new int[w];

        var va = VInt.Load(a, 0);
        var vb = VInt.Load(b, 0);
        (va + vb).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal((i + 1) + i * 2, result[i]);
    }

    [Fact]
    public void VInt_ProgramIndex_IsZeroThroughWidth()
    {
        var idx = VInt.ProgramIndex;
        int w = VInt.LaneCount;

        for (int l = 0; l < w; l++)
            Assert.Equal(l, idx.GetLane(l));
    }

    [Fact]
    public void VInt_ToFloat_ConvertsPerLane()
    {
        int w = VInt.LaneCount;
        var data = new int[w];
        for (int i = 0; i < w; i++) data[i] = i * 5;
        var result = new float[w];

        VInt.Load(data, 0).ToFloat().Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(i * 5f, result[i]);
    }

    [Fact]
    public void VInt_FromFloatTruncate_TruncatesPerLane()
    {
        int w = VInt.LaneCount;
        var data = MakeData(w, i => i + 0.7f);
        var result = new int[w];

        VInt.FromFloatTruncate(VFloat.Load(data, 0)).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(i, result[i]); // truncation, not rounding
    }

    [Fact]
    public void VInt_Select_Blend()
    {
        int w = VInt.LaneCount;
        var a = new int[w];
        var b = new int[w];
        for (int i = 0; i < w; i++) { a[i] = 10; b[i] = 20; }
        var result = new int[w];

        var va = VInt.Load(a, 0);
        var vb = VInt.Load(b, 0);
        var mask = VMask.FirstN(w / 2);
        VInt.Select(mask, va, vb).Store(result, 0);

        for (int i = 0; i < w; i++)
            Assert.Equal(i < w / 2 ? 10 : 20, result[i]);
    }

    [Fact]
    public void VMask_All_HasEveryLaneActive()
    {
        var m = VMask.All;
        int w = VMask.LaneCount;

        for (int l = 0; l < w; l++)
            Assert.True(m.IsLaneActive(l));
        Assert.True(m.AllActive());
        Assert.True(m.Any());
        Assert.False(m.NoneActive());
    }

    [Fact]
    public void VMask_None_HasNoLanesActive()
    {
        var m = VMask.None;
        int w = VMask.LaneCount;

        for (int l = 0; l < w; l++)
            Assert.False(m.IsLaneActive(l));
        Assert.True(m.NoneActive());
        Assert.False(m.Any());
    }

    [Fact]
    public void VMask_FirstN_ActivatesFirstNLanes()
    {
        int w = VMask.LaneCount;
        int n = w / 2;
        var m = VMask.FirstN(n);

        for (int l = 0; l < w; l++)
            Assert.Equal(l < n, m.IsLaneActive(l));
        Assert.Equal(n, m.CountActive());
    }

    [Fact]
    public void VMask_AndOrNot()
    {
        int w = VMask.LaneCount;
        var a = VMask.FirstN(w);
        var b = VMask.FirstN(w / 2);

        var and = a & b;
        var or = a | b;
        var not = !b;

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(l < w / 2, and.IsLaneActive(l));
            Assert.True(or.IsLaneActive(l)); // a is all-active
            Assert.Equal(l >= w / 2, not.IsLaneActive(l));
        }
    }

    [Fact]
    public void VMask_AndNot()
    {
        int w = VMask.LaneCount;
        var a = VMask.All;
        var b = VMask.FirstN(2);

        var result = VMask.AndNot(a, b);

        for (int l = 0; l < w; l++)
            Assert.Equal(l >= 2, result.IsLaneActive(l));
    }
}
