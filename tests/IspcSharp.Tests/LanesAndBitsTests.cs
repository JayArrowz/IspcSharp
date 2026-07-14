using System;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for cross-lane primitives (Lanes.Shuffle/Rotate/Broadcast/ShiftLanes)
/// and bit manipulation (VInt shifts, VFloat/VInt bitcasts).
/// </summary>
public class LanesAndBitsTests
{
    private static VFloat MakeFloat()
    {
        float[] data = new float[VFloat.LaneCount];
        for (int i = 0; i < data.Length; i++)
            data[i] = (i + 1) * 10f;
        return VFloat.Load(data, 0);
    }

    private static VInt MakeInt()
    {
        int[] data = new int[VInt.LaneCount];
        for (int i = 0; i < data.Length; i++)
            data[i] = (i + 1) * 10;
        return VInt.Load(data, 0);
    }

    [Fact]
    public void Shuffle_Float_Reverse()
    {
        int w = VFloat.LaneCount;
        VFloat v = MakeFloat();
        int[] idx = new int[w];
        for (int i = 0; i < w; i++)
            idx[i] = w - 1 - i;

        VFloat r = Lanes.Shuffle(v, VInt.Load(idx, 0));

        for (int l = 0; l < w; l++)
            Assert.Equal(v.GetLane(w - 1 - l), r.GetLane(l));
    }

    [Fact]
    public void Shuffle_Int_Identity()
    {
        VInt v = MakeInt();
        VInt r = Lanes.Shuffle(v, VInt.ProgramIndex);
        for (int l = 0; l < VInt.LaneCount; l++)
            Assert.Equal(v.GetLane(l), r.GetLane(l));
    }

    [Fact]
    public void Shuffle_OutOfRangeIndex_ProducesZero()
    {
        VFloat v = MakeFloat();
        VFloat r = Lanes.Shuffle(v, new VInt(VFloat.LaneCount)); // one past the end
        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(0f, r.GetLane(l));
    }

    [Fact]
    public void Rotate_Float_ByOne()
    {
        int w = VFloat.LaneCount;
        VFloat v = MakeFloat();
        VFloat r = Lanes.Rotate(v, 1);
        for (int l = 0; l < w; l++)
            Assert.Equal(v.GetLane((l + 1) % w), r.GetLane(l));
    }

    [Fact]
    public void Rotate_NegativeOffset_Wraps()
    {
        int w = VInt.LaneCount;
        VInt v = MakeInt();
        VInt r = Lanes.Rotate(v, -1);
        for (int l = 0; l < w; l++)
            Assert.Equal(v.GetLane((((l - 1) % w) + w) % w), r.GetLane(l));
    }

    [Fact]
    public void Broadcast_TakesNamedLane()
    {
        VFloat v = MakeFloat();
        VFloat r = Lanes.Broadcast(v, 2);
        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(v.GetLane(2), r.GetLane(l));
    }

    [Fact]
    public void ShiftLanes_FillsWithZero()
    {
        int w = VFloat.LaneCount;
        VFloat v = MakeFloat();
        VFloat r = Lanes.ShiftLanes(v, 1);   // r[l] = v[l+1]; last lane -> 0
        for (int l = 0; l < w - 1; l++)
            Assert.Equal(v.GetLane(l + 1), r.GetLane(l));
        Assert.Equal(0f, r.GetLane(w - 1));
    }

    [Fact]
    public void ShiftLeft_MultipliesByPowerOfTwo()
    {
        VInt v = MakeInt();
        VInt r = v << 3;
        for (int l = 0; l < VInt.LaneCount; l++)
            Assert.Equal(v.GetLane(l) << 3, r.GetLane(l));
    }

    [Fact]
    public void ShiftRightArithmetic_PreservesSign()
    {
        int[] data = new int[VInt.LaneCount];
        for (int i = 0; i < data.Length; i++)
            data[i] = (i % 2 == 0) ? -(i + 1) * 16 : (i + 1) * 16;
        VInt v = VInt.Load(data, 0);
        VInt r = v >> 2;
        for (int l = 0; l < VInt.LaneCount; l++)
            Assert.Equal(data[l] >> 2, r.GetLane(l));
    }

    [Fact]
    public void ShiftRightLogical_ZeroFills()
    {
        VInt v = new VInt(-16);
        VInt r = VInt.ShiftRightLogical(v, 2);
        for (int l = 0; l < VInt.LaneCount; l++)
            Assert.Equal((int)(unchecked((uint)-16) >> 2), r.GetLane(l));
    }

    [Fact]
    public void Bitcast_RoundTrips()
    {
        VFloat v = MakeFloat();
        VFloat r = v.AsInt().AsFloat();
        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(v.GetLane(l), r.GetLane(l));
    }

    [Fact]
    public void Bitcast_FloatBitsMatchBitConverter()
    {
        VFloat v = new VFloat(1.5f);
        VInt bits = v.AsInt();
        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(BitConverter.SingleToInt32Bits(1.5f), bits.GetLane(l));
    }

    [Fact]
    public void VariableShifts_MatchScalarSemantics()
    {
        int w = VInt.LaneCount;
        Random rng = new Random(9);
        int[] xData = new int[w];
        int[] cData = new int[w];
        for (int i = 0; i < w; i++)
        {
            xData[i] = rng.Next(int.MinValue, int.MaxValue);
            cData[i] = rng.Next(0, 40);   // includes counts >= 32 (C# uses low 5 bits)
        }

        VInt x = VInt.Load(xData, 0);
        VInt c = VInt.Load(cData, 0);

        VInt left = VInt.ShiftLeftVariable(x, c);
        VInt right = VInt.ShiftRightArithmeticVariable(x, c);
        VInt logical = VInt.ShiftRightLogicalVariable(x, c);

        for (int l = 0; l < w; l++)
        {
            Assert.Equal(xData[l] << cData[l], left.GetLane(l));
            Assert.Equal(xData[l] >> cData[l], right.GetLane(l));
            Assert.Equal(xData[l] >>> cData[l], logical.GetLane(l));
        }
    }

    [Fact]
    public void VInt_WidenNarrow_RoundTrips()
    {
        VInt v = MakeInt();
        (VLong lo, VLong hi) = VInt.Widen(v);
        VInt r = VInt.Narrow(lo, hi);
        for (int l = 0; l < VInt.LaneCount; l++)
            Assert.Equal(v.GetLane(l), r.GetLane(l));
    }

    [Fact]
    public void VMask_WidenNarrow_RoundTrips()
    {
        int w = VInt.LaneCount;
        Span<int> bits = stackalloc int[w];
        for (int i = 0; i < w; i++)
            bits[i] = (i % 3 == 0) ? -1 : 0;
        VMask mask = new VMask(new System.Numerics.Vector<int>(bits));

        (VMaskD lo, VMaskD hi) = VMaskD.Widen(mask);
        VMask r = VMask.Narrow(lo, hi);
        for (int l = 0; l < w; l++)
            Assert.Equal(mask.IsLaneActive(l), r.IsLaneActive(l));
    }

    [Fact]
    public void VDouble2_FromFloat_ToFloat_RoundTripsExactly()
    {
        VFloat v = MakeFloat();
        VDouble2 d = VDouble2.FromFloat(v);
        // Widening is exact per lane.
        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal((double)v.GetLane(l), d.GetLane(l));
        VFloat r = d.ToFloat();
        for (int l = 0; l < VFloat.LaneCount; l++)
            Assert.Equal(v.GetLane(l), r.GetLane(l));
    }

    [Fact]
    public void VDouble2_Arithmetic_And_Compare_WorkAtFullWidth()
    {
        int w = VFloat.LaneCount;
        float[] data = new float[w];
        for (int i = 0; i < w; i++)
            data[i] = i - (w / 2f);
        VDouble2 d = VDouble2.FromFloat(VFloat.Load(data, 0));

        VDouble2 doubled = (d * 2.0) + 1.0;
        for (int l = 0; l < w; l++)
            Assert.Equal((data[l] * 2.0) + 1.0, doubled.GetLane(l));

        VMask mask = d > VDouble2.Zero;                       // full-width VMask
        VDouble2 sel = VDouble2.Select(mask, doubled, VDouble2.Zero);
        for (int l = 0; l < w; l++)
            Assert.Equal(data[l] > 0 ? (data[l] * 2.0) + 1.0 : 0.0, sel.GetLane(l));
    }

    [Fact]
    public void VDouble2_ReduceAdd_SumsAllLanes()
    {
        int w = VFloat.LaneCount;
        float[] data = new float[w];
        for (int i = 0; i < w; i++)
            data[i] = i + 1;
        VDouble2 d = VDouble2.FromFloat(VFloat.Load(data, 0));

        double expected = 0;
        for (int i = 0; i < w; i++)
            expected += i + 1;
        Assert.Equal(expected, Reduce.Add(d));
    }

    [Fact]
    public void VLong2_FromInt_ToInt_RoundTripsExactly()
    {
        VInt v = MakeInt();
        VLong2 d = VLong2.FromInt(v);
        // Widening int -> long is exact per lane.
        for (int l = 0; l < VInt.LaneCount; l++)
            Assert.Equal(v.GetLane(l), d.GetLane(l));
        VInt r = d.ToInt();
        for (int l = 0; l < VInt.LaneCount; l++)
            Assert.Equal(v.GetLane(l), r.GetLane(l));
    }

    [Fact]
    public void VLong2_Arithmetic_And_Compare_WorkAtFullWidth()
    {
        int w = VInt.LaneCount;
        int[] data = new int[w];
        for (int i = 0; i < w; i++)
            data[i] = i - (w / 2);
        VLong2 d = VLong2.FromInt(VInt.Load(data, 0));

        // A product that overflows 32 bits stays exact in 64.
        VLong2 scaled = (d * 1_000_000_000L) + 7L;
        for (int l = 0; l < w; l++)
            Assert.Equal((data[l] * 1_000_000_000L) + 7L, scaled.GetLane(l));

        VMask mask = d > VLong2.Zero;                        // full-width VMask
        VLong2 sel = VLong2.Select(mask, scaled, VLong2.Zero);
        for (int l = 0; l < w; l++)
            Assert.Equal(data[l] > 0 ? (data[l] * 1_000_000_000L) + 7L : 0L, sel.GetLane(l));
    }

    [Fact]
    public void VLong2_ReduceAdd_SumsAllLanesExactlyBeyond32Bits()
    {
        int w = VInt.LaneCount;
        int[] data = new int[w];
        for (int i = 0; i < w; i++)
            data[i] = 300_000_000;   // w * 3e8 overflows int for w >= 8
        VLong2 d = VLong2.FromInt(VInt.Load(data, 0));

        long expected = w * 300_000_000L;
        Assert.Equal(expected, Reduce.Add(d));
    }
}
