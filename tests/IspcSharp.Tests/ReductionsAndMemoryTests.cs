using System;
using IspcSharp;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for horizontal reductions (Reduce.Add/Min/Max) and
/// indexed memory access (Memory.Gather/Scatter).
/// </summary>
public class ReductionsAndMemoryTests
{
    [Fact]
    public void Reduce_Add_VFloat_SumsAllLanes()
    {
        int w = VFloat.LaneCount;
        var data = new float[w];
        for (int i = 0; i < w; i++) data[i] = i + 1;

        var v = VFloat.Load(data, 0);
        float result = Reduce.Add(v);

        float expected = 0;
        for (int i = 0; i < w; i++) expected += i + 1;
        Assert.Equal(expected, result, 3);
    }

    [Fact]
    public void Reduce_Add_VFloat_Masked_SumsOnlyActiveLanes()
    {
        int w = VFloat.LaneCount;
        var data = new float[w];
        for (int i = 0; i < w; i++) data[i] = i + 1;

        var v = VFloat.Load(data, 0);
        var mask = VMask.FirstN(3);
        float result = Reduce.Add(v, mask);

        // Only first 3 lanes: 1 + 2 + 3 = 6
        Assert.Equal(6f, result, 3);
    }

    [Fact]
    public void Reduce_Add_VInt_SumsAllLanes()
    {
        int w = VInt.LaneCount;
        var data = new int[w];
        for (int i = 0; i < w; i++) data[i] = i + 1;

        var v = VInt.Load(data, 0);
        int result = Reduce.Add(v);

        int expected = 0;
        for (int i = 0; i < w; i++) expected += i + 1;
        Assert.Equal(expected, result);
    }

    [Fact]
    public void Reduce_Min_VFloat_FindsMinimum()
    {
        int w = VFloat.LaneCount;
        var data = new float[w];
        var rng = new Random(42);
        for (int i = 0; i < w; i++) data[i] = (float)rng.NextDouble() * 100;

        var v = VFloat.Load(data, 0);
        float result = Reduce.Min(v);

        float expected = float.MaxValue;
        for (int i = 0; i < w; i++) expected = Math.Min(expected, data[i]);
        Assert.Equal(expected, result, 3);
    }

    [Fact]
    public void Reduce_Max_VFloat_FindsMaximum()
    {
        int w = VFloat.LaneCount;
        var data = new float[w];
        var rng = new Random(42);
        for (int i = 0; i < w; i++) data[i] = (float)rng.NextDouble() * 100;

        var v = VFloat.Load(data, 0);
        float result = Reduce.Max(v);

        float expected = float.MinValue;
        for (int i = 0; i < w; i++) expected = Math.Max(expected, data[i]);
        Assert.Equal(expected, result, 3);
    }

    [Fact]
    public void Reduce_Min_Masked_IgnoresInactiveLanes()
    {
        int w = VFloat.LaneCount;
        var data = new float[w];
        for (int i = 0; i < w; i++) data[i] = i + 1; // 1, 2, 3, ...

        var v = VFloat.Load(data, 0);
        // Only lanes 2..w-1 active; lane 0 (value 1) is inactive.
        var mask = VMask.FirstN(w) & !VMask.FirstN(2);
        float result = Reduce.Min(v, mask);

        // Minimum of active lanes: 3 (lane 2)
        Assert.Equal(3f, result, 3);
    }

    [Fact]
    public void Reduce_Max_Masked_IgnoresInactiveLanes()
    {
        int w = VFloat.LaneCount;
        var data = new float[w];
        for (int i = 0; i < w; i++) data[i] = i + 1;

        var v = VFloat.Load(data, 0);
        // Only first 3 lanes active.
        var mask = VMask.FirstN(3);
        float result = Reduce.Max(v, mask);

        Assert.Equal(3f, result, 3);
    }

    [Fact]
    public void Gather_Float_PicksIndexedElements()
    {
        int w = VFloat.LaneCount;
        var source = new float[w * 2];
        for (int i = 0; i < source.Length; i++) source[i] = i * 10;

        // Indices: 0, 2, 4, 6, ... (every other element)
        var idxData = new int[w];
        for (int i = 0; i < w; i++) idxData[i] = i * 2;
        var indices = VInt.Load(idxData, 0);

        var result = Memory.Gather(source, indices);

        for (int l = 0; l < w; l++)
            Assert.Equal(l * 2 * 10f, result.GetLane(l));
    }

    [Fact]
    public void Gather_Float_Masked_UsesFallbackForInactive()
    {
        int w = VFloat.LaneCount;
        var source = new float[w * 2];
        for (int i = 0; i < source.Length; i++) source[i] = i;

        var idxData = new int[w];
        for (int i = 0; i < w; i++) idxData[i] = i;
        var indices = VInt.Load(idxData, 0);
        var mask = VMask.FirstN(2);

        var result = Memory.Gather(source, indices, mask, fallback: -1f);

        for (int l = 0; l < w; l++)
        {
            float expected = l < 2 ? l : -1f;
            Assert.Equal(expected, result.GetLane(l));
        }
    }

    [Fact]
    public void Scatter_Float_WritesToIndexedPositions()
    {
        int w = VFloat.LaneCount;
        var dest = new float[w * 2]; // zero-initialized

        var idxData = new int[w];
        for (int i = 0; i < w; i++) idxData[i] = i * 2; // write to even positions
        var indices = VInt.Load(idxData, 0);

        var valData = new float[w];
        for (int i = 0; i < w; i++) valData[i] = (i + 1) * 100;
        var values = VFloat.Load(valData, 0);

        Memory.Scatter(dest, indices, values);

        for (int l = 0; l < w; l++)
        {
            Assert.Equal((l + 1) * 100f, dest[l * 2]);
            Assert.Equal(0f, dest[l * 2 + 1]); // odd positions untouched
        }
    }

    [Fact]
    public void Scatter_Float_Masked_OnlyWritesActiveLanes()
    {
        int w = VFloat.LaneCount;
        var dest = new float[w];

        var indices = VInt.ProgramIndex;
        var values = new VFloat(42f);
        var mask = VMask.FirstN(3);

        Memory.Scatter(dest, indices, values, mask);

        for (int l = 0; l < w; l++)
            Assert.Equal(l < 3 ? 42f : 0f, dest[l]);
    }

    [Fact]
    public void Gather_Int_PicksIndexedElements()
    {
        int w = VInt.LaneCount;
        var source = new int[w * 2];
        for (int i = 0; i < source.Length; i++) source[i] = i * 3;

        var idxData = new int[w];
        for (int i = 0; i < w; i++) idxData[i] = i * 2;
        var indices = VInt.Load(idxData, 0);

        // int Gather requires explicit mask (no convenience overload without mask).
        var result = Memory.Gather(source, indices, VMask.All);

        for (int l = 0; l < w; l++)
            Assert.Equal(l * 2 * 3, result.GetLane(l));
    }

    [Fact]
    public void Gather_Int_Masked_UsesFallbackForInactive()
    {
        int w = VInt.LaneCount;
        var source = new int[w * 2];
        for (int i = 0; i < source.Length; i++) source[i] = i * 3;

        var idxData = new int[w];
        for (int i = 0; i < w; i++) idxData[i] = i * 2;
        var indices = VInt.Load(idxData, 0);
        var mask = VMask.FirstN(2);

        var result = Memory.Gather(source, indices, mask, fallback: -1);

        for (int l = 0; l < w; l++)
            Assert.Equal(l < 2 ? l * 2 * 3 : -1, result.GetLane(l));
    }

    [Fact]
    public void Scatter_Int_WritesToIndexedPositions()
    {
        int w = VInt.LaneCount;
        var dest = new int[w];

        var indices = VInt.ProgramIndex;
        var values = new VInt(77);
        var mask = VMask.FirstN(w / 2);

        Memory.Scatter(dest, indices, values, mask);

        for (int l = 0; l < w; l++)
            Assert.Equal(l < w / 2 ? 77 : 0, dest[l]);
    }

    [Fact]
    public void GatherThenScatter_RoundTrips()
    {
        int w = VFloat.LaneCount;
        var source = new float[w];
        var dest = new float[w];
        for (int i = 0; i < w; i++) source[i] = (i + 1) * 5;

        var indices = VInt.ProgramIndex;
        var gathered = Memory.Gather(source, indices);
        Memory.Scatter(dest, indices, gathered);

        for (int i = 0; i < w; i++)
            Assert.Equal(source[i], dest[i]);
    }

    [Fact]
    public void Reduce_Min_VInt_FindsMinimum()
    {
        int w = VInt.LaneCount;
        var data = new int[w];
        var rng = new Random(7);
        for (int i = 0; i < w; i++) data[i] = rng.Next(-1000, 1000);

        int result = Reduce.Min(VInt.Load(data, 0));

        int expected = int.MaxValue;
        for (int i = 0; i < w; i++) expected = Math.Min(expected, data[i]);
        Assert.Equal(expected, result);
    }

    [Fact]
    public void Reduce_Max_VInt_FindsMaximum()
    {
        int w = VInt.LaneCount;
        var data = new int[w];
        var rng = new Random(7);
        for (int i = 0; i < w; i++) data[i] = rng.Next(-1000, 1000);

        int result = Reduce.Max(VInt.Load(data, 0));

        int expected = int.MinValue;
        for (int i = 0; i < w; i++) expected = Math.Max(expected, data[i]);
        Assert.Equal(expected, result);
    }

    [Fact]
    public void Reduce_Min_VInt_Masked_IgnoresInactiveLanes()
    {
        int w = VInt.LaneCount;
        var data = new int[w];
        for (int i = 0; i < w; i++) data[i] = i - 5; // lane 0 = -5 (global min)

        // Lane 0 inactive: minimum of remaining lanes is -4.
        var mask = VMask.FirstN(w) & !VMask.FirstN(1);
        Assert.Equal(-4, Reduce.Min(VInt.Load(data, 0), mask));
    }

    [Fact]
    public void Reduce_Max_VInt_Masked_IgnoresInactiveLanes()
    {
        int w = VInt.LaneCount;
        var data = new int[w];
        for (int i = 0; i < w; i++) data[i] = i;

        var mask = VMask.FirstN(2);
        Assert.Equal(1, Reduce.Max(VInt.Load(data, 0), mask));
    }

    [Fact]
    public void Scatter_Float_SparseMask_OnlyWritesActiveLanes()
    {
        int w = VFloat.LaneCount;
        var dest = new float[w];

        // Alternating mask (lane 0 active, lane 1 inactive, ...), exercises the
        // movemask bit-iteration path with holes rather than a solid prefix.
        Span<int> bits = stackalloc int[w];
        for (int i = 0; i < w; i++) bits[i] = (i % 2 == 0) ? -1 : 0;
        var mask = new VMask(new System.Numerics.Vector<int>(bits));

        var valData = new float[w];
        for (int i = 0; i < w; i++) valData[i] = i + 1;

        Memory.Scatter(dest, VInt.ProgramIndex, VFloat.Load(valData, 0), mask);

        for (int l = 0; l < w; l++)
            Assert.Equal(l % 2 == 0 ? l + 1 : 0f, dest[l]);
    }

    [Fact]
    public void Scatter_Float_DuplicateIndices_HigherLaneWins()
    {
        int w = VFloat.LaneCount;
        var dest = new float[4];

        // All lanes write to index 0, ISPC semantics: the highest active lane wins.
        var indices = new VInt(0);
        var valData = new float[w];
        for (int i = 0; i < w; i++) valData[i] = i + 1;

        Memory.Scatter(dest, indices, VFloat.Load(valData, 0), VMask.All);

        Assert.Equal(w, dest[0]); // last lane's value (w)
    }

    [Fact]
    public void Scatter_Int_SparseMask_OnlyWritesActiveLanes()
    {
        int w = VInt.LaneCount;
        var dest = new int[w];

        Span<int> bits = stackalloc int[w];
        for (int i = 0; i < w; i++) bits[i] = (i % 2 == 1) ? -1 : 0;
        var mask = new VMask(new System.Numerics.Vector<int>(bits));

        var valData = new int[w];
        for (int i = 0; i < w; i++) valData[i] = (i + 1) * 10;

        Memory.Scatter(dest, VInt.ProgramIndex, VInt.Load(valData, 0), mask);

        for (int l = 0; l < w; l++)
            Assert.Equal(l % 2 == 1 ? (l + 1) * 10 : 0, dest[l]);
    }

    [Fact]
    public void Scatter_Float_NoneActive_WritesNothing()
    {
        int w = VFloat.LaneCount;
        var dest = new float[w];

        Memory.Scatter(dest, VInt.ProgramIndex, new VFloat(9f), VMask.None);

        for (int l = 0; l < w; l++)
            Assert.Equal(0f, dest[l]);
    }

    [Fact]
    public void Transpose_Float_Rectangular()
    {
        int rows = 37, cols = 53;   // non-square, not a multiple of the 32 block
        var src = new float[rows, cols];
        var rng = new Random(1);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                src[i, j] = (float)rng.NextDouble();

        var dst = new float[cols, rows];
        Memory.Transpose(src, dst);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                Assert.Equal(src[i, j], dst[j, i]);
    }

    [Fact]
    public void Transposed_AllocatesAndMatches()
    {
        int rows = 20, cols = 64;
        var src = new int[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                src[i, j] = i * cols + j;

        var dstF = new float[rows, cols];   // ensure the double overload exists too
        var srcD = new double[rows, cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) srcD[i, j] = i - j;
        var dstD = new double[cols, rows];
        Memory.Transpose(srcD, dstD);

        var dstI = new int[cols, rows];
        Memory.Transpose(src, dstI);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                Assert.Equal(src[i, j], dstI[j, i]);
                Assert.Equal(srcD[i, j], dstD[j, i]);
            }
        _ = dstF;

        var ft = Memory.Transposed(new float[3, 5]);
        Assert.Equal(5, ft.GetLength(0));
        Assert.Equal(3, ft.GetLength(1));
    }

    [Fact]
    public void SoaFloat2_Interleave_RoundTrips()
    {
        var xy = new float[] { 1, 2, 3, 4, 5, 6 };
        var soa = SoaFloat2.FromInterleaved(xy);

        Assert.Equal(3, soa.Length);
        Assert.Equal(new float[] { 1, 3, 5 }, soa.X);
        Assert.Equal(new float[] { 2, 4, 6 }, soa.Y);

        var back = new float[6];
        soa.ToInterleaved(back);
        Assert.Equal(xy, back);
    }

    [Fact]
    public void SoaFloat3_Interleave_RoundTrips()
    {
        var xyz = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var soa = SoaFloat3.FromInterleaved(xyz);

        Assert.Equal(3, soa.Length);
        Assert.Equal(new float[] { 1, 4, 7 }, soa.X);
        Assert.Equal(new float[] { 2, 5, 8 }, soa.Y);
        Assert.Equal(new float[] { 3, 6, 9 }, soa.Z);

        var back = new float[9];
        soa.ToInterleaved(back);
        Assert.Equal(xyz, back);
    }
}
