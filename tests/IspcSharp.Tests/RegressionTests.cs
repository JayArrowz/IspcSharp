using System;
using System.Linq;
using Microsoft.CodeAnalysis;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Kernels that each reproduce one bug where the generated <c>_Simd</c> method disagreed with
/// the scalar method it was generated from. The scalar method is the reference in every test.
/// </summary>
public static partial class RegressionKernels
{
    /// <summary>A lane-local loop counter must not be incremented on the iteration that breaks.</summary>
    [Spmd(GenerateParallel = false)]
    public static void CounterAfterBreak(int[] a, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int k = 0;
            for (k = 0; k < 10; k++)
            {
                if (a[i] == k)
                    break;
            }

            o[i] = k;
        }
    }

    /// <summary>'continue' still runs the incrementor; only 'break' skips it.</summary>
    [Spmd(GenerateParallel = false)]
    public static void CounterWithContinue(int[] a, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int k = 0;
            int hits = 0;
            for (k = 0; k < 8; k++)
            {
                if (a[i] > k)
                    continue;
                hits += 1;
                if (hits == 3)
                    break;
            }

            o[i] = (k * 100) + hits;
        }
    }

    /// <summary>A 'for' initializer inside a branch must not reset lanes that skip the branch.</summary>
    [Spmd(GenerateParallel = false)]
    public static void MaskedForInitializer(int[] a, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int k = 5;
            if (a[i] > 2)
            {
                for (k = 0; k < 3; k++)
                {
                    o[i] = 0;
                }
            }

            o[i] = k;
        }
    }

    /// <summary>A lane-local incrementor in an otherwise uniform 'for' inside a branch.</summary>
    [Spmd(GenerateParallel = false)]
    public static void MaskedLaneIncrementor(int[] a, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int j = 1;
            if (a[i] > 2)
            {
                for (int k = 0; k < 3; k++, j += 2)
                {
                    o[i] = k;
                }
            }

            o[i] = j;
        }
    }

    /// <summary>Two sibling loops may reuse the loop variable name, as in C#.</summary>
    [Spmd(GenerateParallel = false)]
    public static void SiblingLoopsShareName(int[] a, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int acc = 0;
            for (int k = 0; k < 6; k++)
            {
                if (a[i] == k)
                    break;
                acc += 1;
            }

            for (int k = 0; k < 6; k++)
            {
                if (a[i] + 2 == k)
                    break;
                acc += 10;
            }

            o[i] = acc;
        }
    }

    /// <summary>An index local advanced inside a 'for' is not the same offset on every pass.</summary>
    [Spmd(GenerateParallel = false)]
    public static void IndexAdvancedInFor(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int j = i;
            float acc = 0f;
            for (int k = 0; k < 3; k++)
            {
                acc += a[j];
                j += 2;
            }

            o[i] = acc;
        }
    }

    /// <summary>The same through a divergent 'while'.</summary>
    [Spmd(GenerateParallel = false)]
    public static void IndexAdvancedInWhile(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int j = i;
            float acc = 0f;
            int k = 0;
            while (k < 3)
            {
                acc += a[j];
                j = j + 1;
                k++;
            }

            o[i] = acc;
        }
    }

    /// <summary>An index local reassigned under a uniform 'if' keeps its old offset when the branch is not taken.</summary>
    [Spmd(GenerateParallel = false)]
    public static void IndexReassignedInUniformIf(float[] a, float[] o, int n, int mode)
    {
        foreach (int i in Spmd.Range(n))
        {
            int j = i;
            if (mode == 1)
                j = i + 3;
            o[i] = a[j];
        }
    }

    /// <summary>Compound assignment on narrow locals wraps to the declared type.</summary>
    [Spmd(GenerateParallel = false)]
    public static void NarrowCompound(byte[] src, int[] asByte, int[] asSByte, int[] asShort, int[] asUShort, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            byte b = src[i];
            b += 100;
            b *= 3;
            asByte[i] = b;

            sbyte sb = (sbyte)src[i];
            sb += 100;
            asSByte[i] = sb;

            short s = (short)(src[i] * 200);
            s += 20000;
            s -= 7;
            asShort[i] = s;

            ushort us = (ushort)(src[i] * 250);
            us += 40000;
            asUShort[i] = us;
        }
    }

    /// <summary>++ / -- on narrow locals wrap too.</summary>
    [Spmd(GenerateParallel = false)]
    public static void NarrowIncrement(byte[] src, int[] up, int[] down, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            byte b = src[i];
            b++;
            up[i] = b;

            byte c = src[i];
            c--;
            --c;
            down[i] = c;
        }
    }

    /// <summary>Only the selected arm of ?: may divide.</summary>
    [Spmd(GenerateParallel = false)]
    public static void TernaryGuardsDivide(int[] num, int[] den, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = den[i] != 0 ? num[i] / den[i] : -1;
        }
    }

    /// <summary>The right side of &amp;&amp; only runs where the left side is true.</summary>
    [Spmd(GenerateParallel = false)]
    public static void AndGuardsDivide(int[] num, int[] den, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            if (den[i] != 0 && num[i] / den[i] > 2)
                o[i] = 1;
        }
    }

    /// <summary>The right side of || only runs where the left side is false.</summary>
    [Spmd(GenerateParallel = false)]
    public static void OrGuardsRemainder(int[] num, int[] den, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            if (den[i] == 0 || num[i] % den[i] == 1)
                o[i] = 7;
        }
    }

    /// <summary>A gather under an 'if' must not read the lanes the 'if' excludes.</summary>
    [Spmd(GenerateParallel = false)]
    public static void GuardedGatherFloat(int[] idx, float[] table, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int j = idx[i];
            if (j >= 0)
                o[i] = table[j];
        }
    }

    /// <summary>The same for an int table, guarded by a ternary instead.</summary>
    [Spmd(GenerateParallel = false)]
    public static void GuardedGatherIntTernary(int[] idx, int[] table, int[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            int j = idx[i];
            o[i] = j >= 0 && j < 8 ? table[j] : -5;
        }
    }

    /// <summary>The same in a double kernel (64-bit index gang).</summary>
    [Spmd(GenerateParallel = false)]
    public static void GuardedGatherDouble(double[] idx, double[] table, double[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            double j = idx[i];
            if (j >= 0)
                o[i] = table[(int)j];
        }
    }

    /// <summary>An unguarded gather with a bad index must throw, as the scalar loop does.</summary>
    [Spmd(GenerateParallel = false)]
    public static void PlainGather(int[] idx, float[] table, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = table[idx[i]];
        }
    }

    /// <summary>A backwards neighbour is a contiguous load, not a gather.</summary>
    [Spmd(GenerateParallel = false)]
    public static void Stencil(float[] a, float[] o, int n, int half)
    {
        foreach (int i in Spmd.Range(4, n - 4))
        {
            o[i] = a[i - 1] + a[i] + a[i + 1] + a[i - half] + a[i - (half - 1)] + a[(i + 2) - 3];
        }
    }

    /// <summary>A uniform condition selects one branch for the whole gang.</summary>
    [Spmd(GenerateParallel = false)]
    public static void UniformBranch(float[] a, float[] o, int n, int mode)
    {
        foreach (int i in Spmd.Range(n))
        {
            float x = a[i];
            if (mode == 1)
                x = x * 2f;
            else if (mode == 2)
                x = x + 1f;
            else
                x = -x;
            o[i] = x;
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void RoundF(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Round(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void RoundAwayF(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Round(a[i], MidpointRounding.AwayFromZero);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void FloorF(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Floor(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void ExpF(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Exp(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void LogF(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Log(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void PowF(float[] a, float[] b, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Pow(a[i], b[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void SinF(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Sin(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void CosF(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = MathF.Cos(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void RoundD(double[] a, double[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = Math.Round(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void FloorD(double[] a, double[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = Math.Floor(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void ExpD(double[] a, double[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = Math.Exp(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void LogD(double[] a, double[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = Math.Log(a[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void PowD(double[] a, double[] b, double[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = Math.Pow(a[i], b[i]);
        }
    }

    [Spmd(GenerateParallel = false)]
    public static void IntDivRem(int[] a, int[] b, int[] q, int[] r, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            q[i] = a[i] / b[i];
            r[i] = a[i] % b[i];
        }
    }

    [Spmd(GenerateParallel = false, Streaming = true)]
    public static float StreamingScale(float[] a, float[] o, float s, int n)
    {
        float sum = 0f;
        foreach (int i in Spmd.Range(n))
        {
            o[i] = a[i] * s;
            sum += a[i];
        }

        return sum;
    }

    /// <summary>Streaming plus a per-lane 'return': the peel loop needs its own label.</summary>
    [Spmd(Streaming = true)]
    public static void StreamingWithReturn(float[] a, float[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            if (a[i] < 0f)
                return;
            o[i] = a[i] * 3f;
        }
    }

    [Spmd(GenerateParallel = false, Streaming = true)]
    public static void StreamingDouble(double[] a, double[] o, int n)
    {
        foreach (int i in Spmd.Range(n))
        {
            o[i] = a[i] + 1d;
        }
    }
}

public class RegressionTests
{
    // A few full gangs plus a ragged tail, so both the vector loop and the scalar tail run.
    private static int N => (5 * Spmd.LaneCount) + 3;

    private static int[] Ints(int n, Func<int, int> f) => [.. Enumerable.Range(0, n).Select(f)];

    [Fact]
    public void ForLoop_BreakSkipsIncrementor()
    {
        int n = N;
        int[] a = Ints(n, i => i % 13);
        int[] expected = new int[n], actual = new int[n];
        RegressionKernels.CounterAfterBreak(a, expected, n);
        RegressionKernels.CounterAfterBreak_Simd(a, actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ForLoop_ContinueStillRunsIncrementor()
    {
        int n = N;
        int[] a = Ints(n, i => i % 9);
        int[] expected = new int[n], actual = new int[n];
        RegressionKernels.CounterWithContinue(a, expected, n);
        RegressionKernels.CounterWithContinue_Simd(a, actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ForLoop_InitializerRespectsMask()
    {
        int n = N;
        int[] a = Ints(n, i => i % 5);
        int[] expected = new int[n], actual = new int[n];
        RegressionKernels.MaskedForInitializer(a, expected, n);
        RegressionKernels.MaskedForInitializer_Simd(a, actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ForLoop_LaneIncrementorRespectsMask()
    {
        int n = N;
        int[] a = Ints(n, i => i % 5);
        int[] expected = new int[n], actual = new int[n];
        RegressionKernels.MaskedLaneIncrementor(a, expected, n);
        RegressionKernels.MaskedLaneIncrementor_Simd(a, actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ForLoop_SiblingLoopsMayReuseVariableName()
    {
        int n = N;
        int[] a = Ints(n, i => i % 7);
        int[] expected = new int[n], actual = new int[n];
        RegressionKernels.SiblingLoopsShareName(a, expected, n);
        RegressionKernels.SiblingLoopsShareName_Simd(a, actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void AffineIndex_AdvancedInsideLoop_IsNotContiguous()
    {
        int n = N;
        float[] a = [.. Enumerable.Range(0, n + 8).Select(i => (float)((i * 3) + 1))];
        float[] expected = new float[n], actual = new float[n];

        RegressionKernels.IndexAdvancedInFor(a, expected, n);
        RegressionKernels.IndexAdvancedInFor_Simd(a, actual, n);
        Assert.Equal(expected, actual);

        Array.Clear(expected);
        Array.Clear(actual);
        RegressionKernels.IndexAdvancedInWhile(a, expected, n);
        RegressionKernels.IndexAdvancedInWhile_Simd(a, actual, n);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void AffineIndex_ReassignedUnderUniformIf(int mode)
    {
        int n = N;
        float[] a = [.. Enumerable.Range(0, n + 8).Select(i => (float)i)];
        float[] expected = new float[n], actual = new float[n];
        RegressionKernels.IndexReassignedInUniformIf(a, expected, n, mode);
        RegressionKernels.IndexReassignedInUniformIf_Simd(a, actual, n, mode);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void NarrowLocals_CompoundAssignmentWraps()
    {
        int n = Math.Max(N, 256 + 3);
        byte[] src = [.. Enumerable.Range(0, n).Select(i => (byte)i)];
        int[][] expected = [new int[n], new int[n], new int[n], new int[n]];
        int[][] actual = [new int[n], new int[n], new int[n], new int[n]];
        RegressionKernels.NarrowCompound(src, expected[0], expected[1], expected[2], expected[3], n);
        RegressionKernels.NarrowCompound_Simd(src, actual[0], actual[1], actual[2], actual[3], n);
        for (int k = 0; k < 4; k++)
            Assert.Equal(expected[k], actual[k]);
    }

    [Fact]
    public void NarrowLocals_IncrementWraps()
    {
        int n = Math.Max(N, 256 + 3);
        byte[] src = [.. Enumerable.Range(0, n).Select(i => (byte)i)];
        int[] eUp = new int[n], eDown = new int[n], aUp = new int[n], aDown = new int[n];
        RegressionKernels.NarrowIncrement(src, eUp, eDown, n);
        RegressionKernels.NarrowIncrement_Simd(src, aUp, aDown, n);
        Assert.Equal(eUp, aUp);
        Assert.Equal(eDown, aDown);
    }

    [Fact]
    public void IntegerDivide_GuardedByTernaryAndShortCircuit()
    {
        int n = N;
        int[] num = Ints(n, i => (i * 5) - 20);
        int[] den = Ints(n, i => (i % 4) - 1);   // -1, 0, 1, 2, ...

        int[] expected = new int[n], actual = new int[n];
        RegressionKernels.TernaryGuardsDivide(num, den, expected, n);
        RegressionKernels.TernaryGuardsDivide_Simd(num, den, actual, n);
        Assert.Equal(expected, actual);

        Array.Clear(expected);
        Array.Clear(actual);
        RegressionKernels.AndGuardsDivide(num, den, expected, n);
        RegressionKernels.AndGuardsDivide_Simd(num, den, actual, n);
        Assert.Equal(expected, actual);

        Array.Clear(expected);
        Array.Clear(actual);
        RegressionKernels.OrGuardsRemainder(num, den, expected, n);
        RegressionKernels.OrGuardsRemainder_Simd(num, den, actual, n);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Gather_UnderMask_DoesNotReadInactiveLanes()
    {
        int n = N;
        // Inactive lanes carry indices that are out of range in both directions.
        int[] idx = Ints(n, i => (i % 3) switch { 0 => -1, 1 => i % 8, _ => 1 << 28 });
        int[] idxNonNegative = Ints(n, i => i % 3 == 0 ? -1 : i % 8);

        float[] table = [.. Enumerable.Range(0, 8).Select(i => i + 0.5f)];
        float[] expected = new float[n], actual = new float[n];
        RegressionKernels.GuardedGatherFloat(idxNonNegative, table, expected, n);
        RegressionKernels.GuardedGatherFloat_Simd(idxNonNegative, table, actual, n);
        Assert.Equal(expected, actual);

        int[] itable = [.. Enumerable.Range(0, 8).Select(i => i * 11)];
        int[] iExpected = new int[n], iActual = new int[n];
        RegressionKernels.GuardedGatherIntTernary(idx, itable, iExpected, n);
        RegressionKernels.GuardedGatherIntTernary_Simd(idx, itable, iActual, n);
        Assert.Equal(iExpected, iActual);

        double[] didx = [.. idxNonNegative.Select(i => (double)i)];
        double[] dtable = [.. Enumerable.Range(0, 8).Select(i => i + 0.25)];
        double[] dExpected = new double[n], dActual = new double[n];
        RegressionKernels.GuardedGatherDouble(didx, dtable, dExpected, n);
        RegressionKernels.GuardedGatherDouble_Simd(didx, dtable, dActual, n);
        Assert.Equal(dExpected, dActual);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-2)]
    [InlineData(8)]
    [InlineData(int.MaxValue)]
    [InlineData(int.MinValue)]
    public void Gather_OutOfRangeIndex_ThrowsLikeTheScalarLoop(int bad)
    {
        int n = N;
        int[] idx = Ints(n, i => i % 8);
        idx[Spmd.LaneCount + 1] = bad;    // inside a full gang, not the scalar tail
        float[] table = new float[8];

        _ = Assert.Throws<IndexOutOfRangeException>(() => RegressionKernels.PlainGather(idx, table, new float[n], n));
        _ = Assert.Throws<IndexOutOfRangeException>(() => RegressionKernels.PlainGather_Simd(idx, table, new float[n], n));
    }

    [Fact]
    public void MemoryGather_BoundsChecksActiveLanesOnly()
    {
        float[] f = [1f, 2f, 3f, 4f];
        int[] i32 = [1, 2, 3, 4];
        double[] d = [1d, 2d, 3d, 4d];
        long[] i64 = [1L, 2L, 3L, 4L];

        var bad = new VInt(4);                       // one past the end, every lane
        var badL = new VLong(-1L);
        var firstOnly = VMask.FirstN(1);
        var firstOnlyD = VMaskD.FirstN(1);

        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.Gather(f, bad, VMask.All));
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.Gather(i32, bad, VMask.All));
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.Gather(d, badL, VMaskD.All));
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.Gather(i64, badL, VMaskD.All));
        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.Gather(f, bad, firstOnly));

        // A bad index in a lane the mask excludes is not an error: that lane takes the fallback.
        VInt mixed = VInt.Select(firstOnly, new VInt(2), bad);
        VFloat g = Memory.Gather(f, mixed, firstOnly, -9f);
        Assert.Equal(3f, g.GetLane(0));
        for (int l = 1; l < VFloat.LaneCount; l++)
            Assert.Equal(-9f, g.GetLane(l));

        VLong mixedL = VLong.Select(firstOnlyD, new VLong(3L), badL);
        VDouble gd = Memory.Gather(d, mixedL, firstOnlyD, -9d);
        Assert.Equal(4d, gd.GetLane(0));
        for (int l = 1; l < VDouble.LaneCount; l++)
            Assert.Equal(-9d, gd.GetLane(l));

        // Nothing active, nothing read: an empty source is fine.
        VFloat none = Memory.Gather(ReadOnlySpan<float>.Empty, bad, VMask.None, 5f);
        Assert.Equal(5f, none.GetLane(0));
    }

    [Fact]
    public void ShiftLanes_OffsetBeyondTheGang_IsAllZero()
    {
        VByte b = VByte.ProgramIndex + (byte)1;
        VShort s = VShort.ProgramIndex + (short)1;
        foreach (int offset in new[] { 250, -250, 256, -256, 65536, -65536, 65537, int.MaxValue, int.MinValue, VByte.LaneCount, -VByte.LaneCount })
        {
            Assert.Equal(VByte.Zero, Lanes.ShiftLanes(b, offset));
            Assert.Equal(VShort.Zero, Lanes.ShiftLanes(s, offset));
        }

        // In-range shifts are unchanged.
        VByte up = Lanes.ShiftLanes(b, 1);
        VByte down = Lanes.ShiftLanes(b, -1);
        VShort sup = Lanes.ShiftLanes(s, 1);
        VShort sdown = Lanes.ShiftLanes(s, -1);
        for (int l = 0; l < VByte.LaneCount; l++)
        {
            Assert.Equal(l + 1 < VByte.LaneCount ? (byte)(l + 2) : (byte)0, up.GetLane(l));
            Assert.Equal(l > 0 ? (byte)l : (byte)0, down.GetLane(l));
        }

        for (int l = 0; l < VShort.LaneCount; l++)
        {
            Assert.Equal(l + 1 < VShort.LaneCount ? (short)(l + 2) : (short)0, sup.GetLane(l));
            Assert.Equal(l > 0 ? (short)l : (short)0, sdown.GetLane(l));
        }
    }

    [Fact]
    public void Stencil_WithSubtractedOffsets_MatchesScalar()
    {
        int n = N + 8;
        float[] a = [.. Enumerable.Range(0, n).Select(i => MathF.Sin(i) * 10f)];
        float[] expected = new float[n], actual = new float[n];
        RegressionKernels.Stencil(a, expected, n, 3);
        RegressionKernels.Stencil_Simd(a, actual, n, 3);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void UniformBranch_MatchesScalar(int mode)
    {
        int n = N;
        float[] a = [.. Enumerable.Range(0, n).Select(i => i - 7.5f)];
        float[] expected = new float[n], actual = new float[n];
        RegressionKernels.UniformBranch(a, expected, n, mode);
        RegressionKernels.UniformBranch_Simd(a, actual, n, mode);
        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Run a one-input float kernel with each probe value placed in a full gang (first copy)
    /// and again in the scalar tail, and require gang, tail and MathF to agree.
    /// </summary>
    private static void CheckFloat(
        Action<float[], float[], int> scalar, Action<float[], float[], int> simd, float[] probes, float relTol)
    {
        int gangs = ((probes.Length + Spmd.LaneCount - 1) / Spmd.LaneCount) + 1;
        int n = (gangs * Spmd.LaneCount) + Math.Min(probes.Length, Spmd.LaneCount - 1);
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = probes[i % probes.Length];
        float[] expected = new float[n], actual = new float[n];
        scalar(a, expected, n);
        simd(a, actual, n);
        for (int i = 0; i < n; i++)
            AssertClose(expected[i], actual[i], relTol, $"f({a[i]:R}) at [{i}]");
    }

    private static void CheckDouble(
        Action<double[], double[], int> scalar, Action<double[], double[], int> simd, double[] probes, double relTol)
    {
        int gangs = ((probes.Length + VDouble.LaneCount - 1) / VDouble.LaneCount) + 1;
        int n = (gangs * VDouble.LaneCount) + Math.Min(probes.Length, VDouble.LaneCount - 1);
        double[] a = new double[n];
        for (int i = 0; i < n; i++)
            a[i] = probes[i % probes.Length];
        double[] expected = new double[n], actual = new double[n];
        scalar(a, expected, n);
        simd(a, actual, n);
        for (int i = 0; i < n; i++)
            AssertClose(expected[i], actual[i], relTol, $"f({a[i]:R}) at [{i}]");
    }

    private static void AssertClose(double expected, double actual, double relTol, string what)
    {
        if (double.IsNaN(expected) || double.IsInfinity(expected) || expected == 0d || relTol == 0d)
        {
            // Special values (and exact functions) must match exactly, sign of infinity included.
            Assert.True(expected.Equals(actual) || (expected == 0d && actual == 0d), $"{what}: expected {expected:R}, got {actual:R}");
            return;
        }

        double allowed = relTol * Math.Max(Math.Abs(expected), double.Epsilon);
        Assert.True(Math.Abs(expected - actual) <= allowed, $"{what}: expected {expected:R}, got {actual:R}");
    }

    [Fact]
    public void Round_MatchesMathF_IncludingTiesAndLargeValues()
    {
        float[] probes =
        [
            0.5f, 1.5f, 2.5f, 3.5f, -0.5f, -1.5f, -2.5f, 0.49999997f, -0.49999997f, 0.50000006f,
            1.4f, 1.6f, -1.4f, -1.6f, 8388607.5f, 8388608f, 3e9f, -3e9f, 1e20f,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0f,
        ];
        CheckFloat(RegressionKernels.RoundF, RegressionKernels.RoundF_Simd, probes, 0f);
        CheckFloat(RegressionKernels.RoundAwayF, RegressionKernels.RoundAwayF_Simd, probes, 0f);
        CheckDouble(RegressionKernels.RoundD, RegressionKernels.RoundD_Simd,
            [0.5, 1.5, 2.5, -0.5, -2.5, 0.49999999999999994, 4503599627370495.5, 1e300, -1e300, double.NaN, double.PositiveInfinity], 0d);
    }

    [Fact]
    public void VectorMathRound_IsTiesAwayFromZero_AndRoundToEvenIsBankers()
    {
        float[] ties = [0.5f, 1.5f, 2.5f, -0.5f, -2.5f, 0.49999997f, 3e9f];
        foreach (float v in ties)
        {
            Assert.Equal(MathF.Round(v, MidpointRounding.AwayFromZero), VectorMath.Round(new VFloat(v)).GetLane(0));
            Assert.Equal(MathF.Round(v), VectorMath.RoundToEven(new VFloat(v)).GetLane(0));
            Assert.Equal(Math.Round((double)v, MidpointRounding.AwayFromZero), VectorMath.Round(new VDouble(v)).GetLane(0));
            Assert.Equal(Math.Round((double)v), VectorMath.RoundToEven(new VDouble(v)).GetLane(0));
        }
    }

    [Fact]
    public void Floor_MatchesMathF_BeyondIntRange()
    {
        CheckFloat(RegressionKernels.FloorF, RegressionKernels.FloorF_Simd,
            [1.5f, -1.5f, -0.5f, 3e9f, -3e9f, 1e20f, -1e20f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f], 0f);
        CheckDouble(RegressionKernels.FloorD, RegressionKernels.FloorD_Simd,
            [1.5, -1.5, -0.5, 1e18, -1e18, 1e300, double.NaN, double.NegativeInfinity], 0d);

        // Truncate (the (int)/(long) cast in double kernels) rounds toward zero at any magnitude.
        foreach (double v in new[] { 1.9, -1.9, -0.5, 1e18 + 4096, -1e300, double.PositiveInfinity })
            Assert.Equal(Math.Truncate(v), VectorMath.Truncate(new VDouble(v)).GetLane(0));
    }

    [Fact]
    public void Exp_MatchesMathF_AcrossTheWholeRange()
    {
        float[] probes =
        [
            0f, 1f, -1f, 10f, -10f, 50f, -50f, 87f, -87f, 88f, 88.5f, 88.7f, 88.73f, 89f, 100f, 1e10f,
            -90f, -95f, -100f, -103f, -104f, -110f, -200f,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN,
        ];
        CheckFloat(RegressionKernels.ExpF, RegressionKernels.ExpF_Simd, probes, 1e-5f);

        double[] dprobes =
        [
            0d, 1d, -1d, 100d, -100d, 700d, 709d, 709.5d, 709.78d, 709.79d, 710d, 1e10,
            -708d, -720d, -740d, -745d, -746d, -800d,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN,
        ];
        CheckDouble(RegressionKernels.ExpD, RegressionKernels.ExpD_Simd, dprobes, 1e-12);
    }

    [Fact]
    public void Log_MatchesMathF_OnEdgeInputs()
    {
        float[] probes =
        [
            1f, 0.5f, 2f, 10f, 1e-30f, 1e30f, float.MaxValue, 1.17549435e-38f,
            0f, -0f, -1f, -1e30f, 1e-40f, 1.4e-45f, float.PositiveInfinity, float.NegativeInfinity, float.NaN,
        ];
        CheckFloat(RegressionKernels.LogF, RegressionKernels.LogF_Simd, probes, 2e-6f);

        double[] dprobes =
        [
            1d, 0.5d, 2d, 1e-300, 1e300, double.MaxValue, 2.2250738585072014e-308,
            0d, -1d, 1e-310, double.Epsilon, double.PositiveInfinity, double.NegativeInfinity, double.NaN,
        ];
        CheckDouble(RegressionKernels.LogD, RegressionKernels.LogD_Simd, dprobes, 1e-13);
    }

    [Fact]
    public void Pow_MatchesMathF_ForNegativeAndZeroBases()
    {
        (float X, float Y)[] cases =
        [
            (3f, 2f), (-2f, 2f), (-2f, 3f), (-0.5f, 2f), (-2f, -2f), (-2f, -3f), (-8f, 1f), (-2f, 0.5f), (-2f, 2.5f),
            (0f, 2f), (0f, 0.5f), (0f, 0f), (0f, -1f), (5f, 0f), (-5f, 0f), (2f, 10f), (2f, -10f), (10f, 0.5f),
            (float.PositiveInfinity, 2f), (float.PositiveInfinity, -2f), (0.5f, 300f), (2f, 300f),
        ];
        int n = (2 * Spmd.LaneCount * ((cases.Length / Spmd.LaneCount) + 1)) + 3;
        float[] x = new float[n], y = new float[n], expected = new float[n], actual = new float[n];
        for (int i = 0; i < n; i++)
            (x[i], y[i]) = cases[i % cases.Length];
        RegressionKernels.PowF(x, y, expected, n);
        RegressionKernels.PowF_Simd(x, y, actual, n);
        for (int i = 0; i < n; i++)
            AssertClose(expected[i], actual[i], 2e-5, $"pow({x[i]:R}, {y[i]:R}) at [{i}]");

        double[] dx = [.. x.Select(v => (double)v)], dy = [.. y.Select(v => (double)v)];
        double[] dExpected = new double[n], dActual = new double[n];
        RegressionKernels.PowD(dx, dy, dExpected, n);
        RegressionKernels.PowD_Simd(dx, dy, dActual, n);
        for (int i = 0; i < n; i++)
            AssertClose(dExpected[i], dActual[i], 1e-11, $"pow({dx[i]:R}, {dy[i]:R}) at [{i}]");
    }

    [Fact]
    public void SinCos_StayAccurateForLargeArguments()
    {
        float[] probes = [0f, 0.1f, 1f, -1f, 3f, -3f, 10f, 100f, 1000f, -1000f, 10000f, 100000f];
        // Past ~1e5 the reduction needs the exact product only a fused multiply-add gives.
        if (System.Runtime.Intrinsics.X86.Fma.IsSupported || System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported)
            probes = [.. probes, 1e6f, -1e6f, 3.2e6f];
        int n = (3 * Spmd.LaneCount) + 3;
        float[] a = new float[n], es = new float[n], ec = new float[n], vs = new float[n], vc = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = probes[i % probes.Length];
        RegressionKernels.SinF(a, es, n);
        RegressionKernels.SinF_Simd(a, vs, n);
        RegressionKernels.CosF(a, ec, n);
        RegressionKernels.CosF_Simd(a, vc, n);
        for (int i = 0; i < n; i++)
        {
            Assert.True(MathF.Abs(es[i] - vs[i]) <= 5e-6f, $"sin({a[i]:R}): expected {es[i]:R}, got {vs[i]:R}");
            Assert.True(MathF.Abs(ec[i] - vc[i]) <= 5e-6f, $"cos({a[i]:R}): expected {ec[i]:R}, got {vc[i]:R}");
        }
    }

    [Fact]
    public void IntegerDivide_MatchesScalar_OverTheFullIntRange()
    {
        var rnd = new Random(12345);
        int[] edge = [0, 1, -1, 2, -2, 3, 7, -7, 46340, 65536, int.MaxValue, int.MinValue, int.MaxValue - 1, int.MinValue + 1];
        int n = (edge.Length * edge.Length) + 4000 + 3;
        int[] a = new int[n], b = new int[n];
        int p = 0;
        foreach (int x in edge)
        {
            foreach (int y in edge)
            {
                // Skip the two pairs scalar C# itself throws on.
                a[p] = x;
                b[p] = y == 0 || (x == int.MinValue && y == -1) ? 1 : y;
                p++;
            }
        }

        for (; p < n; p++)
        {
            a[p] = rnd.Next(int.MinValue, int.MaxValue);
            int d = (p % 3) switch { 0 => rnd.Next(1, 100), 1 => rnd.Next(int.MinValue, int.MaxValue), _ => -rnd.Next(1, 1 << 20) };
            b[p] = d == 0 || (a[p] == int.MinValue && d == -1) ? 3 : d;
        }

        int[] eq = new int[n], er = new int[n], aq = new int[n], ar = new int[n];
        RegressionKernels.IntDivRem(a, b, eq, er, n);
        RegressionKernels.IntDivRem_Simd(a, b, aq, ar, n);
        Assert.Equal(eq, aq);
        Assert.Equal(er, ar);
    }

    [Fact]
    public void IntegerDivide_StillThrowsOnZeroAndOverflow()
    {
        var zero = VInt.Select(VMask.FirstN(1), VInt.Zero, VInt.One);
        _ = Assert.Throws<DivideByZeroException>(() => VInt.Divide(new VInt(10), zero));
        _ = Assert.Throws<DivideByZeroException>(() => VInt.Remainder(new VInt(10), zero));
        _ = Assert.Throws<DivideByZeroException>(() => VInt.Divide(new VInt(10), VInt.Zero));

        var minusOne = VInt.Select(VMask.FirstN(1), new VInt(-1), VInt.One);
        _ = Assert.Throws<OverflowException>(() => VInt.Divide(new VInt(int.MinValue), minusOne));
        // MinValue / 1 and x / -1 are both fine; only the one pair overflows.
        Assert.Equal(new VInt(int.MinValue), VInt.Divide(new VInt(int.MinValue), VInt.One));
        Assert.Equal(new VInt(-7), VInt.Divide(new VInt(7), new VInt(-1)));
    }

    [Fact]
    public void MaskedLoadStore_FullMaskMatchesPartialMaskSemantics()
    {
        int w = VFloat.LaneCount;
        float[] src = [.. Enumerable.Range(0, (2 * w) + 3).Select(i => i + 1f)];

        // All lanes active and in range: identical to a plain load/store.
        Assert.Equal(VFloat.Load(src, 1), VFloat.LoadMasked(src, 1, VMask.All));
        float[] dst = new float[src.Length];
        VFloat.Load(src, 1).StoreMasked(dst, 2, VMask.All);
        Assert.Equal(src.AsSpan(1, w).ToArray(), dst.AsSpan(2, w).ToArray());

        // All lanes active but the gang runs off the end: lanes past the end are dropped / take the fallback.
        int off = src.Length - 2;
        VFloat tail = VFloat.LoadMasked(src, off, VMask.All, -1f);
        Assert.Equal(src[off], tail.GetLane(0));
        Assert.Equal(src[off + 1], tail.GetLane(1));
        for (int l = 2; l < w; l++)
            Assert.Equal(-1f, tail.GetLane(l));

        float[] dst2 = new float[src.Length];
        new VFloat(9f).StoreMasked(dst2, off, VMask.All);
        Assert.Equal(9f, dst2[off]);
        Assert.Equal(9f, dst2[off + 1]);
        Assert.Equal(0f, dst2[off - 1]);

        // Partial mask: only the selected lanes move.
        float[] dst3 = new float[src.Length];
        new VFloat(4f).StoreMasked(dst3, 0, VMask.FirstN(2));
        Assert.Equal(new float[] { 4f, 4f, 0f }, dst3.AsSpan(0, 3).ToArray());

        int[] isrc = [.. Enumerable.Range(0, (2 * w) + 3)];
        int[] idst = new int[isrc.Length];
        VInt.Load(isrc, 3).StoreMasked(idst, 0, VMask.All);
        Assert.Equal(isrc.AsSpan(3, w).ToArray(), idst.AsSpan(0, w).ToArray());

        double[] dsrc = [.. Enumerable.Range(0, (2 * VDouble.LaneCount) + 3).Select(i => i + 0.5)];
        Assert.Equal(VDouble.Load(dsrc, 1), VDouble.LoadMasked(dsrc, 1, VMaskD.All));
        double[] ddst = new double[dsrc.Length];
        VDouble.Load(dsrc, 1).StoreMasked(ddst, 0, VMaskD.All);
        Assert.Equal(dsrc.AsSpan(1, VDouble.LaneCount).ToArray(), ddst.AsSpan(0, VDouble.LaneCount).ToArray());
    }

    [Fact]
    public void LoadStore_OutOfRangeOffsets_Throw()
    {
        float[] buf = new float[VFloat.LaneCount + 2];
        float[] small = new float[VFloat.LaneCount - 1];
        foreach (int bad in new[] { -1, 3, buf.Length, int.MaxValue, int.MinValue })
        {
            _ = Assert.ThrowsAny<ArgumentException>(() => VFloat.Load(buf, bad));
            _ = Assert.ThrowsAny<ArgumentException>(() => VFloat.One.Store(buf, bad));
        }

        _ = Assert.ThrowsAny<ArgumentException>(() => VFloat.Load(small, 0));
        _ = Assert.ThrowsAny<ArgumentException>(() => VFloat.One.Store(small, 0));
        _ = Assert.ThrowsAny<ArgumentException>(() => VInt.Load(new int[VInt.LaneCount], 1));
        _ = Assert.ThrowsAny<ArgumentException>(() => VDouble.Load(new double[VDouble.LaneCount], 1));
        _ = Assert.ThrowsAny<ArgumentException>(() => VLong.One.Store(new long[VLong.LaneCount], 1));

        // The last valid offset still works.
        VFloat.One.Store(buf, 2);
        Assert.Equal(VFloat.One, VFloat.Load(buf, 2));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(1000)]
    [InlineData(100_003)]
    public void StreamingKernel_MatchesScalar_AtAnyAlignment(int n)
    {
        // Slide the start through every alignment the peel loop can see.
        for (int shift = 0; shift < 9; shift++)
        {
            float[] a = [.. Enumerable.Range(0, n + shift).Select(i => (i % 17) - 8f)];
            float[] expected = new float[n + shift], actual = new float[n + shift];
            float es = RegressionKernels.StreamingScale(a, expected, 1.5f, n);
            float vs = RegressionKernels.StreamingScale_Simd(a, actual, 1.5f, n);
            Assert.Equal(expected, actual);
            Assert.Equal(es, vs, 1e-2f * Math.Max(1f, MathF.Abs(es)));

            double[] da = [.. a.Select(v => (double)v)];
            double[] dExpected = new double[n + shift], dActual = new double[n + shift];
            RegressionKernels.StreamingDouble(da, dExpected, n);
            RegressionKernels.StreamingDouble_Simd(da, dActual, n);
            Assert.Equal(dExpected, dActual);
        }
    }

    [Fact]
    public void StreamingKernel_WithLaneReturn_MatchesContinueSemantics()
    {
        // Per-lane 'return' retires that element only (ISPC semantics), in the peel loop, the
        // vector loop and the tail alike. Non-negative input never returns, so scalar agrees.
        int n = 1000;
        float[] a = [.. Enumerable.Range(0, n).Select(i => (float)(i % 13))];
        float[] expected = new float[n], actual = new float[n], parallel = new float[n];
        RegressionKernels.StreamingWithReturn(a, expected, n);
        RegressionKernels.StreamingWithReturn_Simd(a, actual, n);
        RegressionKernels.StreamingWithReturn_ParallelSimd(a, parallel, n, minChunkSize: 64);
        Assert.Equal(expected, actual);
        Assert.Equal(expected, parallel);

        // With negatives: only those elements are skipped.
        a[5] = -1f;
        a[n - 2] = -1f;
        float[] skipped = new float[n];
        RegressionKernels.StreamingWithReturn_Simd(a, skipped, n);
        for (int i = 0; i < n; i++)
            Assert.Equal(a[i] < 0f ? 0f : a[i] * 3f, skipped[i]);
    }

    [Fact]
    public void StoreNonTemporal_NeverWritesPastTheBuffer()
    {
        // Whatever the alignment, a gang that does not fit must throw rather than stream out of bounds.
        for (int extra = 0; extra < 9; extra++)
        {
            float[] buf = new float[(3 * VFloat.LaneCount) + extra];
            for (int off = buf.Length - VFloat.LaneCount + 1; off <= buf.Length + 1; off++)
            {
                int o = off;
                _ = Assert.ThrowsAny<ArgumentException>(() => VFloat.One.StoreNonTemporal(buf, o));
            }

            _ = Assert.ThrowsAny<ArgumentException>(() => VFloat.One.StoreNonTemporal(buf, -1));
            VFloat.One.StoreNonTemporal(buf, buf.Length - VFloat.LaneCount);
            Assert.Equal(1f, buf[^1]);

            double[] dbuf = new double[(3 * VDouble.LaneCount) + extra];
            _ = Assert.ThrowsAny<ArgumentException>(() => VDouble.One.StoreNonTemporal(dbuf, dbuf.Length - VDouble.LaneCount + 1));
            VDouble.One.StoreNonTemporal(dbuf, dbuf.Length - VDouble.LaneCount);
            Assert.Equal(1d, dbuf[^1]);
        }
    }

    [Fact]
    public void Scatter_HigherLaneWinsOnDuplicateIndex()
    {
        float[] dst = new float[4];
        Memory.Scatter(dst, new VInt(2), VInt.ProgramIndex.ToFloat() + 1f, VMask.All);
        Assert.Equal(VFloat.LaneCount, dst[2]);

        float[] dst2 = new float[4];
        Memory.Scatter(dst2, new VInt(1), VInt.ProgramIndex.ToFloat() + 1f, VMask.FirstN(2));
        Assert.Equal(2f, dst2[1]);

        _ = Assert.Throws<IndexOutOfRangeException>(() => Memory.Scatter(new float[4], new VInt(4), VFloat.One, VMask.All));
    }

    [Fact]
    public void Reductions_MatchPerLaneReference()
    {
        var rnd = new Random(3);
        int[] iv = [.. Enumerable.Range(0, VInt.LaneCount).Select(_ => rnd.Next(-1000, 1000))];
        long[] lv = [.. Enumerable.Range(0, VLong.LaneCount).Select(_ => (long)rnd.Next(-1000, 1000) << 20)];
        float[] fv = [.. Enumerable.Range(0, VFloat.LaneCount).Select(_ => (float)rnd.NextDouble() - 0.5f)];
        double[] dv = [.. Enumerable.Range(0, VDouble.LaneCount).Select(_ => rnd.NextDouble() - 0.5)];

        Assert.Equal(iv.Sum(), Reduce.Add(VInt.Load(iv)));
        Assert.Equal(iv.Min(), Reduce.Min(VInt.Load(iv)));
        Assert.Equal(iv.Max(), Reduce.Max(VInt.Load(iv)));
        Assert.Equal(lv.Sum(), Reduce.Add(VLong.Load(lv)));
        Assert.Equal(lv.Min(), Reduce.Min(VLong.Load(lv)));
        Assert.Equal(lv.Max(), Reduce.Max(VLong.Load(lv)));
        Assert.Equal(fv.Min(), Reduce.Min(VFloat.Load(fv)));
        Assert.Equal(fv.Max(), Reduce.Max(VFloat.Load(fv)));
        Assert.Equal(dv.Min(), Reduce.Min(VDouble.Load(dv)));
        Assert.Equal(dv.Max(), Reduce.Max(VDouble.Load(dv)));

        // int.MinValue + int.MinValue wraps the same way the scalar loop does.
        Assert.Equal(unchecked(int.MaxValue * VInt.LaneCount), Reduce.Add(new VInt(int.MaxValue)));
    }

    [Fact]
    public void FirstN_And_Rotate_MatchPerLaneDefinition()
    {
        for (int count = -1; count <= VMask.LaneCount + 1; count++)
        {
            VMask m = VMask.FirstN(count);
            for (int l = 0; l < VMask.LaneCount; l++)
                Assert.Equal(l < count, m.IsLaneActive(l));
        }

        for (int count = -1; count <= VMaskD.LaneCount + 1; count++)
        {
            VMaskD m = VMaskD.FirstN(count);
            for (int l = 0; l < VMaskD.LaneCount; l++)
                Assert.Equal(l < count, m.IsLaneActive(l));
        }

        int w = VInt.LaneCount;
        foreach (int offset in new[] { 0, 1, -1, w, -w, w + 3, -(w + 3), 1000, -1000 })
        {
            VInt r = Lanes.Rotate(VInt.ProgramIndex, offset);
            for (int l = 0; l < w; l++)
                Assert.Equal((((l + offset) % w) + w) % w, r.GetLane(l));
        }

        VByte rb = Lanes.Rotate(VByte.ProgramIndex, -3);
        for (int l = 0; l < VByte.LaneCount; l++)
            Assert.Equal((byte)((l - 3 + VByte.LaneCount) % VByte.LaneCount), rb.GetLane(l));
        VShort rs = Lanes.Rotate(VShort.ProgramIndex, 5);
        for (int l = 0; l < VShort.LaneCount; l++)
            Assert.Equal((short)((l + 5) % VShort.LaneCount), rs.GetLane(l));
    }
}

/// <summary>
/// What the generator emits for the constructs whose lowering was changed for performance:
/// these assert on the generated source, since the results alone can't tell a gather from a load.
/// </summary>
public class LoweringShapeTests
{
    private static string Generate(string body, string parameters = "float[] a, float[] o, int n, int mode", string attr = "GenerateParallel = false")
    {
        string source = $$"""
            using System;
            using IspcSharp;

            namespace Shape;

            public static partial class Kernels
            {
                [Spmd({{attr}})]
                public static void Run({{parameters}})
                {
                    foreach (int i in Spmd.Range(1, n))
                    {
                        {{body}}
                    }
                }
            }
            """;
        var driver = GeneratorCachingTests.CreateDriverFor().RunGenerators(GeneratorCachingTests.CreateCompilationFor(source));
        var result = driver.GetRunResult().Results[0];
        Assert.Empty(result.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error));
        return result.GeneratedSources.Single(s => s.HintName == "Shape.Kernels.Run_Spmd.g.cs").SourceText.ToString();
    }

    [Fact]
    public void SubtractedOffset_IsAContiguousLoad()
    {
        string src = Generate("o[i] = a[i - 1] + a[i - mode] + a[(i + 2) - 3];");
        Assert.DoesNotContain("Memory.Gather", src);
        Assert.Contains("VFloat.Load(a, __i + (-1))", src);
        Assert.Contains("VFloat.Load(a, __i + (-mode))", src);
    }

    [Fact]
    public void ReversedIndex_IsStillAGather()
    {
        // 'mode - i' walks backwards across the gang: not contiguous.
        string src = Generate("o[i] = a[mode - i];");
        Assert.Contains("Memory.Gather", src);
    }

    [Fact]
    public void UniformCondition_IsAPlainBranch()
    {
        string src = Generate("float x = a[i]; if (mode == 1) x = MathF.Sqrt(x); else x = MathF.Exp(x); o[i] = x;");
        Assert.Contains("if (mode == 1)", src);
        Assert.DoesNotContain("VMask __m", src);
    }

    [Fact]
    public void VaryingCondition_IsStillMasked()
    {
        string src = Generate("float x = a[i]; if (x > 0f) x = MathF.Sqrt(x); o[i] = x;");
        Assert.Contains("VMask __m0", src);
    }

    [Fact]
    public void MathRound_LowersToTiesToEven()
    {
        Assert.Contains("VectorMath.RoundToEven(", Generate("o[i] = MathF.Round(a[i]);"));
        Assert.Contains("VectorMath.Round(", Generate("o[i] = MathF.Round(a[i], MidpointRounding.AwayFromZero);"));
    }

    [Fact]
    public void DeepHornerChain_GeneratesInLinearTime()
    {
        // Lowering used to re-lower both factors of every product while testing for a fused
        // multiply-add, doubling the work per nesting level: degree 40 would never finish.
        string expr = "x";
        for (int k = 1; k <= 40; k++)
            expr = $"({expr}) * x + {k}f";
        var sw = System.Diagnostics.Stopwatch.StartNew();
        string src = Generate($"float x = a[i]; o[i] = {expr};");
        sw.Stop();
        Assert.Equal(40, src.Split("VFloat.MulAdd(").Length - 1);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(20), $"generation took {sw.Elapsed}");
    }

    [Fact]
    public void StreamingKernel_PeelsToAlignment()
    {
        string src = Generate("o[i] = a[i] * 2f;", attr: "GenerateParallel = false, Streaming = true");
        Assert.Contains("StoreNonTemporal", src);
        Assert.Contains("Memory.StreamingPeel(o", src);
    }
}
