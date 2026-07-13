using System;
using IspcSharp;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// Tests for the SPMD execution engine: Spmd.Foreach, Spmd.ParallelForeach,
/// Spmd.While (divergent loops), and Spmd.If (coherent if).
/// </summary>
public class SpmdEngineTests
{
    [Fact]
    public void Foreach_CoversAllElements()
    {
        int n = Spmd.LaneCount * 3 + 1; // deliberately hits the tail
        var data = new float[n];
        var result = new float[n];
        for (int i = 0; i < n; i++) data[i] = i;

        Spmd.Foreach(n, (in SpmdContext ctx) =>
        {
            var x = VFloat.LoadMasked(data, ctx.Base, ctx.Active);
            (x * 2f).StoreMasked(result, ctx.Base, ctx.Active);
        });

        for (int i = 0; i < n; i++)
            Assert.Equal(i * 2f, result[i]);
    }

    [Fact]
    public void Foreach_TailMaskIsPartial()
    {
        int w = Spmd.LaneCount;
        int n = w + 1; // one element in the tail
        var result = new int[n];

        Spmd.Foreach(n, (in SpmdContext ctx) =>
        {
            // Write the lane index to verify which lanes ran.
            VInt.ProgramIndex.StoreMasked(result, ctx.Base, ctx.Active);
        });

        // Full gangs: lane index = element index
        for (int i = 0; i < w; i++)
            Assert.Equal(i, result[i]);
        // Tail: only lane 0 active
        Assert.Equal(0, result[w]);
    }

    [Fact]
    public void Foreach_IsFullGangFlagCorrect()
    {
        int w = Spmd.LaneCount;
        int n = w * 2 + 1;
        var fullGangFlags = new bool[n / w + 1];
        int flagIdx = 0;

        Spmd.Foreach(n, (in SpmdContext ctx) =>
        {
            fullGangFlags[flagIdx++] = ctx.IsFullGang;
        });

        Assert.True(fullGangFlags[0]); // first gang full
        Assert.True(fullGangFlags[1]); // second gang full
        Assert.False(fullGangFlags[2]); // tail gang partial
    }

    [Fact]
    public void Foreach_Index_IsCorrectPerLane()
    {
        int n = Spmd.LaneCount * 2;
        var result = new int[n];

        Spmd.Foreach(n, (in SpmdContext ctx) =>
        {
            ctx.Index.Store(result, ctx.Base);
        });

        for (int i = 0; i < n; i++)
            Assert.Equal(i, result[i]);
    }

    [Fact]
    public void ParallelForeach_ProducesSameResultsAsForeach()
    {
        int n = Spmd.LaneCount * 100 + 7;
        var data = new float[n];
        var resultSerial = new float[n];
        var resultParallel = new float[n];
        var rng = new Random(123);
        for (int i = 0; i < n; i++) data[i] = (float)rng.NextDouble() * 10f;

        Spmd.Foreach(n, (in SpmdContext ctx) =>
        {
            var x = VFloat.LoadMasked(data, ctx.Base, ctx.Active);
            (VectorMath.Sqrt(x) + 1f).StoreMasked(resultSerial, ctx.Base, ctx.Active);
        });

        Spmd.ParallelForeach(n, (in SpmdContext ctx) =>
        {
            var x = VFloat.LoadMasked(data, ctx.Base, ctx.Active);
            (VectorMath.Sqrt(x) + 1f).StoreMasked(resultParallel, ctx.Base, ctx.Active);
        });

        for (int i = 0; i < n; i++)
            Assert.Equal(resultSerial[i], resultParallel[i], 4);
    }

    [Fact]
    public void ParallelForeach_SmallCountRunsSingleThreaded()
    {
        // Below minChunkSize, ParallelForeach falls back to Foreach.
        int n = 100;
        var result = new float[n];

        Spmd.ParallelForeach(n, (in SpmdContext ctx) =>
        {
            VFloat.One.StoreMasked(result, ctx.Base, ctx.Active);
        });

        for (int i = 0; i < n; i++)
            Assert.Equal(1f, result[i]);
    }

    [Fact]
    public void While_NewtonSqrt_ConvergesPerLane()
    {
        int w = Spmd.LaneCount;
        var input = new float[w];
        var result = new float[w];
        var rng = new Random(42);
        for (int i = 0; i < w; i++) input[i] = (float)rng.NextDouble() * 4 + 0.01f;

        Spmd.Foreach(w, (in SpmdContext ctx) =>
        {
            var x = VFloat.Load(input, ctx.Base);
            var guess = x;
            var err = new VFloat(1f);
            var next = VFloat.Zero;

            Spmd.While(
                ctx.Active,
                active => active & (err > new VFloat(0.0001f)),
                (ref LoopState s) =>
                {
                    next = new VFloat(0.5f) * (guess + x / guess);
                    err = VectorMath.Abs(next - guess);
                    guess = next;
                },
                maxIterations: 100);

            guess.StoreMasked(result, ctx.Base, ctx.Active);
        });

        for (int i = 0; i < w; i++)
            Assert.Equal(MathF.Sqrt(input[i]), result[i], 3);
    }

    [Fact]
    public void While_Break_ExitsLanesEarly()
    {
        int w = Spmd.LaneCount;
        var result = new int[w];

        Spmd.Foreach(w, (in SpmdContext ctx) =>
        {
            var count = VInt.Zero;
            var breakAt = VInt.ProgramIndex + 1; // lane l breaks after l+1 iterations

            Spmd.While(
                ctx.Active,
                active => active,
                (ref LoopState s) =>
                {
                    count = count.Blend(s.Active, count + 1);
                    s.Break(s.Active & (count >= breakAt));
                },
                maxIterations: w + 1);

            count.StoreMasked(result, ctx.Base, ctx.Active);
        });

        // Lane l should have counted up to l+1 (then broke).
        for (int l = 0; l < w; l++)
            Assert.Equal(l + 1, result[l]);
    }

    [Fact]
    public void While_Continue_SkipsRestOfIteration()
    {
        int w = Spmd.LaneCount;
        var result = new int[w];

        Spmd.Foreach(w, (in SpmdContext ctx) =>
        {
            var count = VInt.Zero;
            var sum = VInt.Zero;

            Spmd.While(
                ctx.Active,
                active => active & (count < new VInt(10)),
                (ref LoopState s) =>
                {
                    count = count.Blend(s.Active, count + 1);
                    // Skip even iterations (continue before the add).
                    // VInt has no % operator; use count - (count / 2) * 2 to get parity.
                    var half = VInt.FromFloatTruncate(count.ToFloat() / new VFloat(2f));
                    var parity = count - half * 2;
                    var skip = s.Active & VInt.Eq(parity, VInt.Zero);
                    s.Continue(skip);
                    sum = sum.Blend(s.Active, sum + count);
                },
                maxIterations: 20);

            sum.StoreMasked(result, ctx.Base, ctx.Active);
        });

        // Sum of odd numbers 1+3+5+7+9 = 25
        for (int l = 0; l < w; l++)
            Assert.Equal(25, result[l]);
    }

    [Fact]
    public void While_Return_PropagatesReturnedMask()
    {
        int w = Spmd.LaneCount;
        var returned = new bool[w];

        Spmd.Foreach(w, (in SpmdContext ctx) =>
        {
            var count = VInt.Zero;

            var state = Spmd.While(
                ctx.Active,
                active => active,
                (ref LoopState s) =>
                {
                    count = count.Blend(s.Active, count + 1);
                    s.Return(s.Active & (count >= new VInt(5)));
                },
                maxIterations: 20);

            // Lanes that returned should be in state.Returned.
            for (int l = 0; l < w; l++)
                if (ctx.Active.IsLaneActive(l))
                    returned[ctx.Base + l] = state.Returned.IsLaneActive(l);
        });

        for (int l = 0; l < w; l++)
            Assert.True(returned[l]);
    }

    [Fact]
    public void While_ExceedingMaxIterations_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
        {
            Spmd.While(
                VMask.All,
                active => active, // never becomes false
                (ref LoopState s) => { },
                maxIterations: 5);
        });
    }

    [Fact]
    public void If_ThenBranchRunsWhenAnyLaneMatches()
    {
        int w = Spmd.LaneCount;
        var data = MakeRange(w);
        var result = new float[w];

        Spmd.Foreach(w, (in SpmdContext ctx) =>
        {
            var x = VFloat.Load(data, ctx.Base);
            int baseIdx = ctx.Base;
            Spmd.If(ctx.Active, x > new VFloat(w / 2f), then: m =>
            {
                (x * 2f).StoreMasked(result, baseIdx, m);
            }, @else: m =>
            {
                x.StoreMasked(result, baseIdx, m);
            });
        });

        for (int i = 0; i < w; i++)
        {
            float expected = i > w / 2f ? i * 2f : i;
            Assert.Equal(expected, result[i]);
        }
    }

    [Fact]
    public void If_SkipsWhenNoLaneMatches()
    {
        bool thenRan = false;
        bool elseRan = false;

        Spmd.If(VMask.All, VMask.None, then: _ => thenRan = true, @else: _ => elseRan = true);

        Assert.False(thenRan);
        Assert.True(elseRan);
    }

    [Fact]
    public void Foreach2D_CoversAllPixels()
    {
        const int W = 64, H = 48;
        var result = new int[W * H];

        Spmd.Foreach2D(W, H, (in SpmdContext ctx, int y) =>
        {
            for (int l = 0; l < VFloat.LaneCount; l++)
            {
                if (ctx.Active.IsLaneActive(l))
                {
                    int x = ctx.Base + l;
                    result[y * W + x] = y * 1000 + x;
                }
            }
        });

        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                Assert.Equal(y * 1000 + x, result[y * W + x]);
    }

    private static float[] MakeRange(int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = i;
        return a;
    }
}
