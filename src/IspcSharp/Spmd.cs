using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace IspcSharp
{
    /// <summary>
    /// Per-iteration context handed to an SPMD kernel body.
    /// Equivalent to ISPC's implicit execution state inside a <c>foreach</c>.
    /// </summary>
    public readonly ref struct SpmdContext
    {
        /// <summary>Index of the first lane in this "gang" (the base element index).</summary>
        public readonly int Base;

        /// <summary>Per-lane element indices: Base + {0,1,...,LaneCount-1}. ISPC's loop variable.</summary>
        public readonly VInt Index;

        /// <summary>
        /// Lanes that correspond to real elements. All-active for full gangs;
        /// partially active for the tail when count % LaneCount != 0.
        /// Kernels must route every memory write through this mask (StoreMasked / Blend).
        /// </summary>
        public readonly VMask Active;

        /// <summary>True when every lane maps to a real element (no tail masking needed).</summary>
        public readonly bool IsFullGang;

        public SpmdContext(int @base, VMask active, bool isFullGang)
        {
            Base = @base;
            Active = active;
            IsFullGang = isFullGang;
            Index = VInt.ProgramIndex + @base;
        }
    }

    /// <summary>Kernel body: executes one gang (LaneCount elements) of the iteration space.</summary>
    public delegate void SpmdKernel(in SpmdContext ctx);

    /// <summary>2D kernel body: one gang along x, with the (uniform) row index y.</summary>
    public delegate void Spmd2DKernel(in SpmdContext ctx, int y);

    /// <summary>Condition for a divergent loop: returns the lanes that want another iteration.</summary>
    public delegate VMask SpmdCondition(VMask active);

    /// <summary>Body of a divergent loop. May shrink the mask (break/return) via <see cref="LoopState"/>.</summary>
    public delegate void SpmdLoopBody(ref LoopState state);

    /// <summary>
    /// Mutable per-gang loop state for divergent while/for loops:
    /// tracks which lanes are still running, and which have hit break/continue/return.
    /// </summary>
    public struct LoopState
    {
        /// <summary>Lanes still executing this iteration (after masking out continue for the rest of the body).</summary>
        public VMask Active;

        /// <summary>Lanes that have exited the loop entirely (break or return).</summary>
        public VMask Exited;

        /// <summary>Lanes that returned from the whole kernel (propagates outward).</summary>
        public VMask Returned;

        internal VMask _continued;

        /// <summary>SPMD <c>break</c>: deactivate <paramref name="lanes"/> for all remaining iterations.</summary>
        public void Break(VMask lanes)
        {
            var b = lanes & Active;
            Exited |= b;
            Active = VMask.AndNot(Active, b);
        }

        /// <summary>SPMD <c>continue</c>: deactivate <paramref name="lanes"/> for the rest of THIS iteration only.</summary>
        public void Continue(VMask lanes)
        {
            var c = lanes & Active;
            _continued |= c;
            Active = VMask.AndNot(Active, c);
        }

        /// <summary>SPMD <c>return</c>: deactivate <paramref name="lanes"/> permanently, and record them as returned.</summary>
        public void Return(VMask lanes)
        {
            var r = lanes & Active;
            Returned |= r;
            Exited |= r;
            Active = VMask.AndNot(Active, r);
        }
    }

    /// <summary>
    /// The SPMD execution engine, ISPC's <c>foreach</c>, <c>foreach_tiled</c>, tasking,
    /// and divergent-loop machinery.
    /// </summary>
    public static class Spmd
    {
        /// <summary>SIMD lane count on this machine (floats per instruction).</summary>
        public static int LaneCount => VFloat.LaneCount;

        /// <summary>
        /// Marker for the [Spmd] source generator: <c>foreach (var i in Spmd.Range(n))</c>.
        /// Never executed at runtime by generated code; executes scalar if called directly.
        /// </summary>
        public static IEnumerable<int> Range(int count)
        {
            for (int i = 0; i < count; i++) yield return i;
        }

        /// <summary>Half-open range [start, end), generator marker, also runs scalar if called directly.</summary>
        public static IEnumerable<int> Range(int start, int end)
        {
            for (int i = start; i < end; i++) yield return i;
        }

        /// <summary>
        /// 2D domain marker for the [Spmd] source generator:
        /// <c>foreach (var (x, y) in Spmd.Range2D(width, height))</c>. Rows are iterated in
        /// y-major order with x-gangs along each row (contiguous loads). Runs scalar if
        /// called directly, so the un-generated method stays a valid reference.
        /// </summary>
        public static IEnumerable<(int X, int Y)> Range2D(int width, int height)
        {
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    yield return (x, y);
        }

        /// <summary>
        /// Cache-blocked 2D domain marker, ISPC's <c>foreach_tiled</c>:
        /// <c>foreach (var (x, y) in Spmd.Range2DTiled(width, height))</c>. The domain is
        /// walked tile by tile (default 64×64) so that kernels touching multiple rows keep
        /// their working set in cache; within a tile, x-gangs stay contiguous. The generated
        /// <c>_ParallelSimd</c> distributes whole tiles across cores. Runs scalar (in the same
        /// tiled order) if called directly.
        /// </summary>
        public static IEnumerable<(int X, int Y)> Range2DTiled(int width, int height, int tileWidth = 64, int tileHeight = 64)
        {
            for (int ty = 0; ty < height; ty += tileHeight)
                for (int tx = 0; tx < width; tx += tileWidth)
                {
                    int yEnd = Math.Min(ty + tileHeight, height);
                    int xEnd = Math.Min(tx + tileWidth, width);
                    for (int y = ty; y < yEnd; y++)
                        for (int x = tx; x < xEnd; x++)
                            yield return (x, y);
                }
        }

        /// <summary>
        /// 2D tiled foreach, ISPC's <c>foreach_tiled</c>. Iterates a width x height domain in
        /// cache-friendly tiles; within a tile, gangs run along x so loads stay contiguous.
        /// The kernel receives the tile row's y (uniform per gang) plus the usual x-gang context.
        /// </summary>
        public static void Foreach2D(int width, int height, Spmd2DKernel kernel, int tileW = 64, int tileH = 64)
        {
            for (int ty = 0; ty < height; ty += tileH)
            {
                int yEnd = Math.Min(ty + tileH, height);
                for (int tx = 0; tx < width; tx += tileW)
                {
                    int xEnd = Math.Min(tx + tileW, width);
                    for (int y = ty; y < yEnd; y++)
                        ForeachRange(tx, xEnd, y, kernel);
                }
            }
        }

        /// <summary>Multi-core 2D tiled foreach: tiles are distributed across the thread pool.</summary>
        public static void ParallelForeach2D(int width, int height, Spmd2DKernel kernel, int tileW = 64, int tileH = 64)
        {
            int tilesX = (width + tileW - 1) / tileW;
            int tilesY = (height + tileH - 1) / tileH;

            Parallel.For(0, tilesX * tilesY, t =>
            {
                int tx = (t % tilesX) * tileW;
                int ty = (t / tilesX) * tileH;
                int xEnd = Math.Min(tx + tileW, width);
                int yEnd = Math.Min(ty + tileH, height);
                for (int y = ty; y < yEnd; y++)
                    ForeachRange(tx, xEnd, y, kernel);
            });
        }

        private static void ForeachRange(int start, int end, int y, Spmd2DKernel kernel)
        {
            int width = LaneCount;
            int i = start;
            int len = end - start;
            int fullEnd = start + len - len % width;

            var all = VMask.All;
            for (; i < fullEnd; i += width)
            {
                var ctx = new SpmdContext(i, all, isFullGang: true);
                kernel(in ctx, y);
            }
            int tail = end - i;
            if (tail > 0)
            {
                var ctx = new SpmdContext(i, VMask.FirstN(tail), isFullGang: false);
                kernel(in ctx, y);
            }
        }

        /// <summary>
        /// ISPC's <c>foreach</c>: run <paramref name="kernel"/> over [0, count) in gangs of LaneCount,
        /// with a masked tail. Single-threaded (SIMD only).
        /// </summary>
        public static void Foreach(int count, SpmdKernel kernel)
        {
            int width = LaneCount;
            int i = 0;
            int fullEnd = count - count % width;

            var all = VMask.All;
            for (; i < fullEnd; i += width)
            {
                var ctx = new SpmdContext(i, all, isFullGang: true);
                kernel(in ctx);
            }

            int tail = count - i;
            if (tail > 0)
            {
                var ctx = new SpmdContext(i, VMask.FirstN(tail), isFullGang: false);
                kernel(in ctx);
            }
        }

        /// <summary>
        /// ISPC's <c>launch</c>/tasking equivalent: SIMD lanes x all CPU cores.
        /// Splits [0, count) into chunks and runs <see cref="Foreach(int, SpmdKernel)"/>-style
        /// gangs inside each chunk on the thread pool.
        /// </summary>
        /// <param name="count">Iteration count.</param>
        /// <param name="kernel">Gang kernel. Must be thread-safe w.r.t. its captured state (disjoint writes by Index are fine).</param>
        /// <param name="minChunkSize">Minimum elements per task, to amortize scheduling overhead. Rounded to a multiple of LaneCount.</param>
        public static void ParallelForeach(int count, SpmdKernel kernel, int minChunkSize = 16384)
        {
            int width = LaneCount;
            if (count <= minChunkSize)
            {
                Foreach(count, kernel);
                return;
            }

            int chunk = Math.Max(width, (minChunkSize / width) * width);
            int numChunks = (count + chunk - 1) / chunk;

            Parallel.For(0, numChunks, c =>
            {
                int start = c * chunk;
                int end = Math.Min(start + chunk, count);
                int i = start;
                int len = end - start;
                int fullEnd = start + len - len % width;

                var all = VMask.All;
                for (; i < fullEnd; i += width)
                {
                    var ctx = new SpmdContext(i, all, isFullGang: true);
                    kernel(in ctx);
                }
                int tail = end - i;
                if (tail > 0)
                {
                    var ctx = new SpmdContext(i, VMask.FirstN(tail), isFullGang: false);
                    kernel(in ctx);
                }
            });
        }

        /// <summary>
        /// Divergent while loop, ISPC's <c>while</c> with per-lane trip counts.
        /// Iterates until no lane's condition holds (or every lane has broken/returned).
        /// </summary>
        /// <param name="initialActive">Lanes entering the loop (usually ctx.Active, AND-ed with any enclosing if).</param>
        /// <param name="condition">Given currently-running lanes, return the subset that wants another iteration.</param>
        /// <param name="body">Loop body; use <see cref="LoopState.Break"/>/<see cref="LoopState.Continue"/>/<see cref="LoopState.Return"/> for divergent exits.</param>
        /// <param name="maxIterations">Safety valve against runaway loops (throw beyond this).</param>
        /// <returns>Final state; <see cref="LoopState.Returned"/> tells the caller which lanes returned from the kernel.</returns>
        public static LoopState While(VMask initialActive, SpmdCondition condition, SpmdLoopBody body, int maxIterations = int.MaxValue)
        {
            var state = new LoopState { Active = initialActive, Exited = VMask.None, Returned = VMask.None };
            var running = initialActive;

            int iter = 0;
            while (true)
            {
                var want = condition(running) & running;
                if (!want.Any()) break;
                if (iter++ >= maxIterations)
                    throw new InvalidOperationException($"Spmd.While exceeded {maxIterations} iterations, likely a lane whose condition never becomes false.");

                state.Active = want;
                state._continued = VMask.None;
                body(ref state);

                // Lanes that 'continue'd rejoin for the next iteration; broken/returned lanes do not.
                running = VMask.AndNot(want | state._continued, state.Exited);
                running = VMask.AndNot(running, state.Returned);
            }

            state.Active = VMask.AndNot(initialActive, state.Exited);
            return state;
        }

        /// <summary>
        /// Coherent-if marker for the [Spmd] source generator, ISPC's <c>cif</c>.
        /// Wrap an <c>if</c> condition in this (<c>if (Spmd.Coherent(x &lt; 0f)) ...</c>) to tell
        /// the generator the branch is usually taken by all lanes or none: it then emits an
        /// <c>Any()</c> guard so whole gangs skip a branch no lane takes. At runtime (in the
        /// scalar reference) this is an identity function, behavior is unchanged.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Coherent(bool condition) => condition;

        /// <summary>
        /// Coherent if, ISPC's <c>cif</c>. Skips a branch entirely when no lane takes it,
        /// and (optionally) lets callers detect the all-lanes-uniform fast path.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void If(VMask active, VMask condition, Action<VMask> then, Action<VMask>? @else = null)
        {
            var t = active & condition;
            if (t.Any()) then(t);
            if (@else != null)
            {
                var e = VMask.AndNot(active, condition);
                if (e.Any()) @else(e);
            }
        }
    }
}
