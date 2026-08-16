using System;

namespace IspcSharp;

/// <summary>
/// Marks a method for SPMD vectorization by the IspcSharp.Generators source generator.
/// The method body must be a single <c>foreach (var i in Spmd.Range(n)) { ... }</c> loop
/// using the supported C# subset (see README). A vectorized companion method named
/// <c>{Name}_Simd</c> is generated in the same partial class.
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SpmdAttribute : Attribute
{
    /// <summary>
    /// Also generate a {Name}_ParallelSimd variant that splits across cores.
    /// </summary>
    public bool GenerateParallel { get; set; } = true;

    /// <summary>
    /// Emit non-temporal (streaming, <c>movntps</c>) stores for the contiguous buffer writes, plus a
    /// store fence after the loop. This bypasses the cache, a win only for large, write-once output
    /// that won't be re-read soon (it avoids read-for-ownership traffic), and a loss if the data is
    /// reused. NT stores need an aligned destination; the runtime falls back to an ordinary store
    /// when a buffer isn't aligned, so it is always safe but only speeds up aligned buffers (large
    /// arrays, where streaming helps, are commonly page-aligned in practice). Applies to float/double
    /// output buffers. Off by default.
    /// </summary>
    public bool Streaming { get; set; } = false;

    /// <summary>
    /// Force the vectorized reduction loop's unroll factor (1-4) instead of letting the generator
    /// choose. 0, the default, means choose automatically.
    ///
    /// Unrolling gives each copy its own accumulator set, which breaks the accumulator dependency
    /// chain — worth it when that chain is the bottleneck, a loss when the copies exhaust the
    /// vector registers. The automatic choice budgets for the 16 registers x86 has below AVX-512
    /// and charges a <c>double</c>/<c>long</c> accumulator two registers (it is a VDouble2/VLong2
    /// pair). On AVX-512, where there are 32, a higher factor may pay: the unroll is baked into
    /// the generated source, but the register count is only known when the JIT runs, so the
    /// generator cannot detect that for you. Measure before setting this.
    /// </summary>
    public int Unroll { get; set; } = 0;
}
