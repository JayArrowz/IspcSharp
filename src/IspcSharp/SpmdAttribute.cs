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
}
