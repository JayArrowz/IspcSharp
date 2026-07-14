using System;
using System.Collections.Generic;

namespace IspcSharp
{
    /// <summary>
    /// Marks a method for SPMD vectorization by the IspcSharp.Generators source generator.
    /// The method body must be a single <c>foreach (var i in Spmd.Range(n)) { ... }</c> loop
    /// using the supported C# subset (see README). A vectorized companion method named
    /// <c>{Name}_Simd</c> is generated in the same partial class.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    public sealed class SpmdAttribute : Attribute
    {
        /// <summary>Also generate a {Name}_ParallelSimd variant that splits across cores.</summary>
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

    /// <summary>
    /// Marks a reusable, side-effect-free helper for SPMD vectorization, ISPC's non-<c>export</c>
    /// function. The generator emits a "varying" companion (each <c>float</c> parameter/return becomes
    /// a <c>VFloat</c> lane, each blittable struct its varying form) that any <c>[Spmd]</c> kernel or
    /// other <c>[SpmdFunction]</c> can call. The body uses the same supported subset as a kernel, but
    /// takes scalar values instead of buffers and has no <c>Spmd.Range</c> loop, it operates on the
    /// gang it is handed. Recursion and buffer parameters are not allowed.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    public sealed class SpmdFunctionAttribute : Attribute
    {
    }

    /// <summary>
    /// Marks a blittable struct as usable inside <c>[Spmd]</c>/<c>[SpmdFunction]</c> bodies. Fields are
    /// <c>float</c>/<c>int</c>/<c>double</c>/<c>long</c>, or ISPC-style fixed-size array members declared
    /// with <see cref="SpmdArrayAttribute"/>. The generator emits a varying companion whose fields are
    /// the per-lane gang types (<c>VFloat</c>/<c>VInt</c>/…, one gang per array element), so the struct
    /// can be a kernel local, a helper argument/return, or a Structure-of-Arrays buffer element accessed
    /// as <c>buf[i].field</c> (buffers require same-width scalar fields and no array members).
    /// </summary>
    [AttributeUsage(AttributeTargets.Struct, AllowMultiple = false, Inherited = false)]
    public sealed class SpmdStructAttribute : Attribute
    {
    }

    /// <summary>
    /// Declares a <c>float[]</c>/<c>int[]</c>/<c>double[]</c>/<c>long[]</c> field of a <c>[SpmdStruct]</c>
    /// as an ISPC-style <b>fixed-size array member</b> of the given length. The varying companion expands
    /// it into that many independent gangs held in registers (Structure-of-Arrays), so element access
    /// <c>s.field[k]</c> resolves to a single gang. The index <c>k</c> must be a compile-time integer
    /// literal (registers have no runtime-indexed lane), and such a struct is a local / helper argument /
    /// return only, not a buffer element. Example: <c>[SpmdArray(4)] public float[] Weights;</c>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    public sealed class SpmdArrayAttribute : Attribute
    {
        /// <summary>Fixed element count of the array member (must be &gt; 0).</summary>
        public int Length { get; }
        public SpmdArrayAttribute(int length) { Length = length; }
    }

    /// <summary>One per-element mismatch found by <see cref="KernelVerifier"/>.</summary>
    public readonly struct Mismatch
    {
        public readonly int Index;
        public readonly float Expected;
        public readonly float Actual;
        public Mismatch(int index, float expected, float actual)
        {
            Index = index; Expected = expected; Actual = actual;
        }
        public override string ToString() => $"[{Index}] expected {Expected:G9}, got {Actual:G9}";
    }

    /// <summary>
    /// Correctness harness: SIMD/masking bugs are silent (wrong numbers, not exceptions),
    /// so every kernel should be checked against a scalar reference implementation.
    /// </summary>
    public static class KernelVerifier
    {
        /// <summary>
        /// Compare a vectorized kernel's output against a scalar reference over the same input.
        /// </summary>
        /// <param name="count">Element count. Use values that exercise the tail: e.g. LaneCount*3 + 1.</param>
        /// <param name="scalarReference">Writes expected results: (index) => expected value.</param>
        /// <param name="runKernel">Runs the vectorized kernel, filling <paramref name="count"/> outputs into the provided buffer.</param>
        /// <param name="tolerance">Absolute+relative tolerance (polynomial transcendentals won't match Math.* bit-exactly).</param>
        public static IReadOnlyList<Mismatch> Verify(
            int count,
            Func<int, float> scalarReference,
            Action<float[]> runKernel,
            float tolerance = 1e-5f)
        {
            var expected = new float[count];
            for (int i = 0; i < count; i++) expected[i] = scalarReference(i);

            var actual = new float[count];
            runKernel(actual);

            var mismatches = new List<Mismatch>();
            for (int i = 0; i < count; i++)
            {
                float e = expected[i], a = actual[i];
                float allowed = tolerance * Math.Max(1f, Math.Abs(e));
                bool bothNaN = float.IsNaN(e) && float.IsNaN(a);
                if (!bothNaN && Math.Abs(e - a) > allowed)
                    mismatches.Add(new Mismatch(i, e, a));
            }
            return mismatches;
        }

        /// <summary>Throws with a readable report if any element mismatches.</summary>
        public static void AssertMatches(
            int count,
            Func<int, float> scalarReference,
            Action<float[]> runKernel,
            float tolerance = 1e-5f,
            int maxReported = 10)
        {
            var mismatches = Verify(count, scalarReference, runKernel, tolerance);
            if (mismatches.Count == 0) return;

            var report = new System.Text.StringBuilder();
            report.AppendLine($"Kernel verification failed: {mismatches.Count}/{count} elements mismatch (LaneCount={Spmd.LaneCount}).");
            for (int i = 0; i < Math.Min(maxReported, mismatches.Count); i++)
                report.AppendLine("  " + mismatches[i]);
            if (mismatches.Count > maxReported)
                report.AppendLine($"  ... and {mismatches.Count - maxReported} more");
            report.AppendLine("Tip: mismatches clustered at the end of the buffer usually mean a missing tail mask (use StoreMasked/ctx.Active).");
            throw new InvalidOperationException(report.ToString());
        }
    }
}
