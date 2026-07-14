using System;
using System.Collections.Generic;

namespace IspcSharp;

/// <summary>
/// One per-element mismatch found by <see cref="KernelVerifier"/>.
/// </summary>
public readonly struct Mismatch(int index, float expected, float actual)
{
    public readonly int Index = index;
    public readonly float Expected = expected;
    public readonly float Actual = actual;

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
        float[] expected = new float[count];
        for (int i = 0; i < count; i++)
            expected[i] = scalarReference(i);

        float[] actual = new float[count];
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

    /// <summary>
    /// Throws with a readable report if any element mismatches.
    /// </summary>
    public static void AssertMatches(
        int count,
        Func<int, float> scalarReference,
        Action<float[]> runKernel,
        float tolerance = 1e-5f,
        int maxReported = 10)
    {
        var mismatches = Verify(count, scalarReference, runKernel, tolerance);
        if (mismatches.Count == 0)
            return;

        var report = new System.Text.StringBuilder();
        _ = report.AppendLine($"Kernel verification failed: {mismatches.Count}/{count} elements mismatch (LaneCount={Spmd.LaneCount}).");
        for (int i = 0; i < Math.Min(maxReported, mismatches.Count); i++)
            _ = report.AppendLine("  " + mismatches[i]);
        if (mismatches.Count > maxReported)
            _ = report.AppendLine($"  ... and {mismatches.Count - maxReported} more");
        _ = report.AppendLine("Tip: mismatches clustered at the end of the buffer usually mean a missing tail mask (use StoreMasked/ctx.Active).");
        throw new InvalidOperationException(report.ToString());
    }
}
