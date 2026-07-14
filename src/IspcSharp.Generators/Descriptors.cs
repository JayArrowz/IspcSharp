using Microsoft.CodeAnalysis;

namespace IspcSharp.Generators;

internal static class Descriptors
{
    internal static readonly DiagnosticDescriptor NotPartial = new(
        "ISPC001", "Containing type must be partial",
        "[Spmd] method '{0}': the containing type must be declared 'partial' so generated code can be added",
        "IspcSharp", DiagnosticSeverity.Error, true);

    internal static readonly DiagnosticDescriptor BadShape = new(
        "ISPC002", "Unsupported [Spmd] method shape",
        "[Spmd] method '{0}': body must be [optional float/int/double locals], one 'foreach (var i in Spmd.Range(...))' or 'foreach (var (x, y) in Spmd.Range2D(...))' loop, then [optional trailing statements]",
        "IspcSharp", DiagnosticSeverity.Error, true);

    internal static readonly DiagnosticDescriptor Unsupported = new(
        "ISPC003", "Construct not vectorizable",
        "[Spmd] method '{0}': {1} is not supported by the SPMD vectorizer (v0.2 subset). Rewrite it, or use the IspcSharp runtime API directly for full control.",
        "IspcSharp", DiagnosticSeverity.Error, true);

    internal static readonly DiagnosticDescriptor BadParam = new(
        "ISPC004", "Unsupported parameter type",
        "[Spmd] method '{0}': parameter '{1}' has unsupported type '{2}'. Supported: float[]/int[]/double[], Span<T>/ReadOnlySpan<T> of float/int/double, and uniform float/int/double.",
        "IspcSharp", DiagnosticSeverity.Error, true);

    internal static readonly DiagnosticDescriptor NoParallel = new(
        "ISPC005", "Parallel variant skipped",
        "[Spmd] method '{0}': {0}_ParallelSimd was not generated: {1}",
        "IspcSharp", DiagnosticSeverity.Info, true);

    internal static readonly DiagnosticDescriptor GatherPerf = new(
        "ISPC101", "Gather (non-contiguous load) in SPMD kernel",
        "'{0}' uses a lane-varying index, lowered to a per-lane gather (Memory.Gather), not a contiguous vector load. Make it contiguous (loop-variable/affine index, a transposed or SoA layout, or presorted indices) to avoid the gather.",
        "IspcSharp.Performance", DiagnosticSeverity.Warning, true);

    internal static readonly DiagnosticDescriptor ScatterPerf = new(
        "ISPC102", "Scatter (non-contiguous store) in SPMD kernel",
        "'{0}' is a lane-varying indexed store, lowered to a per-active-lane scatter (Memory.Scatter); .NET has no hardware scatter instruction. Restructure to a contiguous write where possible.",
        "IspcSharp.Performance", DiagnosticSeverity.Warning, true);

    internal static readonly DiagnosticDescriptor IntDividePerf = new(
        "ISPC103", "Per-lane integer divide in SPMD kernel",
        "integer '{0}' has no SIMD instruction and runs as a per-lane scalar loop. Expect scalar-ish throughput here; if the divisor is a constant power of two use a shift/mask instead.",
        "IspcSharp.Performance", DiagnosticSeverity.Warning, true);

    internal static readonly DiagnosticDescriptor DoubleConvertPerf = new(
        "ISPC104", "Double↔integer conversion in SPMD kernel",
        "'{0}' converts between double and 64-bit integer lanes ((int)/(long) cast), which has no encoding before AVX-512DQ and runs per-lane on AVX2 (Zen 1–3, Haswell–Comet Lake). Keep the value in double, or move the conversion out of the hot loop.",
        "IspcSharp.Performance", DiagnosticSeverity.Warning, true);

    internal static readonly DiagnosticDescriptor AosAccess = new(
        "ISPC100",
        "Array-of-Structs access in SPMD kernel",
        "'{0}' accesses a field through an indexed element, an AoS pattern that becomes a per-lane gather. Restructure to Structure-of-Arrays (separate float[] per field, or SoaFloat2/SoaFloat3) for contiguous vector loads.",
        "IspcSharp.Performance",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);
}
