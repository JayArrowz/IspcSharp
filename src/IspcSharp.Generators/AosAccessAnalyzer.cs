using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace IspcSharp.Generators;

/// <summary>
/// Warns about Array-of-Structs access patterns inside SPMD kernels, the single most
/// common way to lose SIMD performance. `points[i].X` forces a gather (one strided load
/// per lane) instead of one contiguous vector load; a Structure-of-Arrays layout
/// (`xs[i]`, `ys[i]`, or SoaFloat2/3) loads each field with a single instruction.
///
/// Fires inside: [Spmd]-annotated method bodies, and lambdas passed to
/// Spmd.Foreach / Spmd.ParallelForeach / Spmd.Foreach2D / Spmd.ParallelForeach2D.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class AosAccessAnalyzer : DiagnosticAnalyzer
{
    public static readonly DiagnosticDescriptor AosAccess = new(
        "ISPC100",
        "Array-of-Structs access in SPMD kernel",
        "'{0}' accesses a field through an indexed element, an AoS pattern that becomes a per-lane gather. Restructure to Structure-of-Arrays (separate float[] per field, or SoaFloat2/SoaFloat3) for contiguous vector loads.",
        "IspcSharp.Performance",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(AosAccess);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(AnalyzeMemberAccess, SyntaxKind.SimpleMemberAccessExpression);
    }

    private static void AnalyzeMemberAccess(SyntaxNodeAnalysisContext ctx)
    {
        var ma = (MemberAccessExpressionSyntax)ctx.Node;

        // Pattern: <element access>.<member>, e.g. points[i].X
        if (ma.Expression is not ElementAccessExpressionSyntax)
            return;

        // Skip method invocations like list[i].ToString() being flagged twice; still flag property/field reads.
        if (ma.Parent is InvocationExpressionSyntax inv && inv.Expression == ma)
            return;

        if (!IsInsideSpmdKernel(ma))
            return;

        ctx.ReportDiagnostic(Diagnostic.Create(AosAccess, ma.GetLocation(), ma.ToString()));
    }

    private static bool IsInsideSpmdKernel(SyntaxNode node)
    {
        for (var n = node.Parent; n != null; n = n.Parent)
        {
            // Inside a [Spmd] method?
            if (n is MethodDeclarationSyntax m &&
                m.AttributeLists.SelectMany(l => l.Attributes)
                 .Any(a => a.Name.ToString() is "Spmd" or "SpmdAttribute" or "IspcSharp.Spmd" or "IspcSharp.SpmdAttribute"))
            {
                return true;
            }

            // Inside a lambda passed to Spmd.Foreach / ParallelForeach / Foreach2D / ParallelForeach2D?
            if (n is AnonymousFunctionExpressionSyntax &&
                n.Parent is ArgumentSyntax arg &&
                arg.Parent?.Parent is InvocationExpressionSyntax call)
            {
                string callee = call.Expression.ToString().Replace(" ", "");
                if (callee.StartsWith("Spmd.Foreach") || callee.StartsWith("Spmd.ParallelForeach") ||
                    callee.StartsWith("IspcSharp.Spmd.Foreach") || callee.StartsWith("IspcSharp.Spmd.ParallelForeach"))
                {
                    return true;
                }
            }
        }

        return false;
    }
}
