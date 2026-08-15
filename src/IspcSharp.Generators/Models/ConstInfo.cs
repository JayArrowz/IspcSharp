using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Models;

/// <summary>
/// A named scalar constant a kernel body reads: a <c>const</c> field, or a
/// <c>static readonly</c> one (uniform for the whole run either way, so both broadcast).
/// </summary>
/// <param name="Text">The reference exactly as written in the kernel, whitespace stripped
/// (<c>Scale</c>, <c>Dist.Scale</c>, <c>Tuning.Gain</c>).</param>
/// <param name="Code">The fully qualified reference (<c>global::VRandom.Dist.Scale</c>). The
/// generated companion lives in a different file with only <c>System</c>/<c>IspcSharp</c>
/// imported, so it can't rely on the source file's <c>using</c> directives.</param>
/// <param name="Kind">The lane kind the constant broadcasts to (byte/short widen to int,
/// matching C#'s own promotion).</param>
internal sealed record ConstInfo(string Text, string Code, Kind Kind);

/// <summary>
/// Finds the constants a <c>[Spmd]</c>/<c>[SpmdFunction]</c> body reads.
///
/// This is the one place the generator uses the semantic model rather than syntax: constness,
/// the field's type, and its fully qualified name can't be read off a bare identifier. Binding
/// them here (once, into an equatable model) means a constant declared in another part of the
/// partial class, in a base class, or in a shared <c>Constants</c> holder all resolve the same
/// way, and a local that shadows a constant is correctly left alone.
/// </summary>
internal static class ConstScan
{
    /// <summary>Fully qualified, containing type included: <c>global::Ns.Type.Member</c>.</summary>
    private static readonly SymbolDisplayFormat FullName = new(
        globalNamespaceStyle: SymbolDisplayGlobalNamespaceStyle.Included,
        typeQualificationStyle: SymbolDisplayTypeQualificationStyle.NameAndContainingTypesAndNamespaces,
        memberOptions: SymbolDisplayMemberOptions.IncludeContainingType,
        miscellaneousOptions: SymbolDisplayMiscellaneousOptions.EscapeKeywordIdentifiers);

    public static EquatableReadOnlyList<ConstInfo> From(MethodDeclarationSyntax m, SemanticModel? model)
    {
        if (model is null)
            return new EquatableReadOnlyList<ConstInfo>([]);

        var found = new Dictionary<string, ConstInfo>(StringComparer.Ordinal);
        // A name that ever binds to something else in this method (a local shadowing the
        // constant, an overload set, a type) is dropped: substituting it by text would be wrong.
        var ambiguous = new HashSet<string>(StringComparer.Ordinal);

        foreach (var node in m.DescendantNodes())
        {
            if (node is not (IdentifierNameSyntax or MemberAccessExpressionSyntax))
                continue;
            // The '.Member' half of 'a.Member' is not a reference of its own; the whole
            // member access is the node that binds to the field.
            if (node.Parent is MemberAccessExpressionSyntax parent && parent.Name == node)
                continue;

            string text = Normalize(node.ToString());
            var info = AsConstant(model.GetSymbolInfo(node).Symbol, text);
            if (info is null)
            {
                _ = ambiguous.Add(text);
                continue;
            }

            if (found.TryGetValue(text, out var prior) && prior.Code != info.Code)
                _ = ambiguous.Add(text);
            else
                found[text] = info;
        }

        var kept = found.Values
            .Where(c => !ambiguous.Contains(c.Text))
            .OrderBy(c => c.Text, StringComparer.Ordinal)
            .ToList();
        return new EquatableReadOnlyList<ConstInfo>(kept);
    }

    private static ConstInfo? AsConstant(ISymbol? symbol, string text)
    {
        if (symbol is not IFieldSymbol f)
            return null;
        if (!f.IsConst && !(f.IsStatic && f.IsReadOnly))
            return null;
        var kind = KindOf(f.Type.SpecialType);
        if (kind is null)
            return null;
        return new ConstInfo(text, f.ToDisplayString(FullName), kind.Value);
    }

    /// <summary>
    /// Lane kind of a constant's type. Narrow (byte/short) constants widen to int lanes, the
    /// same promotion C# applies to them in every expression.
    /// </summary>
    private static Kind? KindOf(SpecialType t) => t switch
    {
        SpecialType.System_Single => Kind.F,
        SpecialType.System_Int32 => Kind.I,
        SpecialType.System_Double => Kind.D,
        SpecialType.System_Int64 => Kind.L,
        SpecialType.System_Byte or SpecialType.System_SByte or
        SpecialType.System_Int16 or SpecialType.System_UInt16 => Kind.I,
        _ => null,
    };

    /// <summary>
    /// Whitespace-stripped node text, so <c>Dist . Scale</c> and <c>Dist.Scale</c> key alike
    /// (matching how the emitter compares callee names).
    /// </summary>
    public static string Normalize(string s)
        => s.Replace(" ", "").Replace("\t", "").Replace("\r", "").Replace("\n", "");
}
