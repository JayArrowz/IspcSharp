using System;
using System.Collections.Generic;
using IspcSharp.Generators.Models;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Rewriters;

/// <summary>
/// Replaces every constant reference in a kernel with its fully qualified form
/// (<c>Scale</c> → <c>global::VRandom.Dist.Scale</c>).
///
/// The companion is emitted into its own file with only <c>System</c> and <c>IspcSharp</c>
/// imported, and large parts of the method are re-emitted as ordinary scalar C# (pre-loop
/// locals, uniform scaffolding, the scalar tail, uniform shift counts and index offsets). A
/// constant reached through the source file's <c>using</c> directives would not bind there, so
/// the whole method is rewritten once up front and every downstream path inherits the fix.
///
/// Applied only on the generator's re-parsed (detached) copy of the method — never on the
/// analyzer's live syntax, whose locations the ISPC diagnostics point at.
/// </summary>
internal sealed class ConstQualifyRewriter(IReadOnlyDictionary<string, ConstInfo> consts) : CSharpSyntaxRewriter
{
    private readonly IReadOnlyDictionary<string, ConstInfo> _consts = consts;

    public static MethodDeclarationSyntax Apply(MethodDeclarationSyntax m, IReadOnlyDictionary<string, ConstInfo> consts)
        => consts.Count == 0
            ? m
            : (MethodDeclarationSyntax)new ConstQualifyRewriter(consts).Visit(m);

    public override SyntaxNode? VisitIdentifierName(IdentifierNameSyntax node)
        => Qualify(node) ?? base.VisitIdentifierName(node);

    public override SyntaxNode? VisitMemberAccessExpression(MemberAccessExpressionSyntax node)
        => Qualify(node) ?? base.VisitMemberAccessExpression(node);

    private SyntaxNode? Qualify(ExpressionSyntax node)
    {
        // The '.Member' half of 'a.Member' names a member, not the reference itself.
        if (node.Parent is MemberAccessExpressionSyntax parent && parent.Name == node)
            return null;
        if (!_consts.TryGetValue(ConstScan.Normalize(node.ToString()), out var info))
            return null;
        // Already qualified (an entry is keyed by both its written and its qualified form).
        if (info.Code == node.ToString())
            return null;
        return SyntaxFactory.ParseExpression(info.Code).WithTriviaFrom(node);
    }

    /// <summary>
    /// The lookup table the emitter and this rewriter share: every constant keyed by both the
    /// name written in the source and its fully qualified form, so a reference resolves before
    /// and after the rewrite (the analyzer runs the engine on unrewritten syntax).
    /// </summary>
    public static Dictionary<string, ConstInfo> BuildMap(IEnumerable<ConstInfo> consts)
    {
        var map = new Dictionary<string, ConstInfo>(StringComparer.Ordinal);
        foreach (var c in consts)
        {
            map[c.Text] = c;
            map[c.Code] = c;
        }

        return map;
    }
}
