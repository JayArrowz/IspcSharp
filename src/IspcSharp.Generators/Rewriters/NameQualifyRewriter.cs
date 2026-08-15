using System;
using System.Collections.Generic;
using IspcSharp.Generators.Models;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Rewriters;

/// <summary>
/// Rewrites the cross-type names a kernel reads into their fully qualified form:
/// constants (<c>Scale</c> → <c>global::VRandom.Dist.Scale</c>) and qualified
/// <c>[SpmdFunction]</c> calls (<c>Bits.Draw(i)</c> → <c>global::Rng.Bits.Draw(i)</c>).
///
/// The companion is emitted into its own file with only <c>System</c> and <c>IspcSharp</c>
/// imported, and large parts of the method are re-emitted as ordinary scalar C# (pre-loop
/// locals, uniform scaffolding, the scalar tail, uniform shift counts and index offsets). A
/// name the source file reached through a <c>using</c> directive would not bind there, so the
/// whole method is rewritten once up front and every downstream path inherits the fix.
///
/// Applied only on the generator's re-parsed (detached) copy of the method — never on the
/// analyzer's live syntax, whose locations the ISPC diagnostics point at.
/// </summary>
internal sealed class NameQualifyRewriter(
    IReadOnlyDictionary<string, ConstInfo> consts,
    IReadOnlyDictionary<string, FunctionInfo> functions) : CSharpSyntaxRewriter
{
    private readonly IReadOnlyDictionary<string, ConstInfo> _consts = consts;
    private readonly IReadOnlyDictionary<string, FunctionInfo> _functions = functions;

    public static MethodDeclarationSyntax Apply(
        MethodDeclarationSyntax m,
        IReadOnlyDictionary<string, ConstInfo> consts,
        IReadOnlyDictionary<string, FunctionInfo> functions)
        => (MethodDeclarationSyntax)new NameQualifyRewriter(consts, functions).Visit(m);

    public override SyntaxNode? VisitIdentifierName(IdentifierNameSyntax node)
        => QualifyConst(node) ?? base.VisitIdentifierName(node);

    public override SyntaxNode? VisitMemberAccessExpression(MemberAccessExpressionSyntax node)
        => QualifyConst(node) ?? base.VisitMemberAccessExpression(node);

    /// <summary>
    /// A qualified call to a helper in another type. Only the callee is replaced; the arguments
    /// are still visited, so constants inside them are qualified too. A bare call is left alone,
    /// it names a helper in this same type, whose companion lands in the same partial class.
    /// </summary>
    public override SyntaxNode? VisitInvocationExpression(InvocationExpressionSyntax node)
    {
        if (node.Expression is MemberAccessExpressionSyntax ma &&
            _functions.TryGetValue(ma.ToString().Replace(" ", ""), out var fn) &&
            fn.QualifiedName != ma.ToString())
        {
            var args = (ArgumentListSyntax)Visit(node.ArgumentList);
            return node
                .WithExpression(SyntaxFactory.ParseExpression(fn.QualifiedName).WithTriviaFrom(ma))
                .WithArgumentList(args);
        }

        return base.VisitInvocationExpression(node);
    }

    private SyntaxNode? QualifyConst(ExpressionSyntax node)
    {
        // Names that designate a member ('a.Member', 'new S { Member = … }', named arguments)
        // are not value reads and must survive verbatim.
        if (ConstScan.IsMemberDesignator(node))
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
