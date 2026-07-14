using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Rewriters;

internal sealed class ReturnToGotoRewriter : CSharpSyntaxRewriter
{
    public override SyntaxNode? VisitReturnStatement(ReturnStatementSyntax node)
        => node.Expression == null
            ? SyntaxFactory.GotoStatement(
                SyntaxKind.GotoStatement,
                SyntaxFactory.IdentifierName("__tail_next"))
            : base.VisitReturnStatement(node);
}
