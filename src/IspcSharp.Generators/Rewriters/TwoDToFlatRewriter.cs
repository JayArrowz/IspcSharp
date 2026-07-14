using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Rewriters;

internal sealed class TwoDToFlatRewriter(Dictionary<string, (string Flat, string Cols)> map) : CSharpSyntaxRewriter
{
    private readonly Dictionary<string, (string Flat, string Cols)> _map = map;

    public override SyntaxNode? VisitElementAccessExpression(ElementAccessExpressionSyntax node)
    {
        var visited = (ElementAccessExpressionSyntax)base.VisitElementAccessExpression(node)!;
        if (visited.Expression is IdentifierNameSyntax id &&
            _map.TryGetValue(id.Identifier.Text, out var m) &&
            visited.ArgumentList.Arguments.Count == 2)
        {
            var row = visited.ArgumentList.Arguments[0].Expression;
            var col = visited.ArgumentList.Arguments[1].Expression;
            return SyntaxFactory.ParseExpression($"{m.Flat}[({row}) * {m.Cols} + ({col})]")
                .WithTriviaFrom(visited);
        }

        return visited;
    }
}