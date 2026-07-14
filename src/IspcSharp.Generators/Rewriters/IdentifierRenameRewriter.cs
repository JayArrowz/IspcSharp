using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Rewriters;

internal sealed class IdentifierRenameRewriter(HashSet<string> names, string prefix) : CSharpSyntaxRewriter
{
    private readonly HashSet<string> _names = names;
    private readonly string _prefix = prefix;

    public override SyntaxNode? VisitIdentifierName(IdentifierNameSyntax node)
        => _names.Contains(node.Identifier.Text)
            ? node.WithIdentifier(SyntaxFactory.Identifier(_prefix + node.Identifier.Text))
            : base.VisitIdentifierName(node);
}