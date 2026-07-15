using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Models;

/// <summary>
/// Extracts the container facts a companion needs (type header line, partial-ness, namespace)
/// from a declaration's parents, so the models above never have to hold the syntax tree.
/// </summary>
internal static class DeclarationModel
{
    public static (string TypeHeader, string TypeName, bool TypeIsPartial, string Namespace) ContainerOf(MethodDeclarationSyntax m)
    {
        if (m.Parent is not TypeDeclarationSyntax type)
            return ("", "", false, "");

        bool isPartial = type.Modifiers.Any(Microsoft.CodeAnalysis.CSharp.SyntaxKind.PartialKeyword);
        string header = $"{type.Modifiers} {type.Keyword.Text} {type.Identifier.Text}";

        var typeParts = new List<string> { type.Identifier.Text };
        var parts = new List<string>();
        for (var n = type.Parent; n != null; n = n.Parent)
        {
            if (n is TypeDeclarationSyntax outer)
                typeParts.Insert(0, outer.Identifier.Text);
            if (n is BaseNamespaceDeclarationSyntax nds)
                parts.Insert(0, nds.Name.ToString());
        }

        return (header, string.Join(".", typeParts), isPartial, string.Join(".", parts));
    }
}
