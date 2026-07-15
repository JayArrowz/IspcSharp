using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Models;

/// <summary>
/// A <c>[SpmdFunction]</c> helper: its declaration syntax and parsed signature.
/// </summary>
internal sealed record FunctionInfo(
    string Name,
    string ReturnType,
    EquatableReadOnlyList<(string Name, string Type)> Parameters,
    string DeclarationText,
    string TypeHeader,
    string TypeName,
    bool TypeIsPartial,
    string Namespace)
{
    public static FunctionInfo From(MethodDeclarationSyntax m)
    {
        var ps = m.ParameterList.Parameters
            .Select(p => (p.Identifier.Text, p.Type?.ToString().Trim() ?? ""))
            .ToList();
        var (header, typeName, isPartial, ns) = DeclarationModel.ContainerOf(m);
        return new FunctionInfo(
            m.Identifier.Text,
            m.ReturnType.ToString().Trim(),
            new EquatableReadOnlyList<(string Name, string Type)>(ps),
            m.ToFullString(),
            header,
            typeName,
            isPartial,
            ns);
    }
}
