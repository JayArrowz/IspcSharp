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
    string Namespace,
    EquatableReadOnlyList<ConstInfo> Consts)
{
    /// <summary>
    /// How a caller in another type must name this helper's varying companion. The companion is
    /// emitted into <c>Namespace.TypeName</c> as a public static method, and the calling
    /// companion is a separate file that imports only System and IspcSharp, so cross-type calls
    /// are emitted fully qualified.
    /// </summary>
    public string QualifiedName
        => $"global::{(Namespace.Length > 0 ? Namespace + "." : "")}{TypeName}.{Name}";

    public static FunctionInfo From(MethodDeclarationSyntax m, SemanticModel? model = null)
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
            ns,
            ConstScan.From(m, model));
    }
}
