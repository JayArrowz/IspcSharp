using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Models;

/// <summary>
/// A <c>[Spmd]</c> kernel: the method's source text plus its container info.
/// Same text-not-syntax rule as <see cref="FunctionInfo"/>.
/// </summary>
internal sealed record KernelInfo(
    string Name,
    string MethodText,
    string TypeHeader,
    string TypeName,
    bool TypeIsPartial,
    string Namespace,
    EquatableReadOnlyList<ConstInfo> Consts)
{
    public static KernelInfo From(MethodDeclarationSyntax m, SemanticModel? model = null)
    {
        var (header, typeName, isPartial, ns) = DeclarationModel.ContainerOf(m);
        return new KernelInfo(
            m.Identifier.Text, m.ToFullString(), header, typeName, isPartial, ns,
            ConstScan.From(m, model));
    }
}
