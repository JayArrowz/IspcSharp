using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Models;

/// <summary>
/// A <c>[SpmdStruct]</c> struct: name, namespace, and its primitive fields. Drives generation of
/// the varying companion (<c>Name__V</c>, one gang-typed field each) and buffer gather/scatter.
/// </summary>
internal sealed record StructInfo(string Name, string Namespace, EquatableReadOnlyList<StructField> Fields)
{
    public string VName => Name + "__V";

    /// <summary>
    /// True when every field is the same primitive (a homogeneous SoA struct[] buffer).
    /// </summary>
    public bool AllSameKind => Fields.Count > 0 && Fields.All(f => f.Kind == Fields[0].Kind);

    /// <summary>
    /// True when every field is 32-bit (float/int). A mixed-field struct may still be a
    /// buffer element in this case: the AoS view casts to one 4-byte element size, and each field
    /// is gathered/scattered through its own correctly-typed flat view. Mixing 32- and 64-bit
    /// fields would break the uniform element stride, so those stay locals/args only.
    /// </summary>
    public bool AllFields32Bit => Fields.Count > 0 &&
        Fields.All(f => f.Kind is Kind.F or Kind.I);

    public Kind FieldKind => Fields[0].Kind;

    /// <summary>
    /// True when any field is an ISPC-style fixed-size array member. Such structs are
    /// locals / helper args / returns only (the SoA-in-registers view has no buffer layout).
    /// </summary>
    public bool HasArrayField => Fields.Any(f => f.IsArray);

    /// <summary>
    /// The companion's flattened gang fields: a scalar field yields one gang, an array
    /// member of length N yields N (<c>f_0 … f_{N-1}</c>). Drives companion emission and blends.
    /// </summary>
    public IEnumerable<(string Name, Kind Kind)> GangFields()
    {
        foreach (var f in Fields)
        {
            if (f.IsArray)
            {
                for (int i = 0; i < f.ArrayLength; i++)
                    yield return (f.GangName(i), f.Kind);
            }
            else
            {
                yield return (f.Name, f.Kind);
            }
        }
    }

    /// <summary>
    /// Parse a <see cref="StructDeclarationSyntax"/>; returns null if any field is non-primitive
    /// (or an array field lacking a valid <c>[SpmdArray(N)]</c>), so the struct isn't registered.
    /// </summary>
    public static StructInfo? From(StructDeclarationSyntax s)
    {
        var fields = new List<StructField>();
        foreach (var m in s.Members.OfType<FieldDeclarationSyntax>())
        {
            if (m.Modifiers.Any(mod => mod.IsKind(Microsoft.CodeAnalysis.CSharp.SyntaxKind.StaticKeyword) ||
                                       mod.IsKind(Microsoft.CodeAnalysis.CSharp.SyntaxKind.ConstKeyword)))
            {
                continue;
            }

            string typeText = m.Declaration.Type.ToString();
            if (typeText.EndsWith("[]"))
            {
                // Fixed-size array member: requires [SpmdArray(N)] and a primitive element type.
                int len = SpmdArrayLength(m);
                if (len <= 0)
                    return null;
                var elemKind = KindOf(typeText.Substring(0, typeText.Length - 2).Trim());
                if (elemKind is null)
                    return null;
                foreach (var v in m.Declaration.Variables)
                    fields.Add(new StructField(v.Identifier.Text, elemKind.Value, len));
                continue;
            }

            var kind = KindOf(typeText);
            if (kind is null)
                return null;   // non-blittable field → not a supported struct
            foreach (var v in m.Declaration.Variables)
                fields.Add(new StructField(v.Identifier.Text, kind.Value));
        }

        if (fields.Count == 0)
            return null;
        return new StructInfo(s.Identifier.Text, NamespaceOf(s), new EquatableReadOnlyList<StructField>(fields));
    }

    /// <summary>
    /// Read the length from a field's <c>[SpmdArray(N)]</c> attribute, or 0 if absent/invalid.
    /// </summary>
    private static int SpmdArrayLength(FieldDeclarationSyntax m)
    {
        foreach (var list in m.AttributeLists)
        {
            foreach (var attr in list.Attributes)
            {
                string n = attr.Name.ToString();
                int dot = n.LastIndexOf('.');
                if (dot >= 0)
                    n = n.Substring(dot + 1);
                if (n is not ("SpmdArray" or "SpmdArrayAttribute"))
                    continue;
                var arg = attr.ArgumentList?.Arguments.FirstOrDefault();
                if (arg?.Expression is LiteralExpressionSyntax lit &&
                    int.TryParse(lit.Token.ValueText, out int len))
                {
                    return len;
                }
            }
        }

        return 0;
    }

    private static Kind? KindOf(string t) => t switch
    {
        "float" => Kind.F,
        "int" => Kind.I,
        "double" => Kind.D,
        "long" => Kind.L,
        _ => null,
    };

    private static string NamespaceOf(SyntaxNode node)
    {
        for (var n = node.Parent; n != null; n = n.Parent)
        {
            if (n is FileScopedNamespaceDeclarationSyntax fs)
                return fs.Name.ToString();
            if (n is NamespaceDeclarationSyntax ns)
                return ns.Name.ToString();
        }

        return "";
    }
}
