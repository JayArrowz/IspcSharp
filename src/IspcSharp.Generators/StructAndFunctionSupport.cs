using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators
{
    /// <summary>One field of a blittable struct.</summary>
    internal sealed class StructField
    {
        public readonly string Name;
        public readonly SpmdGenerator.Kind Kind;
        public StructField(string name, SpmdGenerator.Kind kind) { Name = name; Kind = kind; }
    }

    /// <summary>
    /// A <c>[SpmdStruct]</c> struct: name, namespace, and its primitive fields. Drives generation of
    /// the varying companion (<c>Name__V</c>, one gang-typed field each) and buffer gather/scatter.
    /// </summary>
    internal sealed class StructInfo
    {
        public readonly string Name;
        public readonly string Namespace;
        public readonly List<StructField> Fields;
        public StructInfo(string name, string ns, List<StructField> fields) { Name = name; Namespace = ns; Fields = fields; }

        public string VName => Name + "__V";
        /// <summary>True when every field is the same primitive, the requirement for a struct[] SoA buffer.</summary>
        public bool AllSameKind => Fields.Count > 0 && Fields.All(f => f.Kind == Fields[0].Kind);
        public SpmdGenerator.Kind FieldKind => Fields[0].Kind;

        /// <summary>Parse a <see cref="StructDeclarationSyntax"/>; returns null if any field is non-primitive.</summary>
        public static StructInfo? From(StructDeclarationSyntax s)
        {
            var fields = new List<StructField>();
            foreach (var m in s.Members.OfType<FieldDeclarationSyntax>())
            {
                if (m.Modifiers.Any(mod => mod.IsKind(Microsoft.CodeAnalysis.CSharp.SyntaxKind.StaticKeyword) ||
                                           mod.IsKind(Microsoft.CodeAnalysis.CSharp.SyntaxKind.ConstKeyword)))
                    continue;
                var kind = KindOf(m.Declaration.Type.ToString());
                if (kind is null) return null;   // non-blittable field → not a supported struct
                foreach (var v in m.Declaration.Variables)
                    fields.Add(new StructField(v.Identifier.Text, kind.Value));
            }
            if (fields.Count == 0) return null;
            return new StructInfo(s.Identifier.Text, NamespaceOf(s), fields);
        }

        private static SpmdGenerator.Kind? KindOf(string t) => t switch
        {
            "float" => SpmdGenerator.Kind.F,
            "int" => SpmdGenerator.Kind.I,
            "double" => SpmdGenerator.Kind.D,
            "long" => SpmdGenerator.Kind.L,
            _ => null,
        };

        private static string NamespaceOf(SyntaxNode node)
        {
            for (var n = node.Parent; n != null; n = n.Parent)
            {
                if (n is FileScopedNamespaceDeclarationSyntax fs) return fs.Name.ToString();
                if (n is NamespaceDeclarationSyntax ns) return ns.Name.ToString();
            }
            return "";
        }
    }

    /// <summary>A <c>[SpmdFunction]</c> helper: its declaration syntax and parsed signature.</summary>
    internal sealed class FunctionInfo
    {
        public readonly string Name;
        public readonly MethodDeclarationSyntax Syntax;
        public readonly List<(string Name, string Type)> Parameters;
        public readonly string ReturnType;
        public FunctionInfo(string name, MethodDeclarationSyntax syntax, List<(string, string)> ps, string ret)
        { Name = name; Syntax = syntax; Parameters = ps; ReturnType = ret; }

        public static FunctionInfo From(MethodDeclarationSyntax m)
        {
            var ps = m.ParameterList.Parameters
                .Select(p => (p.Identifier.Text, p.Type?.ToString().Trim() ?? ""))
                .ToList();
            return new FunctionInfo(m.Identifier.Text, m, ps, m.ReturnType.ToString().Trim());
        }
    }
}
