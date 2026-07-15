using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using IspcSharp.Generators.Contexts;
using IspcSharp.Generators.Exceptions;
using IspcSharp.Generators.Models;
using IspcSharp.Generators.Rewriters;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace IspcSharp.Generators;

/// <summary>
/// Rewrites [Spmd] methods into vectorized companions using IspcSharp's SIMD types.
///
/// Supports a progressively expanding subset of C#, including:
/// - Float, int, double, and long SIMD operations
/// - 1D and 2D range loops (including tiled iteration)
/// - Arithmetic, casts, branching, while loops, and reductions
/// - Parallel reduction kernels
/// - SIMD-friendly structs and reusable helper functions
/// - Scalar return values and trailing scalar code
///
/// Methods must follow the supported SPMD loop structure and language subset.
/// </summary>
[Generator(LanguageNames.CSharp)]
public sealed class SpmdGenerator : IIncrementalGenerator
{
    private const string AttributeFullName = "IspcSharp.SpmdAttribute";
    private const string StructAttributeFullName = "IspcSharp.SpmdStructAttribute";
    private const string FunctionAttributeFullName = "IspcSharp.SpmdFunctionAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Blittable [SpmdStruct] structs (parsed to name + primitive fields).
        var structs = context.SyntaxProvider.ForAttributeWithMetadataName(
            StructAttributeFullName,
            predicate: static (node, _) => node is StructDeclarationSyntax,
            transform: static (ctx, _) => StructInfo.From((StructDeclarationSyntax)ctx.TargetNode))
            .Where(static s => s is not null)
            .Select(static (s, _) => s!)
            .WithTrackingName("SpmdStructs");

        // [SpmdFunction] helpers.
        var functions = context.SyntaxProvider.ForAttributeWithMetadataName(
            FunctionAttributeFullName,
            predicate: static (node, _) => node is MethodDeclarationSyntax,
            transform: static (ctx, _) => FunctionInfo.From((MethodDeclarationSyntax)ctx.TargetNode))
            .WithTrackingName("SpmdFunctions");

        // [Spmd] kernels.
        var kernels = context.SyntaxProvider.ForAttributeWithMetadataName(
            AttributeFullName,
            predicate: static (node, _) => node is MethodDeclarationSyntax,
            transform: static (ctx, _) => KernelInfo.From((MethodDeclarationSyntax)ctx.TargetNode))
            .WithTrackingName("SpmdKernels");

        var structTable = structs.Collect()
            .Select(static (arr, _) => new EquatableReadOnlyList<StructInfo>(
                [.. arr.OrderBy(s => s.Name, StringComparer.Ordinal)]))
            .WithTrackingName("SpmdStructTable");
        var functionTable = functions.Collect()
            .Select(static (arr, _) => new EquatableReadOnlyList<FunctionInfo>(
                [.. arr.OrderBy(f => f.Name, StringComparer.Ordinal)]))
            .WithTrackingName("SpmdFunctionTable");

        var tables = structTable.Combine(functionTable);

        // Varying companions for the blittable structs (one source file for all of them).
        context.RegisterSourceOutput(structTable, static (spc, all) =>
        {
            string? src = EmitStructCompanions(all);
            if (src != null)
                spc.AddSource("__SpmdStructs.g.cs", SourceText.From(src, Encoding.UTF8));
        });

        context.RegisterSourceOutput(tables, static (spc, pair) =>
        {
            var structMap = BuildStructMap(pair.Left);
            var functions = DistinctFunctions(pair.Right);
            foreach (var fn in functions)
            {
                try
                {
                    var method = ParseMethod(fn.DeclarationText);
                    if (method is null)
                        continue;
                    var fnScope = ScopedFunctionMap(functions, fn.Namespace, fn.TypeName);
                    string? src = GenerateFunctionSource(fn, method, structMap, fnScope, DropDiagnostic);
                    if (src != null)
                        spc.AddSource(HintName(fn.Namespace, fn.TypeName, fn.Name, "_SpmdFn.g.cs"), SourceText.From(src, Encoding.UTF8));
                }
                catch (UnsupportedConstructException)
                {
                    // Reported (with a real location) by SpmdDiagnosticsAnalyzer.
                }
            }
        });

        // [Spmd] kernels, with access to the struct/function tables.
        context.RegisterSourceOutput(kernels.Combine(tables), static (spc, pair) =>
        {
            var kernel = pair.Left;
            try
            {
                var method = ParseMethod(kernel.MethodText);
                if (method is null)
                    return;
                var structMap = BuildStructMap(pair.Right.Left);
                var fnMap = ScopedFunctionMap(pair.Right.Right, kernel.Namespace, kernel.TypeName);
                string? src = GenerateKernelSource(kernel, method, structMap, fnMap, DropDiagnostic);
                if (src != null)
                    spc.AddSource(HintName(kernel.Namespace, kernel.TypeName, kernel.Name, "_Spmd.g.cs"), SourceText.From(src, Encoding.UTF8));
            }
            catch (UnsupportedConstructException)
            {
                // Reported (with a real location) by SpmdDiagnosticsAnalyzer.
            }
        });
    }

    /// <summary>
    /// Diagnostic sink for generator-side runs: the analyzer owns reporting, so the
    /// generation pass discards what the engine surfaces.
    /// </summary>
    private static void DropDiagnostic(Diagnostic _)
    {
    }

    /// <summary>
    /// Fully qualified hint name (Namespace.Type.Member + suffix), so kernels with the
    /// same method name in different types/namespaces never collide on the generated
    /// file name (a hintName collision aborts the entire generator pass).
    /// </summary>
    private static string HintName(string ns, string typeName, string member, string suffix)
        => $"{(ns.Length > 0 ? ns + "." : "")}{(typeName.Length > 0 ? typeName + "." : "")}{member}{suffix}";

    /// <summary>
    /// Re-parse a pipeline model's declaration text back to a method node for emission.
    /// </summary>
    private static MethodDeclarationSyntax? ParseMethod(string declarationText)
        => SyntaxFactory.ParseMemberDeclaration(declarationText) as MethodDeclarationSyntax;

    /// <summary>
    /// Read a <c>bool</c> named argument from the method's <c>[Spmd(...)]</c> attribute.
    /// </summary>
    private static bool GetSpmdBoolArg(MethodDeclarationSyntax method, string argName)
    {
        foreach (var list in method.AttributeLists)
        {
            foreach (var attr in list.Attributes)
            {
                string n = attr.Name.ToString();
                int dot = n.LastIndexOf('.');
                if (dot >= 0)
                    n = n.Substring(dot + 1);
                if (n is not ("Spmd" or "SpmdAttribute") || attr.ArgumentList == null)
                    continue;
                foreach (var arg in attr.ArgumentList.Arguments)
                {
                    if (arg.NameEquals?.Name.Identifier.Text == argName)
                        return arg.Expression.IsKind(SyntaxKind.TrueLiteralExpression);
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Syntactic attribute check (matches "Name" or "NameAttribute").
    /// </summary>
    internal static bool HasAttribute(SyntaxList<AttributeListSyntax> lists, string name)
    {
        foreach (var list in lists)
        {
            foreach (var attr in list.Attributes)
            {
                string n = attr.Name.ToString();
                int dot = n.LastIndexOf('.');
                if (dot >= 0)
                    n = n.Substring(dot + 1);
                if (n == name || n == name + "Attribute")
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// The shared kernel engine: validates the [Spmd] method and emits its vectorized
    /// companion source, surfacing diagnostics through <paramref name="report"/>. The
    /// generator calls this with a discarding sink; SpmdDiagnosticsAnalyzer calls it with
    /// the real method syntax and reports what it surfaces.
    /// </summary>
    internal static string? GenerateKernelSource(KernelInfo kernel, MethodDeclarationSyntax method,
        Dictionary<string, StructInfo> structMap, Dictionary<string, FunctionInfo> fnMap,
        Action<Diagnostic> report)
    {
        string name = kernel.Name;
        var location = method.GetLocation();

        if (!kernel.TypeIsPartial)
        {
            report(Diagnostic.Create(Descriptors.NotPartial, location, name));
            return null;
        }

        var paramInfos = new List<ParamInfo>();
        foreach (var p in method.ParameterList.Parameters)
        {
            string t = p.Type?.ToString().Replace(" ", "") ?? "";
            var kind = t switch
            {
                "float[]" => ParamKind.FloatArray,
                "Span<float>" or "System.Span<float>" => ParamKind.FloatSpan,
                "ReadOnlySpan<float>" or "System.ReadOnlySpan<float>" => ParamKind.FloatReadOnlySpan,
                "int[]" => ParamKind.IntArray,
                "Span<int>" or "System.Span<int>" => ParamKind.IntSpan,
                "ReadOnlySpan<int>" or "System.ReadOnlySpan<int>" => ParamKind.IntReadOnlySpan,
                "double[]" => ParamKind.DoubleArray,
                "Span<double>" or "System.Span<double>" => ParamKind.DoubleSpan,
                "ReadOnlySpan<double>" or "System.ReadOnlySpan<double>" => ParamKind.DoubleReadOnlySpan,
                "long[]" => ParamKind.LongArray,
                "Span<long>" or "System.Span<long>" => ParamKind.LongSpan,
                "ReadOnlySpan<long>" or "System.ReadOnlySpan<long>" => ParamKind.LongReadOnlySpan,
                "byte[]" => ParamKind.ByteArray,
                "Span<byte>" or "System.Span<byte>" => ParamKind.ByteSpan,
                "ReadOnlySpan<byte>" or "System.ReadOnlySpan<byte>" => ParamKind.ByteReadOnlySpan,
                "short[]" => ParamKind.ShortArray,
                "Span<short>" or "System.Span<short>" => ParamKind.ShortSpan,
                "ReadOnlySpan<short>" or "System.ReadOnlySpan<short>" => ParamKind.ShortReadOnlySpan,
                "float[,]" => ParamKind.FloatArray2D,
                "int[,]" => ParamKind.IntArray2D,
                "double[,]" => ParamKind.DoubleArray2D,
                "long[,]" => ParamKind.LongArray2D,
                "float" => ParamKind.UniformFloat,
                "int" => ParamKind.UniformInt,
                "double" => ParamKind.UniformDouble,
                "long" => ParamKind.UniformLong,
                "byte" => ParamKind.UniformByte,
                "short" => ParamKind.UniformShort,
                _ => ParamKind.Unsupported,
            };
            // Blittable [SpmdStruct] buffer: 'S[]' where S has a uniform field type (flat SoA view).
            string? structElem = null;
            if (kind == ParamKind.Unsupported && t.EndsWith("[]") &&
                structMap.TryGetValue(t.Substring(0, t.Length - 2), out var bufStruct))
            {
                // Fixed-size array members expand to per-element gangs held in registers; there is
                // no flat buffer layout for them, so such structs are locals / args / returns only.
                if (bufStruct.HasArrayField)
                {
                    throw new UnsupportedConstructException(
                        $"struct buffer '{bufStruct.Name}[]' has fixed-size array member fields (array-member structs are locals, helper args, or returns only; use separate SoA arrays for a buffer)", p.GetLocation());
                }

                if (bufStruct.HasNarrowField)
                {
                    throw new UnsupportedConstructException(
                        $"struct buffer '{bufStruct.Name}[]' has byte/short fields (narrow-field structs are locals, uniform params, helper args, or returns only; use byte[]/short[] SoA buffers)", p.GetLocation());
                }

                // A mixed-field struct[] buffer is allowed when every field is 32-bit (float/int):
                // the AoS array casts to one 4-byte element size, and each field is gathered/
                // scattered through its own correctly-typed flat view. Mixing 32- and 64-bit
                // fields would break the element stride, so those stay locals/args only.
                if (!bufStruct.AllSameKind && !bufStruct.AllFields32Bit)
                {
                    throw new UnsupportedConstructException(
                        $"struct buffer '{bufStruct.Name}[]' mixes field widths; a mixed-field buffer needs all-32-bit (float/int) fields (make every field the same type, or use separate SoA arrays)", p.GetLocation());
                }

                kind = ParamKind.StructArray;
                structElem = bufStruct.Name;
            }

            if (kind == ParamKind.Unsupported && structMap.TryGetValue(t, out var uniformStruct))
            {
                kind = ParamKind.UniformStruct;
                structElem = uniformStruct.Name;
            }

            if (kind == ParamKind.Unsupported)
            {
                report(Diagnostic.Create(Descriptors.BadParam, p.GetLocation(), name, p.Identifier.Text, t));
                return null;
            }

            paramInfos.Add(new ParamInfo(p.Identifier.Text, kind, p.Type!.ToString(), structElem));
        }

        if (method.Body is null)
        {
            report(Diagnostic.Create(Descriptors.BadShape, location, name));
            return null;
        }

        // Find the single Spmd.Range* loop, it may be nested inside uniform control flow
        // (a for/while/if scaffold), as in FFT stages or a matmul's row/col loops.
        var spmdForeaches = method.Body.DescendantNodes().OfType<CommonForEachStatementSyntax>()
            .Where(f => f.Expression is InvocationExpressionSyntax inv && IsSpmdRangeCallee(inv))
            .ToList();
        if (spmdForeaches.Count == 0)
        {
            report(Diagnostic.Create(Descriptors.BadShape, location, name));
            return null;
        }

        if (spmdForeaches.Count > 1)
        {
            throw new UnsupportedConstructException(
                "more than one Spmd.Range loop in a kernel (split into separate [Spmd] methods)",
                spmdForeaches[1].GetLocation());
        }

        var fe = spmdForeaches[0];
        var rangeCall = (InvocationExpressionSyntax)fe.Expression;

        // Simple shape: foreach at the method-body top level, preceded only by local decls.
        // Anything else (foreach nested in for/while/if) is the "scaffold" shape.
        bool simpleShape = fe.Parent == method.Body &&
            method.Body.Statements.TakeWhile(s => s != fe).All(s => s is LocalDeclarationStatementSyntax);

        var pre = new List<LocalDeclarationStatementSyntax>();
        var post = new List<StatementSyntax>();
        if (simpleShape)
        {
            bool seenFe = false;
            foreach (var st in method.Body.Statements)
            {
                if (st == fe)
                {
                    seenFe = true;
                    continue;
                }

                if (!seenFe)
                    pre.Add((LocalDeclarationStatementSyntax)st);
                else
                    post.Add(st);
            }
        }

        string rangeCallee = rangeCall.Expression.ToString().Replace(" ", "");
        string loopVar, rowVar = "", startExpr = "0", endExpr = "", widthExpr = "", heightExpr = "";
        string tileWExpr = "64", tileHExpr = "64";
        bool is2D, isTiled = false;

        if (fe is ForEachStatementSyntax fe1 &&
            rangeCallee is "Spmd.Range" or "IspcSharp.Spmd.Range" &&
            rangeCall.ArgumentList.Arguments.Count is 1 or 2)
        {
            is2D = false;
            loopVar = fe1.Identifier.Text;
            startExpr = rangeCall.ArgumentList.Arguments.Count == 2
                ? rangeCall.ArgumentList.Arguments[0].ToString() : "0";
            endExpr = rangeCall.ArgumentList.Arguments.Count == 2
                ? rangeCall.ArgumentList.Arguments[1].ToString()
                : rangeCall.ArgumentList.Arguments[0].ToString();
        }
        else if (fe is ForEachVariableStatementSyntax fe2 &&
                 ((rangeCallee is "Spmd.Range2D" or "IspcSharp.Spmd.Range2D" &&
                  rangeCall.ArgumentList.Arguments.Count == 2)
                  ||
                  (rangeCallee is "Spmd.Range2DTiled" or "IspcSharp.Spmd.Range2DTiled" &&
                  rangeCall.ArgumentList.Arguments.Count is 2 or 4)) &&
                 fe2.Variable is DeclarationExpressionSyntax
                 {
                     Designation: ParenthesizedVariableDesignationSyntax { Variables.Count: 2 } pvd
                 } &&
                 pvd.Variables[0] is SingleVariableDesignationSyntax xd &&
                 pvd.Variables[1] is SingleVariableDesignationSyntax yd)
        {
            // foreach (var (x, y) in Spmd.Range2D[Tiled](width, height[, tileW, tileH])):
            // x is the gang (lane) variable along rows; y is a uniform row index.
            is2D = true;
            isTiled = rangeCallee.EndsWith("Tiled");
            loopVar = xd.Identifier.Text;
            rowVar = yd.Identifier.Text;
            widthExpr = rangeCall.ArgumentList.Arguments[0].ToString();
            heightExpr = rangeCall.ArgumentList.Arguments[1].ToString();
            if (rangeCall.ArgumentList.Arguments.Count == 4)
            {
                tileWExpr = rangeCall.ArgumentList.Arguments[2].ToString();
                tileHExpr = rangeCall.ArgumentList.Arguments[3].ToString();
            }
        }
        else
        {
            report(Diagnostic.Create(Descriptors.BadShape, location, name));
            return null;
        }

        string returnType = method.ReturnType.ToString();
        // A [SpmdStruct] return is built after the loop from reduced scalar accumulators
        // (e.g. 'return new Stats { Sum = sum, Min = mn };'), the trailing statements run as
        // ordinary scalar C#, so the companion returns a plain struct just like ISPC's export.
        if (returnType is not ("void" or "float" or "int" or "double" or "long") &&
            !structMap.ContainsKey(returnType))
        {
            throw new UnsupportedConstructException(
                $"return type '{returnType}' (supported: void, float, int, double, long, or a [SpmdStruct])", location);
        }

        var body = fe.Statement is BlockSyntax b ? b.Statements : SyntaxFactory.List([fe.Statement]);

        // The scaffold shape only wraps a 1-D Spmd.Range loop (2-D foreach must be top level).
        if (!simpleShape && is2D)
        {
            throw new UnsupportedConstructException(
                "a 2-D Spmd.Range2D loop nested inside other control flow (use it at the method-body top level)",
                fe.GetLocation());
        }

        // Lane mode: a kernel with double BUFFERS runs as a 64-bit float gang (VDouble/VMaskD);
        // a kernel with long BUFFERS runs as a 64-bit integer gang (VLong/VMaskD). Both are
        // "wide" (half the lane count of float/int). Float/int kernels stay 32-bit; double
        // intermediates there become VDouble2 pairs.
        bool doubleMode = paramInfos.Any(p => p.ElemKind == Kind.D && p.IsBuffer);
        bool longMode = paramInfos.Any(p => p.ElemKind == Kind.L && p.IsBuffer);
        if (doubleMode && paramInfos.Any(p => p.IsBuffer && p.ElemKind != Kind.D))
        {
            throw new UnsupportedConstructException(
                "mixing double buffers with other buffer types (64-bit gangs have a different lane count; use separate kernels)",
                location);
        }

        if (longMode && paramInfos.Any(p => p.IsBuffer && p.ElemKind != Kind.L))
        {
            throw new UnsupportedConstructException(
                "mixing long buffers with other buffer types (64-bit gangs have a different lane count; use separate kernels)",
                location);
        }

        // In-scope uniform locals + candidate reduction accumulators. Simple shape: the
        // pre-loop local declarations. Scaffold shape: every local/for-variable declared
        // outside the foreach body (loop counters, lengths, accumulators declared before it).
        var preLocals = new Dictionary<string, Kind>();
        if (simpleShape)
        {
            foreach (var d in pre)
            {
                string t = d.Declaration.Type.ToString();
                var k = t switch
                {
                    "float" => Kind.F,
                    "int" => Kind.I,
                    "double" => Kind.D,
                    "long" => Kind.L,
                    "byte" or "sbyte" or "short" or "ushort" => Kind.I,
                    _ => throw new UnsupportedConstructException($"pre-loop local of type '{t}' (only float/int/double/long/byte/short)", d.GetLocation()),
                };
                foreach (var v in d.Declaration.Variables)
                    preLocals[v.Identifier.Text] = k;
            }
        }
        else
        {
            foreach (var kv in CollectScaffoldUniforms(fe, method.Body))
                preLocals[kv.Key] = kv.Value;
        }

        // The 2D row variable is a uniform int inside the body.
        if (is2D)
            preLocals[rowVar] = Kind.I;

        var reductions = FindReductions(body, preLocals);
        if (is2D)
            _ = reductions.RemoveAll(r => r.Name == rowVar);
        if (doubleMode && reductions.Any(r => r.LaneKind != Kind.D))
        {
            throw new UnsupportedConstructException(
                "float/int reduction accumulators in a double kernel (declare the accumulator as double)",
                location);
        }

        if (longMode && reductions.Any(r => r.LaneKind != Kind.L))
        {
            throw new UnsupportedConstructException(
                "float/int reduction accumulators in a long kernel (declare the accumulator as long)",
                location);
        }

        var uniformPreLocals = new HashSet<string>(
            preLocals.Keys.Where(n => reductions.All(r => r.Name != n)));

        string laneCountExpr = doubleMode ? "VDouble.LaneCount" : longMode ? "VLong.LaneCount" : "VFloat.LaneCount";
        int unroll = UnrollFactor(body, reductions);
        bool streaming = GetSpmdBoolArg(method, "Streaming");

        var emitter = new VectorBodyEmitter(loopVar, paramInfos, uniformPreLocals, preLocals, reductions, doubleMode, longMode, structMap, fnMap, streaming);
        string vectorBody = emitter.EmitStatements(body, maskExpr: null, indent: "            ");
        foreach (var diag in emitter.Diagnostics)
            report(diag);

        // Per-lane 'return' retires one element. In the scalar tail that must advance to
        // the NEXT element (not exit the method), so bare returns become 'goto __tail_next;'
        // with the label at the end of the tail iteration.
        bool hasLaneReturns = body.Any(st => st.DescendantNodesAndSelf()
            .OfType<ReturnStatementSyntax>().Any(r => r.Expression == null));
        var tailBody = hasLaneReturns ? RewriteReturnsToGoto(body) : body;
        // Address the scalar tail through the same flat SoA span the vector loop uses, so both
        // paths index identically (a[row,k] → __flat_a[row*__cols_a + k]) rather than the tail
        // falling back to bounds-checked 2-D indexing.
        if (paramInfos.Any(p => p.Is2D))
            tailBody = Rewrite2DToFlat(tailBody, paramInfos);
        string scalarBody = Reindent(tailBody, "            ");

        bool hasSpanParams = paramInfos.Any(p => p.IsSpan);
        string paramDecl = string.Join(", ", paramInfos.Select(p => $"{p.TypeText} {p.Name}"));
        string paramPass = string.Join(", ", paramInfos.Select(p => p.Name));
        string ns = kernel.Namespace;
        string typeHeader = kernel.TypeHeader;

        var src = new StringBuilder();
        _ = src.AppendLine("// <auto-generated by IspcSharp.Generators, do not edit>");
        _ = src.AppendLine("#pragma warning disable 1591, 0419   // generated companions carry no full xmldoc");
        _ = src.AppendLine("using System;");
        _ = src.AppendLine("using IspcSharp;");
        _ = src.AppendLine();
        if (ns.Length > 0)
        {
            _ = src.AppendLine($"namespace {ns}");
            _ = src.AppendLine("{");
        }

        _ = src.AppendLine(typeHeader);
        _ = src.AppendLine("{");

        _ = src.AppendLine($"    /// <summary>Vectorized version of <see cref=\"{name}\"/> (generated).</summary>");
        _ = src.AppendLine($"    public static {returnType} {name}_Simd({paramDecl})");
        _ = src.AppendLine("    {");
        EmitFlatSpanSetup(src, "        ", paramInfos, structMap);
        if (!simpleShape)
        {
            // Scaffold: emit the whole method body, uniform statements verbatim, with the
            // one Spmd.Range loop replaced by its vectorized SIMD loop + scalar tail.
            var ctx = new ScaffoldContext(fe, loopVar, startExpr, endExpr,
                vectorBody, scalarBody, reductions, laneCountExpr, doubleMode, longMode, unroll, hasLaneReturns);
            EmitScaffold(src, method.Body.Statements, "        ", ctx);
        }
        else
        {
            foreach (var d in pre)
                _ = src.AppendLine("        " + d.NormalizeWhitespace().ToFullString().TrimEnd());
            if (is2D && isTiled)
            {
                // Cache-blocked traversal (ISPC's foreach_tiled): walk tile by tile so the
                // working set of row-crossing kernels stays in cache.
                _ = src.AppendLine($"        int __width = {widthExpr};");
                _ = src.AppendLine($"        int __height = {heightExpr};");
                _ = src.AppendLine($"        int __tw = {tileWExpr};");
                _ = src.AppendLine($"        int __th = {tileHExpr};");
                _ = src.AppendLine("        for (int __ty = 0; __ty < __height; __ty += __th)");
                _ = src.AppendLine("        {");
                _ = src.AppendLine("            int __yEnd = System.Math.Min(__ty + __th, __height);");
                _ = src.AppendLine("            for (int __tx = 0; __tx < __width; __tx += __tw)");
                _ = src.AppendLine("            {");
                _ = src.AppendLine("                int __xEnd = System.Math.Min(__tx + __tw, __width);");
                _ = src.AppendLine("                for (int __y = __ty; __y < __yEnd; __y++)");
                _ = src.AppendLine("                {");
                _ = src.AppendLine($"                    int {rowVar} = __y;");
                EmitLoopCore(src, "                    ", "__tx", "__xEnd", loopVar, vectorBody, scalarBody, reductions, laneCountExpr, doubleMode, longMode, unroll: unroll, hasLaneReturns: hasLaneReturns);
                _ = src.AppendLine("                }");
                _ = src.AppendLine("            }");
                _ = src.AppendLine("        }");
            }
            else if (is2D)
            {
                _ = src.AppendLine($"        int __width = {widthExpr};");
                _ = src.AppendLine($"        int __height = {heightExpr};");
                _ = src.AppendLine("        for (int __y = 0; __y < __height; __y++)");
                _ = src.AppendLine("        {");
                _ = src.AppendLine($"            int {rowVar} = __y;");
                EmitLoopCore(src, "            ", "0", "__width", loopVar, vectorBody, scalarBody, reductions, laneCountExpr, doubleMode, longMode, unroll: unroll, hasLaneReturns: hasLaneReturns);
                _ = src.AppendLine("        }");
            }
            else
            {
                EmitLoopCore(src, "        ", startExpr, endExpr, loopVar, vectorBody, scalarBody, reductions, laneCountExpr, doubleMode, longMode, unroll: unroll, hasLaneReturns: hasLaneReturns);
            }

            foreach (var st in post)
                _ = src.AppendLine("        " + st.NormalizeWhitespace().ToFullString().TrimEnd());
        }

        if (streaming)
            _ = src.AppendLine("        Memory.StoreFence();   // make the non-temporal stores globally visible");
        _ = src.AppendLine("    }");

        bool wantParallel = WantsParallel(method);
        if (!simpleShape)
        {
            report(Diagnostic.Create(Descriptors.NoParallel, location, name,
                "the vectorized loop is nested in uniform control flow; only _Simd is generated (parallelize the outer loop yourself)"));
        }
        else if (paramInfos.Any(p => p.Is2D))
        {
            report(Diagnostic.Create(Descriptors.NoParallel, location, name,
                "2D-array buffers use a flat-span view that cannot be captured across threads; only _Simd is generated"));
        }
        else if (paramInfos.Any(p => p.IsStructBuffer))
        {
            report(Diagnostic.Create(Descriptors.NoParallel, location, name,
                "struct buffers use a flat-span view that cannot be captured across threads; only _Simd is generated (parallelize the outer loop yourself)"));
        }
        else if (wantParallel && hasSpanParams)
        {
            report(Diagnostic.Create(Descriptors.NoParallel, location, name,
                "Span parameters cannot be captured across threads; use float[]/int[]"));
        }
        else if (wantParallel)
        {
            // Reductions get per-chunk (thread-local) partial accumulators, combined
            // deterministically after the parallel loop. The chunk body accumulates into
            // renamed '__loc_' locals so the captured outer accumulator is never raced on.
            string scalarBodyParallel = reductions.Count > 0
                ? Reindent(RenameIdentifiers(tailBody, reductions.Select(r => r.Name), "__loc_"), "            ")
                : scalarBody;

            string fallback = returnType == "void"
                ? $"{{ {name}_Simd({paramPass}); return; }}"
                : $"{{ return {name}_Simd({paramPass}); }}";

            _ = src.AppendLine();
            _ = src.AppendLine($"    /// <summary>Multi-core + SIMD version of <see cref=\"{name}\"/> (generated).</summary>");
            _ = src.AppendLine($"    public static {returnType} {name}_ParallelSimd({paramDecl}, int minChunkSize = 16384)");
            _ = src.AppendLine("    {");
            foreach (var d in pre)
                _ = src.AppendLine("        " + d.NormalizeWhitespace().ToFullString().TrimEnd());
            if (is2D && isTiled)
            {
                // Tiled 2D: parallelize over whole tiles (cache-blocked per core).
                // Reductions get one partial slot per tile.
                _ = src.AppendLine($"        int __width0 = {widthExpr};");
                _ = src.AppendLine($"        int __height0 = {heightExpr};");
                _ = src.AppendLine($"        int __tw0 = {tileWExpr};");
                _ = src.AppendLine($"        int __th0 = {tileHExpr};");
                _ = src.AppendLine("        if ((long)__width0 * __height0 <= minChunkSize) " + fallback);
                _ = src.AppendLine("        int __tilesX = (__width0 + __tw0 - 1) / __tw0;");
                _ = src.AppendLine("        int __tilesY = (__height0 + __th0 - 1) / __th0;");
                _ = src.AppendLine("        int __numTiles = __tilesX * __tilesY;");
                foreach (var r in reductions)
                    _ = src.AppendLine($"        {ScalarType(r)}[] __part_{r.Name} = new {ScalarType(r)}[__numTiles];");
                _ = src.AppendLine("        System.Threading.Tasks.Parallel.For(0, __numTiles, __t =>");
                _ = src.AppendLine("        {");
                _ = src.AppendLine("            int __tx = (__t % __tilesX) * __tw0;");
                _ = src.AppendLine("            int __ty = (__t / __tilesX) * __th0;");
                _ = src.AppendLine("            int __xEnd = System.Math.Min(__tx + __tw0, __width0);");
                _ = src.AppendLine("            int __yEnd = System.Math.Min(__ty + __th0, __height0);");
                foreach (var r in reductions)
                    _ = src.AppendLine($"            {ScalarType(r)} __loc_{r.Name} = {ScalarIdentity(r)};");
                _ = src.AppendLine("            for (int __y = __ty; __y < __yEnd; __y++)");
                _ = src.AppendLine("            {");
                _ = src.AppendLine($"                int {rowVar} = __y;");
                EmitLoopCore(src, "                ", "__tx", "__xEnd", loopVar, vectorBody, scalarBodyParallel,
                    reductions, laneCountExpr, doubleMode, longMode, unroll: unroll, redTargetPrefix: "__loc_", hasLaneReturns: hasLaneReturns);
                _ = src.AppendLine("            }");
                foreach (var r in reductions)
                    _ = src.AppendLine($"            __part_{r.Name}[__t] = __loc_{r.Name};");
                _ = src.AppendLine("        });");
                foreach (var r in reductions)
                {
                    _ = src.AppendLine("        for (int __c = 0; __c < __numTiles; __c++)");
                    _ = src.AppendLine(CombinePartial(r));
                }
            }
            else if (is2D)
            {
                // 2D: parallelize over rows; each row is an x-gang loop. Reductions get
                // one partial slot per row.
                _ = src.AppendLine($"        int __width0 = {widthExpr};");
                _ = src.AppendLine($"        int __height0 = {heightExpr};");
                _ = src.AppendLine("        if ((long)__width0 * __height0 <= minChunkSize) " + fallback);
                foreach (var r in reductions)
                    _ = src.AppendLine($"        {ScalarType(r)}[] __part_{r.Name} = new {ScalarType(r)}[__height0];");
                _ = src.AppendLine("        System.Threading.Tasks.Parallel.For(0, __height0, __y =>");
                _ = src.AppendLine("        {");
                _ = src.AppendLine($"            int {rowVar} = __y;");
                foreach (var r in reductions)
                    _ = src.AppendLine($"            {ScalarType(r)} __loc_{r.Name} = {ScalarIdentity(r)};");
                EmitLoopCore(src, "            ", "0", "__width0", loopVar, vectorBody, scalarBodyParallel,
                    reductions, laneCountExpr, doubleMode, longMode, unroll: unroll, redTargetPrefix: "__loc_", hasLaneReturns: hasLaneReturns);
                foreach (var r in reductions)
                    _ = src.AppendLine($"            __part_{r.Name}[__y] = __loc_{r.Name};");
                _ = src.AppendLine("        });");
                foreach (var r in reductions)
                {
                    _ = src.AppendLine("        for (int __c = 0; __c < __height0; __c++)");
                    _ = src.AppendLine(CombinePartial(r));
                }
            }
            else
            {
                _ = src.AppendLine($"        int __start0 = {startExpr};");
                _ = src.AppendLine($"        int __end0 = {endExpr};");
                _ = src.AppendLine("        int __total = __end0 - __start0;");
                _ = src.AppendLine("        if (__total <= minChunkSize) " + fallback);
                _ = src.AppendLine($"        int __w0 = {laneCountExpr};");
                _ = src.AppendLine("        int __chunk = System.Math.Max(__w0, (minChunkSize / __w0) * __w0);");
                _ = src.AppendLine("        int __numChunks = (__total + __chunk - 1) / __chunk;");
                foreach (var r in reductions)
                    _ = src.AppendLine($"        {ScalarType(r)}[] __part_{r.Name} = new {ScalarType(r)}[__numChunks];");
                _ = src.AppendLine("        System.Threading.Tasks.Parallel.For(0, __numChunks, __c =>");
                _ = src.AppendLine("        {");
                _ = src.AppendLine("            int __cs = __start0 + __c * __chunk;");
                _ = src.AppendLine("            int __ce = System.Math.Min(__cs + __chunk, __end0);");
                foreach (var r in reductions)
                    _ = src.AppendLine($"            {ScalarType(r)} __loc_{r.Name} = {ScalarIdentity(r)};");
                EmitLoopCore(src, "            ", "__cs", "__ce", loopVar, vectorBody, scalarBodyParallel,
                    reductions, laneCountExpr, doubleMode, longMode, unroll: unroll, redTargetPrefix: "__loc_", hasLaneReturns: hasLaneReturns);
                foreach (var r in reductions)
                    _ = src.AppendLine($"            __part_{r.Name}[__c] = __loc_{r.Name};");
                _ = src.AppendLine("        });");
                foreach (var r in reductions)
                {
                    _ = src.AppendLine("        for (int __c = 0; __c < __numChunks; __c++)");
                    _ = src.AppendLine(CombinePartial(r));
                }
            }

            foreach (var st in post)
                _ = src.AppendLine("        " + st.NormalizeWhitespace().ToFullString().TrimEnd());
            if (streaming)
                _ = src.AppendLine("        Memory.StoreFence();   // NT stores (workers self-fence at Parallel.For join)");
            _ = src.AppendLine("    }");
        }

        _ = src.AppendLine("}");
        if (ns.Length > 0)
            _ = src.AppendLine("}");

        return src.ToString();
    }

    private static void EmitLoopCore(StringBuilder src, string ind, string startExpr, string endExpr,
        string loopVar, string vectorBody, string scalarBody, List<ReductionInfo> reductions,
        string laneCountExpr, bool doubleMode, bool longMode, int unroll = 1, string redTargetPrefix = "", bool hasLaneReturns = false)
    {
        // The body strings are generated at a fixed base indent; each append below re-indents them to
        // one level under the enclosing brace at this loop's actual nesting depth (so braces line up).
        int bodyIndent = ind.Length + 4;

        _ = src.AppendLine($"{ind}int __start = {startExpr};");
        _ = src.AppendLine($"{ind}int __end = {endExpr};");
        _ = src.AppendLine($"{ind}int __w = {laneCountExpr};");
        _ = src.AppendLine($"{ind}int __i = __start;");

        if (unroll > 1 && reductions.Count > 0)
        {
            // Multiple-accumulator unroll: 'unroll' independent accumulators per reduction break the
            // FMA/add dependency chain (a single accumulator is latency-bound; independent ones fill
            // the pipeline). Each unrolled copy processes the next gang into its own accumulator.
            for (int k = 0; k < unroll; k++)
            {
                foreach (var r in reductions)
                    _ = src.AppendLine($"{ind}{VType(r.LaneKind, doubleMode, longMode)} __red_{r.Name}_{k} = {VectorIdentity(r, doubleMode, longMode)};");
            }

            _ = src.AppendLine($"{ind}for (; __i <= __end - {unroll} * __w; __i += {unroll} * __w)");
            _ = src.AppendLine($"{ind}{{");
            for (int k = 0; k < unroll; k++)
            {
                _ = src.AppendLine($"{ind}    {{");   // brace-scope so replicated locals don't collide
                _ = src.Append(Reindent(SubstituteUnroll(vectorBody, k, reductions), bodyIndent + 4));
                _ = src.AppendLine($"{ind}    }}");
            }

            _ = src.AppendLine($"{ind}}}");

            // Remainder: leftover full gangs fold into accumulator 0.
            _ = src.AppendLine($"{ind}for (; __i <= __end - __w; __i += __w)");
            _ = src.AppendLine($"{ind}{{");
            _ = src.Append(Reindent(SubstituteUnroll(vectorBody, 0, reductions), bodyIndent));
            _ = src.AppendLine($"{ind}}}");

            foreach (var r in reductions)
            {
                for (int k = 1; k < unroll; k++)
                {
                    _ = src.AppendLine(r.Op switch
                    {
                        ReduceOp.Min => $"{ind}__red_{r.Name}_0 = VectorMath.Min(__red_{r.Name}_0, __red_{r.Name}_{k});",
                        ReduceOp.Max => $"{ind}__red_{r.Name}_0 = VectorMath.Max(__red_{r.Name}_0, __red_{r.Name}_{k});",
                        _ => $"{ind}__red_{r.Name}_0 = __red_{r.Name}_0 + __red_{r.Name}_{k};",
                    });
                }

                EmitFinalReduce(src, ind, r, $"__red_{r.Name}_0", redTargetPrefix);
            }
        }
        else
        {
            foreach (var r in reductions)
                _ = src.AppendLine($"{ind}{VType(r.LaneKind, doubleMode, longMode)} __red_{r.Name} = {VectorIdentity(r, doubleMode, longMode)};");
            _ = src.AppendLine($"{ind}for (; __i <= __end - __w; __i += __w)");
            _ = src.AppendLine($"{ind}{{");
            _ = src.Append(Reindent(vectorBody, bodyIndent));
            _ = src.AppendLine($"{ind}}}");
            foreach (var r in reductions)
                EmitFinalReduce(src, ind, r, $"__red_{r.Name}", redTargetPrefix);
        }

        _ = src.AppendLine($"{ind}// Scalar tail (original body; per-lane returns advance to the next element).");
        _ = src.AppendLine($"{ind}for (; __i < __end; __i++)");
        _ = src.AppendLine($"{ind}{{");
        _ = src.AppendLine($"{ind}    int {loopVar} = __i;");
        _ = src.Append(Reindent(scalarBody, bodyIndent));
        if (hasLaneReturns)
            _ = src.AppendLine($"{ind}    __tail_next: ;");
        _ = src.AppendLine($"{ind}}}");
    }

    /// <summary>
    /// Emit the horizontal reduce that folds a vector accumulator into the scalar target.
    /// </summary>
    private static void EmitFinalReduce(StringBuilder src, string ind, ReductionInfo r, string acc, string redTargetPrefix)
    {
        string target = redTargetPrefix + r.Name;
        _ = src.AppendLine(r.Op switch
        {
            ReduceOp.Min => $"{ind}{target} = Math.Min({target}, Reduce.Min({acc}));",
            ReduceOp.Max => $"{ind}{target} = Math.Max({target}, Reduce.Max({acc}));",
            _ => $"{ind}{target} += Reduce.Add({acc});",
        });
    }

    /// <summary>
    /// Produce unroll copy <paramref name="k"/> of a vectorized reduction body: advance the
    /// gang base by k widths and route each accumulator to its k-th copy.
    /// </summary>
    private static string SubstituteUnroll(string body, int k, List<ReductionInfo> reductions)
    {
        string s = body;
        if (k > 0)
            s = System.Text.RegularExpressions.Regex.Replace(s, @"\b__i\b", $"(__i + {k} * __w)");
        foreach (var r in reductions)
        {
            s = System.Text.RegularExpressions.Regex.Replace(
                s, $@"\b__red_{System.Text.RegularExpressions.Regex.Escape(r.Name)}\b", $"__red_{r.Name}_{k}");
        }

        return s;
    }

    /// <summary>
    /// Unroll factor for the vectorized loop: 4 for a straight-line reduction body (breaks the
    /// accumulator dependency chain), else 1. Anything with nested control flow or per-lane exits is
    /// left un-unrolled, replicating masks/branches isn't worth the complexity.
    /// </summary>
    private static int UnrollFactor(SyntaxList<StatementSyntax> body, List<ReductionInfo> reductions)
    {
        if (reductions.Count == 0)
            return 1;
        foreach (var st in body)
        {
            foreach (var node in st.DescendantNodesAndSelf())
            {
                if (node is IfStatementSyntax or WhileStatementSyntax or ForStatementSyntax or
                         DoStatementSyntax or CommonForEachStatementSyntax or SwitchStatementSyntax or
                         BreakStatementSyntax or ContinueStatementSyntax or ReturnStatementSyntax)
                {
                    return 1;
                }
            }
        }

        return 4;
    }

    /// <summary>
    /// The lane type of a scalar C# type in a float/int gang (structs → varying companion).
    /// </summary>
    internal static string VaryingTypeOf(string scalarType, IReadOnlyDictionary<string, StructInfo> structs)
    {
        switch (scalarType)
        {
            case "float":
                return "VFloat";
            case "int":
            case "byte":
            case "sbyte":
            case "short":
            case "ushort":
                return "VInt";
            case "double":
                return "VDouble2";
            case "long":
                return "VLong2";
            default:
                if (structs.TryGetValue(scalarType, out var si))
                    return si.VName;
                throw new UnsupportedConstructException($"type '{scalarType}' (only float/int/double/long/byte/short and [SpmdStruct] structs)");
        }
    }

    /// <summary>
    /// Emit the varying companion struct (<c>Name__V</c>) for every blittable struct.
    /// </summary>
    internal static string? EmitStructCompanions(IReadOnlyList<StructInfo> structs)
    {
        if (structs.Count == 0)
            return null;
        var byName = structs.GroupBy(s => s.Name).Select(g => g.First()).ToList();

        var src = new StringBuilder();
        _ = src.AppendLine("// <auto-generated by IspcSharp.Generators, do not edit>");
        _ = src.AppendLine("#pragma warning disable 1591, 0419   // generated companions carry no full xmldoc");
        _ = src.AppendLine("using System;");
        _ = src.AppendLine("using IspcSharp;");
        _ = src.AppendLine();

        foreach (var nsGroup in byName.GroupBy(s => s.Namespace))
        {
            bool hasNs = nsGroup.Key.Length > 0;
            if (hasNs)
            {
                _ = src.AppendLine($"namespace {nsGroup.Key}");
                _ = src.AppendLine("{");
            }

            foreach (var s in nsGroup)
            {
                string vt = s.VName;
                // Array members are flattened to one gang per element (f_0, f_1, …), held SoA in registers.
                var gangs = s.GangFields().ToList();
                _ = src.AppendLine($"    /// <summary>Varying (per-lane) companion of <see cref=\"{s.Name}\"/>, one gang per field.</summary>");
                _ = src.AppendLine($"    public struct {vt}");
                _ = src.AppendLine("    {");
                foreach (var g in gangs)
                    _ = src.AppendLine($"        public {VType(LaneKind(g.Kind), false, false)} {g.Name};");
                // Constructor.
                string ctorParams = string.Join(", ", gangs.Select(g => $"{VType(LaneKind(g.Kind), false, false)} {g.Name}"));
                _ = src.AppendLine($"        public {vt}({ctorParams})");
                _ = src.AppendLine("        {");
                foreach (var g in gangs)
                    _ = src.AppendLine($"            this.{g.Name} = {g.Name};");
                _ = src.AppendLine("        }");
                // Per-lane blend.
                _ = src.AppendLine($"        public static {vt} Select(VMask __m, {vt} __t, {vt} __f)");
                string selArgs = string.Join(", ", gangs.Select(g => $"{VType(LaneKind(g.Kind), false, false)}.Select(__m, __t.{g.Name}, __f.{g.Name})"));
                _ = src.AppendLine($"            => new {vt}({selArgs});");
                _ = src.AppendLine("    }");
                _ = src.AppendLine();
            }

            if (hasNs)
                _ = src.AppendLine("}");
        }

        return src.ToString();
    }

    /// <summary>
    /// Emit the varying companion of one <c>[SpmdFunction]</c> helper. Same dual-caller
    /// contract as <see cref="GenerateKernelSource"/>: the generator discards diagnostics,
    /// the analyzer reports them.
    /// </summary>
    internal static string? GenerateFunctionSource(FunctionInfo fn, MethodDeclarationSyntax method,
        Dictionary<string, StructInfo> structMap, Dictionary<string, FunctionInfo> fnMap,
        Action<Diagnostic> report)
    {
        if (!fn.TypeIsPartial)
        {
            report(Diagnostic.Create(Descriptors.NotPartial, method.GetLocation(), fn.Name));
            return null;
        }

        if (method.Body is null && method.ExpressionBody is null)
            throw new UnsupportedConstructException("[SpmdFunction] with no body", method.GetLocation());

        string ret = fn.ReturnType;
        if (ret == "void")
            throw new UnsupportedConstructException("[SpmdFunction] returning void (helpers must return a value)", method.GetLocation());

        string vret = VaryingTypeOf(ret, structMap);

        var paramLocals = new Dictionary<string, Kind>();
        var paramStructs = new Dictionary<string, string>();
        var sb = new StringBuilder();
        string vparams = string.Join(", ", fn.Parameters.Select(p =>
        {
            if (structMap.ContainsKey(p.Type))
                paramStructs[p.Name] = p.Type;
            else
                paramLocals[p.Name] = KindOfScalar(p.Type, method.GetLocation());
            return $"{VaryingTypeOf(p.Type, structMap)} {p.Name}";
        }));

        var emitter = new VectorBodyEmitter(structMap, fnMap, paramLocals, paramStructs);
        string body = emitter.EmitFunctionBody(method, "            ");
        foreach (var diag in emitter.Diagnostics)
            report(diag);

        string ns = fn.Namespace;
        var src = new StringBuilder();
        _ = src.AppendLine("// <auto-generated by IspcSharp.Generators, do not edit>");
        _ = src.AppendLine("#pragma warning disable 1591, 0419   // generated companions carry no full xmldoc");
        _ = src.AppendLine("using System;");
        _ = src.AppendLine("using IspcSharp;");
        _ = src.AppendLine();
        if (ns.Length > 0)
        {
            _ = src.AppendLine($"namespace {ns}");
            _ = src.AppendLine("{");
        }

        _ = src.AppendLine(fn.TypeHeader);
        _ = src.AppendLine("{");
        _ = src.AppendLine($"    /// <summary>Varying companion of <see cref=\"{fn.Name}\"/> (generated).</summary>");
        _ = src.AppendLine($"    public static {vret} {fn.Name}({vparams})");
        _ = src.AppendLine("    {");
        _ = src.Append(body);
        _ = src.AppendLine("    }");
        _ = src.AppendLine("}");
        if (ns.Length > 0)
            _ = src.AppendLine("}");

        return src.ToString();
    }

    internal static Dictionary<string, StructInfo> BuildStructMap(IEnumerable<StructInfo> structs)
        => structs.GroupBy(s => s.Name).ToDictionary(g => g.Key, g => g.First());

    /// <summary>
    /// All distinct [SpmdFunction] helpers (one per container + name), in deterministic order.
    /// Helpers with the same name in DIFFERENT containers are separate entries — each gets its
    /// own varying companion.
    /// </summary>
    internal static List<FunctionInfo> DistinctFunctions(IEnumerable<FunctionInfo> functions)
        => [.. functions
            .GroupBy(f => (f.Namespace, f.TypeName, f.Name))
            .Select(g => g.First())
            .OrderBy(f => f.Namespace, StringComparer.Ordinal)
            .ThenBy(f => f.TypeName, StringComparer.Ordinal)
            .ThenBy(f => f.Name, StringComparer.Ordinal)];

    /// <summary>
    /// The helpers visible to a method by simple name: those in its own containing type
    /// (kernels call helpers as bare identifiers, which C# resolves within the same type —
    /// partial declarations included). Scoping the lookup this way keeps two same-named
    /// helpers in different classes from resolving against each other's signatures.
    /// </summary>
    internal static Dictionary<string, FunctionInfo> ScopedFunctionMap(
        IEnumerable<FunctionInfo> functions, string ns, string typeName)
        => functions
            .Where(f => f.Namespace == ns && f.TypeName == typeName)
            .GroupBy(f => f.Name)
            .ToDictionary(g => g.Key, g => g.First());

    internal static Kind KindOfScalar(string t, Location loc) => t switch
    {
        "float" => Kind.F,
        "int" => Kind.I,
        "double" => Kind.D,
        "long" => Kind.L,
        // Narrow scalars live as int lanes (C# promotes them to int in every expression).
        "byte" or "sbyte" or "short" or "ushort" => Kind.I,
        _ => throw new UnsupportedConstructException($"type '{t}' (only float/int/double/long/byte/short)", loc),
    };

    /// <summary>
    /// Vector type name for a lane kind. In float/int kernels the wide lanes are
    /// full-int/float-gang-width pairs: double lanes are VDouble2, long lanes VLong2. In their
    /// own 64-bit kernels they're plain VDouble / VLong gangs.
    /// </summary>
    /// <summary>
    /// The register (lane-value) kind of a field/element kind: narrow memory kinds widen
    /// to int lanes, everything else is already a lane kind.
    /// </summary>
    internal static Kind LaneKind(Kind k) => k is Kind.B or Kind.S ? Kind.I : k;

    internal static string VType(Kind k, bool doubleMode, bool longMode) => k switch
    {
        Kind.F => "VFloat",
        Kind.I => "VInt",
        Kind.L => longMode ? "VLong" : "VLong2",
        Kind.B or Kind.S => throw new InvalidOperationException("byte/short are memory-only lane kinds (widened to VInt in registers)"),
        _ => doubleMode ? "VDouble" : "VDouble2",
    };

    /// <summary>
    /// Scalar element type name for a lane kind (the flat-view element type of a buffer).
    /// </summary>
    internal static string ElemTypeName(Kind k) => k switch { Kind.I => "int", Kind.D => "double", Kind.L => "long", Kind.B => "byte", Kind.S => "short", _ => "float" };

    /// <summary>
    /// Flat SoA view name for a struct-buffer field. A homogeneous buffer shares one view
    /// (<c>__flat_buf</c>); a mixed 32-bit buffer gets one correctly-typed view per distinct field
    /// kind (<c>__flat_buf_f</c> / <c>__flat_buf_i</c>) so each field gathers through its real type.
    /// </summary>
    internal static string StructFlatView(string bufName, Kind fieldKind, bool mixed)
        => mixed ? $"__flat_{bufName}_{fieldKind switch { Kind.I => "i", Kind.D => "d", Kind.L => "l", _ => "f" }}"
                 : "__flat_" + bufName;

    /// <summary>
    /// Vector-lane identity element for a reduction accumulator.
    /// </summary>
    internal static string VectorIdentity(ReductionInfo r, bool doubleMode, bool longMode)
    {
        string vt = VType(r.LaneKind, doubleMode, longMode);
        return (r.Op, r.LaneKind) switch
        {
            (ReduceOp.Min, Kind.F) => "new VFloat(float.PositiveInfinity)",
            (ReduceOp.Min, Kind.I) => "new VInt(int.MaxValue)",
            (ReduceOp.Min, Kind.D) => $"new {vt}(double.PositiveInfinity)",
            (ReduceOp.Min, Kind.L) => $"new {vt}(long.MaxValue)",
            (ReduceOp.Max, Kind.F) => "new VFloat(float.NegativeInfinity)",
            (ReduceOp.Max, Kind.I) => "new VInt(int.MinValue)",
            (ReduceOp.Max, Kind.D) => $"new {vt}(double.NegativeInfinity)",
            (ReduceOp.Max, Kind.L) => $"new {vt}(long.MinValue)",
            _ => $"{vt}.Zero",
        };
    }

    /// <summary>
    /// Scalar identity element for a per-chunk partial accumulator.
    /// </summary>
    private static string ScalarIdentity(ReductionInfo r) => (r.Op, r.LaneKind) switch
    {
        (ReduceOp.Min, Kind.F) => "float.PositiveInfinity",
        (ReduceOp.Min, Kind.I) => "int.MaxValue",
        (ReduceOp.Min, Kind.D) => "double.PositiveInfinity",
        (ReduceOp.Min, Kind.L) => "long.MaxValue",
        (ReduceOp.Max, Kind.F) => "float.NegativeInfinity",
        (ReduceOp.Max, Kind.I) => "int.MinValue",
        (ReduceOp.Max, Kind.D) => "double.NegativeInfinity",
        (ReduceOp.Max, Kind.L) => "long.MinValue",
        (_, Kind.F) => "0f",
        (_, Kind.D) => "0d",
        (_, Kind.L) => "0L",
        _ => "0",
    };

    /// <summary>
    /// Scalar element type name for a reduction's partials array.
    /// </summary>
    private static string ScalarType(ReductionInfo r) => r.LaneKind switch
    {
        Kind.F => "float",
        Kind.I => "int",
        Kind.D => "double",
        _ => "long",
    };

    /// <summary>
    /// The per-chunk combine statement for the parallel variant.
    /// </summary>
    private static string CombinePartial(ReductionInfo r) => r.Op switch
    {
        ReduceOp.Min => $"            {r.Name} = Math.Min({r.Name}, __part_{r.Name}[__c]);",
        ReduceOp.Max => $"            {r.Name} = Math.Max({r.Name}, __part_{r.Name}[__c]);",
        _ => $"            {r.Name} += __part_{r.Name}[__c];",
    };

    /// <summary>
    /// A reduction is a pre-loop local that appears in the loop body ONLY as
    /// 'name += expr;' (reduce_add), 'name = Math.Min(name, expr);' (reduce_min),
    /// or 'name = Math.Max(name, expr);' (reduce_max), one operator per variable.
    /// </summary>
    private static List<ReductionInfo> FindReductions(SyntaxList<StatementSyntax> body, Dictionary<string, Kind> preLocals)
    {
        var result = new List<ReductionInfo>();
        foreach (var kv in preLocals)
        {
            ReduceOp? op = null;
            bool mixedOps = false, otherUse = false;

            void Record(ReduceOp seen)
            {
                if (op is { } prev && prev != seen)
                    mixedOps = true;
                op = seen;
            }

            foreach (var st in body)
            {
                foreach (var node in st.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>()
                                       .Where(id => id.Identifier.Text == kv.Key))
                {
                    if (node.Parent is AssignmentExpressionSyntax a && a.Left == node)
                    {
                        if (a.OperatorToken.Text == "+=")
                        {
                            Record(ReduceOp.Add);
                            continue;
                        }

                        if (a.OperatorToken.Text == "=" &&
                            IsMinMaxSelfCall(a.Right, kv.Key, out var mm))
                        {
                            Record(mm);
                            continue;
                        }

                        otherUse = true;
                    }
                    // The accumulator occurrence inside its own Min/Max call
                    // ('name = Math.Min(>name<, expr)') is part of the accumulation.
                    else if (!IsAccumulatorArgOfMinMaxSelfAssign(node, kv.Key))
                    {
                        otherUse = true;
                    }
                }
            }

            if (mixedOps)
            {
                throw new UnsupportedConstructException(
                    $"reduction variable '{kv.Key}' mixes accumulation operators (use only one of '+=', 'Math.Min', 'Math.Max')");
            }

            if (op != null && otherUse)
            {
                throw new UnsupportedConstructException(
                    $"reduction variable '{kv.Key}' is also read inside the loop (accumulators may only appear as '{kv.Key} += ...', '{kv.Key} = Math.Min({kv.Key}, ...)', or '{kv.Key} = Math.Max({kv.Key}, ...)')");
            }

            if (op is { } found)
                result.Add(new ReductionInfo { Name = kv.Key, LaneKind = kv.Value, Op = found });
        }

        return result;
    }

    /// <summary>
    /// Matches 'Math.Min(name, expr)' / 'Math.Max(expr, name)' (Math or MathF, exactly one arg is the bare accumulator).
    /// </summary>
    internal static bool IsMinMaxSelfCall(ExpressionSyntax rhs, string name, out ReduceOp op)
    {
        op = ReduceOp.Add;
        if (rhs is not InvocationExpressionSyntax inv ||
            inv.ArgumentList.Arguments.Count != 2)
        {
            return false;
        }

        switch (inv.Expression.ToString().Replace(" ", ""))
        {
            case "Math.Min" or "MathF.Min":
                op = ReduceOp.Min;
                break;
            case "Math.Max" or "MathF.Max":
                op = ReduceOp.Max;
                break;
            default:
                return false;
        }

        int selfArgs = inv.ArgumentList.Arguments.Count(arg =>
            arg.Expression is IdentifierNameSyntax id && id.Identifier.Text == name);
        return selfArgs == 1;
    }

    private static bool IsAccumulatorArgOfMinMaxSelfAssign(IdentifierNameSyntax node, string name)
    {
        return node.Parent is ArgumentSyntax { Parent: ArgumentListSyntax { Parent: InvocationExpressionSyntax inv } } &&
               inv.Parent is AssignmentExpressionSyntax asg &&
               asg.OperatorToken.Text == "=" &&
               asg.Left is IdentifierNameSyntax lhs && lhs.Identifier.Text == name &&
               IsMinMaxSelfCall(inv, name, out _);
    }

    private static bool WantsParallel(MethodDeclarationSyntax method)
    {
        foreach (var list in method.AttributeLists)
        {
            foreach (var attr in list.Attributes)
            {
                foreach (var arg in attr.ArgumentList?.Arguments ?? default)
                {
                    if (arg.NameEquals?.Name.Identifier.Text == "GenerateParallel" &&
                        arg.Expression is LiteralExpressionSyntax { Token.Text: "false" })
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    private static string Reindent(SyntaxList<StatementSyntax> statements, string indent)
    {
        var sb = new StringBuilder();
        foreach (var s in statements)
        {
            foreach (string? line in s.NormalizeWhitespace().ToFullString().Split('\n'))
                _ = sb.AppendLine(indent + line.TrimEnd('\r'));
        }

        return sb.ToString();
    }

    /// <summary>
    /// Shift a pre-rendered block so its least-indented line sits at
    /// <paramref name="targetIndent"/> spaces, preserving relative nesting (blank lines stay blank).
    /// </summary>
    private static string Reindent(string block, int targetIndent)
    {
        if (string.IsNullOrEmpty(block))
            return block;
        int baseIndent = int.MaxValue;
        using (var reader = new System.IO.StringReader(block))
        {
            for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
            {
                if (line.TrimStart(' ').Length == 0)
                    continue;   // blank line
                baseIndent = Math.Min(baseIndent, line.Length - line.TrimStart(' ').Length);
            }
        }

        if (baseIndent == int.MaxValue)
            return block;   // all blank
        int shift = targetIndent - baseIndent;
        if (shift == 0)
            return block;

        var sb = new StringBuilder();
        using (var reader = new System.IO.StringReader(block))
        {
            for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
            {
                if (line.TrimStart(' ').Length == 0)
                {
                    _ = sb.AppendLine();
                    continue;
                }

                _ = shift > 0
                    ? sb.AppendLine(new string(' ', shift) + line)
                    : sb.AppendLine(line.Substring(Math.Min(-shift, line.Length - line.TrimStart(' ').Length)));
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Rewrite bare 'return;' statements to 'goto __tail_next;' for the scalar tail.
    /// </summary>
    private static SyntaxList<StatementSyntax> RewriteReturnsToGoto(SyntaxList<StatementSyntax> statements)
    {
        var rewriter = new ReturnToGotoRewriter();
        return SyntaxFactory.List(statements.Select(s => (StatementSyntax)rewriter.Visit(s)));
    }

    /// <summary>
    /// Rewrite 2-D array accesses <c>arr[i, j]</c> on the given params to flat-span indexing
    /// <c>__flat_arr[(i) * __cols_arr + (j)]</c>, the same view the vector loop loads through.
    /// </summary>
    private static SyntaxList<StatementSyntax> Rewrite2DToFlat(SyntaxList<StatementSyntax> statements, List<ParamInfo> paramInfos)
    {
        var map = paramInfos.Where(p => p.Is2D).ToDictionary(p => p.Name, p => (p.FlatName, p.ColsName));
        var rewriter = new TwoDToFlatRewriter(map);
        return SyntaxFactory.List(statements.Select(s => (StatementSyntax)rewriter.Visit(s)));
    }

    /// <summary>
    /// Rewrite every occurrence of the given identifiers to '{prefix}{name}'.
    /// </summary>
    private static SyntaxList<StatementSyntax> RenameIdentifiers(
        SyntaxList<StatementSyntax> statements, IEnumerable<string> names, string prefix)
    {
        var rewriter = new IdentifierRenameRewriter([.. names], prefix);
        return SyntaxFactory.List(statements.Select(s => (StatementSyntax)rewriter.Visit(s)));
    }

    private static bool IsSpmdRangeCallee(InvocationExpressionSyntax inv)
    {
        string c = inv.Expression.ToString().Replace(" ", "");
        return c is "Spmd.Range" or "IspcSharp.Spmd.Range"
            or "Spmd.Range2D" or "IspcSharp.Spmd.Range2D"
            or "Spmd.Range2DTiled" or "IspcSharp.Spmd.Range2DTiled";
    }

    /// <summary>
    /// Every uniform local / for-loop variable declared OUTSIDE the foreach body: loop
    /// counters, lengths (e.g. 'int n = a.Length'), and reduction accumulators declared
    /// just before the loop. These become the foreach's visible uniforms.
    /// </summary>
    private static Dictionary<string, Kind> CollectScaffoldUniforms(CommonForEachStatementSyntax fe, BlockSyntax methodBody)
    {
        var result = new Dictionary<string, Kind>();
        var feBody = fe.Statement;
        foreach (var vd in methodBody.DescendantNodes().OfType<VariableDeclaratorSyntax>())
        {
            if (feBody.Span.Contains(vd.Span))
                continue;             // declared inside the loop body
            if (vd.Parent is not VariableDeclarationSyntax decl)
                continue;
            Kind? k = decl.Type.ToString() switch
            {
                "int" => Kind.I,
                "float" => Kind.F,
                "double" => Kind.D,
                "long" => Kind.L,
                "byte" or "sbyte" or "short" or "ushort" => Kind.I,
                _ => null,
            };
            if (k is { } kind)
                result[vd.Identifier.Text] = kind;
        }

        return result;
    }

    /// <summary>
    /// Emit the flat row-major span view + column-count local for each 2-D array param,
    /// and the flat SoA view for each struct buffer param.
    /// </summary>
    private static void EmitFlatSpanSetup(StringBuilder src, string indent, List<ParamInfo> paramInfos,
        IReadOnlyDictionary<string, StructInfo> structs)
    {
        foreach (var p in paramInfos.Where(p => p.Is2D))
        {
            string elem = p.ElemKind switch { Kind.I => "int", Kind.D => "double", Kind.L => "long", _ => "float" };
            _ = src.AppendLine($"{indent}int {p.ColsName} = {p.Name}.GetLength(1);");
            _ = src.AppendLine($"{indent}Span<{elem}> {p.FlatName} = System.Runtime.InteropServices.MemoryMarshal.CreateSpan("
                + $"ref System.Runtime.CompilerServices.Unsafe.As<byte, {elem}>("
                + $"ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference({p.Name})), {p.Name}.Length);");
        }

        foreach (var p in paramInfos.Where(p => p.IsStructBuffer))
        {
            var si = structs[p.StructType!];
            bool mixed = !si.AllSameKind;
            // View the AoS S[] as flat elem[] spans: field f of element e sits at e*FieldCount + f.
            // Homogeneous → one view of the shared field type; mixed 32-bit → one correctly-typed
            // view per distinct field kind (so an int field isn't loaded through a float span).
            foreach (var k in mixed ? si.Fields.Select(f => f.Kind).Distinct() : [si.FieldKind])
            {
                _ = src.AppendLine($"{indent}Span<{ElemTypeName(k)}> {StructFlatView(p.Name, k, mixed)} = "
                    + $"System.Runtime.InteropServices.MemoryMarshal.Cast<{p.StructType}, {ElemTypeName(k)}>({p.Name}.AsSpan());");
            }
        }
    }

    private static void EmitScaffold(StringBuilder src, IEnumerable<StatementSyntax> statements, string indent, ScaffoldContext ctx)
    {
        foreach (var st in statements)
            EmitScaffoldStatement(src, st, indent, ctx);
    }

    private static void EmitScaffoldStatement(StringBuilder src, StatementSyntax st, string indent, ScaffoldContext ctx)
    {
        if (st == ctx.Fe)
        {
            EmitLoopCore(src, indent, ctx.StartExpr, ctx.EndExpr, ctx.LoopVar, ctx.VectorBody,
                ctx.ScalarBody, ctx.Reductions, ctx.LaneCountExpr, ctx.DoubleMode, ctx.LongMode, unroll: ctx.Unroll, hasLaneReturns: ctx.HasLaneReturns);
            return;
        }

        if (!st.DescendantNodes().Contains(ctx.Fe))
        {
            // Pure uniform statement, emit verbatim.
            foreach (string? line in st.NormalizeWhitespace().ToFullString().Split('\n'))
                _ = src.AppendLine(indent + line.TrimEnd('\r'));
            return;
        }

        // A control-flow statement wrapping the foreach: reconstruct the header, recurse.
        switch (st)
        {
            case BlockSyntax blk:
                _ = src.AppendLine($"{indent}{{");
                EmitScaffold(src, blk.Statements, indent + "    ", ctx);
                _ = src.AppendLine($"{indent}}}");
                break;
            case ForStatementSyntax f:
                string inits = f.Declaration != null ? f.Declaration.ToString() : string.Join(", ", f.Initializers);
                string cond = f.Condition?.ToString() ?? "";
                string incs = string.Join(", ", f.Incrementors);
                _ = src.AppendLine($"{indent}for ({inits}; {cond}; {incs})");
                EmitScaffoldBraced(src, f.Statement, indent, ctx);
                break;
            case WhileStatementSyntax w:
                _ = src.AppendLine($"{indent}while ({w.Condition})");
                EmitScaffoldBraced(src, w.Statement, indent, ctx);
                break;
            case IfStatementSyntax ifs:
                _ = src.AppendLine($"{indent}if ({ifs.Condition})");
                EmitScaffoldBraced(src, ifs.Statement, indent, ctx);
                if (ifs.Else != null)
                {
                    _ = src.AppendLine($"{indent}else");
                    EmitScaffoldBraced(src, ifs.Else.Statement, indent, ctx);
                }

                break;
            default:
                throw new UnsupportedConstructException(
                    $"'{ScaffoldTrunc(st)}' wrapping the Spmd.Range loop (only for/while/if/block scaffolding is supported)",
                    st.GetLocation());
        }
    }

    private static void EmitScaffoldBraced(StringBuilder src, StatementSyntax body, string indent, ScaffoldContext ctx)
    {
        _ = src.AppendLine($"{indent}{{");
        if (body is BlockSyntax blk)
            EmitScaffold(src, blk.Statements, indent + "    ", ctx);
        else
            EmitScaffoldStatement(src, body, indent + "    ", ctx);
        _ = src.AppendLine($"{indent}}}");
    }

    private static string ScaffoldTrunc(SyntaxNode n)
    {
        string s = n.ToString();
        return s.Length > 50 ? s.Substring(0, 47) + "..." : s;
    }
}
