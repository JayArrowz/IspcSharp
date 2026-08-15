using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using IspcSharp.Generators.Exceptions;
using IspcSharp.Generators.Models;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace IspcSharp.Generators.Analyzers;

/// <summary>
/// Reports every [Spmd]/[SpmdFunction] diagnostic (ISPC001–ISPC005 shape/support errors and
/// the ISPC101–ISPC104 performance warnings)
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class SpmdDiagnosticsAnalyzer : DiagnosticAnalyzer
{
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => [
        Descriptors.NotPartial,
        Descriptors.BadShape,
        Descriptors.Unsupported,
        Descriptors.BadParam,
        Descriptors.NoParallel,
        Descriptors.GatherPerf,
        Descriptors.ScatterPerf,
        Descriptors.IntDividePerf,
        Descriptors.DoubleConvertPerf
    ];

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(static compilationContext =>
        {
            var tables = new Lazy<(Dictionary<string, StructInfo> Structs, List<FunctionInfo> Functions)>(
                () => BuildTables(compilationContext.Compilation));

            compilationContext.RegisterSyntaxNodeAction(
                ctx => AnalyzeMethod(ctx, tables),
                SyntaxKind.MethodDeclaration);
        });
    }

    private static void AnalyzeMethod(
        SyntaxNodeAnalysisContext ctx,
        Lazy<(Dictionary<string, StructInfo> Structs, List<FunctionInfo> Functions)> tables)
    {
        var method = (MethodDeclarationSyntax)ctx.Node;
        bool isKernel = SpmdGenerator.HasAttribute(method.AttributeLists, "Spmd");
        bool isFunction = SpmdGenerator.HasAttribute(method.AttributeLists, "SpmdFunction");
        if (!isKernel && !isFunction)
            return;

        var (structMap, functions) = tables.Value;
        try
        {
            if (isKernel)
            {
                var info = KernelInfo.From(method, ctx.SemanticModel);
                var fnMap = SpmdGenerator.ScopedFunctionMap(functions, info.Namespace, info.TypeName);
                _ = SpmdGenerator.GenerateKernelSource(info, method, structMap, fnMap, ctx.ReportDiagnostic);
            }
            else
            {
                var info = FunctionInfo.From(method, ctx.SemanticModel);
                var fnMap = SpmdGenerator.ScopedFunctionMap(functions, info.Namespace, info.TypeName);
                _ = SpmdGenerator.GenerateFunctionSource(info, method, structMap, fnMap, ctx.ReportDiagnostic);
            }
        }
        catch (UnsupportedConstructException ex)
        {
            ctx.ReportDiagnostic(Diagnostic.Create(
                Descriptors.Unsupported, ex.Location ?? method.GetLocation(),
                method.Identifier.Text, ex.What));
        }
    }

    /// <summary>
    /// Syntax-only scan for [SpmdStruct]/[SpmdFunction] declarations across the compilation
    /// (the same syntactic parse the generator's transforms use).
    /// </summary>
    private static (Dictionary<string, StructInfo> Structs, List<FunctionInfo> Functions) BuildTables(
        Compilation compilation)
    {
        var structs = new List<StructInfo>();
        var functions = new List<FunctionInfo>();
        foreach (var tree in compilation.SyntaxTrees)
        {
            foreach (var node in tree.GetRoot().DescendantNodes())
            {
                switch (node)
                {
                    case StructDeclarationSyntax s when SpmdGenerator.HasAttribute(s.AttributeLists, "SpmdStruct"):
                        if (StructInfo.From(s) is { } si)
                            structs.Add(si);
                        break;
                    case MethodDeclarationSyntax m when SpmdGenerator.HasAttribute(m.AttributeLists, "SpmdFunction"):
                        functions.Add(FunctionInfo.From(m));
                        break;
                    default:
                        break;
                }
            }
        }

        return (SpmdGenerator.BuildStructMap(structs), SpmdGenerator.DistinctFunctions(functions));
    }
}
