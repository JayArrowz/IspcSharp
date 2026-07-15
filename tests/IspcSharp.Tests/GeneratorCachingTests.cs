using System;
using System.IO;
using System.Linq;
using IspcSharp.Generators;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace IspcSharp.Tests;

public class GeneratorCachingTests
{
    private const string KernelSource = """
        using IspcSharp;

        namespace CacheDemo;

        [SpmdStruct]
        public struct Vec2
        {
            public float X;
            public float Y;
        }

        public static partial class Kernels
        {
            [SpmdFunction]
            public static float Lerp(float a, float b, float t) => a + ((b - a) * t);

            [Spmd]
            public static void Blend(float[] a, float[] b, float[] o, float t, int count)
            {
                foreach (int i in Spmd.Range(count))
                {
                    o[i] = Lerp(a[i], b[i], t);
                }
            }
        }
        """;

    private static readonly string[] ModelSteps =
        ["SpmdStructs", "SpmdFunctions", "SpmdKernels", "SpmdStructTable", "SpmdFunctionTable"];

    private static CSharpCompilation CreateCompilation(string source)
    {
        string tpa = (string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!;
        var references = tpa.Split(Path.PathSeparator)
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p))
            .Append(MetadataReference.CreateFromFile(typeof(Spmd).Assembly.Location))
            .ToList();

        return CSharpCompilation.Create(
            "CacheTest",
            [CSharpSyntaxTree.ParseText(source)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
    }

    private static GeneratorDriver CreateDriver()
        => CSharpGeneratorDriver.Create(
            [new SpmdGenerator().AsSourceGenerator()],
            driverOptions: new GeneratorDriverOptions(
                IncrementalGeneratorOutputKind.None,
                trackIncrementalGeneratorSteps: true));

    [Fact]
    public void Generator_ProducesSources_AndNoDiagnostics()
    {
        var driver = CreateDriver().RunGenerators(CreateCompilation(KernelSource));
        GeneratorRunResult result = driver.GetRunResult().Results[0];
        Assert.Empty(result.Diagnostics);
        Assert.Contains(result.GeneratedSources, s => s.HintName == "CacheDemo.Kernels.Blend_Spmd.g.cs");
        Assert.Contains(result.GeneratedSources, s => s.HintName == "CacheDemo.Kernels.Lerp_SpmdFn.g.cs");
        Assert.Contains(result.GeneratedSources, s => s.HintName == "__SpmdStructs.g.cs");
    }

    [Fact]
    public void Pipeline_IsFullyCached_WhenAnUnrelatedFileIsAdded()
    {
        var compilation = CreateCompilation(KernelSource);
        var driver = CreateDriver().RunGenerators(compilation);

        var edited = compilation.AddSyntaxTrees(
            CSharpSyntaxTree.ParseText("namespace CacheDemo { internal sealed class Unrelated { } }"));
        driver = driver.RunGenerators(edited);
        GeneratorRunResult result = driver.GetRunResult().Results[0];

        foreach (string step in ModelSteps)
        {
            Assert.All(
                result.TrackedSteps[step].SelectMany(s => s.Outputs),
                o => Assert.True(
                    o.Reason is IncrementalStepRunReason.Cached or IncrementalStepRunReason.Unchanged,
                    $"step '{step}' re-ran: {o.Reason}"));
        }

        Assert.All(
            result.TrackedOutputSteps.SelectMany(kv => kv.Value).SelectMany(s => s.Outputs),
            o => Assert.True(
                o.Reason == IncrementalStepRunReason.Cached,
                $"an output step re-ran: {o.Reason}"));
    }

    [Fact]
    public void Pipeline_Regenerates_WhenTheKernelBodyChanges()
    {
        var compilation = CreateCompilation(KernelSource);
        var driver = CreateDriver().RunGenerators(compilation);

        var edited = CreateCompilation(
            KernelSource.Replace("o[i] = Lerp(a[i], b[i], t);", "o[i] = Lerp(a[i], b[i], t) * 2f;"));
        driver = driver.RunGenerators(edited);
        GeneratorRunResult result = driver.GetRunResult().Results[0];

        var blend = result.GeneratedSources.Single(s => s.HintName == "CacheDemo.Kernels.Blend_Spmd.g.cs");
        Assert.Contains("2f", blend.SourceText.ToString());
    }

    [Fact]
    public void SameKernelName_InDifferentClasses_BothGenerate()
    {
        const string source = """
            using IspcSharp;

            namespace CacheDemo;

            public static partial class KernelsA
            {
                [Spmd]
                public static void Scale(float[] a, float[] o, int count)
                {
                    foreach (int i in Spmd.Range(count))
                    {
                        o[i] = a[i] * 2f;
                    }
                }
            }

            public static partial class KernelsB
            {
                [Spmd]
                public static void Scale(float[] a, float[] o, int count)
                {
                    foreach (int i in Spmd.Range(count))
                    {
                        o[i] = a[i] * 3f;
                    }
                }
            }
            """;

        var driver = CreateDriver().RunGenerators(CreateCompilation(source));
        GeneratorRunResult result = driver.GetRunResult().Results[0];

        Assert.Empty(result.Diagnostics);
        var a = result.GeneratedSources.Single(s => s.HintName == "CacheDemo.KernelsA.Scale_Spmd.g.cs");
        var b = result.GeneratedSources.Single(s => s.HintName == "CacheDemo.KernelsB.Scale_Spmd.g.cs");
        Assert.Contains("2f", a.SourceText.ToString());
        Assert.Contains("3f", b.SourceText.ToString());
    }
}
