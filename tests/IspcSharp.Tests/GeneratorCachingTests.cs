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

    /// <summary>
    /// A constant reached through a <c>using</c> directive: the companion is a separate file
    /// that imports only System and IspcSharp, so the reference has to be emitted fully
    /// qualified — in the vector body, in the pre-loop local, and in the scalar tail alike.
    /// </summary>
    [Fact]
    public void ConstantFromAnotherNamespace_IsFullyQualified_AndCompiles()
    {
        const string source = """
            using IspcSharp;
            using Tuning;

            namespace Tuning
            {
                public static class Knobs
                {
                    public const float Gain = 2.5f;
                    public const int Shift = 3;
                }
            }

            namespace App
            {
                public static partial class Kernels
                {
                    [Spmd]
                    public static float Weighted(float[] a, int[] bits, float[] o, int count)
                    {
                        float head = Knobs.Gain;
                        float sum = 0f;
                        foreach (int i in Spmd.Range(count))
                        {
                            o[i] = (a[i] * Knobs.Gain) + (bits[i] >>> Knobs.Shift);
                            sum += o[i];
                        }

                        return sum + head + Knobs.Gain;
                    }
                }
            }
            """;

        var driver = CreateDriver().RunGeneratorsAndUpdateCompilation(
            CreateCompilation(source), out var outputCompilation, out _);
        GeneratorRunResult result = driver.GetRunResult().Results[0];

        Assert.Empty(result.Diagnostics);
        string generated = result.GeneratedSources
            .Single(s => s.HintName == "App.Kernels.Weighted_Spmd.g.cs").SourceText.ToString();

        Assert.Contains("new VFloat(global::Tuning.Knobs.Gain)", generated);   // broadcast in the vector body
        Assert.Contains("global::Tuning.Knobs.Shift", generated);              // uniform shift count, not broadcast
        Assert.DoesNotContain("Knobs.Gain)", generated.Replace("global::Tuning.Knobs.Gain)", ""));

        var errors = outputCompilation.GetDiagnostics()
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .ToList();
        Assert.Empty(errors);
    }

    /// <summary>
    /// A constant declared in a different part of the same partial class: found through the
    /// symbol, not the syntax of the part the kernel happens to live in.
    /// </summary>
    [Fact]
    public void ConstantFromAnotherPartialPart_Resolves()
    {
        const string source = """
            using IspcSharp;

            namespace App;

            public static partial class Kernels
            {
                private const float Scale = 1.1920929e-7f;
            }

            public static partial class Kernels
            {
                [SpmdFunction]
                public static float Unit(int bits) => (bits >>> 9) * Scale;

                [Spmd]
                public static void Units(int[] bits, float[] o, int count)
                {
                    foreach (int i in Spmd.Range(count))
                    {
                        o[i] = Unit(bits[i]);
                    }
                }
            }
            """;

        var driver = CreateDriver().RunGeneratorsAndUpdateCompilation(
            CreateCompilation(source), out var outputCompilation, out _);
        GeneratorRunResult result = driver.GetRunResult().Results[0];

        Assert.Empty(result.Diagnostics);
        string fn = result.GeneratedSources
            .Single(s => s.HintName == "App.Kernels.Unit_SpmdFn.g.cs").SourceText.ToString();
        Assert.Contains("new VFloat(global::App.Kernels.Scale)", fn);

        var errors = outputCompilation.GetDiagnostics()
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .ToList();
        Assert.Empty(errors);
    }

    /// <summary>
    /// A qualified call reaches a [SpmdFunction] in another class (and another namespace), and
    /// still picks the right one when the name is ambiguous across classes. The emitted call is
    /// fully qualified, since the companion file has no using directives of its own.
    /// </summary>
    [Fact]
    public void QualifiedCall_ResolvesHelperInAnotherClass()
    {
        const string source = """
            using IspcSharp;
            using Rng;

            namespace Rng
            {
                public static partial class Bits
                {
                    [SpmdFunction]
                    public static float Curve(float x) => x * 2f;
                }
            }

            namespace App
            {
                [SpmdStruct]
                public struct Word2 { public int W0; public int W1; }

                public static partial class Lib
                {
                    [SpmdFunction]
                    public static Word2 Draw(int i) => new Word2 { W0 = i * 3, W1 = i * 5 };
                }

                public static partial class Kernels
                {
                    // Same helper name as Rng.Bits.Curve: a bare call must still mean this one.
                    [SpmdFunction]
                    public static float Curve(float x) => x * 3f;

                    [Spmd]
                    public static void Run(float[] a, float[] o, int[] w, int count)
                    {
                        foreach (int i in Spmd.Range(count))
                        {
                            Word2 d = Lib.Draw(i);                   // another class, struct-returning
                            o[i] = Curve(a[i]) + Bits.Curve(a[i]);   // 'Bits' is reachable only via using
                            w[i] = d.W0 ^ d.W1;
                        }
                    }
                }
            }
            """;

        var driver = CreateDriver().RunGeneratorsAndUpdateCompilation(
            CreateCompilation(source), out var outputCompilation, out _);
        GeneratorRunResult result = driver.GetRunResult().Results[0];

        Assert.Empty(result.Diagnostics);
        string generated = result.GeneratedSources
            .Single(s => s.HintName == "App.Kernels.Run_Spmd.g.cs").SourceText.ToString();

        Assert.Contains("global::App.Lib.Draw(", generated);        // another class, struct-returning
        Assert.Contains("global::Rng.Bits.Curve(", generated);      // another namespace, via using
        Assert.Contains("Curve(VFloat.Load(a, __i))", generated);   // bare call stays this type's

        var errors = outputCompilation.GetDiagnostics()
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .ToList();
        Assert.Empty(errors);
    }

    [Fact]
    public void SameFunctionName_InDifferentClasses_BothGenerate_AndResolvePerClass()
    {
        const string source = """
            using IspcSharp;

            namespace CacheDemo;

            public static partial class DspA
            {
                [SpmdFunction]
                public static float Curve(float x) => x * x;

                [Spmd]
                public static void Apply(float[] input, float[] output, int count)
                {
                    foreach (int i in Spmd.Range(count))
                    {
                        output[i] = Curve(input[i]);
                    }
                }
            }

            public static partial class DspB
            {
                [SpmdFunction]
                public static float Curve(float x, float gain) => x * gain;

                [Spmd]
                public static void Apply(float[] input, float[] output, float gain, int count)
                {
                    foreach (int i in Spmd.Range(count))
                    {
                        output[i] = Curve(input[i], gain);
                    }
                }
            }
            """;

        var driver = CreateDriver().RunGeneratorsAndUpdateCompilation(
            CreateCompilation(source), out var outputCompilation, out _);
        GeneratorRunResult result = driver.GetRunResult().Results[0];

        Assert.Empty(result.Diagnostics);

        var a = result.GeneratedSources.Single(s => s.HintName == "CacheDemo.DspA.Curve_SpmdFn.g.cs");
        var b = result.GeneratedSources.Single(s => s.HintName == "CacheDemo.DspB.Curve_SpmdFn.g.cs");
        Assert.Contains("x * x", a.SourceText.ToString());
        Assert.Contains("gain", b.SourceText.ToString());

        _ = result.GeneratedSources.Single(s => s.HintName == "CacheDemo.DspA.Apply_Spmd.g.cs");
        _ = result.GeneratedSources.Single(s => s.HintName == "CacheDemo.DspB.Apply_Spmd.g.cs");
        var errors = outputCompilation.GetDiagnostics()
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .ToList();
        Assert.Empty(errors);
    }
}
