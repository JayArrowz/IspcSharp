using System;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using IspcSharp.Generators;
using Xunit;

namespace IspcSharp.Tests;

/// <summary>
/// The reduction unroll heuristic. Independent accumulators per unrolled copy break the
/// accumulator dependency chain, but each copy costs registers — and a double or long
/// reduction costs two, being a VDouble2/VLong2 pair. Unrolling a two-double-reduction body
/// four ways pins sixteen vectors, the whole AVX2 register file, and measured ~20% slower
/// than not unrolling. These tests pin the shape of the emitted loop so that trade stays put.
/// </summary>
public class UnrollHeuristicTests
{
    private static string Generate(string body, string signature, string hint, string extra = "", int unroll = 0)
    {
        string attr = unroll > 0 ? $"[Spmd(Unroll = {unroll})]" : "[Spmd]";
        string source = $$"""
            using System;
            using IspcSharp;

            namespace App
            {
                public static partial class K
                {
            {{extra}}

                    {{attr}}
                    public static {{signature}}
                    {
            {{body}}
                    }
                }
            }
            """;

        string tpa = (string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!;
        var references = tpa.Split(Path.PathSeparator)
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p))
            .Append(MetadataReference.CreateFromFile(typeof(Spmd).Assembly.Location))
            .ToList();

        var compilation = CSharpCompilation.Create(
            "UnrollTest", [CSharpSyntaxTree.ParseText(source)], references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        var driver = CSharpGeneratorDriver
            .Create([new SpmdGenerator().AsSourceGenerator()])
            .RunGenerators(compilation);

        return driver.GetRunResult().Results[0].GeneratedSources
            .Single(s => s.HintName == hint).SourceText.ToString();
    }

    /// <summary>
    /// The unroll factor is visible in the main vector loop's stride. An unrolled kernel also
    /// emits a single-gang remainder loop after it, so the widest stride must win — matching
    /// the "- __w" form first would report 1 for every kernel.
    /// </summary>
    private static int UnrollOf(string generated)
    {
        foreach (int k in new[] { 4, 3, 2 })
        {
            if (generated.Contains($"for (; __i <= __end - {k} * __w;", StringComparison.Ordinal))
                return k;
        }

        if (generated.Contains("for (; __i <= __end - __w;", StringComparison.Ordinal))
            return 1;

        throw new InvalidOperationException("no vector loop found in:\n" + generated);
    }

    [Fact]
    public void FloatReduction_SimpleBody_UnrollsFour()
    {
        string g = Generate(
            """
                        float sum = 0f;
                        foreach (int i in Spmd.Range(count))
                        {
                            sum += a[i] * b[i];
                        }
                        result[0] = sum;
            """,
            "void Dot(float[] a, float[] b, float[] result, int count)",
            "App.K.Dot_Spmd.g.cs");

        Assert.Equal(4, UnrollOf(g));   // one accumulator register per copy: 4 copies fit easily
    }

    [Fact]
    public void SingleDoubleReduction_UnrollsTwo()
    {
        // One VDouble2 accumulator is 2 registers per copy, so the budget of 4 allows 2 copies.
        // Measured neutral against 1 and 4 on AVX2 — the halves already add independently.
        string g = Generate(
            """
                        double sum = 0.0;
                        foreach (int i in Spmd.Range(count))
                        {
                            sum += a[i];
                        }
                        return sum;
            """,
            "double PreciseSum(float[] a, int count)",
            "App.K.PreciseSum_Spmd.g.cs");

        Assert.Equal(2, UnrollOf(g));
    }

    [Fact]
    public void TwoDoubleReductions_DoNotUnroll()
    {
        // Two VDouble2 accumulators is 4 registers per copy, the whole budget. Measured: 2 copies
        // ran 2.2x slower than 1 on an L2-resident loop.
        string g = Generate(
            """
                        double sum = 0.0;
                        double sumSq = 0.0;
                        foreach (int i in Spmd.Range(count))
                        {
                            float x = a[i];
                            sum += x;
                            sumSq += x * x;
                        }
                        return sum + sumSq;
            """,
            "double SumAndSquares(float[] a, int count)",
            "App.K.SumAndSquares_Spmd.g.cs");

        Assert.Equal(1, UnrollOf(g));
    }

    /// <summary>
    /// A one-line helper inlines away to a couple of instructions. It must not veto unrolling —
    /// the cost is charged through the call, not assumed from its presence.
    /// </summary>
    [Fact]
    public void TinyHelperInBody_StillUnrolls()
    {
        string g = Generate(
            """
                        float sum = 0f;
                        foreach (int i in Spmd.Range(count))
                        {
                            sum += Lerp(a[i], b[i], 0.25f);
                        }
                        result[0] = sum;
            """,
            "void Blend(float[] a, float[] b, float[] result, int count)",
            "App.K.Blend_Spmd.g.cs",
            """
                    [SpmdFunction]
                    public static float Lerp(float x, float y, float t) => x + ((y - x) * t);
            """);

        Assert.Equal(4, UnrollOf(g));
    }

    /// <summary>
    /// Cost is transitive: a small-looking wrapper around an expensive helper is expensive.
    /// </summary>
    [Fact]
    public void HelperCallingExpensiveHelper_DisablesUnroll()
    {
        string g = Generate(
            """
                        float sum = 0f;
                        foreach (int i in Spmd.Range(count))
                        {
                            sum += Outer(a[i]);
                        }
                        result[0] = sum;
            """,
            "void Nested(float[] a, float[] result, int count)",
            "App.K.Nested_Spmd.g.cs",
            """
                    [SpmdFunction]
                    public static float Outer(float x) => Inner(x) + Inner(x + 1f);

                    [SpmdFunction]
                    public static float Inner(float x)
                    {
                        float u = MathF.Exp(x) + MathF.Log(x + 2f);
                        float v = MathF.Sin(u) * MathF.Cos(u);
                        return MathF.Atan2(u, v) + MathF.Tanh(u * v) + MathF.Pow(u, v);
                    }
            """);

        Assert.Equal(1, UnrollOf(g));
    }

    [Fact]
    public void ExplicitUnroll_OverridesHeuristic()
    {
        // The body is heavy enough that the heuristic would pick 1; the attribute wins.
        string source = """
                        double sum = 0.0;
                        foreach (int i in Spmd.Range(count))
                        {
                            sum += Heavy(a[i]);
                        }
                        return sum;
            """;
        string helper = """
                    [SpmdFunction]
                    public static float Heavy(float x)
                    {
                        float u = MathF.Exp(x) + MathF.Log(x + 2f);
                        return MathF.Sin(u) * MathF.Cos(u) + MathF.Atan2(u, x) + MathF.Pow(u, x);
                    }
            """;

        Assert.Equal(1, UnrollOf(Generate(
            source, "double H(float[] a, int count)", "App.K.H_Spmd.g.cs", helper)));

        Assert.Equal(2, UnrollOf(Generate(
            source, "double H(float[] a, int count)", "App.K.H_Spmd.g.cs", helper, unroll: 2)));
    }

    [Fact]
    public void MathIntrinsicInBody_StillUnrolls()
    {
        // MathF.* lowers to inline VectorMath polynomials, not a call — it must not veto unrolling.
        string g = Generate(
            """
                        float sum = 0f;
                        foreach (int i in Spmd.Range(count))
                        {
                            sum += MathF.Sqrt(a[i]);
                        }
                        result[0] = sum;
            """,
            "void SqrtSum(float[] a, float[] result, int count)",
            "App.K.SqrtSum_Spmd.g.cs");

        Assert.Equal(4, UnrollOf(g));
    }

    [Fact]
    public void ControlFlowInBody_DisablesUnroll()
    {
        string g = Generate(
            """
                        float sum = 0f;
                        foreach (int i in Spmd.Range(count))
                        {
                            sum += a[i] > 0f ? a[i] : 0f;
                            for (int j = 0; j < 2; j++)
                                sum += 1f;
                        }
                        result[0] = sum;
            """,
            "void Guarded(float[] a, float[] result, int count)",
            "App.K.Guarded_Spmd.g.cs");

        Assert.Equal(1, UnrollOf(g));
    }
}
