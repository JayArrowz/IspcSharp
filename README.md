# IspcSharp

[![Build Status](https://github.com/jayarrowz/IspcSharp/workflows/CI/badge.svg)](https://github.com/jayarrowz/IspcSharp/actions)
[![NuGet](https://img.shields.io/nuget/v/IspcSharp.svg)](https://www.nuget.org/packages/IspcSharp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow.svg)](https://buymeacoffee.com/jayarrowz)

**SPMD-on-SIMD programming for C#**, write scalar-looking kernels, run them across every SIMD lane and every CPU core. Inspired by [Intel ISPC](https://ispc.github.io/), built on `System.Numerics.Vector<T>` so the same code uses SSE, AVX2, AVX-512, or NEON at runtime.

- **`IspcSharp`**, the runtime: varying types (`VFloat`, `VInt`, `VDouble`, `VLong`, masks), the SPMD execution engine (`Spmd.Foreach`, `Spmd.ParallelForeach`, `Spmd.While`), a vectorized math library, reductions, gather/scatter, cross-lane shuffles, SoA helpers, and a correctness verifier.
- **`IspcSharp.Generators`**, a Roslyn source generator: write a plain scalar loop, mark it `[Spmd]`, and get vectorized `_Simd` / `_ParallelSimd` companions generated at compile time.

## Why

The .NET JIT auto-vectorizes almost nothing beyond trivial loops. Hardware intrinsics (`Vector256<T>`, `Avx2.*`) get you the performance but the code is write-only and per-ISA. ISPC solves this for C/C++ with the **SPMD model**: you write the *scalar logic for one element*, and the compiler maps it across SIMD lanes, converting branches into masks so all lanes stay in lockstep. IspcSharp brings that model to C#:

| | Plain C# loop | Hand-written intrinsics | ISPC (native) | **IspcSharp** |
|---|---|---|---|---|
| Readable, scalar-shaped code | ✅ | ❌ | ✅ | ✅ |
| Uses SIMD reliably | ❌ | ✅ | ✅ | ✅ |
| Portable across SSE/AVX/AVX-512/NEON | ✅ | ❌ (per-ISA paths) | ✅ | ✅ (runtime-sized `Vector<T>`) |
| Multi-core tasking built in | ❌ | ❌ | ✅ | ✅ |
| No FFI / separate toolchain / P/Invoke | ✅ | ✅ | ❌ | ✅ |
| One debugger, one profiler, one build | ✅ | ✅ | ❌ | ✅ |

**Realistic wins** (compute-bound kernels): 4–16× from SIMD depending on lane width (SSE/NEON = 4 floats, AVX2 = 8, AVX-512 = 16), multiplied by core count with `ParallelForeach`. Branchy kernels land lower (2–8×) because both sides of masked branches execute. Memory-bandwidth-bound loops may see little or nothing, profile first, vectorize the compute-bound hot spots.


## Performance
Benchmarks are run automatically on Linux x64 (AVX2), Windows x64 (AVX2), and macOS ARM64 (Apple Silicon / NEON) using GitHub Actions [here](https://github.com/JayArrowz/IspcSharp/actions/runs/29326842215/attempts/2#summary-87066998054).

Across the benchmark suite:

- Compute-heavy kernels: typically 4–12× faster than scalar C#
- Best cases: over 20× faster when combining SIMD with multicore execution
- Memory-bandwidth-bound kernels: around 1.5–2×, approaching hardware bandwidth limits
- Branch-heavy kernels: still commonly 2–7× faster, depending on divergence

Gather-heavy workloads are intentionally included to demonstrate the limits of SIMD. When every lane reads unrelated memory locations, memory latency dominates and vectorization may not outperform scalar code.

Every benchmark compares the generated SIMD implementation against the original scalar implementation written by the user.

## Getting started

```xml
<ItemGroup>
  <PackageReference Include="IspcSharp" Version="1.0.3" />
  <PackageReference Include="IspcSharp.Generators" Version="1.0.3" OutputItemType="Analyzer" ReferenceOutputAssembly="false" />
</ItemGroup>
```

Then verify everything works on your machine:

```bash
dotnet test tests/IspcSharp.Tests                                  # 272 tests
dotnet run -c Release --project samples/IspcSharp.Samples          # smoke-test samples
```

## Usage

### Style 1, `[Spmd]` source generator (write scalar, get SIMD)

```csharp
using IspcSharp;

public static partial class Kernels          // must be partial
{
    [Spmd]
    public static void ClampScale(float[] input, float[] output, float scale, int count)
    {
        foreach (var i in Spmd.Range(count))
        {
            float x = input[i];
            if (x < 0f)      x = 0f;
            else if (x > 1f) x = 1f;
            output[i] = x * scale;
        }
    }
}
```

The generator emits, at compile time:

```csharp
// ClampScale_Simd(...)        , SIMD, single thread
// ClampScale_ParallelSimd(...), SIMD × all cores
```

where the `if/else` became branch-free per-lane selects over full-width contiguous loads, plus a scalar tail loop for the `count % LaneCount` leftovers. Your original scalar method is untouched and remains the debuggable reference implementation.

More of what kernels can look like:

```csharp
// Reductions, returned directly (or written to output buffers after the loop):
[Spmd]
public static float SumOfSquares(float[] input, int count)
{
    float sum = 0f;
    foreach (var i in Spmd.Range(count))
        sum += input[i] * input[i];
    return sum;                            // _ParallelSimd uses thread-local partials
}

// 2D domains + divergent loops (per-lane trip counts):
[Spmd]
public static void Mandelbrot(int[] iters, float minX, float minY, float dx, float dy,
                              int maxIter, int width, int height)
{
    foreach (var (x, y) in Spmd.Range2D(width, height))    // or Spmd.Range2DTiled(w, h)
    {
        float cx = minX + x * dx, cy = minY + y * dy;
        float zx = 0f, zy = 0f;
        int i = 0;
        while (zx * zx + zy * zy < 4f && i < maxIter)      // masked per-lane iteration
        {
            float n = zx * zx - zy * zy + cx;
            zy = 2f * zx * zy + cy;
            zx = n;
            i++;
        }
        iters[y * width + x] = i;                          // stays a contiguous store
    }
}

// Uniform control flow around the loop + 2-D matrices (matrix multiply):
[Spmd]
public static void MatMul(float[,] a, float[,] b, float[,] c, int n, int m, int p)
{
    for (int row = 0; row < n; row++)          // uniform loops emit as plain C#
        for (int col = 0; col < p; col++)
        {
            float sum = 0f;
            foreach (var k in Spmd.Range(m))   // only this loop is vectorized
                sum += a[row, k] * b[k, col];  // a[row,k] contiguous; b[k,col] gathered
            c[row, col] = sum;
        }
}

// 64-bit integer (long) lanes, bit-manipulation over a long[] (a splitmix-style hash):
[Spmd]
public static void Hash(long[] input, long[] output, int count)
{
    foreach (var i in Spmd.Range(count))       // runs as a VLong/VMaskD 64-bit gang
    {
        long x = input[i];
        x = x ^ (x >> 33);                     // bitwise + shift on 64-bit lanes
        x = x * 0x5555555555555555L;
        x = x ^ (x >> 29);
        output[i] = x;
    }
}

// Cross-gang-width: an exact 64-bit sum over int[] data. No long buffer, so this stays a
// 32-bit int gang; the accumulator widens into a VLong2 pair (VInt.LaneCount longs). A plain
// int accumulator would overflow, the integer mirror of a double accumulator over float data.
[Spmd]
public static long WideSum(int[] input, int count)
{
    long sum = 0;
    foreach (var i in Spmd.Range(count))       // 32-bit int gang; sum accumulates in VLong2
        sum += input[i];
    return sum;                                // returned as a scalar long
}

// Blittable structs ([SpmdStruct]) + reusable helpers ([SpmdFunction]), ISPC's structs + functions.
[SpmdStruct] public struct Complex { public float Re, Im; }

[SpmdFunction]                                 // a varying helper: Complex → VFloat gangs, callable from kernels
public static Complex CMul(Complex a, Complex b)
    => new Complex { Re = a.Re * b.Re - a.Im * b.Im, Im = a.Re * b.Im + a.Im * b.Re };

[Spmd]
public static void MulSpectra(float[] ar, float[] ai, float[] br, float[] bi, float[] cr, float[] ci, int n)
{
    foreach (var i in Spmd.Range(n))
    {
        Complex a = new Complex { Re = ar[i], Im = ai[i] };   // struct locals (SoA in registers)
        Complex b = new Complex { Re = br[i], Im = bi[i] };
        Complex c = CMul(a, b);                                // call the varying helper per gang
        cr[i] = c.Re; ci[i] = c.Im;
    }
}

// Return a struct of reductions (ISPC's "export a struct"): each field is one reduced accumulator,
// built after the loop and returned by both _Simd and _ParallelSimd.
[SpmdStruct] public struct Stats { public float Sum, Min, Max; }

[Spmd]
public static Stats Summarize(float[] a, int count)
{
    float sum = 0f, mn = float.MaxValue, mx = float.NegativeInfinity;
    foreach (var i in Spmd.Range(count))
    {
        sum += a[i];                       // reduce_add
        mn = Math.Min(mn, a[i]);           // reduce_min
        mx = Math.Max(mx, a[i]);           // reduce_max
    }
    return new Stats { Sum = sum, Min = mn, Max = mx };   // trailing scalar C# over the reduced values
}

// Mixed-field struct buffer: fields differ in type but share a 32-bit width, so Hit[] is a valid
// AoS buffer, T is gathered/scattered as float, Id as int, through their own typed views.
[SpmdStruct] public struct Hit { public float T; public int Id; }

[Spmd]
public static void BumpHits(Hit[] hits, float dt, int count)
{
    foreach (var i in Spmd.Range(count))
    {
        Hit h = hits[i];                   // AoS gather of both fields (ISPC100/101, flagged)
        h.T = h.T + dt;                    // float field
        h.Id = h.Id + 1;                   // int field
        hits[i] = h;                       // AoS scatter
    }
}

// Fixed-size array members (ISPC's `float Coef[3]`): [SpmdArray(N)] expands each into N register
// gangs. Element indices are compile-time literals; array members of different types are fine.
[SpmdStruct]
public struct Poly
{
    [SpmdArray(3)] public float[] Coef;    // 3 float gangs, held SoA in registers
    [SpmdArray(2)] public int[]   Tag;     // 2 int gangs (a differently-typed array member)
    public float Bias;                     // a plain scalar field alongside the arrays
}

[SpmdFunction]                             // helper indexing the array members by constant index
public static float EvalPoly(Poly p, float x)
    => p.Coef[0] + p.Coef[1] * x + p.Coef[2] * (x * x) + p.Bias;

[Spmd]
public static void ApplyPoly(float[] c0, float[] c1, float[] c2, float[] bias,
                             float[] x, float[] o, int count)
{
    foreach (var i in Spmd.Range(count))
    {
        // object-initializer construction: `new float[]{...}` fills the gangs, `new int[2]` zero-fills
        Poly p = new Poly { Coef = new float[] { c0[i], c1[i], c2[i] }, Tag = new int[2], Bias = bias[i] };
        o[i] = EvalPoly(p, x[i]);
    }
}
```

The `Spmd.Range` loop can sit inside `for`/`while`/`if` scaffolding, those statements (loop counters, `MathF.PI`, tuple swaps, `x ^= y`, `arr.Length`, …) run as ordinary scalar C#, and only the inner loop vectorizes. That's enough to write an iterative FFT or a matrix multiply as one `[Spmd]` method. Anything the generator can't vectorize is a **compile error with an `ISPC00x` diagnostic naming the construct**, never a silent scalar fallback, never silently wrong codegen. The scalar method always still compiles; only the `_Simd` companion is withheld.

### Style 2, runtime API (explicit control)

```csharp
using IspcSharp;

public static void Tonemap(float[] input, float[] output, float exposure)
{
    Spmd.ParallelForeach(input.Length, (in SpmdContext ctx) =>
    {
        if (ctx.IsFullGang)
        {
            // Hot path: full-speed contiguous SIMD load/store.
            var x = VFloat.Load(input, ctx.Base);
            (VFloat.One - VectorMath.Exp(-(x * exposure))).Store(output, ctx.Base);
        }
        else
        {
            // Tail only (at most one gang per call): masked, lane-by-lane.
            var x = VFloat.LoadMasked(input, ctx.Base, ctx.Active);
            (VFloat.One - VectorMath.Exp(-(x * exposure))).StoreMasked(output, ctx.Base, ctx.Active);
        }
    });
}
```

> **Performance trap:** `LoadMasked`/`StoreMasked` are *scalar per-lane loops*, they exist for
> correctness at buffer tails, not as the default access path. Using them for every gang makes the
> "SIMD" version slower than scalar. Always branch on `ctx.IsFullGang` as above (the `[Spmd]`
> generator does this for you automatically), and hoist any mutable state out of per-gang lambdas —
> a lambda that captures reassigned locals allocates a closure per gang.

For divergent loops at runtime, `Spmd.While` maintains the execution mask for you, with per-lane `Break` / `Continue` / `Return`:

```csharp
Spmd.While(
    ctx.Active,
    active => active & ((zx*zx + zy*zy) < new VFloat(4f)) & (count < new VInt(maxIter)),
    (ref LoopState s) =>
    {
        var nzx = zx*zx - zy*zy + cx;
        var nzy = zx*zy*2f + cy;
        zx = zx.Blend(s.Active, nzx);      // masked assignment: only running lanes update
        zy = zy.Blend(s.Active, nzy);
        count = count.Blend(s.Active, count + 1);
    });
```

### Verify before you trust

SIMD/masking bugs are **silent**, wrong numbers, not exceptions. Always check kernels against a scalar reference, with a count that exercises the tail:

```csharp
KernelVerifier.AssertMatches(
    count: Spmd.LaneCount * 3 + 1,                        // deliberately hits the tail path
    scalarReference: i => 1f - MathF.Exp(-input[i] * 1.5f),
    runKernel: output => Kernels.Tonemap(input, output, 1.5f));
```

## The ISPC concept map

| ISPC | IspcSharp |
|---|---|
| `varying float` / `int` / `double` / `long` | `VFloat` / `VInt` / `VDouble` / `VLong` (+ `VDouble2` / `VLong2` for double / 64-bit-int precision at float-gang width) |
| `varying bool` / execution mask | `VMask` (32-bit float/int gangs) / `VMaskD` (64-bit double/long gangs), `Widen`/`Narrow` to convert |
| `uniform float` | plain `float` (broadcasts implicitly in mixed expressions) |
| `programIndex` / `programCount` | `VInt.ProgramIndex` / `Spmd.LaneCount` |
| `foreach (i = 0 ... n)` | `Spmd.Foreach(n, kernel)` or `[Spmd]` + `Spmd.Range(n)` |
| `foreach` 2D / `foreach_tiled` | `Spmd.Range2D(w, h)` / `Spmd.Range2DTiled(w, h[, tw, th])` in `[Spmd]`; `Spmd.Foreach2D` at runtime |
| `launch` / tasking | `_ParallelSimd` (generated) / `Spmd.ParallelForeach(n, kernel)` |
| divergent `while` / `break` / `continue` / `return` | plain C# statements in `[Spmd]` (per-lane masks); `Spmd.While` + `LoopState` at runtime |
| `cif` (coherent if) | `if (Spmd.Coherent(cond))` in `[Spmd]` / `Spmd.If` / `mask.Any()` |
| `select(cond, a, b)` | ternary `?:` in `[Spmd]`; `VFloat.Select(mask, a, b)` / `x.Blend(mask, v)` |
| `reduce_add/min/max` | `sum += x` / `m = Math.Min(m, x)` patterns in `[Spmd]`; `Reduce.Add/Min/Max` (masked overloads) |
| gather / scatter | `a[expr]` / `a[expr] = x` in `[Spmd]`; `Memory.Gather` / `Memory.Scatter` (masked; float/int/double) |
| 2-D arrays (`float m[][]`) | `float[,]`/`int[,]`/`double[,]` params in `[Spmd]` (`a[row, col]`, row-major flat view) |
| `struct` + member functions | `[SpmdStruct]` + `[SpmdFunction]` (varying companion held SoA in registers) |
| `struct` array members (`float c[3]`) | `[SpmdArray(3)] float[] c;` (N register gangs; `c[k]` with constant `k`; mixed element types) |
| return / export a `struct` | `[Spmd]`/`[SpmdFunction]` returning a `[SpmdStruct]` (`return new Stats { … };`) |
| array of structs (AoS) | `S[]` buffer param (`buf[i].field`; same-width fields, incl. mixed `float`+`int`) |
| uniform scaffolding (FFT, matmul) | `Spmd.Range` loop nested in uniform `for`/`while`/`if`, scaffolding runs as scalar C# |
| stdlib (`sin`, `exp`, `rsqrt`, ...) | `VectorMath`, branch-free approximations, float + double |
| `shuffle` / `rotate` / `broadcast` / `shift` | `Lanes.Shuffle` / `Rotate` / `Broadcast` / `ShiftLanes` |
| SoA layout idioms | `SoaFloat2` / `SoaFloat3` |

## What the `[Spmd]` generator supports

The generator handles a deliberately constrained, verifiable subset. Kernel shape: optional pre-loop locals, one `foreach` over a `Spmd.Range*` marker, optional trailing statements (including `return`). The foreach may also be **nested inside uniform control flow**, `for`/`while`/`if` scaffolding whose statements (loop counters, swaps, `MathF.PI`, `j ^= bit`, `for (i += len)`, `arr.Length`, …) run as plain scalar C# around the vectorized loop. This is what lets an iterative FFT or a matrix multiply be a single `[Spmd]` method (see the matmul example above); only the innermost `Spmd.Range` loop is vectorized, so its body must still be per-lane parallel (no loop-carried recurrence).

- **Types:** `float`, `int`, `double`, and `long` lane variables (locals declared in the loop body), parameters of `float[]`/`int[]`/`double[]`/`long[]`, `float[,]`/`int[,]`/`double[,]`/`long[,]` (2-D matrices), `Span<T>`/`ReadOnlySpan<T>` of those, and uniform `float`/`int`/`double`/`long`. Implicit int→float promotion; explicit `(float)`/`(int)`/`(double)` casts. **Wide (64-bit) kernels:** a kernel with `double` **buffers** becomes a `VDouble`/`VMaskD` gang; a kernel with `long` **buffers** becomes a `VLong`/`VMaskD` gang (64-bit integers: arithmetic, bitwise `& | ^ ~`, shifts, `/` `%`, `Math.Min/Max/Abs`, gather/scatter, reductions). Both are half the lane count of float/int, and a wide kernel can't mix in 32-bit buffers. In float/int kernels, `double` locals/uniforms/`d`-literals/`(double)` casts give **double-precision intermediates at full gang width** (`VDouble2` pairs), e.g. a `double` accumulator over float data; likewise `long` locals/`L`-literals/`(long)` casts give **64-bit integer intermediates at full gang width** (`VLong2` pairs), e.g. an exact `long` sum over `int` data, or an overflow-free `(long)a[i] * b[i]` widening multiply.
- **Return values:** kernels may return `float`/`int`/`double`/`long`, reduce into a pre-loop local and `return` it after the loop; both `_Simd` and `_ParallelSimd` return it directly. (A `long` return from a 32-bit kernel is the natural home for a `VLong2` accumulator, an exact 64-bit sum over `int` data.) A kernel may also **return a `[SpmdStruct]`** built from several reduced accumulators (`return new Stats { Sum = sum, Min = mn, Max = mx };`), ISPC's "export a struct" — the trailing `return` runs as scalar C# over the already-reduced scalars, so it works in both variants. (`[SpmdFunction]` helpers return structs too; see below.)
- **Memory:** `a[i]` reads/writes indexed by the loop variable, including affine offsets `a[i + u1 + u2]` (uniform terms), this keeps `buf[y*width + x]` and an FFT's `buf[i + k + half]` contiguous loads. **Affine index locals** are tracked too: `int even = i + k; … real[even] = …` stays a contiguous load/store (the offset even propagates through `int odd = even + half`), so you don't have to inline the index arithmetic to keep the fast path. **2-D arrays** index as `a[row, col]` (viewed as a flat row-major span): contiguous when the last index is unit-stride in the lane (`a[row, k]`), a strided gather otherwise (`b[k, col]`). Any genuinely lane-varying index becomes a gather (`a[expr]`) or scatter (`a[expr] = x`) via `Memory.Gather`/`Scatter`; double kernels index with `VLong` through `table[(int)x]`-style casts.
- **Constants:** `MathF.PI`/`Math.PI`, `.E`, `.Tau`, `float.MaxValue`/`int.MinValue`/`double.Epsilon`/`…` broadcast automatically inside vectorized expressions.
- **Expressions:** `+ - * /`, integer `/` and `%` (per-lane loops; inactive-lane divisors are masked to 1 so masked-off zero divisors can't throw), unary `-` and `~`, shifts `<<` `>>` `>>>` with **uniform or per-lane counts** (per-lane uses AVX2 variable shifts where available; C#'s low-5-bits semantics), bitwise `& | ^` on int lanes, ternary `?:`, and `Math`/`MathF` calls with `VectorMath` equivalents: `Sqrt`, `Abs`, `Min`, `Max`, `Sin`, `Cos`, `Tan`, `Exp`, `Log`, `Pow`, `Tanh`, `Floor`, `Round`, `Clamp`, `Atan`, `Atan2`, `Asin`, `Acos`, `Cbrt`, `FusedMultiplyAdd` (double overloads resolve automatically for double arguments).
- **Assignments:** `=`, `+=`, `-=`, `*=`, `/=`, compound bitwise `&=` `|=` `^=` (int lanes), and `++`/`--` statements on lane locals.
- **Control flow:** `if`/`else if`/`else`, arbitrarily nested, with `< > <= >= == != && || !` conditions, lowered to masked selects; wrap a condition in `Spmd.Coherent(...)` to add an `Any()` guard so whole gangs skip branches no lane takes (ISPC's `cif`; identity function at runtime). Varying `while`/`for` loops lower to execution-mask iteration; uniform `for` loops emit as plain C#. `break`/`continue` work in both (per-lane masks in varying loops), including `continue` at the `foreach` top level. A bare `return;` **retires the lane** (ISPC semantics): the element exits every varying loop and skips the rest of its body, while other elements keep processing, see limitations for how this differs from scalar C#. Local declarations inside masked branches are auto-blended.
- **Reductions:** a pre-loop local accumulated *only* via `x += expr` (add), `x = Math.Min(x, expr)`, or `x = Math.Max(x, expr)` becomes a vector accumulator, horizontally reduced after the loop. Multiple accumulators per loop are fine, one operator each. `_ParallelSimd` gives each chunk thread-local partials, combined deterministically afterward.
- **Blittable structs (`[SpmdStruct]`):** a struct whose fields are all `float`/`int`/`double`/`long` can be a **lane local**, a helper argument/return, or, when every field shares one type, a **buffer element**. The generator emits a varying companion (`Name__V`, one gang per field, held Structure-of-Arrays in registers), so `Vec3 v = new Vec3 { X = a[i], … };`, field reads/writes `v.X`, whole-struct assignment `v = w;` (masked-blended in divergent branches), and construction (`new S { f = … }` or `new S(a, b, …)` in field order) all work. **Struct buffers** (`Vec3[] pts`) view the array as a flat SoA span: `pts[i].X` and `pts[i] = new Vec3 { … }` lower to contiguous loads (single-field structs) or strided gather/scatter (AoS, the `ISPC100` case), and make the kernel `_Simd`-only. **Mixed-field buffers** are allowed when every field is 32-bit — `Hit { float T; int Id; }[]` works, each field gathered/scattered through its own correctly-typed flat view (`T` as `float`, `Id` as `int`). Mixing 32- and 64-bit fields in a buffer (e.g. `float` + `double`) would break the element stride, so those structs stay locals/args only; pass separate SoA arrays instead. The struct keeps its default sequential layout. **Fixed-size array members** (ISPC's `float x[4]`) are declared `[SpmdArray(N)] public float[] Coef;` — the companion expands each into N independent gangs (`Coef_0 … Coef_{N-1}`), so `p.Coef[k]` reads/writes one gang. The index `k` must be a **compile-time integer literal** (SoA-in-registers has no runtime-indexed lane), array members of differing element types are fine (`float[]` + `int[]`), and construction is by object initializer (`new Poly { Coef = new float[]{ a, b, c }, Tag = new int[2] }`, a sized-but-empty `new int[2]` zero-fills). An array-member struct is a **local / helper arg / return only**, never a buffer element.
- **Vectorized helper functions (`[SpmdFunction]`):** a side-effect-free helper with primitive/struct arguments and return (ISPC's non-`export` function) gets a varying companion, each `float` becomes a `VFloat`, each struct its `__V` form. Any `[Spmd]` kernel or other `[SpmdFunction]` can call it (`o[i] = Lerp(a[i], b[i], t);`, `Complex c = CMul(a, b);`), so common math composes without inlining by hand. The body uses the same subset as a kernel but takes scalar values instead of buffers and has no `Spmd.Range` loop; it operates on the gang it is handed. No recursion or buffer parameters.
- **Domains:** `Spmd.Range(n)`, `Spmd.Range(start, end)`, `Spmd.Range2D(w, h)` (rows of x-gangs, parallel over rows), `Spmd.Range2DTiled(w, h[, tileW, tileH])` (cache-blocked 64×64 tiles by default, parallel over tiles).

Notes:
- The containing type must be `partial`.
- `_ParallelSimd` is generated only for the simple shape (a single top-level foreach) with array parameters. It is **not** generated when the foreach is nested in uniform control flow (parallelize the outer loop yourself) or when a `float[,]`/`int[,]`/`double[,]` parameter is present (the flat-span view can't cross threads), you get an informational `ISPC005` diagnostic and only `_Simd`. Parallel float sums reassociate across chunks, expect tiny drift vs. serial order, exactly like ISPC tasks.
- **Performance diagnostics, every slow path is flagged where the generator emits it** (accurate to the exact expression, since the generator has the affine analysis). Warnings, so they surface in the build, not just the IDE; suppress an individual code via `.editorconfig` (`dotnet_diagnostic.ISPC101.severity = none`) when the slow path is intentional:
  - `ISPC100`, Array-of-Structs field access (`points[i].X`), each field read is a per-lane gather; use SoA (`SoaFloat2/3` or separate arrays). *(analyzer; also fires for runtime-API lambdas)*
  - `ISPC101`, **gather**: a lane-varying indexed load (`table[idx[i]]`, `a[i*2]`, `b[k, col]`, a whole `struct[]` element), not a contiguous vector load.
  - `ISPC102`, **scatter**: a lane-varying indexed store (`out[idx[i]] = …`), .NET has no hardware scatter.
  - `ISPC103`, **integer `/` `%`**: no SIMD instruction, runs as a per-lane scalar loop (use a shift/mask for power-of-two divisors).
  - `ISPC104`, **double↔integer conversion** (`(int)`/`(long)` on double lanes): scalarizes on AVX2 (no AVX-512DQ), keep the value in `double` or hoist the conversion.
- Generated code assumes buffers are at least `n` elements, same contract as your scalar loop.
- **`[Spmd(Streaming = true)]`** emits **non-temporal (streaming) stores** for contiguous `float`/`double` output writes, plus a store fence after the loop. This bypasses the cache, a win only for large, write-once output that isn't re-read soon (it skips read-for-ownership traffic), a loss if the data is reused. NT stores need an aligned destination; the runtime falls back to an ordinary store when a buffer isn't aligned, so it is always correct but only speeds up aligned buffers (large arrays are commonly page-aligned in practice). Off by default.

## Runtime library tour

| Area | Types / members |
|---|---|
| Varying types | `VFloat`, `VInt`, `VDouble`, `VLong`, arithmetic/comparison operators, bitwise, shifts, `Load/Store(+Masked)`, `Select`/`Blend`, conversions, `AsInt()`/`AsFloat()` bitcasts, hardware FMA `MulAdd`, variable per-lane shifts (`VInt.Shift*Variable`) |
| Cross-gang-width | `VDouble2` / `VLong2`, double / 64-bit-int precision at full float/int-gang width (two `VDouble`/`VLong` halves, full operator/math surface; `FromInt`/`FromFloat`↔`ToInt`/`ToFloat`); `VInt.Widen/Narrow` ↔ `VLong`, `VMaskD.Widen`/`VMask.Narrow` |
| Masks | `VMask`, `VMaskD`, `& \| ! ^`, `Any()`, `AllActive()`, `FirstN(n)`, `AndNot` |
| Execution | `Spmd.Foreach`, `ParallelForeach`, `Foreach2D`, `ParallelForeach2D`, `Spmd.While` (+ `LoopState.Break/Continue/Return`), `Spmd.If`, `Spmd.Coherent` |
| Math | `VectorMath`, `Sqrt Rsqrt Rcp Abs Min Max Clamp Lerp Floor Round Truncate Exp Log Pow Sin Cos Tan Tanh Sigmoid Atan Atan2 Asin Acos Cbrt Hypot`, float + double |
| Reductions | `Reduce.Add/Min/Max` for `VFloat`/`VInt`/`VDouble`, with masked overloads |
| Indexed memory | `Memory.Gather`/`Scatter` for float/int/double, AVX2 hardware gather, movemask-driven scatter |
| Layout helpers | `Memory.Transpose`/`Transposed` (cache-blocked, float/int/double); `SoaFloat2/3` + `FromInterleaved`/`ToInterleaved` de-interleavers |
| Cross-lane | `Lanes.Shuffle` (hardware permutes), `Rotate`, `Broadcast`, `ShiftLanes` |
| Data layout | `SoaFloat2`, `SoaFloat3` |
| Testing | `KernelVerifier.AssertMatches`, asserts a kernel against a scalar reference |

Accuracy: `VectorMath` transcendentals are branch-free polynomial/rational approximations, not bit-identical to `MathF`/`Math`, roughly **1e-6 relative for float, 1e-13 for double** over their stated ranges. `Log` requires x > 0; float `Exp` clamps to [-87, 88], double to [-708, 709]; double `Sin`/`Cos` hold accuracy for |x| < ~1e6. Use `KernelVerifier` with an appropriate tolerance.

**FMA:** on float/double lanes the generator fuses `acc += x*y` (dot product, matmul, sum of squares) and `a*b + c` (AXPY, Lerp, polynomials) into a hardware **fused multiply-add**, one instruction, higher throughput, and a single rounding. Because C# does **not** contract `a*b+c` by default, a `_Simd` result can differ from the naive scalar reference by ~1 ULP (FMA is *more* accurate, one rounding instead of two). Compare with a small tolerance, not bit-exact. FMA is used only where no operand needs a precision-changing coercion, so it still tracks the scalar computation's precision.

**Reduction unrolling:** a straight-line reduction loop (`sum += …`, `Math.Min/Max`) is unrolled 4× with **four independent accumulators**, then combined and horizontally reduced after the loop. A single accumulator is latency-bound (each `+=`/FMA waits on the previous); independent accumulators fill the pipeline, so wide compute-bound reductions (dot product, matmul, sum of squares) run closer to peak. Transparent, no source change, results still match within the FMA tolerance above.

## Performance guidance

1. **Profile first.** SIMD only helps compute-bound loops. If you're bound on cache misses or memory bandwidth, fix the data layout before vectorizing.
2. **Use SoA layout.** `x[i], y[i], z[i]` in separate arrays loads with one instruction each; an array of `struct Vec3` needs gathers. Use `SoaFloat3` (or `SoaFloat3.FromInterleaved` to de-interleave AoS data once), or roll your own.
3. **Keep gangs busy.** Masked branches execute *both* sides. If one side is expensive and rarely taken, mark it coherent, `if (Spmd.Coherent(cond))`, so whole gangs can skip it.
4. **Avoid gather/scatter when a layout change would make access contiguous**, even hardware gathers are several times slower than contiguous loads. Classic case: matmul's `b[k, col]` is a column gather; `Memory.Transpose(b, bT)` once, then `bT[col, k]` is contiguous. The transpose is O(size) and amortizes across every reuse (see `MatMulBench` for the measured win).
5. **Amortize threading.** `ParallelForeach`/`_ParallelSimd` default to ≥16K elements per task; below that they run single-threaded because scheduling overhead would dominate.
6. **Prefer the generator for hot loops.** It emits full-width loads with a scalar tail, keeps state in registers, and never allocates. Hand-written runtime-API kernels are easy to get subtly wrong (masked access on every gang, closures allocated per gang), the benchmark suite runs entirely on generated kernels for this reason.
7. **Benchmark with [BenchmarkDotNet](https://benchmarkdotnet.org/)** in Release: `dotnet run -c Release --project benchmarks/IspcSharp.Benchmarks -- --filter '*'`.

## Limitations

- **`return` in `[Spmd]` kernels is per-lane (ISPC semantics), not scalar C# semantics**, a bare `return;` retires the *current element* (skips the rest of its body, exits its loops) while other elements keep processing. In the scalar method, that same `return` exits the whole loop and stops all later elements, so once a return fires, the scalar method is no longer a per-element reference for the elements after it. Use `continue` when you want identical scalar/SIMD behavior; `return <value>` inside the loop is rejected (value returns go after the loop).
- **No true x86 scatter instruction**, .NET doesn't expose `vscatterdps` (through .NET 10, all scatter intrinsics are ARM SVE). `Memory.Scatter` uses the fastest managed path: one hardware movemask (AVX-512 kmask / AVX / NEON), then active-lane-only stores.
- **Gather is hardware where possible, but latency-bound**, `Memory.Gather` uses the AVX2 `vpgatherdd`/`vgatherqpd` family (also for **16-wide AVX-512 gangs**, by running the 256-bit gather on each half, .NET exposes no 512-bit gather intrinsic); other widths use a portable path that copies indices to the stack once and indexes them, avoiding per-lane `Vector<T>` spills. Even so, a gather over a large random-access table is bound by memory *latency*, not the instruction, for a table that overflows cache, hardware gather is barely faster than a scalar indexed load. The real fix is a contiguous layout (transpose / SoA), not the gather itself.
- **Double precision is half the lane count**, a `double`/`long` gang is half as wide as `float`/`int`, so expect roughly half the speedup; use it when you need the range/accuracy. The double transcendentals (`Exp`/`Log`/`Sin`/… and everything built on them, `Pow`, `Cbrt`, `Tanh`, `Cos`, `Tan`) do their range reduction with a branchless magic-number round, **not** `Vector.ConvertToInt64`/`ConvertToDouble`, those have no encoding before AVX-512DQ and would otherwise scalarize to a per-lane loop on Zen 1–3 / Haswell–Comet Lake (it made the double `Exp` ~3× slower than it should be). One residual spot: `Math.Floor`/`Round`/`Truncate` and `(int)`/`(long)` casts *inside a double kernel* still use those conversions (they need correct directed rounding across the full 2⁵² range), so a double kernel dominated by casts/floor rather than transcendentals sees less benefit on pre-AVX-512 hardware.
- **`Vector<T>` hides low-level control**, no explicit AVX-512 mask registers, no FMA contraction guarantee, no compile-time ISA specialization; lane width is chosen at runtime.
- **No *automatic* SoA / transpose**, and this is deliberate (ISPC doesn't either). If data is physically AoS/row-major in memory, accessing one field/column across lanes *is* a gather; the only fix is to physically reorder into a contiguous buffer, which means **allocating and an O(size) copy pass**, a space/time tradeoff a compiler shouldn't make silently (and generated kernels otherwise never allocate). So layout is an explicit choice, but the building blocks are provided: **`Memory.Transpose`/`Transposed`** (turn a matmul's `b[k,col]` gather into a contiguous `bT[col,k]` load, one transpose amortizes across all output rows), **`SoaFloat2/3.FromInterleaved`/`ToInterleaved`** (de-interleave AoS `x,y,z,x,y,z…` into contiguous arrays once), and the `[Spmd]` 2-D array support to consume them. The `ISPC100` analyzer warns you where an AoS gather is happening.
- **Only the innermost `Spmd.Range` loop vectorizes**, its body must be per-lane parallel. A loop-carried recurrence (e.g. an FFT's `w *= wStep` twiddle update, or a running prefix sum) can't be auto-vectorized by any SPMD compiler; rewrite it in closed form (compute each lane's value directly, like `cos(k·angle)`). Uniform scaffolding around the loop has no such restriction, it's ordinary scalar C#.
- **`long` lanes accelerate partially**, add/subtract/bitwise/compare/shift/min-max vectorize well on AVX2+, but there's no single-instruction 64-bit packed multiply before AVX-512DQ (`vpmullq`; else a JIT sequence) and integer divide is scalar on all ISAs. So `long` kernels fly for bitwise/additive/hashing/indexing work and are only partly accelerated for multiply/divide-heavy code.
- **`float[,]`/`int[,]`/`double[,]`/`long[,]` kernels are `_Simd`-only**, the flat row-major span view can't be captured across threads, so no `_ParallelSimd` (parallelize the outer loop yourself).
- **Integer `/` `%`, masked tail load/store, and non-contiguous 2-D access are per-lane loops**, there's no SIMD instruction for them; expect scalar-ish speed on those specific ops.
- The generator matches `Math`/`Span` symbols syntactically rather than via full semantic-model resolution, so exotic shadowing of those names could confuse it.

## Repository layout

```
src/IspcSharp/                  Runtime library
src/IspcSharp.Generators/       Roslyn source generator + AoS analyzer
samples/IspcSharp.Samples/      Tonemap, reduction, Newton-sqrt (varying while), Mandelbrot
benchmarks/IspcSharp.Benchmarks BenchmarkDotNet suite on generated kernels: branchless, reductions,
                                transcendental, divergent while, 2D Mandelbrot, FFT, matrix
                                multiply, bandwidth-bound control, grid route finding
tests/IspcSharp.Tests/          xUnit suite (272 tests) covering every feature above, in both
                                hardware-SIMD and software-fallback modes
.github/workflows/ci.yml        Test matrix: Linux/Windows x64 + macOS ARM64 (NEON), plus an
                                on-demand benchmark job
```

## License

MIT
