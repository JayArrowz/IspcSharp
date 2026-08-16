using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp;

/// <summary>
/// Vectorized math, the equivalent of ISPC's standard library.
/// <see cref="Vector{T}"/> only provides Sqrt/Abs/Min/Max natively, so the transcendentals
/// here are branch-free polynomial approximations (the same technique ISPC's stdlib uses),
/// accurate to roughly single-precision ULP levels over their stated ranges.
/// All functions operate on every lane unconditionally; combine with masks at the call site.
/// </summary>
public static class VectorMath
{
    private const float Ln2 = 0.6931471805599453f;
    private const float Log2E = 1.4426950408889634f;
    private const float Pi = 3.14159265358979323846f;
    private const float TwoPi = 6.28318530717958647692f;
    private const float InvTwoPi = 0.15915494309189533577f;
    private const int MantissaShift = 1 << 23;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat Sqrt(VFloat x) => new(Vector.SquareRoot(x.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat Abs(VFloat x) => new(Vector.Abs(x.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat Min(VFloat a, VFloat b) => new(Vector.Min(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat Max(VFloat a, VFloat b) => new(Vector.Max(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt Abs(VInt x) => new(Vector.Abs(x.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt Min(VInt a, VInt b) => new(Vector.Min(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt Max(VInt a, VInt b) => new(Vector.Max(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong Abs(VLong x) => VLong.Abs(x);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong Min(VLong a, VLong b) => VLong.Min(a, b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong Max(VLong a, VLong b) => VLong.Max(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Clamp(VFloat x, VFloat lo, VFloat hi) => Min(Max(x, lo), hi);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Clamp(VInt x, VInt lo, VInt hi) => Min(Max(x, lo), hi);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Lerp(VFloat a, VFloat b, VFloat t) => a + ((b - a) * t);

    /// <summary>
    /// Approximate reciprocal 1/x.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Rcp(VFloat x) => VFloat.One / x;

    /// <summary>
    /// Approximate reciprocal square root 1/sqrt(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Rsqrt(VFloat x) => VFloat.One / Sqrt(x);

    /// <summary>
    /// Per-lane floor. Correct for |x| &lt; 2^31.
    /// </summary>
    public static VFloat Floor(VFloat x)
    {
        var truncated = VInt.FromFloatTruncate(x).ToFloat();          // rounds toward zero
        var needsFix = truncated > x;                                  // negative non-integers
        return VFloat.Select(needsFix, truncated - VFloat.One, truncated);
    }

    /// <summary>
    /// Per-lane round-to-nearest (half away from zero). Correct for |x| &lt; 2^31.
    /// </summary>
    public static VFloat Round(VFloat x)
    {
        var half = new VFloat(0.5f);
        var sign = VFloat.Select(x < VFloat.Zero, new VFloat(-1f), VFloat.One);
        return VInt.FromFloatTruncate(x + (sign * half)).ToFloat();
    }

    /// <summary>
    /// Per-lane e^x. Range-reduced (x = k*ln2 + r) + degree-5 polynomial + exponent bit assembly.
    /// Accurate to ~1-2 ulp for x in [-87, 88]; clamps outside to avoid Inf/denormal bit tricks.
    /// </summary>
    public static VFloat Exp(VFloat x)
    {
        x = Clamp(x, new VFloat(-87.0f), new VFloat(88.0f));

        var k = Round(x * new VFloat(Log2E));                 // nearest integer multiple of ln2
        var r = x - (k * new VFloat(Ln2));                      // r in [-ln2/2, ln2/2]

        // exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5!
        var r2 = r * r;
        var p = new VFloat(1.0f / 120.0f);
        p = VFloat.MulAdd(p, r, new VFloat(1.0f / 24.0f));
        p = VFloat.MulAdd(p, r, new VFloat(1.0f / 6.0f));
        p = VFloat.MulAdd(p, r, new VFloat(0.5f));
        p = VFloat.MulAdd(p, r2, r + VFloat.One);

        // 2^k via IEEE754 bits: ((k + 127) << 23), shift done as integer multiply.
        var ki = VInt.FromFloatTruncate(k);
        var bits = (ki + 127) * MantissaShift;
        var scale = new VFloat(Vector.AsVectorSingle(bits.V));

        return p * scale;
    }

    /// <summary>
    /// Per-lane natural log. Valid for x &gt; 0 (lanes &lt;= 0 produce garbage, mask them).
    /// Mantissa/exponent split via IEEE754 bits + atanh-form polynomial; ~2 ulp.
    /// </summary>
    public static VFloat Log(VFloat x)
    {
        var bits = new VInt(Vector.AsVectorInt32(x.V));

        // exponent = (bits >> 23) - 127 (bits are positive for x > 0, so a logical shift —
        // vpsrld, one instruction — replaces the former per-lane scalar divide).
        var e = VInt.ShiftRightLogical(bits, 23) - 127;

        // mantissa in [1, 2): keep fraction bits, force exponent to 0 (0x3F800000)
        var mBits = (bits & new VInt(0x007FFFFF)) | new VInt(0x3F800000);
        var m = new VFloat(Vector.AsVectorSingle(mBits.V));

        // Normalize m to [sqrt(0.5), sqrt(2)) for a symmetric polynomial around 1.
        var tooBig = m > new VFloat(1.41421356f);
        m = VFloat.Select(tooBig, m * new VFloat(0.5f), m);
        var ef = VInt.Select(tooBig, e + 1, e).ToFloat();

        // log(m) via atanh identity: log(m) = 2*atanh(t), t = (m-1)/(m+1)
        var t = (m - VFloat.One) / (m + VFloat.One);
        var t2 = t * t;
        var p = new VFloat(2.0f / 9.0f);
        p = VFloat.MulAdd(p, t2, new VFloat(2.0f / 7.0f));
        p = VFloat.MulAdd(p, t2, new VFloat(2.0f / 5.0f));
        p = VFloat.MulAdd(p, t2, new VFloat(2.0f / 3.0f));
        p = VFloat.MulAdd(p, t2, new VFloat(2.0f));
        var logM = p * t;

        return (ef * new VFloat(Ln2)) + logM;
    }

    /// <summary>
    /// Per-lane x^y for x &gt; 0 (uses exp(y*log(x))).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Pow(VFloat x, VFloat y) => Exp(y * Log(x));

    /// <summary>
    /// Per-lane sine. Range-reduced to [-π/2, π/2] then degree-9 minimax polynomial; ~1e-6 abs error.
    /// </summary>
    public static VFloat Sin(VFloat x)
    {
        // Reduce to [-π, π]: x -= 2π * round(x / 2π)
        var k = Round(x * new VFloat(InvTwoPi));
        var r = x - (k * new VFloat(TwoPi));

        // Further reduce to [-π/2, π/2] for polynomial accuracy.
        //   sin(r) = sin(π - r)   for r in [π/2, π]
        //   sin(r) = sin(-π - r)  for r in [-π, -π/2]
        var pi = new VFloat(Pi);
        var halfPi = new VFloat(Pi * 0.5f);
        var foldHigh = r > halfPi;
        var foldLow = r < -halfPi;
        var needsFold = foldHigh | foldLow;
        var folded = VFloat.Select(foldHigh, pi - r, -pi - r);
        r = VFloat.Select(needsFold, folded, r);

        var r2 = r * r;
        var p = new VFloat(2.6019031e-6f);       // ~ +x⁹/9!
        p = VFloat.MulAdd(p, r2, new VFloat(-1.9807418e-4f)); // ~ -x⁷/7!
        p = VFloat.MulAdd(p, r2, new VFloat(8.3330173e-3f));  // ~ +x⁵/5!
        p = VFloat.MulAdd(p, r2, new VFloat(-1.6666568e-1f)); // ~ -x³/3!
        return VFloat.MulAdd(p * r2, r, r);
    }

    /// <summary>
    /// Per-lane cosine (sin(x + π/2)).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Cos(VFloat x) => Sin(x + new VFloat(Pi * 0.5f));

    /// <summary>
    /// Per-lane tangent (sin/cos, loses accuracy near odd multiples of π/2).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Tan(VFloat x) => Sin(x) / Cos(x);

    /// <summary>
    /// Per-lane hyperbolic tangent via exp; clamps input to keep exp in range.
    /// </summary>
    public static VFloat Tanh(VFloat x)
    {
        x = Clamp(x, new VFloat(-10f), new VFloat(10f));
        var e2x = Exp(x * new VFloat(2f));
        return (e2x - VFloat.One) / (e2x + VFloat.One);
    }

    /// <summary>
    /// Sigmoid 1/(1+e^-x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Sigmoid(VFloat x) => VFloat.One / (VFloat.One + Exp(-x));

    /// <summary>
    /// Per-lane arctangent. Cephes-style two-step range reduction (fold at tan(3π/8) and
    /// tan(π/8)) + degree-7 polynomial; ~1e-7 abs error.
    /// </summary>
    public static VFloat Atan(VFloat x)
    {
        var sign = x < VFloat.Zero;
        var t = Abs(x);

        // Reduce to [0, tan(π/8)]:
        //   t > tan(3π/8) (≈2.414): atan(t) = π/2 + atan(-1/t)
        //   t > tan(π/8)  (≈0.414): atan(t) = π/4 + atan((t-1)/(t+1))
        var big = t > new VFloat(2.414213562373095f);
        var mid = t > new VFloat(0.4142135623730950f);
        var baseAngle = VFloat.Select(big, new VFloat(Pi * 0.5f),
                        VFloat.Select(mid, new VFloat(Pi * 0.25f), VFloat.Zero));
        var u = VFloat.Select(big, new VFloat(-1f) / t,
                VFloat.Select(mid, (t - VFloat.One) / (t + VFloat.One), t));

        var z = u * u;
        var p = new VFloat(8.05374449538e-2f);
        p = VFloat.MulAdd(p, z, new VFloat(-1.38776856032e-1f));
        p = VFloat.MulAdd(p, z, new VFloat(1.99777106478e-1f));
        p = VFloat.MulAdd(p, z, new VFloat(-3.33329491539e-1f));
        var r = baseAngle + VFloat.MulAdd(p * z, u, u);

        return VFloat.Select(sign, -r, r);
    }

    /// <summary>
    /// Per-lane atan2(y, x) with full quadrant handling. (0, 0) lanes return 0.
    /// </summary>
    public static VFloat Atan2(VFloat y, VFloat x)
    {
        var r = Atan(y / x);
        // Left half-plane: shift by ±π toward y's sign.
        var yNeg = y < VFloat.Zero;
        var shift = VFloat.Select(yNeg, new VFloat(-Pi), new VFloat(Pi));
        r = VFloat.Select(x < VFloat.Zero, r + shift, r);
        // Origin: y/x = 0/0 = NaN, define as 0.
        var origin = VFloat.Eq(x, VFloat.Zero) & VFloat.Eq(y, VFloat.Zero);
        return VFloat.Select(origin, VFloat.Zero, r);
    }

    /// <summary>
    /// Per-lane arcsine for x in [-1, 1] (π/2 − acos, so asin/acos stay complementary).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Asin(VFloat x) => new VFloat(Pi * 0.5f) - Acos(x);

    /// <summary>
    /// Per-lane arccosine for x in [-1, 1] (via the atan half-angle identity).
    /// </summary>
    public static VFloat Acos(VFloat x)
    {
        // acos(x) = 2·atan(sqrt((1-x)/(1+x))), well-conditioned over [-1, 1].
        var q = (VFloat.One - x) / (VFloat.One + x);
        var r = Atan(Sqrt(q)) * new VFloat(2f);
        // x = -1: q = inf → sqrt = inf → atan = π/2 → r = π. Guard NaN from -1 exactly.
        return VFloat.Select(x <= new VFloat(-1f), new VFloat(Pi), r);
    }

    /// <summary>
    /// Per-lane cube root (handles negatives; exp/log seed + one Newton step).
    /// </summary>
    public static VFloat Cbrt(VFloat x)
    {
        var sign = VFloat.Select(x < VFloat.Zero, new VFloat(-1f), VFloat.One);
        var a = Abs(x);
        var y = Exp(Log(a) * new VFloat(1f / 3f));
        // Newton polish: y ← (2y + a/y²) / 3.
        y = (y + y + (a / (y * y))) * new VFloat(1f / 3f);
        var r = sign * y;
        return VFloat.Select(VFloat.Eq(x, VFloat.Zero), VFloat.Zero, r);
    }

    /// <summary>
    /// Per-lane hypot: sqrt(x² + y²) without intermediate overflow.
    /// </summary>
    public static VFloat Hypot(VFloat x, VFloat y)
    {
        var a = Abs(x);
        var b = Abs(y);
        var m = Max(a, b);
        var q = Min(a, b) / m;                       // NaN when m == 0, masked below
        var r = m * Sqrt(VFloat.MulAdd(q, q, VFloat.One));
        return VFloat.Select(VFloat.Eq(m, VFloat.Zero), VFloat.Zero, r);
    }

    private const double LnTwoD = 0.6931471805599453;
    private const double Log2ED = 1.4426950408889634;
    private const double PiD = 3.141592653589793;
    // ln2 split (Cody-Waite): LnTwoHiD + LnTwoLoD == ln2 to ~100 bits.
    private const double LnTwoHiD = 6.93145751953125e-1;
    private const double LnTwoLoD = 1.42860682030941723212e-6;
    // 2π split: TwoPiHiD has 33 significant bits (k * TwoPiHiD stays exact for k < 2^20),
    // TwoPiHiD + TwoPiLoD == 2π to ~86 bits.
    private const double TwoPiHiD = 6.28318530693650245668;
    private const double TwoPiLoD = 2.43084020260247689973e-10;
    private const double InvTwoPiD = 0.15915494309189533577;

    // (x << 52) == x * 2^52 for long lanes, double's exponent-field shift.
    private const long MantissaShiftD = 1L << 52;

    // "Magic number" for branchless double→integer rounding. Adding 2^52+2^51 to a double
    // (|value| < 2^51) snaps it onto integer boundaries, the FPU does round-to-nearest-even —
    // and the low mantissa bits then hold the integer. This lets the double transcendentals
    // round WITHOUT Vector.ConvertToInt64/ConvertToDouble, which have no AVX2 encoding (they
    // need AVX-512DQ) and otherwise scalarize to a per-lane loop on Zen 1–3 / Haswell–Comet Lake.
    private const double RoundMagicD = 6755399441055744.0;   // 2^52 + 2^51
    private static readonly Vector<long> RoundMagicBits =
        new(BitConverter.DoubleToInt64Bits(RoundMagicD));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble Sqrt(VDouble x) => VDouble.Sqrt(x);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble Abs(VDouble x) => VDouble.Abs(x);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble Min(VDouble a, VDouble b) => VDouble.Min(a, b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble Max(VDouble a, VDouble b) => VDouble.Max(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Clamp(VDouble x, VDouble lo, VDouble hi) => Min(Max(x, lo), hi);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Lerp(VDouble a, VDouble b, VDouble t) => a + ((b - a) * t);

    /// <summary>
    /// Approximate reciprocal 1/x.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Rcp(VDouble x) => VDouble.One / x;

    /// <summary>
    /// Approximate reciprocal square root 1/sqrt(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Rsqrt(VDouble x) => VDouble.One / Sqrt(x);

    /// <summary>
    /// Per-lane floor. Correct for |x| &lt; 2^52.
    /// </summary>
    public static VDouble Floor(VDouble x)
    {
        var truncated = new VDouble(Vector.ConvertToDouble(Vector.ConvertToInt64(x.V))); // rounds toward zero
        var needsFix = truncated > x;                                                     // negative non-integers
        return VDouble.Select(needsFix, truncated - VDouble.One, truncated);
    }

    /// <summary>
    /// Per-lane truncation toward zero. Correct for |x| &lt; 2^52. This is what
    /// <c>(int)</c>/<c>(long)</c> casts lower to in double [Spmd] kernels.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Truncate(VDouble x)
        => new(Vector.ConvertToDouble(Vector.ConvertToInt64(x.V)));

    /// <summary>
    /// Per-lane round-to-nearest (half away from zero). Correct for |x| &lt; 2^52.
    /// </summary>
    public static VDouble Round(VDouble x)
    {
        var half = new VDouble(0.5);
        var sign = VDouble.Select(x < VDouble.Zero, new VDouble(-1d), VDouble.One);
        return new VDouble(Vector.ConvertToDouble(Vector.ConvertToInt64((x + (sign * half)).V)));
    }

    /// <summary>
    /// Per-lane e^x. Range-reduced (x = k*ln2 + r, split-constant subtraction) + a divide-free
    /// degree-11 Taylor polynomial for exp(r) + exponent bit assembly. Accurate to ~1e-14 relative
    /// (a few ulp) for x in [-708, 709]; clamps outside to avoid Inf/denormal bit tricks.
    ///
    /// The reduced-argument evaluation is a polynomial, not a Cephes P/Q rational: the rational
    /// form needed a vector divide (the slowest lane op, and only 2-4 lanes wide for doubles),
    /// which made the double exp — and everything built on it (Sigmoid, Tanh, Pow, Tonemap) —
    /// throughput-bound on the divide unit. Two independent Horner chains in r² (even/odd powers)
    /// keep the dependency chain short and use both FMA ports, matching the float exp's shape.
    /// Degree 11 (not 13) is the throughput sweet spot: it holds ~1e-14 relative here (well inside
    /// the 1e-13 double contract) while the two extra terms of degree 13 only cost FMA throughput.
    /// </summary>
    public static VDouble Exp(VDouble x)
    {
        x = Clamp(x, new VDouble(-708.0), new VDouble(709.0));

        // k = round(x·log2e), as BOTH a double (for the reduced argument r) and an int64 (for
        // 2^k), via the magic constant, no double↔int64 conversion. Round-to-nearest-even here
        // vs. away-from-zero is harmless: at an exact half, k shifts by 1 but 2^k·exp(r∓ln2)
        // gives the identical result, and r stays inside [-ln2/2, ln2/2].
        var t = (x * new VDouble(Log2ED)) + new VDouble(RoundMagicD);
        var k = t - new VDouble(RoundMagicD);
        var ki = Vector.AsVectorInt64(t.V) - RoundMagicBits + new Vector<long>(1023L);
        var r = x - (k * new VDouble(LnTwoHiD)) - (k * new VDouble(LnTwoLoD)); // r in [-ln2/2, ln2/2]

        // exp(r) = Σ r^n/n! to degree 11 (truncation ~6e-15 over |r| <= ln2/2, well under the
        // 1e-13 double contract). Evaluated as even(r²) + r·odd(r²), two independent chains.
        var r2 = r * r;
        var po = new VDouble(1.0 / 39916800.0);                // 1/11!
        po = VDouble.MulAdd(po, r2, new VDouble(1.0 / 362880.0));     // 1/9!
        po = VDouble.MulAdd(po, r2, new VDouble(1.0 / 5040.0));       // 1/7!
        po = VDouble.MulAdd(po, r2, new VDouble(1.0 / 120.0));        // 1/5!
        po = VDouble.MulAdd(po, r2, new VDouble(1.0 / 6.0));          // 1/3!
        po = VDouble.MulAdd(po, r2, VDouble.One);                     // 1/1!
        var pe = new VDouble(1.0 / 3628800.0);                 // 1/10!
        pe = VDouble.MulAdd(pe, r2, new VDouble(1.0 / 40320.0));      // 1/8!
        pe = VDouble.MulAdd(pe, r2, new VDouble(1.0 / 720.0));        // 1/6!
        pe = VDouble.MulAdd(pe, r2, new VDouble(1.0 / 24.0));         // 1/4!
        pe = VDouble.MulAdd(pe, r2, new VDouble(0.5));                // 1/2!
        pe = VDouble.MulAdd(pe, r2, VDouble.One);                     // 1/0!
        var expR = VDouble.MulAdd(r, po, pe);                         // even(r²) + r·odd(r²)

        // 2^k via IEEE754 bits: (k + 1023) << 52.
#if NET8_0_OR_GREATER
        // A real vector shift (vpsllq), one instruction on any AVX2+.
        var bits = Vector.ShiftLeft(ki, 52);
#else
        var bits = ki * MantissaShiftD;
#endif
        var scale = new VDouble(Vector.AsVectorDouble(bits));

        return expR * scale;
    }

    /// <summary>
    /// Per-lane natural log. Valid for x &gt; 0 (lanes &lt;= 0 produce garbage, mask them).
    /// Mantissa/exponent split via IEEE754 bits + atanh-form polynomial; ~1-2 ulp.
    /// </summary>
    public static VDouble Log(VDouble x)
    {
        var bits = Vector.AsVectorInt64(x.V);

        // exponent = (bits >> 52) - 1023 (bits are positive for x > 0, so a logical shift —
        // vpsrlq, one AVX2 instruction, replaces the former per-lane scalar 64-bit divide).
#if NET8_0_OR_GREATER
        var e = Vector.AsVectorInt64(Vector.ShiftRightLogical(Vector.AsVectorUInt64(bits), 52)) - new Vector<long>(1023L);
#else
        var e = Divide(bits, MantissaShiftD) - new Vector<long>(1023L);
#endif

        // mantissa in [1, 2): keep fraction bits, force exponent to 0 (0x3FF00000_00000000)
        var mBits = (bits & new Vector<long>(0x000FFFFFFFFFFFFFL)) | new Vector<long>(0x3FF0000000000000L);
        var m = new VDouble(Vector.AsVectorDouble(mBits));

        // Normalize m to [sqrt(0.5), sqrt(2)) for a symmetric polynomial around 1.
        var tooBig = m > new VDouble(1.4142135623730951);
        m = VDouble.Select(tooBig, m * new VDouble(0.5), m);
        var eAdj = Vector.ConditionalSelect(tooBig.Bits, e + Vector<long>.One, e);
        // int64 exponent → double via the reverse magic-number trick (no ConvertToDouble, which
        // scalarizes without AVX-512DQ): reinterpret (eAdj + magicBits) then subtract the magic.
        var ef = new VDouble(Vector.AsVectorDouble(eAdj + RoundMagicBits)) - new VDouble(RoundMagicD);

        // log(m) via atanh identity: log(m) = 2*atanh(t), t = (m-1)/(m+1), |t| <= 0.1716.
        var t = (m - VDouble.One) / (m + VDouble.One);
        var t2 = t * t;
        var p = new VDouble(2.0 / 21.0);
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 19.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 17.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 15.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 13.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 11.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 9.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 7.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 5.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0 / 3.0));
        p = VDouble.MulAdd(p, t2, new VDouble(2.0));
        var logM = p * t;

        // e*ln2 with the split constant so the product doesn't erode logM's precision.
        return VDouble.MulAdd(ef, new VDouble(LnTwoHiD), VDouble.MulAdd(ef, new VDouble(LnTwoLoD), logM));
    }

    /// <summary>
    /// Per-lane x^y for x &gt; 0 (uses exp(y*log(x))).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Pow(VDouble x, VDouble y) => Exp(y * Log(x));

    /// <summary>
    /// Per-lane sine. Range-reduced to [-π/2, π/2] (split-constant 2π subtraction, exact
    /// for |x| &lt; ~6e6) then degree-19 polynomial; ~1e-15 abs error over that range.
    /// </summary>
    public static VDouble Sin(VDouble x)
    {
        // Reduce to [-π, π]: x -= 2π * round(x / 2π), with 2π split so k*hi stays exact.
        // round(x/2π) via the magic constant (no ConvertToInt64/Double round-trip); k differing
        // by 1 at an exact half is harmless, it shifts r by a full 2π period.
        var kt = (x * new VDouble(InvTwoPiD)) + new VDouble(RoundMagicD);
        var k = kt - new VDouble(RoundMagicD);
        var r = x - (k * new VDouble(TwoPiHiD)) - (k * new VDouble(TwoPiLoD));

        // Further reduce to [-π/2, π/2] for polynomial accuracy.
        //   sin(r) = sin(π - r)   for r in [π/2, π]
        //   sin(r) = sin(-π - r)  for r in [-π, -π/2]
        var pi = new VDouble(PiD);
        var halfPi = new VDouble(PiD * 0.5);
        var foldHigh = r > halfPi;
        var foldLow = r < -halfPi;
        var needsFold = foldHigh | foldLow;
        var folded = VDouble.Select(foldHigh, pi - r, -pi - r);
        r = VDouble.Select(needsFold, folded, r);

        // sin(r) = r + r·r²·(a1 + r²·(a2 + ...)), aₖ = (-1)ᵏ/(2k+1)!  through 1/19!.
        var r2 = r * r;
        var p = new VDouble(-8.220635246624330e-18);            // -1/19!
        p = VDouble.MulAdd(p, r2, new VDouble(2.811457254345521e-15));   // +1/17!
        p = VDouble.MulAdd(p, r2, new VDouble(-7.647163731819816e-13));  // -1/15!
        p = VDouble.MulAdd(p, r2, new VDouble(1.605904383682161e-10));   // +1/13!
        p = VDouble.MulAdd(p, r2, new VDouble(-2.505210838544172e-8));   // -1/11!
        p = VDouble.MulAdd(p, r2, new VDouble(2.755731922398589e-6));    // +1/9!
        p = VDouble.MulAdd(p, r2, new VDouble(-1.984126984126984e-4));   // -1/7!
        p = VDouble.MulAdd(p, r2, new VDouble(8.333333333333333e-3));    // +1/5!
        p = VDouble.MulAdd(p, r2, new VDouble(-1.666666666666667e-1));   // -1/3!
        return VDouble.MulAdd(p * r2, r, r);
    }

    /// <summary>
    /// Per-lane cosine (sin(x + π/2)).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Cos(VDouble x) => Sin(x + new VDouble(PiD * 0.5));

    /// <summary>
    /// Per-lane tangent (sin/cos, loses accuracy near odd multiples of π/2).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Tan(VDouble x) => Sin(x) / Cos(x);

    /// <summary>
    /// Per-lane hyperbolic tangent via exp; clamps input to keep exp in range.
    /// </summary>
    public static VDouble Tanh(VDouble x)
    {
        x = Clamp(x, new VDouble(-20d), new VDouble(20d));
        var e2x = Exp(x * new VDouble(2d));
        return (e2x - VDouble.One) / (e2x + VDouble.One);
    }

    /// <summary>
    /// Sigmoid 1/(1+e^-x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Sigmoid(VDouble x) => VDouble.One / (VDouble.One + Exp(-x));

    /// <summary>
    /// Per-lane arctangent. Cephes rational approximation with two-step range reduction
    /// (fold at tan(3π/8) and 0.66); ~1e-15 relative.
    /// </summary>
    public static VDouble Atan(VDouble x)
    {
        var sign = x < VDouble.Zero;
        var t = Abs(x);

        var big = t > new VDouble(2.414213562373095);   // tan(3π/8)
        var mid = t > new VDouble(0.66);
        var baseAngle = VDouble.Select(big, new VDouble(PiD * 0.5),
                        VDouble.Select(mid, new VDouble(PiD * 0.25), VDouble.Zero));
        var u = VDouble.Select(big, new VDouble(-1d) / t,
                VDouble.Select(mid, (t - VDouble.One) / (t + VDouble.One), t));

        // atan(u) ≈ u + u·z·P(z)/Q(z), z = u² (Cephes atan, |u| ≤ 0.66).
        var z = u * u;
        var p = new VDouble(-8.750608600031904122785e-1);
        p = VDouble.MulAdd(p, z, new VDouble(-1.615753718733365076637e1));
        p = VDouble.MulAdd(p, z, new VDouble(-7.500855792314704667340e1));
        p = VDouble.MulAdd(p, z, new VDouble(-1.228866684490136173410e2));
        p = VDouble.MulAdd(p, z, new VDouble(-6.485021904942025371773e1));
        var q = z + new VDouble(2.485846490142306297962e1);
        q = VDouble.MulAdd(q, z, new VDouble(1.650270098316988542046e2));
        q = VDouble.MulAdd(q, z, new VDouble(4.328810604912902668951e2));
        q = VDouble.MulAdd(q, z, new VDouble(4.853903996359136964868e2));
        q = VDouble.MulAdd(q, z, new VDouble(1.945506571482613964425e2));
        var r = baseAngle + VDouble.MulAdd(z * (p / q), u, u);

        return VDouble.Select(sign, -r, r);
    }

    /// <summary>
    /// Per-lane atan2(y, x) with full quadrant handling. (0, 0) lanes return 0.
    /// </summary>
    public static VDouble Atan2(VDouble y, VDouble x)
    {
        var r = Atan(y / x);
        var yNeg = y < VDouble.Zero;
        var shift = VDouble.Select(yNeg, new VDouble(-PiD), new VDouble(PiD));
        r = VDouble.Select(x < VDouble.Zero, r + shift, r);
        var origin = VDouble.Eq(x, VDouble.Zero) & VDouble.Eq(y, VDouble.Zero);
        return VDouble.Select(origin, VDouble.Zero, r);
    }

    /// <summary>
    /// Per-lane arcsine for x in [-1, 1] (π/2 − acos, so asin/acos stay complementary).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Asin(VDouble x) => new VDouble(PiD * 0.5) - Acos(x);

    /// <summary>
    /// Per-lane arccosine for x in [-1, 1] (via the atan half-angle identity).
    /// </summary>
    public static VDouble Acos(VDouble x)
    {
        var q = (VDouble.One - x) / (VDouble.One + x);
        var r = Atan(Sqrt(q)) * new VDouble(2d);
        return VDouble.Select(x <= new VDouble(-1d), new VDouble(PiD), r);
    }

    /// <summary>
    /// Per-lane cube root (handles negatives; exp/log seed + two Newton steps).
    /// </summary>
    public static VDouble Cbrt(VDouble x)
    {
        var sign = VDouble.Select(x < VDouble.Zero, new VDouble(-1d), VDouble.One);
        var a = Abs(x);
        var y = Exp(Log(a) * new VDouble(1d / 3d));
        y = (y + y + (a / (y * y))) * new VDouble(1d / 3d);
        y = (y + y + (a / (y * y))) * new VDouble(1d / 3d);
        var r = sign * y;
        return VDouble.Select(VDouble.Eq(x, VDouble.Zero), VDouble.Zero, r);
    }

    /// <summary>
    /// Per-lane hypot: sqrt(x² + y²) without intermediate overflow.
    /// </summary>
    public static VDouble Hypot(VDouble x, VDouble y)
    {
        var a = Abs(x);
        var b = Abs(y);
        var m = Max(a, b);
        var q = Min(a, b) / m;                        // NaN when m == 0, masked below
        var r = m * Sqrt(VDouble.MulAdd(q, q, VDouble.One));
        return VDouble.Select(VDouble.Eq(m, VDouble.Zero), VDouble.Zero, r);
    }

    // Portable per-lane long divide (Vector<long> has no division or shift op on netstandard2.1).
    private static Vector<long> Divide(Vector<long> a, long divisor)
    {
        Span<long> tmp = stackalloc long[Vector<long>.Count];
        for (int l = 0; l < Vector<long>.Count; l++)
            tmp[l] = a[l] / divisor;
        return new Vector<long>(tmp);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Sqrt(VDouble2 x) => new(Sqrt(x.Lower), Sqrt(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Abs(VDouble2 x) => new(Abs(x.Lower), Abs(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Min(VDouble2 a, VDouble2 b) => new(Min(a.Lower, b.Lower), Min(a.Upper, b.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Max(VDouble2 a, VDouble2 b) => new(Max(a.Lower, b.Lower), Max(a.Upper, b.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Clamp(VDouble2 x, VDouble2 lo, VDouble2 hi) => Min(Max(x, lo), hi);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Lerp(VDouble2 a, VDouble2 b, VDouble2 t) => a + ((b - a) * t);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Rcp(VDouble2 x) => new(Rcp(x.Lower), Rcp(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Rsqrt(VDouble2 x) => new(Rsqrt(x.Lower), Rsqrt(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Floor(VDouble2 x) => new(Floor(x.Lower), Floor(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Round(VDouble2 x) => new(Round(x.Lower), Round(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Truncate(VDouble2 x) => new(Truncate(x.Lower), Truncate(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Exp(VDouble2 x) => new(Exp(x.Lower), Exp(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Log(VDouble2 x) => new(Log(x.Lower), Log(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Pow(VDouble2 x, VDouble2 y) => new(Pow(x.Lower, y.Lower), Pow(x.Upper, y.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Sin(VDouble2 x) => new(Sin(x.Lower), Sin(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Cos(VDouble2 x) => new(Cos(x.Lower), Cos(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Tan(VDouble2 x) => new(Tan(x.Lower), Tan(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Tanh(VDouble2 x) => new(Tanh(x.Lower), Tanh(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Sigmoid(VDouble2 x) => new(Sigmoid(x.Lower), Sigmoid(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Atan(VDouble2 x) => new(Atan(x.Lower), Atan(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Atan2(VDouble2 y, VDouble2 x) => new(Atan2(y.Lower, x.Lower), Atan2(y.Upper, x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Asin(VDouble2 x) => new(Asin(x.Lower), Asin(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Acos(VDouble2 x) => new(Acos(x.Lower), Acos(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Cbrt(VDouble2 x) => new(Cbrt(x.Lower), Cbrt(x.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 Hypot(VDouble2 x, VDouble2 y) => new(Hypot(x.Lower, y.Lower), Hypot(x.Upper, y.Upper));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 Abs(VLong2 x) => VLong2.Abs(x);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 Min(VLong2 a, VLong2 b) => VLong2.Min(a, b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 Max(VLong2 a, VLong2 b) => VLong2.Max(a, b);
}
