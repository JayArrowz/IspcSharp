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
    // ln2 split (Cody–Waite): Ln2Hi has 9 significant bits, so k*Ln2Hi is exact for any exponent k.
    private const float Ln2Hi = 0.693359375f;
    private const float Ln2Lo = -2.12194440e-4f;
    // 2π split in three (Cody–Waite): TwoPiA has 8 significant bits and TwoPiB 11, so k*TwoPiA
    // and k*TwoPiB are exact products even without a fused multiply-add (k up to 2^16 / 2^13);
    // TwoPiC is the rest. A + B + C == 2π to ~7e-15.
    private const float TwoPiA = 6.28125f;
    private const float TwoPiB = 0.0019350051879882812f;
    private const float TwoPiC = 3.019916050561733e-07f;
    // 1.5 * 2^23: adding it to |v| < 2^22 rounds v to the nearest integer (ties to even).
    private const float RoundMagicF = 12582912f;
    private const float MinNormalF = 1.17549435e-38f;
    private const double MinNormalD = 2.2250738585072014e-308;

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
    /// Per-lane floor (hardware <c>roundps</c>/<c>frintm</c>). Exact for every input, including
    /// values beyond the int range, infinities and NaN.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Floor(VFloat x) => new(Vector.Floor(x.V));

    /// <summary>
    /// Per-lane round to nearest, ties to even: the same result as <see cref="MathF.Round(float)"/>
    /// (2.5 → 2, 3.5 → 4), which is what a plain <c>MathF.Round(x)</c> in a [Spmd] kernel lowers
    /// to. Exact for every input. <see cref="Round(VFloat)"/> is the ties-away-from-zero variant.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat RoundToEven(VFloat x)
    {
        // Adding and subtracting ±2^23 snaps |x| < 2^23 onto the integer grid in the FPU's own
        // round-to-nearest-even mode. The sign is OR-ed back so -0.4 rounds to -0, like MathF.
        var sign = Vector.AsVectorInt32(x.V) & new Vector<int>(int.MinValue);
        var magic = Vector.AsVectorSingle(sign | new Vector<int>(0x4B000000));   // ±2^23
        var r = Vector.AsVectorSingle(Vector.AsVectorInt32(x.V + magic - magic) | sign);
        // |x| >= 2^23 is already integral (and NaN compares false): pass those through.
        var small = Vector.LessThan(Vector.Abs(x.V), new Vector<float>(8388608f));
        return new VFloat(Vector.ConditionalSelect(small, r, x.V));
    }

    /// <summary>
    /// Per-lane round to nearest, ties away from zero (2.5 → 3, -2.5 → -3): the same result as
    /// <c>MathF.Round(x, MidpointRounding.AwayFromZero)</c>. Exact for every input. Note that a
    /// plain <c>MathF.Round(x)</c> rounds ties to even instead: see <see cref="RoundToEven(VFloat)"/>.
    /// </summary>
    public static VFloat Round(VFloat x)
    {
        var a = Abs(x);
        var t = Floor(a);
        var r = t + VFloat.Select((a - t) >= new VFloat(0.5f), VFloat.One, VFloat.Zero);
        var sign = Vector.AsVectorInt32(x.V) & new Vector<int>(int.MinValue);
        return new VFloat(Vector.AsVectorSingle(Vector.AsVectorInt32(r.V) | sign));
    }

    /// <summary>
    /// Per-lane e^x over the whole float range: range-reduced (x = k*ln2 + r, split-constant
    /// subtraction) + degree-5 polynomial + exponent bit assembly, ~2e-6 relative. Like
    /// <see cref="MathF.Exp(float)"/> it overflows to +Inf (x &gt; 88.72), underflows through the
    /// denormals to 0 (x &lt; -103.97), and returns NaN for NaN.
    /// </summary>
    // Inlined into the caller's loop on purpose: out of line, every gang pays a call with the
    // vector passed through memory and reloads all of the constants below.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Exp(VFloat x)
    {
        // Beyond these the result is already +Inf / 0; clamping keeps k inside the range the
        // two-step scaling below can represent, and turns ±Inf into ordinary lanes.
        var xc = Clamp(x, new VFloat(-104.0f), new VFloat(89.0f));

        // k = round(x * log2e) without a float→int→float round trip: adding 1.5*2^23 lands in
        // [2^23, 2^24) where the spacing is 1, so the FPU rounds to an integer for us.
        var t = (xc * new VFloat(Log2E)) + new VFloat(RoundMagicF);
        var k = t - new VFloat(RoundMagicF);
        // r = x - k*ln2 with ln2 split in two (Cody–Waite): k*Ln2Hi is exact, so r keeps its
        // precision at large |x| instead of losing the low bits of the product.
        var r = VFloat.MulAdd(k, new VFloat(-Ln2Hi), xc);
        r = VFloat.MulAdd(k, new VFloat(-Ln2Lo), r);               // r in [-ln2/2, ln2/2]

        // exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5!
        var r2 = r * r;
        var p = new VFloat(1.0f / 120.0f);
        p = VFloat.MulAdd(p, r, new VFloat(1.0f / 24.0f));
        p = VFloat.MulAdd(p, r, new VFloat(1.0f / 6.0f));
        p = VFloat.MulAdd(p, r, new VFloat(0.5f));
        p = VFloat.MulAdd(p, r2, r + VFloat.One);

        // 2^k as 2^k1 * 2^k2 (k1 + k2 = k): one factor cannot hold 2^128 (the exponent field
        // would read as Inf even where the result is still finite) or anything below 2^-126.
        // kb = k + 2*127 is positive, so the halves are valid biased exponents as they stand.
        var kb = VInt.FromFloatTruncate(k) + 254;
        var k1 = kb >> 1;
        var k2 = kb - k1;
        var result = p * (k1 << 23).AsFloat() * (k2 << 23).AsFloat();

        return VFloat.Select(VFloat.Eq(x, x), result, x);           // NaN in, NaN out
    }

    /// <summary>
    /// Per-lane natural log, ~2 ulp. Mantissa/exponent split via IEEE754 bits + atanh-form
    /// polynomial. Matches <see cref="MathF.Log(float)"/> on the edges too: log(0) = -Inf,
    /// log(negative) = NaN, log(+Inf) = +Inf, log(NaN) = NaN, and denormals are exact.
    /// </summary>
    // Inlined into the caller's loop on purpose: out of line, every gang pays a call with the
    // vector passed through memory and reloads all of the constants below.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

        var r = (ef * new VFloat(Ln2)) + logM;

        // The bit tricks above only hold for positive normal finite x: bit patterns
        // 0x00800000..0x7F7FFFFF. That is every lane of almost every gang, so it is one add and
        // one signed compare (adding 0x7F800000 maps exactly that range onto the bottom of the
        // int range); anything else is patched up out of line.
        if (Vector.LessThanAll(bits.V + new Vector<int>(0x7F800000), new Vector<int>(unchecked((int)0xFF000000))))
            return r;

        return LogSpecialLanes(x, r);
    }

    /// <summary>
    /// The IEEE results for the inputs the bit-level log cannot handle: +Inf and NaN pass
    /// through, 0 → -Inf, negatives → NaN, and denormals (no implicit leading 1, so the
    /// mantissa split is wrong) are recomputed per lane. Ordinary lanes keep <paramref name="r"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VFloat LogSpecialLanes(VFloat x, VFloat r)
    {
        r = VFloat.Select(x < new VFloat(float.PositiveInfinity), r, x);          // +Inf, NaN
        var positive = x > VFloat.Zero;
        var nonPositive = VFloat.Select(VFloat.Eq(x, VFloat.Zero), new VFloat(float.NegativeInfinity), new VFloat(float.NaN));
        r = VFloat.Select(positive, r, nonPositive);                               // 0, negatives

        var denormal = positive & (x < new VFloat(MinNormalF));
        if (!denormal.Any())
            return r;

        Span<float> xs = stackalloc float[VFloat.LaneCount];
        Span<float> rs = stackalloc float[VFloat.LaneCount];
        x.V.CopyTo(xs);
        r.V.CopyTo(rs);
        for (ulong b = denormal.ToBitmask(); b != 0; b &= b - 1)
        {
            int l = BitOperations.TrailingZeroCount(b);
            rs[l] = MathF.Log(xs[l]);
        }

        return VFloat.Load(rs);
    }

    /// <summary>
    /// Per-lane x^y (exp(y*log|x|) with the sign and edge cases of <see cref="MathF.Pow(float, float)"/>):
    /// a negative base works for integer exponents (<c>Pow(-2, 2) = 4</c>, <c>Pow(-2, 3) = -8</c>)
    /// and is NaN otherwise; <c>Pow(x, 0) = 1</c>; <c>Pow(0, y)</c> is 0 or +Inf by the sign of y.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Pow(VFloat x, VFloat y)
    {
        var r = Exp(y * Log(Abs(x)));

        // Negative base: defined only for integer y, negative when that integer is odd.
        // (Every float >= 2^24 is an even integer, which is what the halving test reports.)
        var negative = x < VFloat.Zero;
        var yIsInt = VFloat.Eq(Floor(y), y);
        var halfY = Floor(y * new VFloat(0.5f));
        var yIsOdd = yIsInt & VFloat.Neq(halfY + halfY, y);
        r = VFloat.Select(negative & yIsOdd, -r, r);
        r = VFloat.Select(VMask.AndNot(negative, yIsInt), new VFloat(float.NaN), r);
        // x^0 is 1 for every x (the log route gives 0 * -Inf = NaN at x = 0).
        return VFloat.Select(VFloat.Eq(y, VFloat.Zero), VFloat.One, r);
    }

    /// <summary>
    /// Per-lane sine. Reduced to [-π/2, π/2] (split-constant 2π subtraction, so accuracy holds
    /// for large arguments) then a degree-9 minimax polynomial; ~3e-6 abs error for |x| up to
    /// ~1e7 with hardware FMA (~1e5 without). Beyond that a float's spacing approaches the
    /// period and the result is meaningless.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Sin(VFloat x) => SinReduced(ReduceTwoPi(x));

    /// <summary>
    /// Per-lane cosine: sin(r + π/2), with the quarter turn added <i>after</i> range reduction
    /// so large arguments keep their precision.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Cos(VFloat x)
    {
        var r = ReduceTwoPi(x) + new VFloat(Pi * 0.5f);             // [-π/2, 3π/2]
        var wrapped = r - new VFloat(TwoPi);
        return SinReduced(VFloat.Select(r > new VFloat(Pi), wrapped, r));
    }

    /// <summary>
    /// x - 2π*round(x / 2π), in [-π, π]. 2π is subtracted in three parts whose products with k
    /// are exact, so the cancellation against x keeps the digits that matter once |x| is large
    /// (a single rounded k*2π loses them: sin(1000) was off by 1e-5, sin(1e6) by 0.02).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static VFloat ReduceTwoPi(VFloat x)
    {
        var t = (x * new VFloat(InvTwoPi)) + new VFloat(RoundMagicF);   // round via 1.5*2^23
        var k = t - new VFloat(RoundMagicF);
        var r = VFloat.MulAdd(k, new VFloat(-TwoPiA), x);
        r = VFloat.MulAdd(k, new VFloat(-TwoPiB), r);
        return VFloat.MulAdd(k, new VFloat(-TwoPiC), r);
    }

    /// <summary>sin(r) for r already in [-π, π].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static VFloat SinReduced(VFloat r)
    {
        // Fold to [-π/2, π/2] for polynomial accuracy.
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
    /// Per-lane floor (hardware <c>roundpd</c>/<c>frintm</c>). Exact for every input.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Floor(VDouble x) => new(Vector.Floor(x.V));

    /// <summary>
    /// Per-lane truncation toward zero, exact for every input. This is what
    /// <c>(int)</c>/<c>(long)</c> casts lower to in double [Spmd] kernels.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Truncate(VDouble x)
        // Two hardware rounds and a blend — no double↔int64 conversion, which has no encoding
        // before AVX-512DQ and used to run per lane.
        => new(Vector.ConditionalSelect(Vector.LessThan(x.V, Vector<double>.Zero), Vector.Ceiling(x.V), Vector.Floor(x.V)));

    /// <summary>
    /// Per-lane round to nearest, ties to even: the same result as <see cref="Math.Round(double)"/>
    /// (2.5 → 2, 3.5 → 4), which is what a plain <c>Math.Round(x)</c> in a [Spmd] kernel lowers to.
    /// Exact for every input. <see cref="Round(VDouble)"/> is the ties-away-from-zero variant.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble RoundToEven(VDouble x)
    {
        // ±2^52 add/subtract snaps |x| < 2^52 onto the integer grid (round-to-nearest-even);
        // the sign is OR-ed back so -0.4 rounds to -0, like Math.Round.
        var sign = Vector.AsVectorInt64(x.V) & new Vector<long>(long.MinValue);
        var magic = Vector.AsVectorDouble(sign | new Vector<long>(0x4330000000000000L));   // ±2^52
        var r = Vector.AsVectorDouble(Vector.AsVectorInt64(x.V + magic - magic) | sign);
        var small = Vector.LessThan(Vector.Abs(x.V), new Vector<double>(4503599627370496.0));
        return new VDouble(Vector.ConditionalSelect(small, r, x.V));
    }

    /// <summary>
    /// Per-lane round to nearest, ties away from zero (2.5 → 3, -2.5 → -3): the same result as
    /// <c>Math.Round(x, MidpointRounding.AwayFromZero)</c>. Exact for every input. Note that a
    /// plain <c>Math.Round(x)</c> rounds ties to even instead: see <see cref="RoundToEven(VDouble)"/>.
    /// </summary>
    public static VDouble Round(VDouble x)
    {
        var a = Abs(x);
        var t = Floor(a);
        var r = t + VDouble.Select((a - t) >= new VDouble(0.5), VDouble.One, VDouble.Zero);
        var sign = Vector.AsVectorInt64(x.V) & new Vector<long>(long.MinValue);
        return new VDouble(Vector.AsVectorDouble(Vector.AsVectorInt64(r.V) | sign));
    }

    /// <summary>
    /// Per-lane e^x. Range-reduced (x = k*ln2 + r, split-constant subtraction) + a divide-free
    /// degree-11 Taylor polynomial for exp(r) + exponent bit assembly. Accurate to ~1e-14 relative
    /// (a few ulp) over the whole double range: like <see cref="Math.Exp(double)"/> it overflows
    /// to +Inf (x &gt; 709.78), underflows through the denormals to 0, and returns NaN for NaN.
    ///
    /// The reduced-argument evaluation is a polynomial, not a Cephes P/Q rational: the rational
    /// form needed a vector divide (the slowest lane op, and only 2-4 lanes wide for doubles),
    /// which made the double exp — and everything built on it (Sigmoid, Tanh, Pow, Tonemap) —
    /// throughput-bound on the divide unit. Two independent Horner chains in r² (even/odd powers)
    /// keep the dependency chain short and use both FMA ports, matching the float exp's shape.
    /// Degree 11 (not 13) is the throughput sweet spot: it holds ~1e-14 relative here (well inside
    /// the 1e-13 double contract) while the two extra terms of degree 13 only cost FMA throughput.
    /// </summary>
    // Inlined into the caller's loop on purpose: out of line, every gang pays a call with the
    // vector passed through memory and reloads all of the constants below.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Exp(VDouble x)
    {
        // Beyond these the result is already +Inf / 0; clamping keeps k inside the range the
        // two-step scaling below can represent, and turns ±Inf into ordinary lanes.
        var x0 = x;
        x = Clamp(x, new VDouble(-746.0), new VDouble(710.0));

        // k = round(x·log2e), as BOTH a double (for the reduced argument r) and an int64 (for
        // 2^k), via the magic constant, no double↔int64 conversion. Round-to-nearest-even here
        // vs. away-from-zero is harmless: at an exact half, k shifts by 1 but 2^k·exp(r∓ln2)
        // gives the identical result, and r stays inside [-ln2/2, ln2/2].
        var t = (x * new VDouble(Log2ED)) + new VDouble(RoundMagicD);
        var k = t - new VDouble(RoundMagicD);
        // kb = k + 2*1023: positive, so its two halves are valid biased exponents as they stand.
        var kb = Vector.AsVectorInt64(t.V) - RoundMagicBits + new Vector<long>(2046L);
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

        // 2^k as 2^k1 * 2^k2 (k1 + k2 = k): one factor cannot hold 2^1024 (the exponent field
        // would read as Inf even where the result is still finite) or anything below 2^-1022.
        // Logical shifts only (vpsrlq/vpsllq) — AVX2 has no 64-bit arithmetic shift.
        var k1 = Vector.AsVectorInt64(Vector.ShiftRightLogical(Vector.AsVectorUInt64(kb), 1));
        var k2 = kb - k1;
        var s1 = new VDouble(Vector.AsVectorDouble(Vector.ShiftLeft(k1, 52)));
        var s2 = new VDouble(Vector.AsVectorDouble(Vector.ShiftLeft(k2, 52)));
        var result = expR * s1 * s2;

        return VDouble.Select(VDouble.Eq(x0, x0), result, x0);        // NaN in, NaN out
    }

    /// <summary>
    /// Per-lane natural log, ~1-2 ulp. Mantissa/exponent split via IEEE754 bits + atanh-form
    /// polynomial. Matches <see cref="Math.Log(double)"/> on the edges too: log(0) = -Inf,
    /// log(negative) = NaN, log(+Inf) = +Inf, log(NaN) = NaN, and denormals are exact.
    /// </summary>
    // Inlined into the caller's loop on purpose: out of line, every gang pays a call with the
    // vector passed through memory and reloads all of the constants below.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Log(VDouble x)
    {
        var bits = Vector.AsVectorInt64(x.V);

        // exponent = (bits >> 52) - 1023 (bits are positive for x > 0, so a logical shift —
        // vpsrlq, one AVX2 instruction, replaces the former per-lane scalar 64-bit divide).
        var e = Vector.AsVectorInt64(Vector.ShiftRightLogical(Vector.AsVectorUInt64(bits), 52)) - new Vector<long>(1023L);

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
        var r = VDouble.MulAdd(ef, new VDouble(LnTwoHiD), VDouble.MulAdd(ef, new VDouble(LnTwoLoD), logM));

        // Positive normal finite lanes (bit patterns 0x0010…0 .. 0x7FEF…F) are done; anything
        // else is patched up out of line (see the float overload for the one-compare test).
        if (Vector.LessThanAll(bits + new Vector<long>(0x7FF0000000000000L), new Vector<long>(unchecked((long)0xFFE0000000000000UL))))
            return r;

        return LogSpecialLanes(x, r);
    }

    /// <inheritdoc cref="LogSpecialLanes(VFloat, VFloat)"/>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VDouble LogSpecialLanes(VDouble x, VDouble r)
    {
        r = VDouble.Select(x < new VDouble(double.PositiveInfinity), r, x);       // +Inf, NaN
        var positive = x > VDouble.Zero;
        var nonPositive = VDouble.Select(VDouble.Eq(x, VDouble.Zero), new VDouble(double.NegativeInfinity), new VDouble(double.NaN));
        r = VDouble.Select(positive, r, nonPositive);                              // 0, negatives

        var denormal = positive & (x < new VDouble(MinNormalD));
        if (!denormal.Any())
            return r;

        Span<double> xs = stackalloc double[VDouble.LaneCount];
        Span<double> rs = stackalloc double[VDouble.LaneCount];
        x.V.CopyTo(xs);
        r.V.CopyTo(rs);
        for (ulong b = denormal.ToBitmask(); b != 0; b &= b - 1)
        {
            int l = BitOperations.TrailingZeroCount(b);
            rs[l] = Math.Log(xs[l]);
        }

        return VDouble.Load(rs);
    }

    /// <summary>
    /// Per-lane x^y (exp(y*log|x|) with the sign and edge cases of <see cref="Math.Pow(double, double)"/>):
    /// a negative base works for integer exponents and is NaN otherwise; <c>Pow(x, 0) = 1</c>;
    /// <c>Pow(0, y)</c> is 0 or +Inf by the sign of y.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Pow(VDouble x, VDouble y)
    {
        var r = Exp(y * Log(Abs(x)));

        var negative = x < VDouble.Zero;
        var yIsInt = VDouble.Eq(Floor(y), y);
        var halfY = Floor(y * new VDouble(0.5));
        var yIsOdd = yIsInt & VDouble.Neq(halfY + halfY, y);
        r = VDouble.Select(negative & yIsOdd, -r, r);
        r = VDouble.Select(VMaskD.AndNot(negative, yIsInt), new VDouble(double.NaN), r);
        return VDouble.Select(VDouble.Eq(y, VDouble.Zero), VDouble.One, r);
    }

    /// <summary>
    /// Per-lane sine. Range-reduced to [-π/2, π/2] (split-constant 2π subtraction, exact
    /// for |x| &lt; ~6e6) then degree-19 polynomial; ~1e-15 abs error over that range.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 RoundToEven(VDouble2 x) => new(RoundToEven(x.Lower), RoundToEven(x.Upper));
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
