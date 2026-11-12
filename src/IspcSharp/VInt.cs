using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace IspcSharp;

/// <summary>
/// A "varying int", one 32-bit integer per SIMD lane. The ISPC equivalent of <c>varying int</c>.
/// </summary>
public readonly struct VInt : IEquatable<VInt>
{
    public readonly Vector<int> V;

    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<int>.Count;
    }

    public static VInt Zero => new(Vector<int>.Zero);
    public static VInt One => new(Vector<int>.One);

    private static readonly Vector<int> LaneIndex = CreateLaneIndex();
    private static Vector<int> CreateLaneIndex()
    {
        Span<int> tmp = stackalloc int[Vector<int>.Count];
        for (int i = 0; i < tmp.Length; i++)
            tmp[i] = i;
        return new Vector<int>(tmp);
    }

    /// <summary>
    /// {0, 1, 2, ... LaneCount-1}, ISPC's <c>programIndex</c>.
    /// </summary>
    public static VInt ProgramIndex => new(LaneIndex);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VInt(Vector<int> v) => V = v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VInt(int uniform) => V = new Vector<int>(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Load(ReadOnlySpan<int> source, int offset = 0)
        => new(LaneIO.Load(source, offset));

    /// <summary>
    /// Contiguous load of LaneCount elements from an array (same as the span overload, without
    /// rebuilding a span on every call).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Load(int[] source, int offset) => new(LaneIO.Load(source, offset));

    /// <summary>
    /// Contiguous store of LaneCount elements into an array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(int[] destination, int offset) => LaneIO.Store(V, destination, offset);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<int> destination, int offset = 0) => LaneIO.Store(V, destination, offset);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void StoreMasked(Span<int> destination, int offset, VMask mask)
    {
        // A full gang that fits is one plain store; the per-lane path is only for real tails.
        if (mask.AllActive() && LaneIO.Fits<int>(destination.Length, offset))
            LaneIO.StoreUnchecked(V, destination, offset);
        else
            LaneIO.StoreMaskedSlow(V, destination, offset, mask.ToBitmask());
    }

    public int GetLane(int lane) => V[lane];

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator +(VInt a, VInt b) => new(a.V + b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator -(VInt a, VInt b) => new(a.V - b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator *(VInt a, VInt b) => new(a.V * b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator +(VInt a, int b) => new(a.V + new Vector<int>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator -(VInt a, int b) => new(a.V - new Vector<int>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator *(VInt a, int b) => new(a.V * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator &(VInt a, VInt b) => new(a.V & b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator |(VInt a, VInt b) => new(a.V | b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator ^(VInt a, VInt b) => new(a.V ^ b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator ~(VInt a) => new(~a.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator -(VInt a) => new(-a.V);

    /// <summary>
    /// Per-lane integer division (see <see cref="Divide"/>).
    /// Throws <see cref="DivideByZeroException"/> if ANY lane's divisor is zero, in masked
    /// SPMD code, select a nonzero divisor into inactive lanes first (the [Spmd] generator
    /// does this automatically).
    /// </summary>
    public static VInt operator /(VInt a, VInt b) => Divide(a, b);

    /// <summary>
    /// Per-lane integer remainder (see <see cref="Remainder"/>; <see cref="op_Division"/> about zero divisors).
    /// </summary>
    public static VInt operator %(VInt a, VInt b) => Remainder(a, b);

    /// <summary>
    /// Per-lane integer division, truncating toward zero like C#'s <c>/</c>. On x86 and ARM64 the
    /// quotient is computed in double precision (<c>vcvtdq2pd</c> / <c>vdivpd</c> / <c>vcvttpd2dq</c>,
    /// or NEON <c>scvtf</c> / <c>fdiv</c> / <c>fcvtzs</c>): a
    /// double holds every int exactly and the correctly rounded quotient of two ints truncates
    /// to the exact integer quotient, so this is bit-identical to the scalar divide at roughly a
    /// tenth of the cost of a per-lane <c>idiv</c>. Throws exactly where the scalar code does:
    /// <see cref="DivideByZeroException"/> for a zero divisor in any lane, <see cref="OverflowException"/>
    /// for <c>int.MinValue / -1</c>.
    /// </summary>
    public static VInt Divide(VInt a, VInt b)
    {
#if NET8_0_OR_GREATER
        if ((System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount is 8 or 4) ||
            (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && LaneCount == 4))
        {
            ThrowIfDivideFaults(a, b);
            return new VInt(QuotientViaDouble(a.V, b.V));
        }
#endif
        return DividePortable(a, b);
    }

    /// <summary>
    /// Per-lane integer remainder with the sign of the dividend, like C#'s <c>%</c>
    /// (<c>a - (a / b) * b</c> on x86; see <see cref="Divide"/> for the method and the exceptions).
    /// </summary>
    public static VInt Remainder(VInt a, VInt b)
    {
#if NET8_0_OR_GREATER
        if ((System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount is 8 or 4) ||
            (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && LaneCount == 4))
        {
            ThrowIfDivideFaults(a, b);
            return new VInt(a.V - (QuotientViaDouble(a.V, b.V) * b.V));
        }
#endif
        return RemainderPortable(a, b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ThrowIfDivideFaults(VInt a, VInt b)
    {
        if (Vector.EqualsAny(b.V, Vector<int>.Zero))
            ThrowDivideByZero();
        var minusOne = new Vector<int>(-1);
        if (Vector.EqualsAny(b.V, minusOne) &&
            (Vector.Equals(a.V, new Vector<int>(int.MinValue)) & Vector.Equals(b.V, minusOne)) != Vector<int>.Zero)
        {
            ThrowDivideOverflow();
        }
    }

    private static void ThrowDivideByZero() => throw new DivideByZeroException();
    private static void ThrowDivideOverflow() => throw new OverflowException();

#if NET8_0_OR_GREATER
    /// <summary>
    /// trunc(a / b) per lane through double precision. Callers have ruled out b == 0 and
    /// MinValue / -1, the only pairs whose quotient does not fit an int.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> QuotientViaDouble(Vector<int> a, Vector<int> b)
    {
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported)
        {
            // Sign-extend each half to 64-bit lanes, convert, divide, truncate, narrow back.
            var x = Vector128.AsVector128(a);
            var y = Vector128.AsVector128(b);
            var qLo = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.ConvertToInt64RoundToZero(System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Divide(
                System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.ConvertToDouble(System.Runtime.Intrinsics.Arm.AdvSimd.SignExtendWideningLower(x.GetLower())),
                System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.ConvertToDouble(System.Runtime.Intrinsics.Arm.AdvSimd.SignExtendWideningLower(y.GetLower()))));
            var qHi = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.ConvertToInt64RoundToZero(System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Divide(
                System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.ConvertToDouble(System.Runtime.Intrinsics.Arm.AdvSimd.SignExtendWideningUpper(x)),
                System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.ConvertToDouble(System.Runtime.Intrinsics.Arm.AdvSimd.SignExtendWideningUpper(y))));
            return Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingUpper(System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingLower(qLo), qHi));
        }

        if (LaneCount == 8)
        {
            var a256 = Vector256.AsVector256(a);
            var b256 = Vector256.AsVector256(b);
            var lo = System.Runtime.Intrinsics.X86.Avx.ConvertToVector128Int32WithTruncation(
                System.Runtime.Intrinsics.X86.Avx.Divide(
                    System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(a256.GetLower()),
                    System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(b256.GetLower())));
            var hi = System.Runtime.Intrinsics.X86.Avx.ConvertToVector128Int32WithTruncation(
                System.Runtime.Intrinsics.X86.Avx.Divide(
                    System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(a256.GetUpper()),
                    System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(b256.GetUpper())));
            return Vector256.AsVector(Vector256.Create(lo, hi));
        }

        // 128-bit gang: cvtdq2pd converts the low two ints, so the upper pair is moved down first.
        var a128 = Vector128.AsVector128(a);
        var b128 = Vector128.AsVector128(b);
        var q01 = System.Runtime.Intrinsics.X86.Sse2.ConvertToVector128Int32WithTruncation(
            System.Runtime.Intrinsics.X86.Sse2.Divide(
                System.Runtime.Intrinsics.X86.Sse2.ConvertToVector128Double(a128),
                System.Runtime.Intrinsics.X86.Sse2.ConvertToVector128Double(b128)));
        var q23 = System.Runtime.Intrinsics.X86.Sse2.ConvertToVector128Int32WithTruncation(
            System.Runtime.Intrinsics.X86.Sse2.Divide(
                System.Runtime.Intrinsics.X86.Sse2.ConvertToVector128Double(System.Runtime.Intrinsics.X86.Sse2.Shuffle(a128, 0xEE)),
                System.Runtime.Intrinsics.X86.Sse2.ConvertToVector128Double(System.Runtime.Intrinsics.X86.Sse2.Shuffle(b128, 0xEE))));
        return Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.UnpackLow(q01.AsInt64(), q23.AsInt64()).AsInt32());
    }
#endif

    /// <summary>Per-lane scalar divide for ISAs without a vector path.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt DividePortable(VInt a, VInt b)
    {
        Span<int> x = stackalloc int[LaneCount];
        Span<int> y = stackalloc int[LaneCount];
        a.V.CopyTo(x);
        b.V.CopyTo(y);
        for (int l = 0; l < x.Length; l++)
            x[l] /= y[l];
        return new VInt(new Vector<int>(x));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt RemainderPortable(VInt a, VInt b)
    {
        Span<int> x = stackalloc int[LaneCount];
        Span<int> y = stackalloc int[LaneCount];
        a.V.CopyTo(x);
        b.V.CopyTo(y);
        for (int l = 0; l < x.Length; l++)
            x[l] %= y[l];
        return new VInt(new Vector<int>(x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator VInt(int uniform) => new(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt operator <<(VInt a, int count)
    {
#if NET8_0_OR_GREATER
        return new VInt(Vector.ShiftLeft(a.V, count));
#else
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = a.V[l] << count;
        return new VInt(new Vector<int>(tmp));
#endif
    }

    /// <summary>
    /// Arithmetic (sign-extending) right shift.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt operator >>(VInt a, int count)
    {
#if NET8_0_OR_GREATER
        return new VInt(Vector.ShiftRightArithmetic(a.V, count));
#else
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = a.V[l] >> count;
        return new VInt(new Vector<int>(tmp));
#endif
    }

    /// <summary>
    /// Logical (zero-filling) right shift.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt ShiftRightLogical(VInt a, int count)
    {
#if NET8_0_OR_GREATER
        return new VInt(Vector.AsVectorInt32(Vector.ShiftRightLogical(Vector.AsVectorUInt32(a.V), count)));
#else
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (int)((uint)a.V[l] >> count);
        return new VInt(new Vector<int>(tmp));
#endif
    }

    /// <summary>
    /// High 32 bits of the <b>unsigned</b> 32x32 → 64 product, per lane:
    /// <c>result[l] = (int)(((ulong)(uint)a[l] * (uint)b[l]) >> 32)</c>.
    ///
    /// The low half is just <c>a * b</c> — the bottom 32 bits of a product are the same
    /// signed or unsigned — so this plus <c>operator *</c> gives the full widening multiply
    /// without ever forming 64-bit lanes. That matters: <c>Vector&lt;long&gt;</c> multiply has
    /// no hardware instruction below AVX-512DQ, so widening to a <see cref="VLong2"/> pair and
    /// multiplying there costs an emulated sequence per half plus the widen/narrow round trip.
    /// <c>vpmuludq</c> is precisely a 32x32 → 64 widening multiply and does the whole job.
    ///
    /// Counter-based RNGs (Philox, Threefry) and fixed-point scaling are the usual callers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt MultiplyHighUnsigned(VInt a, VInt b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            // NEON: umull / umull2 give all four 64-bit products directly, and shift-and-narrow
            // by 32 (shrn / shrn2) keeps each one's high word.
            var x = Vector128.AsVector128(a.V).AsUInt32();
            var y = Vector128.AsVector128(b.V).AsUInt32();
            var lo = System.Runtime.Intrinsics.Arm.AdvSimd.MultiplyWideningLower(x.GetLower(), y.GetLower());
            var hi32 = System.Runtime.Intrinsics.Arm.AdvSimd.MultiplyWideningUpper(x, y);
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.ShiftRightLogicalNarrowingUpper(
                System.Runtime.Intrinsics.Arm.AdvSimd.ShiftRightLogicalNarrowingLower(lo, 32), hi32, 32).AsInt32()));
        }
#endif
        var au = Vector.AsVectorUInt64(a.V);
        var bu = Vector.AsVectorUInt64(b.V);

        // vpmuludq multiplies the even-indexed 32-bit lanes and writes 64-bit products, so the
        // odd lanes get a second pass after being shifted down into even position.
        var pe = MultiplyEvenUnsigned(au, bu);
        var po = MultiplyEvenUnsigned(
            Vector.ShiftRightLogical(au, 32),
            Vector.ShiftRightLogical(bu, 32));

        // Recombine: >>32 drops each even product's high word into the even 32-bit slot, while
        // each odd product's high word already sits in the odd slot. The slots each side
        // contributes are disjoint and the rest are zero, so OR merges them — no blend needed.
        var hi = Vector.ShiftRightLogical(pe, 32) | (po & new Vector<ulong>(0xFFFFFFFF00000000UL));
        return new VInt(Vector.AsVectorInt32(hi));
    }

    /// <summary>
    /// <c>vpmuludq</c>: 64-bit products of the low halves of each 64-bit lane — that is, of the
    /// even-indexed 32-bit lanes. Callers pass the odd lanes by shifting them down first.
    ///
    /// The portable fallback lives in its own method on purpose: <c>stackalloc</c> emits
    /// <c>localloc</c>, and RyuJIT refuses to inline <i>any</i> method containing it, no matter
    /// what <see cref="MethodImplOptions.AggressiveInlining"/> says. Left inline here that would
    /// turn every multiply into a real call — which is exactly how the per-lane
    /// <c>VectorMath.Log</c> divide helper this primitive replaces became a bottleneck.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<ulong> MultiplyEvenUnsigned(Vector<ulong> a, Vector<ulong> b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            return Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.Multiply(
                Vector512.AsVector512(a).AsUInt32(), Vector512.AsVector512(b).AsUInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            return Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.Multiply(
                Vector256.AsVector256(a).AsUInt32(), Vector256.AsVector256(b).AsUInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 4)
        {
            return Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.Multiply(
                Vector128.AsVector128(a).AsUInt32(), Vector128.AsVector128(b).AsUInt32()));
        }
#endif
        return MultiplyEvenUnsignedPortable(a, b);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Vector<ulong> MultiplyEvenUnsignedPortable(Vector<ulong> a, Vector<ulong> b)
    {
        Span<ulong> tmp = stackalloc ulong[Vector<ulong>.Count];
        for (int l = 0; l < tmp.Length; l++)
            tmp[l] = (ulong)(uint)a[l] * (uint)b[l];
        return new Vector<ulong>(tmp);
    }

    /// <summary>
    /// Per-lane left shift: result[l] = a[l] &lt;&lt; (counts[l] &amp; 31).
    /// Hardware <c>vpsllvd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt ShiftLeftVariable(VInt a, VInt counts)
    {
#if NET8_0_OR_GREATER
        var cnt = (counts & new VInt(31)).V;
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ShiftLeftLogicalVariable(
                Vector512.AsVector512(a.V).AsUInt32(), Vector512.AsVector512(cnt).AsUInt32()).AsInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftLeftLogicalVariable(
                Vector256.AsVector256(a.V), Vector256.AsVector256(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 4)
        {
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftLeftLogicalVariable(
                Vector128.AsVector128(a.V), Vector128.AsVector128(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            // NEON ushl: a positive per-lane count shifts left.
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.ShiftLogical(
                Vector128.AsVector128(a.V), Vector128.AsVector128(cnt))));
        }
#endif
        return ShiftLeftVariablePortable(a, counts);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt ShiftLeftVariablePortable(VInt a, VInt counts)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] << counts.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Per-lane arithmetic right shift: result[l] = a[l] &gt;&gt; (counts[l] &amp; 31).
    /// Hardware <c>vpsravd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt ShiftRightArithmeticVariable(VInt a, VInt counts)
    {
#if NET8_0_OR_GREATER
        var cnt = (counts & new VInt(31)).V;
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ShiftRightArithmeticVariable(
                Vector512.AsVector512(a.V), Vector512.AsVector512(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightArithmeticVariable(
                Vector256.AsVector256(a.V), Vector256.AsVector256(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 4)
        {
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightArithmeticVariable(
                Vector128.AsVector128(a.V), Vector128.AsVector128(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            // NEON has no right-shift-by-vector: sshl with a negated count shifts right, sign-filling.
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.ShiftArithmetic(
                Vector128.AsVector128(a.V), System.Runtime.Intrinsics.Arm.AdvSimd.Negate(Vector128.AsVector128(cnt)))));
        }
#endif
        return ShiftRightArithmeticVariablePortable(a, counts);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt ShiftRightArithmeticVariablePortable(VInt a, VInt counts)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] >> counts.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Per-lane logical right shift: result[l] = (uint)a[l] &gt;&gt; (counts[l] &amp; 31).
    /// Hardware <c>vpsrlvd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt ShiftRightLogicalVariable(VInt a, VInt counts)
    {
#if NET8_0_OR_GREATER
        var cnt = (counts & new VInt(31)).V;
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ShiftRightLogicalVariable(
                Vector512.AsVector512(a.V).AsUInt32(), Vector512.AsVector512(cnt).AsUInt32()).AsInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogicalVariable(
                Vector256.AsVector256(a.V).AsUInt32(), Vector256.AsVector256(cnt).AsUInt32()).AsInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 4)
        {
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogicalVariable(
                Vector128.AsVector128(a.V).AsUInt32(), Vector128.AsVector128(cnt).AsUInt32()).AsInt32()));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            // NEON ushl with a negated count shifts right, zero-filling.
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.ShiftLogical(
                Vector128.AsVector128(a.V).AsUInt32(), System.Runtime.Intrinsics.Arm.AdvSimd.Negate(Vector128.AsVector128(cnt))).AsInt32()));
        }
#endif
        return ShiftRightLogicalVariablePortable(a, counts);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt ShiftRightLogicalVariablePortable(VInt a, VInt counts)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (int)((uint)a.V[l] >> counts.V[l]);
        return new VInt(new Vector<int>(tmp));
    }

    // ------------------------------------------------------------------------------------------
    // Narrow (byte/short) buffers at int-gang width: one int lane per element. Every entry point
    // does a single range check, then hands a ref to the first element to a core routine that
    // loads or stores straight through a vector register. Array overloads exist so a kernel that
    // was handed arrays does not rebuild a span (null test, pointer, length) on every gang.
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Widening load from a byte buffer: one int lane per byte, zero-extended (0..255).
    /// Reads exactly LaneCount bytes. Hardware <c>vpmovzxbd</c> (SSE4.1/AVX2/AVX-512F) or
    /// NEON <c>ushll</c>; portable loop otherwise. This is the ISPC-style bridge that runs
    /// byte data at full int-gang width: compute in int lanes (C#'s own promotion rules),
    /// touch memory one byte per lane.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt LoadZeroExtend(ReadOnlySpan<byte> source, int offset = 0)
    {
        if (!LaneIO.Fits(source.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        return LoadZeroExtendCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source), (nint)(uint)offset));
    }

    /// <inheritdoc cref="LoadZeroExtend(ReadOnlySpan{byte}, int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt LoadZeroExtend(byte[] source, int offset)
    {
        if (!LaneIO.Fits(source.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        return LoadZeroExtendCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(source), (nint)(uint)offset));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static VInt LoadZeroExtendCore(ref byte p)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(Vector128.LoadUnsafe(ref p))));

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            var b = Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(ref p)).AsByte();
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32(b)));
        }

        if (System.Runtime.Intrinsics.X86.Sse41.IsSupported && LaneCount == 4)
        {
            var b = Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<uint>(ref p)).AsByte();
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32(b)));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            var b = Vector64.CreateScalarUnsafe(Unsafe.ReadUnaligned<uint>(ref p)).AsByte();
            var w16 = System.Runtime.Intrinsics.Arm.AdvSimd.ZeroExtendWideningLower(b);
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.ZeroExtendWideningLower(w16.GetLower()).AsInt32()));
        }
#endif
        return LoadZeroExtendPortable(ref p);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt LoadZeroExtendPortable(ref byte p)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < tmp.Length; l++)
            tmp[l] = Unsafe.Add(ref p, l);
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Widening load from a short buffer: one int lane per short, sign-extended.
    /// Reads exactly LaneCount shorts. Hardware <c>vpmovsxwd</c> (SSE4.1/AVX2/AVX-512F) or
    /// NEON <c>sshll</c>; portable loop otherwise. See <see cref="LoadZeroExtend(ReadOnlySpan{byte}, int)"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt LoadSignExtend(ReadOnlySpan<short> source, int offset = 0)
    {
        if (!LaneIO.Fits(source.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        return LoadSignExtendCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source), (nint)(uint)offset));
    }

    /// <inheritdoc cref="LoadSignExtend(ReadOnlySpan{short}, int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt LoadSignExtend(short[] source, int offset)
    {
        if (!LaneIO.Fits(source.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        return LoadSignExtendCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(source), (nint)(uint)offset));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static VInt LoadSignExtendCore(ref short p)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(Vector256.LoadUnsafe(ref p))));

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32(Vector128.LoadUnsafe(ref p))));

        if (System.Runtime.Intrinsics.X86.Sse41.IsSupported && LaneCount == 4)
        {
            var s = Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(ref Unsafe.As<short, byte>(ref p))).AsInt16();
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32(s)));
        }

        // NEON: four shorts are exactly one 64-bit vector register (ldr d), widened by sshll.
        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.SignExtendWideningLower(Vector64.LoadUnsafe(ref p))));
#endif
        return LoadSignExtendPortable(ref p);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt LoadSignExtendPortable(ref short p)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < tmp.Length; l++)
            tmp[l] = Unsafe.Add(ref p, l);
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Truncating narrowing store to a byte buffer: each int lane's low 8 bits, exactly
    /// C#'s <c>(byte)</c> cast. Writes exactly LaneCount bytes. Hardware <c>vpmovdb</c>
    /// (AVX-512F), a byte shuffle (SSSE3/AVX2) or NEON <c>xtn</c>; portable otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void StoreNarrow(Span<byte> destination, int offset = 0)
    {
        if (!LaneIO.Fits(destination.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        StoreNarrowCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nint)(uint)offset));
    }

    /// <inheritdoc cref="StoreNarrow(Span{byte}, int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void StoreNarrow(byte[] destination, int offset)
    {
        if (!LaneIO.Fits(destination.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        StoreNarrowCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(destination), (nint)(uint)offset));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void StoreNarrowCore(ref byte p)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector128Byte(Vector512.AsVector512(V)).StoreUnsafe(ref p);
            return;
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            // Byte 0 of each int (bytes 0,4,8,12; 0x80 zeroes) to the bottom of its 128-bit half,
            // then the two halves' low dwords side by side: one vpshufb + one vpermd, no masking and
            // no saturating packs. The control vector is written here, not shared through a static:
            // the JIT folds a Create call into a constant but calls a property getter every gang.
            var picked = System.Runtime.Intrinsics.X86.Avx2.Shuffle(Vector256.AsVector256(V).AsByte(), Vector256.Create(
                (byte)0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));
            var packed = System.Runtime.Intrinsics.X86.Avx2.PermuteVar8x32(picked.AsInt32(), Vector256.Create(0, 4, 0, 0, 0, 0, 0, 0));
            Unsafe.WriteUnaligned(ref p, packed.AsUInt64().ToScalar());
            return;
        }

        if (System.Runtime.Intrinsics.X86.Ssse3.IsSupported && LaneCount == 4)
        {
            var picked = System.Runtime.Intrinsics.X86.Ssse3.Shuffle(Vector128.AsVector128(V).AsByte(), Vector128.Create(
                (byte)0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));
            Unsafe.WriteUnaligned(ref p, picked.AsUInt32().ToScalar());
            return;
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            var n16 = System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingLower(Vector128.AsVector128(V));
            var n8 = System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingLower(Vector128.Create(n16, Vector64<short>.Zero));
            Unsafe.WriteUnaligned(ref p, n8.AsUInt32().ToScalar());
            return;
        }
#endif
        for (int l = 0; l < LaneCount; l++)
            Unsafe.Add(ref p, l) = (byte)V[l];
    }

    /// <summary>
    /// Truncating narrowing store to a short buffer: each int lane's low 16 bits, exactly
    /// C#'s <c>(short)</c> cast. Writes exactly LaneCount shorts. Hardware <c>vpmovdw</c>
    /// (AVX-512F), a byte shuffle (SSSE3/AVX2) or NEON <c>xtn</c>; portable otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void StoreNarrow(Span<short> destination, int offset = 0)
    {
        if (!LaneIO.Fits(destination.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        StoreNarrowCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nint)(uint)offset));
    }

    /// <inheritdoc cref="StoreNarrow(Span{short}, int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void StoreNarrow(short[] destination, int offset)
    {
        if (!LaneIO.Fits(destination.Length, offset, LaneCount))
            LaneIO.ThrowGangOutOfRange();
        StoreNarrowCore(ref Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(destination), (nint)(uint)offset));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void StoreNarrowCore(ref short p)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector256Int16(Vector512.AsVector512(V)).StoreUnsafe(ref p);
            return;
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            // Low word of each int to the bottom of its 128-bit half (vpshufb), then the two
            // halves' low qwords side by side (vpermq): plain truncation, no sign-fit shifts.
            var picked = System.Runtime.Intrinsics.X86.Avx2.Shuffle(Vector256.AsVector256(V).AsByte(), Vector256.Create(
                (byte)0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));
            System.Runtime.Intrinsics.X86.Avx2.Permute4x64(picked.AsInt64(), 0b_10_00).GetLower().AsInt16().StoreUnsafe(ref p);
            return;
        }

        if (System.Runtime.Intrinsics.X86.Ssse3.IsSupported && LaneCount == 4)
        {
            var picked = System.Runtime.Intrinsics.X86.Ssse3.Shuffle(Vector128.AsVector128(V).AsByte(), Vector128.Create(
                (byte)0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));
            Unsafe.WriteUnaligned(ref Unsafe.As<short, byte>(ref p), picked.AsUInt64().ToScalar());
            return;
        }

        // NEON: xtn leaves four shorts in one 64-bit vector register, stored as is (str d).
        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingLower(Vector128.AsVector128(V)).StoreUnsafe(ref p);
            return;
        }
#endif
        for (int l = 0; l < LaneCount; l++)
            Unsafe.Add(ref p, l) = (short)V[l];
    }

    /// <summary>
    /// Widen a full int gang into its two long-gang halves (cross-gang-width bridge).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (VLong Lower, VLong Upper) Widen(VInt v)
    {
        Vector.Widen(v.V, out var lo, out var hi);
        return (new VLong(lo), new VLong(hi));
    }

    /// <summary>
    /// Narrow two long-gang halves back into one int gang (truncating).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Narrow(VLong lower, VLong upper)
        => new(Vector.Narrow(lower.V, upper.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator <(VInt a, VInt b) => new(Vector.LessThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator >(VInt a, VInt b) => new(Vector.GreaterThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator <=(VInt a, VInt b) => new(Vector.LessThanOrEqual(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator >=(VInt a, VInt b) => new(Vector.GreaterThanOrEqual(a.V, b.V));
    public static VMask Eq(VInt a, VInt b) => new(Vector.Equals(a.V, b.V));
    public static VMask Neq(VInt a, VInt b) => new(~Vector.Equals(a.V, b.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Select(VMask mask, VInt ifTrue, VInt ifFalse)
        => new(Vector.ConditionalSelect(mask.Bits, ifTrue.V, ifFalse.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VInt Blend(VMask mask, VInt newValue) => Select(mask, newValue, this);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VFloat ToFloat() => new(Vector.ConvertToSingle(V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt FromFloatTruncate(VFloat f) => new(Vector.ConvertToInt32(f.V));

    /// <summary>
    /// Bitcast: reinterpret the 32 bits of each lane as a float (no conversion).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VFloat AsFloat() => new(Vector.AsVectorSingle(V));

    public bool Equals(VInt other) => V.Equals(other.V);
    public override bool Equals(object? obj) => obj is VInt i && Equals(i);
    public override int GetHashCode() => V.GetHashCode();
    public override string ToString() => V.ToString();

    public static bool operator ==(VInt left, VInt right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VInt left, VInt right)
    {
        return !(left == right);
    }
}
