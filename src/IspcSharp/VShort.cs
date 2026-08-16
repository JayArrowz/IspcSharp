using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace IspcSharp;

/// <summary>
/// A "varying short", one 16-bit signed integer per SIMD lane. The ISPC equivalent of
/// <c>varying int16</c>. Twice the lane count of <see cref="VInt"/> (SSE/NEON = 8 lanes,
/// AVX2 = 16, AVX-512 = 32), so short-heavy kernels process 2x the elements per instruction.
/// Masks are <see cref="VMaskS"/>. Arithmetic wraps like scalar C# casts; use
/// <see cref="AddSaturate"/>/<see cref="SubtractSaturate"/> for clamped DSP-style math
/// (hardware <c>paddsw</c>/<c>psubsw</c>/<c>sqadd</c>).
/// </summary>
public readonly struct VShort : IEquatable<VShort>
{
    public readonly Vector<short> V;

    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<short>.Count;
    }

    public static VShort Zero => new(Vector<short>.Zero);
    public static VShort One => new(Vector<short>.One);

    private static readonly Vector<short> LaneIndex = CreateLaneIndex();
    private static Vector<short> CreateLaneIndex()
    {
        Span<short> tmp = stackalloc short[Vector<short>.Count];
        for (int i = 0; i < tmp.Length; i++)
            tmp[i] = (short)i;
        return new Vector<short>(tmp);
    }

    /// <summary>
    /// {0, 1, 2, ... LaneCount-1}, ISPC's <c>programIndex</c> for 16-bit gangs.
    /// </summary>
    public static VShort ProgramIndex => new(LaneIndex);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VShort(Vector<short> v) => V = v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VShort(short uniform) => V = new Vector<short>(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort Load(ReadOnlySpan<short> source, int offset = 0)
        => new(new Vector<short>(source[offset..]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<short> destination, int offset = 0) => V.CopyTo(destination[offset..]);

    /// <summary>
    /// Masked load: inactive lanes receive <paramref name="fallback"/>. Safe at buffer tails.
    /// </summary>
    public static VShort LoadMasked(ReadOnlySpan<short> source, int offset, VMaskS mask, short fallback = 0)
    {
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++)
        {
            int idx = offset + l;
            tmp[l] = (mask.IsLaneActive(l) && idx < source.Length) ? source[idx] : fallback;
        }

        return Load(tmp);
    }

    /// <summary>
    /// Masked store: only active lanes write. Safe at buffer tails.
    /// </summary>
    public void StoreMasked(Span<short> destination, int offset, VMaskS mask)
    {
        for (int l = 0; l < LaneCount; l++)
        {
            int idx = offset + l;
            if (mask.IsLaneActive(l) && idx < destination.Length)
                destination[idx] = V[l];
        }
    }

    public short GetLane(int lane) => V[lane];

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator +(VShort a, VShort b) => new(a.V + b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator -(VShort a, VShort b) => new(a.V - b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator *(VShort a, VShort b) => new(a.V * b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator +(VShort a, short b) => new(a.V + new Vector<short>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator -(VShort a, short b) => new(a.V - new Vector<short>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator *(VShort a, short b) => new(a.V * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator -(VShort a) => new(-a.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator &(VShort a, VShort b) => new(a.V & b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator |(VShort a, VShort b) => new(a.V | b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator ^(VShort a, VShort b) => new(a.V ^ b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort operator ~(VShort a) => new(~a.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator VShort(short uniform) => new(uniform);

    /// <summary>
    /// Per-lane integer division (no SIMD integer divide exists, scalar loop).
    /// Throws <see cref="DivideByZeroException"/> if ANY lane's divisor is zero.
    /// </summary>
    public static VShort operator /(VShort a, VShort b) => Divide(a, b);

    /// <summary>
    /// Per-lane integer remainder (scalar loop; see <see cref="op_Division"/> about zero divisors).
    /// </summary>
    public static VShort operator %(VShort a, VShort b) => Remainder(a, b);

    public static VShort Divide(VShort a, VShort b)
    {
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (short)(a.V[l] / b.V[l]);
        return new VShort(new Vector<short>(tmp));
    }

    public static VShort Remainder(VShort a, VShort b)
    {
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (short)(a.V[l] % b.V[l]);
        return new VShort(new Vector<short>(tmp));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort operator <<(VShort a, int count)
    {
#if NET8_0_OR_GREATER
        return new VShort(Vector.ShiftLeft(a.V, count));
#else
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (short)(a.V[l] << count);
        return new VShort(new Vector<short>(tmp));
#endif
    }

    /// <summary>
    /// Arithmetic (sign-extending) right shift.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort operator >>(VShort a, int count)
    {
#if NET8_0_OR_GREATER
        return new VShort(Vector.ShiftRightArithmetic(a.V, count));
#else
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (short)(a.V[l] >> count);
        return new VShort(new Vector<short>(tmp));
#endif
    }

    /// <summary>
    /// Logical (zero-filling) right shift.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort ShiftRightLogical(VShort a, int count)
    {
#if NET8_0_OR_GREATER
        return new VShort(Vector.AsVectorInt16(Vector.ShiftRightLogical(Vector.AsVectorUInt16(a.V), count)));
#else
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (short)((ushort)a.V[l] >> count);
        return new VShort(new Vector<short>(tmp));
#endif
    }

    /// <summary>
    /// Saturating add: lanes clamp to [-32768, 32767] instead of wrapping.
    /// Hardware <c>paddsw</c> (SSE2/AVX2/AVX-512BW) or NEON <c>sqadd</c>; portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort AddSaturate(VShort a, VShort b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported && LaneCount == 32)
        {
            return new VShort(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512BW.AddSaturate(
                Vector512.AsVector512(a.V), Vector512.AsVector512(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 16)
        {
            return new VShort(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.AddSaturate(
                Vector256.AsVector256(a.V), Vector256.AsVector256(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 8)
        {
            return new VShort(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.AddSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 8)
        {
            return new VShort(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.AddSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }
#endif
        return AddSaturatePortable(a, b);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VShort AddSaturatePortable(VShort a, VShort b)
    {
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (short)Math.Clamp(a.V[l] + b.V[l], short.MinValue, short.MaxValue);
        return new VShort(new Vector<short>(tmp));
    }

    /// <summary>
    /// Saturating subtract: lanes clamp to [-32768, 32767] instead of wrapping.
    /// Hardware <c>psubsw</c> (SSE2/AVX2/AVX-512BW) or NEON <c>sqsub</c>; portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort SubtractSaturate(VShort a, VShort b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported && LaneCount == 32)
        {
            return new VShort(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512BW.SubtractSaturate(
                Vector512.AsVector512(a.V), Vector512.AsVector512(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 16)
        {
            return new VShort(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.SubtractSaturate(
                Vector256.AsVector256(a.V), Vector256.AsVector256(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 8)
        {
            return new VShort(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.SubtractSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 8)
        {
            return new VShort(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.SubtractSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }
#endif
        return SubtractSaturatePortable(a, b);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VShort SubtractSaturatePortable(VShort a, VShort b)
    {
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (short)Math.Clamp(a.V[l] - b.V[l], short.MinValue, short.MaxValue);
        return new VShort(new Vector<short>(tmp));
    }

    /// <summary>
    /// Per-lane high half of the 32-bit product: result[l] = (short)((a[l] * b[l]) &gt;&gt; 16).
    /// The core of Q15/Q16 fixed-point multiplies. Hardware <c>pmulhw</c>
    /// (SSE2/AVX2/AVX-512BW); portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort MultiplyHigh(VShort a, VShort b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported && LaneCount == 32)
        {
            return new VShort(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512BW.MultiplyHigh(
                Vector512.AsVector512(a.V), Vector512.AsVector512(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 16)
        {
            return new VShort(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.MultiplyHigh(
                Vector256.AsVector256(a.V), Vector256.AsVector256(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 8)
        {
            return new VShort(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.MultiplyHigh(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }
#endif
        return MultiplyHighPortable(a, b);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VShort MultiplyHighPortable(VShort a, VShort b)
    {
        Span<short> tmp = stackalloc short[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (short)((a.V[l] * b.V[l]) >> 16);
        return new VShort(new Vector<short>(tmp));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort Min(VShort a, VShort b) => new(Vector.Min(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort Max(VShort a, VShort b) => new(Vector.Max(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VShort Abs(VShort a) => new(Vector.Abs(a.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator <(VShort a, VShort b) => new(Vector.LessThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator >(VShort a, VShort b) => new(Vector.GreaterThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator <=(VShort a, VShort b) => new(Vector.LessThanOrEqual(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator >=(VShort a, VShort b) => new(Vector.GreaterThanOrEqual(a.V, b.V));
    public static VMaskS Eq(VShort a, VShort b) => new(Vector.Equals(a.V, b.V));
    public static VMaskS Neq(VShort a, VShort b) => new(~Vector.Equals(a.V, b.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort Select(VMaskS mask, VShort ifTrue, VShort ifFalse)
        => new(Vector.ConditionalSelect(mask.Bits, ifTrue.V, ifFalse.V));

    /// <summary>
    /// Masked assignment: returns <c>mask ? newValue : current</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VShort Blend(VMaskS mask, VShort newValue) => Select(mask, newValue, this);

    /// <summary>
    /// Widen a full short gang into its two int-gang halves (cross-gang-width bridge).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (VInt Lower, VInt Upper) Widen(VShort v)
    {
        Vector.Widen(v.V, out var lo, out var hi);
        return (new VInt(lo), new VInt(hi));
    }

    /// <summary>
    /// Narrow two int-gang halves back into one short gang (truncating).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort Narrow(VInt lower, VInt upper)
        => new(Vector.Narrow(lower.V, upper.V));

    public bool Equals(VShort other) => V.Equals(other.V);
    public override bool Equals(object? obj) => obj is VShort s && Equals(s);
    public override int GetHashCode() => V.GetHashCode();
    public override string ToString() => V.ToString();

    public static bool operator ==(VShort left, VShort right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VShort left, VShort right)
    {
        return !(left == right);
    }
}
