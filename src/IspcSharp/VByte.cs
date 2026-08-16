using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace IspcSharp;

/// <summary>
/// A "varying byte", one 8-bit unsigned integer per SIMD lane. The ISPC equivalent of
/// <c>varying uint8</c>. Four times the lane count of <see cref="VInt"/> (SSE/NEON = 16 lanes,
/// AVX2 = 32, AVX-512 = 64), the widest gang in the library — ideal for image/pixel work.
/// Masks are <see cref="VMaskB"/>. Arithmetic wraps like scalar C# casts; use
/// <see cref="AddSaturate"/>/<see cref="SubtractSaturate"/> for clamped pixel math
/// (hardware <c>paddusb</c>/<c>psubusb</c>/<c>uqadd</c>). Comparisons are unsigned.
/// </summary>
public readonly struct VByte : IEquatable<VByte>
{
    public readonly Vector<byte> V;

    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<byte>.Count;
    }

    public static VByte Zero => new(Vector<byte>.Zero);
    public static VByte One => new(Vector<byte>.One);

    private static readonly Vector<byte> LaneIndex = CreateLaneIndex();
    private static Vector<byte> CreateLaneIndex()
    {
        Span<byte> tmp = stackalloc byte[Vector<byte>.Count];
        for (int i = 0; i < tmp.Length; i++)
            tmp[i] = (byte)i;
        return new Vector<byte>(tmp);
    }

    /// <summary>
    /// {0, 1, 2, ... LaneCount-1}, ISPC's <c>programIndex</c> for 8-bit gangs.
    /// </summary>
    public static VByte ProgramIndex => new(LaneIndex);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VByte(Vector<byte> v) => V = v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VByte(byte uniform) => V = new Vector<byte>(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte Load(ReadOnlySpan<byte> source, int offset = 0)
        => new(new Vector<byte>(source[offset..]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<byte> destination, int offset = 0) => V.CopyTo(destination[offset..]);

    /// <summary>
    /// Masked load: inactive lanes receive <paramref name="fallback"/>. Safe at buffer tails.
    /// </summary>
    public static VByte LoadMasked(ReadOnlySpan<byte> source, int offset, VMaskB mask, byte fallback = 0)
    {
        Span<byte> tmp = stackalloc byte[LaneCount];
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
    public void StoreMasked(Span<byte> destination, int offset, VMaskB mask)
    {
        for (int l = 0; l < LaneCount; l++)
        {
            int idx = offset + l;
            if (mask.IsLaneActive(l) && idx < destination.Length)
                destination[idx] = V[l];
        }
    }

    public byte GetLane(int lane) => V[lane];

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator +(VByte a, VByte b) => new(a.V + b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator -(VByte a, VByte b) => new(a.V - b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator *(VByte a, VByte b) => new(a.V * b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator +(VByte a, byte b) => new(a.V + new Vector<byte>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator -(VByte a, byte b) => new(a.V - new Vector<byte>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator *(VByte a, byte b) => new(a.V * b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator &(VByte a, VByte b) => new(a.V & b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator |(VByte a, VByte b) => new(a.V | b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator ^(VByte a, VByte b) => new(a.V ^ b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte operator ~(VByte a) => new(~a.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator VByte(byte uniform) => new(uniform);

    /// <summary>
    /// Per-lane integer division (no SIMD integer divide exists, scalar loop).
    /// Throws <see cref="DivideByZeroException"/> if ANY lane's divisor is zero.
    /// </summary>
    public static VByte operator /(VByte a, VByte b) => Divide(a, b);

    /// <summary>
    /// Per-lane integer remainder (scalar loop; see <see cref="op_Division"/> about zero divisors).
    /// </summary>
    public static VByte operator %(VByte a, VByte b) => Remainder(a, b);

    public static VByte Divide(VByte a, VByte b)
    {
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (byte)(a.V[l] / b.V[l]);
        return new VByte(new Vector<byte>(tmp));
    }

    public static VByte Remainder(VByte a, VByte b)
    {
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (byte)(a.V[l] % b.V[l]);
        return new VByte(new Vector<byte>(tmp));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte operator <<(VByte a, int count)
    {
#if NET8_0_OR_GREATER
        return new VByte(Vector.ShiftLeft(a.V, count));
#else
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (byte)(a.V[l] << count);
        return new VByte(new Vector<byte>(tmp));
#endif
    }

    /// <summary>
    /// Logical (zero-filling) right shift — bytes are unsigned, so this is the only right shift.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte operator >>(VByte a, int count)
    {
#if NET8_0_OR_GREATER
        return new VByte(Vector.ShiftRightLogical(a.V, count));
#else
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (byte)(a.V[l] >> count);
        return new VByte(new Vector<byte>(tmp));
#endif
    }

    /// <summary>
    /// Saturating add: lanes clamp to 255 instead of wrapping (the pixel-brighten primitive).
    /// Hardware <c>paddusb</c> (SSE2/AVX2/AVX-512BW) or NEON <c>uqadd</c>; portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte AddSaturate(VByte a, VByte b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported && LaneCount == 64)
        {
            return new VByte(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512BW.AddSaturate(
                Vector512.AsVector512(a.V), Vector512.AsVector512(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 32)
        {
            return new VByte(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.AddSaturate(
                Vector256.AsVector256(a.V), Vector256.AsVector256(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 16)
        {
            return new VByte(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.AddSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 16)
        {
            return new VByte(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.AddSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }
#endif
        return AddSaturatePortable(a, b);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VByte AddSaturatePortable(VByte a, VByte b)
    {
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (byte)Math.Min(a.V[l] + b.V[l], byte.MaxValue);
        return new VByte(new Vector<byte>(tmp));
    }

    /// <summary>
    /// Saturating subtract: lanes clamp to 0 instead of wrapping (the pixel-darken primitive).
    /// Hardware <c>psubusb</c> (SSE2/AVX2/AVX-512BW) or NEON <c>uqsub</c>; portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte SubtractSaturate(VByte a, VByte b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported && LaneCount == 64)
        {
            return new VByte(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512BW.SubtractSaturate(
                Vector512.AsVector512(a.V), Vector512.AsVector512(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 32)
        {
            return new VByte(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.SubtractSaturate(
                Vector256.AsVector256(a.V), Vector256.AsVector256(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 16)
        {
            return new VByte(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.SubtractSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 16)
        {
            return new VByte(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.SubtractSaturate(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }
#endif
        return SubtractSaturatePortable(a, b);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VByte SubtractSaturatePortable(VByte a, VByte b)
    {
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (byte)Math.Max(a.V[l] - b.V[l], 0);
        return new VByte(new Vector<byte>(tmp));
    }

    /// <summary>
    /// Per-lane average rounded up: (a[l] + b[l] + 1) / 2, without intermediate overflow.
    /// Hardware <c>pavgb</c> (SSE2/AVX2/AVX-512BW); portable loop otherwise. The bilinear-blend
    /// / mipmap primitive.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte Average(VByte a, VByte b)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported && LaneCount == 64)
        {
            return new VByte(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512BW.Average(
                Vector512.AsVector512(a.V), Vector512.AsVector512(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 32)
        {
            return new VByte(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.Average(
                Vector256.AsVector256(a.V), Vector256.AsVector256(b.V))));
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 16)
        {
            return new VByte(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse2.Average(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 16)
        {
            return new VByte(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.FusedAddRoundedHalving(
                Vector128.AsVector128(a.V), Vector128.AsVector128(b.V))));
        }
#endif
        return AveragePortable(a, b);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VByte AveragePortable(VByte a, VByte b)
    {
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (byte)((a.V[l] + b.V[l] + 1) >> 1);
        return new VByte(new Vector<byte>(tmp));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte Min(VByte a, VByte b) => new(Vector.Min(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VByte Max(VByte a, VByte b) => new(Vector.Max(a.V, b.V));

    // Unsigned per-lane comparisons (200 > 100 is true; no sign trap at 0x80).
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator <(VByte a, VByte b) => new(Vector.LessThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator >(VByte a, VByte b) => new(Vector.GreaterThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator <=(VByte a, VByte b) => new(Vector.LessThanOrEqual(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator >=(VByte a, VByte b) => new(Vector.GreaterThanOrEqual(a.V, b.V));
    public static VMaskB Eq(VByte a, VByte b) => new(Vector.Equals(a.V, b.V));
    public static VMaskB Neq(VByte a, VByte b) => new(~Vector.Equals(a.V, b.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte Select(VMaskB mask, VByte ifTrue, VByte ifFalse)
        => new(Vector.ConditionalSelect(mask.Bits, ifTrue.V, ifFalse.V));

    /// <summary>
    /// Masked assignment: returns <c>mask ? newValue : current</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VByte Blend(VMaskB mask, VByte newValue) => Select(mask, newValue, this);

    /// <summary>
    /// Widen a full byte gang into its two short-gang halves (zero-extending; values 0..255).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (VShort Lower, VShort Upper) Widen(VByte v)
    {
        Vector.Widen(v.V, out var lo, out var hi);
        return (new VShort(Vector.AsVectorInt16(lo)), new VShort(Vector.AsVectorInt16(hi)));
    }

    /// <summary>
    /// Narrow two short-gang halves back into one byte gang (truncating to the low 8 bits).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte Narrow(VShort lower, VShort upper)
        => new(Vector.Narrow(Vector.AsVectorUInt16(lower.V), Vector.AsVectorUInt16(upper.V)));

    public bool Equals(VByte other) => V.Equals(other.V);
    public override bool Equals(object? obj) => obj is VByte b && Equals(b);
    public override int GetHashCode() => V.GetHashCode();
    public override string ToString() => V.ToString();

    public static bool operator ==(VByte left, VByte right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VByte left, VByte right)
    {
        return !(left == right);
    }
}
