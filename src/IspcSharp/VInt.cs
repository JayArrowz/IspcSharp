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
        => new(new Vector<int>(source[offset..]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<int> destination, int offset = 0) => V.CopyTo(destination[offset..]);

    public void StoreMasked(Span<int> destination, int offset, VMask mask)
    {
        for (int l = 0; l < LaneCount; l++)
        {
            int idx = offset + l;
            if (mask.IsLaneActive(l) && idx < destination.Length)
                destination[idx] = V[l];
        }
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
    /// Per-lane integer division (no SIMD integer divide exists, scalar loop).
    /// Throws <see cref="DivideByZeroException"/> if ANY lane's divisor is zero, in masked
    /// SPMD code, select a nonzero divisor into inactive lanes first (the [Spmd] generator
    /// does this automatically).
    /// </summary>
    public static VInt operator /(VInt a, VInt b) => Divide(a, b);

    /// <summary>
    /// Per-lane integer remainder (scalar loop; see <see cref="op_Division"/> about zero divisors).
    /// </summary>
    public static VInt operator %(VInt a, VInt b) => Remainder(a, b);

    public static VInt Divide(VInt a, VInt b)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] / b.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    public static VInt Remainder(VInt a, VInt b)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] % b.V[l];
        return new VInt(new Vector<int>(tmp));
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
    /// Per-lane left shift: result[l] = a[l] &lt;&lt; (counts[l] &amp; 31).
    /// Hardware <c>vpsllvd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
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
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] << counts.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Per-lane arithmetic right shift: result[l] = a[l] &gt;&gt; (counts[l] &amp; 31).
    /// Hardware <c>vpsravd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
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
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] >> counts.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Per-lane logical right shift: result[l] = (uint)a[l] &gt;&gt; (counts[l] &amp; 31).
    /// Hardware <c>vpsrlvd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
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
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (int)((uint)a.V[l] >> counts.V[l]);
        return new VInt(new Vector<int>(tmp));
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
