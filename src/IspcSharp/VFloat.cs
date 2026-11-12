using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp;

/// <summary>
/// A "varying float", one float per SIMD lane. The ISPC equivalent of <c>varying float</c>.
/// Wraps <see cref="Vector{T}"/> so the width automatically matches the best SIMD ISA
/// available at runtime (SSE = 4 lanes, AVX2 = 8, AVX-512 = 16, NEON = 4).
/// </summary>
public readonly struct VFloat : IEquatable<VFloat>
{
    public readonly Vector<float> V;

    /// <summary>
    /// Number of lanes (floats) processed per instruction on this machine.
    /// </summary>
    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<float>.Count;
    }

    public static VFloat Zero => new(Vector<float>.Zero);
    public static VFloat One => new(Vector<float>.One);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VFloat(Vector<float> v) => V = v;

    /// <summary>
    /// Broadcast a uniform (scalar) value to all lanes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VFloat(float uniform) => V = new Vector<float>(uniform);

    /// <summary>
    /// Contiguous (aligned or unaligned) load of LaneCount floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Load(ReadOnlySpan<float> source) => new(new Vector<float>(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Load(ReadOnlySpan<float> source, int offset)
        => new(LaneIO.Load(source, offset));

    /// <summary>
    /// Contiguous load of LaneCount elements from an array (same as the span overload, without
    /// rebuilding a span on every call).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Load(float[] source, int offset) => new(LaneIO.Load(source, offset));

    /// <summary>
    /// Contiguous store of LaneCount elements into an array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(float[] destination, int offset) => LaneIO.Store(V, destination, offset);

    /// <summary>
    /// Masked load: inactive lanes receive <paramref name="fallback"/>. Safe at buffer tails.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat LoadMasked(ReadOnlySpan<float> source, int offset, VMask mask, float fallback = 0f)
        // A full gang that fits is one plain load; the per-lane path is only for real tails.
        => mask.AllActive() && LaneIO.Fits<float>(source.Length, offset)
            ? new(LaneIO.LoadUnchecked(source, offset))
            : new(LaneIO.LoadMaskedSlow(source, offset, mask.ToBitmask(), fallback));

    /// <summary>
    /// Contiguous store of LaneCount floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<float> destination) => V.CopyTo(destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<float> destination, int offset) => LaneIO.Store(V, destination, offset);

    /// <summary>
    /// Non-temporal (streaming) store: bypasses the cache when the target is 32-byte aligned,
    /// else falls back to an ordinary store. Weakly ordered, pair with <see cref="Memory.StoreFence"/>
    /// after the loop. See <see cref="SpmdAttribute.Streaming"/>.
    /// </summary>
    public unsafe void StoreNonTemporal(Span<float> destination, int offset = 0)
    {
        // The streaming instructions write through a raw pointer, so the whole gang has to be
        // inside the buffer before anything else happens.
        if (!LaneIO.Fits<float>(destination.Length, offset))
            LaneIO.ThrowGangOutOfRange();
#if NET8_0_OR_GREATER
        fixed (float* start = destination)
        {
            float* p = start + offset;
            if (System.Runtime.Intrinsics.X86.Avx.IsSupported && LaneCount == 8)
            {
                if (((nuint)p & 31u) == 0)
                {
                    System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(p, System.Runtime.Intrinsics.Vector256.AsVector256(V));
                    return;
                }
            }
            else if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
            {
                if (((nuint)p & 63u) == 0)
                {
                    System.Runtime.Intrinsics.X86.Avx512F.StoreAlignedNonTemporal(p, System.Runtime.Intrinsics.Vector512.AsVector512(V));
                    return;
                }
            }
            else if (System.Runtime.Intrinsics.X86.Sse.IsSupported && LaneCount == 4)
            {
                if (((nuint)p & 15u) == 0)
                {
                    System.Runtime.Intrinsics.X86.Sse.StoreAlignedNonTemporal(p, System.Runtime.Intrinsics.Vector128.AsVector128(V));
                    return;
                }
            }
        }
#endif
        LaneIO.StoreUnchecked(V, destination, offset);
    }

    /// <summary>
    /// Masked store: only active lanes write. Safe at buffer tails.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void StoreMasked(Span<float> destination, int offset, VMask mask)
    {
        // A full gang that fits is one plain store; the per-lane path is only for real tails.
        if (mask.AllActive() && LaneIO.Fits<float>(destination.Length, offset))
            LaneIO.StoreUnchecked(V, destination, offset);
        else
            LaneIO.StoreMaskedSlow(V, destination, offset, mask.ToBitmask());
    }

    /// <summary>
    /// Extract a single lane (slow, for debugging/remainders, not hot loops).
    /// </summary>
    public float GetLane(int lane) => V[lane];

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator +(VFloat a, VFloat b) => new(a.V + b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator -(VFloat a, VFloat b) => new(a.V - b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator *(VFloat a, VFloat b) => new(a.V * b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator /(VFloat a, VFloat b) => new(a.V / b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator -(VFloat a) => new(-a.V);

    // Mixed uniform/varying arithmetic, mirrors ISPC's implicit uniform->varying promotion.
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator +(VFloat a, float b) => new(a.V + new Vector<float>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator +(float a, VFloat b) => b + a;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator -(VFloat a, float b) => new(a.V - new Vector<float>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator -(float a, VFloat b) => new(new Vector<float>(a) - b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator *(VFloat a, float b) => new(a.V * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator *(float a, VFloat b) => b * a;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator /(VFloat a, float b) => new(a.V / new Vector<float>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VFloat operator /(float a, VFloat b) => new(new Vector<float>(a) / b.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator VFloat(float uniform) => new(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator <(VFloat a, VFloat b) => new(Vector.LessThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator >(VFloat a, VFloat b) => new(Vector.GreaterThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator <=(VFloat a, VFloat b) => new(Vector.LessThanOrEqual(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator >=(VFloat a, VFloat b) => new(Vector.GreaterThanOrEqual(a.V, b.V));
    public static VMask Eq(VFloat a, VFloat b) => new(Vector.Equals(a.V, b.V));
    public static VMask Neq(VFloat a, VFloat b) => new(~Vector.Equals(a.V, b.V));

    /// <summary>
    /// Fused multiply-add: a*b + c. Uses hardware FMA (one instruction, one rounding)
    /// on net8.0+ when available; falls back to mul+add elsewhere.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat MulAdd(VFloat a, VFloat b, VFloat c)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && Vector<float>.Count == 16)
        {
            var r = System.Runtime.Intrinsics.X86.Avx512F.FusedMultiplyAdd(
                System.Runtime.Intrinsics.Vector512.AsVector512(a.V),
                System.Runtime.Intrinsics.Vector512.AsVector512(b.V),
                System.Runtime.Intrinsics.Vector512.AsVector512(c.V));
            return new VFloat(System.Runtime.Intrinsics.Vector512.AsVector(r));
        }

        if (System.Runtime.Intrinsics.X86.Fma.IsSupported && Vector<float>.Count == 8)
        {
            var r = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                System.Runtime.Intrinsics.Vector256.AsVector256(a.V),
                System.Runtime.Intrinsics.Vector256.AsVector256(b.V),
                System.Runtime.Intrinsics.Vector256.AsVector256(c.V));
            return new VFloat(System.Runtime.Intrinsics.Vector256.AsVector(r));
        }

        if (System.Runtime.Intrinsics.X86.Fma.IsSupported && Vector<float>.Count == 4)
        {
            var r = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                System.Runtime.Intrinsics.Vector128.AsVector128(a.V),
                System.Runtime.Intrinsics.Vector128.AsVector128(b.V),
                System.Runtime.Intrinsics.Vector128.AsVector128(c.V));
            return new VFloat(System.Runtime.Intrinsics.Vector128.AsVector(r));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && Vector<float>.Count == 4)
        {
            // NEON fmla computes addend + left*right (one instruction, one rounding).
            var r = System.Runtime.Intrinsics.Arm.AdvSimd.FusedMultiplyAdd(
                System.Runtime.Intrinsics.Vector128.AsVector128(c.V),
                System.Runtime.Intrinsics.Vector128.AsVector128(a.V),
                System.Runtime.Intrinsics.Vector128.AsVector128(b.V));
            return new VFloat(System.Runtime.Intrinsics.Vector128.AsVector(r));
        }
#endif
        return (a * b) + c;
    }

    /// <summary>
    /// Bitcast: reinterpret the 32 bits of each lane as an int (no conversion).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VInt AsInt() => new(Vector.AsVectorInt32(V));

    /// <summary>
    /// Per-lane: <c>mask ? ifTrue : ifFalse</c>. This is how <c>if</c> statements execute in SPMD code.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Select(VMask mask, VFloat ifTrue, VFloat ifFalse)
        => new(Vector.ConditionalSelect(mask.Bits, ifTrue.V, ifFalse.V));

    /// <summary>
    /// Masked assignment: returns <c>mask ? newValue : current</c>. Use for writes inside divergent control flow.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VFloat Blend(VMask mask, VFloat newValue) => Select(mask, newValue, this);

    public bool Equals(VFloat other) => V.Equals(other.V);
    public override bool Equals(object? obj) => obj is VFloat f && Equals(f);
    public override int GetHashCode() => V.GetHashCode();
    public override string ToString() => V.ToString();

    public static bool operator ==(VFloat left, VFloat right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VFloat left, VFloat right)
    {
        return !(left == right);
    }
}
