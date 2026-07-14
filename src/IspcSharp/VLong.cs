using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
#endif

namespace IspcSharp;

/// <summary>
/// A "varying long", one 64-bit integer per SIMD lane, the same width as <see cref="VDouble"/>
/// (AVX2 = 4 lanes, AVX-512 = 8, SSE/NEON = 2). A full integer lane type: arithmetic, bitwise,
/// shifts, comparisons, gather/scatter, and reductions. Masks are <see cref="VMaskD"/>.
/// Note: 64-bit packed multiply needs AVX-512DQ (else a JIT sequence), and integer divide is
/// scalar on all ISAs, so multiply/divide-heavy long kernels accelerate only partially.
/// </summary>
public readonly struct VLong : IEquatable<VLong>
{
    public readonly Vector<long> V;

    public static int LaneCount => Vector<long>.Count;
    public static VLong Zero => new(Vector<long>.Zero);
    public static VLong One => new(Vector<long>.One);

    private static readonly Vector<long> LaneIndex = CreateLaneIndex();
    private static Vector<long> CreateLaneIndex()
    {
        Span<long> tmp = stackalloc long[Vector<long>.Count];
        for (int i = 0; i < tmp.Length; i++)
            tmp[i] = i;
        return new Vector<long>(tmp);
    }

    /// <summary>
    /// {0, 1, 2, ... LaneCount-1}, ISPC's <c>programIndex</c> for 64-bit gangs.
    /// </summary>
    public static VLong ProgramIndex => new(LaneIndex);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VLong(Vector<long> v) => V = v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VLong(long uniform) => V = new Vector<long>(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong Load(ReadOnlySpan<long> source, int offset = 0)
        => new(new Vector<long>(source[offset..]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<long> destination, int offset = 0) => V.CopyTo(destination[offset..]);

    /// <summary>
    /// Masked load: inactive lanes receive <paramref name="fallback"/>. Safe at buffer tails.
    /// </summary>
    public static VLong LoadMasked(ReadOnlySpan<long> source, int offset, VMaskD mask, long fallback = 0L)
    {
        Span<long> tmp = stackalloc long[LaneCount];
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
    public void StoreMasked(Span<long> destination, int offset, VMaskD mask)
    {
        for (int l = 0; l < LaneCount; l++)
        {
            int idx = offset + l;
            if (mask.IsLaneActive(l) && idx < destination.Length)
                destination[idx] = V[l];
        }
    }

    public long GetLane(int lane) => V[lane];

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator +(VLong a, VLong b) => new(a.V + b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator -(VLong a, VLong b) => new(a.V - b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator *(VLong a, VLong b) => new(a.V * b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator +(VLong a, long b) => new(a.V + new Vector<long>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator -(VLong a, long b) => new(a.V - new Vector<long>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator *(VLong a, long b) => new(a.V * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator -(VLong a) => new(-a.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator &(VLong a, VLong b) => new(a.V & b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator |(VLong a, VLong b) => new(a.V | b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator ^(VLong a, VLong b) => new(a.V ^ b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong operator ~(VLong a) => new(~a.V);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator VLong(long uniform) => new(uniform);

    public static VLong operator /(VLong a, VLong b) => Divide(a, b);
    public static VLong operator %(VLong a, VLong b) => Remainder(a, b);

    public static VLong Divide(VLong a, VLong b)
    {
        Span<long> tmp = stackalloc long[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] / b.V[l];
        return new VLong(new Vector<long>(tmp));
    }

    public static VLong Remainder(VLong a, VLong b)
    {
        Span<long> tmp = stackalloc long[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] % b.V[l];
        return new VLong(new Vector<long>(tmp));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong operator <<(VLong a, int count)
    {
#if NET8_0_OR_GREATER
        return new VLong(Vector.ShiftLeft(a.V, count));
#else
        Span<long> tmp = stackalloc long[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = a.V[l] << count;
        return new VLong(new Vector<long>(tmp));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong operator >>(VLong a, int count)
    {
#if NET8_0_OR_GREATER
        return new VLong(Vector.ShiftRightArithmetic(a.V, count));
#else
        Span<long> tmp = stackalloc long[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = a.V[l] >> count;
        return new VLong(new Vector<long>(tmp));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong ShiftRightLogical(VLong a, int count)
    {
#if NET8_0_OR_GREATER
        return new VLong(Vector.AsVectorInt64(Vector.ShiftRightLogical(Vector.AsVectorUInt64(a.V), count)));
#else
        Span<long> tmp = stackalloc long[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (long)((ulong)a.V[l] >> count);
        return new VLong(new Vector<long>(tmp));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong Min(VLong a, VLong b) => new(Vector.Min(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong Max(VLong a, VLong b) => new(Vector.Max(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong Abs(VLong a) => new(Vector.Abs(a.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator <(VLong a, VLong b) => new(Vector.LessThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator >(VLong a, VLong b) => new(Vector.GreaterThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator <=(VLong a, VLong b) => new(Vector.LessThanOrEqual(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator >=(VLong a, VLong b) => new(Vector.GreaterThanOrEqual(a.V, b.V));
    public static VMaskD Eq(VLong a, VLong b) => new(Vector.Equals(a.V, b.V));
    public static VMaskD Neq(VLong a, VLong b) => new(~Vector.Equals(a.V, b.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong Select(VMaskD mask, VLong ifTrue, VLong ifFalse)
        => new(Vector.ConditionalSelect(mask.Bits, ifTrue.V, ifFalse.V));

    /// <summary>
    /// Masked assignment: returns <c>mask ? newValue : current</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VLong Blend(VMaskD mask, VLong newValue) => Select(mask, newValue, this);

    /// <summary>
    /// Per-lane truncation of doubles to 64-bit indices.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong FromDoubleTruncate(VDouble d) => new(Vector.ConvertToInt64(d.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VDouble ToDouble() => new(Vector.ConvertToDouble(V));

    public bool Equals(VLong other) => V.Equals(other.V);
    public override bool Equals(object? obj) => obj is VLong l && Equals(l);
    public override int GetHashCode() => V.GetHashCode();
    public override string ToString() => V.ToString();

    public static bool operator ==(VLong left, VLong right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VLong left, VLong right)
    {
        return !(left == right);
    }
}
