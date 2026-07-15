using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp;

/// <summary>
/// A per-lane execution mask for 8-bit gangs (<see cref="VByte"/>).
/// Each lane is either all-ones (0xFF, active/true) or all-zeros (inactive/false),
/// the same convention as <see cref="VMask"/> but at byte-lane width
/// (SSE/NEON = 16 lanes, AVX2 = 32, AVX-512 = 64).
/// </summary>
[method: MethodImpl(MethodImplOptions.AggressiveInlining)]
public readonly struct VMaskB(Vector<byte> bits) : IEquatable<VMaskB>
{
    /// <summary>
    /// Raw per-lane bits: 0xFF (all ones) = active, 0 = inactive.
    /// </summary>
    public readonly Vector<byte> Bits = bits;

    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<byte>.Count;
    }

    /// <summary>
    /// All lanes active.
    /// </summary>
    public static VMaskB All => new(Vector.Equals(Vector<byte>.Zero, Vector<byte>.Zero));

    /// <summary>
    /// No lanes active.
    /// </summary>
    public static VMaskB None => new(Vector<byte>.Zero);

    /// <summary>
    /// Mask with the first <paramref name="count"/> lanes active, used for remainder/tail iterations.
    /// </summary>
    public static VMaskB FirstN(int count)
    {
        Span<byte> tmp = stackalloc byte[LaneCount];
        for (int i = 0; i < LaneCount; i++)
            tmp[i] = i < count ? (byte)0xFF : (byte)0;
        return new VMaskB(new Vector<byte>(tmp));
    }

    /// <summary>
    /// Widen a byte-gang mask into its two short-gang halves (cross-gang-width bridge).
    /// Widening goes through sbyte so the all-ones lanes sign-extend to all-ones shorts.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (VMaskS Lower, VMaskS Upper) Widen(VMaskB mask)
    {
        Vector.Widen(Vector.AsVectorSByte(mask.Bits), out var lo, out var hi);
        return (new VMaskS(lo), new VMaskS(hi));
    }

    /// <summary>
    /// Narrow two short-gang masks back into one byte-gang mask (cross-gang-width bridge).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VMaskB Narrow(VMaskS lower, VMaskS upper)
        => new(Vector.AsVectorByte(Vector.Narrow(lower.Bits, upper.Bits)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator &(VMaskB a, VMaskB b) => new(a.Bits & b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator |(VMaskB a, VMaskB b) => new(a.Bits | b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator ^(VMaskB a, VMaskB b) => new(a.Bits ^ b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator !(VMaskB a) => new(~a.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskB operator ~(VMaskB a) => new(~a.Bits);

    /// <summary>
    /// Subtract: lanes active in <paramref name="a"/> but not in <paramref name="b"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VMaskB AndNot(VMaskB a, VMaskB b) => new(Vector.AndNot(a.Bits, b.Bits));

    /// <summary>
    /// True if at least one lane is active.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Any() => !Vector.EqualsAll(Bits, Vector<byte>.Zero);

    /// <summary>
    /// True if every lane is active.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool AllActive() => !Vector.EqualsAny(Bits, Vector<byte>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool NoneActive() => Vector.EqualsAll(Bits, Vector<byte>.Zero);

    /// <summary>
    /// Number of active lanes.
    /// </summary>
    public int CountActive()
    {
        int c = 0;
        for (int l = 0; l < LaneCount; l++)
        {
            if (Bits[l] != 0)
                c++;
        }

        return c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsLaneActive(int lane) => Bits[lane] != 0;

    public bool Equals(VMaskB other) => Bits.Equals(other.Bits);
    public override bool Equals(object? obj) => obj is VMaskB m && Equals(m);
    public override int GetHashCode() => Bits.GetHashCode();

    public override string ToString()
    {
        Span<char> c = stackalloc char[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            c[l] = IsLaneActive(l) ? '1' : '0';
        return $"VMaskB[{c}]";
    }

    public static bool operator ==(VMaskB left, VMaskB right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VMaskB left, VMaskB right)
    {
        return !(left == right);
    }
}
