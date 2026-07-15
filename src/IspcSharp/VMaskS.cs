using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp;

/// <summary>
/// A per-lane execution mask for 16-bit gangs (<see cref="VShort"/>).
/// Each lane is either all-ones (active/true) or all-zeros (inactive/false),
/// the same convention as <see cref="VMask"/> but at short-lane width
/// (SSE/NEON = 8 lanes, AVX2 = 16, AVX-512 = 32).
/// </summary>
[method: MethodImpl(MethodImplOptions.AggressiveInlining)]
public readonly struct VMaskS(Vector<short> bits) : IEquatable<VMaskS>
{
    /// <summary>
    /// Raw per-lane bits: -1 (all ones) = active, 0 = inactive.
    /// </summary>
    public readonly Vector<short> Bits = bits;

    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<short>.Count;
    }

    /// <summary>
    /// All lanes active.
    /// </summary>
    public static VMaskS All => new(Vector.Equals(Vector<short>.Zero, Vector<short>.Zero));

    /// <summary>
    /// No lanes active.
    /// </summary>
    public static VMaskS None => new(Vector<short>.Zero);

    /// <summary>
    /// Mask with the first <paramref name="count"/> lanes active, used for remainder/tail iterations.
    /// </summary>
    public static VMaskS FirstN(int count)
    {
        Span<short> tmp = stackalloc short[LaneCount];
        for (int i = 0; i < LaneCount; i++)
            tmp[i] = i < count ? (short)-1 : (short)0;
        return new VMaskS(new Vector<short>(tmp));
    }

    /// <summary>
    /// Widen a short-gang mask into its two int-gang halves (cross-gang-width bridge).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (VMask Lower, VMask Upper) Widen(VMaskS mask)
    {
        Vector.Widen(mask.Bits, out var lo, out var hi);
        return (new VMask(lo), new VMask(hi));
    }

    /// <summary>
    /// Narrow two int-gang masks back into one short-gang mask (cross-gang-width bridge).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VMaskS Narrow(VMask lower, VMask upper)
        => new(Vector.Narrow(lower.Bits, upper.Bits));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator &(VMaskS a, VMaskS b) => new(a.Bits & b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator |(VMaskS a, VMaskS b) => new(a.Bits | b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator ^(VMaskS a, VMaskS b) => new(a.Bits ^ b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator !(VMaskS a) => new(~a.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskS operator ~(VMaskS a) => new(~a.Bits);

    /// <summary>
    /// Subtract: lanes active in <paramref name="a"/> but not in <paramref name="b"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VMaskS AndNot(VMaskS a, VMaskS b) => new(Vector.AndNot(a.Bits, b.Bits));

    /// <summary>
    /// True if at least one lane is active.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Any() => !Vector.EqualsAll(Bits, Vector<short>.Zero);

    /// <summary>
    /// True if every lane is active.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool AllActive() => !Vector.EqualsAny(Bits, Vector<short>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool NoneActive() => Vector.EqualsAll(Bits, Vector<short>.Zero);

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

    public bool Equals(VMaskS other) => Bits.Equals(other.Bits);
    public override bool Equals(object? obj) => obj is VMaskS m && Equals(m);
    public override int GetHashCode() => Bits.GetHashCode();

    public override string ToString()
    {
        Span<char> c = stackalloc char[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            c[l] = IsLaneActive(l) ? '1' : '0';
        return $"VMaskS[{c}]";
    }

    public static bool operator ==(VMaskS left, VMaskS right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VMaskS left, VMaskS right)
    {
        return !(left == right);
    }
}
