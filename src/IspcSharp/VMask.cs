using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp;

/// <summary>
/// A per-lane execution mask, ISPC's "varying bool" / execution mask.
/// Each lane is either all-ones (active/true) or all-zeros (inactive/false).
/// All divergent control flow (if/else, break, continue, return, while) is expressed
/// by AND-ing and inverting masks instead of actually branching.
/// </summary>
[method: MethodImpl(MethodImplOptions.AggressiveInlining)]
public readonly struct VMask(Vector<int> bits) : IEquatable<VMask>
{
    /// <summary>
    /// Raw per-lane bits: -1 (all ones) = active, 0 = inactive.
    /// </summary>
    public readonly Vector<int> Bits = bits;

    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<int>.Count;
    }

    /// <summary>
    /// All lanes active.
    /// </summary>
    public static VMask All => new(Vector.Equals(Vector<int>.Zero, Vector<int>.Zero));

    /// <summary>
    /// No lanes active.
    /// </summary>
    public static VMask None => new(Vector<int>.Zero);

    /// <summary>
    /// Mask with the first <paramref name="count"/> lanes active, used for remainder/tail iterations.
    /// </summary>
    public static VMask FirstN(int count)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int i = 0; i < LaneCount; i++)
            tmp[i] = i < count ? -1 : 0;
        return new VMask(new Vector<int>(tmp));
    }

    /// <summary>
    /// Narrow two double-gang masks back into one float-gang mask (cross-gang-width bridge).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VMask Narrow(VMaskD lower, VMaskD upper)
        => new(Vector.Narrow(lower.Bits, upper.Bits));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator &(VMask a, VMask b) => new(a.Bits & b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator |(VMask a, VMask b) => new(a.Bits | b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator ^(VMask a, VMask b) => new(a.Bits ^ b.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator !(VMask a) => new(~a.Bits);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator ~(VMask a) => new(~a.Bits);

    /// <summary>
    /// Subtract: lanes active in <paramref name="a"/> but not in <paramref name="b"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VMask AndNot(VMask a, VMask b) => new(Vector.AndNot(a.Bits, b.Bits));

    /// <summary>
    /// True if at least one lane is active. Use to skip whole blocks when no lane needs them ("coherent" control flow).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Any() => !Vector.EqualsAll(Bits, Vector<int>.Zero);

    /// <summary>
    /// True if every lane is active. Use to take a fast uniform path with no masking.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool AllActive() => !Vector.EqualsAny(Bits, Vector<int>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool NoneActive() => Vector.EqualsAll(Bits, Vector<int>.Zero);

    /// <summary>
    /// Number of active lanes, ISPC's popcnt(lanemask()).
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

    public bool Equals(VMask other) => Bits.Equals(other.Bits);
    public override bool Equals(object? obj) => obj is VMask m && Equals(m);
    public override int GetHashCode() => Bits.GetHashCode();

    public override string ToString()
    {
        Span<char> c = stackalloc char[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            c[l] = IsLaneActive(l) ? '1' : '0';
        return $"VMask[{c}]";
    }

    public static bool operator ==(VMask left, VMask right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VMask left, VMask right)
    {
        return !(left == right);
    }
}
