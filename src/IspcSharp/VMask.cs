using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

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
    /// ISPC's <c>lanemask()</c>: bit <c>l</c> is set when lane <c>l</c> is active.
    ///
    /// One <c>vmovmskps</c>-class instruction. This is the bridge out of vector-land — once the
    /// mask is an integer you can <see cref="System.Numerics.BitOperations.TrailingZeroCount(uint)"/>
    /// to walk active lanes one at a time, <c>PopCount</c> it for a compaction offset, or test it
    /// against a constant. Prefer <see cref="Any"/>/<see cref="AllActive"/> for plain branches;
    /// reach for this when you need the lane <i>positions</i>, not just whether any exist.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong ToBitmask()
    {
#if NET8_0_OR_GREATER
        if (LaneCount == 16)
            return Vector512.AsVector512(Bits).ExtractMostSignificantBits();
        if (LaneCount == 8)
            return Vector256.AsVector256(Bits).ExtractMostSignificantBits();
        if (LaneCount == 4)
            return Vector128.AsVector128(Bits).ExtractMostSignificantBits();
#endif
        return ToBitmaskPortable();
    }

    /// <summary>
    /// Any gang width without a movemask instruction, including one wider than this build knows
    /// about. <c>ulong</c> is the widest bitmask there is, so past 64 lanes this throws rather
    /// than silently wrapping the shift — C# masks a shift count to 63, which would fold lane 64
    /// onto bit 0 and return a plausible-looking wrong answer. Not reachable while
    /// <c>Vector&lt;T&gt;</c> tops out at 512 bits; the check is here so that if it ever is, it
    /// fails loudly.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private ulong ToBitmaskPortable()
    {
        if (LaneCount > 64)
            throw new NotSupportedException($"ToBitmask needs <= 64 lanes, gang is {LaneCount}");

        ulong m = 0;
        for (int l = 0; l < LaneCount; l++)
        {
            if (Bits[l] != 0)
                m |= 1UL << l;
        }

        return m;
    }

    /// <summary>
    /// Number of active lanes, ISPC's popcnt(lanemask()).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int CountActive() => System.Numerics.BitOperations.PopCount(ToBitmask());

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
