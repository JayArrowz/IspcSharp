using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

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
    /// <summary>
    /// ISPC's <c>lanemask()</c>: bit <c>l</c> is set when lane <c>l</c> is active. One
    /// <c>movmsk</c>-class instruction. Mirrors <see cref="VMask.ToBitmask"/> at byte-lane width.
    ///
    /// Once the mask is an integer you can <c>TrailingZeroCount</c> it to walk active lanes one
    /// at a time, <c>PopCount</c> it for a compaction offset, or test it against a constant.
    /// Prefer <see cref="Any"/>/<see cref="AllActive"/> for a plain branch; reach for this when
    /// you need lane <i>positions</i>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong ToBitmask()
    {
#if NET8_0_OR_GREATER
        if (LaneCount == 64)
            return Vector512.AsVector512(Bits).ExtractMostSignificantBits();
        if (LaneCount == 32)
            return Vector256.AsVector256(Bits).ExtractMostSignificantBits();
        if (LaneCount == 16)
            return Vector128.AsVector128(Bits).ExtractMostSignificantBits();
#endif
        return ToBitmaskPortable();
    }

    /// <summary>
    /// Any gang width without a movemask instruction, including one wider than this build knows
    /// about. <c>ulong</c> is the widest bitmask there is, so past 64 lanes this throws rather
    /// than silently wrapping the shift — C# masks a shift count to 63, which would fold lane 64
    /// onto bit 0 and return a plausible-looking wrong answer.
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int CountActive() => System.Numerics.BitOperations.PopCount(ToBitmask());

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
