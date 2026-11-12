using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace IspcSharp;

/// <summary>
/// Shared contiguous load/store plumbing for the lane types. A gang access is validated with
/// one range check (does a whole gang fit at this offset?) and then read or written directly,
/// instead of slicing a span and letting the <see cref="Vector{T}"/> constructor check it again.
/// </summary>
internal static class LaneIO
{
    /// <summary>
    /// True when a full gang fits in a buffer of <paramref name="length"/> elements at
    /// <paramref name="offset"/>. A negative offset fails too (it becomes a huge unsigned value).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool Fits<T>(int length, int offset)
        where T : struct
        => (ulong)(uint)offset + (uint)Vector<T>.Count <= (uint)length;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Load<T>(ReadOnlySpan<T> source, int offset)
        where T : struct
    {
        if (!Fits<T>(source.Length, offset))
            ThrowGangOutOfRange();
        return Vector.LoadUnsafe(ref MemoryMarshal.GetReference(source), (uint)offset);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store<T>(Vector<T> value, Span<T> destination, int offset)
        where T : struct
    {
        if (!Fits<T>(destination.Length, offset))
            ThrowGangOutOfRange();
        value.StoreUnsafe(ref MemoryMarshal.GetReference(destination), (uint)offset);
    }

    /// <summary>Caller has already established <see cref="Fits{T}"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> LoadUnchecked<T>(ReadOnlySpan<T> source, int offset)
        where T : struct
        => Vector.LoadUnsafe(ref MemoryMarshal.GetReference(source), (uint)offset);

    /// <summary>Caller has already established <see cref="Fits{T}"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreUnchecked<T>(Vector<T> value, Span<T> destination, int offset)
        where T : struct
        => value.StoreUnsafe(ref MemoryMarshal.GetReference(destination), (uint)offset);

    /// <summary>
    /// Per-lane masked load for partial gangs: lanes in <paramref name="activeBits"/> that fall
    /// inside the buffer are read, every other lane takes <paramref name="fallback"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static Vector<T> LoadMaskedSlow<T>(ReadOnlySpan<T> source, int offset, ulong activeBits, T fallback)
        where T : unmanaged
    {
        Span<T> tmp = stackalloc T[Vector<T>.Count];
        tmp.Fill(fallback);
        for (ulong bits = activeBits; bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            int idx = offset + l;
            if (idx < source.Length)
                tmp[l] = source[idx];
        }

        return new Vector<T>(tmp);
    }

    /// <summary>
    /// Per-lane masked store for partial gangs: lanes in <paramref name="activeBits"/> that fall
    /// inside the buffer are written, nothing else is touched.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void StoreMaskedSlow<T>(Vector<T> value, Span<T> destination, int offset, ulong activeBits)
        where T : unmanaged
    {
        Span<T> tmp = stackalloc T[Vector<T>.Count];
        value.CopyTo(tmp);
        for (ulong bits = activeBits; bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            int idx = offset + l;
            if (idx < destination.Length)
                destination[idx] = tmp[l];
        }
    }

    [DoesNotReturn]
    public static void ThrowGangOutOfRange()
        => throw new ArgumentOutOfRangeException("offset", "A full gang does not fit in the buffer at this offset.");
}
