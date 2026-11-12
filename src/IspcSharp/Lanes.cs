using System;
using System.Runtime.CompilerServices;

namespace IspcSharp;

/// <summary>
/// Cross-lane primitives, ISPC's <c>shuffle</c> / <c>rotate</c> / <c>broadcast</c> /
/// <c>shift</c> stdlib family. On net8.0+ full-width shuffles use hardware permutes via
/// <c>Vector128/256/512.Shuffle</c>; elsewhere a portable per-lane loop.
/// Out-of-range lane indices produce 0 (matching <c>VectorXXX.Shuffle</c> semantics).
/// </summary>
public static class Lanes
{
    /// <summary>
    /// result[l] = v[indices[l]], arbitrary lane permutation (ISPC's shuffle).
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="indices">The indices to shuffle.</param>
    /// <returns>The shuffled vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Shuffle(VFloat v, VInt indices)
    {
#if NET8_0_OR_GREATER
        if (VFloat.LaneCount == 16)
        {
            return new VFloat(System.Runtime.Intrinsics.Vector512.AsVector(
                System.Runtime.Intrinsics.Vector512.Shuffle(
                    System.Runtime.Intrinsics.Vector512.AsVector512(v.V),
                    System.Runtime.Intrinsics.Vector512.AsVector512(indices.V))));
        }

        if (VFloat.LaneCount == 8)
        {
            return new VFloat(System.Runtime.Intrinsics.Vector256.AsVector(
                System.Runtime.Intrinsics.Vector256.Shuffle(
                    System.Runtime.Intrinsics.Vector256.AsVector256(v.V),
                    System.Runtime.Intrinsics.Vector256.AsVector256(indices.V))));
        }

        if (VFloat.LaneCount == 4)
        {
            return new VFloat(System.Runtime.Intrinsics.Vector128.AsVector(
                System.Runtime.Intrinsics.Vector128.Shuffle(
                    System.Runtime.Intrinsics.Vector128.AsVector128(v.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(indices.V))));
        }
#endif
        return ShufflePortable(v, indices);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VFloat ShufflePortable(VFloat v, VInt indices)
    {
        Span<float> tmp = stackalloc float[VFloat.LaneCount];
        for (int l = 0; l < VFloat.LaneCount; l++)
        {
            int idx = indices.V[l];
            tmp[l] = (uint)idx < (uint)VFloat.LaneCount ? v.V[idx] : 0f;
        }

        return VFloat.Load(tmp);
    }

    /// <summary>
    /// result[l] = v[indices[l]], arbitrary lane permutation (ISPC's shuffle).
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="indices">The indices to shuffle.</param>
    /// <returns>The shuffled vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Shuffle(VInt v, VInt indices)
    {
#if NET8_0_OR_GREATER
        if (VInt.LaneCount == 16)
        {
            return new VInt(System.Runtime.Intrinsics.Vector512.AsVector(
                System.Runtime.Intrinsics.Vector512.Shuffle(
                    System.Runtime.Intrinsics.Vector512.AsVector512(v.V),
                    System.Runtime.Intrinsics.Vector512.AsVector512(indices.V))));
        }

        if (VInt.LaneCount == 8)
        {
            return new VInt(System.Runtime.Intrinsics.Vector256.AsVector(
                System.Runtime.Intrinsics.Vector256.Shuffle(
                    System.Runtime.Intrinsics.Vector256.AsVector256(v.V),
                    System.Runtime.Intrinsics.Vector256.AsVector256(indices.V))));
        }

        if (VInt.LaneCount == 4)
        {
            return new VInt(System.Runtime.Intrinsics.Vector128.AsVector(
                System.Runtime.Intrinsics.Vector128.Shuffle(
                    System.Runtime.Intrinsics.Vector128.AsVector128(v.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(indices.V))));
        }
#endif
        return ShufflePortable(v, indices);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VInt ShufflePortable(VInt v, VInt indices)
    {
        Span<int> tmp = stackalloc int[VInt.LaneCount];
        for (int l = 0; l < VInt.LaneCount; l++)
        {
            int idx = indices.V[l];
            tmp[l] = (uint)idx < (uint)VInt.LaneCount ? v.V[idx] : 0;
        }

        return VInt.Load(tmp);
    }

    /// <summary>
    /// result[l] = v[(l + offset) mod LaneCount], ISPC's rotate.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to rotate by.</param>
    /// <returns>The rotated vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Rotate(VFloat v, int offset)
        => Shuffle(v, RotateIndices(offset));

    /// <summary>
    /// result[l] = v[(l + offset) mod LaneCount], ISPC's rotate.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to rotate by.</param>
    /// <returns>The rotated vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Rotate(VInt v, int offset)
        => Shuffle(v, RotateIndices(offset));

    /// <summary>
    /// All lanes take v[lane], ISPC's broadcast.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="lane">The lane to broadcast.</param>
    /// <returns>The broadcast vector.</returns>
    public static VFloat Broadcast(VFloat v, int lane) => new(v.V[lane]);

    /// <summary>
    /// All lanes take v[lane], ISPC's broadcast.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="lane">The lane to broadcast.</param>
    /// <returns>The broadcast vector.</returns>
    public static VInt Broadcast(VInt v, int lane) => new(v.V[lane]);

    /// <summary>
    /// result[l] = v[l + offset], lanes shifted past the edge become 0, ISPC's shift.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to shift by.</param>
    /// <returns>The shifted vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat ShiftLanes(VFloat v, int offset)
        => Shuffle(v, VInt.ProgramIndex + offset);   // out-of-range indices → 0

    /// <summary>
    /// result[l] = v[l + offset], lanes shifted past the edge become 0, ISPC's shift.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to shift by.</param>
    /// <returns>The shifted vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt ShiftLanes(VInt v, int offset)
        => Shuffle(v, VInt.ProgramIndex + offset);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static VInt RotateIndices(int offset)
    {
        // (l + offset) mod w. The lane count is a power of two, so the wrap is a mask — which
        // also folds negative offsets the right way, with no per-lane loop.
        int w = VInt.LaneCount;
        return (VInt.ProgramIndex + (offset & (w - 1))) & new VInt(w - 1);
    }

    /// <summary>
    /// result[l] = v[indices[l]], arbitrary lane permutation (ISPC's shuffle) for short gangs.
    /// Hardware permutes via <c>VectorXXX.Shuffle</c> on net8.0+.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="indices">The indices to shuffle.</param>
    /// <returns>The shuffled vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort Shuffle(VShort v, VShort indices)
    {
#if NET8_0_OR_GREATER
        if (VShort.LaneCount == 32)
        {
            return new VShort(System.Runtime.Intrinsics.Vector512.AsVector(
                System.Runtime.Intrinsics.Vector512.Shuffle(
                    System.Runtime.Intrinsics.Vector512.AsVector512(v.V),
                    System.Runtime.Intrinsics.Vector512.AsVector512(indices.V))));
        }

        if (VShort.LaneCount == 16)
        {
            return new VShort(System.Runtime.Intrinsics.Vector256.AsVector(
                System.Runtime.Intrinsics.Vector256.Shuffle(
                    System.Runtime.Intrinsics.Vector256.AsVector256(v.V),
                    System.Runtime.Intrinsics.Vector256.AsVector256(indices.V))));
        }

        if (VShort.LaneCount == 8)
        {
            return new VShort(System.Runtime.Intrinsics.Vector128.AsVector(
                System.Runtime.Intrinsics.Vector128.Shuffle(
                    System.Runtime.Intrinsics.Vector128.AsVector128(v.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(indices.V))));
        }
#endif
        return ShufflePortable(v, indices);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VShort ShufflePortable(VShort v, VShort indices)
    {
        Span<short> tmp = stackalloc short[VShort.LaneCount];
        for (int l = 0; l < VShort.LaneCount; l++)
        {
            int idx = indices.V[l];
            tmp[l] = (uint)idx < (uint)VShort.LaneCount ? v.V[idx] : (short)0;
        }

        return VShort.Load(tmp);
    }

    /// <summary>
    /// result[l] = v[indices[l]], arbitrary lane permutation (ISPC's shuffle) for byte gangs.
    /// Hardware <c>pshufb</c>-class permutes via <c>VectorXXX.Shuffle</c> on net8.0+.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="indices">The indices to shuffle.</param>
    /// <returns>The shuffled vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte Shuffle(VByte v, VByte indices)
    {
#if NET8_0_OR_GREATER
        if (VByte.LaneCount == 64)
        {
            return new VByte(System.Runtime.Intrinsics.Vector512.AsVector(
                System.Runtime.Intrinsics.Vector512.Shuffle(
                    System.Runtime.Intrinsics.Vector512.AsVector512(v.V),
                    System.Runtime.Intrinsics.Vector512.AsVector512(indices.V))));
        }

        if (VByte.LaneCount == 32)
        {
            return new VByte(System.Runtime.Intrinsics.Vector256.AsVector(
                System.Runtime.Intrinsics.Vector256.Shuffle(
                    System.Runtime.Intrinsics.Vector256.AsVector256(v.V),
                    System.Runtime.Intrinsics.Vector256.AsVector256(indices.V))));
        }

        if (VByte.LaneCount == 16)
        {
            return new VByte(System.Runtime.Intrinsics.Vector128.AsVector(
                System.Runtime.Intrinsics.Vector128.Shuffle(
                    System.Runtime.Intrinsics.Vector128.AsVector128(v.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(indices.V))));
        }
#endif
        return ShufflePortable(v, indices);
    }

    /// <summary>Portable fallback, kept out of line: <c>stackalloc</c> emits <c>localloc</c>,
    /// which makes the enclosing method un-inlinable no matter what path actually runs.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static VByte ShufflePortable(VByte v, VByte indices)
    {
        Span<byte> tmp = stackalloc byte[VByte.LaneCount];
        for (int l = 0; l < VByte.LaneCount; l++)
        {
            int idx = indices.V[l];
            tmp[l] = idx < VByte.LaneCount ? v.V[idx] : (byte)0;
        }

        return VByte.Load(tmp);
    }

    /// <summary>
    /// result[l] = v[(l + offset) mod LaneCount], ISPC's rotate.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to rotate by.</param>
    /// <returns>The rotated vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort Rotate(VShort v, int offset)
        => Shuffle(v, RotateIndicesS(offset));

    /// <summary>
    /// result[l] = v[(l + offset) mod LaneCount], ISPC's rotate.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to rotate by.</param>
    /// <returns>The rotated vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte Rotate(VByte v, int offset)
        => Shuffle(v, RotateIndicesB(offset));

    /// <summary>
    /// All lanes take v[lane], ISPC's broadcast.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="lane">The lane to broadcast.</param>
    /// <returns>The broadcast vector.</returns>
    public static VShort Broadcast(VShort v, int lane) => new(v.V[lane]);

    /// <summary>
    /// All lanes take v[lane], ISPC's broadcast.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="lane">The lane to broadcast.</param>
    /// <returns>The broadcast vector.</returns>
    public static VByte Broadcast(VByte v, int lane) => new(v.V[lane]);

    /// <summary>
    /// result[l] = v[l + offset], lanes shifted past the edge become 0, ISPC's shift.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to shift by.</param>
    /// <returns>The shifted vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VShort ShiftLanes(VShort v, int offset)
    {
        // The index gang is only 16 bits wide, so an offset beyond the gang must be answered
        // here: truncated to short it could wrap back into range and return live lanes.
        if ((uint)(offset + VShort.LaneCount) >= (uint)(2 * VShort.LaneCount))
            return VShort.Zero;
        return Shuffle(v, VShort.ProgramIndex + (short)offset);   // wrapped negatives land out of range → 0
    }

    /// <summary>
    /// result[l] = v[l + offset], lanes shifted past the edge become 0, ISPC's shift.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to shift by.</param>
    /// <returns>The shifted vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte ShiftLanes(VByte v, int offset)
    {
        // The index gang is only 8 bits wide, so an offset beyond the gang must be answered
        // here: truncated to byte it could wrap back into range and return live lanes.
        if ((uint)(offset + VByte.LaneCount) >= (uint)(2 * VByte.LaneCount))
            return VByte.Zero;
        return Shuffle(v, VByte.ProgramIndex + unchecked((byte)offset));   // wrapped negatives land out of range → 0
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static VShort RotateIndicesS(int offset)
    {
        int w = VShort.LaneCount;
        return (VShort.ProgramIndex + (short)(offset & (w - 1))) & new VShort((short)(w - 1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static VByte RotateIndicesB(int offset)
    {
        int w = VByte.LaneCount;
        return (VByte.ProgramIndex + (byte)(offset & (w - 1))) & new VByte((byte)(w - 1));
    }
}
