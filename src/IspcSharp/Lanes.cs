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

    private static VInt RotateIndices(int offset)
    {
        int w = VInt.LaneCount;
        // ((l + offset) mod w + w) mod w, branch-free for negative offsets.
        int normalized = ((offset % w) + w) % w;
        Span<int> tmp = stackalloc int[w];
        for (int l = 0; l < w; l++)
            tmp[l] = (l + normalized) % w;
        return VInt.Load(tmp);
    }

    /// <summary>
    /// result[l] = v[indices[l]], arbitrary lane permutation (ISPC's shuffle) for short gangs.
    /// Hardware permutes via <c>VectorXXX.Shuffle</c> on net8.0+.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="indices">The indices to shuffle.</param>
    /// <returns>The shuffled vector.</returns>
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
        => Shuffle(v, VShort.ProgramIndex + (short)offset);   // wrapped negatives land out of range → 0

    /// <summary>
    /// result[l] = v[l + offset], lanes shifted past the edge become 0, ISPC's shift.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="offset">The offset to shift by.</param>
    /// <returns>The shifted vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VByte ShiftLanes(VByte v, int offset)
        => Shuffle(v, VByte.ProgramIndex + unchecked((byte)offset));   // wrapped negatives land out of range → 0

    private static VShort RotateIndicesS(int offset)
    {
        int w = VShort.LaneCount;
        int normalized = ((offset % w) + w) % w;
        Span<short> tmp = stackalloc short[w];
        for (int l = 0; l < w; l++)
            tmp[l] = (short)((l + normalized) % w);
        return VShort.Load(tmp);
    }

    private static VByte RotateIndicesB(int offset)
    {
        int w = VByte.LaneCount;
        int normalized = ((offset % w) + w) % w;
        Span<byte> tmp = stackalloc byte[w];
        for (int l = 0; l < w; l++)
            tmp[l] = (byte)((l + normalized) % w);
        return VByte.Load(tmp);
    }
}
