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
}
