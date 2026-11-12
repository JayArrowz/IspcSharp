using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace IspcSharp;

/// <summary>
/// Horizontal reductions across lanes, ISPC's reduce_add / reduce_min / reduce_max family —
/// plus gather/scatter for indexed (non-contiguous) memory access.
/// </summary>
public static class Reduce
{
    /// <summary>
    /// Sum of all lanes, ISPC's reduce_add.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Add(VFloat v) => Vector.Dot(v.V, Vector<float>.One);

    /// <summary>
    /// Sum of active lanes only (inactive lanes contribute 0).
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="mask">The mask for active lanes.</param>
    /// <returns>The sum of active lanes.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Add(VFloat v, VMask mask)
        => Add(VFloat.Select(mask, v, VFloat.Zero));

    /// <summary>
    /// Sum of all lanes (wrapping, like the scalar loop).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Add(VInt v) => Vector.Sum(v.V);

    /// <summary>
    /// Minimum across all lanes.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <returns>The minimum value.</returns>
    public static float Min(VFloat v)
    {
        float m = v.V[0];
        for (int l = 1; l < VFloat.LaneCount; l++)
            m = Math.Min(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Minimum across active lanes (inactive lanes ignored via +Inf).
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="mask">The mask for active lanes.</param>
    /// <returns>The minimum value.</returns>
    public static float Min(VFloat v, VMask mask)
        => Min(VFloat.Select(mask, v, new VFloat(float.PositiveInfinity)));

    /// <summary>
    /// Maximum across all lanes.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <returns>The maximum value.</returns>
    public static float Max(VFloat v)
    {
        float m = v.V[0];
        for (int l = 1; l < VFloat.LaneCount; l++)
            m = Math.Max(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Maximum across active lanes (inactive lanes ignored via -Inf).
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="mask">The mask for active lanes.</param>
    /// <returns>The maximum value.</returns>
    public static float Max(VFloat v, VMask mask)
        => Max(VFloat.Select(mask, v, new VFloat(float.NegativeInfinity)));

    /// <summary>
    /// Minimum across all lanes.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <returns>The minimum value.</returns>
    public static int Min(VInt v)
    {
        int m = v.V[0];
        for (int l = 1; l < VInt.LaneCount; l++)
            m = Math.Min(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Minimum across active lanes (inactive lanes ignored via int.MaxValue).
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="mask">The mask for active lanes.</param>
    /// <returns>The minimum value.</returns>
    public static int Min(VInt v, VMask mask)
        => Min(VInt.Select(mask, v, new VInt(int.MaxValue)));

    /// <summary>
    /// Maximum across all lanes.
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <returns>The maximum value.</returns>
    public static int Max(VInt v)
    {
        int m = v.V[0];
        for (int l = 1; l < VInt.LaneCount; l++)
            m = Math.Max(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Maximum across active lanes (inactive lanes ignored via int.MinValue).
    /// </summary>
    /// <param name="v">The input vector.</param>
    /// <param name="mask">The mask for active lanes.</param>
    /// <returns>The maximum value.</returns>
    public static int Max(VInt v, VMask mask)
        => Max(VInt.Select(mask, v, new VInt(int.MinValue)));

    /// <summary>
    /// Sum of all lanes, ISPC's reduce_add (double).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Add(VDouble v) => VDouble.ReduceAdd(v);

    /// <summary>
    /// Sum of active lanes only (inactive lanes contribute 0).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Add(VDouble v, VMaskD mask) => VDouble.ReduceAdd(v, mask);

    /// <summary>
    /// Minimum across all lanes.
    /// </summary>
    public static double Min(VDouble v)
    {
        double m = v.V[0];
        for (int l = 1; l < VDouble.LaneCount; l++)
            m = Math.Min(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Minimum across active lanes (inactive lanes ignored via +Inf).
    /// </summary>
    public static double Min(VDouble v, VMaskD mask)
        => Min(VDouble.Select(mask, v, new VDouble(double.PositiveInfinity)));

    /// <summary>
    /// Maximum across all lanes.
    /// </summary>
    public static double Max(VDouble v)
    {
        double m = v.V[0];
        for (int l = 1; l < VDouble.LaneCount; l++)
            m = Math.Max(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Maximum across active lanes (inactive lanes ignored via -Inf).
    /// </summary>
    public static double Max(VDouble v, VMaskD mask)
        => Max(VDouble.Select(mask, v, new VDouble(double.NegativeInfinity)));

    /// <summary>
    /// Sum of all lanes of a full-float-gang-width double pair.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Add(VDouble2 v) => Add(v.Lower) + Add(v.Upper);

    /// <summary>
    /// Sum of active lanes only (full-width VMask).
    /// </summary>
    public static double Add(VDouble2 v, VMask mask)
    {
        var (lo, hi) = VMaskD.Widen(mask);
        return Add(v.Lower, lo) + Add(v.Upper, hi);
    }

    /// <summary>
    /// Minimum across all lanes of a double pair.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Min(VDouble2 v) => Math.Min(Min(v.Lower), Min(v.Upper));

    /// <summary>
    /// Maximum across all lanes of a double pair.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Max(VDouble2 v) => Math.Max(Max(v.Lower), Max(v.Upper));

    /// <summary>
    /// Sum of all lanes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Add(VLong v) => Vector.Sum(v.V);

    /// <summary>
    /// Sum of active lanes only (inactive lanes contribute 0).
    /// </summary>
    public static long Add(VLong v, VMaskD mask) => Add(VLong.Select(mask, v, VLong.Zero));

    /// <summary>
    /// Minimum across all lanes.
    /// </summary>
    public static long Min(VLong v)
    {
        long m = v.V[0];
        for (int l = 1; l < VLong.LaneCount; l++)
            m = Math.Min(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Minimum across active lanes (inactive lanes ignored via long.MaxValue).
    /// </summary>
    public static long Min(VLong v, VMaskD mask) => Min(VLong.Select(mask, v, new VLong(long.MaxValue)));

    /// <summary>
    /// Maximum across all lanes.
    /// </summary>
    public static long Max(VLong v)
    {
        long m = v.V[0];
        for (int l = 1; l < VLong.LaneCount; l++)
            m = Math.Max(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Maximum across active lanes (inactive lanes ignored via long.MinValue).
    /// </summary>
    public static long Max(VLong v, VMaskD mask) => Max(VLong.Select(mask, v, new VLong(long.MinValue)));

    /// <summary>
    /// Sum of all lanes of a full-int-gang-width long pair.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Add(VLong2 v) => Add(v.Lower) + Add(v.Upper);

    /// <summary>
    /// Sum of active lanes only (full-width VMask).
    /// </summary>
    public static long Add(VLong2 v, VMask mask)
    {
        var (lo, hi) = VMaskD.Widen(mask);
        return Add(v.Lower, lo) + Add(v.Upper, hi);
    }

    /// <summary>
    /// Minimum across all lanes of a long pair.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Min(VLong2 v) => Math.Min(Min(v.Lower), Min(v.Upper));

    /// <summary>
    /// Maximum across all lanes of a long pair.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Max(VLong2 v) => Math.Max(Max(v.Lower), Max(v.Upper));

    /// <summary>
    /// Sum of all byte lanes, widened to int (a full gang can exceed 255).
    /// One hardware <c>psadbw</c> against zero (SSE2/AVX2/AVX-512BW) sums 8 bytes per
    /// 64-bit lane in a single instruction, and NEON <c>uaddlv</c> sums the whole gang in one;
    /// portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Add(VByte v)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VByte.LaneCount == 16)
            return System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.AddAcrossWidening(Vector128.AsVector128(v.V)).ToScalar();

        if (System.Runtime.Intrinsics.X86.Avx512BW.IsSupported && VByte.LaneCount == 64)
        {
            var sad = System.Runtime.Intrinsics.X86.Avx512BW.SumAbsoluteDifferences(
                Vector512.AsVector512(v.V), Vector512<byte>.Zero);
            return (int)Vector512.Sum(sad.AsUInt64());
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VByte.LaneCount == 32)
        {
            var sad = System.Runtime.Intrinsics.X86.Avx2.SumAbsoluteDifferences(
                Vector256.AsVector256(v.V), Vector256<byte>.Zero);
            return (int)Vector256.Sum(sad.AsUInt64());
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && VByte.LaneCount == 16)
        {
            var sad = System.Runtime.Intrinsics.X86.Sse2.SumAbsoluteDifferences(
                Vector128.AsVector128(v.V), Vector128<byte>.Zero);
            return (int)Vector128.Sum(sad.AsUInt64());
        }
#endif
        int sum = 0;
        for (int l = 0; l < VByte.LaneCount; l++)
            sum += v.V[l];
        return sum;
    }

    /// <summary>
    /// Sum of active byte lanes only (inactive lanes contribute 0), widened to int.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Add(VByte v, VMaskB mask) => Add(VByte.Select(mask, v, VByte.Zero));

    /// <summary>
    /// Minimum across all lanes.
    /// </summary>
    public static byte Min(VByte v)
    {
        byte m = v.V[0];
        for (int l = 1; l < VByte.LaneCount; l++)
            m = Math.Min(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Minimum across active lanes (inactive lanes ignored via byte.MaxValue).
    /// </summary>
    public static byte Min(VByte v, VMaskB mask) => Min(VByte.Select(mask, v, new VByte(byte.MaxValue)));

    /// <summary>
    /// Maximum across all lanes.
    /// </summary>
    public static byte Max(VByte v)
    {
        byte m = v.V[0];
        for (int l = 1; l < VByte.LaneCount; l++)
            m = Math.Max(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Maximum across active lanes (inactive lanes ignored via 0).
    /// </summary>
    public static byte Max(VByte v, VMaskB mask) => Max(VByte.Select(mask, v, VByte.Zero));

    /// <summary>
    /// Sum of all short lanes, widened to int (a full gang can exceed short range).
    /// SIMD widen + one vector add + horizontal sum on net8.0+; portable loop otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Add(VShort v)
    {
#if NET8_0_OR_GREATER
        // NEON saddlv: one widening across-vector add.
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VShort.LaneCount == 8)
            return System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.AddAcrossWidening(Vector128.AsVector128(v.V)).ToScalar();
        Vector.Widen(v.V, out var lo, out var hi);
        return Vector.Sum(lo + hi);
#else
        int sum = 0;
        for (int l = 0; l < VShort.LaneCount; l++)
            sum += v.V[l];
        return sum;
#endif
    }

    /// <summary>
    /// Sum of active short lanes only (inactive lanes contribute 0), widened to int.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Add(VShort v, VMaskS mask) => Add(VShort.Select(mask, v, VShort.Zero));

    /// <summary>
    /// Minimum across all lanes.
    /// </summary>
    public static short Min(VShort v)
    {
        short m = v.V[0];
        for (int l = 1; l < VShort.LaneCount; l++)
            m = Math.Min(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Minimum across active lanes (inactive lanes ignored via short.MaxValue).
    /// </summary>
    public static short Min(VShort v, VMaskS mask) => Min(VShort.Select(mask, v, new VShort(short.MaxValue)));

    /// <summary>
    /// Maximum across all lanes.
    /// </summary>
    public static short Max(VShort v)
    {
        short m = v.V[0];
        for (int l = 1; l < VShort.LaneCount; l++)
            m = Math.Max(m, v.V[l]);
        return m;
    }

    /// <summary>
    /// Maximum across active lanes (inactive lanes ignored via short.MinValue).
    /// </summary>
    public static short Max(VShort v, VMaskS mask) => Max(VShort.Select(mask, v, new VShort(short.MinValue)));
}

/// <summary>
/// Indexed (non-contiguous) memory access. Note: gather/scatter is inherently slower than
/// contiguous Load/Store even with hardware support, restructure to Structure-of-Arrays
/// layout where possible instead of reaching for these.
/// </summary>
public static class Memory
{
    /// <summary>
    /// Ordering fence after non-temporal (streaming) stores, so their writes are globally
    /// visible before the buffer is read. Emitted automatically at the end of a <see cref="SpmdAttribute.Streaming"/>
    /// kernel; call it manually if you use <c>StoreNonTemporal</c> directly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreFence()
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
            System.Runtime.Intrinsics.X86.Sse.StoreFence();
#endif
    }

    /// <summary>
    /// How many leading elements to process one at a time so that <c>buffer[index + result]</c>
    /// starts on a vector-width boundary — the alignment a non-temporal store needs to take
    /// effect. Never more than <paramref name="maxCount"/>; 0 where streaming stores have no
    /// hardware path. Emitted before the vector loop of a <see cref="SpmdAttribute.Streaming"/>
    /// kernel. (Advisory: if the GC later moves the buffer the stores fall back to ordinary
    /// ones, they never fault.)
    /// </summary>
    public static unsafe int StreamingPeel(ReadOnlySpan<float> buffer, int index, int maxCount)
    {
        fixed (float* p = buffer)
            return StreamingPeel((nuint)p, sizeof(float), Vector<float>.Count, buffer.Length, index, maxCount);
    }

    /// <inheritdoc cref="StreamingPeel(ReadOnlySpan{float}, int, int)"/>
    public static unsafe int StreamingPeel(ReadOnlySpan<double> buffer, int index, int maxCount)
    {
        fixed (double* p = buffer)
            return StreamingPeel((nuint)p, sizeof(double), Vector<double>.Count, buffer.Length, index, maxCount);
    }

    private static int StreamingPeel(nuint start, int elemSize, int laneCount, int length, int index, int maxCount)
    {
#if NET8_0_OR_GREATER
        if (!System.Runtime.Intrinsics.X86.Sse.IsSupported || (uint)index >= (uint)length || maxCount <= 0)
            return 0;
        nuint vectorBytes = (nuint)(elemSize * laneCount);
        nuint misaligned = (start + ((nuint)index * (nuint)elemSize)) & (vectorBytes - 1);
        if (misaligned == 0)
            return 0;
        nuint bytesToBoundary = vectorBytes - misaligned;
        // A buffer that is not even element-aligned can never reach a vector boundary.
        if (bytesToBoundary % (nuint)elemSize != 0)
            return 0;
        return Math.Min((int)(bytesToBoundary / (nuint)elemSize), maxCount);
#else
        return 0;
#endif
    }

    /// <summary>
    /// Bounds check for the hardware gather paths, which read through a raw pointer: throws
    /// <see cref="IndexOutOfRangeException"/> if any <b>active</b> lane's index is outside
    /// [0, <paramref name="length"/>), exactly as the per-lane paths (and the scalar loop the
    /// kernel was written as) would. One unsigned compare catches negative and too-large indices.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CheckGatherIndices(VInt indices, VMask mask, int length)
    {
        // index >= length (unsigned)  <=>  max(index, length) == index: vpmaxud + vpcmpeqd,
        // cheaper than an unsigned compare, which AVX2 has to synthesize.
        var idx = Vector.AsVectorUInt32(indices.V);
        var outOfRange = Vector.Equals(Vector.Max(idx, new Vector<uint>((uint)length)), idx);
        if ((Vector.AsVectorInt32(outOfRange) & mask.Bits) != Vector<int>.Zero)
            ThrowGatherIndexOutOfRange();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CheckGatherIndices(VLong indices, VMaskD mask, int length)
    {
        var outOfRange = Vector.GreaterThanOrEqual(Vector.AsVectorUInt64(indices.V), new Vector<ulong>((ulong)length));
        if ((Vector.AsVectorInt64(outOfRange) & mask.Bits) != Vector<long>.Zero)
            ThrowGatherIndexOutOfRange();
    }

    [System.Diagnostics.CodeAnalysis.DoesNotReturn]
    private static void ThrowGatherIndexOutOfRange()
        => throw new IndexOutOfRangeException("Gather index was outside the bounds of the source buffer.");

    /// <summary>
    /// Gather: result[lane] = source[indices[lane]] for active lanes; <paramref name="fallback"/> otherwise.
    /// Uses AVX2 hardware gather on net8.0+ when available (8-wide, doubled for a 16-wide AVX-512
    /// gang). Inactive lanes are never read; an out-of-range index in an active lane throws
    /// <see cref="IndexOutOfRangeException"/> on every path.
    /// </summary>
    public static unsafe VFloat Gather(ReadOnlySpan<float> source, VInt indices, VMask mask, float fallback = 0f)
    {
#if NET8_0_OR_GREATER
        // AVX-512 gangs (16 float lanes): .NET exposes no 512-bit gather intrinsic, so run the
        // AVX2 256-bit hardware gather on each 8-lane half. Without this a 16-wide gang would
        // drop to the portable per-lane path, the Linux/AVX-512 gather regression.
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VFloat.LaneCount == 16)
        {
            CheckGatherIndices(indices, mask, source.Length);
            fixed (float* p = source)
            {
                var idx = Vector512.AsVector512(indices.V);
                var idxLo = idx.GetLower();
                var idxHi = idx.GetUpper();
                var mm = Vector512.AsVector512(mask.Bits).AsSingle();
                var src = Vector256.Create(fallback);
                var lo = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idxLo, mm.GetLower(), 4);
                var hi = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idxHi, mm.GetUpper(), 4);
                return new VFloat(Vector512.AsVector(
                    Vector512.Create(lo, hi)));
            }
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VFloat.LaneCount == 8)
        {
            // Hardware masked gather: inactive lanes are not read (safe even if their index is junk).
            CheckGatherIndices(indices, mask, source.Length);
            fixed (float* p = source)
            {
                var idx = Vector256.AsVector256(indices.V);
                var m = Vector256.AsVector256(mask.Bits).AsSingle();
                var src = Vector256.Create(fallback);
                var r = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx, m, 4);
                return new VFloat(Vector256.AsVector(r));
            }
        }
        // NEON/SSE 4-lane fast path: full gangs build the vector directly, skipping the
        // stackalloc round-trip and per-lane mask tests (ARM has no hardware gather).
        if (VFloat.LaneCount == 4 && mask.AllActive())
        {
            var r = Vector128.Create(
                source[indices.V[0]], source[indices.V[1]],
                source[indices.V[2]], source[indices.V[3]]);
            return new VFloat(Vector128.AsVector(r));
        }
#endif
        // Portable fallback (no hardware gather at this width). Copy the index vector to the stack
        // once, indexing Vector<T> per lane instead spills the whole vector on every access, which
        // is what makes a 16-wide gang crawl here.
        int w = VFloat.LaneCount;
        Span<int> idxs = stackalloc int[w];
        indices.V.CopyTo(idxs);
        Span<float> tmp = stackalloc float[w];
        if (mask.AllActive())
        {
            for (int l = 0; l < w; l++)
                tmp[l] = source[idxs[l]];
        }
        else
        {
            for (int l = 0; l < w; l++)
                tmp[l] = mask.IsLaneActive(l) ? source[idxs[l]] : fallback;
        }

        return VFloat.Load(tmp);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat Gather(ReadOnlySpan<float> source, VInt indices)
        => Gather(source, indices, VMask.All);

    /// <summary>
    /// Scatter: destination[indices[lane]] = values[lane] for active lanes.
    /// If two active lanes carry the same index, the higher lane wins (matches ISPC).
    /// On net8.0+ the active-lane set is extracted with one hardware movemask
    /// (AVX-512 kmask / AVX vmovmskps / NEON) and only active lanes store, .NET does
    /// not expose x86 vscatterdps, so this is the fastest scatter available from managed code.
    /// </summary>
    public static void Scatter(Span<float> destination, VInt indices, VFloat values, VMask mask)
    {
        // Ascending lane order, so on a duplicate index the higher lane writes last and wins,
        // matching ISPC. ToBitmask always yields an answer, so there is no per-lane fallback.
        for (ulong bits = mask.ToBitmask(); bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            destination[indices.V[l]] = values.V[l];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Scatter(Span<float> destination, VInt indices, VFloat values)
        => Scatter(destination, indices, values, VMask.All);

    /// <summary>
    /// Gather for int lanes; same hardware paths as the float overload
    /// (AVX2 <c>vpgatherdd</c>, doubled for a 16-wide AVX-512 gang).
    /// </summary>
    public static unsafe VInt Gather(ReadOnlySpan<int> source, VInt indices, VMask mask, int fallback = 0)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VInt.LaneCount == 16)
        {
            CheckGatherIndices(indices, mask, source.Length);
            fixed (int* p = source)
            {
                var idx = Vector512.AsVector512(indices.V);
                var m = Vector512.AsVector512(mask.Bits);
                var src = Vector256.Create(fallback);
                var lo = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx.GetLower(), m.GetLower(), 4);
                var hi = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx.GetUpper(), m.GetUpper(), 4);
                return new VInt(Vector512.AsVector(
                    Vector512.Create(lo, hi)));
            }
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VInt.LaneCount == 8)
        {
            CheckGatherIndices(indices, mask, source.Length);
            fixed (int* p = source)
            {
                var idx = Vector256.AsVector256(indices.V);
                var m = Vector256.AsVector256(mask.Bits);
                var src = Vector256.Create(fallback);
                var r = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx, m, 4);
                return new VInt(Vector256.AsVector(r));
            }
        }
#endif
        int w = VInt.LaneCount;
        Span<int> idxs = stackalloc int[w];
        indices.V.CopyTo(idxs);
        Span<int> tmp = stackalloc int[w];
        if (mask.AllActive())
        {
            for (int l = 0; l < w; l++)
                tmp[l] = source[idxs[l]];
        }
        else
        {
            for (int l = 0; l < w; l++)
                tmp[l] = mask.IsLaneActive(l) ? source[idxs[l]] : fallback;
        }

        return VInt.Load(tmp);
    }

    /// <summary>
    /// Scatter for int lanes; same semantics and hardware mask path as the float overload.
    /// </summary>
    public static void Scatter(Span<int> destination, VInt indices, VInt values, VMask mask)
    {
        // Ascending lane order, so on a duplicate index the higher lane writes last and wins,
        // matching ISPC. ToBitmask always yields an answer, so there is no per-lane fallback.
        for (ulong bits = mask.ToBitmask(); bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            destination[indices.V[l]] = values.V[l];
        }
    }

    /// <summary>
    /// Gather for double lanes: result[lane] = source[indices[lane]] for active lanes.
    /// Uses AVX2 vgatherqpd on net8.0+ when available (inactive lanes are never read).
    /// </summary>
    public static unsafe VDouble Gather(ReadOnlySpan<double> source, VLong indices, VMaskD mask, double fallback = 0d)
    {
#if NET8_0_OR_GREATER
        // AVX-512 gang (8 double lanes): no 512-bit gather intrinsic, so run the AVX2 vgatherqpd
        // on each 4-lane half (otherwise a 16-wide double gang drops to the portable path).
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VDouble.LaneCount == 8)
        {
            CheckGatherIndices(indices, mask, source.Length);
            fixed (double* p = source)
            {
                var idx = Vector512.AsVector512(indices.V);
                var m = Vector512.AsVector512(mask.Bits).AsDouble();
                var src = Vector256.Create(fallback);
                var lo = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx.GetLower(), m.GetLower(), 8);
                var hi = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx.GetUpper(), m.GetUpper(), 8);
                return new VDouble(Vector512.AsVector(
                    Vector512.Create(lo, hi)));
            }
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VDouble.LaneCount == 4)
        {
            CheckGatherIndices(indices, mask, source.Length);
            fixed (double* p = source)
            {
                var idx = Vector256.AsVector256(indices.V);
                var m = Vector256.AsVector256(mask.Bits).AsDouble();
                var src = Vector256.Create(fallback);
                var r = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx, m, 8);
                return new VDouble(Vector256.AsVector(r));
            }
        }
#endif
        int w = VDouble.LaneCount;
        Span<long> idxs = stackalloc long[w];
        indices.V.CopyTo(idxs);
        Span<double> tmp = stackalloc double[w];
        if (mask.AllActive())
        {
            for (int l = 0; l < w; l++)
                tmp[l] = source[(int)idxs[l]];
        }
        else
        {
            for (int l = 0; l < w; l++)
                tmp[l] = mask.IsLaneActive(l) ? source[(int)idxs[l]] : fallback;
        }

        return VDouble.Load(tmp);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble Gather(ReadOnlySpan<double> source, VLong indices)
        => Gather(source, indices, VMaskD.All);

    /// <summary>
    /// Scatter for double lanes; same semantics and hardware mask path as the float overload.
    /// </summary>
    public static void Scatter(Span<double> destination, VLong indices, VDouble values, VMaskD mask)
    {
        // Ascending lane order, so on a duplicate index the higher lane writes last and wins,
        // matching ISPC. ToBitmask always yields an answer, so there is no per-lane fallback.
        for (ulong bits = mask.ToBitmask(); bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            destination[(int)indices.V[l]] = values.V[l];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Scatter(Span<double> destination, VLong indices, VDouble values)
        => Scatter(destination, indices, values, VMaskD.All);

    /// <summary>
    /// Gather for long lanes: result[lane] = source[indices[lane]] for active lanes.
    /// Uses AVX2 vpgatherqq on net8.0+ when available.
    /// </summary>
    public static unsafe VLong Gather(ReadOnlySpan<long> source, VLong indices, VMaskD mask, long fallback = 0L)
    {
#if NET8_0_OR_GREATER
        // AVX-512 gang (8 long lanes): run the AVX2 vpgatherqq on each 4-lane half.
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VLong.LaneCount == 8)
        {
            CheckGatherIndices(indices, mask, source.Length);
            fixed (long* p = source)
            {
                var idx = Vector512.AsVector512(indices.V);
                var m = Vector512.AsVector512(mask.Bits);
                var src = Vector256.Create(fallback);
                var lo = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx.GetLower(), m.GetLower(), 8);
                var hi = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx.GetUpper(), m.GetUpper(), 8);
                return new VLong(Vector512.AsVector(
                    Vector512.Create(lo, hi)));
            }
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VLong.LaneCount == 4)
        {
            CheckGatherIndices(indices, mask, source.Length);
            fixed (long* p = source)
            {
                var idx = Vector256.AsVector256(indices.V);
                var m = Vector256.AsVector256(mask.Bits);
                var src = Vector256.Create(fallback);
                var r = System.Runtime.Intrinsics.X86.Avx2.GatherMaskVector256(src, p, idx, m, 8);
                return new VLong(Vector256.AsVector(r));
            }
        }
#endif
        int w = VLong.LaneCount;
        Span<long> idxs = stackalloc long[w];
        indices.V.CopyTo(idxs);
        Span<long> tmp = stackalloc long[w];
        if (mask.AllActive())
        {
            for (int l = 0; l < w; l++)
                tmp[l] = source[(int)idxs[l]];
        }
        else
        {
            for (int l = 0; l < w; l++)
                tmp[l] = mask.IsLaneActive(l) ? source[(int)idxs[l]] : fallback;
        }

        return VLong.Load(tmp);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong Gather(ReadOnlySpan<long> source, VLong indices)
        => Gather(source, indices, VMaskD.All);

    /// <summary>
    /// Gather double lanes at full int-gang width (a <see cref="VDouble2"/> pair): the int
    /// index gang widens into its two long-gang halves, one hardware gather each. Used for
    /// double values indexed per-lane inside 32-bit (float/int) kernels.
    /// </summary>
    public static VDouble2 Gather(ReadOnlySpan<double> source, VInt indices, VMask mask, double fallback = 0d)
    {
        var (ilo, ihi) = VInt.Widen(indices);
        var (mlo, mhi) = VMaskD.Widen(mask);
        return new VDouble2(Gather(source, ilo, mlo, fallback), Gather(source, ihi, mhi, fallback));
    }

    /// <summary>
    /// Gather 64-bit integer lanes at full int-gang width (a <see cref="VLong2"/> pair);
    /// the integer mirror of the <see cref="VDouble2"/> overload above.
    /// </summary>
    public static VLong2 Gather(ReadOnlySpan<long> source, VInt indices, VMask mask, long fallback = 0L)
    {
        var (ilo, ihi) = VInt.Widen(indices);
        var (mlo, mhi) = VMaskD.Widen(mask);
        return new VLong2(Gather(source, ilo, mlo, fallback), Gather(source, ihi, mhi, fallback));
    }

    /// <summary>
    /// Widening gather: float source into double lanes (float data indexed per-lane inside a
    /// double kernel). Portable per-lane loop — there is no hardware float→double gather.
    /// </summary>
    public static VDouble Gather(ReadOnlySpan<float> source, VLong indices, VMaskD mask, float fallback = 0f)
    {
        int w = VDouble.LaneCount;
        Span<long> idxs = stackalloc long[w];
        indices.V.CopyTo(idxs);
        Span<double> tmp = stackalloc double[w];
        for (int l = 0; l < w; l++)
            tmp[l] = mask.IsLaneActive(l) ? source[(int)idxs[l]] : fallback;

        return VDouble.Load(tmp);
    }

    /// <summary>
    /// Widening gather: int source into 64-bit integer lanes (int data indexed per-lane
    /// inside a long kernel). Portable per-lane loop.
    /// </summary>
    public static VLong Gather(ReadOnlySpan<int> source, VLong indices, VMaskD mask, int fallback = 0)
    {
        int w = VLong.LaneCount;
        Span<long> idxs = stackalloc long[w];
        indices.V.CopyTo(idxs);
        Span<long> tmp = stackalloc long[w];
        for (int l = 0; l < w; l++)
            tmp[l] = mask.IsLaneActive(l) ? source[(int)idxs[l]] : fallback;

        return VLong.Load(tmp);
    }

    /// <summary>
    /// Widening gather: byte source into int lanes, zero-extended (byte data indexed
    /// per-lane, ISPC's <c>uint8</c> gather on i32 targets). Portable per-lane loop —
    /// there is no 8-bit hardware gather.
    /// </summary>
    public static VInt Gather(ReadOnlySpan<byte> source, VInt indices, VMask mask, byte fallback = 0)
    {
        int w = VInt.LaneCount;
        Span<int> idxs = stackalloc int[w];
        indices.V.CopyTo(idxs);
        Span<int> tmp = stackalloc int[w];
        for (int l = 0; l < w; l++)
            tmp[l] = mask.IsLaneActive(l) ? source[idxs[l]] : fallback;

        return VInt.Load(tmp);
    }

    /// <summary>
    /// Widening gather: short source into int lanes, sign-extended. Portable per-lane loop.
    /// </summary>
    public static VInt Gather(ReadOnlySpan<short> source, VInt indices, VMask mask, short fallback = 0)
    {
        int w = VInt.LaneCount;
        Span<int> idxs = stackalloc int[w];
        indices.V.CopyTo(idxs);
        Span<int> tmp = stackalloc int[w];
        for (int l = 0; l < w; l++)
            tmp[l] = mask.IsLaneActive(l) ? source[idxs[l]] : fallback;

        return VInt.Load(tmp);
    }

    /// <summary>
    /// Widening gather: byte source into 64-bit integer lanes, zero-extended (byte data
    /// indexed per-lane inside a double/long kernel). Portable per-lane loop.
    /// </summary>
    public static VLong Gather(ReadOnlySpan<byte> source, VLong indices, VMaskD mask, byte fallback = 0)
    {
        int w = VLong.LaneCount;
        Span<long> idxs = stackalloc long[w];
        indices.V.CopyTo(idxs);
        Span<long> tmp = stackalloc long[w];
        for (int l = 0; l < w; l++)
            tmp[l] = mask.IsLaneActive(l) ? source[(int)idxs[l]] : fallback;

        return VLong.Load(tmp);
    }

    /// <summary>
    /// Widening gather: short source into 64-bit integer lanes, sign-extended. Portable per-lane loop.
    /// </summary>
    public static VLong Gather(ReadOnlySpan<short> source, VLong indices, VMaskD mask, short fallback = 0)
    {
        int w = VLong.LaneCount;
        Span<long> idxs = stackalloc long[w];
        indices.V.CopyTo(idxs);
        Span<long> tmp = stackalloc long[w];
        for (int l = 0; l < w; l++)
            tmp[l] = mask.IsLaneActive(l) ? source[(int)idxs[l]] : fallback;

        return VLong.Load(tmp);
    }

    /// <summary>
    /// Truncating scatter into a byte buffer: destination[indices[lane]] = (byte)values[lane]
    /// for active lanes (higher lane wins on duplicate indices, matching ISPC). Same
    /// movemask-driven active-lane path as the other scatters.
    /// </summary>
    public static void Scatter(Span<byte> destination, VInt indices, VInt values, VMask mask)
    {
        // Ascending lane order, so on a duplicate index the higher lane writes last and wins,
        // matching ISPC. ToBitmask always yields an answer, so there is no per-lane fallback.
        for (ulong bits = mask.ToBitmask(); bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            destination[indices.V[l]] = (byte)values.V[l];
        }
    }

    /// <summary>
    /// Truncating scatter into a short buffer; same semantics as the byte overload.
    /// </summary>
    public static void Scatter(Span<short> destination, VInt indices, VInt values, VMask mask)
    {
        // Ascending lane order, so on a duplicate index the higher lane writes last and wins,
        // matching ISPC. ToBitmask always yields an answer, so there is no per-lane fallback.
        for (ulong bits = mask.ToBitmask(); bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            destination[indices.V[l]] = (short)values.V[l];
        }
    }

    /// <summary>
    /// Scatter for long lanes; same semantics and hardware mask path as the double overload.
    /// </summary>
    public static void Scatter(Span<long> destination, VLong indices, VLong values, VMaskD mask)
    {
        // Ascending lane order, so on a duplicate index the higher lane writes last and wins,
        // matching ISPC. ToBitmask always yields an answer, so there is no per-lane fallback.
        for (ulong bits = mask.ToBitmask(); bits != 0; bits &= bits - 1)
        {
            int l = BitOperations.TrailingZeroCount(bits);
            destination[(int)indices.V[l]] = values.V[l];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Scatter(Span<long> destination, VLong indices, VLong values)
        => Scatter(destination, indices, values, VMaskD.All);

    // ------------------------------------------------------------------------------------------
    // Strided (Array-of-Structs field) access: element l of the gang is source[(base + l) * stride
    // + field]. This is how 'buf[i].field' on a [SpmdStruct] buffer is lowered. A gather works on
    // any shape, but the layout is perfectly regular, so where the platform has a de-interleave it
    // is used instead: NEON ld2/ld3/ld4 + st2/st3/st4 (.NET 9+), or Vector.Unzip/Zip for strides
    // 2 and 4 (.NET 11+, any ISA). Everything else keeps the gather/scatter.
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Strided load: <c>result[l] = source[(baseIndex + l) * stride + field]</c> for every lane —
    /// one field of a gang of consecutive structs viewed as a flat buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VFloat LoadStrided(ReadOnlySpan<float> source, int baseIndex, int stride, int field)
    {
        // These tests are spelled out here (not behind a helper) so the JIT folds them while
        // importing this method: stride is a constant at the call site, and a branch that is
        // already dead never charges its callee to the caller's inlining budget.
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VFloat.LaneCount * sizeof(float) == 16 && stride is 2 or 3 or 4)
            return new VFloat(LoadStridedFast(source, baseIndex, stride, field));
#endif
#if NET11_0_OR_GREATER
        if (stride is 2 or 4)
            return new VFloat(LoadStridedFast(source, baseIndex, stride, field));
#endif
        return Gather(source, ((VInt.ProgramIndex + baseIndex) * stride) + field);
    }

    /// <inheritdoc cref="LoadStrided(ReadOnlySpan{float}, int, int, int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt LoadStrided(ReadOnlySpan<int> source, int baseIndex, int stride, int field)
    {
        // These tests are spelled out here (not behind a helper) so the JIT folds them while
        // importing this method: stride is a constant at the call site, and a branch that is
        // already dead never charges its callee to the caller's inlining budget.
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VInt.LaneCount * sizeof(int) == 16 && stride is 2 or 3 or 4)
            return new VInt(LoadStridedFast(source, baseIndex, stride, field));
#endif
#if NET11_0_OR_GREATER
        if (stride is 2 or 4)
            return new VInt(LoadStridedFast(source, baseIndex, stride, field));
#endif
        return Gather(source, ((VInt.ProgramIndex + baseIndex) * stride) + field, VMask.All, 0);
    }

    /// <inheritdoc cref="LoadStrided(ReadOnlySpan{float}, int, int, int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VDouble LoadStrided(ReadOnlySpan<double> source, int baseIndex, int stride, int field)
    {
        // These tests are spelled out here (not behind a helper) so the JIT folds them while
        // importing this method: stride is a constant at the call site, and a branch that is
        // already dead never charges its callee to the caller's inlining budget.
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VDouble.LaneCount * sizeof(double) == 16 && stride is 2 or 3 or 4)
            return new VDouble(LoadStridedFast(source, baseIndex, stride, field));
#endif
#if NET11_0_OR_GREATER
        if (stride is 2 or 4)
            return new VDouble(LoadStridedFast(source, baseIndex, stride, field));
#endif
        return Gather(source, ((VLong.ProgramIndex + baseIndex) * stride) + field);
    }

    /// <inheritdoc cref="LoadStrided(ReadOnlySpan{float}, int, int, int)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VLong LoadStrided(ReadOnlySpan<long> source, int baseIndex, int stride, int field)
    {
        // These tests are spelled out here (not behind a helper) so the JIT folds them while
        // importing this method: stride is a constant at the call site, and a branch that is
        // already dead never charges its callee to the caller's inlining budget.
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VLong.LaneCount * sizeof(long) == 16 && stride is 2 or 3 or 4)
            return new VLong(LoadStridedFast(source, baseIndex, stride, field));
#endif
#if NET11_0_OR_GREATER
        if (stride is 2 or 4)
            return new VLong(LoadStridedFast(source, baseIndex, stride, field));
#endif
        return Gather(source, ((VLong.ProgramIndex + baseIndex) * stride) + field);
    }

    /// <summary>
    /// Strided store: <c>destination[(baseIndex + l) * stride + field] = values[l]</c> for active
    /// lanes — one field of a gang of consecutive structs. The other fields of those structs are
    /// left as they are.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStrided(Span<float> destination, int baseIndex, int stride, int field, VFloat values, VMask mask)
    {
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VFloat.LaneCount * sizeof(float) == 16 && stride is 2 or 3 or 4)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, Vector.AsVectorSingle(mask.Bits));
            return;
        }
#endif
#if NET11_0_OR_GREATER
        if (stride == 2)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, Vector.AsVectorSingle(mask.Bits));
            return;
        }
#endif
        Scatter(destination, ((VInt.ProgramIndex + baseIndex) * stride) + field, values, mask);
    }

    /// <inheritdoc cref="StoreStrided(Span{float}, int, int, int, VFloat, VMask)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStrided(Span<int> destination, int baseIndex, int stride, int field, VInt values, VMask mask)
    {
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VInt.LaneCount * sizeof(int) == 16 && stride is 2 or 3 or 4)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, mask.Bits);
            return;
        }
#endif
#if NET11_0_OR_GREATER
        if (stride == 2)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, mask.Bits);
            return;
        }
#endif
        Scatter(destination, ((VInt.ProgramIndex + baseIndex) * stride) + field, values, mask);
    }

    /// <inheritdoc cref="StoreStrided(Span{float}, int, int, int, VFloat, VMask)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStrided(Span<double> destination, int baseIndex, int stride, int field, VDouble values, VMaskD mask)
    {
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VDouble.LaneCount * sizeof(double) == 16 && stride is 2 or 3 or 4)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, Vector.AsVectorDouble(mask.Bits));
            return;
        }
#endif
#if NET11_0_OR_GREATER
        if (stride == 2)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, Vector.AsVectorDouble(mask.Bits));
            return;
        }
#endif
        Scatter(destination, ((VLong.ProgramIndex + baseIndex) * stride) + field, values, mask);
    }

    /// <inheritdoc cref="StoreStrided(Span{float}, int, int, int, VFloat, VMask)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreStrided(Span<long> destination, int baseIndex, int stride, int field, VLong values, VMaskD mask)
    {
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && VLong.LaneCount * sizeof(long) == 16 && stride is 2 or 3 or 4)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, mask.Bits);
            return;
        }
#endif
#if NET11_0_OR_GREATER
        if (stride == 2)
        {
            StoreStridedFast(destination, baseIndex, stride, field, values.V, mask.Bits);
            return;
        }
#endif
        Scatter(destination, ((VLong.ProgramIndex + baseIndex) * stride) + field, values, mask);
    }

    /// <summary>
    /// Range check for a de-interleaving access: the whole block of <c>stride * LaneCount</c>
    /// elements starting at <c>baseIndex * stride</c> must be inside the buffer. Returns the
    /// block's first element index.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int StridedStart<T>(int length, int baseIndex, int stride)
        where T : struct
    {
        ulong start = (ulong)(uint)baseIndex * (uint)stride;
        if (baseIndex < 0 || start + (ulong)(stride * Vector<T>.Count) > (uint)length)
            ThrowGatherIndexOutOfRange();
        return (int)start;
    }

    /// <summary>De-interleaving load; the caller has checked that this stride has one here.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector<T> LoadStridedFast<T>(ReadOnlySpan<T> source, int baseIndex, int stride, int field)
        where T : unmanaged
    {
        int start = StridedStart<T>(source.Length, baseIndex, stride);
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && Vector<T>.Count * sizeof(T) == 16)
        {
            fixed (T* p = source)
            {
                return sizeof(T) == 4
                    ? Vector128.AsVector(LoadFieldNeon((float*)(p + start), stride, field)).As<float, T>()
                    : Vector128.AsVector(LoadFieldNeon((double*)(p + start), stride, field)).As<double, T>();
            }
        }
#endif
#if NET11_0_OR_GREATER
        int w = Vector<T>.Count;
        if (stride == 2)
        {
            var lo = LaneIO.LoadUnchecked(source, start);
            var hi = LaneIO.LoadUnchecked(source, start + w);
            return field == 0 ? Vector.UnzipEven(lo, hi) : Vector.UnzipOdd(lo, hi);
        }

        // Stride 4: one unzip separates fields {0,2} from {1,3}, a second separates the pair.
        var v0 = LaneIO.LoadUnchecked(source, start);
        var v1 = LaneIO.LoadUnchecked(source, start + w);
        var v2 = LaneIO.LoadUnchecked(source, start + (2 * w));
        var v3 = LaneIO.LoadUnchecked(source, start + (3 * w));
        Vector<T> a, b;
        if ((field & 1) == 0)
        {
            a = Vector.UnzipEven(v0, v1);
            b = Vector.UnzipEven(v2, v3);
        }
        else
        {
            a = Vector.UnzipOdd(v0, v1);
            b = Vector.UnzipOdd(v2, v3);
        }

        return (field & 2) == 0 ? Vector.UnzipEven(a, b) : Vector.UnzipOdd(a, b);
#else
        return default;   // unreachable: callers only come here when a path above exists
#endif
    }

    /// <summary>De-interleaving store; the caller has checked that this stride has one here.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreStridedFast<T>(Span<T> destination, int baseIndex, int stride, int field, Vector<T> values, Vector<T> mask)
        where T : unmanaged
    {
        int start = StridedStart<T>(destination.Length, baseIndex, stride);
#if NET9_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && Vector<T>.Count * sizeof(T) == 16)
        {
            fixed (T* p = destination)
            {
                if (sizeof(T) == 4)
                    StoreFieldNeon((float*)(p + start), stride, field, Vector128.AsVector128(values).As<T, float>(), Vector128.AsVector128(mask).As<T, float>());
                else
                    StoreFieldNeon((double*)(p + start), stride, field, Vector128.AsVector128(values).As<T, double>(), Vector128.AsVector128(mask).As<T, double>());
            }

            return;
        }
#endif
#if NET11_0_OR_GREATER
        // Stride 2: split the block into its two fields, blend the new values into one, zip back.
        int w = Vector<T>.Count;
        var (even, odd) = Vector.Unzip(LaneIO.LoadUnchecked<T>(destination, start), LaneIO.LoadUnchecked<T>(destination, start + w));
        if (field == 0)
            even = Vector.ConditionalSelect(mask, values, even);
        else
            odd = Vector.ConditionalSelect(mask, values, odd);
        var (lo, hi) = Vector.Zip(even, odd);
        LaneIO.StoreUnchecked(lo, destination, start);
        LaneIO.StoreUnchecked(hi, destination, start + w);
#endif
    }

#if NET9_0_OR_GREATER
    // NEON de-interleaving loads/stores. They only look at the element size, so int and long
    // buffers go through the float and double entry points.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector128<float> LoadFieldNeon(float* p, int stride, int field)
    {
        if (stride == 2)
        {
            var (a, b) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load2xVector128AndUnzip(p);
            return field == 0 ? a : b;
        }

        if (stride == 3)
        {
            var (a, b, c) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load3xVector128AndUnzip(p);
            return field == 0 ? a : field == 1 ? b : c;
        }

        var (q0, q1, q2, q3) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load4xVector128AndUnzip(p);
        return field == 0 ? q0 : field == 1 ? q1 : field == 2 ? q2 : q3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector128<double> LoadFieldNeon(double* p, int stride, int field)
    {
        if (stride == 2)
        {
            var (a, b) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load2xVector128AndUnzip(p);
            return field == 0 ? a : b;
        }

        if (stride == 3)
        {
            var (a, b, c) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load3xVector128AndUnzip(p);
            return field == 0 ? a : field == 1 ? b : c;
        }

        var (q0, q1, q2, q3) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load4xVector128AndUnzip(p);
        return field == 0 ? q0 : field == 1 ? q1 : field == 2 ? q2 : q3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreFieldNeon(float* p, int stride, int field, Vector128<float> values, Vector128<float> mask)
    {
        if (stride == 2)
        {
            var (a, b) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load2xVector128AndUnzip(p);
            if (field == 0)
                a = Vector128.ConditionalSelect(mask, values, a);
            else
                b = Vector128.ConditionalSelect(mask, values, b);
            System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.StoreVectorAndZip(p, (a, b));
        }
        else if (stride == 3)
        {
            var (a, b, c) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load3xVector128AndUnzip(p);
            if (field == 0)
                a = Vector128.ConditionalSelect(mask, values, a);
            else if (field == 1)
                b = Vector128.ConditionalSelect(mask, values, b);
            else
                c = Vector128.ConditionalSelect(mask, values, c);
            System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.StoreVectorAndZip(p, (a, b, c));
        }
        else
        {
            var (q0, q1, q2, q3) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load4xVector128AndUnzip(p);
            if (field == 0)
                q0 = Vector128.ConditionalSelect(mask, values, q0);
            else if (field == 1)
                q1 = Vector128.ConditionalSelect(mask, values, q1);
            else if (field == 2)
                q2 = Vector128.ConditionalSelect(mask, values, q2);
            else
                q3 = Vector128.ConditionalSelect(mask, values, q3);
            System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.StoreVectorAndZip(p, (q0, q1, q2, q3));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreFieldNeon(double* p, int stride, int field, Vector128<double> values, Vector128<double> mask)
    {
        if (stride == 2)
        {
            var (a, b) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load2xVector128AndUnzip(p);
            if (field == 0)
                a = Vector128.ConditionalSelect(mask, values, a);
            else
                b = Vector128.ConditionalSelect(mask, values, b);
            System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.StoreVectorAndZip(p, (a, b));
        }
        else if (stride == 3)
        {
            var (a, b, c) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load3xVector128AndUnzip(p);
            if (field == 0)
                a = Vector128.ConditionalSelect(mask, values, a);
            else if (field == 1)
                b = Vector128.ConditionalSelect(mask, values, b);
            else
                c = Vector128.ConditionalSelect(mask, values, c);
            System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.StoreVectorAndZip(p, (a, b, c));
        }
        else
        {
            var (q0, q1, q2, q3) = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.Load4xVector128AndUnzip(p);
            if (field == 0)
                q0 = Vector128.ConditionalSelect(mask, values, q0);
            else if (field == 1)
                q1 = Vector128.ConditionalSelect(mask, values, q1);
            else if (field == 2)
                q2 = Vector128.ConditionalSelect(mask, values, q2);
            else
                q3 = Vector128.ConditionalSelect(mask, values, q3);
            System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.StoreVectorAndZip(p, (q0, q1, q2, q3));
        }
    }
#endif

    /// <summary>
    /// Turns a strided access into a contiguous one. Classic use: matmul's b[k, col] is a
    /// gather (column of a row-major matrix); transpose b once, then bT[col, k] is a
    /// contiguous load. The transpose is O(rows*cols) and amortizes when the consumer reuses
    /// the data (a matmul reads b once per output row). dst must be [cols, rows].
    /// </summary>
    private const int TransposeBlock = 32;

    public static void Transpose(float[,] src, float[,] dst)
    {
        int rows = src.GetLength(0), cols = src.GetLength(1);
        for (int i0 = 0; i0 < rows; i0 += TransposeBlock)
        {
            for (int j0 = 0; j0 < cols; j0 += TransposeBlock)
            {
                int iMax = Math.Min(i0 + TransposeBlock, rows);
                int jMax = Math.Min(j0 + TransposeBlock, cols);
                for (int i = i0; i < iMax; i++)
                {
                    for (int j = j0; j < jMax; j++)
                        dst[j, i] = src[i, j];
                }
            }
        }
    }

    public static void Transpose(int[,] src, int[,] dst)
    {
        int rows = src.GetLength(0), cols = src.GetLength(1);
        for (int i0 = 0; i0 < rows; i0 += TransposeBlock)
        {
            for (int j0 = 0; j0 < cols; j0 += TransposeBlock)
            {
                int iMax = Math.Min(i0 + TransposeBlock, rows);
                int jMax = Math.Min(j0 + TransposeBlock, cols);
                for (int i = i0; i < iMax; i++)
                {
                    for (int j = j0; j < jMax; j++)
                        dst[j, i] = src[i, j];
                }
            }
        }
    }

    public static void Transpose(double[,] src, double[,] dst)
    {
        int rows = src.GetLength(0), cols = src.GetLength(1);
        for (int i0 = 0; i0 < rows; i0 += TransposeBlock)
        {
            for (int j0 = 0; j0 < cols; j0 += TransposeBlock)
            {
                int iMax = Math.Min(i0 + TransposeBlock, rows);
                int jMax = Math.Min(j0 + TransposeBlock, cols);
                for (int i = i0; i < iMax; i++)
                {
                    for (int j = j0; j < jMax; j++)
                        dst[j, i] = src[i, j];
                }
            }
        }
    }

    public static void Transpose(long[,] src, long[,] dst)
    {
        int rows = src.GetLength(0), cols = src.GetLength(1);
        for (int i0 = 0; i0 < rows; i0 += TransposeBlock)
        {
            for (int j0 = 0; j0 < cols; j0 += TransposeBlock)
            {
                int iMax = Math.Min(i0 + TransposeBlock, rows);
                int jMax = Math.Min(j0 + TransposeBlock, cols);
                for (int i = i0; i < iMax; i++)
                {
                    for (int j = j0; j < jMax; j++)
                        dst[j, i] = src[i, j];
                }
            }
        }
    }

    /// <summary>
    /// Allocate and return the transpose of a 2-D array.
    /// </summary>
    public static float[,] Transposed(float[,] src)
    {
        float[,] dst = new float[src.GetLength(1), src.GetLength(0)];
        Transpose(src, dst);
        return dst;
    }
}
