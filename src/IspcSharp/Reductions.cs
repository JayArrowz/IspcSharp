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

    public static int Add(VInt v)
    {
        int sum = 0;
        for (int l = 0; l < VInt.LaneCount; l++)
            sum += v.V[l];
        return sum;
    }

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
    public static long Add(VLong v)
    {
        long sum = 0;
        for (int l = 0; l < VLong.LaneCount; l++)
            sum += v.V[l];
        return sum;
    }

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
    /// 64-bit lane in a single instruction; portable loop otherwise.
    /// </summary>
    public static int Add(VByte v)
    {
#if NET8_0_OR_GREATER
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
    public static int Add(VShort v)
    {
#if NET8_0_OR_GREATER
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
    /// Gather: result[lane] = source[indices[lane]] for active lanes; <paramref name="fallback"/> otherwise.
    /// Uses AVX-512 (16-wide) or AVX2 (8-wide) hardware gather on net8.0+ when available (indices of
    /// active lanes must still be in bounds).
    /// </summary>
    public static unsafe VFloat Gather(ReadOnlySpan<float> source, VInt indices, VMask mask, float fallback = 0f)
    {
#if NET8_0_OR_GREATER
        // AVX-512 gangs (16 float lanes): .NET exposes no 512-bit gather intrinsic, so run the
        // AVX2 256-bit hardware gather on each 8-lane half. Without this a 16-wide gang would
        // drop to the portable per-lane path, the Linux/AVX-512 gather regression.
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && VFloat.LaneCount == 16)
        {
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
        if (VFloat.LaneCount == 4 && TryGetActiveBits(mask, out uint gbits) && gbits == 0xF)
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
#if NET8_0_OR_GREATER
        if (TryGetActiveBits(mask, out uint bits))
        {
            while (bits != 0)
            {
                int l = BitOperations.TrailingZeroCount(bits);
                bits &= bits - 1;
                destination[indices.V[l]] = values.V[l];
            }

            return;
        }
#endif
        for (int l = 0; l < VFloat.LaneCount; l++)
        {
            if (mask.IsLaneActive(l))
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
#if NET8_0_OR_GREATER
        if (TryGetActiveBits(mask, out uint bits))
        {
            while (bits != 0)
            {
                int l = BitOperations.TrailingZeroCount(bits);
                bits &= bits - 1;
                destination[indices.V[l]] = values.V[l];
            }

            return;
        }
#endif
        for (int l = 0; l < VInt.LaneCount; l++)
        {
            if (mask.IsLaneActive(l))
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
#if NET8_0_OR_GREATER
        if (TryGetActiveBitsD(mask, out uint bits))
        {
            while (bits != 0)
            {
                int l = BitOperations.TrailingZeroCount(bits);
                bits &= bits - 1;
                destination[(int)indices.V[l]] = values.V[l];
            }

            return;
        }
#endif
        for (int l = 0; l < VDouble.LaneCount; l++)
        {
            if (mask.IsLaneActive(l))
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
    /// Scatter for long lanes; same semantics and hardware mask path as the double overload.
    /// </summary>
    public static void Scatter(Span<long> destination, VLong indices, VLong values, VMaskD mask)
    {
#if NET8_0_OR_GREATER
        if (TryGetActiveBitsD(mask, out uint bits))
        {
            while (bits != 0)
            {
                int l = BitOperations.TrailingZeroCount(bits);
                bits &= bits - 1;
                destination[(int)indices.V[l]] = values.V[l];
            }

            return;
        }
#endif
        for (int l = 0; l < VLong.LaneCount; l++)
        {
            if (mask.IsLaneActive(l))
                destination[(int)indices.V[l]] = values.V[l];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Scatter(Span<long> destination, VLong indices, VLong values)
        => Scatter(destination, indices, values, VMaskD.All);

#if NET8_0_OR_GREATER
    /// <summary>
    /// Extract the active-lane set as a bitmask with one hardware instruction:
    /// AVX-512 mask-register compare (16 lanes), AVX vmovmskps (8 lanes), or
    /// SSE movmskps / NEON narrowing shift (4 lanes). Returns false when the lane
    /// width has no fast path (callers fall back to the portable per-lane loop).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryGetActiveBits(VMask mask, out uint bits)
    {
        if (VInt.LaneCount == 16 && System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
        {
            bits = (uint)Vector512.ExtractMostSignificantBits(
                Vector512.AsVector512(mask.Bits));
            return true;
        }

        if (VInt.LaneCount == 8 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
        {
            bits = Vector256.ExtractMostSignificantBits(
                Vector256.AsVector256(mask.Bits));
            return true;
        }

        if (VInt.LaneCount == 4 &&
            (System.Runtime.Intrinsics.X86.Sse2.IsSupported || System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported))
        {
            bits = Vector128.ExtractMostSignificantBits(
                Vector128.AsVector128(mask.Bits));
            return true;
        }

        bits = 0;
        return false;
    }

    /// <summary>
    /// 64-bit-lane variant of <see cref="TryGetActiveBits"/> for VMaskD masks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryGetActiveBitsD(VMaskD mask, out uint bits)
    {
        if (VDouble.LaneCount == 8 && System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
        {
            bits = (uint)Vector512.ExtractMostSignificantBits(
                Vector512.AsVector512(mask.Bits));
            return true;
        }

        if (VDouble.LaneCount == 4 && System.Runtime.Intrinsics.X86.Avx.IsSupported)
        {
            bits = Vector256.AsVector256(mask.Bits).ExtractMostSignificantBits(
);
            return true;
        }

        if (VDouble.LaneCount == 2 &&
            (System.Runtime.Intrinsics.X86.Sse2.IsSupported || System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported))
        {
            bits = Vector128.AsVector128(mask.Bits).ExtractMostSignificantBits(
);
            return true;
        }

        bits = 0;
        return false;
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
