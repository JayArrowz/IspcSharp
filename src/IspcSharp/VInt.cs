using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace IspcSharp;

/// <summary>
/// A "varying int", one 32-bit integer per SIMD lane. The ISPC equivalent of <c>varying int</c>.
/// </summary>
public readonly struct VInt : IEquatable<VInt>
{
    public readonly Vector<int> V;

    public static int LaneCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector<int>.Count;
    }

    public static VInt Zero => new(Vector<int>.Zero);
    public static VInt One => new(Vector<int>.One);

    private static readonly Vector<int> LaneIndex = CreateLaneIndex();
    private static Vector<int> CreateLaneIndex()
    {
        Span<int> tmp = stackalloc int[Vector<int>.Count];
        for (int i = 0; i < tmp.Length; i++)
            tmp[i] = i;
        return new Vector<int>(tmp);
    }

    /// <summary>
    /// {0, 1, 2, ... LaneCount-1}, ISPC's <c>programIndex</c>.
    /// </summary>
    public static VInt ProgramIndex => new(LaneIndex);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VInt(Vector<int> v) => V = v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VInt(int uniform) => V = new Vector<int>(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Load(ReadOnlySpan<int> source, int offset = 0)
        => new(new Vector<int>(source[offset..]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Store(Span<int> destination, int offset = 0) => V.CopyTo(destination[offset..]);

    public void StoreMasked(Span<int> destination, int offset, VMask mask)
    {
        for (int l = 0; l < LaneCount; l++)
        {
            int idx = offset + l;
            if (mask.IsLaneActive(l) && idx < destination.Length)
                destination[idx] = V[l];
        }
    }

    public int GetLane(int lane) => V[lane];

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator +(VInt a, VInt b) => new(a.V + b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator -(VInt a, VInt b) => new(a.V - b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator *(VInt a, VInt b) => new(a.V * b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator +(VInt a, int b) => new(a.V + new Vector<int>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator -(VInt a, int b) => new(a.V - new Vector<int>(b));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator *(VInt a, int b) => new(a.V * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator &(VInt a, VInt b) => new(a.V & b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator |(VInt a, VInt b) => new(a.V | b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator ^(VInt a, VInt b) => new(a.V ^ b.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator ~(VInt a) => new(~a.V);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VInt operator -(VInt a) => new(-a.V);

    /// <summary>
    /// Per-lane integer division (no SIMD integer divide exists, scalar loop).
    /// Throws <see cref="DivideByZeroException"/> if ANY lane's divisor is zero, in masked
    /// SPMD code, select a nonzero divisor into inactive lanes first (the [Spmd] generator
    /// does this automatically).
    /// </summary>
    public static VInt operator /(VInt a, VInt b) => Divide(a, b);

    /// <summary>
    /// Per-lane integer remainder (scalar loop; see <see cref="op_Division"/> about zero divisors).
    /// </summary>
    public static VInt operator %(VInt a, VInt b) => Remainder(a, b);

    public static VInt Divide(VInt a, VInt b)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] / b.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    public static VInt Remainder(VInt a, VInt b)
    {
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] % b.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator VInt(int uniform) => new(uniform);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt operator <<(VInt a, int count)
    {
#if NET8_0_OR_GREATER
        return new VInt(Vector.ShiftLeft(a.V, count));
#else
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = a.V[l] << count;
        return new VInt(new Vector<int>(tmp));
#endif
    }

    /// <summary>
    /// Arithmetic (sign-extending) right shift.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt operator >>(VInt a, int count)
    {
#if NET8_0_OR_GREATER
        return new VInt(Vector.ShiftRightArithmetic(a.V, count));
#else
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = a.V[l] >> count;
        return new VInt(new Vector<int>(tmp));
#endif
    }

    /// <summary>
    /// Logical (zero-filling) right shift.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt ShiftRightLogical(VInt a, int count)
    {
#if NET8_0_OR_GREATER
        return new VInt(Vector.AsVectorInt32(Vector.ShiftRightLogical(Vector.AsVectorUInt32(a.V), count)));
#else
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++) tmp[l] = (int)((uint)a.V[l] >> count);
        return new VInt(new Vector<int>(tmp));
#endif
    }

    /// <summary>
    /// Per-lane left shift: result[l] = a[l] &lt;&lt; (counts[l] &amp; 31).
    /// Hardware <c>vpsllvd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
    public static VInt ShiftLeftVariable(VInt a, VInt counts)
    {
#if NET8_0_OR_GREATER
        var cnt = (counts & new VInt(31)).V;
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ShiftLeftLogicalVariable(
                Vector512.AsVector512(a.V).AsUInt32(), Vector512.AsVector512(cnt).AsUInt32()).AsInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftLeftLogicalVariable(
                Vector256.AsVector256(a.V), Vector256.AsVector256(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 4)
        {
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftLeftLogicalVariable(
                Vector128.AsVector128(a.V), Vector128.AsVector128(cnt).AsUInt32())));
        }
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] << counts.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Per-lane arithmetic right shift: result[l] = a[l] &gt;&gt; (counts[l] &amp; 31).
    /// Hardware <c>vpsravd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
    public static VInt ShiftRightArithmeticVariable(VInt a, VInt counts)
    {
#if NET8_0_OR_GREATER
        var cnt = (counts & new VInt(31)).V;
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ShiftRightArithmeticVariable(
                Vector512.AsVector512(a.V), Vector512.AsVector512(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightArithmeticVariable(
                Vector256.AsVector256(a.V), Vector256.AsVector256(cnt).AsUInt32())));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 4)
        {
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightArithmeticVariable(
                Vector128.AsVector128(a.V), Vector128.AsVector128(cnt).AsUInt32())));
        }
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = a.V[l] >> counts.V[l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Per-lane logical right shift: result[l] = (uint)a[l] &gt;&gt; (counts[l] &amp; 31).
    /// Hardware <c>vpsrlvd</c> at 128/256/512-bit widths (AVX2 / AVX-512F); portable loop otherwise.
    /// </summary>
    public static VInt ShiftRightLogicalVariable(VInt a, VInt counts)
    {
#if NET8_0_OR_GREATER
        var cnt = (counts & new VInt(31)).V;
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ShiftRightLogicalVariable(
                Vector512.AsVector512(a.V).AsUInt32(), Vector512.AsVector512(cnt).AsUInt32()).AsInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogicalVariable(
                Vector256.AsVector256(a.V).AsUInt32(), Vector256.AsVector256(cnt).AsUInt32()).AsInt32()));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 4)
        {
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogicalVariable(
                Vector128.AsVector128(a.V).AsUInt32(), Vector128.AsVector128(cnt).AsUInt32()).AsInt32()));
        }
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = (int)((uint)a.V[l] >> counts.V[l]);
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Widening load from a byte buffer: one int lane per byte, zero-extended (0..255).
    /// Reads exactly LaneCount bytes. Hardware <c>vpmovzxbd</c> (SSE4.1/AVX2/AVX-512F) or
    /// NEON <c>ushll</c>; portable loop otherwise. This is the ISPC-style bridge that runs
    /// byte data at full int-gang width: compute in int lanes (C#'s own promotion rules),
    /// touch memory one byte per lane.
    /// </summary>
    public static VInt LoadZeroExtend(ReadOnlySpan<byte> source, int offset = 0)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            var b = Vector128.Create(source.Slice(offset, 16));
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(b)));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            ulong bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(source[offset..]);
            var b = Vector128.CreateScalar(bits).AsByte();
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32(b)));
        }

        if (System.Runtime.Intrinsics.X86.Sse41.IsSupported && LaneCount == 4)
        {
            uint bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(source[offset..]);
            var b = Vector128.CreateScalar(bits).AsByte();
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32(b)));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            uint bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(source[offset..]);
            var b = Vector64.CreateScalar(bits).AsByte();
            var w16 = System.Runtime.Intrinsics.Arm.AdvSimd.ZeroExtendWideningLower(b);
            var w32 = System.Runtime.Intrinsics.Arm.AdvSimd.ZeroExtendWideningLower(w16.GetLower());
            return new VInt(Vector128.AsVector(w32.AsInt32()));
        }
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = source[offset + l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Widening load from a short buffer: one int lane per short, sign-extended.
    /// Reads exactly LaneCount shorts. Hardware <c>vpmovsxwd</c> (SSE4.1/AVX2/AVX-512F) or
    /// NEON <c>sshll</c>; portable loop otherwise. See <see cref="LoadZeroExtend"/>.
    /// </summary>
    public static VInt LoadSignExtend(ReadOnlySpan<short> source, int offset = 0)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            var s = Vector256.Create(source.Slice(offset, 16));
            return new VInt(Vector512.AsVector(System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector512Int32(s)));
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            var s = Vector128.Create(source.Slice(offset, 8));
            return new VInt(Vector256.AsVector(System.Runtime.Intrinsics.X86.Avx2.ConvertToVector256Int32(s)));
        }

        if (System.Runtime.Intrinsics.X86.Sse41.IsSupported && LaneCount == 4)
        {
            ulong bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(source.Slice(offset, 4)));
            var s = Vector128.CreateScalar(bits).AsInt16();
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.X86.Sse41.ConvertToVector128Int32(s)));
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            ulong bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(source.Slice(offset, 4)));
            var s = Vector64.CreateScalar(bits).AsInt16();
            return new VInt(Vector128.AsVector(System.Runtime.Intrinsics.Arm.AdvSimd.SignExtendWideningLower(s)));
        }
#endif
        Span<int> tmp = stackalloc int[LaneCount];
        for (int l = 0; l < LaneCount; l++)
            tmp[l] = source[offset + l];
        return new VInt(new Vector<int>(tmp));
    }

    /// <summary>
    /// Truncating narrowing store to a byte buffer: each int lane's low 8 bits, exactly
    /// C#'s <c>(byte)</c> cast. Writes exactly LaneCount bytes. Hardware <c>vpmovdb</c>
    /// (AVX-512F) or a mask+pack sequence (SSE2/AVX2) or NEON <c>xtn</c>; portable otherwise.
    /// </summary>
    public void StoreNarrow(Span<byte> destination, int offset = 0)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            var b = System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector128Byte(Vector512.AsVector512(V));
            b.CopyTo(destination.Slice(offset, 16));
            return;
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 8)
        {
            // Mask to 0..255 so the saturating packs are exact truncation.
            var v = Vector256.AsVector256(V) & Vector256.Create(0xFF);
            var p16 = System.Runtime.Intrinsics.X86.Sse2.PackSignedSaturate(v.GetLower(), v.GetUpper());
            var p8 = System.Runtime.Intrinsics.X86.Sse2.PackUnsignedSaturate(p16, p16);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(destination[offset..], p8.AsUInt64().ToScalar());
            return;
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 4)
        {
            var v = Vector128.AsVector128(V) & Vector128.Create(0xFF);
            var p16 = System.Runtime.Intrinsics.X86.Sse2.PackSignedSaturate(v, v);
            var p8 = System.Runtime.Intrinsics.X86.Sse2.PackUnsignedSaturate(p16, p16);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(destination[offset..], p8.AsUInt32().ToScalar());
            return;
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            var n16 = System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingLower(Vector128.AsVector128(V));
            var n8 = System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingLower(
                Vector128.Create(n16, Vector64<short>.Zero));
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(destination[offset..], n8.AsUInt32().ToScalar());
            return;
        }
#endif
        for (int l = 0; l < LaneCount; l++)
            destination[offset + l] = (byte)V[l];
    }

    /// <summary>
    /// Truncating narrowing store to a short buffer: each int lane's low 16 bits, exactly
    /// C#'s <c>(short)</c> cast. Writes exactly LaneCount shorts. Hardware <c>vpmovdw</c>
    /// (AVX-512F) or a sign-fit+pack sequence (SSE2/AVX2) or NEON <c>xtn</c>; portable otherwise.
    /// </summary>
    public void StoreNarrow(Span<short> destination, int offset = 0)
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 16)
        {
            var s = System.Runtime.Intrinsics.X86.Avx512F.ConvertToVector256Int16(Vector512.AsVector512(V));
            s.CopyTo(destination.Slice(offset, 16));
            return;
        }

        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && LaneCount == 8)
        {
            // Sign-fit each lane into short range so the saturating pack is exact truncation.
            var v = Vector256.AsVector256(V);
            var t = System.Runtime.Intrinsics.X86.Avx2.ShiftRightArithmetic(
                System.Runtime.Intrinsics.X86.Avx2.ShiftLeftLogical(v, 16), 16);
            var p = System.Runtime.Intrinsics.X86.Sse2.PackSignedSaturate(t.GetLower(), t.GetUpper());
            p.CopyTo(destination.Slice(offset, 8));
            return;
        }

        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 4)
        {
            var v = Vector128.AsVector128(V);
            var t = System.Runtime.Intrinsics.X86.Sse2.ShiftRightArithmetic(
                System.Runtime.Intrinsics.X86.Sse2.ShiftLeftLogical(v, 16), 16);
            var p = System.Runtime.Intrinsics.X86.Sse2.PackSignedSaturate(t, t);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(destination.Slice(offset, 4)),
                p.AsUInt64().ToScalar());
            return;
        }

        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported && LaneCount == 4)
        {
            var n = System.Runtime.Intrinsics.Arm.AdvSimd.ExtractNarrowingLower(Vector128.AsVector128(V));
            System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(destination.Slice(offset, 4)),
                n.AsUInt64().ToScalar());
            return;
        }
#endif
        for (int l = 0; l < LaneCount; l++)
            destination[offset + l] = (short)V[l];
    }

    /// <summary>
    /// Widen a full int gang into its two long-gang halves (cross-gang-width bridge).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (VLong Lower, VLong Upper) Widen(VInt v)
    {
        Vector.Widen(v.V, out var lo, out var hi);
        return (new VLong(lo), new VLong(hi));
    }

    /// <summary>
    /// Narrow two long-gang halves back into one int gang (truncating).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Narrow(VLong lower, VLong upper)
        => new(Vector.Narrow(lower.V, upper.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator <(VInt a, VInt b) => new(Vector.LessThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator >(VInt a, VInt b) => new(Vector.GreaterThan(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator <=(VInt a, VInt b) => new(Vector.LessThanOrEqual(a.V, b.V));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMask operator >=(VInt a, VInt b) => new(Vector.GreaterThanOrEqual(a.V, b.V));
    public static VMask Eq(VInt a, VInt b) => new(Vector.Equals(a.V, b.V));
    public static VMask Neq(VInt a, VInt b) => new(~Vector.Equals(a.V, b.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt Select(VMask mask, VInt ifTrue, VInt ifFalse)
        => new(Vector.ConditionalSelect(mask.Bits, ifTrue.V, ifFalse.V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VInt Blend(VMask mask, VInt newValue) => Select(mask, newValue, this);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VFloat ToFloat() => new(Vector.ConvertToSingle(V));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static VInt FromFloatTruncate(VFloat f) => new(Vector.ConvertToInt32(f.V));

    /// <summary>
    /// Bitcast: reinterpret the 32 bits of each lane as a float (no conversion).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VFloat AsFloat() => new(Vector.AsVectorSingle(V));

    public bool Equals(VInt other) => V.Equals(other.V);
    public override bool Equals(object? obj) => obj is VInt i && Equals(i);
    public override int GetHashCode() => V.GetHashCode();
    public override string ToString() => V.ToString();

    public static bool operator ==(VInt left, VInt right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(VInt left, VInt right)
    {
        return !(left == right);
    }
}
