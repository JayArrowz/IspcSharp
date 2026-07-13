using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp
{
    /// <summary>64-bit-lane execution mask for <see cref="VDouble"/> (lanes are long: -1 active, 0 inactive).</summary>
    public readonly struct VMaskD : IEquatable<VMaskD>
    {
        public readonly Vector<long> Bits;

        public static int LaneCount => Vector<long>.Count;
        public static VMaskD All => new VMaskD(Vector.Equals(Vector<long>.Zero, Vector<long>.Zero));
        public static VMaskD None => new VMaskD(Vector<long>.Zero);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VMaskD(Vector<long> bits) => Bits = bits;

        public static VMaskD FirstN(int count)
        {
            Span<long> tmp = stackalloc long[LaneCount];
            for (int i = 0; i < LaneCount; i++) tmp[i] = i < count ? -1L : 0L;
            return new VMaskD(new Vector<long>(tmp));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator &(VMaskD a, VMaskD b) => new VMaskD(a.Bits & b.Bits);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator |(VMaskD a, VMaskD b) => new VMaskD(a.Bits | b.Bits);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator !(VMaskD a) => new VMaskD(~a.Bits);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator ~(VMaskD a) => new VMaskD(~a.Bits);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMaskD AndNot(VMaskD a, VMaskD b) => new VMaskD(Vector.AndNot(a.Bits, b.Bits));

        /// <summary>Widen a float-gang mask into its two double-gang halves (cross-gang-width bridge).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static (VMaskD Lower, VMaskD Upper) Widen(VMask mask)
        {
            Vector.Widen(mask.Bits, out Vector<long> lo, out Vector<long> hi);
            return (new VMaskD(lo), new VMaskD(hi));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Any() => !Vector.EqualsAll(Bits, Vector<long>.Zero);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AllActive() => !Vector.EqualsAny(Bits, Vector<long>.Zero);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool NoneActive() => Vector.EqualsAll(Bits, Vector<long>.Zero);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsLaneActive(int lane) => Bits[lane] != 0;

        public bool Equals(VMaskD other) => Bits.Equals(other.Bits);
        public override bool Equals(object? obj) => obj is VMaskD m && Equals(m);
        public override int GetHashCode() => Bits.GetHashCode();
    }

    /// <summary>
    /// A "varying double", one double per SIMD lane (half the lane count of <see cref="VFloat"/>:
    /// AVX2 = 4 doubles, AVX-512 = 8). Use for kernels where float precision drifts (long
    /// accumulations, geometric predicates). Sqrt/Abs/Min/Max are hardware ops; transcendentals
    /// (Sin/Cos/Exp/Log/Pow/Tanh/...) live in <see cref="VectorMath"/> as VDouble overloads.
    /// </summary>
    public readonly struct VDouble : IEquatable<VDouble>
    {
        public readonly Vector<double> V;

        public static int LaneCount => Vector<double>.Count;
        public static VDouble Zero => new VDouble(Vector<double>.Zero);
        public static VDouble One => new VDouble(Vector<double>.One);

        private static readonly Vector<double> s_laneIndex = CreateLaneIndex();
        private static Vector<double> CreateLaneIndex()
        {
            Span<double> tmp = stackalloc double[Vector<double>.Count];
            for (int i = 0; i < tmp.Length; i++) tmp[i] = i;
            return new Vector<double>(tmp);
        }

        /// <summary>{0, 1, 2, ... LaneCount-1}, ISPC's <c>programIndex</c> for double gangs.</summary>
        public static VDouble ProgramIndex => new VDouble(s_laneIndex);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VDouble(Vector<double> v) => V = v;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VDouble(double uniform) => V = new Vector<double>(uniform);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble Load(ReadOnlySpan<double> source, int offset = 0)
            => new VDouble(new Vector<double>(source.Slice(offset)));

        public static VDouble LoadMasked(ReadOnlySpan<double> source, int offset, VMaskD mask, double fallback = 0d)
        {
            Span<double> tmp = stackalloc double[LaneCount];
            for (int l = 0; l < LaneCount; l++)
            {
                int idx = offset + l;
                tmp[l] = (mask.IsLaneActive(l) && idx < source.Length) ? source[idx] : fallback;
            }
            return Load(tmp);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Store(Span<double> destination, int offset = 0) => V.CopyTo(destination.Slice(offset));

        /// <summary>Non-temporal (streaming) store when the target is aligned, else an ordinary store.
        /// Weakly ordered, pair with <see cref="Memory.StoreFence"/>. See <see cref="SpmdAttribute.Streaming"/>.</summary>
        public unsafe void StoreNonTemporal(Span<double> destination, int offset = 0)
        {
#if NET8_0_OR_GREATER
            var slice = destination.Slice(offset);
            if (System.Runtime.Intrinsics.X86.Avx.IsSupported && LaneCount == 4)
            {
                fixed (double* p = slice)
                    if (((nuint)p & 31u) == 0)
                    {
                        System.Runtime.Intrinsics.X86.Avx.StoreAlignedNonTemporal(p, System.Runtime.Intrinsics.Vector256.AsVector256(V));
                        return;
                    }
            }
            else if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && LaneCount == 8)
            {
                fixed (double* p = slice)
                    if (((nuint)p & 63u) == 0)
                    {
                        System.Runtime.Intrinsics.X86.Avx512F.StoreAlignedNonTemporal(p, System.Runtime.Intrinsics.Vector512.AsVector512(V));
                        return;
                    }
            }
            else if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && LaneCount == 2)
            {
                fixed (double* p = slice)
                    if (((nuint)p & 15u) == 0)
                    {
                        System.Runtime.Intrinsics.X86.Sse2.StoreAlignedNonTemporal(p, System.Runtime.Intrinsics.Vector128.AsVector128(V));
                        return;
                    }
            }
#endif
            V.CopyTo(destination.Slice(offset));
        }

        public void StoreMasked(Span<double> destination, int offset, VMaskD mask)
        {
            for (int l = 0; l < LaneCount; l++)
            {
                int idx = offset + l;
                if (mask.IsLaneActive(l) && idx < destination.Length)
                    destination[idx] = V[l];
            }
        }

        public double GetLane(int lane) => V[lane];

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator +(VDouble a, VDouble b) => new VDouble(a.V + b.V);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator -(VDouble a, VDouble b) => new VDouble(a.V - b.V);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator *(VDouble a, VDouble b) => new VDouble(a.V * b.V);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator /(VDouble a, VDouble b) => new VDouble(a.V / b.V);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator -(VDouble a) => new VDouble(-a.V);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator *(VDouble a, double b) => new VDouble(a.V * b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator *(double a, VDouble b) => b * a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator +(VDouble a, double b) => new VDouble(a.V + new Vector<double>(b));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble operator -(VDouble a, double b) => new VDouble(a.V - new Vector<double>(b));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator VDouble(double uniform) => new VDouble(uniform);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator <(VDouble a, VDouble b)  => new VMaskD(Vector.LessThan(a.V, b.V));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator >(VDouble a, VDouble b)  => new VMaskD(Vector.GreaterThan(a.V, b.V));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator <=(VDouble a, VDouble b) => new VMaskD(Vector.LessThanOrEqual(a.V, b.V));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VMaskD operator >=(VDouble a, VDouble b) => new VMaskD(Vector.GreaterThanOrEqual(a.V, b.V));
        public static VMaskD Eq(VDouble a, VDouble b)  => new VMaskD(Vector.Equals(a.V, b.V));
        public static VMaskD Neq(VDouble a, VDouble b) => new VMaskD(~Vector.Equals(a.V, b.V));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble Select(VMaskD mask, VDouble ifTrue, VDouble ifFalse)
            => new VDouble(Vector.ConditionalSelect(mask.Bits, ifTrue.V, ifFalse.V));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VDouble Blend(VMaskD mask, VDouble newValue) => Select(mask, newValue, this);

        /// <summary>
        /// Fused multiply-add: a*b + c. Uses hardware FMA (one instruction, one rounding)
        /// on net8.0+ when available (x86 FMA3 or ARM NEON); falls back to mul+add elsewhere.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble MulAdd(VDouble a, VDouble b, VDouble c)
        {
#if NET8_0_OR_GREATER
            if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported && Vector<double>.Count == 8)
            {
                var r = System.Runtime.Intrinsics.X86.Avx512F.FusedMultiplyAdd(
                    System.Runtime.Intrinsics.Vector512.AsVector512(a.V),
                    System.Runtime.Intrinsics.Vector512.AsVector512(b.V),
                    System.Runtime.Intrinsics.Vector512.AsVector512(c.V));
                return new VDouble(System.Runtime.Intrinsics.Vector512.AsVector(r));
            }
            if (System.Runtime.Intrinsics.X86.Fma.IsSupported && Vector<double>.Count == 4)
            {
                var r = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                    System.Runtime.Intrinsics.Vector256.AsVector256(a.V),
                    System.Runtime.Intrinsics.Vector256.AsVector256(b.V),
                    System.Runtime.Intrinsics.Vector256.AsVector256(c.V));
                return new VDouble(System.Runtime.Intrinsics.Vector256.AsVector(r));
            }
            if (System.Runtime.Intrinsics.X86.Fma.IsSupported && Vector<double>.Count == 2)
            {
                var r = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                    System.Runtime.Intrinsics.Vector128.AsVector128(a.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(b.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(c.V));
                return new VDouble(System.Runtime.Intrinsics.Vector128.AsVector(r));
            }
            if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported && Vector<double>.Count == 2)
            {
                // ARM fmla computes addend + left*right.
                var r = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.FusedMultiplyAdd(
                    System.Runtime.Intrinsics.Vector128.AsVector128(c.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(a.V),
                    System.Runtime.Intrinsics.Vector128.AsVector128(b.V));
                return new VDouble(System.Runtime.Intrinsics.Vector128.AsVector(r));
            }
#endif
            return a * b + c;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble Sqrt(VDouble x) => new VDouble(Vector.SquareRoot(x.V));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble Abs(VDouble x) => new VDouble(Vector.Abs(x.V));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble Min(VDouble a, VDouble b) => new VDouble(Vector.Min(a.V, b.V));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble Max(VDouble a, VDouble b) => new VDouble(Vector.Max(a.V, b.V));

        /// <summary>Sum of all lanes.</summary>
        public static double ReduceAdd(VDouble v) => Vector.Dot(v.V, Vector<double>.One);

        /// <summary>Sum of active lanes only.</summary>
        public static double ReduceAdd(VDouble v, VMaskD mask) => ReduceAdd(Select(mask, v, Zero));

        public bool Equals(VDouble other) => V.Equals(other.V);
        public override bool Equals(object? obj) => obj is VDouble d && Equals(d);
        public override int GetHashCode() => V.GetHashCode();
        public override string ToString() => V.ToString();
    }
}
