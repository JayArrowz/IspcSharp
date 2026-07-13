using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp
{
    /// <summary>
    /// A full-float-gang-width varying double: <see cref="VFloat.LaneCount"/> doubles held as
    /// two <see cref="VDouble"/> halves. This is the cross-gang-width bridge, double gangs are
    /// half as wide as float gangs, so a float kernel that needs double-precision intermediates
    /// (accumulators, geometric predicates) widens into a VDouble2, computes, and narrows back.
    /// Masks are plain <see cref="VMask"/> (one bit per float lane), widened internally.
    /// The [Spmd] generator uses this automatically for double locals/casts in float/int kernels.
    /// </summary>
    public readonly struct VDouble2 : IEquatable<VDouble2>
    {
        public readonly VDouble Lower;
        public readonly VDouble Upper;

        /// <summary>Lane count, matches <see cref="VFloat.LaneCount"/> (twice a VDouble gang).</summary>
        public static int LaneCount => Vector<float>.Count;

        public static VDouble2 Zero => new VDouble2(VDouble.Zero, VDouble.Zero);
        public static VDouble2 One => new VDouble2(VDouble.One, VDouble.One);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VDouble2(VDouble lower, VDouble upper) { Lower = lower; Upper = upper; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VDouble2(double uniform) { Lower = new VDouble(uniform); Upper = new VDouble(uniform); }

        /// <summary>Widen a full float gang to double precision (exact).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble2 FromFloat(VFloat v)
        {
            Vector.Widen(v.V, out Vector<double> lo, out Vector<double> hi);
            return new VDouble2(new VDouble(lo), new VDouble(hi));
        }

        /// <summary>Widen a full int gang to double precision (exact).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble2 FromInt(VInt v)
        {
            Vector.Widen(v.V, out Vector<long> lo, out Vector<long> hi);
            return new VDouble2(new VDouble(Vector.ConvertToDouble(lo)), new VDouble(Vector.ConvertToDouble(hi)));
        }

        /// <summary>Narrow back to a float gang (rounds to nearest float).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VFloat ToFloat() => new VFloat(Vector.Narrow(Lower.V, Upper.V));

        /// <summary>Truncate toward zero to an int gang.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VInt ToIntTruncate()
            => new VInt(Vector.Narrow(Vector.ConvertToInt64(Lower.V), Vector.ConvertToInt64(Upper.V)));

        public double GetLane(int lane)
            => lane < VDouble.LaneCount ? Lower.GetLane(lane) : Upper.GetLane(lane - VDouble.LaneCount);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator +(VDouble2 a, VDouble2 b) => new VDouble2(a.Lower + b.Lower, a.Upper + b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator -(VDouble2 a, VDouble2 b) => new VDouble2(a.Lower - b.Lower, a.Upper - b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator *(VDouble2 a, VDouble2 b) => new VDouble2(a.Lower * b.Lower, a.Upper * b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator /(VDouble2 a, VDouble2 b) => new VDouble2(a.Lower / b.Lower, a.Upper / b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator -(VDouble2 a) => new VDouble2(-a.Lower, -a.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator +(VDouble2 a, double b) => a + new VDouble2(b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator -(VDouble2 a, double b) => a - new VDouble2(b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator *(VDouble2 a, double b) => a * new VDouble2(b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VDouble2 operator *(double a, VDouble2 b) => b * a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator VDouble2(double uniform) => new VDouble2(uniform);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator <(VDouble2 a, VDouble2 b)
            => NarrowMask(a.Lower < b.Lower, a.Upper < b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator >(VDouble2 a, VDouble2 b)
            => NarrowMask(a.Lower > b.Lower, a.Upper > b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator <=(VDouble2 a, VDouble2 b)
            => NarrowMask(a.Lower <= b.Lower, a.Upper <= b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator >=(VDouble2 a, VDouble2 b)
            => NarrowMask(a.Lower >= b.Lower, a.Upper >= b.Upper);
        public static VMask Eq(VDouble2 a, VDouble2 b)
            => NarrowMask(VDouble.Eq(a.Lower, b.Lower), VDouble.Eq(a.Upper, b.Upper));
        public static VMask Neq(VDouble2 a, VDouble2 b)
            => NarrowMask(VDouble.Neq(a.Lower, b.Lower), VDouble.Neq(a.Upper, b.Upper));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static VMask NarrowMask(VMaskD lo, VMaskD hi) => VMask.Narrow(lo, hi);

        /// <summary>Per-lane: <c>mask ? ifTrue : ifFalse</c> with a full-width VMask.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble2 Select(VMask mask, VDouble2 ifTrue, VDouble2 ifFalse)
        {
            var (lo, hi) = VMaskD.Widen(mask);
            return new VDouble2(
                VDouble.Select(lo, ifTrue.Lower, ifFalse.Lower),
                VDouble.Select(hi, ifTrue.Upper, ifFalse.Upper));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VDouble2 Blend(VMask mask, VDouble2 newValue) => Select(mask, newValue, this);

        /// <summary>Fused multiply-add per half (hardware FMA when available).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VDouble2 MulAdd(VDouble2 a, VDouble2 b, VDouble2 c)
            => new VDouble2(VDouble.MulAdd(a.Lower, b.Lower, c.Lower), VDouble.MulAdd(a.Upper, b.Upper, c.Upper));

        public bool Equals(VDouble2 other) => Lower.Equals(other.Lower) && Upper.Equals(other.Upper);
        public override bool Equals(object? obj) => obj is VDouble2 d && Equals(d);
        public override int GetHashCode() => HashCode.Combine(Lower, Upper);
        public override string ToString() => $"({Lower}, {Upper})";
    }
}
