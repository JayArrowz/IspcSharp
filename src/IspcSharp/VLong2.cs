using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IspcSharp
{
    /// <summary>
    /// A full-int-gang-width varying long: <see cref="VInt.LaneCount"/> 64-bit integers held as
    /// two <see cref="VLong"/> halves. This is the cross-gang-width bridge, long gangs are
    /// half as wide as int/float gangs, so a float/int kernel that needs a 64-bit accumulator
    /// (a long sum over int[] data, an exact product) widens into a VLong2, computes, and narrows
    /// back. Masks are plain <see cref="VMask"/> (one bit per int lane), widened internally.
    /// The [Spmd] generator uses this automatically for long locals/casts in float/int kernels —
    /// the integer mirror of <see cref="VDouble2"/>.
    /// </summary>
    public readonly struct VLong2 : IEquatable<VLong2>
    {
        public readonly VLong Lower;
        public readonly VLong Upper;

        /// <summary>Lane count, matches <see cref="VInt.LaneCount"/> (twice a VLong gang).</summary>
        public static int LaneCount => Vector<int>.Count;

        public static VLong2 Zero => new VLong2(VLong.Zero, VLong.Zero);
        public static VLong2 One => new VLong2(VLong.One, VLong.One);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VLong2(VLong lower, VLong upper) { Lower = lower; Upper = upper; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VLong2(long uniform) { Lower = new VLong(uniform); Upper = new VLong(uniform); }

        /// <summary>Widen a full int gang to 64 bits (exact).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VLong2 FromInt(VInt v)
        {
            Vector.Widen(v.V, out Vector<long> lo, out Vector<long> hi);
            return new VLong2(new VLong(lo), new VLong(hi));
        }

        /// <summary>Widen a full float gang to 64-bit integers, truncating toward zero.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VLong2 FromFloatTruncate(VFloat v)
        {
            Vector.Widen(v.V, out Vector<double> lo, out Vector<double> hi);
            return new VLong2(new VLong(Vector.ConvertToInt64(lo)), new VLong(Vector.ConvertToInt64(hi)));
        }

        /// <summary>Narrow back to an int gang (wraps, like a <c>(int)</c> cast).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VInt ToInt() => new VInt(Vector.Narrow(Lower.V, Upper.V));

        /// <summary>Convert to a float gang (rounds to nearest, like a <c>(float)</c> cast).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VFloat ToFloat()
            => new VFloat(Vector.Narrow(Vector.ConvertToDouble(Lower.V), Vector.ConvertToDouble(Upper.V)));

        /// <summary>Widen to double precision (exact for |value| &lt; 2^53).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VDouble2 ToDouble2() => new VDouble2(Lower.ToDouble(), Upper.ToDouble());

        public long GetLane(int lane)
            => lane < VLong.LaneCount ? Lower.GetLane(lane) : Upper.GetLane(lane - VLong.LaneCount);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator +(VLong2 a, VLong2 b) => new VLong2(a.Lower + b.Lower, a.Upper + b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator -(VLong2 a, VLong2 b) => new VLong2(a.Lower - b.Lower, a.Upper - b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator *(VLong2 a, VLong2 b) => new VLong2(a.Lower * b.Lower, a.Upper * b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator -(VLong2 a) => new VLong2(-a.Lower, -a.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator +(VLong2 a, long b) => a + new VLong2(b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator -(VLong2 a, long b) => a - new VLong2(b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator *(VLong2 a, long b) => a * new VLong2(b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator *(long a, VLong2 b) => b * a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator &(VLong2 a, VLong2 b) => new VLong2(a.Lower & b.Lower, a.Upper & b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator |(VLong2 a, VLong2 b) => new VLong2(a.Lower | b.Lower, a.Upper | b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator ^(VLong2 a, VLong2 b) => new VLong2(a.Lower ^ b.Lower, a.Upper ^ b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator ~(VLong2 a) => new VLong2(~a.Lower, ~a.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator <<(VLong2 a, int count) => new VLong2(a.Lower << count, a.Upper << count);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 operator >>(VLong2 a, int count) => new VLong2(a.Lower >> count, a.Upper >> count);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 ShiftRightLogical(VLong2 a, int count) => new VLong2(VLong.ShiftRightLogical(a.Lower, count), VLong.ShiftRightLogical(a.Upper, count));
        public static VLong2 operator /(VLong2 a, VLong2 b) => Divide(a, b);
        public static VLong2 operator %(VLong2 a, VLong2 b) => Remainder(a, b);
        public static VLong2 Divide(VLong2 a, VLong2 b) => new VLong2(VLong.Divide(a.Lower, b.Lower), VLong.Divide(a.Upper, b.Upper));
        public static VLong2 Remainder(VLong2 a, VLong2 b) => new VLong2(VLong.Remainder(a.Lower, b.Lower), VLong.Remainder(a.Upper, b.Upper));

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 Min(VLong2 a, VLong2 b) => new VLong2(VLong.Min(a.Lower, b.Lower), VLong.Min(a.Upper, b.Upper));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 Max(VLong2 a, VLong2 b) => new VLong2(VLong.Max(a.Lower, b.Lower), VLong.Max(a.Upper, b.Upper));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static VLong2 Abs(VLong2 a) => new VLong2(VLong.Abs(a.Lower), VLong.Abs(a.Upper));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator VLong2(long uniform) => new VLong2(uniform);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator <(VLong2 a, VLong2 b)
            => NarrowMask(a.Lower < b.Lower, a.Upper < b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator >(VLong2 a, VLong2 b)
            => NarrowMask(a.Lower > b.Lower, a.Upper > b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator <=(VLong2 a, VLong2 b)
            => NarrowMask(a.Lower <= b.Lower, a.Upper <= b.Upper);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VMask operator >=(VLong2 a, VLong2 b)
            => NarrowMask(a.Lower >= b.Lower, a.Upper >= b.Upper);
        public static VMask Eq(VLong2 a, VLong2 b)
            => NarrowMask(VLong.Eq(a.Lower, b.Lower), VLong.Eq(a.Upper, b.Upper));
        public static VMask Neq(VLong2 a, VLong2 b)
            => NarrowMask(VLong.Neq(a.Lower, b.Lower), VLong.Neq(a.Upper, b.Upper));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static VMask NarrowMask(VMaskD lo, VMaskD hi) => VMask.Narrow(lo, hi);

        /// <summary>Per-lane: <c>mask ? ifTrue : ifFalse</c> with a full-width VMask.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VLong2 Select(VMask mask, VLong2 ifTrue, VLong2 ifFalse)
        {
            var (lo, hi) = VMaskD.Widen(mask);
            return new VLong2(
                VLong.Select(lo, ifTrue.Lower, ifFalse.Lower),
                VLong.Select(hi, ifTrue.Upper, ifFalse.Upper));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public VLong2 Blend(VMask mask, VLong2 newValue) => Select(mask, newValue, this);

        /// <summary>Fused multiply-add per half (<c>a*b + c</c>; integer, so no rounding).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static VLong2 MulAdd(VLong2 a, VLong2 b, VLong2 c) => a * b + c;

        public bool Equals(VLong2 other) => Lower.Equals(other.Lower) && Upper.Equals(other.Upper);
        public override bool Equals(object? obj) => obj is VLong2 l && Equals(l);
        public override int GetHashCode() => HashCode.Combine(Lower, Upper);
        public override string ToString() => $"({Lower}, {Upper})";
    }
}
