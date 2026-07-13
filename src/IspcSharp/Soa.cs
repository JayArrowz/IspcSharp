using System;

namespace IspcSharp
{
    /// <summary>
    /// Structure-of-Arrays containers. SIMD wants each field contiguous in memory
    /// (x[0..n], y[0..n], z[0..n]) rather than interleaved structs (xyz, xyz, ...),
    /// because contiguous fields load with one instruction while interleaved fields
    /// need gathers. These helpers make the SoA layout convenient.
    /// </summary>
    public sealed class SoaFloat2
    {
        public readonly float[] X;
        public readonly float[] Y;
        public int Length { get; }

        public SoaFloat2(int length)
        {
            Length = length;
            X = new float[length];
            Y = new float[length];
        }

        /// <summary>Load lane-parallel X/Y for the gang starting at <paramref name="offset"/>.</summary>
        public (VFloat X, VFloat Y) LoadGang(int offset)
            => (VFloat.Load(X, offset), VFloat.Load(Y, offset));

        public void StoreGang(int offset, VFloat x, VFloat y)
        {
            x.Store(X, offset);
            y.Store(Y, offset);
        }

        public void StoreGangMasked(int offset, VFloat x, VFloat y, VMask mask)
        {
            x.StoreMasked(X, offset, mask);
            y.StoreMasked(Y, offset, mask);
        }

        /// <summary>De-interleave an Array-of-Structs buffer (x,y,x,y,…) into SoA (the explicit,
        /// one-time cost that turns per-field gathers into contiguous loads).</summary>
        public static SoaFloat2 FromInterleaved(ReadOnlySpan<float> xy)
        {
            int n = xy.Length / 2;
            var soa = new SoaFloat2(n);
            for (int i = 0; i < n; i++) { soa.X[i] = xy[2 * i]; soa.Y[i] = xy[2 * i + 1]; }
            return soa;
        }

        /// <summary>Re-interleave SoA back into an AoS buffer (x,y,x,y,…).</summary>
        public void ToInterleaved(Span<float> xy)
        {
            for (int i = 0; i < Length; i++) { xy[2 * i] = X[i]; xy[2 * i + 1] = Y[i]; }
        }
    }

    public sealed class SoaFloat3
    {
        public readonly float[] X;
        public readonly float[] Y;
        public readonly float[] Z;
        public int Length { get; }

        public SoaFloat3(int length)
        {
            Length = length;
            X = new float[length];
            Y = new float[length];
            Z = new float[length];
        }

        public (VFloat X, VFloat Y, VFloat Z) LoadGang(int offset)
            => (VFloat.Load(X, offset), VFloat.Load(Y, offset), VFloat.Load(Z, offset));

        public void StoreGang(int offset, VFloat x, VFloat y, VFloat z)
        {
            x.Store(X, offset);
            y.Store(Y, offset);
            z.Store(Z, offset);
        }

        public void StoreGangMasked(int offset, VFloat x, VFloat y, VFloat z, VMask mask)
        {
            x.StoreMasked(X, offset, mask);
            y.StoreMasked(Y, offset, mask);
            z.StoreMasked(Z, offset, mask);
        }

        /// <summary>Per-lane 3D dot product.</summary>
        public static VFloat Dot(
            (VFloat X, VFloat Y, VFloat Z) a,
            (VFloat X, VFloat Y, VFloat Z) b)
            => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

        /// <summary>De-interleave an Array-of-Structs buffer (x,y,z,x,y,z,…) into SoA.</summary>
        public static SoaFloat3 FromInterleaved(ReadOnlySpan<float> xyz)
        {
            int n = xyz.Length / 3;
            var soa = new SoaFloat3(n);
            for (int i = 0; i < n; i++)
            {
                soa.X[i] = xyz[3 * i];
                soa.Y[i] = xyz[3 * i + 1];
                soa.Z[i] = xyz[3 * i + 2];
            }
            return soa;
        }

        /// <summary>Re-interleave SoA back into an AoS buffer (x,y,z,x,y,z,…).</summary>
        public void ToInterleaved(Span<float> xyz)
        {
            for (int i = 0; i < Length; i++)
            {
                xyz[3 * i] = X[i];
                xyz[3 * i + 1] = Y[i];
                xyz[3 * i + 2] = Z[i];
            }
        }
    }
}
