using System;

namespace IspcSharp;

/// <summary>
/// Declares a <c>float[]</c>/<c>int[]</c>/<c>double[]</c>/<c>long[]</c> field of a <c>[SpmdStruct]</c>
/// as an ISPC-style <b>fixed-size array member</b> of the given length. The varying companion expands
/// it into that many independent gangs held in registers (Structure-of-Arrays), so element access
/// <c>s.field[k]</c> resolves to a single gang. The index <c>k</c> must be a compile-time integer
/// literal (registers have no runtime-indexed lane), and such a struct is a local / helper argument /
/// return only, not a buffer element. Example: <c>[SpmdArray(4)] public float[] Weights;</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class SpmdArrayAttribute(int length) : Attribute
{
    /// <summary>
    /// Fixed element count of the array member (must be &gt; 0).
    /// </summary>
    public int Length { get; } = length;
}
