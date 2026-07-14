namespace IspcSharp.Generators.Models;

/// <summary>
/// One field of a blittable struct. <see cref="ArrayLength"/> is 0 for a scalar field,
/// or the fixed element count for an ISPC-style array member (<c>[SpmdArray(N)] float[] f</c>).
/// </summary>
internal sealed record StructField(string Name, Kind Kind, int ArrayLength = 0)
{
    public bool IsArray => ArrayLength > 0;

    /// <summary>
    /// Gang name of element <paramref name="i"/> of an array member (<c>f_0</c>, <c>f_1</c>, …).
    /// </summary>
    public string GangName(int i) => Name + "_" + i;
}
