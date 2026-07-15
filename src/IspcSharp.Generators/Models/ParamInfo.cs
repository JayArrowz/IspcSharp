namespace IspcSharp.Generators.Models;

internal readonly struct ParamInfo(string name, ParamKind kind, string typeText, string? structType = null)
{
    public readonly string Name = name;
    public readonly ParamKind PKind = kind;
    public readonly string TypeText = typeText;

    /// <summary>
    /// Element struct type name for a <see cref="ParamKind.StructArray"/> parameter.
    /// </summary>
    public readonly string? StructType = structType;

    public bool IsStructBuffer => PKind == ParamKind.StructArray;

    public bool Is2D => PKind is ParamKind.FloatArray2D or ParamKind.IntArray2D or ParamKind.DoubleArray2D or ParamKind.LongArray2D;

    public bool IsBuffer => PKind is ParamKind.FloatArray or ParamKind.FloatSpan or ParamKind.FloatReadOnlySpan
                                  or ParamKind.IntArray or ParamKind.IntSpan or ParamKind.IntReadOnlySpan
                                  or ParamKind.DoubleArray or ParamKind.DoubleSpan or ParamKind.DoubleReadOnlySpan
                                  or ParamKind.LongArray or ParamKind.LongSpan or ParamKind.LongReadOnlySpan
                                  or ParamKind.ByteArray or ParamKind.ByteSpan or ParamKind.ByteReadOnlySpan
                                  or ParamKind.ShortArray or ParamKind.ShortSpan or ParamKind.ShortReadOnlySpan
                                  or ParamKind.FloatArray2D or ParamKind.IntArray2D or ParamKind.DoubleArray2D or ParamKind.LongArray2D;

    public bool IsReadOnly => PKind is ParamKind.FloatReadOnlySpan or ParamKind.IntReadOnlySpan or ParamKind.DoubleReadOnlySpan
                                    or ParamKind.LongReadOnlySpan or ParamKind.ByteReadOnlySpan or ParamKind.ShortReadOnlySpan;

    public bool IsSpan => PKind is ParamKind.FloatSpan or ParamKind.FloatReadOnlySpan
                                or ParamKind.IntSpan or ParamKind.IntReadOnlySpan
                                or ParamKind.DoubleSpan or ParamKind.DoubleReadOnlySpan
                                or ParamKind.LongSpan or ParamKind.LongReadOnlySpan
                                or ParamKind.ByteSpan or ParamKind.ByteReadOnlySpan
                                or ParamKind.ShortSpan or ParamKind.ShortReadOnlySpan;

    public Kind ElemKind => PKind switch
    {
        ParamKind.IntArray or ParamKind.IntSpan or ParamKind.IntReadOnlySpan or ParamKind.IntArray2D => Kind.I,
        ParamKind.DoubleArray or ParamKind.DoubleSpan or ParamKind.DoubleReadOnlySpan or ParamKind.DoubleArray2D => Kind.D,
        ParamKind.LongArray or ParamKind.LongSpan or ParamKind.LongReadOnlySpan or ParamKind.LongArray2D => Kind.L,
        ParamKind.ByteArray or ParamKind.ByteSpan or ParamKind.ByteReadOnlySpan => Kind.B,
        ParamKind.ShortArray or ParamKind.ShortSpan or ParamKind.ShortReadOnlySpan => Kind.S,
        _ => Kind.F,
    };

    /// <summary>
    /// The flat 1-D span name used to view a 2-D array's row-major storage.
    /// </summary>
    public string FlatName => "__flat_" + Name;

    /// <summary>
    /// The local holding the 2-D array's column count (GetLength(1)).
    /// </summary>
    public string ColsName => "__cols_" + Name;
}