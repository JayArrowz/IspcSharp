namespace IspcSharp.Generators.Models;

internal enum ParamKind
{
    FloatArray, FloatSpan, FloatReadOnlySpan,
    IntArray, IntSpan, IntReadOnlySpan,
    DoubleArray, DoubleSpan, DoubleReadOnlySpan,
    LongArray, LongSpan, LongReadOnlySpan,
    ByteArray, ByteSpan, ByteReadOnlySpan,
    ShortArray, ShortSpan, ShortReadOnlySpan,
    FloatArray2D, IntArray2D, DoubleArray2D, LongArray2D,
    UniformFloat, UniformInt, UniformDouble, UniformLong, UniformByte, UniformShort,
    StructArray, UniformStruct, Unsupported
}
