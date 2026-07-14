using System;

namespace IspcSharp;

/// <summary>
/// Marks a blittable struct as usable inside <c>[Spmd]</c>/<c>[SpmdFunction]</c> bodies. Fields are
/// <c>float</c>/<c>int</c>/<c>double</c>/<c>long</c>, or ISPC-style fixed-size array members declared
/// with <see cref="SpmdArrayAttribute"/>. The generator emits a varying companion whose fields are
/// the per-lane gang types (<c>VFloat</c>/<c>VInt</c>/…, one gang per array element), so the struct
/// can be a kernel local, a helper argument/return, or a Structure-of-Arrays buffer element accessed
/// as <c>buf[i].field</c> (buffers require same-width scalar fields and no array members).
/// </summary>
[AttributeUsage(AttributeTargets.Struct, AllowMultiple = false, Inherited = false)]
public sealed class SpmdStructAttribute : Attribute
{
}
