using System;

namespace IspcSharp;

/// <summary>
/// Marks a reusable, side-effect-free helper for SPMD vectorization, ISPC's non-<c>export</c>
/// function. The generator emits a "varying" companion (each <c>float</c> parameter/return becomes
/// a <c>VFloat</c> lane, each blittable struct its varying form) that any <c>[Spmd]</c> kernel or
/// other <c>[SpmdFunction]</c> can call. The body uses the same supported subset as a kernel, but
/// takes scalar values instead of buffers and has no <c>Spmd.Range</c> loop, it operates on the
/// gang it is handed. Recursion and buffer parameters are not allowed.
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SpmdFunctionAttribute : Attribute
{
}
