namespace IspcSharp.Generators.Models;

/// <summary>
/// Lane kinds. F/I/D/L are value kinds (a lane variable can hold them). B (byte) and
/// S (short) are memory-only kinds, ISPC-style narrow buffer elements: loads widen them
/// into int lanes (<c>VInt.LoadZeroExtend</c>/<c>LoadSignExtend</c>) and stores narrow back
/// (<c>StoreNarrow</c>), so in-register compute is always F/I/D/L at a fixed gang width.
/// </summary>
internal enum Kind { F, I, D, L, B, S }
