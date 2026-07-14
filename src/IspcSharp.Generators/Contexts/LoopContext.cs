namespace IspcSharp.Generators.Contexts;

/// <summary>
/// Tracks break/continue state for the innermost loop being emitted.
/// For uniform loops (plain C# for), break/continue map to C# keywords.
/// For varying loops (mask iteration), break/continue manipulate masks.
/// </summary>
internal readonly struct LoopContext
{
    public readonly string BreakMask;
    public readonly string ContinueMask;
    public readonly string LoopMask;
    public readonly bool IsUniform;

    public LoopContext(string breakMask, string continueMask, string loopMask)
    {
        BreakMask = breakMask;
        ContinueMask = continueMask;
        LoopMask = loopMask;
        IsUniform = false;
    }

    private LoopContext(bool isUniform)
    {
        BreakMask = "";
        ContinueMask = "";
        LoopMask = "";
        IsUniform = isUniform;
    }

    public static LoopContext Uniform() => new(true);
}