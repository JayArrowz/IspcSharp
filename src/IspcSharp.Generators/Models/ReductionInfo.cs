namespace IspcSharp.Generators.Models;

internal sealed class ReductionInfo
{
    public string Name { get; set; } = "";
    public Kind LaneKind { get; set; }
    public ReduceOp Op { get; set; }
}