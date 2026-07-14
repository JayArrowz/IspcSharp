using System.Collections.Generic;
using IspcSharp.Generators.Models;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators.Contexts;

internal sealed class ScaffoldContext(
    CommonForEachStatementSyntax fe,
    string loopVar,
    string startExpr,
    string endExpr,
    string vectorBody,
    string scalarBody,
    List<ReductionInfo> reductions,
    string laneCountExpr,
    bool doubleMode,
    bool longMode,
    int unroll,
    bool hasLaneReturns)
{
    public readonly CommonForEachStatementSyntax Fe = fe;
    public readonly string LoopVar = loopVar;
    public readonly string StartExpr = startExpr;
    public readonly string EndExpr = endExpr;
    public readonly string VectorBody = vectorBody;
    public readonly string ScalarBody = scalarBody;
    public readonly string LaneCountExpr = laneCountExpr;
    public readonly List<ReductionInfo> Reductions = reductions;
    public readonly bool DoubleMode = doubleMode;
    public readonly bool LongMode = longMode;
    public readonly bool HasLaneReturns = hasLaneReturns;
    public readonly int Unroll = unroll;
}
