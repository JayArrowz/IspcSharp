using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace IspcSharp.Generators
{
    /// <summary>
    /// Walks the scalar loop body and emits equivalent masked-vector code, tracking whether
    /// each expression is float-lane (VFloat) or int-lane (VInt).
    ///
    /// Core lowering rules:
    ///   float/int local       -> VFloat / VInt
    ///   buf[i]                -> VFloat.Load / VInt.Load  (contiguous gang access)
    ///   buf[i] = e            -> Store, or masked read-modify-write inside divergent flow
    ///   if (c) A else B       -> mask m = c; writes in A blended by m, in B by !m
    ///   while (c) { A }       -> VMask loop = c & outer; while (loop.Any()) { A@loop; loop &= c; }
    ///   acc += e (reduction)  -> __red_acc += Select(mask, e, 0)
    ///   (float)x / (int)x     -> lane conversions
    /// </summary>
    internal sealed class VectorBodyEmitter
    {
        private readonly string _loopVar;
        private readonly Dictionary<string, SpmdGenerator.ParamInfo> _params;
        private readonly HashSet<string> _uniformPreLocals;
        private readonly Dictionary<string, SpmdGenerator.Kind> _preLocalKinds;
        private readonly Dictionary<string, SpmdGenerator.Kind> _locals = new();

        /// <summary>
        /// Int lane locals that are affine functions of the loop variable with unit stride:
        /// 'int even = i + k;' → at lane l, even = (i + __i) + l. Maps name → the uniform base
        /// offset ("i", "i + half", "0", …). Indexing such a local (real[even]) lowers to a
        /// contiguous Load/Store at __i + offset instead of a gather/scatter.
        /// </summary>
        private readonly Dictionary<string, string> _affineOffset = new();

        private readonly Dictionary<string, SpmdGenerator.ReductionInfo> _reductions;
        private readonly bool _d;                       // double-lane kernel (VDouble/VMaskD gangs)
        private readonly bool _l;                       // long-lane kernel (VLong/VMaskD gangs)
        private readonly bool _streaming;               // emit non-temporal stores ([Spmd(Streaming=true)])
        private bool Wide => _d || _l;                  // 64-bit gang (double or long)

        /// <summary>
        /// Blittable structs (by name) and [SpmdFunction] helpers (by name), plus the struct-typed
        /// locals/params in scope (local name → struct type name).
        /// </summary>
        private readonly IReadOnlyDictionary<string, StructInfo> _structs;
        private readonly IReadOnlyDictionary<string, FunctionInfo> _functions;
        private readonly Dictionary<string, string> _structLocals = new();
        // Function mode: no gang loop, pure (unmasked) body returning a value.
        private readonly bool _functionMode;
        private SpmdGenerator.Kind _retKind;
        private string? _retStruct;
        private int _maskCounter;
        private readonly Stack<LoopContext> _loopStack = new();

        /// <summary>
        /// The execution mask governing the expression currently being lowered. Integer
        /// division uses it to substitute 1 into inactive lanes' divisors so masked-off
        /// lanes can't raise DivideByZeroException (both branch sides always execute).
        /// </summary>
        private string? _exprMask;
        public List<Diagnostic> Diagnostics { get; } = new();
        private void Warn(DiagnosticDescriptor d, SyntaxNode at, string arg)
            => Diagnostics.Add(Diagnostic.Create(d, at.GetLocation(), arg));

        /// <summary>
        /// Mask type names, 64-bit (double/long) kernels use the 64-bit-lane mask.
        /// </summary>
        private string MT => Wide ? "VMaskD" : "VMask";
        private string MTAll => Wide ? "VMaskD.All" : "VMask.All";
        private string MTNone => Wide ? "VMaskD.None" : "VMask.None";
        private SpmdGenerator.Kind LaneFloatKind => _l ? SpmdGenerator.Kind.L : _d ? SpmdGenerator.Kind.D : SpmdGenerator.Kind.F;
        private string IntT => _l ? "VLong" : "VInt";
        private string VType(SpmdGenerator.Kind k) => SpmdGenerator.VType(k, _d, _l);
        private string SelectOf(SpmdGenerator.Kind k) => SpmdGenerator.VType(k, _d, _l) + ".Select";

        public VectorBodyEmitter(
            string loopVar,
            List<SpmdGenerator.ParamInfo> parameters,
            HashSet<string> uniformPreLocals,
            Dictionary<string, SpmdGenerator.Kind> preLocalKinds,
            List<SpmdGenerator.ReductionInfo> reductions,
            bool doubleMode,
            bool longMode,
            IReadOnlyDictionary<string, StructInfo> structs,
            IReadOnlyDictionary<string, FunctionInfo> functions,
            bool streaming = false)
        {
            _loopVar = loopVar;
            _params = parameters.ToDictionary(p => p.Name, p => p);
            _uniformPreLocals = uniformPreLocals;
            _preLocalKinds = preLocalKinds;
            _reductions = reductions.ToDictionary(r => r.Name);
            _d = doubleMode;
            _l = longMode;
            _streaming = streaming;
            _structs = structs;
            _functions = functions;
            foreach (var p in parameters)
                if (p.IsStructBuffer) _structBuffers[p.Name] = p.StructType!;
        }

        /// <summary>Function mode: lower a pure [SpmdFunction] body (float/int gang, no loop, unmasked).</summary>
        public VectorBodyEmitter(
            IReadOnlyDictionary<string, StructInfo> structs,
            IReadOnlyDictionary<string, FunctionInfo> functions,
            Dictionary<string, SpmdGenerator.Kind> paramLocals,
            Dictionary<string, string> paramStructs)
        {
            _loopVar = "";
            _params = new Dictionary<string, SpmdGenerator.ParamInfo>();
            _uniformPreLocals = new HashSet<string>();
            _preLocalKinds = new Dictionary<string, SpmdGenerator.Kind>();
            _reductions = new Dictionary<string, SpmdGenerator.ReductionInfo>();
            _structs = structs;
            _functions = functions;
            _functionMode = true;
            foreach (var kv in paramLocals) _locals[kv.Key] = kv.Value;
            foreach (var kv in paramStructs) _structLocals[kv.Key] = kv.Value;
        }

        /// <summary>Lower a [SpmdFunction] body to <c>return</c> the varying result.</summary>
        public string EmitFunctionBody(MethodDeclarationSyntax method, string indent)
        {
            string ret = method.ReturnType.ToString().Trim();
            if (_structs.TryGetValue(ret, out var rs)) { _retStruct = rs.Name; }
            else _retKind = SpmdGenerator.KindOfScalar(ret, method.GetLocation());

            var sb = new StringBuilder();
            if (method.ExpressionBody is { } arrow)
            {
                sb.AppendLine($"{indent}return {CoerceResult(arrow.Expression)};");
                return sb.ToString();
            }
            foreach (var st in method.Body!.Statements)
                EmitFunctionStatement(st, indent, sb);
            return sb.ToString();
        }

        private void EmitFunctionStatement(StatementSyntax st, string indent, StringBuilder sb)
        {
            switch (st)
            {
                case LocalDeclarationStatementSyntax decl:
                    EmitLocalDecl(decl, null, indent, sb);
                    return;
                case ReturnStatementSyntax { Expression: { } rexpr }:
                    sb.AppendLine($"{indent}return {CoerceResult(rexpr)};");
                    return;
                case ExpressionStatementSyntax es when es.Expression is AssignmentExpressionSyntax asg:
                    EmitAssignment(asg, null, indent, sb);
                    return;
                case BlockSyntax block:
                    foreach (var inner in block.Statements) EmitFunctionStatement(inner, indent, sb);
                    return;
                default:
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"statement '{Trunc(st)}' in a [SpmdFunction] (use locals, assignments, and a final 'return expr')", st.GetLocation());
            }
        }

        /// <summary>Coerce an expression to the function's return type (primitive or struct).</summary>
        private string CoerceResult(ExpressionSyntax e)
            => _retStruct != null ? VecStruct(e).Code : Coerce(Vec(e), _retKind, e);

        public string EmitStatements(SyntaxList<StatementSyntax> statements, string? maskExpr, string indent)
        {
            var sb = new StringBuilder();

            // Fast path: no per-lane 'break'/'continue'/'return' targets this foreach, so every lane
            // runs the whole body. Emit statements with the incoming mask unchanged (null at the top
            // level), no foreach continue/break masks, so top-level stores/reductions skip the
            // redundant Select-with-reload entirely.
            if (!BodyTargetsForeachExit(statements))
            {
                foreach (var st in statements)
                    EmitStatement(st, maskExpr, indent, sb);
                return sb.ToString();
            }

            // A per-lane 'continue'/'break'/'return' can retire a lane mid-body, so the foreach is a
            // "loop" context with continue/break masks that later statements must respect.
            string foreachContMask = $"__fcnt{_maskCounter++}";
            string foreachBrkMask = $"__fbrk{_maskCounter++}";
            sb.AppendLine($"{indent}{MT} {foreachContMask} = {MTNone};");
            sb.AppendLine($"{indent}{MT} {foreachBrkMask} = {MTNone};");
            _loopStack.Push(new LoopContext(foreachBrkMask, foreachContMask, maskExpr ?? MTAll));

            // Each statement's mask is the base mask minus any lane that has already continued/broken —
            // computed fresh (not accumulated) so it stays 'base & !cnt & !brk', not a growing chain.
            string baseMask = maskExpr ?? MTAll;
            foreach (var st in statements)
            {
                // Local declarations at the foreach top level are not divergent, they allocate a gang;
                // pass the original mask (null at top level) so EmitLocalDecl doesn't blend them.
                if (st is LocalDeclarationStatementSyntax)
                    EmitStatement(st, maskExpr, indent, sb);
                else
                    EmitStatement(st, $"{baseMask} & !{foreachContMask} & !{foreachBrkMask}", indent, sb);
            }

            _loopStack.Pop();
            return sb.ToString();
        }

        /// <summary>True if a per-lane <c>return</c>, or a <c>break</c>/<c>continue</c> that targets the
        /// foreach itself (not a nested for/while), appears in the body, i.e. a lane can retire mid-body
        /// and the foreach-level continue/break masks are actually needed.</summary>
        private static bool BodyTargetsForeachExit(SyntaxList<StatementSyntax> statements)
        {
            var bodyBlock = statements.Count > 0 ? statements[0].Parent : null;
            foreach (var st in statements)
                foreach (var node in st.DescendantNodesAndSelf())
                {
                    if (node is ReturnStatementSyntax) return true;
                    if (node is BreakStatementSyntax or ContinueStatementSyntax)
                    {
                        bool nested = false;
                        for (var a = node.Parent; a != null && a != bodyBlock; a = a.Parent)
                            if (a is ForStatementSyntax or WhileStatementSyntax or DoStatementSyntax or CommonForEachStatementSyntax)
                            { nested = true; break; }
                        if (!nested) return true;
                    }
                }
            return false;
        }

        private void EmitStatement(StatementSyntax st, string? mask, string indent, StringBuilder sb)
        {
            _exprMask = mask;
            switch (st)
            {
                case LocalDeclarationStatementSyntax decl:
                    EmitLocalDecl(decl, mask, indent, sb);
                    break;

                case ExpressionStatementSyntax { Expression: AssignmentExpressionSyntax asg }:
                    EmitAssignment(asg, mask, indent, sb);
                    break;

                case ExpressionStatementSyntax { Expression: PostfixUnaryExpressionSyntax post }
                    when post.OperatorToken.Text is "++" or "--":
                    EmitIncDec(post.Operand, post.OperatorToken.Text, mask, indent, sb, post.GetLocation());
                    break;

                case ExpressionStatementSyntax { Expression: PrefixUnaryExpressionSyntax pre }
                    when pre.OperatorToken.Text is "++" or "--":
                    EmitIncDec(pre.Operand, pre.OperatorToken.Text, mask, indent, sb, pre.GetLocation());
                    break;

                case IfStatementSyntax ifs:
                    EmitIf(ifs, mask, indent, sb);
                    break;

                case WhileStatementSyntax whs:
                    EmitWhile(whs, mask, indent, sb);
                    break;

                case BlockSyntax block:
                    EmitBlock(block, mask, indent, sb);
                    break;

                case ForStatementSyntax fors:
                    EmitFor(fors, mask, indent, sb);
                    break;

                case BreakStatementSyntax bs:
                    EmitBreak(mask, indent, sb, bs.GetLocation());
                    break;

                case ContinueStatementSyntax cs:
                    EmitContinue(mask, indent, sb, cs.GetLocation());
                    break;

                case ReturnStatementSyntax rs:
                    EmitReturn(rs, mask, indent, sb);
                    break;

                case EmptyStatementSyntax:
                    break;

                default:
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"statement '{Trunc(st)}' ({st.Kind()})", st.GetLocation());
            }
        }

        /// <summary>
        /// Emit a block of statements, threading the continue and break masks: after a 'continue'
        /// or 'break', subsequent statements in the same block are masked with !continueMask &amp; !breakMask.
        /// Local declarations are exempt, they allocate per-gang, not per-iteration.
        /// </summary>
        private void EmitBlock(BlockSyntax block, string? mask, string indent, StringBuilder sb)
        {
            string? currentMask = mask;
            foreach (var inner in block.Statements)
            {
                // If inside a varying loop and a continue/break was emitted, update the mask
                // so subsequent statements don't execute for lanes that continued or broke.
                if (_loopStack.Count > 0 && !_loopStack.Peek().IsUniform)
                {
                    var ctx = _loopStack.Peek();
                    if (currentMask != null)
                        currentMask = $"{currentMask} & !{ctx.ContinueMask} & !{ctx.BreakMask}";
                    else
                        currentMask = $"{MTAll} & !{ctx.ContinueMask} & !{ctx.BreakMask}";
                }
                // Local declarations are not divergent, they allocate VFloat/VInt per gang.
                if (inner is LocalDeclarationStatementSyntax)
                    EmitStatement(inner, mask, indent, sb);
                else
                    EmitStatement(inner, currentMask, indent, sb);
            }
        }

        private void EmitLocalDecl(LocalDeclarationStatementSyntax decl, string? mask, string indent, StringBuilder sb)
        {
            string t = decl.Declaration.Type.ToString();

            // Struct locals: 'S v = <struct expr>;' → 'S__V v = ...;' (blended for masked branches).
            if (_structs.TryGetValue(t, out var si))
            {
                foreach (var v in decl.Declaration.Variables)
                {
                    _structLocals[v.Identifier.Text] = t;
                    string init = v.Initializer != null
                        ? VecStruct(v.Initializer.Value).Code
                        : "default";
                    if (mask != null && v.Initializer != null)
                        sb.AppendLine($"{indent}{si.VName} {v.Identifier.Text} = {si.VName}.Select({mask}, {init}, default);");
                    else
                        sb.AppendLine($"{indent}{si.VName} {v.Identifier.Text} = {init};");
                }
                return;
            }

            SpmdGenerator.Kind kind = t switch
            {
                "float" => SpmdGenerator.Kind.F,
                "int" => SpmdGenerator.Kind.I,
                "double" => SpmdGenerator.Kind.D,
                "long" => SpmdGenerator.Kind.L,
                "var" => throw new SpmdGenerator.UnsupportedConstructException(
                    "'var' local (declare as float, int, double, long, or a [SpmdStruct] type)", decl.GetLocation()),
                _ => throw new SpmdGenerator.UnsupportedConstructException(
                    $"local of type '{t}' (only float/int/double/long and [SpmdStruct] locals)", decl.GetLocation()),
            };
            if (_l && kind != SpmdGenerator.Kind.L)
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"'{t}' lane local in a long kernel (64-bit integer gang; declare it as long)",
                    decl.GetLocation());
            if (_d && kind != SpmdGenerator.Kind.D)
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"'{t}' lane local in a double kernel (double gangs have a different lane count; declare it as double)",
                    decl.GetLocation());
            // double/long locals in float/int kernels are fine: they become VDouble2 / VLong2 pairs
            // (64-bit precision at full float/int-gang width, the cross-gang-width bridge).
            // Note: we no longer reject local declarations inside masked branches.
            // The continue/break mask threading handles correctness, locals are
            // allocated once per gang, and their initializers are masked appropriately
            // by the enclosing if/while/for mask. A local declared after a 'continue'
            // simply gets initialized with a masked select.

            string vtype = VType(kind);
            foreach (var v in decl.Declaration.Variables)
            {
                _locals[v.Identifier.Text] = kind;
                // Record affine-in-lane int locals declared unmasked, so uses as array indices
                // become contiguous loads/stores. (In a masked branch the initializer is blended
                // with 0 for inactive lanes, which breaks the affine property, so skip those.)
                if (kind is SpmdGenerator.Kind.I or SpmdGenerator.Kind.L && mask == null &&
                    v.Initializer != null && TryClassifyAffine(v.Initializer.Value, out var aoff))
                    _affineOffset[v.Identifier.Text] = aoff;
                else
                    _affineOffset.Remove(v.Identifier.Text);

                // An affine int/long local used only as a contiguous buffer index never needs a vector
                //, every use lowers to the affine offset directly, so its declaration would be dead.
                // Track it (for the affine offset) but skip emitting the VInt/VLong.
                if (_affineOffset.ContainsKey(v.Identifier.Text) &&
                    AllOccurrencesAreContiguousIndices(v.Identifier.Text, decl))
                    continue;

                string init = v.Initializer != null
                    ? Coerce(Vec(v.Initializer.Value), kind, v.Initializer.Value)
                    : $"{vtype}.Zero";
                // If inside a masked branch, blend the initializer with zero for inactive lanes.
                if (mask != null)
                    sb.AppendLine($"{indent}{vtype} {v.Identifier.Text} = {SelectOf(kind)}({mask}, {init}, {vtype}.Zero);");
                else
                    sb.AppendLine($"{indent}{vtype} {v.Identifier.Text} = {init};");
            }
        }

        /// <summary>True if every occurrence of an affine local is a contiguous 1-D buffer index
        /// (so the local never needs to be materialized as a vector). Conservative: any other use
        ///, arithmetic, a store value, a gather/2-D index, another local's initializer, returns false.</summary>
        private bool AllOccurrencesAreContiguousIndices(string name, LocalDeclarationStatementSyntax decl)
        {
            if (decl.Parent is not { } scope) return false;
            foreach (var id in scope.DescendantNodes().OfType<IdentifierNameSyntax>())
            {
                if (id.Identifier.Text != name) continue;
                if (!IsContiguousIndexOccurrence(id)) return false;
            }
            return true;   // all occurrences (or none) are contiguous indices → the VInt is dead
        }

        private bool IsContiguousIndexOccurrence(IdentifierNameSyntax id)
        {
            for (SyntaxNode? a = id.Parent; a != null; a = a.Parent)
            {
                if (a is ElementAccessExpressionSyntax ea)
                {
                    if (ea.Expression.Span.Contains(id.Span)) return false;   // it's the buffer name, not the index
                    return ea.ArgumentList.Arguments.Count == 1 &&
                           TryGetContiguousIndex(ea, out var buf, out _) &&
                           _params.TryGetValue(buf, out var p) && p.IsBuffer;
                }
                if (a is StatementSyntax) return false;   // hit a statement with no enclosing index → a value use
            }
            return false;
        }

        private void EmitAssignment(AssignmentExpressionSyntax asg, string? mask, string indent, StringBuilder sb)
        {
            _exprMask = mask;
            string op = asg.OperatorToken.Text;

            // Whole-struct assignment: 'v = <struct expr>;'
            if (asg.Left is IdentifierNameSyntax sid && _structLocals.TryGetValue(sid.Identifier.Text, out var sst))
            {
                if (op != "=")
                    throw new SpmdGenerator.UnsupportedConstructException($"compound '{op}' on struct '{sid.Identifier.Text}'", asg.GetLocation());
                string rhs = VecStruct(asg.Right).Code;
                sb.AppendLine(mask == null
                    ? $"{indent}{sid.Identifier.Text} = {rhs};"
                    : $"{indent}{sid.Identifier.Text} = {VName(sst)}.Select({mask}, {rhs}, {sid.Identifier.Text});");
                return;
            }

            // Struct field assignment: 'v.field = e;' (struct local) or 'buf[i].field = e;' (struct buffer).
            if (asg.Left is MemberAccessExpressionSyntax fma)
            {
                if (fma.Expression is IdentifierNameSyntax fid && _structLocals.TryGetValue(fid.Identifier.Text, out var fst))
                {
                    var k = FieldKind(fst, fma.Name.Identifier.Text, fma);
                    string target = $"{fid.Identifier.Text}.{fma.Name.Identifier.Text}";
                    string rhs = CombineCompoundOnExpr(op, target, asg, k);
                    sb.AppendLine(mask == null
                        ? $"{indent}{target} = {rhs};"
                        : $"{indent}{target} = {SelectOf(k)}({mask}, {rhs}, {target});");
                    return;
                }
                if (fma.Expression is ElementAccessExpressionSyntax fea && TryStructBufferField(fea, out var bn, out var bs, out var bidx))
                {
                    EmitStructFieldStore(bn, bs, fma.Name.Identifier.Text, bidx, asg, op, mask, indent, sb);
                    return;
                }
            }

            // Whole-struct store to a buffer element: 'buf[i] = <struct expr>;'
            if (asg.Left is ElementAccessExpressionSyntax sea && TryStructBufferField(sea, out var sbuf, out var sbs, out var sbidx))
            {
                if (op != "=")
                    throw new SpmdGenerator.UnsupportedConstructException($"compound '{op}' on struct-buffer element '{sbuf}[...]'", asg.GetLocation());
                if (_structs[sbs].Fields.Count > 1) Warn(SpmdGenerator.ScatterPerf, sea, sea.ToString());
                string tmp = $"__sv{_maskCounter++}";
                sb.AppendLine($"{indent}var {tmp} = {VecStruct(asg.Right).Code};");
                foreach (var f in _structs[sbs].Fields)
                    EmitStructFieldStoreRaw(sbuf, sbs, f.Name, sbidx, $"{tmp}.{f.Name}", mask, indent, sb);
                return;
            }

            if (asg.Left is IdentifierNameSyntax rid && _reductions.TryGetValue(rid.Identifier.Text, out var rinfo))
            {
                string rname = rid.Identifier.Text;
                var rkind = rinfo.LaneKind;
                string sel = SelectOf(rkind);

                if (rinfo.Op == SpmdGenerator.ReduceOp.Add)
                {
                    if (op != "+=")
                        throw new SpmdGenerator.UnsupportedConstructException(
                            $"reduction '{rname}' with operator '{op}' (this accumulator uses '+=')", asg.GetLocation());
                    // 'acc += x * y' on float/double lanes fuses into one FMA (matmul / dot product /
                    // sum of squares): higher throughput and a single rounding. Only when no operand
                    // needs a precision-changing coercion, so it still tracks the scalar reference.
                    if (mask == null && TryFmaFactors(asg.Right, rkind, out var fa, out var fb))
                    {
                        sb.AppendLine($"{indent}__red_{rname} = {VType(rkind)}.MulAdd({fa}, {fb}, __red_{rname});");
                        return;
                    }
                    string rhs = Coerce(Vec(asg.Right), rkind, asg.Right);
                    string zero = $"{VType(rkind)}.Zero";
                    sb.AppendLine(mask == null
                        ? $"{indent}__red_{rname} += {rhs};"
                        : $"{indent}__red_{rname} += {sel}({mask}, {rhs}, {zero});");
                    return;
                }

                // Min/Max: FindReductions guarantees the shape 'acc = Math.Min(acc, expr)'
                // (or Max, or with the args swapped); vectorize the non-accumulator argument.
                if (op != "=" ||
                    asg.Right is not InvocationExpressionSyntax minMaxCall ||
                    !SpmdGenerator.IsMinMaxSelfCall(minMaxCall, rname, out var callOp) ||
                    callOp != rinfo.Op)
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"reduction '{rname}' accumulation (this accumulator uses '{rname} = Math.{rinfo.Op}({rname}, ...)')", asg.GetLocation());

                var otherArg = minMaxCall.ArgumentList.Arguments
                    .First(a => a.Expression is not IdentifierNameSyntax id || id.Identifier.Text != rname)
                    .Expression;
                string val = Coerce(Vec(otherArg), rkind, otherArg);
                string fn = rinfo.Op == SpmdGenerator.ReduceOp.Min ? "Min" : "Max";
                string identity = SpmdGenerator.VectorIdentity(rinfo, _d, _l);
                sb.AppendLine(mask == null
                    ? $"{indent}__red_{rname} = VectorMath.{fn}(__red_{rname}, {val});"
                    : $"{indent}__red_{rname} = VectorMath.{fn}(__red_{rname}, {sel}({mask}, {val}, {identity}));");
                return;
            }

            // Local variable
            if (asg.Left is IdentifierNameSyntax id && _locals.TryGetValue(id.Identifier.Text, out var lkind))
            {
                string name = id.Identifier.Text;
                // Reassignment can change (or destroy) the affine-in-lane property.
                if (lkind == SpmdGenerator.Kind.I && op == "=" && mask == null &&
                    TryClassifyAffine(asg.Right, out var aoff))
                    _affineOffset[name] = aoff;
                else
                    _affineOffset.Remove(name);

                string rhs = CombineCompound(op, name, asg, lkind);
                sb.AppendLine(mask == null
                    ? $"{indent}{name} = {rhs};"
                    : $"{indent}{name} = {SelectOf(lkind)}({mask}, {rhs}, {name});");
                return;
            }

            // 2-D array store: a[i, j] = ... on a float[,]/int[,]/double[,] param.
            if (asg.Left is ElementAccessExpressionSyntax ea2d && ea2d.ArgumentList.Arguments.Count == 2 &&
                ea2d.Expression is IdentifierNameSyntax a2d &&
                _params.TryGetValue(a2d.Identifier.Text, out var p2d) && p2d.Is2D)
            {
                Emit2DStore(ea2d, p2d, asg, op, mask, indent, sb);
                return;
            }

            // Buffer store: buf[i] / buf[uniform + i] = ... (contiguous gang store)
            if (asg.Left is ElementAccessExpressionSyntax ea && TryGetContiguousIndex(ea, out var bufName, out var storeIdx))
            {
                if (!_params.TryGetValue(bufName, out var p) || !p.IsBuffer)
                    throw new SpmdGenerator.UnsupportedConstructException($"store to unknown buffer '{bufName}'", asg.GetLocation());
                if (p.IsReadOnly)
                    throw new SpmdGenerator.UnsupportedConstructException($"store to ReadOnlySpan '{bufName}'", asg.GetLocation());

                var ek = p.ElemKind;
                string vtype = VType(ek);
                string load = $"{vtype}.Load({bufName}, {storeIdx})";
                string rhs = CombineCompoundOnExpr(op, load, asg, ek);
                // Streaming: a full-gang, non-compound contiguous write to a float/double buffer can use a
                // non-temporal store (cache-bypassing). Masked/compound/other paths keep the ordinary store.
                string store = _streaming && op == "=" && ek is SpmdGenerator.Kind.F or SpmdGenerator.Kind.D
                    ? "StoreNonTemporal" : "Store";
                if (mask == null)
                    sb.AppendLine($"{indent}({rhs}).{store}({bufName}, {storeIdx});");
                else
                    sb.AppendLine($"{indent}{vtype}.Select({mask}, {rhs}, {load}).Store({bufName}, {storeIdx});");
                return;
            }

            // Scatter: buf[non-loop-index] = ... (indexed store)
            if (asg.Left is ElementAccessExpressionSyntax eaScatter &&
                eaScatter.Expression is IdentifierNameSyntax scatterId &&
                _params.TryGetValue(scatterId.Identifier.Text, out var scatterParam) && scatterParam.IsBuffer)
            {
                if (scatterParam.IsReadOnly)
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"scatter to ReadOnlySpan '{scatterId.Identifier.Text}'", asg.GetLocation());
                if (op != "=")
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"compound scatter '{op}' on '{scatterId.Identifier.Text}' (only '=' supported for indexed stores)", asg.GetLocation());

                Warn(SpmdGenerator.ScatterPerf, eaScatter, eaScatter.ToString());
                // Lower: Memory.Scatter(buf, indices, values, mask).
                // Long kernels index with VLong directly; double kernels truncate to VLong.
                var ek = scatterParam.ElemKind;
                var idxExpr = eaScatter.ArgumentList.Arguments[0].Expression;
                var val = Coerce(Vec(asg.Right), ek, asg.Right);
                string idx = _l
                    ? Coerce(Vec(idxExpr), SpmdGenerator.Kind.L, idxExpr)
                    : _d
                    ? $"VLong.FromDoubleTruncate({Coerce(Vec(idxExpr), SpmdGenerator.Kind.D, idxExpr)})"
                    : Coerce(Vec(idxExpr), SpmdGenerator.Kind.I, idxExpr);
                string maskArg = mask ?? MTAll;
                sb.AppendLine($"{indent}Memory.Scatter({scatterId.Identifier.Text}, {idx}, {val}, {maskArg});");
                return;
            }

            throw new SpmdGenerator.UnsupportedConstructException($"assignment target '{Trunc(asg.Left)}'", asg.GetLocation());
        }

        /// <summary>
        /// Per-lane 'return', ISPC semantics: the lane retires for its current element,
        /// exiting every enclosing varying loop and skipping the rest of the foreach body;
        /// other elements keep processing. (Scalar C# 'return' would stop ALL later elements —
        /// that global semantic is not vectorizable, so kernels using 'return' diverge from
        /// their scalar reference once a return fires; use 'continue' for exact parity.)
        /// Lowered by OR-ing the lane set into every enclosing varying loop's break mask,
        /// including the foreach-level mask, so existing mask threading does the rest.
        /// </summary>
        private void EmitReturn(ReturnStatementSyntax rs, string? mask, string indent, StringBuilder sb)
        {
            if (rs.Expression != null)
                throw new SpmdGenerator.UnsupportedConstructException(
                    "'return <value>' inside the loop (value returns belong after the loop; use a bare 'return;' to retire the lane)",
                    rs.GetLocation());
            if (_loopStack.Any(ctx => ctx.IsUniform))
                throw new SpmdGenerator.UnsupportedConstructException(
                    "'return' inside a uniform for loop (per-lane returns can't exit a shared scalar loop; use a varying loop or restructure)",
                    rs.GetLocation());

            string activeMask = mask ?? MTAll;
            foreach (var ctx in _loopStack)
                sb.AppendLine($"{indent}{ctx.BreakMask} |= {activeMask};");
        }

        /// <summary>Lower 'x++' / '--x' statements on lane locals to masked 'x = x ± 1'.</summary>
        private void EmitIncDec(ExpressionSyntax operand, string op, string? mask, string indent, StringBuilder sb, Location loc)
        {
            if (operand is not IdentifierNameSyntax id || !_locals.TryGetValue(id.Identifier.Text, out var kind))
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"'{op}' on '{Trunc(operand)}' (only lane locals support ++/-- inside the loop body)", loc);

            string name = id.Identifier.Text;
            _affineOffset.Remove(name);   // ++/-- keeps affine-ness but conservatively drop it
            string one = kind switch
            {
                SpmdGenerator.Kind.F => "new VFloat(1f)",
                SpmdGenerator.Kind.D => $"new {VType(SpmdGenerator.Kind.D)}(1d)",
                SpmdGenerator.Kind.L => $"new {VType(SpmdGenerator.Kind.L)}(1)",
                _ => "new VInt(1)",
            };
            string rhs = $"({name} {(op == "++" ? "+" : "-")} {one})";
            sb.AppendLine(mask == null
                ? $"{indent}{name} = {rhs};"
                : $"{indent}{name} = {SelectOf(kind)}({mask}, {rhs}, {name});");
        }

        private string CombineCompound(string op, string currentName, AssignmentExpressionSyntax asg, SpmdGenerator.Kind kind)
            => CombineCompoundOnExpr(op, currentName, asg, kind);

        private string CombineCompoundOnExpr(string op, string current, AssignmentExpressionSyntax asg, SpmdGenerator.Kind kind)
        {
            string rhs = Coerce(Vec(asg.Right), kind, asg.Right);
            bool isInt = kind is SpmdGenerator.Kind.I or SpmdGenerator.Kind.L;
            return op switch
            {
                "=" => rhs,
                "+=" => $"({current} + {rhs})",
                "-=" => $"({current} - {rhs})",
                "*=" => $"({current} * {rhs})",
                "/=" when !isInt => $"({current} / {rhs})",
                "/=" => $"{VType(kind)}.Divide({current}, {GuardedDivisor(rhs, kind)})",
                "&=" when isInt => $"({current} & {rhs})",
                "|=" when isInt => $"({current} | {rhs})",
                "^=" when isInt => $"({current} ^ {rhs})",
                _ => throw new SpmdGenerator.UnsupportedConstructException(
                    $"assignment operator '{op}' on {kind} lanes", asg.GetLocation()),
            };
        }

        private void EmitIf(IfStatementSyntax ifs, string? mask, string indent, StringBuilder sb)
        {
            // Coherent if, ISPC's 'cif'. 'if (Spmd.Coherent(cond))' asks for an Any() guard
            // around each branch so whole gangs skip a branch no lane takes.
            bool coherent = TryUnwrapCoherent(ifs.Condition, out var condition);

            // If the condition is uniform (no lane variables) and we're inside a uniform
            // loop context, emit a plain C# if/else, break/continue inside need C# semantics.
            bool inUniformLoop = _loopStack.Count > 0 && _loopStack.Peek().IsUniform;
            if (inUniformLoop && IsUniformExpression(condition))
            {
                string condExpr = condition.ToString();
                sb.AppendLine($"{indent}if ({condExpr})");
                sb.AppendLine($"{indent}{{");
                EmitStatement(ifs.Statement, mask, indent + "    ", sb);
                sb.AppendLine($"{indent}}}");
                if (ifs.Else != null)
                {
                    sb.AppendLine($"{indent}else");
                    sb.AppendLine($"{indent}{{");
                    EmitStatement(ifs.Else.Statement, mask, indent + "    ", sb);
                    sb.AppendLine($"{indent}}}");
                }
                return;
            }

            string m = $"__m{_maskCounter++}";
            string cond = Mask(condition);
            sb.AppendLine(mask == null
                ? $"{indent}{MT} {m} = {cond};"
                : $"{indent}{MT} {m} = ({cond}) & {mask};");

            // Every statement in the branch is blended by the branch mask, so skipping the
            // whole branch when no lane is active is a pure optimization, never a semantic change.
            if (coherent)
            {
                sb.AppendLine($"{indent}if ({m}.Any())");
                sb.AppendLine($"{indent}{{");
                EmitStatement(ifs.Statement, m, indent + "    ", sb);
                sb.AppendLine($"{indent}}}");
            }
            else
            {
                EmitStatement(ifs.Statement, m, indent, sb);
            }

            if (ifs.Else != null)
            {
                string me = $"__m{_maskCounter++}";
                sb.AppendLine(mask == null
                    ? $"{indent}{MT} {me} = !{m};"
                    : $"{indent}{MT} {me} = (!{m}) & {mask};");
                if (coherent)
                {
                    sb.AppendLine($"{indent}if ({me}.Any())");
                    sb.AppendLine($"{indent}{{");
                    EmitStatement(ifs.Else.Statement, me, indent + "    ", sb);
                    sb.AppendLine($"{indent}}}");
                }
                else
                {
                    EmitStatement(ifs.Else.Statement, me, indent, sb);
                }
            }
        }

        /// <summary>Recognize 'Spmd.Coherent(cond)' and unwrap to the inner condition.</summary>
        private static bool TryUnwrapCoherent(ExpressionSyntax e, out ExpressionSyntax condition)
        {
            if (e is InvocationExpressionSyntax inv &&
                inv.ArgumentList.Arguments.Count == 1 &&
                inv.Expression.ToString().Replace(" ", "") is "Spmd.Coherent" or "IspcSharp.Spmd.Coherent")
            {
                condition = inv.ArgumentList.Arguments[0].Expression;
                return true;
            }
            condition = e;
            return false;
        }

        /// <summary>
        /// Divergent while: iterate until no lane's condition holds. Writes inside the body are
        /// blended by the shrinking loop mask, so exited lanes keep their values frozen.
        /// break/continue are supported via mask manipulation.
        /// </summary>
        private void EmitWhile(WhileStatementSyntax whs, string? mask, string indent, StringBuilder sb)
        {
            string m = $"__loop{_maskCounter++}";
            string cond = Mask(whs.Condition);
            sb.AppendLine(mask == null
                ? $"{indent}{MT} {m} = {cond};"
                : $"{indent}{MT} {m} = ({cond}) & {mask};");

            // Set up break/continue context for this loop.
            string breakMask = $"__brk{_maskCounter}";
            string contMask = $"__cnt{_maskCounter}";
            _maskCounter++;
            sb.AppendLine($"{indent}{MT} {breakMask} = {MTNone};");
            sb.AppendLine($"{indent}{MT} {contMask} = {MTNone};");
            _loopStack.Push(new LoopContext(breakMask, contMask, m));

            sb.AppendLine($"{indent}while ({m}.Any())");
            sb.AppendLine($"{indent}{{");
            sb.AppendLine($"{indent}    {contMask} = {MTNone};");
            // Emit the body through EmitBlock so break/continue masks are threaded.
            if (whs.Statement is BlockSyntax whileBlock)
                EmitBlock(whileBlock, m, indent + "    ", sb);
            else
                EmitStatement(whs.Statement, m, indent + "    ", sb);
            // Continue mask: lanes that 'continue'd rejoin for next iteration.
            // Break mask: lanes that 'break'd are removed from the loop mask.
            _exprMask = m;   // loop-epilogue condition re-check runs under the loop mask
            sb.AppendLine($"{indent}    {m} = ({m} & ({Mask(whs.Condition)})) & !{breakMask};");
            sb.AppendLine($"{indent}}}");

            _loopStack.Pop();
        }

        /// <summary>
        /// Uniform for loop: the loop variable is a plain int (not a lane), so this emits
        /// a normal C# for loop around the vectorized body. break/continue inside map to
        /// C# break/continue (they affect the uniform loop, not per-lane masks).
        ///
        /// If the for loop has a varying condition (depends on lane variables), it is lowered
        /// as a divergent while loop instead.
        /// </summary>
        private void EmitFor(ForStatementSyntax fors, string? mask, string indent, StringBuilder sb)
        {
            // Register for-loop variables as uniform ints so Vec() broadcasts them.
            if (fors.Declaration != null)
            {
                string varType = fors.Declaration.Type.ToString();
                if (varType != "int")
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"for-loop variable of type '{varType}' (only 'int' uniform loop variables supported)",
                        fors.Declaration.GetLocation());
                foreach (var v in fors.Declaration.Variables)
                {
                    _uniformPreLocals.Add(v.Identifier.Text);
                    if (!_preLocalKinds.ContainsKey(v.Identifier.Text))
                        _preLocalKinds[v.Identifier.Text] = SpmdGenerator.Kind.I;
                }
            }

            // Check if the condition is uniform (no lane variables).
            bool isUniform = IsUniformExpression(fors.Condition);

            // Even if the loop condition is uniform, if break/continue inside the body
            // are guarded by varying conditions (references to lane variables), we must
            // lower as a divergent loop, C# break/continue can't be per-lane.
            if (isUniform && HasVaryingBreakContinue(fors))
                isUniform = false;

            if (isUniform)
            {
                // Uniform for loop: emit as a plain C# for loop so that
                // break/continue map to C# keywords (continue jumps to the
                // incrementor, which is the correct for-loop semantics).
                string condExpr = fors.Condition?.ToString() ?? "true";
                string inits = fors.Declaration != null
                    ? fors.Declaration.ToString()
                    : string.Join(", ", fors.Initializers);
                string incs = string.Join(", ", fors.Incrementors);
                sb.AppendLine($"{indent}for ({inits}; {condExpr}; {incs})");
                sb.AppendLine($"{indent}{{");

                // Push a uniform loop context (break/continue use C# keywords).
                _loopStack.Push(LoopContext.Uniform());

                var body = fors.Statement is BlockSyntax b ? b.Statements : new SyntaxList<StatementSyntax>().Add(fors.Statement);
                foreach (var st in body)
                    EmitStatement(st, mask, indent + "    ", sb);

                _loopStack.Pop();
                sb.AppendLine($"{indent}}}");
            }
            else
            {
                // Varying for loop: lower as a divergent while loop.
                // The condition depends on lane variables, so we need mask iteration.

                // Emit the initializer before the while loop.
                if (fors.Declaration != null)
                {
                    foreach (var v in fors.Declaration.Variables)
                    {
                        string init = v.Initializer != null ? v.Initializer.Value.ToString() : "0";
                        sb.AppendLine($"{indent}int {v.Identifier.Text} = {init};");
                    }
                }
                else if (fors.Initializers.Count > 0)
                {
                    foreach (var init in fors.Initializers)
                        sb.AppendLine($"{indent}{init};");
                }

                string m = $"__loop{_maskCounter++}";
                string cond = Mask(fors.Condition!);
                sb.AppendLine(mask == null
                    ? $"{indent}{MT} {m} = {cond};"
                    : $"{indent}{MT} {m} = ({cond}) & {mask};");

                string breakMask = $"__brk{_maskCounter}";
                string contMask = $"__cnt{_maskCounter}";
                _maskCounter++;
                sb.AppendLine($"{indent}{MT} {breakMask} = {MTNone};");
                sb.AppendLine($"{indent}{MT} {contMask} = {MTNone};");
                _loopStack.Push(new LoopContext(breakMask, contMask, m));

                sb.AppendLine($"{indent}while ({m}.Any())");
                sb.AppendLine($"{indent}{{");
                sb.AppendLine($"{indent}    {contMask} = {MTNone};");

                // Emit the body through EmitBlock so break/continue masks are threaded.
                if (fors.Statement is BlockSyntax forBlock)
                    EmitBlock(forBlock, m, indent + "    ", sb);
                else
                    EmitStatement(fors.Statement, m, indent + "    ", sb);

                // Emit incrementors (they run at the end of each iteration).
                // Lane-local ++/-- gets masked lowering; scalar (uniform) incrementors emit verbatim.
                foreach (var inc in fors.Incrementors)
                {
                    if (inc is AssignmentExpressionSyntax asg)
                        EmitAssignment(asg, m, indent + "    ", sb);
                    else if (inc is PostfixUnaryExpressionSyntax { Operand: IdentifierNameSyntax pid } post &&
                             post.OperatorToken.Text is "++" or "--" && _locals.ContainsKey(pid.Identifier.Text))
                        EmitIncDec(post.Operand, post.OperatorToken.Text, m, indent + "    ", sb, post.GetLocation());
                    else if (inc is PrefixUnaryExpressionSyntax { Operand: IdentifierNameSyntax nid } pre &&
                             pre.OperatorToken.Text is "++" or "--" && _locals.ContainsKey(nid.Identifier.Text))
                        EmitIncDec(pre.Operand, pre.OperatorToken.Text, m, indent + "    ", sb, pre.GetLocation());
                    else
                        sb.AppendLine($"{indent}    {inc};");
                }

                _exprMask = m;   // loop-epilogue condition re-check runs under the loop mask
                sb.AppendLine($"{indent}    {m} = ({m} & ({Mask(fors.Condition!)})) & !{breakMask};");
                sb.AppendLine($"{indent}}}");

                _loopStack.Pop();
            }
        }

        private void EmitBreak(string? mask, string indent, StringBuilder sb, Location loc)
        {
            if (_loopStack.Count == 0)
                throw new SpmdGenerator.UnsupportedConstructException(
                    "'break' outside of a loop", loc);

            var ctx = _loopStack.Peek();
            if (ctx.IsUniform)
            {
                sb.AppendLine($"{indent}break;");
            }
            else
            {
                // Set the break mask for active lanes in the current mask.
                string activeMask = mask ?? MTAll;
                sb.AppendLine($"{indent}{ctx.BreakMask} |= {activeMask};");
            }
        }

        private void EmitContinue(string? mask, string indent, StringBuilder sb, Location loc)
        {
            if (_loopStack.Count == 0)
                throw new SpmdGenerator.UnsupportedConstructException(
                    "'continue' outside of a loop", loc);

            var ctx = _loopStack.Peek();
            if (ctx.IsUniform)
            {
                sb.AppendLine($"{indent}continue;");
            }
            else
            {
                // Set the continue mask and deactivate lanes for the rest of this iteration.
                string activeMask = mask ?? MTAll;
                sb.AppendLine($"{indent}{ctx.ContinueMask} |= {activeMask};");
                // We can't actually 'skip' the rest of the body in the generated code,
                // so we rely on the mask being passed to subsequent statements.
                // The caller (EmitStatement for blocks) will use the updated mask.
                // However, since we process statements sequentially, we need a way to
                // communicate that the mask changed. The simplest approach: emit a
                // comment and let the loop re-check the mask at the top.
                // For correctness, statements after 'continue' in the same block will
                // still execute but with the continued lanes masked out, IF the mask
                // is threaded through. Since we emit inline, we need to re-derive the
                // mask. The cleanest way: wrap the rest of the block in a conditional.
                // For now, we use the approach of AND-ing the current mask with !continue.
                // This is handled by the caller seeing the continue mask in the loop epilogue.
            }
        }

        /// <summary>
        /// Check if an expression only references uniform variables (parameters, pre-locals,
        /// and uniform for-loop variables), not lane locals or the loop variable.
        /// </summary>
        private bool IsUniformExpression(ExpressionSyntax e)
        {
            foreach (var id in e.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>())
            {
                string name = id.Identifier.Text;
                // Lane local → varying.
                if (_locals.ContainsKey(name)) return false;
                // The foreach loop variable → varying.
                if (name == _loopVar) return false;
                // If it's not a known uniform, assume varying.
                if (!_uniformPreLocals.Contains(name) &&
                    !_params.ContainsKey(name) &&
                    !_preLocalKinds.ContainsKey(name))
                {
                    // Could be a type name (MathF, etc.), check if parent is member access.
                    if (id.Parent is MemberAccessExpressionSyntax)
                        continue;
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Check if a for-loop body contains break/continue inside a varying if-condition.
        /// If so, the loop must be lowered as divergent (per-lane break/continue).
        /// </summary>
        private bool HasVaryingBreakContinue(ForStatementSyntax fors)
        {
            // Walk the body and find every break/continue, then check if any enclosing
            // if-statement has a varying condition.
            var body = fors.Statement;
            foreach (var node in body.DescendantNodesAndSelf())
            {
                if (node is not (BreakStatementSyntax or ContinueStatementSyntax)) continue;

                // Walk up the tree to find enclosing if-statements.
                for (var parent = node.Parent; parent != null; parent = parent.Parent)
                {
                    if (parent is IfStatementSyntax ifs && !IsUniformExpression(ifs.Condition))
                        return true;
                    // Stop at the for-loop itself, we only care about ifs inside it.
                    if (parent == fors) break;
                }
            }
            return false;
        }

        private (string Code, SpmdGenerator.Kind Kind) Vec(ExpressionSyntax e) => e switch
        {
            ParenthesizedExpressionSyntax p => WrapParen(Vec(p.Expression)),

            LiteralExpressionSyntax lit => LitExpr(lit),

            CastExpressionSyntax cast => CastExpr(cast),

            PrefixUnaryExpressionSyntax { OperatorToken.Text: "-" } neg
                => Neg(Vec(neg.Operand)),

            PrefixUnaryExpressionSyntax { OperatorToken.Text: "~" } inv
                => BitNot(Vec(inv.Operand), inv),

            IdentifierNameSyntax id when _locals.TryGetValue(id.Identifier.Text, out var k)
                => (id.Identifier.Text, k),

            IdentifierNameSyntax id when _uniformPreLocals.Contains(id.Identifier.Text)
                => Broadcast(id.Identifier.Text, _l ? SpmdGenerator.Kind.L : _d ? SpmdGenerator.Kind.D : _preLocalKinds[id.Identifier.Text]),

            IdentifierNameSyntax id when _params.TryGetValue(id.Identifier.Text, out var p) &&
                                         p.PKind is SpmdGenerator.ParamKind.UniformFloat
                                                 or SpmdGenerator.ParamKind.UniformInt
                                                 or SpmdGenerator.ParamKind.UniformDouble
                                                 or SpmdGenerator.ParamKind.UniformLong
                => Broadcast(id.Identifier.Text, _l
                        ? SpmdGenerator.Kind.L
                        : _d
                        ? SpmdGenerator.Kind.D
                        : p.PKind switch
                        {
                            SpmdGenerator.ParamKind.UniformInt => SpmdGenerator.Kind.I,
                            SpmdGenerator.ParamKind.UniformDouble => SpmdGenerator.Kind.D,
                            SpmdGenerator.ParamKind.UniformLong => SpmdGenerator.Kind.L,
                            _ => SpmdGenerator.Kind.F,
                        }),

            IdentifierNameSyntax id when id.Identifier.Text == _loopVar
                => _l
                    ? ("(VLong.ProgramIndex + __i)", SpmdGenerator.Kind.L)
                    : _d
                    ? ("(VDouble.ProgramIndex + (double)__i)", SpmdGenerator.Kind.D)
                    : ("(VInt.ProgramIndex + __i)", SpmdGenerator.Kind.I),

            // Struct field access: 'v.field' on a struct local, or 'buf[i].field' on a struct buffer.
            MemberAccessExpressionSyntax sfa when TryStructFieldRead(sfa, out var sfr) => sfr,

            // Known scalar constants: MathF.PI / Math.PI / float.MaxValue / int.MinValue / ...
            MemberAccessExpressionSyntax cma when TryConstMember(cma, out var cc) => cc,

            // Uniform '.Length' on a buffer/array param → broadcast int.
            MemberAccessExpressionSyntax lma when lma.Name.Identifier.Text == "Length" && IsUniformExpression(lma)
                => _d ? ($"new VDouble({lma})", SpmdGenerator.Kind.D) : ($"new VInt({lma})", SpmdGenerator.Kind.I),

            // 2-D array access a[i, j] on a float[,]/int[,]/double[,] param.
            ElementAccessExpressionSyntax ea2 when ea2.ArgumentList.Arguments.Count == 2 &&
                ea2.Expression is IdentifierNameSyntax a2 &&
                _params.TryGetValue(a2.Identifier.Text, out var p2) && p2.Is2D
                => Emit2DLoad(ea2, p2),

            ElementAccessExpressionSyntax ea when TryGetContiguousIndex(ea, out var buf, out var idx) &&
                                                  _params.TryGetValue(buf, out var p) && p.IsBuffer
                => ($"{VType(p.ElemKind)}.Load({buf}, {idx})", p.ElemKind),

            // Gather: buf[non-loop-index], lowered to Memory.Gather(buf, indices).
            ElementAccessExpressionSyntax eaGather when
                eaGather.Expression is IdentifierNameSyntax gatherId &&
                _params.TryGetValue(gatherId.Identifier.Text, out var gatherParam) && gatherParam.IsBuffer
                => EmitGather(eaGather, gatherId.Identifier.Text, gatherParam),

            BinaryExpressionSyntax bin => BinExpr(bin),

            ConditionalExpressionSyntax tern => TernExpr(tern),

            InvocationExpressionSyntax call => VecCall(call),

            _ => throw new SpmdGenerator.UnsupportedConstructException($"expression '{Trunc(e)}' ({e.Kind()})", e.GetLocation()),
        };

        private static (string, SpmdGenerator.Kind) WrapParen((string Code, SpmdGenerator.Kind Kind) inner)
            => ($"({inner.Code})", inner.Kind);

        private static (string, SpmdGenerator.Kind) Neg((string Code, SpmdGenerator.Kind Kind) inner)
            => ($"(-{inner.Code})", inner.Kind);

        private static (string, SpmdGenerator.Kind) BitNot((string Code, SpmdGenerator.Kind Kind) inner, SyntaxNode at)
            => inner.Kind == SpmdGenerator.Kind.I
                ? ($"(~{inner.Code})", SpmdGenerator.Kind.I)
                : throw new SpmdGenerator.UnsupportedConstructException("'~' on non-int lanes", at.GetLocation());

        private (string, SpmdGenerator.Kind) Broadcast(string name, SpmdGenerator.Kind k)
            => ($"new {VType(k)}({name})", k);

        private (string, SpmdGenerator.Kind) LitExpr(LiteralExpressionSyntax lit)
        {
            string text = lit.Token.Text;

            // In a long kernel, every integer literal is a VLong lane; floats are invalid.
            if (_l)
            {
                if (text.EndsWith("f") || text.EndsWith("F") || text.EndsWith("d") || text.EndsWith("D") || text.Contains('.'))
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"floating literal '{text}' in a long kernel", lit.GetLocation());
                return ($"new VLong({text.TrimEnd('L', 'l', 'u', 'U')})", SpmdGenerator.Kind.L);
            }

            // Hex/binary literals are always int lanes ("0xFF" ends in 'F' but is not a float).
            bool isHexOrBinary = text.StartsWith("0x") || text.StartsWith("0X") ||
                                 text.StartsWith("0b") || text.StartsWith("0B");
            if (isHexOrBinary)
                return _d
                    ? ($"new VDouble({text})", SpmdGenerator.Kind.D)
                    : ($"new VInt({text})", SpmdGenerator.Kind.I);

            if (_d)
                return ($"new VDouble({text.TrimEnd('f', 'F', 'd', 'D')})", SpmdGenerator.Kind.D);
            // A 'd'-suffixed literal in a float/int kernel opts that expression into
            // double precision (VDouble2 pairs at full gang width).
            if (text.EndsWith("d") || text.EndsWith("D"))
                return ($"new VDouble2({text.TrimEnd('d', 'D')})", SpmdGenerator.Kind.D);
            // An 'L'-suffixed literal opts into 64-bit integers (VLong2 pairs at full gang width).
            if (text.EndsWith("L") || text.EndsWith("l"))
                return ($"new VLong2({text.TrimEnd('u', 'U')})", SpmdGenerator.Kind.L);
            if (text.EndsWith("f") || text.EndsWith("F") || text.Contains('.'))
                return ($"new VFloat({text.TrimEnd('f', 'F')}f)", SpmdGenerator.Kind.F);
            return ($"new VInt({text})", SpmdGenerator.Kind.I);
        }

        private (string, SpmdGenerator.Kind) CastExpr(CastExpressionSyntax cast)
        {
            var inner = Vec(cast.Expression);
            string t = cast.Type.ToString();
            if (_l)
            {
                // Long kernel: (long)/(int) are identity on the 64-bit integer lane; no floats.
                if (t is "long" or "int") return (inner.Code, SpmdGenerator.Kind.L);
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"cast to '{t}' in a long kernel (only (long)/(int) casts supported)", cast.GetLocation());
            }
            if (_d)
            {
                // Double kernels: every lane value is already VDouble; (double) is the identity,
                // and (int)/(long) truncate toward zero, the double stays the carrier type
                // (used mostly as gather/scatter indices, where VLong truncation reapplies).
                if (t == "double") return (inner.Code, SpmdGenerator.Kind.D);
                if (t is "int" or "long") { Warn(SpmdGenerator.DoubleConvertPerf, cast, cast.ToString()); return ($"VectorMath.Truncate({inner.Code})", SpmdGenerator.Kind.D); }
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"cast to '{t}' in a double kernel (only (double)/(int)/(long) casts supported)", cast.GetLocation());
            }
            // Float/int kernels: cross-gang-width double conversions via VDouble2.
            if (inner.Kind == SpmdGenerator.Kind.D)
            {
                if (t is "int" or "long") Warn(SpmdGenerator.DoubleConvertPerf, cast, cast.ToString());
                return t switch
                {
                    "float" => ($"({inner.Code}).ToFloat()", SpmdGenerator.Kind.F),
                    "int" => ($"({inner.Code}).ToIntTruncate()", SpmdGenerator.Kind.I),
                    "double" => (inner.Code, SpmdGenerator.Kind.D),
                    _ => throw new SpmdGenerator.UnsupportedConstructException($"cast to '{t}'", cast.GetLocation()),
                };
            }
            // ...and cross-gang-width long conversions via VLong2.
            if (inner.Kind == SpmdGenerator.Kind.L)
            {
                return t switch
                {
                    "long" => (inner.Code, SpmdGenerator.Kind.L),
                    "int" => ($"({inner.Code}).ToInt()", SpmdGenerator.Kind.I),
                    "float" => ($"({inner.Code}).ToFloat()", SpmdGenerator.Kind.F),
                    "double" => ($"({inner.Code}).ToDouble2()", SpmdGenerator.Kind.D),
                    _ => throw new SpmdGenerator.UnsupportedConstructException($"cast to '{t}'", cast.GetLocation()),
                };
            }
            return t switch
            {
                "float" => (Coerce(inner, SpmdGenerator.Kind.F, cast), SpmdGenerator.Kind.F),
                "int" => (Coerce(inner, SpmdGenerator.Kind.I, cast), SpmdGenerator.Kind.I),
                "long" => (Coerce(inner, SpmdGenerator.Kind.L, cast), SpmdGenerator.Kind.L),
                "double" => (Coerce(inner, SpmdGenerator.Kind.D, cast), SpmdGenerator.Kind.D),
                _ => throw new SpmdGenerator.UnsupportedConstructException($"cast to '{t}'", cast.GetLocation()),
            };
        }

        private (string, SpmdGenerator.Kind) BinExpr(BinaryExpressionSyntax bin)
        {
            string op = bin.OperatorToken.Text;

            // Shifts on int lanes. Uniform counts use Vector<T> whole-gang shifts; varying
            // (per-lane) counts use VInt.Shift*Variable (AVX2 vpsllvd/vpsravd/vpsrlvd, or a
            // per-lane loop elsewhere). C# semantics: only the low 5 bits of each count matter.
            if (op is "<<" or ">>" or ">>>")
            {
                var sh = Vec(bin.Left);
                if (sh.Kind is not (SpmdGenerator.Kind.I or SpmdGenerator.Kind.L))
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"shift '{op}' on non-integer lanes", bin.GetLocation());
                var shKind = sh.Kind;
                if (IsUniformExpression(bin.Right))
                {
                    string cnt = bin.Right.ToString();
                    return op switch
                    {
                        "<<" => ($"({sh.Code} << {cnt})", shKind),
                        ">>" => ($"({sh.Code} >> {cnt})", shKind),
                        _ => ($"{VType(shKind)}.ShiftRightLogical({sh.Code}, {cnt})", shKind),
                    };
                }
                if (shKind == SpmdGenerator.Kind.L)
                    throw new SpmdGenerator.UnsupportedConstructException(
                        "varying (per-lane) shift count on long lanes (only uniform shift counts on long)", bin.GetLocation());
                string counts = Coerce(Vec(bin.Right), SpmdGenerator.Kind.I, bin.Right);
                return op switch
                {
                    "<<" => ($"VInt.ShiftLeftVariable({sh.Code}, {counts})", SpmdGenerator.Kind.I),
                    ">>" => ($"VInt.ShiftRightArithmeticVariable({sh.Code}, {counts})", SpmdGenerator.Kind.I),
                    _ => ($"VInt.ShiftRightLogicalVariable({sh.Code}, {counts})", SpmdGenerator.Kind.I),
                };
            }

            var l = Vec(bin.Left);
            var r = Vec(bin.Right);

            // Implicit int->long->float->double promotion when kinds are mixed (mirrors C#).
            var kind = Promote(l.Kind, r.Kind);
            string lc = CoerceCode(l, kind, bin.Left);
            string rc = CoerceCode(r, kind, bin.Right);
            bool isInt = kind is SpmdGenerator.Kind.I or SpmdGenerator.Kind.L;

            // Integer / and % have no SIMD instruction, a per-lane scalar loop. Flag it unless the
            // divisor is a compile-time constant (the JIT can then strength-reduce; still note it).
            if (isInt && op is "/" or "%")
                Warn(SpmdGenerator.IntDividePerf, bin, bin.ToString());

            // 'a * b + c' on float/double lanes fuses into one FMA (higher throughput, single rounding).
            if (op == "+")
            {
                if (TryFmaFactors(bin.Left, kind, out var fla, out var flb))
                    return ($"{VType(kind)}.MulAdd({fla}, {flb}, {rc})", kind);
                if (TryFmaFactors(bin.Right, kind, out var fra, out var frb))
                    return ($"{VType(kind)}.MulAdd({fra}, {frb}, {lc})", kind);
            }

            return op switch
            {
                "+" or "-" or "*" => ($"({lc} {op} {rc})", kind),
                "/" when !isInt => ($"({lc} / {rc})", kind),
                // Integer / and %: per-lane scalar loops. The divisor gets 1 selected into
                // inactive lanes so masked-off lanes can't raise DivideByZeroException.
                "/" => ($"{VType(kind)}.Divide({lc}, {GuardedDivisor(rc, kind)})", kind),
                "%" when isInt => ($"{VType(kind)}.Remainder({lc}, {GuardedDivisor(rc, kind)})", kind),
                "%" => throw new SpmdGenerator.UnsupportedConstructException(
                    "'%' on float/double lanes (use int/long lanes, or x - Floor(x/y)*y)", bin.GetLocation()),
                "&" or "|" or "^" when isInt => ($"({lc} {op} {rc})", kind),
                _ => throw new SpmdGenerator.UnsupportedConstructException($"operator '{op}'", bin.GetLocation()),
            };
        }

        /// <summary>Recognize a fusable <c>x * y</c> on float/double lanes where neither factor needs a
        /// precision-changing coercion (both already the target kind), so <c>MulAdd</c> still tracks the
        /// scalar reference. Yields the two factor codes.</summary>
        private bool TryFmaFactors(ExpressionSyntax e, SpmdGenerator.Kind target, out string a, out string b)
        {
            a = b = "";
            if (target is not (SpmdGenerator.Kind.F or SpmdGenerator.Kind.D)) return false;
            if (Unparen(e) is not BinaryExpressionSyntax { OperatorToken.Text: "*" } mul) return false;
            var la = Vec(mul.Left);
            var lb = Vec(mul.Right);
            if (la.Kind != target || lb.Kind != target) return false;
            a = la.Code;
            b = lb.Code;
            return true;
        }

        private string GuardedDivisor(string divisor, SpmdGenerator.Kind kind)
            => _exprMask == null ? divisor : $"{VType(kind)}.Select({_exprMask}, {divisor}, {VType(kind)}.One)";

        private static SpmdGenerator.Kind Promote(SpmdGenerator.Kind a, SpmdGenerator.Kind b)
        {
            if (a == SpmdGenerator.Kind.D || b == SpmdGenerator.Kind.D) return SpmdGenerator.Kind.D;
            if (a == SpmdGenerator.Kind.F || b == SpmdGenerator.Kind.F) return SpmdGenerator.Kind.F;
            if (a == SpmdGenerator.Kind.L || b == SpmdGenerator.Kind.L) return SpmdGenerator.Kind.L;
            return SpmdGenerator.Kind.I;
        }

        private (string, SpmdGenerator.Kind) TernExpr(ConditionalExpressionSyntax tern)
        {
            var t = Vec(tern.WhenTrue);
            var f = Vec(tern.WhenFalse);
            var kind = Promote(t.Kind, f.Kind);
            return ($"{SelectOf(kind)}({Mask(tern.Condition)}, {CoerceCode(t, kind, tern.WhenTrue)}, {CoerceCode(f, kind, tern.WhenFalse)})", kind);
        }

        /// <summary>Lane kind of a struct field.</summary>
        private SpmdGenerator.Kind FieldKind(string structType, string field, SyntaxNode at)
        {
            if (_structs.TryGetValue(structType, out var si))
                foreach (var f in si.Fields)
                    if (f.Name == field) return f.Kind;
            throw new SpmdGenerator.UnsupportedConstructException($"field '{field}' on struct '{structType}'", at.GetLocation());
        }

        private string VName(string structType) => _structs[structType].VName;

        /// <summary>Read of a struct field: <c>v.field</c> (struct local) or <c>buf[i].field</c> (struct buffer).</summary>
        private bool TryStructFieldRead(MemberAccessExpressionSyntax ma, out (string, SpmdGenerator.Kind) result)
        {
            result = default;
            string field = ma.Name.Identifier.Text;
            if (ma.Expression is IdentifierNameSyntax id && _structLocals.TryGetValue(id.Identifier.Text, out var st))
            {
                result = ($"{id.Identifier.Text}.{field}", FieldKind(st, field, ma));
                return true;
            }
            if (ma.Expression is ElementAccessExpressionSyntax ea && TryStructBufferField(ea, out var bufName, out var bufStruct, out var idx))
            {
                var k = FieldKind(bufStruct, field, ma);
                result = (StructFieldGather(bufName, bufStruct, field, idx), k);
                return true;
            }
            return false;
        }

        /// <summary>A struct-valued expression: local, <c>new S{...}</c>/<c>new S(...)</c>, helper call, or ternary.</summary>
        private (string Code, string StructType) VecStruct(ExpressionSyntax e)
        {
            switch (e)
            {
                case ParenthesizedExpressionSyntax p:
                    return VecStruct(p.Expression);
                case IdentifierNameSyntax id when _structLocals.TryGetValue(id.Identifier.Text, out var st):
                    return (id.Identifier.Text, st);
                case ElementAccessExpressionSyntax ea when TryStructBufferField(ea, out var bn, out var bs, out var bidx):
                {
                    // Whole-struct read 'buf[i]' → construct the varying struct from each field's gang load.
                    var si = _structs[bs];
                    if (si.Fields.Count > 1) Warn(SpmdGenerator.GatherPerf, ea, ea.ToString());
                    string fields = string.Join(", ", si.Fields.Select(f => StructFieldGather(bn, bs, f.Name, bidx)));
                    return ($"new {si.VName}({fields})", bs);
                }
                case ObjectCreationExpressionSyntax oc:
                    return VecStructNew(oc);
                case InvocationExpressionSyntax call when call.Expression is IdentifierNameSyntax fid &&
                        _functions.TryGetValue(fid.Identifier.Text, out var fnInfo) && _structs.ContainsKey(fnInfo.ReturnType):
                    return ($"{fnInfo.Name}({EmitCallArgs(call, fnInfo)})", fnInfo.ReturnType);
                case ConditionalExpressionSyntax tern:
                {
                    var t = VecStruct(tern.WhenTrue);
                    var f = VecStruct(tern.WhenFalse);
                    return ($"{VName(t.StructType)}.Select({Mask(tern.Condition)}, {t.Code}, {f.Code})", t.StructType);
                }
                default:
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"struct expression '{Trunc(e)}' (use a local, 'new S{{...}}', 'new S(...)', a helper call, or a ternary)", e.GetLocation());
            }
        }

        private (string Code, string StructType) VecStructNew(ObjectCreationExpressionSyntax oc)
        {
            string type = oc.Type.ToString();
            if (!_structs.TryGetValue(type, out var si))
                throw new SpmdGenerator.UnsupportedConstructException($"'new {type}' (not a [SpmdStruct])", oc.GetLocation());
            string vt = si.VName;

            // Object initializer: new S { x = a, y = b, ... }
            if (oc.Initializer != null)
            {
                var sets = new List<string>();
                foreach (var ex in oc.Initializer.Expressions)
                {
                    if (ex is not AssignmentExpressionSyntax a || a.Left is not IdentifierNameSyntax fn)
                        throw new SpmdGenerator.UnsupportedConstructException($"struct initializer '{Trunc(ex)}'", ex.GetLocation());
                    var k = FieldKind(type, fn.Identifier.Text, ex);
                    sets.Add($"{fn.Identifier.Text} = {Coerce(Vec(a.Right), k, a.Right)}");
                }
                return ($"new {vt} {{ {string.Join(", ", sets)} }}", type);
            }
            // Positional constructor: new S(a, b, ...), args map to fields in declaration order.
            var ctorArgs = oc.ArgumentList?.Arguments ?? default;
            if (ctorArgs.Count != si.Fields.Count)
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"'new {type}(...)' expects {si.Fields.Count} field args (or use 'new {type} {{ field = ... }}')", oc.GetLocation());
            var ctor = new List<string>();
            for (int i = 0; i < ctorArgs.Count; i++)
                ctor.Add(Coerce(Vec(ctorArgs[i].Expression), si.Fields[i].Kind, ctorArgs[i].Expression));
            return ($"new {vt}({string.Join(", ", ctor)})", type);
        }

        /// <summary>Coerce a call's arguments to the helper's parameter types (primitive or struct).</summary>
        private string EmitCallArgs(InvocationExpressionSyntax call, FunctionInfo fn)
        {
            var args = call.ArgumentList.Arguments;
            if (args.Count != fn.Parameters.Count)
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"call to '{fn.Name}' expects {fn.Parameters.Count} arguments", call.GetLocation());
            var parts = new List<string>();
            for (int i = 0; i < args.Count; i++)
            {
                string ptype = fn.Parameters[i].Type;
                if (_structs.ContainsKey(ptype)) parts.Add(VecStruct(args[i].Expression).Code);
                else parts.Add(Coerce(Vec(args[i].Expression), SpmdGenerator.KindOfScalar(ptype, call.GetLocation()), args[i].Expression));
            }
            return string.Join(", ", parts);
        }

        // Struct-buffer parameter access (buf[i].field). Implemented in phase 2.
        private readonly Dictionary<string, string> _structBuffers = new();   // param name → struct type
        private bool TryStructBufferField(ElementAccessExpressionSyntax ea, out string bufName, out string bufStruct, out string idx)
        {
            bufName = ""; bufStruct = ""; idx = "__i";
            if (ea.Expression is not IdentifierNameSyntax id) return false;
            if (!_structBuffers.TryGetValue(id.Identifier.Text, out bufStruct!)) return false;
            bufName = id.Identifier.Text;
            if (ea.ArgumentList.Arguments.Count != 1) return false;
            if (!TryClassifyAffine(ea.ArgumentList.Arguments[0].Expression, out var off)) return false;
            idx = off == "0" ? "__i" : $"(__i + ({off}))";
            return true;
        }

        /// <summary>Load one field of a gang of struct-buffer elements. The struct buffer is viewed as a
        /// flat array of its (single) field type; field <c>f</c> of element e sits at <c>e*N + f</c>, a
        /// stride-N slice, a contiguous gang load when N==1, else a strided gather.</summary>
        private string StructFieldGather(string bufName, string structType, string field, string baseIdx)
        {
            var si = _structs[structType];
            int n = si.Fields.Count;
            int fi = si.Fields.FindIndex(f => f.Name == field);
            var k = si.Fields[fi].Kind;
            string flat = "__flat_" + bufName;
            if (n == 1)
                return $"{VType(k)}.Load({flat}, {baseIdx})";
            string idxVec = $"(({IntT}.ProgramIndex + ({baseIdx})) * {n} + {fi})";
            return k == SpmdGenerator.Kind.F
                ? $"Memory.Gather({flat}, {idxVec})"
                : $"Memory.Gather({flat}, {idxVec}, VMask.All, 0)";
        }

        /// <summary>Store to <c>buf[i].field = expr</c> (with optional compound op).</summary>
        private void EmitStructFieldStore(string bufName, string structType, string field, string baseIdx,
            AssignmentExpressionSyntax asg, string op, string? mask, string indent, StringBuilder sb)
        {
            var k = FieldKind(structType, field, asg);
            if (op != "=" && _structs[structType].Fields.Count > 1)
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"compound '{op}' on struct-buffer field '{field}' (only '=' supported for AoS scatter)", asg.GetLocation());
            string rhs = op == "="
                ? Coerce(Vec(asg.Right), k, asg.Right)
                : CombineCompoundOnExpr(op, StructFieldGather(bufName, structType, field, baseIdx), asg, k);
            EmitStructFieldStoreRaw(bufName, structType, field, baseIdx, rhs, mask, indent, sb);
        }

        /// <summary>Store an already-lowered value into one field of a gang of struct-buffer elements
        /// (contiguous when the struct has one field, else an AoS scatter).</summary>
        private void EmitStructFieldStoreRaw(string bufName, string structType, string field, string baseIdx,
            string valueCode, string? mask, string indent, StringBuilder sb)
        {
            var si = _structs[structType];
            int n = si.Fields.Count;
            int fi = si.Fields.FindIndex(f => f.Name == field);
            var k = si.Fields[fi].Kind;
            string flat = "__flat_" + bufName;
            string vt = VType(k);
            if (n == 1)
            {
                sb.AppendLine(mask == null
                    ? $"{indent}({valueCode}).Store({flat}, {baseIdx});"
                    : $"{indent}{vt}.Select({mask}, {valueCode}, {vt}.Load({flat}, {baseIdx})).Store({flat}, {baseIdx});");
                return;
            }
            string idxVec = $"(({IntT}.ProgramIndex + ({baseIdx})) * {n} + {fi})";
            sb.AppendLine($"{indent}Memory.Scatter({flat}, {idxVec}, {valueCode}, {mask ?? MTAll});");
        }

        private (string Code, SpmdGenerator.Kind Kind) VecCall(InvocationExpressionSyntax call)
        {
            string callee = call.Expression.ToString().Replace(" ", "");

            // A call to another [SpmdFunction] returning a primitive → its varying companion.
            if (call.Expression is IdentifierNameSyntax fid && _functions.TryGetValue(fid.Identifier.Text, out var fnInfo))
            {
                if (_structs.ContainsKey(fnInfo.ReturnType))
                    throw new SpmdGenerator.UnsupportedConstructException(
                        $"struct-returning call '{fnInfo.Name}' used where a scalar is expected", call.GetLocation());
                string callArgs = EmitCallArgs(call, fnInfo);
                return ($"{fnInfo.Name}({callArgs})", SpmdGenerator.KindOfScalar(fnInfo.ReturnType, call.GetLocation()));
            }

            string? fn = callee switch
            {
                "Math.Sqrt" or "MathF.Sqrt" => "Sqrt",
                "Math.Abs" or "MathF.Abs" => "Abs",
                "Math.Min" or "MathF.Min" => "Min",
                "Math.Max" or "MathF.Max" => "Max",
                "Math.Sin" or "MathF.Sin" => "Sin",
                "Math.Cos" or "MathF.Cos" => "Cos",
                "Math.Tan" or "MathF.Tan" => "Tan",
                "Math.Exp" or "MathF.Exp" => "Exp",
                "Math.Log" or "MathF.Log" => "Log",
                "Math.Pow" or "MathF.Pow" => "Pow",
                "Math.Tanh" or "MathF.Tanh" => "Tanh",
                "Math.Floor" or "MathF.Floor" => "Floor",
                "Math.Round" or "MathF.Round" => "Round",
                "Math.Clamp" or "MathF.Clamp" => "Clamp",
                "Math.Atan" or "MathF.Atan" => "Atan",
                "Math.Atan2" or "MathF.Atan2" => "Atan2",
                "Math.Asin" or "MathF.Asin" => "Asin",
                "Math.Acos" or "MathF.Acos" => "Acos",
                "Math.Cbrt" or "MathF.Cbrt" => "Cbrt",
                "MathF.FusedMultiplyAdd" or "Math.FusedMultiplyAdd" => "!FMA",
                _ => null,
            };
            if (fn == null)
                throw new SpmdGenerator.UnsupportedConstructException($"call to '{callee}' (no VectorMath mapping)", call.GetLocation());

            // Long kernels only have integer math, Min/Max/Abs (no transcendentals on 64-bit ints).
            if (_l && fn is not ("Min" or "Max" or "Abs"))
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"'{callee}' on long lanes (only Math.Min/Max/Abs are available for 64-bit integers)", call.GetLocation());

            // VectorMath has VFloat, VDouble, and VDouble2 overloads under the same names,
            // so the emitted call text is identical, only the argument coercion differs.
            // If any argument is double (a VDouble2 in float kernels), the whole call is
            // computed in double precision, mirroring C# overload resolution.
            var vecArgs = call.ArgumentList.Arguments
                .Select(a => (Value: Vec(a.Expression), Node: a.Expression))
                .ToArray();
            // Min/Max/Abs over 64-bit integer lanes stay in long precision (VLong2), the same way
            // a double argument pins the call to double, VectorMath has matching VLong2 overloads.
            bool longIntCall = !_l && fn is "Min" or "Max" or "Abs" &&
                vecArgs.Any(a => a.Value.Kind == SpmdGenerator.Kind.L) &&
                vecArgs.All(a => a.Value.Kind is SpmdGenerator.Kind.L or SpmdGenerator.Kind.I);
            var target = _d || vecArgs.Any(a => a.Value.Kind == SpmdGenerator.Kind.D)
                ? SpmdGenerator.Kind.D
                : longIntCall
                ? SpmdGenerator.Kind.L
                : LaneFloatKind;
            var args = vecArgs.Select(a => CoerceCode(a.Value, target, a.Node)).ToArray();
            string code = fn == "!FMA"
                ? $"{VType(target)}.MulAdd({string.Join(", ", args)})"
                : $"VectorMath.{fn}({string.Join(", ", args)})";
            return (code, target);
        }

        private string Mask(ExpressionSyntax e) => e switch
        {
            ParenthesizedExpressionSyntax p => $"({Mask(p.Expression)})",
            PrefixUnaryExpressionSyntax { OperatorToken.Text: "!" } not => $"(!{Mask(not.Operand)})",
            BinaryExpressionSyntax bin => MaskBin(bin),
            _ => throw new SpmdGenerator.UnsupportedConstructException($"condition '{Trunc(e)}'", e.GetLocation()),
        };

        private string MaskBin(BinaryExpressionSyntax bin)
        {
            string op = bin.OperatorToken.Text;
            if (op is "&&") return $"({Mask(bin.Left)} & {Mask(bin.Right)})";
            if (op is "||") return $"({Mask(bin.Left)} | {Mask(bin.Right)})";

            var l = Vec(bin.Left);
            var r = Vec(bin.Right);
            var kind = Promote(l.Kind, r.Kind);
            string lc = CoerceCode(l, kind, bin.Left);
            string rc = CoerceCode(r, kind, bin.Right);
            string vt = VType(kind);

            return op switch
            {
                "<" or ">" or "<=" or ">=" => $"({lc} {op} {rc})",
                "==" => $"{vt}.Eq({lc}, {rc})",
                "!=" => $"{vt}.Neq({lc}, {rc})",
                _ => throw new SpmdGenerator.UnsupportedConstructException($"condition operator '{op}'", bin.GetLocation()),
            };
        }

        private string Coerce((string Code, SpmdGenerator.Kind Kind) v, SpmdGenerator.Kind target, SyntaxNode at)
            => CoerceCode(v, target, at);

        private string CoerceCode((string Code, SpmdGenerator.Kind Kind) v, SpmdGenerator.Kind target, SyntaxNode at)
        {
            if (v.Kind == target) return v.Code;
            if (v.Kind == SpmdGenerator.Kind.I && target == SpmdGenerator.Kind.F)
                return $"({v.Code}).ToFloat()";
            // float -> int requires an explicit cast in the source, which arrives via CastExpr.
            if (v.Kind == SpmdGenerator.Kind.F && target == SpmdGenerator.Kind.I)
                return $"VInt.FromFloatTruncate({v.Code})";
            // Implicit float/int -> double widening in float/int kernels (mirrors C#):
            // values become VDouble2 pairs at full gang width.
            if (!_d && target == SpmdGenerator.Kind.D && v.Kind == SpmdGenerator.Kind.F)
                return $"VDouble2.FromFloat({v.Code})";
            if (!_d && target == SpmdGenerator.Kind.D && v.Kind == SpmdGenerator.Kind.I)
                return $"VDouble2.FromInt({v.Code})";
            // int -> long widening in float/int kernels (mirrors C#): values become VLong2 pairs.
            // (float -> long only via an explicit cast; both flow through here.)
            if (!_l && target == SpmdGenerator.Kind.L && v.Kind == SpmdGenerator.Kind.I)
                return $"VLong2.FromInt({v.Code})";
            if (!_l && target == SpmdGenerator.Kind.L && v.Kind == SpmdGenerator.Kind.F)
                return $"VLong2.FromFloatTruncate({v.Code})";
            // long -> double widening in float/int kernels (mirrors C#).
            if (!_d && !_l && target == SpmdGenerator.Kind.D && v.Kind == SpmdGenerator.Kind.L)
                return $"({v.Code}).ToDouble2()";
            // double -> float/int and long -> float/int narrowing requires an explicit cast (CastExpr).
            throw new SpmdGenerator.UnsupportedConstructException($"lane conversion at '{Trunc(at)}'", at.GetLocation());
        }

        /// <summary>
        /// Recognize a contiguous gang index: an affine sum where exactly one additive term has
        /// unit stride in the loop variable (the loop var itself, or an affine-in-lane local) and
        /// every other term is uniform, 'buf[i]', 'buf[i + u]', 'buf[i + a + b]', 'buf[even]'
        /// (where 'int even = i + k'), 'buf[y*width + x]', etc.
        /// Returns the scalar base-index expression ("__i" or "__i + (uniform sum)").
        /// </summary>
        private bool TryGetContiguousIndex(ElementAccessExpressionSyntax ea, out string bufferName, out string indexExpr)
        {
            bufferName = ea.Expression.ToString();
            indexExpr = "__i";
            if (ea.ArgumentList.Arguments.Count != 1) return false;
            if (!TryClassifyAffine(ea.ArgumentList.Arguments[0].Expression, out var offset)) return false;
            indexExpr = offset == "0" ? "__i" : $"__i + ({offset})";
            return true;
        }

        /// <summary>
        /// Classify an index expression as affine-in-lane with unit stride: exactly one additive
        /// term is the loop variable or an affine-in-lane local, and every other term is uniform.
        /// Yields the uniform base offset ("0" when there is none). Used for both contiguous
        /// buffer access and to propagate affine-ness through 'int odd = even + half'.
        /// </summary>
        private bool TryClassifyAffine(ExpressionSyntax e, out string offset)
        {
            offset = "0";
            var terms = new List<ExpressionSyntax>();
            FlattenAdditive(e, terms);

            var uniformParts = new List<string>();
            int unitStride = 0;
            foreach (var t in terms)
            {
                var tt = Unparen(t);
                if (tt is IdentifierNameSyntax lv && lv.Identifier.Text == _loopVar) { unitStride++; continue; }
                if (tt is IdentifierNameSyntax al && _affineOffset.TryGetValue(al.Identifier.Text, out var innerOff))
                {
                    unitStride++;
                    if (innerOff != "0") uniformParts.Add(innerOff);
                    continue;
                }
                if (IsUniformExpression(tt)) { uniformParts.Add(tt.ToString()); continue; }
                return false;   // a lane-varying term that isn't unit-stride → not contiguous
            }
            if (unitStride != 1) return false;

            offset = uniformParts.Count == 0 ? "0" : string.Join(" + ", uniformParts);
            return true;
        }

        private static void FlattenAdditive(ExpressionSyntax e, List<ExpressionSyntax> terms)
        {
            e = Unparen(e);
            if (e is BinaryExpressionSyntax { OperatorToken.Text: "+" } b)
            {
                FlattenAdditive(b.Left, terms);
                FlattenAdditive(b.Right, terms);
            }
            else terms.Add(e);
        }

        private static ExpressionSyntax Unparen(ExpressionSyntax e)
            => e is ParenthesizedExpressionSyntax p ? Unparen(p.Expression) : e;

        /// <summary>
        /// Lower buf[non-loop-index] to Memory.Gather(buf, indices).
        /// The index expression is evaluated as a VInt (per-lane indices), then gathered.
        /// </summary>
        private (string Code, SpmdGenerator.Kind Kind) EmitGather(
            ElementAccessExpressionSyntax ea, string bufName, SpmdGenerator.ParamInfo p)
        {
            if (ea.ArgumentList.Arguments.Count != 1)
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"multi-dimensional gather from '{bufName}' (only single-index a[expr] supported)", ea.GetLocation());

            Warn(SpmdGenerator.GatherPerf, ea, ea.ToString());
            var idxExpr = ea.ArgumentList.Arguments[0].Expression;

            // Long kernels index with VLong directly (the lane expression is already 64-bit).
            if (_l)
            {
                var idxL = Coerce(Vec(idxExpr), SpmdGenerator.Kind.L, idxExpr);
                return ($"Memory.Gather({bufName}, {idxL})", SpmdGenerator.Kind.L);
            }

            // Double kernels index with VLong (truncated from the double lane expression).
            if (_d)
            {
                var idxD = Coerce(Vec(idxExpr), SpmdGenerator.Kind.D, idxExpr);
                return ($"Memory.Gather({bufName}, VLong.FromDoubleTruncate({idxD}))", SpmdGenerator.Kind.D);
            }

            var idx = Coerce(Vec(idxExpr), SpmdGenerator.Kind.I, idxExpr);

            // Memory.Gather(ReadOnlySpan<float>, VInt) → VFloat
            // Memory.Gather(ReadOnlySpan<int>, VInt, VMask, int) → VInt
            if (p.ElemKind == SpmdGenerator.Kind.F)
                return ($"Memory.Gather({bufName}, {idx})", SpmdGenerator.Kind.F);
            else
                return ($"Memory.Gather({bufName}, {idx}, VMask.All, 0)", SpmdGenerator.Kind.I);
        }

        /// <summary>Recognize a known scalar constant member (MathF.PI, float.MaxValue, int.MinValue, ...).</summary>
        private bool TryConstMember(MemberAccessExpressionSyntax ma, out (string, SpmdGenerator.Kind) result)
        {
            result = default;
            string member = ma.Name.Identifier.Text;
            if (member is not ("PI" or "E" or "Tau" or "MaxValue" or "MinValue"
                or "Epsilon" or "NaN" or "PositiveInfinity" or "NegativeInfinity"))
                return false;

            string full = ma.ToString().Replace(" ", "");
            SpmdGenerator.Kind k;
            if (full.StartsWith("MathF.") || full.StartsWith("float.")) k = SpmdGenerator.Kind.F;
            else if (full.StartsWith("Math.") || full.StartsWith("double.")) k = SpmdGenerator.Kind.D;
            else if (full.StartsWith("int.")) k = SpmdGenerator.Kind.I;
            else return false;

            if (_d) k = SpmdGenerator.Kind.D;   // double kernel: everything is a VDouble lane
            result = ($"new {VType(k)}({ma})", k);
            return true;
        }

        /// <summary>Lower a 2-D array read a[i, j] to a contiguous load (row-major) or a gather.</summary>
        private (string Code, SpmdGenerator.Kind Kind) Emit2DLoad(ElementAccessExpressionSyntax ea, SpmdGenerator.ParamInfo p)
        {
            var idx0 = ea.ArgumentList.Arguments[0].Expression;
            var idx1 = ea.ArgumentList.Arguments[1].Expression;
            if (TryGet2DContiguous(idx0, idx1, p.ColsName, out var flatBase))
                return ($"{VType(p.ElemKind)}.Load({p.FlatName}, {flatBase})", p.ElemKind);

            Warn(SpmdGenerator.GatherPerf, ea, ea.ToString());
            string flatIdx = Build2DFlatIndex(idx0, idx1, p.ColsName);
            if (_l) return ($"Memory.Gather({p.FlatName}, {flatIdx})", SpmdGenerator.Kind.L);
            if (_d) return ($"Memory.Gather({p.FlatName}, {flatIdx})", SpmdGenerator.Kind.D);
            if (p.ElemKind == SpmdGenerator.Kind.F) return ($"Memory.Gather({p.FlatName}, {flatIdx})", SpmdGenerator.Kind.F);
            return ($"Memory.Gather({p.FlatName}, {flatIdx}, VMask.All, 0)", SpmdGenerator.Kind.I);
        }

        /// <summary>Lower a 2-D array store a[i, j] = ... to a contiguous store or a scatter.</summary>
        private void Emit2DStore(ElementAccessExpressionSyntax ea, SpmdGenerator.ParamInfo p,
            AssignmentExpressionSyntax asg, string op, string? mask, string indent, StringBuilder sb)
        {
            var idx0 = ea.ArgumentList.Arguments[0].Expression;
            var idx1 = ea.ArgumentList.Arguments[1].Expression;
            var ek = p.ElemKind;

            if (TryGet2DContiguous(idx0, idx1, p.ColsName, out var flatBase))
            {
                string vtype = VType(ek);
                string load = $"{vtype}.Load({p.FlatName}, {flatBase})";
                string rhs = CombineCompoundOnExpr(op, load, asg, ek);
                if (mask == null)
                    sb.AppendLine($"{indent}({rhs}).Store({p.FlatName}, {flatBase});");
                else
                    sb.AppendLine($"{indent}{vtype}.Select({mask}, {rhs}, {load}).Store({p.FlatName}, {flatBase});");
                return;
            }

            if (op != "=")
                throw new SpmdGenerator.UnsupportedConstructException(
                    $"compound scatter '{op}' on 2-D array '{p.Name}' (only '=' supported for non-contiguous stores)", asg.GetLocation());
            Warn(SpmdGenerator.ScatterPerf, ea, ea.ToString());
            string flatIdx = Build2DFlatIndex(idx0, idx1, p.ColsName);
            string val = Coerce(Vec(asg.Right), ek, asg.Right);
            string maskArg = mask ?? MTAll;
            sb.AppendLine($"{indent}Memory.Scatter({p.FlatName}, {flatIdx}, {val}, {maskArg});");
        }

        /// <summary>
        /// A 2-D access is contiguous iff the row index is uniform and the column index is the
        /// loop variable (optionally plus a uniform offset): a[row, k] → flat 'row*cols + k',
        /// coefficient 1 on the lane. Returns the scalar base index for a gang Load/Store.
        /// </summary>
        private bool TryGet2DContiguous(ExpressionSyntax idx0, ExpressionSyntax idx1, string cols, out string flatBase)
        {
            flatBase = "";
            // Contiguous iff the row index is uniform and the column index is unit-stride in the
            // loop variable (the loop var, loop var + uniform, or an affine-in-lane local).
            if (!IsUniformExpression(idx0)) return false;
            if (!TryClassifyAffine(idx1, out var colOff)) return false;

            string rowOffset = $"({idx0}) * ({cols})";
            string off = colOff == "0" ? rowOffset : $"{rowOffset} + ({colOff})";
            flatBase = $"__i + ({off})";
            return true;
        }

        /// <summary>Build the per-lane flat index 'idx0*cols + idx1' as a VInt (or VLong in double kernels).</summary>
        private string Build2DFlatIndex(ExpressionSyntax idx0, ExpressionSyntax idx1, string cols)
        {
            string colsB = Wide ? $"new VLong((long){cols})" : $"new VInt({cols})";
            return $"({BuildIntIndex(idx0)} * {colsB} + {BuildIntIndex(idx1)})";
        }

        /// <summary>Build an integer index lane expression (VInt, or VLong in 64-bit kernels) from
        /// affine index syntax, the loop variable becomes ProgramIndex+base, uniforms broadcast.</summary>
        private string BuildIntIndex(ExpressionSyntax e)
        {
            if (IsUniformExpression(e))
                return Wide ? $"new VLong((long)({e}))" : $"new VInt({e})";
            return e switch
            {
                ParenthesizedExpressionSyntax p => $"({BuildIntIndex(p.Expression)})",
                IdentifierNameSyntax id when id.Identifier.Text == _loopVar
                    => Wide ? "(VLong.ProgramIndex + (long)__i)" : "(VInt.ProgramIndex + __i)",
                BinaryExpressionSyntax bin when bin.OperatorToken.Text is "+" or "-" or "*"
                    => $"({BuildIntIndex(bin.Left)} {bin.OperatorToken.Text} {BuildIntIndex(bin.Right)})",
                _ => throw new SpmdGenerator.UnsupportedConstructException(
                    $"non-affine 2-D index '{Trunc(e)}' (indices must be affine in the loop variable and uniforms)", e.GetLocation()),
            };
        }

        private static string Trunc(SyntaxNode n)
        {
            var s = n.ToString();
            return s.Length > 60 ? s.Substring(0, 57) + "..." : s;
        }

        /// <summary>
        /// Tracks break/continue state for the innermost loop being emitted.
        /// For uniform loops (plain C# for), break/continue map to C# keywords.
        /// For varying loops (mask iteration), break/continue manipulate masks.
        /// </summary>
        private readonly struct LoopContext
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

            public static LoopContext Uniform() => new LoopContext(true);
        }
    }
}
