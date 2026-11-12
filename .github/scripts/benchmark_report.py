#!/usr/bin/env python3
"""Turn BenchmarkDotNet results from several runners into one readable Markdown report.

Usage:  benchmark_report.py <results-root> <output.md>

<results-root> holds one directory per runner (benchmark-results-<label>/), each containing the
BenchmarkDotNet "*-report-github.md" files that runner produced.

The report leads with *speedup over the scalar baseline measured on the same runner in the same
job*. That ratio is the number that survives a change of runner: hosted runners differ by 2x or
more from one run to the next, so raw times from two runs (or two platforms) are not comparable.
Raw times are still listed, in a collapsed section at the end.

Standard library only, so it runs on a bare GitHub runner.
"""

from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

BASELINE = "Scalar"
# Method name -> (section title, one-line description). Anything else lands in "Hand-written".
GENERATED = {
    "GeneratedSimd": ("Generated SIMD, single thread", "`_Simd`: the generated kernel on one core."),
    "GeneratedParallelSimd": ("Generated SIMD, all cores", "`_ParallelSimd`: SIMD on every core. Depends heavily on the runner's core count."),
}
NOISY = 0.10          # relative standard deviation above which a measurement is flagged
BAR_WIDTH = 10        # characters in a full-length table bar
UNITS_NS = {"ns": 1.0, "us": 1e3, "μs": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}


@dataclass
class Measurement:
    mean: float                 # nanoseconds
    stddev: float | None        # nanoseconds

    @property
    def noise(self) -> float:
        return (self.stddev / self.mean) if self.stddev and self.mean else 0.0


@dataclass
class Runner:
    label: str
    os: str = "?"
    cpu: str = "?"
    runtime: str = "?"
    jit: str = "?"
    # results[(bench, params)][method] = Measurement
    results: dict = field(default_factory=lambda: defaultdict(dict))


def parse_time(cell: str) -> float | None:
    """'9,129.446 μs' -> nanoseconds. Returns None for NA / '-' / anything unparseable."""
    m = re.fullmatch(r"([\d,]*\.?\d+)\s*([^\d\s]+)", cell.strip())
    if not m or m.group(2) not in UNITS_NS:
        return None
    return float(m.group(1).replace(",", "")) * UNITS_NS[m.group(2)]


def split_row(line: str) -> list[str]:
    # BenchmarkDotNet bolds the first row of each parameter group; the markers are not data.
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def parse_report(path: Path, runner: Runner) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    bench = path.name.split("-report")[0].split(".")[-1]
    bench = re.sub(r"Bench$", "", bench)

    # Environment block: "BenchmarkDotNet vX, <OS>" / "<CPU>, 1 CPU, ..." / "[Host] : .NET x (...), <JIT>"
    for i, line in enumerate(lines):
        if line.startswith("BenchmarkDotNet v") and "," in line:
            runner.os = line.split(",", 1)[1].strip()
            if i + 1 < len(lines):
                runner.cpu = re.sub(r",\s*\d+ CPU.*", "", lines[i + 1]).strip()
                cores = re.search(r"(\d+) logical", lines[i + 1])
                if cores:
                    runner.cpu += f" ({cores.group(1)} logical cores)"
        host = re.match(r"\s*\[Host\]\s*:\s*(\.NET [\d.]+)\s*(?:\([^)]*\))?,\s*(.+)", line)
        if host:
            runner.runtime, runner.jit = host.group(1), host.group(2).strip()

    try:
        start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("| Method"))
    except StopIteration:
        return
    header = split_row(lines[start])
    if "Mean" not in header:
        return
    mean_i = header.index("Mean")
    sd_i = header.index("StdDev") if "StdDev" in header else None
    param_names = header[1:mean_i]          # every column between Method and Mean is a [Params]

    for line in lines[start + 2:]:
        if not line.lstrip().startswith("|"):
            break
        cells = split_row(line)
        if len(cells) <= mean_i or not cells[0]:
            continue                        # blank separator row between parameter groups
        mean = parse_time(cells[mean_i])
        if mean is None:
            continue                        # 'NA': the benchmark failed on this runner
        sd = parse_time(cells[sd_i]) if sd_i is not None and sd_i < len(cells) else None
        params = tuple((n, cells[1 + k]) for k, n in enumerate(param_names))
        runner.results[(bench, params)][cells[0]] = Measurement(mean, sd)


def load(root: Path) -> list[Runner]:
    runners = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        runner = Runner(d.name.replace("benchmark-results-", ""))
        for md in sorted(d.rglob("*-report-github.md")):
            parse_report(md, runner)
        if runner.results:
            runners.append(runner)
    return runners


# ---------------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------------

def fmt_time(ns: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3), ("ns", 1.0)):
        if ns >= scale or unit == "ns":
            v = ns / scale
            return f"{v:.0f} {unit}" if v >= 100 else f"{v:.1f} {unit}" if v >= 10 else f"{v:.2f} {unit}"
    return "?"


def fmt_speedup(x: float) -> str:
    return f"{x:.0f}×" if x >= 100 else f"{x:.1f}×" if x >= 10 else f"{x:.2f}×"


def fmt_param_value(v: str) -> str:
    return f"{int(v):,}" if v.isdigit() else v


def fmt_param_compact(v: str) -> str:
    """4194304 -> '4M', 65536 -> '64K': short enough for a chart axis label."""
    if not v.isdigit():
        return v
    n = int(v)
    for shift, suffix in ((30, "G"), (20, "M"), (10, "K")):
        if n >= 1 << shift and n % (1 << shift) == 0:
            return f"{n >> shift}{suffix}"
    return f"{n:,}"


def chart_label(key, multi_param_benches) -> str:
    bench, params = key
    name = re.sub(r"^Generated", "", bench)
    if bench in multi_param_benches and params:
        name += " " + "/".join(fmt_param_compact(v) for _, v in params)
    return name


def bar(value: float, vmax: float, width: int = BAR_WIDTH) -> str:
    """A horizontal bar in eighth-block resolution; never empty, so tiny values stay visible."""
    eighths = max(1, round(value / vmax * width * 8)) if vmax > 0 else 1
    eighths = min(eighths, width * 8)
    return "█" * (eighths // 8) + ("", "▏", "▎", "▍", "▌", "▋", "▊", "▉")[eighths % 8]


def geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def kernel_label(key, multi_param_benches) -> str:
    bench, params = key
    if bench in multi_param_benches and params:
        return f"{bench} <sub>{', '.join(f'{n}={fmt_param_value(v)}' for n, v in params)}</sub>"
    return bench


def mermaid_bar_chart(title: str, labels: list[str], values: list[float], y_title: str) -> list[str]:
    """Horizontal Mermaid xychart; GitHub renders these in job summaries, issues and READMEs."""
    top = max(values) * 1.08
    height = 90 + 26 * len(labels)
    quoted = ", ".join('"' + l.replace('"', "'") + '"' for l in labels)
    return [
        "```mermaid",
        f'%%{{init: {{"xyChart": {{"width": 900, "height": {height}, "xAxis": {{"labelFontSize": 11}}}}, "themeVariables": {{"xyChart": {{"plotColorPalette": "#2f81f7"}}}}}}}}%%',
        "xychart-beta horizontal",
        f'    title "{title}"',
        f"    x-axis [{quoted}]",
        f'    y-axis "{y_title}" 0 --> {top:.1f}',
        f"    bar [{', '.join(f'{v:.2f}' for v in values)}]",
        "```",
    ]


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def build(runners: list[Runner]) -> str:
    out: list[str] = []
    labels = [r.label for r in runners]
    def natural(key):
        # Sort parameter values numerically where they are numbers (N=65,536 before N=4,194,304).
        bench, params = key
        return bench, tuple((n, (0, int(v), "") if v.isdigit() else (1, 0, v)) for n, v in params)

    keys = sorted({k for r in runners for k in r.results}, key=natural)
    param_sets = defaultdict(set)
    for bench, params in keys:
        param_sets[bench].add(params)
    multi = {b for b, s in param_sets.items() if len(s) > 1}

    def speedup(runner: Runner, key, method) -> tuple[float, bool] | None:
        """(scalar time / method time, noisy?) on one runner, or None when either is missing."""
        row = runner.results.get(key, {})
        base, m = row.get(BASELINE), row.get(method)
        if not base or not m or m.mean <= 0:
            return None
        return base.mean / m.mean, (base.noise > NOISY or m.noise > NOISY)

    methods = sorted({m for r in runners for row in r.results.values() for m in row if m != BASELINE})
    hand_written = [m for m in methods if m not in GENERATED]

    out += ["# IspcSharp benchmark report", ""]
    out += [
        "**Speedup = scalar time ÷ SIMD time, both measured on the same runner in the same job.** "
        "Higher is better; 1× means no gain. Compare speedups between runs, not raw times: "
        "hosted runners differ by 2× or more from one run to the next.",
        "",
        f"≈ marks a speedup where either measurement varied by more than {NOISY:.0%} between "
        "iterations (treat it as approximate). ⚠️ marks SIMD slower than scalar.",
        "",
    ]

    # --- Runners -----------------------------------------------------------------------------
    out += ["## Runners", "", "| Runner | CPU | OS | Runtime | JIT target |", "|---|---|---|---|---|"]
    for r in runners:
        out.append(f"| **{r.label}** | {r.cpu} | {r.os} | {r.runtime} | {r.jit} |")
    out.append("")

    # --- Summary -----------------------------------------------------------------------------
    summary = {}
    for method in [m for m in GENERATED if m in methods] + hand_written:
        summary[method] = [geomean([s[0] for k in keys if (s := speedup(r, k, method))]) for r in runners]
    smax = max((v for row in summary.values() for v in row if v), default=1.0)
    out += ["## Summary", "", "Geometric mean speedup over scalar, across every kernel that has the method.", ""]
    out += ["| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for method, row in summary.items():
        name = GENERATED[method][0] if method in GENERATED else f"Hand-written `{method}`"
        cells = [f"{bar(v, smax)} **{fmt_speedup(v)}**" if v else "–" for v in row]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("")

    # --- One table per method ----------------------------------------------------------------
    def table(method: str, title: str, blurb: str) -> None:
        rows = []
        for k in keys:
            vals = [speedup(r, k, method) for r in runners]
            if any(vals):
                rows.append((k, vals, geomean([v[0] for v in vals if v]) or 0.0))
        if not rows:
            return
        rows.sort(key=lambda t: -t[2])
        vmax = max(v[0] for _, vals, _ in rows for v in vals if v)
        out.extend([f"## {title}", "", blurb + " Sorted by average speedup.", ""])
        out.extend(["| Kernel | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)])
        for k, vals, _ in rows:
            cells = []
            for v in vals:
                if not v:
                    cells.append("–")
                    continue
                x, noisy = v
                text = f"{bar(x, vmax)} {'≈' if noisy else ''}{fmt_speedup(x)}"
                cells.append(text + (" ⚠️" if x < 0.95 else ""))
            out.append(f"| {kernel_label(k, multi)} | " + " | ".join(cells) + " |")
        out.append("")

    for method, (title, blurb) in GENERATED.items():
        if method in methods:
            table(method, title, blurb)
    for method in hand_written:
        table(method, f"Hand-written `{method}`", f"A loop written directly against the lane type (`{method}`).")

    # --- Charts ------------------------------------------------------------------------------
    chart_method = next((m for m in GENERATED if m in methods), None)
    if chart_method:
        out += ["## Charts", "", f"{GENERATED[chart_method][0]}: speedup over scalar for every kernel, one chart per runner.", ""]
        for r in runners:
            pts = sorted(((s[0], k) for k in keys if (s := speedup(r, k, chart_method))), key=lambda t: -t[0])
            if not pts:
                continue
            names = [chart_label(k, multi) for _, k in pts]
            out += [f"<details{' open' if r is runners[0] else ''}><summary><b>{r.label}</b></summary>", ""]
            out += mermaid_bar_chart(f"{r.label}: {GENERATED[chart_method][0]}", names, [x for x, _ in pts], "times faster than scalar")
            out += ["", "</details>", ""]

    # --- Raw times ---------------------------------------------------------------------------
    out += ["## Raw times", "", "<details><summary>Mean time per call on each runner (not comparable between runs)</summary>", ""]
    for bench in sorted(param_sets):
        bench_keys = [k for k in keys if k[0] == bench]
        params_present = any(k[1] for k in bench_keys)
        out += [f"#### {bench}", ""]
        out += ["| Method | " + ("Params | " if params_present else "") + " | ".join(labels) + " |"]
        out += ["|---|" + ("---|" if params_present else "") + "---:|" * len(labels)]
        for k in bench_keys:
            row_methods = sorted({m for r in runners for m in r.results.get(k, {})}, key=lambda m: (m != BASELINE, m))
            for m in row_methods:
                cells = []
                for r in runners:
                    x = r.results.get(k, {}).get(m)
                    cells.append("–" if not x else fmt_time(x.mean) + (f" (±{x.noise:.0%})" if x.noise > NOISY else ""))
                p = ", ".join(f"{n}={fmt_param_value(v)}" for n, v in k[1])
                out.append(f"| {'**Scalar**' if m == BASELINE else m} | " + (f"{p} | " if params_present else "") + " | ".join(cells) + " |")
        out.append("")
    out += ["</details>", ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    runners = load(Path(argv[1]))
    if not runners:
        print(f"No BenchmarkDotNet reports found under {argv[1]}", file=sys.stderr)
        Path(argv[2]).write_text("# IspcSharp benchmark report\n\nNo benchmark results were produced.\n", encoding="utf-8")
        return 1
    Path(argv[2]).write_text(build(runners), encoding="utf-8")
    print(f"Wrote {argv[2]}: {len(runners)} runner(s), {len({k for r in runners for k in r.results})} kernel configuration(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
