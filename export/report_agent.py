#!/usr/bin/env python3
"""
Standalone HTML Report Composer (Gemini optional)

Usage (Windows PowerShell):
  # strongly recommended: use the interpreter from your venv
  .\.venv\Scripts\python.exe .\report_agent.py ^
    --insights .\insights.json ^
    --asset-dirs .\assets ^
    --out-dir .\out ^
    --verbose

Optional Gemini polish:
  setx GEMINI_API_KEY YOUR_KEY
  .\.venv\Scripts\python.exe .\report_agent.py ^
    --insights .\insights.json ^
    --asset-dirs .\assets ^
    --out-dir .\out ^
    --polish --model gemini-1.5-flash ^
    --verbose
"""

from __future__ import annotations
import argparse
import html as _html
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
load_dotenv()  # does nothing if .env is absent


# -------------------- Data model --------------------

@dataclass
class InsightRefs:
    png_path: Optional[str] = None
    svg_path: Optional[str] = None
    html_path: Optional[str] = None

@dataclass
class Insight:
    visualization_id: str
    title: str
    primary_insight: str
    supporting_evidence: List[str] = field(default_factory=list)
    decision_relevance: str = ""
    caveats: List[str] = field(default_factory=list)
    references: InsightRefs = field(default_factory=InsightRefs)
    narrative: str = ""

@dataclass
class ReportContent:
    report_title: str = ""
    executive_summary: str = ""
    narrative_sections: List[Dict[str, str]] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    actionable_insights: List[str] = field(default_factory=list)
    clarifications_needed: List[str] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)

# -------------------- Loaders --------------------

def load_report(path: Path, verbose: bool=False) -> ReportContent:
    if not path.exists():
        raise FileNotFoundError(f"Insights file not found: {path.resolve()}")
    if verbose:
        print(f"→ Loading insights from {path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))

    def _normalize_data(d):
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            for key in ("per_chart_insights", "insights", "items", "charts"):
                v = d.get(key)
                if isinstance(v, list):
                    return v
            # If it looks like a single insight object, wrap it
            if "visualization_id" in d and "title" in d:
                return [d]
        raise ValueError(
            "Insights JSON must be a list of insight objects or contain a 'per_chart_insights' array."
        )

    items = _normalize_data(data)

    def _to_insight(x: Dict[str, Any]) -> Insight:
        refs = x.get("references", {}) or {}
        return Insight(
            visualization_id=str(x.get("visualization_id") or x.get("id") or x.get("viz_id") or "viz"),
            title=str(x.get("title") or x.get("name") or "Untitled"),
            primary_insight=str(x.get("primary_insight") or x.get("insight") or ""),
            supporting_evidence=list(x.get("supporting_evidence") or x.get("evidence") or []),
            decision_relevance=str(x.get("decision_relevance") or x.get("so_what") or ""),
            caveats=list(x.get("caveats") or x.get("limitations") or []),
            references=InsightRefs(
                html_path=refs.get("html_path"),
                png_path=refs.get("png_path"),
                svg_path=refs.get("svg_path"),
            ),
            narrative=str(x.get("narrative") or ""),
        )

    insights = [_to_insight(x) for x in items if isinstance(x, dict)]

    # Extract report-level sections
    report_title = ""
    executive_summary = ""
    narrative_sections = []
    key_findings = []
    actionable_insights = []
    clarifications_needed = []
    if isinstance(data, dict):
        report_title = data.get("report_title", "") or ""
        executive_summary = data.get("executive_summary", "") or ""
        narrative_sections = data.get("narrative_sections", []) or []
        key_findings = data.get("key_findings", []) or []
        actionable_insights = data.get("actionable_insights", []) or []
        clarifications_needed = data.get("clarifications_needed", []) or []

    # Align narrative bodies to charts (by index or matching heading↔title)
    bodies = [ns.get("body", "") if isinstance(ns, dict) else str(ns) for ns in (narrative_sections or [])]
    if len(bodies) == len(insights):
        for i, it in enumerate(insights):
            it.narrative = bodies[i]
    else:
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (s or "").lower())
        hmap = {}
        for ns in (narrative_sections or []):
            if isinstance(ns, dict):
                hmap[_norm(ns.get("heading", ""))] = ns.get("body", "")
        for it in insights:
            it.narrative = hmap.get(_norm(it.title), it.narrative)

    if verbose:
        print(f"✓ Loaded {len(insights)} insights; exec_summary={'yes' if executive_summary else 'no'}")

    return ReportContent(
        report_title=report_title,
        executive_summary=executive_summary,
        narrative_sections=narrative_sections,
        key_findings=key_findings,
        actionable_insights=actionable_insights,
        clarifications_needed=clarifications_needed,
        insights=insights,
    )

def load_asset_map(path: Optional[Path], verbose: bool=False) -> Dict[str, Dict[str, str]]:
    if path and path.exists():
        if verbose:
            print(f"→ Loading explicit asset map from {path.resolve()}")
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

# -------------------- Asset Resolver --------------------

def resolve_assets(
    insights: List[Insight],
    asset_dirs: List[Path],
    explicit_map: Dict[str, Dict[str, str]],
    verbose: bool=False
) -> Tuple[List[Insight], List[str]]:
    logs: List[str] = []
    if verbose:
        if asset_dirs:
            print("→ Searching asset dirs:")
            for d in asset_dirs: print("   -", d.resolve())
        else:
            print("→ No asset dirs supplied; will use CWD for search")

    def _find_file(viz: str, exts: Tuple[str, ...]) -> Optional[Path]:
        pat = re.compile(re.escape(viz), re.IGNORECASE)
        for d in asset_dirs or [Path.cwd()]:
            if not d.exists(): continue
            for p in d.glob("*"):
                if p.is_file() and p.suffix.lower() in exts and pat.search(p.stem):
                    return p
        return None

    for it in insights:
        if verbose:
            print(f"   · Resolving assets for {it.visualization_id}")

        # explicit map first
        if it.visualization_id in explicit_map:
            mp = explicit_map[it.visualization_id]
            it.references.html_path = mp.get("html_path") or it.references.html_path
            it.references.png_path  = mp.get("png_path")  or it.references.png_path
            it.references.svg_path  = mp.get("svg_path")  or it.references.svg_path

        # search for missing
        if not it.references.html_path:
            f = _find_file(it.visualization_id, (".html",))
            if f:
                it.references.html_path = str(f.resolve())
                if verbose: print(f"     → html: {it.references.html_path}")
        if not it.references.png_path:
            f = _find_file(it.visualization_id, (".png", ".jpg", ".jpeg"))
            if f:
                it.references.png_path = str(f.resolve())
                if verbose: print(f"     → png : {it.references.png_path}")
        if not it.references.svg_path:
            f = _find_file(it.visualization_id, (".svg",))
            if f:
                it.references.svg_path = str(f.resolve())
                if verbose: print(f"     → svg : {it.references.svg_path}")

        if not any([it.references.html_path, it.references.png_path, it.references.svg_path]):
            msg = f"[warn] Missing chart asset for {it.visualization_id}"
            logs.append(msg)
            if verbose: print("     →", msg)

    return insights, logs


# -------------------- Gemini (optional polish) --------------------

SYSTEM_INSTRUCTION = """
You are a senior data analyst polishing copy for an executive business report.

STRICT RULES
1) Do NOT change facts or numbers. Never invent new metrics, causes, forecasts, or comparisons.
2) Tone: concise, assertive, decision-oriented. Active voice. No hedging (“may”, “might”, “appears”).
3) Primary Insight: ONE punchy headline sentence (≤ 18 words). Start with the result, not the setup.
4) Supporting Evidence: 1–3 compact bullets (≤ 10 words each). Only what’s already provided.
5) Decision Relevance: Start with an imperative verb (Prioritize, Reallocate, Increase, Reduce, Adjust, Monitor).
6) Caveats: 0–3 short bullets (≤ 10 words). Only if present or self-evident from the input.
7) Ban filler: “chart/graph shows”, “visualizes”, “indicates”, “helps to”, “significantly” (unless quantified).
8) Output EXACTLY this JSON shape (no extra keys, no prose):
   {
     "primary_insight": string,
     "supporting_evidence": string[],
     "decision_relevance": string,
     "caveats": string[]
   }
"""


def maybe_polish_with_gemini(
    insights: List[Insight],
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    max_chars: int = 900,
    verbose: bool=False
) -> List[Insight]:
    # This function requires a recent version of the google-generativeai library
    # If you get an error like "'google.generativeai' has no attribute 'GenerativeModel'",
    # update the library by running: pip install --upgrade google-generativeai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        if verbose: print("ⓘ GEMINI_API_KEY not set; skipping polish")
        return insights

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # The line below is correct but requires an up-to-date library version.
        # The error message indicates your installed version is too old.
        gmodel = genai.GenerativeModel(model,  system_instruction=SYSTEM_INSTRUCTION)
        gen_cfg = {
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                }
        if verbose: print(f"→ Polishing text with Gemini model={model}")
    except Exception as e:
        if verbose: print(f"ⓘ Could not init Gemini: {e}; skipping polish")
        return insights

    def _trim(s: str, n: int) -> str:
        return s if len(s) <= n else s[: n - 1] + "…"

    out: List[Insight] = []
    for it in insights:
        # **IMPROVEMENT**: Include title in the JSON payload for better context
        payload = {
            "title": it.title,
            "primary_insight": _trim(it.primary_insight, max_chars),
            "supporting_evidence": [_trim(x, max_chars) for x in (it.supporting_evidence or [])],
            "decision_relevance": _trim(it.decision_relevance, max_chars),
            "caveats": [_trim(x, max_chars) for x in (it.caveats or [])],
        }
        prompt = (
            "Polish this section for an executive report using the system rules.\n"
            "Do not add facts. Keep all numbers unchanged.\n"
            "Return ONLY a JSON object with keys: "
            "{primary_insight, supporting_evidence, decision_relevance, caveats}.\n"
            "Do NOT include the 'title' in your output.\n\n"
            f"JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        # (
        #     "You tidy short analytical bullets for a business report.\n"
        #     "Constraints:\n"
        #     " - Preserve the exact meaning and any numbers. Do not invent.\n"
        #     " - Make wording concise and clear.\n"
        #     " - The 'title' in the JSON is for context only; do not include it in your output.\n"
        #     "Return ONLY a compact JSON object with keys: "
        #     "{primary_insight, supporting_evidence, decision_relevance, caveats}.\n\n"
        #     f"CURRENT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        # )
        try:
            resp = gmodel.generate_content(prompt, generation_config=gen_cfg)
            text = (resp.text or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
            
            start = text.find("{"); end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                j = json.loads(text[start:end+1])
                it.primary_insight      = j.get("primary_insight", it.primary_insight)
                it.supporting_evidence  = j.get("supporting_evidence", it.supporting_evidence) or []
                it.decision_relevance   = j.get("decision_relevance", it.decision_relevance)
                it.caveats              = j.get("caveats", it.caveats) or []
        except Exception as e:
            if verbose: print(f"     · polish failed for {it.visualization_id}: {e}")
        out.append(it)
    return out

# -------------------- HTML --------------------

HTML_CSS = """
:root { --bg:#0b0c10; --panel:#12151a; --text:#e6edf3; --muted:#9aa4af; --accent:#7aa2f7; --border:#222833; }
*{box-sizing:border-box}
html,body{background:var(--bg);color:var(--text);margin:0;line-height:1.55;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
.container{max-width:1100px;margin:2rem auto;padding:0 1rem 3rem}
header.report-header{background:linear-gradient(180deg,rgba(122,162,247,.15),transparent 70%);
  padding:2rem 1rem 1rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem}
.report-title{font-size:1.75rem;margin:0 0 .25rem}
.report-subtitle{margin:0;color:var(--muted);font-size:.95rem}
.toc{margin:1rem 0 2rem;padding:.75rem 1rem;border:1px solid var(--border);border-radius:12px;background:var(--panel)}
.toc h3{margin-top:0;font-size:1rem;color:var(--muted)}
.toc a{color:var(--accent);text-decoration:none}
.toc ul{margin:.25rem 0 0 1.25rem}
.report-section{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:1rem;
  margin:1rem 0 1.25rem;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.section-title{margin:.25rem 0 1rem;font-size:1.25rem}
.chart{background:#0d1016;border:1px dashed var(--border);border-radius:12px;padding:.25rem;margin-bottom:1rem}
.chart-iframe{width:100%;height:460px;border:none;background:#0d1016;border-radius:10px}
.chart-img{width:100%;height:auto;display:block;border-radius:10px}
.chart--missing .placeholder{padding:1rem;color:var(--muted)}
.insight-kicker{text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;color:var(--muted);margin-top:.5rem}
footer{margin-top:2rem;text-align:center;color:var(--muted);font-size:.9rem}
"""

def _esc(s: str) -> str: return _html.escape(s or "")

def render_section(it: Insight) -> str:
    chart_html = it.references.html_path
    chart_png  = it.references.png_path
    chart_svg  = it.references.svg_path

    if chart_html and Path(chart_html).exists():
        rel = Path(chart_html).name
        chart_block = f"""
        <div class="chart">
          <iframe src="{_esc(rel)}" class="chart-iframe" loading="lazy"
                  sandbox="allow-scripts allow-same-origin"></iframe>
        </div>"""
    elif chart_png and Path(chart_png).exists():
        rel = Path(chart_png).name
        chart_block = f"""
        <div class="chart">
          <img src="{_esc(rel)}" alt="{_esc(it.title)}" class="chart-img"/>
        </div>"""
    elif chart_svg and Path(chart_svg).exists():
        rel = Path(chart_svg).name
        chart_block = f"""
        <div class="chart">
          <img src="{_esc(rel)}" alt="{_esc(it.title)}" class="chart-img"/>
        </div>"""
    else:
        chart_block = f"""
        <div class="chart chart--missing">
          <div class="placeholder">
            <strong>Chart not available:</strong> <code>{_esc(it.visualization_id)}</code><br/>
            Provide an HTML/PNG/SVG for this visualization to render here.
          </div>
        </div>"""

    evidence_list = "".join(f"<li>{_esc(e)}</li>" for e in (it.supporting_evidence or [])) or "<li>—</li>"
    caveats_list  = "".join(f"<li>{_esc(c)}</li>" for c in (it.caveats or [])) or "<li>None noted</li>"
    narrative_block = f"<p>{_esc(it.narrative)}</p>" if (it.narrative or "").strip() else ""

    return f"""
    <section id="{_esc(it.visualization_id)}" class="report-section">
      <h2 class="section-title">{_esc(it.title)}</h2>
      {chart_block}
      {narrative_block}
      <div class="insight">
        <div class="insight-kicker">Primary Insight</div>
        <p>{_esc(it.primary_insight)}</p>
        <div class="insight-kicker">Supporting Evidence</div>
        <ul>{evidence_list}</ul>
        <div class="insight-kicker">Decision Relevance</div>
        <p>{_esc(it.decision_relevance)}</p>
        <div class="insight-kicker">Caveats</div>
        <ul>{caveats_list}</ul>
      </div>
    </section>"""

def compose_html(report: ReportContent, title: str) -> str:
    insights = report.insights
    toc = "\n".join(f'<li><a href="#{_esc(x.visualization_id)}">{_esc(x.title)}</a></li>' for x in insights)
    sections = "\n".join(render_section(x) for x in insights)

    exec_summary_block = ""
    if (report.executive_summary or "").strip():
        exec_summary_block = f"""
    <section id="executive_summary" class="report-section">
      <h2 class="section-title">Executive Summary</h2>
      <p>{_esc(report.executive_summary)}</p>
    </section>"""

    def _bullets(items: List[str], empty: str = "—") -> str:
        return ("".join(f"<li>{_esc(x)}</li>" for x in (items or []))) or f"<li>{empty}</li>"

    key_findings_block = f"""
    <section id="key_findings" class="report-section">
      <h2 class="section-title">Key Findings</h2>
      <ul>{_bullets(report.key_findings)}</ul>
    </section>"""

    actionable_block = f"""
    <section id="actionable_insights" class="report-section">
      <h2 class="section-title">Actionable Insights</h2>
      <ul>{_bullets(report.actionable_insights)}</ul>
    </section>"""

    clarifications_block = f"""
    <section id="clarifications_needed" class="report-section">
      <h2 class="section-title">Clarifications needed</h2>
      <ul>{_bullets(report.clarifications_needed)}</ul>
    </section>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_esc(title)}</title><style>{HTML_CSS}</style>
</head>
<body>
  <div class="container">
    <header class="report-header">
      <h1 class="report-title">{_esc(title)}</h1>
      <p class="report-subtitle">Auto-generated Executive Summary, Charts with Narrative and Insights, Key Findings, Actionables, Clarifications</p>
    </header>
    <div class="toc"><h3>Contents</h3><ul>{toc}</ul></div>
    {exec_summary_block}
    {sections}
    {key_findings_block}
    {actionable_block}
    {clarifications_block}
  </div>
</body></html>"""

# -------------------- Exports --------------------

def write_outputs(
    report: ReportContent,
    out_dir: Path,
    verbose: bool=False,
    html_name: str = "sales_insights_report.html",
    resolved_name: str = "resolved_insights.json",
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose: print(f"→ Writing outputs to {out_dir.resolve()}")

    # Copy assets next to the report so it’s portable
    for it in report.insights:
        for p in [it.references.html_path, it.references.png_path, it.references.svg_path]:
            if p and Path(p).exists():
                src = Path(p); dst = out_dir / src.name
                if src.resolve() != dst.resolve():
                    try:
                        dst.write_bytes(src.read_bytes())
                        if verbose: print(f"   · copied {src.name}")
                    except Exception as e:
                        print(f"[warn] could not copy {src} → {dst}: {e}", file=sys.stderr)

    # Derive title and filename from report.report_title if provided
    title = (report.report_title or "").strip() or "Sales Insights Report"
    def _slugify_filename(s: str) -> str:
        base = re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")
        return f"{base}.html" if base else "report.html"

    computed_html_name = _slugify_filename(title) if report.report_title else html_name
    html_path = out_dir / computed_html_name
    html_path.write_text(compose_html(report, title), encoding="utf-8")
    if verbose: print(f"✓ HTML report: {html_path.resolve()}")

    def _localize(path_str: Optional[str]) -> Optional[str]:
        return Path(path_str).name if path_str else None

    resolved = []
    for it in report.insights:
        resolved.append({
            "visualization_id": it.visualization_id,
            "title": it.title,
            "primary_insight": it.primary_insight,
            "supporting_evidence": it.supporting_evidence,
            "decision_relevance": it.decision_relevance,
            "caveats": it.caveats,
            "narrative": it.narrative,
            "references": {
                "html_path": _localize(it.references.html_path),
                "png_path": _localize(it.references.png_path),
                "svg_path": _localize(it.references.svg_path),
            }
        })
    resolved_path = out_dir / resolved_name
    resolved_path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose: print(f"✓ Resolved JSON: {resolved_path.resolve()}")

    return html_path, resolved_path

# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Standalone HTML Report Composer (Gemini optional)")
    ap.add_argument("--insights", type=Path, required=True, help="Path to Insights JSON")
    ap.add_argument("--asset-dirs", type=str, default="", help="Comma-separated folders to search for charts")
    ap.add_argument("--asset-map", type=Path, default=None, help="Optional JSON mapping viz_id -> {html_path|png_path|svg_path}")
    ap.add_argument("--out-dir", type=Path, default=Path("./out"), help="Output directory")
    ap.add_argument("--polish", action="store_true", help="Polish text via Gemini (requires GEMINI_API_KEY)")
    ap.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model for polishing")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap.parse_args()

def main():
    args = parse_args()
    report = load_report(args.insights, verbose=args.verbose)
    explicit_map = load_asset_map(args.asset_map, verbose=args.verbose)
    asset_dirs = [Path(p.strip()) for p in (args.asset_dirs.split(",") if args.asset_dirs else []) if p.strip()]

    report.insights, logs = resolve_assets(report.insights, asset_dirs, explicit_map, verbose=args.verbose)
    for w in logs:
        print(w, file=sys.stderr)

    if args.polish:
        report.insights = maybe_polish_with_gemini(report.insights, model=args.model, verbose=args.verbose)

    write_outputs(report, args.out_dir, verbose=args.verbose)

if __name__ == "__main__":
    main()
