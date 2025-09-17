from math import ceil
from plotly.subplots import make_subplots

import argparse, json, os, sys, warnings
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, continue without it

# Fix for chart_studio compatibility with plotly 6.3.0
# The issue is that chart_studio expects plotly.version to have a stable_semver() method
# but in plotly 6.3.0, version is just a string
try:
    import plotly
    if hasattr(plotly, 'version') and isinstance(plotly.version, str):
        # Create a mock version object with stable_semver method
        class MockVersion:
            def __init__(self, version_string):
                self.version_string = version_string
            
            def stable_semver(self):
                return self.version_string
        
        # Replace the string version with our mock object
        plotly.version = MockVersion(plotly.version)
        # print(f"🔧 Fixed plotly version compatibility: {plotly.version.stable_semver()}")
except Exception as e:
    print(f"⚠️ Could not fix plotly version compatibility: {e}")

# =========================
# v5: schema simplification helpers
# =========================

def _ensure_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(',') if s.strip()]
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []

# normalize a single chart spec from v5 shorthand into v4-compatible keys
# Accepted v5 shorthands:
# - agg: "sum" | "mean" | "min" | "max" | ... (string)
# - by: "Year,Region" | ["Year","Region"]
# - top_n: 5 (int) or {n:5, by: <col>, category: <col>, scope: overall|per_group, group: <col>}
# - You can omit x/y/color/size/value; they will be auto-inferred.
# - y_title/x_title/legend_* remain optional but are rarely needed.
# - height_weight optional for combined layout weighting.

def normalize_v5_spec(spec: Dict[str, Any], schema: Dict[str, List[str]]) -> Dict[str, Any]:
    s = dict(spec)

    # top_n can be int in v5
    if isinstance(s.get("top_n"), int):
        s["top_n"] = {"n": int(s["top_n"]) }

    # Support simple aggregation: agg + by (+ optional measure)
    # If user passed complex aggregate already, leave as-is
    if "aggregate" not in s:
        agg = s.get("agg") or s.get("Agg") or s.get("AGG")
        by = s.get("by")
        measure = s.get("measure") or s.get("value") or s.get("y")
        if agg or by or measure:
            how = str(agg).lower() if agg else "sum"
            groupby = _ensure_list(by)
            # choose measure fallback from y if not provided
            if not measure:
                # try to infer y from schema numerics
                nums = [c for c in schema.get("numerics", []) if c in (s.get("y"), s.get("value"))]
                measure = nums[0] if nums else (s.get("y") or s.get("value"))
            aggregate = {"how": how}
            if groupby:
                aggregate["groupby"] = groupby
            if measure:
                aggregate["measure"] = measure
            s["aggregate"] = aggregate

    # Friendly aliases: allow title -> name fallback and vice versa
    if not s.get("title") and s.get("name"):
        s["title"] = s["name"]

    return s

# =========================
# Helpers reused from v4 (copied)
# =========================

def _is_domain_trace_type(t: str) -> bool:
    return t in {"pie","treemap","sunburst","icicle","table","funnelarea",
                 "parcats","parcoords","sankey","indicator"}

def load_any_csv(csv_path: str, try_parse_dates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[c] = coerced
    if try_parse_dates:
        for c in df.columns:
            if df[c].dtype == "object":
                import warnings as _warn
                with _warn.catch_warnings():
                    _warn.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.7:
                    df[c] = parsed
    return df

def infer_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    dates = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    numerics = df.select_dtypes(include=["number"]).columns.tolist()
    categoricals = [c for c in df.columns if c not in numerics and c not in dates]
    return {"dates": dates, "numerics": numerics, "categoricals": categoricals, "all": df.columns.tolist()}

def apply_filters(df: pd.DataFrame, flt: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not flt: return df.copy()
    out = df.copy()
    if flt.get("query"): out = out.query(flt["query"])
    if flt.get("in"):
        for col, allowed in flt["in"].items():
            if col in out.columns: out = out[out[col].isin(allowed)]
    if flt.get("range"):
        for col, (lo, hi) in flt["range"].items():
            if col in out.columns: out = out[(out[col] >= lo) & (out[col] <= hi)]
    return out

# =========================
# Builders (same as v4 with light wrappers)
# =========================

def style_fig(fig: go.Figure, spec: Dict[str, Any]) -> go.Figure:
    fig.update_layout(title=spec.get("title"),
                      xaxis_title=spec.get("x_title"),
                      yaxis_title=spec.get("y_title"),
                      legend_title=spec.get("legend_title"),
                      template=spec.get("template", "plotly_white"),
                      margin=dict(l=40, r=20, t=60, b=40),
                      height=spec.get("height", 450),
                      legend=dict(
                          orientation=spec.get("legend_orientation", "h"),
                          y=spec.get("legend_y", None),
                          x=spec.get("legend_x", None)
                      ))
    if spec.get("x_tickformat"): fig.update_xaxes(tickformat=spec["x_tickformat"])  
    if spec.get("y_tickformat"): fig.update_yaxes(tickformat=spec["y_tickformat"])  
    return fig

def resolve_field(spec: Dict[str, Any], key: str, schema: Dict[str, List[str]], exclude: Optional[List[str]] = None):
    val = spec.get(key)
    if val and val not in ("auto", "auto_x", "auto_y", "auto_color"):
        return val
    exclude_set = set(exclude or [])
    if key == "x":
        for lst in [schema["dates"], schema["categoricals"], schema["numerics"]]:
            for c in lst:
                if c not in exclude_set: return c
    if key == "y":
        for c in schema["numerics"]:
            if c not in exclude_set: return c
    if key == "color":
        x = spec.get("x")
        for c in schema["categoricals"]:
            if c not in exclude_set and c != x: return c
    if key == "size":
        y = spec.get("y")
        for c in schema["numerics"]:
            if c not in exclude_set and c != y: return c
    return None

def expand_auto_aliases(spec: Dict[str, Any], schema: Dict[str, List[str]]):
    alias = {"auto_x": resolve_field(spec, "x", schema),
             "auto_y": resolve_field(spec, "y", schema),
             "auto_color": resolve_field(spec, "color", schema)}
    # Map $aliases to actual column keys, not display labels
    for k in (USER_ALIASES or {}).keys():
        alias[k.lstrip("$")] = k.lstrip("$")

    def map_value(v):
        if v is None: return v
        key = str(v)
        if key.startswith("$"): 
            bare_key = key[1:]
            # Return bare column name if alias not found, instead of keeping the $prefix
            return alias.get(bare_key, bare_key)
        return alias.get(key, v)

    def map_list(lst):
        return [map_value(x) for x in lst]

    # Map common scalar fields
    for k in ["x", "y", "color", "size", "value"]:
        if k in spec:
            spec[k] = map_value(spec[k])

    # Map list fields like treemap path
    if isinstance(spec.get("path"), list):
        spec["path"] = [map_value(p) for p in spec["path"]]

    # Map inside aggregates/top_n/pivot if present
    agg = spec.get("aggregate")
    if agg:
        if isinstance(agg.get("groupby"), list):
            agg["groupby"] = map_list(agg["groupby"])
        if isinstance(agg.get("measures"), list):
            agg["measures"] = map_list(agg["measures"])
        if isinstance(agg.get("measure"), str):
            agg["measure"] = map_value(agg["measure"])

    tn = spec.get("top_n")
    if tn:
        for k in ["by", "category", "group"]:
            if k in tn:
                tn[k] = map_value(tn[k])

    pv = spec.get("pivot")
    if pv:
        if isinstance(pv.get("index"), list):
            pv["index"] = map_list(pv["index"])
        for k in ["columns", "values"]:
            if k in pv:
                pv[k] = map_value(pv[k])

def aggregate(df: pd.DataFrame, groupby: List[str], measures: List[str], how) -> pd.DataFrame:
    if not groupby: groupby = []
    if isinstance(how, str):
        agg_map = {m: how for m in measures}
    elif isinstance(how, list):
        agg_map = {m: how[min(i, len(how)-1)] for i, m in enumerate(measures)}
    elif isinstance(how, dict):
        agg_map = how
    else:
        raise ValueError("aggregate.how must be str | list[str] | dict")
    gdf = df.groupby(groupby, dropna=False).agg(agg_map)
    if isinstance(gdf.columns, pd.MultiIndex):
        gdf.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in gdf.columns]
    else:
        gdf.columns = [str(c) for c in gdf.columns]
    return gdf.reset_index()


def prepare_chart_data(df: pd.DataFrame, spec: Dict[str, Any], schema: Dict[str, List[str]]) -> pd.DataFrame:
    data = apply_filters(df, spec.get("filter"))

    # 1) Derived columns (for row counts etc.)
    if "compute" in spec:
        params = spec.get("params", {})
        local_env = {k: v for k, v in params.items()}
        for new_col, expr in spec["compute"].items():
            data[new_col] = data.eval(expr, local_dict=local_env, engine="python")

    # 2) Top-N
    tn = spec.get("top_n")
    if isinstance(tn, dict):
        n = int(tn.get("n", 5))
        by = tn.get("by") or resolve_field(spec, "y", schema)
        category = tn.get("category") or resolve_field(spec, "color", schema) or resolve_field(spec, "x", schema)
        scope = tn.get("scope", "overall")
        group = tn.get("group") or resolve_field(spec, "x", schema)
        if by and by in data.columns and category and category in data.columns:
            if scope == "per_group" and group and group in data.columns:
                keep_mask = pd.Series(False, index=data.index)
                for _, sub in data.groupby(group):
                    winners = (sub.groupby(category)[by].sum().sort_values(ascending=False).head(n).index)
                    keep_mask |= data.index.isin(sub[sub[category].isin(winners)].index)
                data = data[keep_mask]
            else:
                tot = data.groupby(category, dropna=False)[by].sum().reset_index()
                keep = set(tot.sort_values(by=by, ascending=False).head(n)[category].tolist())
                data = data[data[category].isin(keep)]

    # 3) Aggregation — now robust when groupby is omitted
    agg_conf = spec.get("aggregate")
    if agg_conf:
        groupby = _ensure_list(agg_conf.get("groupby"))

        # If groupby missing, default to x (and color if present)
        if not groupby:
            candidates = [spec.get("x"), spec.get("color")]
            groupby = [c for c in candidates if c and c in data.columns]
            agg_conf["groupby"] = groupby  # write back so builders can use it too

        # Measures (fallback to y)
        measures = agg_conf.get("measures") or ([agg_conf["measure"]] if "measure" in agg_conf else [])
        if not measures:
            y_guess = resolve_field(spec, "y", schema)
            measures = [y_guess] if y_guess else []
            agg_conf["measures"] = measures

        how = agg_conf.get("how", "sum")

        if not groupby:
            # Global aggregate -> single row
            if not measures:
                return data  # nothing to aggregate
            g = data[measures].agg(how)
            data = pd.DataFrame([g.to_dict()]) if isinstance(g, pd.Series) else g.reset_index(drop=True)
        else:
            # Normal grouped aggregate
            for gcol in list(groupby):
                if gcol not in data.columns:
                    raise ValueError(f"groupby '{gcol}' not in columns")
            for m in measures:
                if m not in data.columns:
                    raise ValueError(f"measure '{m}' not in columns")
            data = aggregate(data, groupby, measures, how)

    # 4) Optional pivot
    if "pivot" in spec:
        pv = spec["pivot"]
        data = pd.pivot_table(
            data,
            index=pv.get("index", []),
            columns=pv["columns"],
            values=pv["values"],
            aggfunc=pv.get("aggfunc", "sum"),
            fill_value=pv.get("fill_value", 0),
        ).reset_index()

    # 5) Sorting
    sort = spec.get("sort")
    if sort and sort.get("by"):
        by = [sort["by"]] if isinstance(sort["by"], str) else sort["by"]
        try:
            data = data.sort_values(by=by, ascending=bool(sort.get("ascending", True)))
        except Exception:
            pass

    return data

# builders

def fallback_table(df: pd.DataFrame, spec: Dict[str, Any], schema: Dict[str, List[str]]) -> go.Figure:
    title = spec.get("title") or "Data preview"
    limit = int(spec.get("limit", 50))
    data = df.head(limit)
    header_vals = list(data.columns)
    cell_vals = [data[c] for c in data.columns]
    fig = go.Figure(data=[go.Table(header=dict(values=header_vals, align="left"),
                                   cells=dict(values=cell_vals, align="left"))])
    fig.update_layout(title=title, template="plotly_white", height=450, margin=dict(l=40, r=20, t=60, b=40))
    return fig

def build_line(df, spec, schema):
    spec = dict(spec)
    # prefer first groupby as x when user omitted x
    gb = _ensure_list((spec.get("aggregate") or {}).get("groupby"))
    if (spec.get("x") in (None, "auto", "auto_x")) and gb:
        spec["x"] = gb[0]

    spec.setdefault("x", resolve_field(spec, "x", schema))
    spec.setdefault("y", resolve_field(spec, "y", schema))
    expand_auto_aliases(spec, schema)
    if spec.get("x") is None or spec.get("y") is None: return fallback_table(df, spec, schema)

    data = prepare_chart_data(df, spec, schema)
    color = spec.get("color")
    if color not in data.columns: color = None

    fig = px.line(data, x=spec["x"], y=spec["y"], color=color, markers=spec.get("markers", True))
    return style_fig(fig, spec)


def build_bar(df, spec, schema):
    spec = dict(spec)
    gb = _ensure_list((spec.get("aggregate") or {}).get("groupby"))
    if (spec.get("x") in (None, "auto", "auto_x")) and gb:
        spec["x"] = gb[0]

    spec.setdefault("x", resolve_field(spec, "x", schema))
    spec.setdefault("y", resolve_field(spec, "y", schema, exclude=[spec["x"]]))
    expand_auto_aliases(spec, schema)
    if spec.get("x") is None or spec.get("y") is None: return fallback_table(df, spec, schema)

    # Robust aggregation for count and similar cases to avoid inflated axes and Chart Studio issues
    data = prepare_chart_data(df, spec, schema)
    x_name = spec.get("x")
    y_name = spec.get("y")
    agg_conf = spec.get("aggregate") or {}
    how = str((agg_conf.get("how") if isinstance(agg_conf, dict) else spec.get("agg")) or "").lower()

    try:
        if x_name in data.columns:
            data = data[data[x_name].notna()].copy()
        # Determine if we must compute counts from raw rows
        from pandas.api.types import is_numeric_dtype as _isnum
        has_y = y_name in data.columns
        x_has_dupes = x_name in data.columns and (data[x_name].nunique(dropna=False) < len(data))
        needs_count = (how == "count") and (not has_y or not _isnum(data[y_name]) or x_has_dupes)
        if needs_count:
            # Compute one-row-per-category counts
            if y_name in data.columns:
                agg_df = data.groupby(x_name, dropna=False)[y_name].count().reset_index(name="value")
            else:
                agg_df = data.groupby(x_name, dropna=False).size().reset_index(name="value")
            agg_df = agg_df.sort_values("value", ascending=False, kind="stable")
            color = spec.get("color")
            if color == x_name or color not in agg_df.columns:
                color = None
            barmode = "stack" if spec.get("stack", False) else "group"
            fig = px.bar(agg_df, x=x_name, y="value", color=color, barmode=barmode,
                         text_auto=spec.get("text_auto", False))
            fig.update_yaxes(title_text="count")
            return style_fig(fig, spec)
        else:
            # Numeric y with optional sum/avg/min/max OR already-counted values
            if has_y:
                data[y_name] = pd.to_numeric(data[y_name], errors="coerce")
                data = data[data[y_name].notna()]
            if how in ("sum", "avg", "mean", "min", "max") and has_y:
                func = "mean" if how in ("avg", "mean") else how
                agg_df = data.groupby(x_name, dropna=False)[y_name].agg(func).reset_index(name="value")
                agg_df = agg_df.sort_values("value", ascending=False, kind="stable")
                color = spec.get("color")
                if color == x_name or color not in agg_df.columns:
                    color = None
                barmode = "stack" if spec.get("stack", False) else "group"
                fig = px.bar(agg_df, x=x_name, y="value", color=color, barmode=barmode,
                             text_auto=spec.get("text_auto", False))
                fig.update_yaxes(title_text=y_name)
                return style_fig(fig, spec)
            # No explicit aggregation: just plot cleaned data
            if has_y and "sort" not in spec:
                data = data.sort_values(by=y_name, ascending=False, kind="stable")
            color = spec.get("color")
            if color not in data.columns:
                color = None
            barmode = "stack" if spec.get("stack", False) else "group"
            fig = px.bar(data, x=x_name, y=y_name if has_y else None, color=color, barmode=barmode,
                         text_auto=spec.get("text_auto", False))
            return style_fig(fig, spec)
    except Exception:
        pass

    # Fallback: try plotting whatever we have
    color = spec.get("color") if spec.get("color") in data.columns else None
    barmode = "stack" if spec.get("stack", False) else "group"
    fig = px.bar(data, x=x_name, y=y_name, color=color, barmode=barmode,
                 text_auto=spec.get("text_auto", False))
    # Enforce stable categorical ordering for readability across backends
    try:
        fig.update_xaxes(categoryorder="total descending")
    except Exception:
        pass
    return style_fig(fig, spec)

def build_scatter(df, spec, schema):
    spec = dict(spec)
    gb = _ensure_list((spec.get("aggregate") or {}).get("groupby"))
    if (spec.get("x") in (None, "auto", "auto_x")) and gb:
        spec["x"] = gb[0]

    spec.setdefault("x", resolve_field(spec, "x", schema))
    spec.setdefault("y", resolve_field(spec, "y", schema, exclude=[spec["x"]]))
    expand_auto_aliases(spec, schema)
    if spec.get("x") is None or spec.get("y") is None: return fallback_table(df, spec, schema)

    data = prepare_chart_data(df, spec, schema)
    color = spec.get("color");  size = spec.get("size")
    if color not in data.columns: color = None
    if size  not in data.columns: size  = None

    fig = px.scatter(data, x=spec["x"], y=spec["y"], color=color, size=size,
                     hover_data=spec.get("hover", []),
                     trendline="ols" if spec.get("trendline") else None)
    return style_fig(fig, spec)

def build_heatmap(df, spec, schema):
    spec = dict(spec)
    spec.setdefault("x", resolve_field(spec, "x", schema))
    spec.setdefault("y", resolve_field(spec, "y", schema, exclude=[spec["x"]]))
    spec["value"] = spec.get("value") or resolve_field({"y": "auto"}, "y", schema, exclude=[spec["x"]])
    expand_auto_aliases(spec, schema)
    if not spec.get("x") or not spec.get("y") or not spec.get("value"): return fallback_table(df, spec, schema)
    data = prepare_chart_data(df, spec, schema)
    x, y, v = spec["x"], spec["y"], spec["value"]
    if v not in data.columns or x not in data.columns or y not in data.columns: return fallback_table(df, spec, schema)
    pivot = pd.pivot_table(data, index=y, columns=x, values=v, aggfunc="sum", fill_value=0.0)
    if pivot.empty: return fallback_table(df, spec, schema)
    fig = px.imshow(pivot.values, x=[str(c) for c in pivot.columns], y=pivot.index.astype(str),
                    aspect="auto", origin="lower", labels=dict(x=x, y=y, color=v))
    if isinstance(spec.get("colorbar"), dict):
        fig.update_traces(colorbar=spec["colorbar"])
    fig.update_traces(colorbar_title=v)
    return style_fig(fig, spec)

def build_treemap(df, spec, schema):
    spec = dict(spec)
    spec.setdefault("value", resolve_field(spec, "y", schema))
    if "path" not in spec:
        cat = resolve_field(spec, "color", schema)
        spec["path"] = [cat] if cat else [resolve_field(spec, "x", schema)]
    expand_auto_aliases(spec, schema)
    if not spec.get("value"): return fallback_table(df, spec, schema)
    data = prepare_chart_data(df, spec, schema)
    path = [p for p in spec["path"] if p in data.columns]
    if not path or spec["value"] not in data.columns: return fallback_table(df, spec, schema)
    try:
        fig = px.treemap(data, path=path, values=spec["value"])
    except Exception:
        return fallback_table(df, spec, schema)
    return style_fig(fig, spec)

def build_table(df, spec, schema):
    spec = dict(spec)
    data = prepare_chart_data(df, spec, schema) if spec.get("aggregate") or spec.get("filter") else df
    limit = int(spec.get("limit", 20))
    data = data.head(limit)
    header_vals = list(data.columns)
    cell_vals = [data[c] for c in data.columns]
    fig = go.Figure(data=[go.Table(header=dict(values=header_vals, align="left"),
                                   cells=dict(values=cell_vals, align="left"))])
    return style_fig(fig, spec)

def build_histogram(df, spec, schema):
    spec = dict(spec)
    # x defaults to any column (prefer numeric/date then categorical via resolve_field)
    spec.setdefault("x", resolve_field(spec, "x", schema))
    expand_auto_aliases(spec, schema)
    if not spec.get("x"):
        return fallback_table(df, spec, schema)

    # Use prepared data when user specified filters/compute/aggregate; else raw df
    data = prepare_chart_data(df, spec, schema) if (spec.get("filter") or spec.get("compute") or spec.get("aggregate")) else df

    x = spec.get("x")
    if x not in data.columns:
        return fallback_table(df, spec, schema)

    # ✅ Fix: Robust handling for Chart Studio publishing
    # - Coerce x to numeric (when possible)
    # - Drop nulls
    # - Provide a sensible default bin count
    try:
        data[x] = pd.to_numeric(data[x], errors="coerce")
    except Exception:
        pass
    data = data.dropna(subset=[x])
    if len(data) == 0:
        return fallback_table(df, spec, schema)

    color = spec.get("color") if spec.get("color") in data.columns else None

    # Default to count per bin; do not require y
    ycol = spec.get("y") if spec.get("y") in data.columns else None
    histfunc = spec.get("histfunc") or ("count" if ycol is None else "sum")

    nbins = spec.get("nbins") or 20
    barmode = "relative" if spec.get("stack") else ("overlay" if spec.get("overlay") else None)

    fig = px.histogram(data, x=x, y=ycol, color=color, nbins=nbins, histfunc=histfunc, barmode=barmode)
    return style_fig(fig, spec)

def build_pie(df, spec, schema):
    spec = dict(spec)
    # Names: prefer color, then x; count values by default
    spec.pop("value", None)
    names = spec.get("names") or spec.get("color") or spec.get("x")
    if not names:
        names = resolve_field(spec, "color", schema) or resolve_field(spec, "x", schema)

    expand_auto_aliases(spec, schema)

    # Prepare data (handles filters/aggregate/top_n)
    data = prepare_chart_data(df, spec, schema)

    # If names column missing after transforms, try to resolve post-alias
    if not names or names not in data.columns:
        names = spec.get("color") or spec.get("x")
        if not names or names not in data.columns:
            return fallback_table(df, spec, schema)

    # compute counts per category for simplicity
    counts = data.groupby(names, dropna=False).size().reset_index(name="count")
    data = counts
    values = "count"

    try:
        fig = px.pie(data, names=names, values=values, hole=spec.get("hole", 0))
    except Exception:
        return fallback_table(df, spec, schema)
    return style_fig(fig, spec)

CHART_BUILDERS = {
    "line": build_line,
    "bar": build_bar,
    "scatter": build_scatter,
    "heatmap": build_heatmap,
    "treemap": build_treemap,
    "table": build_table,
    "histogram": build_histogram,
    "pie": build_pie,
}

# =========================
# HTML assembly
# =========================

def figure_to_div(fig: go.Figure, include_js: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=("cdn" if include_js else False),
                       config={"displaylogo": False, "responsive": True})

def wrap_html(title: str, sections: List[Tuple[str, str]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    grid = "".join(f"""<section class=\"card\"><h2>{heading}</h2>{fig_div}</section>""" for heading, fig_div in sections)
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#fafafa}}
header{{padding:24px;background:#111;color:#fff}}
main{{padding:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}}
.card{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 12px rgba(0,0,0,.06)}}
h1{{margin:0 0 6px;font-size:24px}} h2{{margin:0 0 12px;font-size:18px;color:#333}}
.meta{{color:#bbb;font-size:12px}} footer{{text-align:center;padding:18px;color:#777}}
@media(max-width:480px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body>
<header><h1>{title}</h1><div class="meta">Generated at {now}</div></header>
<main><div class="grid">{grid}</div></main>
<footer>Plotly report • optionally published to Chart Studio</footer>
</body></html>"""

# =========================
# Chart Studio publishing
# =========================

def init_chart_studio(username: Optional[str], api_key: Optional[str], domain: Optional[str]):
    try:
        import chart_studio
        import chart_studio.tools as tls
        import chart_studio.plotly as py
    except Exception as e:
        raise RuntimeError("chart_studio package is required. pip install chart_studio") from e

    user = username or os.getenv("PLOTLY_USERNAME")
    key = api_key or os.getenv("PLOTLY_API_KEY")
    
    # Debug: Print what credentials were found
    # print(f"🔍 Chart Studio Debug - Username: {user}")
    # print(f"🔍 Chart Studio Debug - API Key: {'*' * len(key) if key else 'None'}")
    
    if not (user and key):
        raise RuntimeError("Missing Chart Studio credentials. Provide --cs-username/--cs-api-key or set PLOTLY_USERNAME and PLOTLY_API_KEY.")

    tls.set_credentials_file(username=user, api_key=key)
    if domain:
        chart_studio.tools.set_config_file(
            plotly_domain=domain,
            plotly_api_domain=domain,
            plotly_streaming_domain=domain
        )
    print(f"✅ Chart Studio credentials set successfully for user: {user}")
    return py

def publish_cs(py_module, fig: go.Figure, filename: str, sharing: str = "public", auto_open: bool = False) -> str:
    """Publish figure to Chart Studio and return the URL"""
    url = py_module.plot(fig, filename=filename, auto_open=auto_open, sharing=sharing)
    print(f"✅ Successfully published to Chart Studio: {url}")
    return url

# =========================
# v5 combined subplot builder (copied from v4 with our latest spacing/width features)
# =========================

def combine_figures_to_subplots(figs: List[go.Figure],
                                titles: List[str],
                                cols: int = 1,
                                title: str = "Report",
                                height_per_row: int = 600,
                                vspace: float = 0.20,
                                row_height_weights: Optional[List[float]] = None,
                                legend_orientation: str = "h",
                                legend_y: float = -0.08,
                                legend_x: float = 0.5,
                                hspace: float = 0.06,
                                total_width: Optional[int] = None,
                                col_width_weights: Optional[List[float]] = None) -> go.Figure:
    n = len(figs)
    if n == 0:
        raise ValueError("No figures to combine.")
    rows = ceil(n / max(1, cols))

    types: List[str] = []
    any_colorbar = False
    for f in figs:
        domain = any(_is_domain_trace_type(getattr(tr, "type", "")) for tr in f.data)
        types.append("domain" if domain else "xy")
        for tr in f.data:
            ttype = getattr(tr, "type", "")
            if getattr(tr, "showscale", False) or ttype in {"heatmap", "imshow", "surface", "choropleth", "densitymapbox"}:
                any_colorbar = True

    specs, k = [], 0
    for r in range(rows):
        row_specs = []
        for c in range(cols):
            row_specs.append({"type": types[k]} if k < n else {"type": "xy"})
            k += 1
        specs.append(row_specs)

    normalized_row_heights = None
    total_height_px = max(520, rows * height_per_row)
    if row_height_weights:
        weights = list(row_height_weights[:rows]) + [1.0] * max(0, rows - len(row_height_weights))
        sum_w = sum(w if w > 0 else 0.0001 for w in weights)
        normalized_row_heights = [w / sum_w for w in weights]
        total_height_px = int(height_per_row * sum_w)

    normalized_col_widths = None
    if cols > 1:
        if col_width_weights:
            cw = list(col_width_weights[:cols]) + [1.0] * max(0, cols - len(col_width_weights))
        else:
            est = [1.0] * cols
            for i, f in enumerate(figs):
                c = i % cols
                w = 1.0
                if any(_is_domain_trace_type(getattr(tr, "type", "")) for tr in f.data):
                    w = max(w, 1.25)
                if any(getattr(tr, "showscale", False) or getattr(tr, "type", "") in {"heatmap","imshow"} for tr in f.data):
                    w = max(w, 1.35)
                est[c] = max(est[c], w)
            cw = est
        s = sum(x if x > 0 else 0.0001 for x in cw)
        normalized_col_widths = [x / s for x in cw]

    fig_all = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=vspace,
        horizontal_spacing=hspace,
        row_heights=normalized_row_heights,
        column_widths=normalized_col_widths,
    )

    for i, f in enumerate(figs):
        r, c = i // cols + 1, i % cols + 1
        for tr in f.data:
            fig_all.add_trace(tr, row=r, col=c)
        if types[i] == "xy":
            xt = f.layout.xaxis.title.text if f.layout.xaxis and f.layout.xaxis.title else None
            yt = f.layout.yaxis.title.text if f.layout.yaxis and f.layout.yaxis.title else None
            if xt: fig_all.update_xaxes(title_text=xt, title_standoff=12, row=r, col=c)
            if yt: fig_all.update_yaxes(title_text=yt, title_standoff=12, row=r, col=c)

    right_margin = 40 + (90 if any_colorbar else 0)
    # fig_all.update_layout(
    #     title=title, title_x=0.02,
    #     template="plotly_white",
    #     height=total_height_px,
    #     width=total_width,
    #     margin=dict(l=70, r=right_margin, t=96, b=110),
    #     legend=dict(orientation=legend_orientation, yanchor="top", y=legend_y, xanchor="center", x=legend_x),
    #     showlegend=True,
    # )
    fig_all.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white",
        height=total_height_px,
        width=total_width,
        margin=dict(l=70, r=right_margin, t=96, b=110),
        legend=dict(orientation=legend_orientation, yanchor="top", y=legend_y, xanchor="center", x=legend_x),
        showlegend=True,
        title_font_size=22
    )
    for ann in fig_all.layout.annotations or []:
        ann.font.size = 13 if n > 4 else 15
        ann.yshift = 12 if rows <= 2 else 16

    # Ensure bar/histogram axes are linear and autorange to data; enforce category ordering
    try:
        for i, f in enumerate(figs):
            r, c = i // cols + 1, i % cols + 1
            has_bar = any(getattr(tr, "type", "") == "bar" for tr in f.data)
            has_hist = any(getattr(tr, "type", "") == "histogram" for tr in f.data)
            if has_bar or has_hist:
                fig_all.update_yaxes(type="linear", autorange=True, row=r, col=c)
                fig_all.update_xaxes(categoryorder="total descending", row=r, col=c)
    except Exception:
        pass

    return fig_all

# =========================
# Build report (v5)
# =========================

def build_report(df: pd.DataFrame,
                 guide: Dict[str, Any],
                 out_html: str,
                 publish_py=None,
                 publish_sharing: str = "public",
                 no_html: bool = False,
                 publish_combined: bool = False,
                 publish_cols: int = 1,
                 publish_row_height: int = 520,
                 publish_vspace: float = 0.20,
                 publish_auto_layout: bool = False,
                 publish_row_height_weights: Optional[List[float]] = None,
                 publish_hspace: float = 0.06,
                 publish_width: Optional[int] = None,
                 publish_col_width_weights: Optional[List[float]] = None) -> List[Tuple[str, str]]:
    global USER_ALIASES
    USER_ALIASES = guide.get("columns") or guide.get("aliases") or {}

    title = guide.get("report_title", "Plotly Report")
    charts = guide.get("charts", [])
    schema = infer_schema(df)

    sections: List[Tuple[str, str]] = []
    built_figs: List[go.Figure] = []
    built_names: List[str] = []

    include_js_next = True
    for i, raw in enumerate(charts, start=1):
        spec = normalize_v5_spec(raw, schema)
        ctype = spec.get("type")
        name = spec.get("name") or spec.get("title") or f"Chart {i}"
        if ctype not in CHART_BUILDERS:
            warnings.warn(f"Skipping '{name}': unsupported chart type '{ctype}'")
            continue
        try:
            fig = CHART_BUILDERS[ctype](df, spec, schema)
        except Exception as e:
            warnings.warn(f"Skipping '{name}' due to error: {e}")
            continue

        built_figs.append(fig)
        built_names.append(name)

        if not no_html:
            fig_div = figure_to_div(fig, include_js=include_js_next)
            include_js_next = False
            sections.append((name, fig_div))

    published: List[Tuple[str, str]] = []
    if publish_py is not None and built_figs:
        if publish_combined:
            try:
                # Build Chart Studio-safe figures directly from df/specs when possible
                def _cs_fig_from_spec(spec: Dict[str, Any]) -> Optional[go.Figure]:
                    try:
                        stype = (spec.get('type') or '').lower()
                        if stype == 'bar':
                            x_name = spec.get('x')
                            y_name = spec.get('y')
                            how = (spec.get('aggregate') or {}).get('how') or spec.get('agg') or ''
                            how = str(how).lower()
                            if x_name and how == 'count':
                                # Recompute robust counts from raw df
                                data0 = apply_filters(df, spec.get('filter'))
                                data0 = data0[data0[x_name].notna()] if x_name in data0.columns else data0
                                if y_name and y_name in data0.columns:
                                    counts = data0.groupby(x_name, dropna=False)[y_name].count().reset_index(name='value')
                                else:
                                    counts = data0.groupby(x_name, dropna=False).size().reset_index(name='value')
                                counts = counts.sort_values('value', ascending=False, kind='stable')
                                figb = px.bar(counts, x=x_name, y='value')
                                figb.update_yaxes(title_text='count', type='linear', autorange=True)
                                figb.update_xaxes(categoryorder='total descending')
                                return figb
                        if stype == 'histogram':
                            x = spec.get('x')
                            if x and x in df.columns:
                                data0 = apply_filters(df, spec.get('filter'))
                                vals = pd.to_numeric(data0[x], errors='coerce').dropna().tolist()
                                nbins = int(spec.get('nbins') or 20)
                                if len(vals) > 0:
                                    counts, edges = np.histogram(vals, bins=nbins)
                                    centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(counts))]
                                    figh = px.bar(x=[str(c) for c in centers], y=[float(c) for c in counts])
                                    figh.update_yaxes(title_text='count', type='linear', autorange=True)
                                    return figh
                        return None
                    except Exception:
                        return None

                # Sanitize figures for Chart Studio (avoid typed arrays, enforce simple axes) when we cannot rebuild
                def _sanitize_for_chart_studio(fig_in: go.Figure) -> go.Figure:
                    try:
                        f = go.Figure(fig_in)  # shallow copy is fine here
                        # Convert histograms to plain bars with precomputed counts
                        if any(getattr(tr, 'type', '') == 'histogram' for tr in f.data):
                            # Use the first histogram trace as source
                            src = next(tr for tr in f.data if getattr(tr, 'type', '') == 'histogram')
                            x_vals = []
                            try:
                                x_vals = [float(v) for v in list(getattr(src, 'x', []) or []) if v is not None]
                            except Exception:
                                x_vals = []
                            nbins = int(getattr(src, 'nbinsx', 20) or 20)
                            if x_vals:
                                counts, edges = np.histogram(x_vals, bins=nbins)
                                centers = [float((edges[i] + edges[i+1]) / 2.0) for i in range(len(counts))]
                                newf = go.Figure()
                                newf.add_bar(x=[str(c) for c in centers], y=[float(c) for c in counts])
                                newf.update_layout(f.layout)
                                newf.update_yaxes(type='linear', autorange=True)
                                newf.update_xaxes(categoryorder='array')
                                return newf
                        # For bar charts, collapse to one trace with summed values per category
                        if any(getattr(tr, 'type', '') == 'bar' for tr in f.data):
                            from collections import defaultdict
                            acc = defaultdict(float)
                            for tr in f.data:
                                if getattr(tr, 'type', '') != 'bar':
                                    continue
                                x_list = list(getattr(tr, 'x', []) or [])
                                y_list = list(getattr(tr, 'y', []) or [])
                                for xv, yv in zip(x_list, y_list):
                                    try:
                                        yv = float(yv)
                                    except Exception:
                                        continue
                                    key = str(xv) if xv is not None else ''
                                    acc[key] += yv
                            if acc:
                                xs = sorted(acc.keys(), key=lambda k: -acc[k])
                                ys = [float(acc[k]) for k in xs]
                                newf = go.Figure()
                                newf.add_bar(x=xs, y=ys)
                                newf.update_layout(f.layout)
                                newf.update_yaxes(type='linear', autorange=True)
                                newf.update_xaxes(categoryorder='total descending')
                                return newf
                        # Generic sanitization: force plain lists
                        for tr in f.data:
                            if hasattr(tr, 'x') and getattr(tr, 'x', None) is not None:
                                try:
                                    tr.x = [str(v) if v is not None else None for v in list(tr.x)]
                                except Exception:
                                    pass
                            if hasattr(tr, 'y') and getattr(tr, 'y', None) is not None:
                                try:
                                    tr.y = [float(v) if v is not None else None for v in list(tr.y)]
                                except Exception:
                                    pass
                        f.update_yaxes(type='linear', autorange=True)
                        f.update_xaxes(categoryorder='total descending')
                        return f
                    except Exception:
                        return fig_in

                n = len(built_figs)
                cols = publish_cols
                if publish_auto_layout:
                    if n == 1: cols = 1
                    elif n <= 4: cols = 2
                    elif n <= 9: cols = 3
                    else: cols = 4
                rows = ceil(n / max(1, cols))
                weights = publish_row_height_weights
                if weights is None:
                    chart_weights = [float(ch.get("height_weight", 1.0)) for ch in charts[:n]]
                    weights = []
                    for r in range(rows):
                        start = r * cols
                        end = min(start + cols, n)
                        if start < end:
                            weights.append(max(chart_weights[start:end]))
                vspace = publish_vspace
                hspace = publish_hspace
                base_row_h = publish_row_height
                total_width = publish_width
                col_width_weights = publish_col_width_weights
                if publish_auto_layout:
                    if rows <= 1:
                        vspace = 0.26; base_row_h = max(base_row_h, 680)
                    elif rows == 2:
                        vspace = 0.20; base_row_h = max(base_row_h, 620)
                    elif rows == 3:
                        vspace = 0.14; base_row_h = max(base_row_h, 560)
                    else:
                        vspace = 0.10; base_row_h = max(base_row_h, 520)
                    if cols == 1: hspace = 0.06
                    elif cols == 2: hspace = 0.08
                    elif cols == 3: hspace = 0.09
                    else: hspace = 0.10
                    if total_width is None:
                        base_col_w = 560 if cols == 1 else (540 if cols == 2 else 520)
                        total_width = max(960, cols * base_col_w + 100)
                    if col_width_weights is None and cols > 1:
                        col_weight_est = [1.0] * cols
                        for i, f in enumerate(built_figs):
                            c = i % cols
                            w = 1.0
                            if any(getattr(tr, "type", "") in {"heatmap","imshow"} for tr in f.data):
                                w = max(w, 1.35)
                            if any(_is_domain_trace_type(getattr(tr, "type", "")) for tr in f.data):
                                w = max(w, 1.20)
                            col_weight_est[c] = max(col_weight_est[c], w)
                        col_width_weights = col_weight_est

                # Prefer rebuilding CS-safe figures per spec when possible
                cs_figs: List[go.Figure] = []
                for spec, f in zip(charts, built_figs):
                    rebuilt = _cs_fig_from_spec(spec)
                    cs_figs.append(rebuilt if rebuilt is not None else _sanitize_for_chart_studio(f))
                sanitized_figs = cs_figs
                combo = combine_figures_to_subplots(
                    sanitized_figs, built_names,
                    cols=cols,
                    title=title,
                    height_per_row=base_row_h,
                    vspace=vspace,
                    row_height_weights=weights,
                    legend_orientation="h",
                    legend_y=-0.10,
                    legend_x=0.5,
                    hspace=hspace,
                    total_width=total_width,
                    col_width_weights=col_width_weights,
                )
                fname = (guide.get("filename") or title).replace(" ", "-").lower()
                url = publish_cs(publish_py, combo, filename=fname, sharing=publish_sharing, auto_open=False)
                published.append((title, url))
                print(f"📤 Published combined report: {url}")
            except Exception as e:
                warnings.warn(f"Failed to publish combined figure: {e}")
        else:
            for name, fig in zip(built_names, built_figs):
                try:
                    fname = name.replace(" ", "-").lower()
                    url = publish_cs(publish_py, fig, filename=fname, sharing=publish_sharing, auto_open=False)
                    published.append((name, url))
                    print(f"📤 Published '{name}': {url}")
                except Exception as e:
                    warnings.warn(f"Failed to publish '{name}': {e}")

    if not no_html:
        if not sections and published:
            links = "".join(f"<li><a href='{url}' target='_blank' rel='noopener'>{name}</a></li>"
                            for name, url in published)
            placeholder = f"<section class='card'><h2>Published Charts</h2><ul>{links}</ul></section>"
            sections.append(("Published Charts", placeholder))
        if not sections:
            fallback_fig = fallback_table(df, {"title": "Data preview"}, schema)
            sections.append(("Data preview", figure_to_div(fallback_fig, include_js=True)))

        html = wrap_html(title, sections)
        Path(os.path.dirname(out_html) or ".").mkdir(parents=True, exist_ok=True)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ Wrote report: {out_html}")

    return published

# =========================
# Config loader (YAML/JSON)
# =========================

def load_guide(config_path: str) -> Dict[str, Any]:
    ext = os.path.splitext(config_path.lower())[1]
    with open(config_path, "r", encoding="utf-8") as f:
        if ext in [".yml", ".yaml"]:
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError("Config must be .yaml/.yml or .json")

# =========================
# CLI (mirrors v4, defaults to charts_v5.yaml)
# =========================

def main():
    p = argparse.ArgumentParser(description="Plotly report builder (v5 simplified schema). Publish to Chart Studio or write HTML.")
    p.add_argument("--csv", help="Path to CSV file.")
    p.add_argument("--config", required=False, default="charts_v5.yaml", help="Path to YAML/JSON chart guide (default: charts_v5.yaml).")
    p.add_argument("--out", default="out/plotly_index.html", help="Output HTML path (default: out/plotly_index.html).")
    p.add_argument("--no-html", action="store_true", help="Do not write local HTML (publish only).")

    # Combined figure options
    p.add_argument("--cs-single", action="store_true", help="Publish ONE combined multi-panel figure (single URL) instead of one per chart.")
    p.add_argument("--cs-cols", type=int, default=1, help="Columns in the combined Chart Studio figure (default: 1 = vertical stack).")
    p.add_argument("--cs-vspace", type=float, default=0.20, help="Vertical spacing between subplot rows (0–1).")
    p.add_argument("--cs-row-height", type=int, default=520, help="Height per row (px) for the combined figure (default: 520).")
    p.add_argument("--cs-auto-layout", action="store_true", help="Automatically determine combined layout (cols, row heights, vspace) based on chart count and types.")
    p.add_argument("--cs-row-heights", help="Comma-separated row height weights (e.g., '1,2,1'). Overrides auto weights.")
    p.add_argument("--cs-hspace", type=float, default=0.06, help="Horizontal spacing between subplot columns (0–1).")
    p.add_argument("--cs-width", type=int, help="Total width (px) of the combined figure. If omitted, auto when --cs-auto-layout.")
    p.add_argument("--cs-col-widths", help="Comma-separated column width weights (e.g., '1,1.3,1'). Optional, only when --cs-cols>1.")

    # Chart Studio
    p.add_argument("--publish-chartstudio", action="store_true", help="Publish each chart to your Chart Studio account.")
    p.add_argument("--sharing", default="public", choices=["public","private","secret"], help="Chart privacy for published charts (default: public).")
    p.add_argument("--cs-username", help="Chart Studio username (or set env PLOTLY_USERNAME).")
    p.add_argument("--cs-api-key", help="Chart Studio API key (or set env PLOTLY_API_KEY).")
    p.add_argument("--cs-domain", help="Chart Studio Enterprise domain (e.g., https://plotly.your-company.com)")

    # Dummy data
    p.add_argument("--make-dummy", action="store_true", help="Generate a dummy CSV (and exit) at the --csv path.")
    p.add_argument("--rows", type=int, default=500, help="Rows for dummy CSV (default 500).")
    p.add_argument("--cats", help="Comma-separated labels for the category column (e.g., 'North,South').")
    p.add_argument("--col2", default="Category", help="Name of the categorical column for dummy data (default: Category)")

    args = p.parse_args()

    if args.make_dummy:
        if not args.csv: sys.exit("--csv is required when using --make-dummy")
        cats = [s.strip() for s in args.cats.split(",")] if args.cats else None
        # reuse v4 dummy behavior
        np.random.seed(7)
        DEFAULT_CATS = [f"Category_{i}" for i in range(1, 11)]
        cats = cats or DEFAULT_CATS
        data = []
        for _ in range(args.rows):
            year = np.random.randint(2020, 2026)
            cat = np.random.choice(cats)
            base = np.random.lognormal(mean=10.0, sigma=0.25)
            bias = (cats.index(cat) + 1) * 1000
            drift = (year - 2020) * 3000
            val = round(base / 500 + bias + drift + np.random.normal(0, 2000), 2)
            data.append((year, cat, max(0.0, val)))
        df_dummy = pd.DataFrame(data, columns=["Year", args.col2, "Sales"])
        Path(os.path.dirname(args.csv) or ".").mkdir(parents=True, exist_ok=True)
        df_dummy.to_csv(args.csv, index=False)
        print(f"🧪 Dummy CSV written: {args.csv} (rows={len(df_dummy)}; col2='{args.col2}'; cats={len(cats)})")
        return

    if not args.csv: sys.exit("--csv is required (or use --make-dummy to create one)")
    if not args.config and not args.publish_chartstudio:
        sys.exit("--config is required unless you only wanted to generate dummy data.")

    df = load_any_csv(args.csv)
    guide = load_guide(args.config) if args.config else {"report_title":"No Charts","charts":[]}

    publish_py = None
    if args.publish_chartstudio:
        publish_py = init_chart_studio(args.cs_username, args.cs_api_key, args.cs_domain)

    # Parse weights if provided
    row_height_weights = None
    if args.cs_row_heights:
        try:
            row_height_weights = [float(x.strip()) for x in args.cs_row_heights.split(',') if x.strip()]
            if not row_height_weights:
                row_height_weights = None
        except Exception:
            warnings.warn("Ignoring --cs-row-heights; failed to parse comma-separated floats.")
            row_height_weights = None

    col_width_weights = None
    if args.cs_col_widths:
        try:
            col_width_weights = [float(x.strip()) for x in args.cs_col_widths.split(',') if x.strip()]
            if not col_width_weights:
                col_width_weights = None
        except Exception:
            warnings.warn("Ignoring --cs-col-widths; failed to parse comma-separated floats.")
            col_width_weights = None

    urls = build_report(df, guide, args.out,
                        publish_py=publish_py,
                        publish_sharing=args.sharing,
                        no_html=args.no_html,
                        publish_combined=args.cs_single,
                        publish_cols=args.cs_cols,
                        publish_row_height=args.cs_row_height,
                        publish_vspace=args.cs_vspace,
                        publish_auto_layout=args.cs_auto_layout,
                        publish_row_height_weights=row_height_weights,
                        publish_hspace=args.cs_hspace,
                        publish_width=args.cs_width,
                        publish_col_width_weights=col_width_weights)

    if urls:
        print("\n=== Chart Studio URLs ===")
        for name, url in urls:
            print(f"- {name}: {url}")

if __name__ == "__main__":
    main()
