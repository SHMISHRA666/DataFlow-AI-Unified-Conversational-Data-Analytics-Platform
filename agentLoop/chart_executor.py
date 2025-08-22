# chart_executor.py - ChartExecutorAgent for executing generated visualization code

import asyncio
import os
import sys
import subprocess
import tempfile
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from agentLoop.agents import AgentRunner
from utils.utils import log_step, log_error
import importlib.util


class ChartExecutor:
    """
    ChartExecutor handles the safe execution of generated visualization code
    and creates actual chart files from GenerationAgent output.
    """
    
    def __init__(self, output_directory: str = "generated_charts"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Create subdirectories for different formats
        (self.output_directory / "png").mkdir(exist_ok=True)
        (self.output_directory / "svg").mkdir(exist_ok=True)
        (self.output_directory / "html").mkdir(exist_ok=True)
        (self.output_directory / "pdf").mkdir(exist_ok=True)
        
        # Required packages for visualization
        self.required_packages = [
            "matplotlib", "plotly", "seaborn", "pandas", "numpy"
        ]
        
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required visualization packages are available"""
        available = {}
        for package in self.required_packages:
            try:
                importlib.import_module(package)
                available[package] = True
            except ImportError:
                available[package] = False
        return available
    
    def install_missing_dependencies(self, missing_packages: List[str]) -> bool:
        """Install missing packages using uv or pip"""
        if not missing_packages:
            return True
            
        try:
            for package in missing_packages:
                log_step(f"Installing {package}...", symbol="📦")
                
                # Try uv first (since the project uses uv)
                try:
                    result = subprocess.run(["uv", "add", package], 
                                          capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        log_step(f"✅ Installed {package} using uv", symbol="✅")
                        continue
                    else:
                        log_step(f"uv add failed for {package}: {result.stderr}", symbol="⚠️")
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    log_step(f"uv not available or failed, trying pip...", symbol="⚠️")
                
                # Try pip installation methods
                pip_methods = [
                    # Method 1: Direct pip with current Python
                    [sys.executable, "-m", "pip", "install", package],
                    # Method 2: Try ensuring pip is available first
                    ["python", "-m", "ensurepip", "--default-pip"],
                    # Method 3: Try with uv pip
                    ["uv", "pip", "install", package],
                    # Method 4: Try system pip
                    ["pip", "install", package]
                ]
                
                success = False
                for i, method in enumerate(pip_methods):
                    try:
                        if i == 1:  # ensurepip method
                            subprocess.run(method, capture_output=True, text=True, timeout=30)
                            # After ensuring pip, try installing the package
                            subprocess.check_call(["python", "-m", "pip", "install", package], timeout=60)
                        else:
                            subprocess.check_call(method, timeout=60)
                        
                        log_step(f"✅ Installed {package} using method {i+1}", symbol="✅")
                        success = True
                        break
                        
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                        if i < len(pip_methods) - 1:
                            continue  # Try next method
                        else:
                            log_error(f"All installation methods failed for {package}: {e}")
                
                if not success:
                    log_error(f"Failed to install {package} with all methods")
                    return False
                    
            return True
            
        except Exception as e:
            log_error(f"Unexpected error during package installation: {e}")
            return False
    
    def build_df_loader_preamble(self, data_context: Dict[str, Any]) -> str:
        """Build Python code to load a DataFrame named df from data_context.

        Supported keys (first match wins):
          - df_csv_path: str (+ optional csv_read_kwargs: dict)
          - df_tsv_path: str (+ optional tsv_read_kwargs: dict)
          - df_excel_path: str (+ optional excel_read_kwargs: dict)
          - df_parquet_path: str
          - df_feather_path: str
          - df_pickle_path: str
          - df_json_records_path: str (expects lines JSON)
          - df_json_path: str (regular JSON)
          - df_path: str (generic path; infer by extension; supports csv, tsv, xlsx/xls, parquet, feather, json)
        """
        if not data_context:
            # Fail fast to prevent silent sample data usage
            return (
                "df = None\n"
                "raise RuntimeError('No data_context provided to ChartExecutor; cannot materialize df')\n"
            )

        def _q(path: str) -> str:
            # Quote a filesystem path safely for Python source
            return repr(str(path))

        if isinstance(data_context, dict):
            if data_context.get("df_excel_path"):
                excel_kwargs = data_context.get("excel_read_kwargs") or {}
                import json as _json
                excel_kwargs_literal = _json.dumps(excel_kwargs)
                return (
                    f"excel_path = {_q(data_context['df_excel_path'])}\n"
                    f"_excel_kwargs = {excel_kwargs_literal}\n"
                    "df = pd.read_excel(excel_path, **_excel_kwargs)\n"
                )
            if data_context.get("df_tsv_path"):
                tsv_kwargs = data_context.get("tsv_read_kwargs") or {}
                import json as _json
                tsv_kwargs_literal = _json.dumps(tsv_kwargs)
                return (
                    f"tsv_path = {_q(data_context['df_tsv_path'])}\n"
                    f"_tsv_kwargs = {tsv_kwargs_literal}\n"
                    "df = pd.read_csv(tsv_path, sep='\t', **_tsv_kwargs)\n"
                )
            if data_context.get("df_parquet_path"):
                return (
                    f"parquet_path = {_q(data_context['df_parquet_path'])}\n"
                    "df = pd.read_parquet(parquet_path)\n"
                )
            if data_context.get("df_feather_path"):
                return (
                    f"feather_path = {_q(data_context['df_feather_path'])}\n"
                    "df = pd.read_feather(feather_path)\n"
                )
            if data_context.get("df_pickle_path"):
                return (
                    f"pickle_path = {_q(data_context['df_pickle_path'])}\n"
                    "df = pd.read_pickle(pickle_path)\n"
                )
            if data_context.get("df_json_records_path"):
                return (
                    f"json_path = {_q(data_context['df_json_records_path'])}\n"
                    "df = pd.read_json(json_path, lines=True)\n"
                )
            if data_context.get("df_json_path"):
                return (
                    f"json_path = {_q(data_context['df_json_path'])}\n"
                    "df = pd.read_json(json_path)\n"
                )
            if data_context.get("df_csv_path"):
                csv_kwargs = data_context.get("csv_read_kwargs") or {}
                # Serialize kwargs to literal Python
                import json as _json
                kwargs_literal = _json.dumps(csv_kwargs)
                return (
                    f"csv_path = {_q(data_context['df_csv_path'])}\n"
                    f"_csv_kwargs = {kwargs_literal}\n"
                    "df = pd.read_csv(csv_path, **_csv_kwargs)\n"
                )
            if data_context.get("df_path"):
                csv_kwargs = data_context.get("csv_read_kwargs") or {}
                tsv_kwargs = data_context.get("tsv_read_kwargs") or {}
                excel_kwargs = data_context.get("excel_read_kwargs") or {}
                json_kwargs = data_context.get("json_read_kwargs") or {}
                import json as _json
                _csv_kwargs = _json.dumps(csv_kwargs)
                _tsv_kwargs = _json.dumps(tsv_kwargs)
                _excel_kwargs = _json.dumps(excel_kwargs)
                _json_kwargs = _json.dumps(json_kwargs)
                return (
                    f"_generic_path = {_q(data_context['df_path'])}\n"
                    f"_csv_kwargs = {_csv_kwargs}\n"
                    f"_tsv_kwargs = {_tsv_kwargs}\n"
                    f"_excel_kwargs = {_excel_kwargs}\n"
                    f"_json_kwargs = {_json_kwargs}\n"
                    "_ext = Path(_generic_path).suffix.lower()\n"
                    "if _ext in ['.csv']:\n"
                    "    df = pd.read_csv(_generic_path, **_csv_kwargs)\n"
                    "elif _ext in ['.tsv']:\n"
                    "    df = pd.read_csv(_generic_path, sep='\t', **_tsv_kwargs)\n"
                    "elif _ext in ['.xlsx', '.xls']:\n"
                    "    df = pd.read_excel(_generic_path, **_excel_kwargs)\n"
                    "elif _ext == '.json':\n"
                    "    df = pd.read_json(_generic_path, **_json_kwargs)\n"
                    "elif _ext == '.parquet':\n"
                    "    df = pd.read_parquet(_generic_path)\n"
                    "elif _ext == '.feather':\n"
                    "    df = pd.read_feather(_generic_path)\n"
                    "else:\n"
                    "    raise RuntimeError(f'Unsupported file extension for df_path: {_ext}')\n"
                )

        return (
            "df = None\n"
            "raise RuntimeError('Unsupported or incomplete data_context; provide one of df_*_path keys')\n"
        )
    
    def enhance_code_for_execution(self, original_code: str, chart_id: str, data_context: Dict[str, Any], fallback_mode: bool = False) -> str:
        """Enhance generated code for safe execution and file output"""
        
        if fallback_mode:
            # Simple matplotlib-only fallback
            enhanced_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Set up output paths
chart_id = "{chart_id}"
output_dir = Path("{self.output_directory}")
png_path = output_dir / "png" / f"{{chart_id}}.png"
svg_path = output_dir / "svg" / f"{{chart_id}}.svg"

# DataFrame loading from pipeline-provided data_context
{self.build_df_loader_preamble(data_context)}

try:
    # Generic matplotlib fallbacks using simple heuristics on df
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if "px.bar" in '''{original_code}''':
        # Bar chart: aggregate first numeric by first non-numeric
        if not numeric_cols:
            raise ValueError('No numeric columns available for bar chart fallback')
        cat_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        plt.bar(grouped.index.astype(str), grouped.values)
        plt.title(f'{{num_col}} by {{cat_col}}')
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
    elif "px.line" in '''{original_code}''':
        # Line chart: plot first numeric over first datetime (or index)
        if not numeric_cols:
            raise ValueError('No numeric columns available for line chart fallback')
        x_col = datetime_cols[0] if datetime_cols else (df.columns[0] if len(df.columns) > 0 else None)
        y_col = numeric_cols[0]
        if x_col is None:
            raise ValueError('No suitable x-axis column for line chart fallback')
        series = df.sort_values(by=x_col).set_index(x_col)[y_col]
        plt.figure(figsize=(10, 6))
        plt.plot(series.index, series.values)
        plt.title(f'{{y_col}} over {{x_col}}')
        plt.xlabel(str(x_col))
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
    elif "px.pie" in '''{original_code}''':
        # Pie chart: share of first numeric by first non-numeric
        if not numeric_cols:
            raise ValueError('No numeric columns available for pie chart fallback')
        cat_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].sum()
        plt.figure(figsize=(8, 8))
        plt.pie(grouped.values, labels=grouped.index.astype(str), autopct='%1.1f%%')
        plt.title(f'{{num_col}} distribution by {{cat_col}}')
    else:
        # Try to naively execute original code with matplotlib replacements
{self._indent_code(original_code.replace("px.", "plt.").replace("fig.", "plt."), 8)}
    
    # Save the plot
    plt.savefig(str(png_path), dpi=300, bbox_inches='tight')
    plt.savefig(str(svg_path), format='svg', bbox_inches='tight')
    plt.close('all')
    
    print(f"SUCCESS: Chart saved to {{png_path}}")
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""
        else:
            # Full feature mode with all libraries
            enhanced_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    print("Plotly not available, using matplotlib fallback")
    px = None

try:
    import seaborn as sns
except ImportError:
    print("Seaborn not available")
    sns = None

# Set up output paths
chart_id = "{chart_id}"
output_dir = Path("{self.output_directory}")
png_path = output_dir / "png" / f"{{chart_id}}.png"
svg_path = output_dir / "svg" / f"{{chart_id}}.svg"
html_path = output_dir / "html" / f"{{chart_id}}.html"

# DataFrame loading from pipeline-provided data_context
{self.build_df_loader_preamble(data_context)}

try:
    # Execute original code
{self._indent_code(original_code, 4)}
    
    # Handle different visualization libraries
    if 'fig' in locals():
        # Plotly figure
        if hasattr(fig, 'write_image'):
            try:
                fig.write_image(str(png_path), width=800, height=600)
                fig.write_html(str(html_path))
                fig.write_image(str(svg_path), format='svg', width=800, height=600)
            except Exception as e:
                print(f"Plotly image export failed: {{e}}, saving as HTML only")
                fig.write_html(str(html_path))
        
        # Matplotlib figure
        elif hasattr(fig, 'savefig'):
            fig.savefig(str(png_path), dpi=300, bbox_inches='tight')
            fig.savefig(str(svg_path), format='svg', bbox_inches='tight')
            plt.close(fig)
    
    # Handle matplotlib pyplot
    elif plt.get_fignums():
        plt.savefig(str(png_path), dpi=300, bbox_inches='tight')
        plt.savefig(str(svg_path), format='svg', bbox_inches='tight')
        plt.close('all')
    
    # Handle seaborn plots
    elif 'ax' in locals() and hasattr(ax, 'figure'):
        ax.figure.savefig(str(png_path), dpi=300, bbox_inches='tight')
        ax.figure.savefig(str(svg_path), format='svg', bbox_inches='tight')
        plt.close(ax.figure)
    
    print(f"SUCCESS: Chart saved to {{png_path}}")
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""
        
        return enhanced_code
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Add indentation to code lines"""
        lines = code.split('\n')
        indented_lines = [' ' * spaces + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)
    
    async def execute_code_safely(self, code: str, chart_id: str) -> Dict[str, Any]:
        """Execute code in a safe environment and capture results"""
        
        start_time = time.time()
        execution_result = {
            "status": "failed",
            "output": "",
            "errors": [],
            "warnings": [],
            "execution_time": 0.0,
            "files_created": []
        }
        
        try:
            # Create temporary file for code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            execution_result["execution_time"] = time.time() - start_time
            execution_result["output"] = result.stdout
            
            if result.returncode == 0:
                execution_result["status"] = "success"
                
                # Check for created files
                for format_dir in ["png", "svg", "html"]:
                    file_path = self.output_directory / format_dir / f"{chart_id}.{format_dir}"
                    if file_path.exists():
                        execution_result["files_created"].append({
                            "format": format_dir,
                            "path": str(file_path),
                            "size_bytes": file_path.stat().st_size
                        })
            else:
                execution_result["status"] = "failed"
                execution_result["errors"].append(result.stderr)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
        except subprocess.TimeoutExpired:
            execution_result["errors"].append("Code execution timed out after 30 seconds")
        except Exception as e:
            execution_result["errors"].append(str(e))
            execution_result["execution_time"] = time.time() - start_time
        
        return execution_result
    
    async def process_generation_output(self, generation_output: Dict[str, Any], data_context: Dict[str, Any] = None, recommendations: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process GenerationAgent output and create actual charts"""
        
        log_step("🎨 Starting chart execution process", symbol="🚀")
        
        execution_summary = {
            "execution_status": "success",
            "charts_created": [],
            "executed_code": [],
            "data_processing": {
                "data_source": "pipeline_data_context",
                "records_processed": None,
                "data_preparation_steps": ["df_load_from_data_context"]
            },
            "performance_metrics": {
                "total_execution_time_seconds": 0.0,
                "charts_per_second": 0.0
            },
            "execution_summary": {
                "total_charts_requested": 0,
                "total_charts_created": 0,
                "success_rate": 0.0
            },
            "chart_data_summaries": []
        }
        
        start_time = time.time()
        
        # Check dependencies
        dependencies = self.check_dependencies()
        missing = [pkg for pkg, available in dependencies.items() if not available]
        
        if missing:
            log_step(f"Installing missing dependencies: {missing}", symbol="📦")
            if not self.install_missing_dependencies(missing):
                log_step("⚠️ Package installation failed, trying minimal fallback mode", symbol="🔄")
                # Try with minimal dependencies that might already be available
                minimal_deps = self.check_dependencies()
                if not minimal_deps.get("matplotlib", False):
                    log_error("❌ Matplotlib not available and installation failed. Cannot create charts.")
                    execution_summary["execution_status"] = "failed"
                    execution_summary["error"] = "Required dependencies not available and installation failed"
                    return execution_summary
                else:
                    log_step("✅ Using matplotlib fallback mode", symbol="🎨")
        
        # Validate data context early
        supported_keys = [
            "df_parquet_path", "df_csv_path", "df_tsv_path", "df_excel_path", "df_pickle_path", "df_json_records_path", "df_json_path", "df_feather_path", "df_path"
        ]
        if not data_context or not any(k in (data_context or {}) for k in supported_keys):
            log_error("❌ No valid data_context provided; expected one of df_*_path keys")
            return {
                "execution_status": "failed",
                "charts_created": [],
                "executed_code": [],
                "error": "Missing or invalid data_context; cannot materialize df"
            }
        
        # Prepare rec specs for numeric summaries
        rec_by_id: Dict[str, Any] = {}
        if isinstance(recommendations, dict):
            for rv in recommendations.get("recommended_visualizations", []) or []:
                if isinstance(rv, dict) and rv.get("id"):
                    rec_by_id[rv["id"]] = rv

        # Attempt to load df locally for summaries
        df_local: Optional[pd.DataFrame] = None
        try:
            df_local = self._load_df_from_context(data_context)
            if df_local is not None:
                try:
                    execution_summary["data_processing"]["records_processed"] = int(len(df_local))
                except Exception:
                    pass
        except Exception as e:
            log_error(f"Failed to load df for summaries: {e}")

        # Process visualizations from generation output
        generated_visualizations = generation_output.get("generated_visualizations", [])
        execution_summary["execution_summary"]["total_charts_requested"] = len(generated_visualizations)
        
        for i, viz in enumerate(generated_visualizations):
            chart_id = viz.get("id", f"chart_{i}")
            
            try:
                # Extract Python code
                python_code = viz.get("code", {}).get("python", "")
                if not python_code:
                    log_error(f"No Python code found for chart {chart_id}")
                    continue
                
                log_step(f"🎨 Executing chart: {chart_id}", symbol="⚙️")
                
                # Check if we need fallback mode
                current_deps = self.check_dependencies()
                fallback_mode = not all([current_deps.get("plotly", False), current_deps.get("seaborn", False)])
                
                if fallback_mode:
                    log_step(f"Using matplotlib fallback mode for {chart_id}", symbol="🔄")
                
                # Enhance code for execution
                enhanced_code = self.enhance_code_for_execution(python_code, chart_id, data_context, fallback_mode)
                
                # Execute code safely
                exec_result = await self.execute_code_safely(enhanced_code, chart_id)
                
                # Record execution details
                execution_summary["executed_code"].append({
                    "code_block_id": chart_id,
                    "original_code": python_code,
                    "executed_code": enhanced_code,
                    "execution_time_seconds": exec_result["execution_time"],
                    "status": exec_result["status"],
                    "output_captured": exec_result["output"],
                    "errors": exec_result["errors"],
                    "warnings": exec_result["warnings"]
                })
                
                # Process created files
                for file_info in exec_result["files_created"]:
                    chart_info = {
                        "chart_id": chart_id,
                        "title": viz.get("title", f"Chart {i+1}"),
                        "chart_type": viz.get("type", "unknown"),
                        "file_path": file_info["path"],
                        "file_format": file_info["format"],
                        "file_size_bytes": file_info["size_bytes"],
                        "generation_time_seconds": exec_result["execution_time"],
                        "interactive": file_info["format"] == "html"
                    }
                    
                    execution_summary["charts_created"].append(chart_info)
                    log_step(f"✅ Created {file_info['format'].upper()}: {file_info['path']}", symbol="📊")

                # Compute numeric summary for narrative support
                if df_local is not None:
                    spec = rec_by_id.get(chart_id, {}).get("spec", {})
                    try:
                        summary = self._compute_chart_summary(df_local, viz.get("type"), spec)
                        summary.update({"chart_id": chart_id, "title": viz.get("title", chart_id)})
                        execution_summary["chart_data_summaries"].append(summary)
                    except Exception as e:
                        log_error(f"Summary computation failed for {chart_id}: {e}")
                
            except Exception as e:
                log_error(f"Failed to execute chart {chart_id}: {e}")
                execution_summary["executed_code"].append({
                    "code_block_id": chart_id,
                    "status": "error",
                    "errors": [str(e)]
                })
        
        # Calculate summary metrics
        total_time = time.time() - start_time
        total_created = len(execution_summary["charts_created"])
        total_requested = execution_summary["execution_summary"]["total_charts_requested"]
        
        execution_summary["performance_metrics"]["total_execution_time_seconds"] = total_time
        execution_summary["performance_metrics"]["charts_per_second"] = total_created / total_time if total_time > 0 else 0
        
        execution_summary["execution_summary"]["total_charts_created"] = total_created
        execution_summary["execution_summary"]["success_rate"] = total_created / total_requested if total_requested > 0 else 0
        
        if total_created == 0:
            execution_summary["execution_status"] = "failed"
        elif total_created < total_requested:
            execution_summary["execution_status"] = "partial_success"
        
        log_step(f"🎉 Chart execution completed: {total_created}/{total_requested} charts created", symbol="✅")
        
        return execution_summary

    def _load_df_from_context(self, data_context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if not isinstance(data_context, dict):
            return None
        if data_context.get("df_excel_path"):
            return pd.read_excel(data_context["df_excel_path"], **(data_context.get("excel_read_kwargs") or {}))
        if data_context.get("df_csv_path"):
            return pd.read_csv(data_context["df_csv_path"], **(data_context.get("csv_read_kwargs") or {}))
        if data_context.get("df_tsv_path"):
            return pd.read_csv(data_context["df_tsv_path"], sep="\t", **(data_context.get("tsv_read_kwargs") or {}))
        if data_context.get("df_parquet_path"):
            return pd.read_parquet(data_context["df_parquet_path"]) 
        if data_context.get("df_feather_path"):
            return pd.read_feather(data_context["df_feather_path"]) 
        if data_context.get("df_json_records_path"):
            return pd.read_json(data_context["df_json_records_path"], lines=True)
        if data_context.get("df_json_path"):
            return pd.read_json(data_context["df_json_path"]) 
        if data_context.get("df_pickle_path"):
            return pd.read_pickle(data_context["df_pickle_path"]) 
        if data_context.get("df_path"):
            p = Path(data_context["df_path"])
            ext = p.suffix.lower()
            if ext == ".csv":
                return pd.read_csv(p, **(data_context.get("csv_read_kwargs") or {}))
            if ext == ".tsv":
                return pd.read_csv(p, sep="\t", **(data_context.get("tsv_read_kwargs") or {}))
            if ext in [".xlsx", ".xls"]:
                return pd.read_excel(p, **(data_context.get("excel_read_kwargs") or {}))
            if ext == ".json":
                return pd.read_json(p, **(data_context.get("json_read_kwargs") or {}))
            if ext == ".parquet":
                return pd.read_parquet(p)
            if ext == ".feather":
                return pd.read_feather(p)
        return None

    def _compute_chart_summary(self, df: pd.DataFrame, chart_type: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color")
        agg = (spec.get("aggregation") or "none").lower()

        # Try to parse y like 'sum(col)'
        if isinstance(y, str) and "(" in y and ")" in y:
            func = y.split("(")[0].strip().lower()
            inner = y.split("(", 1)[1].rsplit(")", 1)[0].strip()
            if func in ["sum", "avg", "mean", "count", "min", "max"]:
                agg = "avg" if func == "mean" else func
                y = inner

        def apply_agg(series):
            if agg == "sum":
                return series.sum()
            if agg in ["avg", "mean"]:
                return series.mean()
            if agg == "count":
                return series.count()
            if agg == "min":
                return series.min()
            if agg == "max":
                return series.max()
            return series

        # Build basic groupby summary
        if x and y and x in df.columns and y in df.columns:
            grp = df.groupby(x)[y]
            values = grp.apply(apply_agg)
            values = values.dropna()
            if not values.empty:
                top = values.sort_values(ascending=False).head(5)
                bottom = values.sort_values(ascending=True).head(5)
                summary["x"] = x
                summary["y"] = y
                summary["agg"] = agg
                summary["top"] = [{"label": str(k), "value": float(v)} for k, v in top.items()]
                summary["bottom"] = [{"label": str(k), "value": float(v)} for k, v in bottom.items()]
                summary["max"] = {"label": str(top.index[0]), "value": float(top.iloc[0])}
                summary["min"] = {"label": str(bottom.index[0]), "value": float(bottom.iloc[0])}
                try:
                    total = float(values.sum()) if agg in ["sum", "count"] else float(values.mean())
                    summary["total_or_average"] = total
                except Exception:
                    pass
                if len(top) >= 2 and top.iloc[1] != 0:
                    summary["top_vs_second_ratio"] = float(top.iloc[0] / top.iloc[1])

        # Additional summaries per type can be added here
        summary["chart_type"] = chart_type
        return summary


class ChartExecutorAgent:
    """
    Enhanced ChartExecutorAgent that integrates with the agent system
    """
    
    def __init__(self, multi_mcp):
        self.multi_mcp = multi_mcp
        self.agent_runner = AgentRunner(multi_mcp)
        self.chart_executor = ChartExecutor()
    
    async def execute_charts(self, generation_output: Dict[str, Any], 
                           execution_config: Dict[str, Any] = None,
                           data_context: Dict[str, Any] = None,
                           recommendations: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute generated charts and create actual visualization files
        
        Args:
            generation_output: Output from GenerationAgent
            execution_config: Chart execution configuration
            
        Returns:
            Chart execution results with file paths and status
        """
        try:
            log_step("🎨 Starting ChartExecutorAgent", symbol="🚀")
            
            # Use ChartExecutor to process the generation output
            execution_result = await self.chart_executor.process_generation_output(generation_output, data_context, recommendations)
            
            # Run the ChartExecutorAgent to enhance the results with AI analysis
            input_data = {
                "generation_output": generation_output,
                "execution_result": execution_result,
                "execution_config": execution_config or {},
                # Provide summary of data context without leaking paths unnecessarily
                "data_context_keys": list((data_context or {}).keys()),
                "chart_data_summaries": execution_result.get("chart_data_summaries", []),
                "task": "analyze_and_enhance_chart_execution",
                "objective": "Analyze chart execution results and provide insights and recommendations"
            }
            
            agent_result = await self.agent_runner.run_agent("ChartExecutorAgent", input_data)
            
            if agent_result["success"]:
                # Combine execution results with agent analysis
                enhanced_result = agent_result["output"]
                enhanced_result.update(execution_result)
                return {"success": True, "output": enhanced_result}
            else:
                # Return basic execution results if agent fails
                log_error(f"ChartExecutorAgent analysis failed: {agent_result.get('error')}")
                return {"success": True, "output": execution_result}
            
        except Exception as e:
            log_error(f"ChartExecutorAgent failed: {e}")
            return {"success": False, "error": str(e)}


