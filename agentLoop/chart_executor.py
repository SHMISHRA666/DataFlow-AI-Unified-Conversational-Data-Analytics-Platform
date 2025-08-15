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
    
    def prepare_sample_data(self, data_context: Dict[str, Any]) -> str:
        """Prepare sample data for chart generation"""
        # Create sample data based on context
        sample_data_code = """
import pandas as pd
import numpy as np

# Sample data for demonstration
np.random.seed(42)
sample_data = {
    'region': ['North', 'South', 'East', 'West', 'Central'] * 20,
    'sales': np.random.normal(15000, 5000, 100).astype(int),
    'product': ['Product A', 'Product B', 'Product C', 'Product D'] * 25,
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'customer_segment': ['Enterprise', 'SMB', 'Consumer'] * 33 + ['Enterprise']
}
df = pd.DataFrame(sample_data)
"""
        return sample_data_code
    
    def enhance_code_for_execution(self, original_code: str, chart_id: str, fallback_mode: bool = False) -> str:
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

# Sample data preparation
{self.prepare_sample_data({})}

try:
    # Convert plotly code to matplotlib fallback
    if "px.bar" in '''{original_code}''':
        # Simple bar chart fallback
        plt.figure(figsize=(10, 6))
        plt.bar(df['region'], df['sales'])
        plt.title('Sales Performance by Region')
        plt.xlabel('Region')
        plt.ylabel('Sales Amount ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
    elif "px.line" in '''{original_code}''':
        # Simple line chart fallback
        plt.figure(figsize=(10, 6))
        daily_sales = df.groupby('date')['sales'].sum()
        plt.plot(daily_sales.index, daily_sales.values)
        plt.title('Sales Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
    elif "px.pie" in '''{original_code}''':
        # Simple pie chart fallback
        plt.figure(figsize=(8, 8))
        product_sales = df.groupby('product')['sales'].sum()
        plt.pie(product_sales.values, labels=product_sales.index, autopct='%1.1f%%')
        plt.title('Sales Distribution by Product')
    else:
        # Try to execute original code with matplotlib only
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

# Sample data preparation
{self.prepare_sample_data({})}

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
    
    async def process_generation_output(self, generation_output: Dict[str, Any]) -> Dict[str, Any]:
        """Process GenerationAgent output and create actual charts"""
        
        log_step("🎨 Starting chart execution process", symbol="🚀")
        
        execution_summary = {
            "execution_status": "success",
            "charts_created": [],
            "executed_code": [],
            "data_processing": {
                "data_source": "sample_generated_data",
                "records_processed": 100,
                "data_preparation_steps": ["sample_data_creation", "type_conversion"]
            },
            "performance_metrics": {
                "total_execution_time_seconds": 0.0,
                "charts_per_second": 0.0
            },
            "execution_summary": {
                "total_charts_requested": 0,
                "total_charts_created": 0,
                "success_rate": 0.0
            }
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
                enhanced_code = self.enhance_code_for_execution(python_code, chart_id, fallback_mode)
                
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


class ChartExecutorAgent:
    """
    Enhanced ChartExecutorAgent that integrates with the agent system
    """
    
    def __init__(self, multi_mcp):
        self.multi_mcp = multi_mcp
        self.agent_runner = AgentRunner(multi_mcp)
        self.chart_executor = ChartExecutor()
    
    async def execute_charts(self, generation_output: Dict[str, Any], 
                           execution_config: Dict[str, Any] = None) -> Dict[str, Any]:
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
            execution_result = await self.chart_executor.process_generation_output(generation_output)
            
            # Run the ChartExecutorAgent to enhance the results with AI analysis
            input_data = {
                "generation_output": generation_output,
                "execution_result": execution_result,
                "execution_config": execution_config or {},
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


# Test function
async def test_chart_executor():
    """Test the ChartExecutor with sample generation output"""
    
    # Sample generation output from GenerationAgent
    sample_generation_output = {
        "generated_visualizations": [
            {
                "id": "sales_by_region",
                "title": "Sales Performance by Region",
                "type": "bar",
                "code": {
                    "python": """
import plotly.express as px
fig = px.bar(df, x='region', y='sales', title='Sales Performance by Region')
fig.update_layout(
    xaxis_title='Region',
    yaxis_title='Sales Amount ($)',
    showlegend=False
)
"""
                }
            },
            {
                "id": "sales_trend",
                "title": "Sales Trend Over Time",
                "type": "line",
                "code": {
                    "python": """
import plotly.express as px
fig = px.line(df, x='date', y='sales', title='Sales Trend Over Time')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Sales Amount ($)'
)
"""
                }
            }
        ]
    }
    
    # Test chart execution
    executor = ChartExecutor()
    result = await executor.process_generation_output(sample_generation_output)
    
    print("Chart execution test completed!")
    print(f"Charts created: {len(result['charts_created'])}")
    for chart in result['charts_created']:
        print(f"  - {chart['title']}: {chart['file_path']}")
    
    return result


if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_chart_executor())
