# flow.py – SIMPLIFIED Output Chain System

import networkx as nx
import asyncio
from agentLoop.contextManager import ExecutionContextManager
from agentLoop.agents import AgentRunner
from utils.utils import log_step, log_error
from agentLoop.visualizer import ExecutionVisualizer
from agentLoop.intelligence_flow import IntelligenceWorkflow
from agentLoop.orchestration_flow import OrchestrationWorkflow
from agentLoop.data_processing_flow import DataProcessingWorkflow
from rich.console import Console
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from action.executor import run_user_code
"""Exporters"""
try:
    # Plotly report builder (charts_v5-compatible)
    from export.plotly_v6 import load_any_csv as _plotly_load_csv
    from export.plotly_v6 import load_guide as _plotly_load_guide
    from export.plotly_v6 import build_report as _plotly_build_report
    from export.plotly_v6 import init_chart_studio as _plotly_init_cs
except Exception:
    _plotly_load_csv = None
    _plotly_load_guide = None
    _plotly_build_report = None
    _plotly_init_cs = None

try:
    # Executive HTML report composer
    from export.report_agent import load_report as _ra_load_report
    from export.report_agent import resolve_assets as _ra_resolve_assets
    from export.report_agent import load_asset_map as _ra_load_asset_map
    from export.report_agent import write_outputs as _ra_write_outputs
except Exception:
    _ra_load_report = None
    _ra_resolve_assets = None
    _ra_load_asset_map = None
    _ra_write_outputs = None

class AgentLoop4:
    def __init__(self, multi_mcp, strategy="conservative"):
        self.multi_mcp = multi_mcp
        self.strategy = strategy
        self.agent_runner = AgentRunner(multi_mcp)
        self.intelligence_workflow = IntelligenceWorkflow(multi_mcp)
        self.orchestration_workflow = OrchestrationWorkflow(multi_mcp)
        self.data_processing_workflow = DataProcessingWorkflow(multi_mcp)

    async def run(self, query, file_manifest, uploaded_files):
        # Phase 1: File Profiling (if files exist)
        file_profiles = {}
        if uploaded_files:
            file_list_text = "\n".join([f"- File {i+1}: {Path(f).name} (full path: {f})" 
                                       for i, f in enumerate(uploaded_files)])
            
            grounded_instruction = f"""Profile and summarize each file's structure, columns, content type.

IMPORTANT: Use these EXACT file names in your response:
{file_list_text}

Profile each file separately and return details."""

            file_result = await self.agent_runner.run_agent(
                "DistillerAgent",
                {
                    "task": "profile_files",
                    "files": uploaded_files,
                    "instruction": grounded_instruction,
                    "writes": ["file_profiles"]
                }
            )
            if file_result["success"]:
                file_profiles = file_result["output"]

        # Phase 2: Planning
        plan_result = await self.agent_runner.run_agent(
            "PlannerAgent",
            {
                "original_query": query,
                "planning_strategy": self.strategy,
                "file_manifest": file_manifest,
                "file_profiles": file_profiles
            }
        )

        if not plan_result["success"]:
            raise RuntimeError(f"Planning failed: {plan_result['error']}")

        if 'plan_graph' not in plan_result['output']:
            raise RuntimeError(f"PlannerAgent output missing 'plan_graph' key")
        
        plan_graph = plan_result["output"]["plan_graph"]

        # Phase 3: Simple Output Chain Execution
        context = ExecutionContextManager(
            plan_graph,
            session_id=None,
            original_query=query,
            file_manifest=file_manifest
        )
        
        context.set_multi_mcp(self.multi_mcp)
        # Ensure downstream exporters can discover dataset paths
        try:
            context.plan_graph.graph['file_manifest'] = file_manifest or []
        except Exception:
            pass
        
        # Store initial files in output chain
        if file_profiles:
            context.plan_graph.graph['output_chain']['file_profiles'] = file_profiles

        # Store uploaded files directly
        for file_info in file_manifest:
            context.plan_graph.graph['output_chain'][file_info['name']] = file_info['path']

        # Phase 4: Execute with simple output chaining
        await self._execute_dag(context)
        return context

    async def _execute_dag(self, context):
        """Execute DAG with simple output chaining"""
        visualizer = ExecutionVisualizer(context)
        console = Console()
        
        MAX_CONCURRENT_AGENTS = 4
        max_iterations = 20
        iteration = 0

        while not context.all_done() and iteration < max_iterations:
            iteration += 1
            console.print(visualizer.get_layout())
            
            ready_steps = context.get_ready_steps()
            if not ready_steps:
                if any(context.plan_graph.nodes[n]['status'] == 'failed' 
                       for n in context.plan_graph.nodes):
                    break
                await asyncio.sleep(0.3)
                continue

            # Rate limiting
            batch_size = min(len(ready_steps), MAX_CONCURRENT_AGENTS)
            current_batch = ready_steps[:batch_size]
            
            print(f"🚀 Executing batch: {current_batch}")

            # Mark running
            for step_id in current_batch:
                context.mark_running(step_id)
            
            # Execute batch
            tasks = [self._execute_step(step_id, context) for step_id in current_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results - SIMPLE!
            for step_id, result in zip(current_batch, results):
                if isinstance(result, Exception):
                    context.mark_failed(step_id, str(result))
                elif result["success"]:
                    await context.mark_done(step_id, result["output"])
                else:
                    context.mark_failed(step_id, result["error"])

            if len(ready_steps) > batch_size:
                await asyncio.sleep(5)

    async def _execute_step(self, step_id, context):
        """SIMPLE: Execute step with direct output passing and code execution"""
        step_data = context.get_step_data(step_id)
        agent_type = step_data["agent"]
        
        # SIMPLE: Get raw outputs from previous steps
        inputs = context.get_inputs(step_data.get("reads", []))
        
        # Build agent input
        def build_agent_input(instruction=None, previous_output=None):
            return {
                "step_id": step_id,
                "agent_prompt": instruction or step_data.get("agent_prompt", step_data["description"]),
                "reads": step_data.get("reads", []),
                "writes": step_data.get("writes", []),
                "inputs": inputs,  # Direct output passing!
                "original_query": context.plan_graph.graph['original_query'],
                "session_context": {
                    "session_id": context.plan_graph.graph['session_id'],
                    "file_manifest": context.plan_graph.graph['file_manifest']
                },
                **({"previous_output": previous_output} if previous_output else {})
            }

        # Execute first iteration
        agent_input = build_agent_input()
        result = await self.agent_runner.run_agent(agent_type, agent_input)
        
        # NEW: Handle code execution if agent returned code variants
        if result["success"] and "code" in result["output"]:
            log_step(f"🔧 {step_id}: Agent returned code variants, executing...", symbol="⚙️")
            
            # Prepare executor input
            executor_input = {
                "code_variants": result["output"]["code"],  # CODE_1, CODE_2, etc.
            }
            
            # Execute code variants sequentially until one succeeds
            try:
                execution_result = await run_user_code(
                    executor_input, 
                    self.multi_mcp, 
                    context.plan_graph.graph['session_id'] or "default_session",
                    inputs  # Pass inputs to code execution
                )
                
                # Handle execution results
                if execution_result["status"] == "success":
                    log_step(f"✅ {step_id}: Code execution succeeded", symbol="🎉")
                    
                    # Extract the actual result from code execution
                    code_output = execution_result.get("code_results", {}).get("result", {})
                    
                    # Combine agent output with code execution results
                    combined_output = {
                        **result["output"].get("output", {}),  # Agent's direct output
                        **code_output  # Code execution results
                    }
                    
                    # Update result with combined output
                    result["output"] = combined_output
                    
                elif execution_result["status"] == "partial_failure":
                    log_step(f"⚠️ {step_id}: Code execution partial failure", symbol="⚠️")
                    
                    # Try to extract any successful results
                    code_output = execution_result.get("code_results", {}).get("result", {})
                    if code_output:
                        combined_output = {
                            **result["output"].get("output", {}),
                            **code_output
                        }
                        result["output"] = combined_output
                    else:
                        # Mark as failed
                        result["success"] = False
                        result["error"] = f"Code execution failed: {execution_result.get('error', 'Unknown error')}"
                        
                else:
                    log_step(f"❌ {step_id}: Code execution failed", symbol="🚨")
                    result["success"] = False
                    result["error"] = f"Code execution failed: {execution_result.get('error', 'Unknown error')}"
                    
            except Exception as e:
                log_step(f"💥 {step_id}: Code execution exception: {e}", symbol="❌")
                result["success"] = False
                result["error"] = f"Code execution exception: {str(e)}"
        
        # Handle call_self if needed
        if result["success"] and result["output"].get("call_self"):
            log_step(f"🔄 CALL_SELF triggered for {step_id}", symbol="🔄")
            
            # Second iteration with previous output
            second_input = build_agent_input(
                instruction=result["output"].get("next_instruction", "Continue"),
                previous_output=result["output"]
            )
            
            second_result = await self.agent_runner.run_agent(agent_type, second_input)
            
            # Handle code execution for second iteration too
            if second_result["success"] and "code" in second_result["output"]:
                log_step(f"🔧 {step_id}: Second iteration returned code variants", symbol="⚙️")
                
                executor_input = {
                    "code_variants": second_result["output"]["code"],
                }
                
                try:
                    execution_result = await run_user_code(
                        executor_input,
                        self.multi_mcp,
                        context.plan_graph.graph['session_id'] or "default_session",
                        inputs
                    )
                    
                    if execution_result["status"] == "success":
                        code_output = execution_result.get("code_results", {}).get("result", {})
                        combined_output = {
                            **second_result["output"].get("output", {}),
                            **code_output
                        }
                        second_result["output"] = combined_output
                    else:
                        second_result["success"] = False
                        second_result["error"] = f"Code execution failed: {execution_result.get('error')}"
                        
                except Exception as e:
                    second_result["success"] = False
                    second_result["error"] = f"Code execution exception: {str(e)}"
            
            # Store iteration data
            step_data['iterations'] = [
                {"iteration": 1, "output": result["output"]},
                {"iteration": 2, "output": second_result["output"] if second_result["success"] else None}
            ]
            step_data['call_self_used'] = True
            
            return second_result if second_result["success"] else result
        
        if result["success"] and "clarification_request" in result:
            log_step(f"🤔 {step_id}: Clarification needed", symbol="❓")
            
            # Get user input
            clarification = result["clarification_request"]
            user_response = await self._get_user_input(clarification)
            
            # CREATE the actual node output (ClarificationAgent doesn't do this)
            result["output"] = {
                "user_choice": user_response,
                "clarification_provided": clarification["message"]
            }
            
            # Mark as successful
            result["success"] = True
        
        return result

    async def run_intelligence_layer(self, analysis_data, business_context=None, original_query=None, dag_context=None, data_context=None):
        """
        Run the Intelligence Layer workflow for DataFlow AI
        
        Args:
            analysis_data: Results from data analysis phase
            business_context: Business domain and objectives
            original_query: Original user request
            
        Returns:
            Intelligence layer outputs (recommendations, artifacts, narratives)
        """
        try:
            log_step("🧠 Initiating Intelligence Layer processing", symbol="🚀")
            
            # Use the intelligence workflow
            # Inject numeric session_id into business_context for per-session outputs
            try:
                sid = dag_context.plan_graph.graph.get('session_id') if dag_context else None
                if not business_context:
                    business_context = {}
                if sid:
                    # enforce numeric only
                    digits_only = ''.join(ch for ch in str(sid) if ch.isdigit())
                    business_context['session_id'] = digits_only or str(int(__import__('time').time()))[-8:]
                else:
                    # ensure we always have a numeric session id
                    business_context.setdefault('session_id', str(int(__import__('time').time()))[-8:])
            except Exception:
                business_context.setdefault('session_id', str(int(__import__('time').time()))[-8:])

            result = await self.intelligence_workflow.process_dataflow_request(
                original_query or "Data analysis and visualization request",
                analysis_data,
                business_context,
                data_context
            )
            
            # After intelligence outputs are ready, attempt exports here (flow-level integration)
            try:
                # Derive a per-session output directory
                session_id = business_context.get('session_id')
                log_step(f"🚀 Calling Export Agent with session_id: {session_id}", symbol="📤")
                exports = self._run_exports_if_available(result, context=dag_context, session_id=session_id)
                if exports:
                    result.setdefault("exports", {}).update(exports)
                    log_step(f"✅ Export Agent completed successfully with {len(exports)} exports", symbol="🎉")
                else:
                    log_step("⚠️ Export Agent: No exports generated", symbol="⚠️")
            except Exception as e:
                log_error(f"Export step skipped due to error: {e}")

            log_step("✅ Intelligence Layer processing completed", symbol="🎉")
            return {"success": True, "output": result}
            
        except Exception as e:
            log_error(f"Intelligence Layer failed: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_intelligence_step(self, step_id, context):
        """
        Handle intelligence layer steps in the DAG execution
        """
        step_data = context.get_step_data(step_id)
        
        # Get inputs from previous steps
        inputs = context.get_inputs(step_data.get("reads", []))
        
        # Extract analysis data from inputs
        analysis_data = {}
        business_context = {}
        
        # Look for analysis results in inputs - prioritize data processing layer output
        data_context = {}
        for key, value in inputs.items():
            if 'data_processing' in key.lower() or 'intelligence_layer_ready_data' in key.lower():
                # Data from Data Processing Layer - extract ready data
                if isinstance(value, dict) and 'intelligence_layer_ready_data' in value:
                    analysis_data.update(value['intelligence_layer_ready_data'])
                    # Extract data_context for chart execution
                    if 'data_context' in value['intelligence_layer_ready_data']:
                        data_context.update(value['intelligence_layer_ready_data']['data_context'])
                    # ✅ Extract session_id from data processing layer
                    if 'session_id' in value:
                        business_context['session_id'] = value['session_id']
                elif isinstance(value, dict):
                    analysis_data.update(value)
                    # Extract data_context if present
                    if 'data_context' in value:
                        data_context.update(value['data_context'])
                    # ✅ Extract session_id from data processing layer
                    if 'session_id' in value:
                        business_context['session_id'] = value['session_id']
            elif 'analysis' in key.lower() or 'data' in key.lower():
                analysis_data.update(value if isinstance(value, dict) else {key: value})
            elif 'context' in key.lower() or 'business' in key.lower():
                business_context.update(value if isinstance(value, dict) else {key: value})
        
        # If no specific analysis data found, use all inputs as analysis data
        if not analysis_data:
            analysis_data = inputs
        
        # Get original query from context
        original_query = context.plan_graph.graph.get('original_query', 'Data analysis request')
        
        # Run intelligence layer
        # Use data_context from Data Processing Layer if available, otherwise infer from manifest
        if not data_context:
            data_context = self._build_data_context_from_manifest(context)

        result = await self.run_intelligence_layer(
            analysis_data, 
            business_context, 
            original_query,
            dag_context=context,
            data_context=data_context
        )
        
        return result

    async def run_data_processing_layer(self, file_paths, processing_context=None, original_query=None, dag_context=None):
        """
        Run the Data Processing Layer workflow for DataFlow AI
        
        Args:
            file_paths: List of file paths to process (CSV, JSON, Excel)
            processing_context: Processing preferences and configuration
            original_query: Original user request
            dag_context: DAG execution context for integration
            
        Returns:
            Data processing outputs ready for Intelligence Layer consumption
        """
        try:
            log_step("🔧 Initiating Data Processing Layer", symbol="🚀")
            
            # Prepare processing context
            if not processing_context:
                processing_context = {}
            
            # Inject session_id for per-session outputs
            try:
                sid = dag_context.plan_graph.graph.get('session_id') if dag_context else None
                if sid:
                    # enforce numeric only
                    digits_only = ''.join(ch for ch in str(sid) if ch.isdigit())
                    processing_context['session_id'] = digits_only or str(int(__import__('time').time()))[-8:]
                else:
                    processing_context.setdefault('session_id', str(int(__import__('time').time()))[-8:])
            except Exception:
                processing_context.setdefault('session_id', str(int(__import__('time').time()))[-8:])
            
            # Use the data processing workflow
            result = await self.data_processing_workflow.process_files_request(
                original_query or "Process and analyze uploaded data files",
                file_paths,
                processing_context
            )
            
            log_step("✅ Data Processing Layer completed", symbol="🎉")
            return {"success": True, "output": result}
            
        except Exception as e:
            log_error(f"Data Processing Layer failed: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_data_processing_step(self, step_id, context):
        """
        Handle data processing layer steps in the DAG execution
        """
        step_data = context.get_step_data(step_id)
        
        # Get inputs from previous steps
        inputs = context.get_inputs(step_data.get("reads", []))
        
        # Extract file paths from inputs or context
        file_paths = []
        processing_context = {}
        
        # Look for files in various sources
        if 'file_paths' in inputs:
            file_paths = inputs['file_paths']
        elif 'files' in inputs:
            file_paths = inputs['files']
        else:
            # Extract from file manifest
            file_manifest = context.plan_graph.graph.get('file_manifest', [])
            file_paths = [item.get('path') for item in file_manifest if item.get('path')]
        
        # Extract processing preferences from inputs
        for key, value in inputs.items():
            if 'processing' in key.lower() or 'config' in key.lower():
                if isinstance(value, dict):
                    processing_context.update(value)
                else:
                    processing_context[key] = value
        
        # Get original query from context
        original_query = context.plan_graph.graph.get('original_query', 'Data processing request')
        
        # Run data processing layer
        result = await self.run_data_processing_layer(
            file_paths, 
            processing_context, 
            original_query,
            dag_context=context
        )
        
        return result

    def _filter_failed_charts_from_guide(self, guide: dict, intelligence_output: dict) -> dict:
        """Filter out charts that failed to execute from the guide configuration"""
        if not intelligence_output or "chart_execution" not in intelligence_output:
            return guide
        
        chart_execution = intelligence_output["chart_execution"]
        if "charts_created" not in chart_execution:
            return guide
        
        # Get list of successfully created charts
        successful_charts = []
        for chart_info in chart_execution["charts_created"]:
            if chart_info.get("files_created"):
                successful_charts.append(chart_info["chart_id"])
        
        # Filter the guide charts to only include successful ones
        if "charts" in guide:
            original_charts = guide["charts"]
            filtered_charts = []
            
            for chart in original_charts:
                # Try to match chart by name or title
                chart_name = chart.get("name", "")
                chart_title = chart.get("title", "")
                
                # Check if this chart was successfully created
                is_successful = False
                for successful_id in successful_charts:
                    if (successful_id in chart_name.lower().replace(" ", "_") or 
                        successful_id in chart_title.lower().replace(" ", "_")):
                        is_successful = True
                        break
                
                if is_successful:
                    filtered_charts.append(chart)
                else:
                    log_step(f"🔍 Export Layer: Filtering out failed chart: {chart_name or chart_title}", symbol="⚠️")
            
            guide["charts"] = filtered_charts
            log_step(f"🔍 Export Layer: Filtered guide from {len(original_charts)} to {len(filtered_charts)} charts", symbol="📊")
        
        return guide

    def _modify_guide_for_raw_data(self, guide: dict, intelligence_output: dict) -> dict:
        """Convert aggregation format to be compatible with export layer"""
        if not guide or "charts" not in guide:
            return guide
        
        updated_charts = []
        for chart in guide["charts"]:
            # Create a copy of the chart
            updated_chart = chart.copy()
            
            # Handle aggregation conversion for export layer compatibility
            if "agg" in updated_chart:
                agg_type = updated_chart["agg"]
                chart_type = updated_chart.get("type")
                
                # For count aggregations, convert to proper format
                if agg_type == "count" and chart_type == "bar":
                    x_field = updated_chart.get("x", "").replace("$", "")
                    y_field = updated_chart.get("y", "").replace("$", "")
                    
                    # Set up proper aggregation structure
                    updated_chart["aggregate"] = {
                        "groupby": [x_field],
                        "measures": [y_field],  # Use the actual y field for counting (note: plural)
                        "how": "count"
                    }
                    # Remove the simple agg field
                    del updated_chart["agg"]
                    # Keep y as the field to be counted
                    updated_chart["y"] = f"${y_field}"
                    
                elif agg_type == "count" and chart_type == "histogram":
                    # For histograms, remove agg as they don't need aggregation
                    del updated_chart["agg"]
                    
                else:
                    # For other aggregation types, remove the simple agg
                    del updated_chart["agg"]
            
            # Remove any existing aggregate field to avoid conflicts
            if "aggregate" in updated_chart and "agg" not in chart:
                # Keep existing aggregate structure
                pass
            
            updated_charts.append(updated_chart)
            log_step(f"🔍 Export Layer: Processed chart '{chart.get('title', 'Unknown')}' for export compatibility", symbol="📊")
        
        guide["charts"] = updated_charts
        log_step(f"🔍 Export Layer: Processed {len(updated_charts)} charts for export compatibility", symbol="📊")
        
        return guide

    def _select_dataset_path_for_export(self, context) -> Path | None:
        """Best-effort dataset path selection for Plotly export.

        Preference order (no hardcoded fallbacks):
        - First CSV in graph-level file_manifest
        - First Excel in graph-level file_manifest
        """
        try:
            manifest = context.plan_graph.graph.get('file_manifest') or []
            # Try CSV
            for item in manifest:
                p = Path(item.get('path', ''))
                if p.suffix.lower() == '.csv' and p.exists():
                    return p
            # Try Excel
            for item in manifest:
                p = Path(item.get('path', ''))
                if p.suffix.lower() in ('.xlsx', '.xls') and p.exists():
                    return p
        except Exception:
            pass
        return None

    def _run_exports_if_available(self, intelligence_output: dict, context=None, session_id: str | None = None) -> dict:
        """Run exporters using artifacts persisted by the Intelligence Layer.

        - Plotly gallery: uses generated_charts/charts.yaml + a dataset CSV
        - Executive report: uses generated_charts/narrative_insights.json + generated assets
        Returns a dict of export paths/urls that can be attached to outputs.
        """
        exports: dict = {}

        # Per-session directory to avoid overwrites
        out_dir = Path('generated_charts') / (str(session_id) if session_id else '')
        charts_yaml = out_dir / 'charts.yaml'
        insights_json = out_dir / 'narrative_insights.json'

        # 1) Plotly gallery export (optional; requires dataset + charts.yaml)
        try:
            # Ensure .env variables are loaded (PLOTLY_USERNAME, PLOTLY_API_KEY)
            try:
                load_dotenv()
            except Exception:
                pass
            
            # Debug: Log export layer status
            log_step(f"🔍 Export Layer Debug - Session ID: {session_id}")
            log_step(f"🔍 Export Layer Debug - Output Dir: {out_dir}")
            log_step(f"🔍 Export Layer Debug - Charts YAML exists: {charts_yaml.exists()}")
            log_step(f"🔍 Export Layer Debug - Insights JSON exists: {insights_json.exists()}")
            
            if _plotly_build_report and charts_yaml.exists():
                dataset_path: Path | None = None
                # If a context is available (DAG step), try to infer from file_manifest
                if context is not None:
                    dataset_path = self._select_dataset_path_for_export(context)
                # Final fallback: None -> skip Plotly export
                if dataset_path and dataset_path.exists():
                    # Load DataFrame from CSV or Excel
                    df = None
                    if dataset_path.suffix.lower() == '.csv':
                        df = _plotly_load_csv(str(dataset_path)) if _plotly_load_csv else None
                    elif dataset_path.suffix.lower() in ('.xlsx', '.xls'):
                        try:
                            df = pd.read_excel(str(dataset_path))
                        except Exception as e:
                            log_error(f"Failed to read Excel dataset for Plotly export: {e}")
                    guide = _plotly_load_guide(str(charts_yaml)) if _plotly_load_guide else None
                    if df is not None and guide is not None:
                        # Ensure a unique Chart Studio filename per session/run to avoid overwriting
                        try:
                            from datetime import datetime
                            safe_title = str(guide.get('report_title') or 'plotly-report').replace(' ', '-').lower()
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            unique_name = f"{safe_title}-{session_id}-{ts}" if session_id else f"{safe_title}-{ts}"
                            guide['filename'] = unique_name
                        except Exception:
                            # Best-effort; if this fails, plotly_v6 will still derive a name
                            pass
                        # Filter out failed charts from the guide
                        guide = self._filter_failed_charts_from_guide(guide, intelligence_output)
                        # Modify guide to match chart executor's raw data approach
                        guide = self._modify_guide_for_raw_data(guide, intelligence_output)
                        log_step(f"🔍 Export Layer: Data loaded ({len(df)} rows), guide loaded", symbol="✅")
                        out_html = str(out_dir / 'plotly_index.html')
                        # Initialize Chart Studio publishing if credentials are available
                        publish_py = None
                        if _plotly_init_cs is not None:
                            try:
                                # Prefer env vars PLOTLY_USERNAME/PLOTLY_API_KEY; passing None uses env in init_chart_studio
                                publish_py = _plotly_init_cs(None, None, None)
                            except Exception as e:
                                # Credentials missing or package not installed; proceed without publishing
                                log_error(f"Chart Studio init skipped: {e}")
                        urls = _plotly_build_report(
                            df,
                            guide,
                            out_html,
                            publish_py=publish_py,
                            publish_sharing='public',
                            no_html=False,
                            publish_combined=True,
                            publish_cols=1,
                            publish_row_height=520,
                            publish_vspace=0.20,
                            publish_auto_layout=True,
                            publish_row_height_weights=None,
                            publish_hspace=0.08,
                            publish_width=None,
                            publish_col_width_weights=None,
                        )
                        exports['plotly_html_path'] = out_html
                        
                        # If no URLs were generated (Chart Studio not available), skip export
                        if not urls:
                            log_step("⚠️ Chart Studio not available, skipping plotly export", symbol="🔄")
                        
                        if urls:
                            exports['chart_studio_urls'] = urls
                            # ✅ Display complete dashboard plotly link in console
                            log_step(f"🎉 Export Agent created complete dashboard with {len(urls)} plotly links", symbol="📊")
                            for name, url in urls:
                                log_step(f"🌐 Complete Dashboard Plotly Link: {url}", symbol="🔗")
                            # Mirror the combined URL into the intelligence output so callers always have it
                            try:
                                if isinstance(intelligence_output, dict):
                                    intelligence_output.setdefault('exports', {})['chart_studio_urls'] = urls
                            except Exception:
                                pass
                        else:
                            log_step("⚠️ Export Agent: No plotly URLs generated", symbol="⚠️")
        except Exception as e:
            log_error(f"Plotly export failed: {e}")

        # 2) Executive report export via report_agent (insights + assets)
        try:
            if _ra_load_report and insights_json.exists():
                report = _ra_load_report(insights_json, verbose=False)
                asset_dirs = [out_dir / 'html', out_dir / 'png', out_dir / 'svg']
                explicit_map = _ra_load_asset_map(None) if _ra_load_asset_map else {}
                report.insights, _ = _ra_resolve_assets(report.insights, asset_dirs, explicit_map, verbose=False)  # type: ignore[arg-type]
                html_path, resolved_path = _ra_write_outputs(
                    report,
                    out_dir,
                    verbose=False,
                    chart_studio_urls=exports.get('chart_studio_urls')
                )
                exports['report_html_path'] = str(html_path)
                exports['resolved_insights_path'] = str(resolved_path)
        except Exception as e:
            log_error(f"Executive report export failed: {e}")

        return exports

    def _build_data_context_from_manifest(self, context) -> dict | None:
        """Build a data_context dict from user-provided file_manifest in the DAG context.

        Returns:
            dict with one of {df_csv_path|df_excel_path} when available; otherwise None.
        """
        try:
            manifest = (context.plan_graph.graph or {}).get('file_manifest') or []
            for item in manifest:
                p = Path(item.get('path', ''))
                if p.suffix.lower() == '.csv' and p.exists():
                    return {"df_csv_path": str(p)}
            for item in manifest:
                p = Path(item.get('path', ''))
                if p.suffix.lower() in ('.xlsx', '.xls') and p.exists():
                    return {"df_excel_path": str(p)}
        except Exception:
            return None
        return None

    async def run_discovery_orchestration(self, organization_context, discovery_constraints=None, user_request=None):
        """
        Run the Discovery Agent for data source discovery and cataloging
        
        Args:
            organization_context: Organization infrastructure and requirements
            discovery_constraints: Optional scope and constraints for discovery
            user_request: Original user request for discovery
            
        Returns:
            Discovery orchestration outputs (sources, catalog, recommendations)
        """
        try:
            log_step("🔍 Initiating Discovery orchestration", symbol="🚀")
            
            # Use the orchestration workflow
            result = await self.orchestration_workflow.process_discovery_request(
                user_request or "Discover and catalog available data sources",
                organization_context,
                discovery_constraints
            )
            
            log_step("✅ Discovery orchestration completed", symbol="🎉")
            return {"success": True, "output": result}
            
        except Exception as e:
            log_error(f"Discovery orchestration failed: {e}")
            return {"success": False, "error": str(e)}

    async def run_monitoring_orchestration(self, system_context, user_request=None):
        """
        Run the Monitoring Agent for system health and performance monitoring
        
        Args:
            system_context: Current system state, metrics, and monitoring data
            user_request: Original user request for monitoring
            
        Returns:
            Monitoring orchestration outputs (health status, alerts, recommendations)
        """
        try:
            log_step("📊 Initiating Monitoring orchestration", symbol="🚀")
            
            # Use the orchestration workflow
            result = await self.orchestration_workflow.process_monitoring_request(
                user_request or "Monitor system health and performance",
                system_context
            )
            
            log_step("✅ Monitoring orchestration completed", symbol="🎉")
            return {"success": True, "output": result}
            
        except Exception as e:
            log_error(f"Monitoring orchestration failed: {e}")
            return {"success": False, "error": str(e)}

    async def run_full_orchestration(self, organization_context, system_context, 
                                   discovery_constraints=None, user_request=None):
        """
        Run both Discovery and Monitoring agents in a coordinated workflow
        
        Args:
            organization_context: Organization info for discovery
            system_context: System state for monitoring
            discovery_constraints: Optional discovery limitations
            user_request: Original user request
            
        Returns:
            Complete orchestration outputs (discovery + monitoring)
        """
        try:
            log_step("🎯 Initiating full orchestration", symbol="🚀")
            
            # Use the orchestration workflow
            result = await self.orchestration_workflow.process_full_orchestration(
                user_request or "Perform comprehensive discovery and monitoring",
                organization_context,
                system_context,
                discovery_constraints
            )
            
            log_step("✅ Full orchestration completed", symbol="🎉")
            return {"success": True, "output": result}
            
        except Exception as e:
            log_error(f"Full orchestration failed: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_orchestration_step(self, step_id, context):
        """
        Handle orchestration layer steps in the DAG execution
        """
        step_data = context.get_step_data(step_id)
        
        # Get inputs from previous steps
        inputs = context.get_inputs(step_data.get("reads", []))
        
        # Determine orchestration type based on step configuration
        orchestration_type = step_data.get("orchestration_type", "discovery")
        
        # Extract relevant data from inputs
        organization_context = {}
        system_context = {}
        discovery_constraints = {}
        
        # Look for context data in inputs
        for key, value in inputs.items():
            if 'organization' in key.lower() or 'infra' in key.lower():
                organization_context.update(value if isinstance(value, dict) else {key: value})
            elif 'system' in key.lower() or 'monitoring' in key.lower() or 'metrics' in key.lower():
                system_context.update(value if isinstance(value, dict) else {key: value})
            elif 'discovery' in key.lower() or 'constraints' in key.lower():
                discovery_constraints.update(value if isinstance(value, dict) else {key: value})
        
        # If no specific context found, distribute inputs appropriately
        if not organization_context and not system_context:
            # Split inputs based on orchestration type
            if orchestration_type == "discovery":
                organization_context = inputs
            elif orchestration_type == "monitoring":
                system_context = inputs
            else:  # full orchestration
                organization_context = inputs
                system_context = inputs
        
        # Get original query from context
        original_query = context.plan_graph.graph.get('original_query', 'Orchestration request')
        
        # Run appropriate orchestration
        if orchestration_type == "discovery":
            result = await self.run_discovery_orchestration(
                organization_context, 
                discovery_constraints or None,
                original_query
            )
        elif orchestration_type == "monitoring":
            result = await self.run_monitoring_orchestration(
                system_context,
                original_query
            )
        elif orchestration_type == "full":
            result = await self.run_full_orchestration(
                organization_context,
                system_context,
                discovery_constraints or None,
                original_query
            )
        else:
            # Default to discovery
            result = await self.run_discovery_orchestration(
                organization_context,
                discovery_constraints or None,
                original_query
            )
        
        return result

