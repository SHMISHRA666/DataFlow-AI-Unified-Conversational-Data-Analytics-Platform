# intelligence_flow.py - DataFlow AI Intelligence Layer Workflow

import asyncio
from agentLoop.agents import AgentRunner
from agentLoop.chart_executor import ChartExecutorAgent
from utils.utils import log_step, log_error
from typing import Dict, Any, List
from pathlib import Path
import json
try:
    import yaml  # PyYAML is already used elsewhere in the project
except Exception:
    yaml = None


class IntelligenceLayer:
    """
    Intelligence Layer for DataFlow AI that orchestrates the three core agents:
    - RecommendationAgent: Analyzes data and provides KPI/visualization recommendations
    - GenerationAgent: Generates code, dashboards, and BI configurations
    - NarrativeAgent: Creates human-readable reports and narratives
    """
    
    def __init__(self, multi_mcp):
        self.multi_mcp = multi_mcp
        self.agent_runner = AgentRunner(multi_mcp)
        self.chart_executor_agent = ChartExecutorAgent(multi_mcp)
    
    async def process_data_analysis(self, analysis_data: Dict[str, Any], 
                                  business_context: Dict[str, Any] = None,
                                  data_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete Intelligence Layer processing pipeline.
        
        Args:
            analysis_data: Results from data analysis (schemas, stats, patterns)
            business_context: Business objectives, domain, audience info
            
        Returns:
            Complete intelligence output with recommendations, generated artifacts, and narratives
        """
        try:
            log_step("🧠 Starting Intelligence Layer processing", symbol="🚀")
            # Ensure a stable numeric session_id in business_context
            if business_context is None:
                business_context = {}
            sid = business_context.get("session_id")
            if not sid:
                # derive 8-digit time suffix
                business_context["session_id"] = str(int(__import__('time').time()))[-8:]
            
            # Phase 1: Generate Recommendations
            log_step("📊 Phase 1: Generating recommendations", symbol="1️⃣")
            recommendations = await self._generate_recommendations(analysis_data, business_context)
            
            # Phase 2: Generate Artifacts
            log_step("🔧 Phase 2: Generating dashboards and code", symbol="2️⃣")
            generated_artifacts = await self._generate_artifacts(recommendations, analysis_data, business_context, data_context)
            
            # Phase 3: Create Narratives
            log_step("📝 Phase 3: Creating narratives and reports", symbol="3️⃣")
            narratives = await self._create_narratives(recommendations, generated_artifacts, analysis_data, business_context)
            
            # Combine all outputs
            intelligence_output = {
                "recommendations": recommendations,
                "generated_artifacts": generated_artifacts,
                "narratives": narratives,
                "processing_summary": {
                    "status": "completed",
                    "phases_completed": ["recommendations", "generation", "narratives"],
                    "intelligence_layer_version": "1.0"
                }
            }
            
            # Persist outputs (charts.yaml, narrative insights, full results)
            try:
                self._persist_outputs(
                    recommendations,
                    generated_artifacts,
                    narratives,
                    business_context or {},
                )
            except Exception as persist_err:
                log_error(f"Failed to persist output artifacts: {persist_err}")

            log_step("✅ Intelligence Layer processing completed", symbol="🎉")
            return intelligence_output
            
        except Exception as e:
            log_error(f"Intelligence Layer processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "processing_summary": {
                    "status": "failed",
                    "error": str(e)
                }
            }
    
    async def _generate_recommendations(self, analysis_data: Dict[str, Any], 
                                      business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Phase 1: Generate KPI and visualization recommendations"""
        
        input_data = {
            "user_query": (business_context or {}).get("original_query", ""),
            "analysis_results": analysis_data,
            "business_context": business_context or {},
            "task": "analyze_data_and_recommend",
            "objective": "Provide intelligent recommendations for KPIs, visualizations, and insights"
        }
        
        result = await self.agent_runner.run_agent("RecommendationAgent", input_data)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(f"RecommendationAgent failed: {result.get('error', 'Unknown error')}")
    
    async def _generate_artifacts(self, recommendations: Dict[str, Any], 
                                analysis_data: Dict[str, Any],
                                business_context: Dict[str, Any] = None,
                                data_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Phase 2: Generate dashboards, code, and BI configurations with chart execution"""
        
        input_data = {
            "recommendations": recommendations,
            "analysis_results": analysis_data,
            "business_context": business_context or {},
            "task": "generate_data_artifacts",
            "objective": "Create production-ready visualizations, dashboards, and BI integrations"
        }
        
        # Step 1: Generate code and configurations
        log_step("🔧 Generating visualization code and configurations", symbol="1️⃣")
        generation_result = await self.agent_runner.run_agent("GenerationAgent", input_data)
        
        if not generation_result["success"]:
            raise Exception(f"GenerationAgent failed: {generation_result.get('error', 'Unknown error')}")
        
        generation_output = generation_result["output"]
        # Ensure YAML artifacts from GenerationAgent are preserved
        files_section = generation_output.get("files", {}) if isinstance(generation_output, dict) else {}
        charts_yaml = files_section.get("charts.yaml")
        
        # Step 2: Execute generated charts and create actual visualization files
        log_step("🎨 Executing chart code and creating visualization files", symbol="2️⃣")
        
        # Check if there are visualizations to execute
        has_visualizations = (
            "generated_visualizations" in generation_output and 
            len(generation_output["generated_visualizations"]) > 0
        )
        
        if has_visualizations:
            # Determine per-session output directory for chart files
            session_out_dir = self._safe_get_output_dir((business_context or {}).get("session_id") or (business_context or {}).get("original_query"))
            session_out_dir.mkdir(parents=True, exist_ok=True)
            chart_execution_result = await self.chart_executor_agent.execute_charts(
                generation_output,
                execution_config={
                    "output_formats": ["png", "svg", "html"],
                    "quality": "high",
                    "interactive": True
                },
                data_context=data_context or {},
                recommendations=recommendations,
                output_directory=str(session_out_dir)
            )
            
            if chart_execution_result["success"]:
                # Combine generation output with chart execution results
                enhanced_output = generation_output.copy()
                enhanced_output["chart_execution"] = chart_execution_result["output"]
                # Re-attach charts.yaml if present
                if charts_yaml is not None:
                    enhanced_output.setdefault("files", {})["charts.yaml"] = charts_yaml
                
                # Update visualization entries with file paths
                executed_charts = chart_execution_result["output"].get("charts_created", [])
                for viz in enhanced_output.get("generated_visualizations", []):
                    viz_id = viz.get("id", "")
                    # Find matching executed chart
                    for chart in executed_charts:
                        if chart.get("chart_id") == viz_id:
                            viz["executed_files"] = {
                                "png_path": next((c["file_path"] for c in executed_charts 
                                                if c["chart_id"] == viz_id and c["file_format"] == "png"), None),
                                "svg_path": next((c["file_path"] for c in executed_charts 
                                                if c["chart_id"] == viz_id and c["file_format"] == "svg"), None),
                                "html_path": next((c["file_path"] for c in executed_charts 
                                                 if c["chart_id"] == viz_id and c["file_format"] == "html"), None),
                                "file_sizes": {fmt: c["file_size_bytes"] for c in executed_charts 
                                             if c["chart_id"] == viz_id for fmt in [c["file_format"]]}
                            }
                            break
                
                log_step(f"✅ Generated and executed {len(executed_charts)} charts successfully", symbol="🎉")
                return enhanced_output
            else:
                log_error(f"Chart execution failed: {chart_execution_result.get('error')}")
                # Return generation output without chart execution
                generation_output["chart_execution_error"] = chart_execution_result.get("error")
                # Keep YAML even on failure
                if charts_yaml is not None:
                    generation_output.setdefault("files", {})["charts.yaml"] = charts_yaml
                return generation_output
        else:
            log_step("ℹ️ No visualizations to execute, returning generation output only", symbol="ℹ️")
            return generation_output
    
    async def _create_narratives(self, recommendations: Dict[str, Any],
                               generated_artifacts: Dict[str, Any],
                               analysis_data: Dict[str, Any],
                               business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Phase 3: Create human-readable narratives and reports"""
        
        input_data = {
            "user_query": (business_context or {}).get("original_query", ""),
            "recommendations": recommendations,
            "generated_artifacts": generated_artifacts,
            "analysis_results": analysis_data,
            "business_context": business_context or {},
            # Provide expected chart ids and file references to enforce per-chart insights
            "expected_chart_ids": [viz.get("id") for viz in generated_artifacts.get("generated_visualizations", [])],
            "expected_chart_files": {
                viz.get("id"): viz.get("executed_files", {})
                for viz in generated_artifacts.get("generated_visualizations", [])
            },
            "task": "create_data_narratives",
            "objective": "Transform insights into compelling, actionable narratives for stakeholders"
        }
        
        result = await self.agent_runner.run_agent("NarrativeAgent", input_data)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(f"NarrativeAgent failed: {result.get('error', 'Unknown error')}")

    def _safe_get_output_dir(self, session_id: str | None = None) -> Path:
        """Return the output directory used across artifacts.

        If a session_id is available, nest outputs under generated_charts/<session_id>/
        to avoid overwrites across runs.
        """
        base = Path("generated_charts")
        if session_id:
            # Enforce numeric-only folder name to keep it clean and sortable
            digits_only = ''.join(ch for ch in str(session_id) if ch.isdigit())
            if digits_only:
                return base / digits_only
            # fallback to 8-digit time suffix
            return base / str(int(__import__('time').time()))[-8:]
        return base

    def _persist_outputs(self, recommendations: Dict[str, Any],
                        generated_artifacts: Dict[str, Any],
                        narratives: Dict[str, Any],
                        business_context: Dict[str, Any]) -> None:
        """Persist charts.yaml, narrative insights JSON, and full results JSON.

        - charts.yaml: contains details for all generated charts (with file paths)
        - narrative_insights.json: raw narrative output
        - results_intelligence_layer.json: combined top-level results for downstream use
        """
        session_id = business_context.get("session_id") or business_context.get("original_query") or "session"
        output_dir = self._safe_get_output_dir(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Write charts.yaml (EXACT schema: report_title, columns, charts)
        try:
            if yaml is not None:
                charts_yaml_obj = self._build_charts_yaml(recommendations, generated_artifacts, business_context)
                charts_yaml_path = output_dir / "charts.yaml"
                with charts_yaml_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(charts_yaml_obj, f, sort_keys=False, allow_unicode=True)
            else:
                log_error("PyYAML not available; skipping charts.yaml write")
        except Exception as e:
            log_error(f"Error writing charts.yaml: {e}")

        # 2) Write narrative_insights.json
        try:
            # Normalize executed file paths to be relative subfolders so exporters do not duplicate files at root
            safe_artifacts = dict(generated_artifacts or {})
            for viz in safe_artifacts.get("generated_visualizations", []) or []:
                files = viz.get("executed_files") or {}
                for key in ["html_path", "png_path", "svg_path"]:
                    p = files.get(key)
                    if p:
                        name = Path(p).name
                        if key == "html_path":
                            files[key] = str(Path("html") / name)
                        elif key == "png_path":
                            files[key] = str(Path("png") / name)
                        elif key == "svg_path":
                            files[key] = str(Path("svg") / name)
                viz["executed_files"] = files

            narrative_path = output_dir / "narrative_insights.json"
            with narrative_path.open("w", encoding="utf-8") as f:
                json.dump(narratives or {}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_error(f"Error writing narrative_insights.json: {e}")

        # 3) Write results_intelligence_layer.json (combined subset)
        try:
            results_path = output_dir / "results_intelligence_layer.json"
            combined = {
                "recommendations": recommendations,
                "generated_artifacts": generated_artifacts,
                "narratives": narratives,
                "metadata": {
                    "report_title": business_context.get("report_title") or business_context.get("original_query") or "DataFlow AI Intelligence Report",
                    "session_id": output_dir.name
                }
            }
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_error(f"Error writing results_intelligence_layer.json: {e}")

    def _build_charts_yaml(self, recommendations: Dict[str, Any],
                           generated_artifacts: Dict[str, Any],
                           business_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a YAML object resembling the provided example, filled with available details.

        Includes chart specs from recommendations and file paths from chart execution results.
        """
        report_title = business_context.get("report_title") or business_context.get("original_query") or "DataFlow AI Report"

        # Map recommended viz by id for spec
        rec_viz_list = (recommendations or {}).get("recommended_visualizations", [])
        rec_by_id = {viz.get("id"): viz for viz in rec_viz_list if viz.get("id")}

        # Aggregate executed file paths by chart_id
        charts_created = ((generated_artifacts or {}).get("chart_execution", {}) or {}).get("charts_created", [])
        files_by_id: Dict[str, Dict[str, str]] = {}
        for c in charts_created:
            cid = c.get("chart_id")
            if not cid:
                continue
            files_by_id.setdefault(cid, {})[c.get("file_format", "")] = c.get("file_path")

        def _parse_y_and_agg(y_val: Any, agg_val: Any) -> (str, str):
            """Normalize y/agg when y is like 'count(col)' or similar."""
            y_field = None
            agg_norm = (agg_val or "").lower() if isinstance(agg_val, str) else ""
            if isinstance(y_val, str) and "(" in y_val and ")" in y_val:
                fn = y_val.split("(", 1)[0].strip().lower()
                inner = y_val.split("(", 1)[1].rsplit(")", 1)[0].strip()
                y_field = inner
                if fn in ["count", "sum", "avg", "mean", "min", "max"]:
                    agg_norm = "avg" if fn == "mean" else fn
            else:
                y_field = y_val
            return y_field, agg_norm

        # Use generated_visualizations for titles/types/ids
        gen_list = (generated_artifacts or {}).get("generated_visualizations", [])
        charts_yaml_list: List[Dict[str, Any]] = []
        for viz in gen_list:
            cid = viz.get("id")
            title = viz.get("title", cid)
            vtype = viz.get("type", "chart")
            rec = rec_by_id.get(cid, {})
            spec = rec.get("spec", {}) if isinstance(rec, dict) else {}

            chart_entry: Dict[str, Any] = {
                "name": title,
                "type": vtype,
                "title": title
            }

            # Encodings/aggregations with normalization (e.g., y='count(col)')
            x_field = spec.get("x")
            y_field, agg_norm = _parse_y_and_agg(spec.get("y"), spec.get("aggregation"))
            color_field = spec.get("color")

            if x_field:
                chart_entry["x"] = f"${x_field}"
            # For bar/line/scatter include y/agg/color; for pie/histogram keep it minimal
            if vtype not in ["pie", "histogram"]:
                if y_field:
                    chart_entry["y"] = f"${y_field}"
                # Avoid color=x duplication
                if color_field and color_field != x_field:
                    chart_entry["color"] = f"${color_field}"
                if agg_norm:
                    chart_entry["agg"] = agg_norm
            else:
                # pie: use names via color or x; histogram: use x only
                if vtype == "pie":
                    names_field = color_field or x_field
                    if names_field:
                        chart_entry["color"] = f"${names_field}"
                # do not set agg/y/value for pie/hist – builder computes counts automatically

            # Attach executed file paths
            if cid in files_by_id:
                chart_entry["files"] = files_by_id[cid]

            charts_yaml_list.append(chart_entry)

        # Build columns alias map from the actual fields used (exclude computed names like 'count')
        alias_fields = set()
        for item in charts_yaml_list:
            for key in ["x", "y", "color", "value"]:
                val = item.get(key)
                if isinstance(val, str) and val.startswith("$"):
                    alias_fields.add(val[1:])

        columns_map: Dict[str, str] = {alias: alias.replace("_", " ").title() for alias in sorted(alias_fields)}

        yaml_obj: Dict[str, Any] = {
            "report_title": report_title,
            "columns": columns_map,
            "charts": charts_yaml_list
        }

        return yaml_obj


class IntelligenceWorkflow:
    """
    Simplified workflow for integrating Intelligence Layer into DataFlow AI
    """
    
    def __init__(self, multi_mcp):
        self.intelligence_layer = IntelligenceLayer(multi_mcp)
    
    async def process_dataflow_request(self, user_query: str, analysis_results: Dict[str, Any],
                                     business_context: Dict[str, Any] = None,
                                     data_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for DataFlow AI Intelligence processing
        
        Args:
            user_query: Original user request
            analysis_results: Results from data ingestion/transformation/analysis
            business_context: Business domain, objectives, audience
            
        Returns:
            Complete DataFlow AI intelligence output
        """
        
        # Prepare business context if not provided
        if not business_context:
            business_context = {
                "domain": "general",
                "audience": ["business_users", "analysts"],
                "objectives": ["insights", "reporting", "visualization"],
                "query_context": user_query
            }
        
        # Add user query to context
        business_context["original_query"] = user_query
        
        # Process through Intelligence Layer
        intelligence_output = await self.intelligence_layer.process_data_analysis(
            analysis_results, business_context, data_context
        )
        
        # Add workflow metadata
        intelligence_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_intelligence",
            "user_query": user_query,
            "processing_timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "intelligence_layer_status": intelligence_output.get("processing_summary", {}).get("status", "unknown")
        }
        
        return intelligence_output

