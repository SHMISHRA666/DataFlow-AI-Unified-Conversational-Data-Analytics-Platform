# intelligence_flow.py - DataFlow AI Intelligence Layer Workflow

import asyncio
from agentLoop.agents import AgentRunner
from agentLoop.chart_executor import ChartExecutorAgent
from utils.utils import log_step, log_error
from typing import Dict, Any, List


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
                                  business_context: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
            # Phase 1: Generate Recommendations
            log_step("📊 Phase 1: Generating recommendations", symbol="1️⃣")
            recommendations = await self._generate_recommendations(analysis_data, business_context)
            
            # Phase 2: Generate Artifacts
            log_step("🔧 Phase 2: Generating dashboards and code", symbol="2️⃣")
            generated_artifacts = await self._generate_artifacts(recommendations, analysis_data, business_context)
            
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
            "analysis_data": analysis_data,
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
                                business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Phase 2: Generate dashboards, code, and BI configurations with chart execution"""
        
        input_data = {
            "recommendations": recommendations,
            "analysis_data": analysis_data,
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
        
        # Step 2: Execute generated charts and create actual visualization files
        log_step("🎨 Executing chart code and creating visualization files", symbol="2️⃣")
        
        # Check if there are visualizations to execute
        has_visualizations = (
            "generated_visualizations" in generation_output and 
            len(generation_output["generated_visualizations"]) > 0
        )
        
        if has_visualizations:
            chart_execution_result = await self.chart_executor_agent.execute_charts(
                generation_output,
                execution_config={
                    "output_formats": ["png", "svg", "html"],
                    "quality": "high",
                    "interactive": True
                }
            )
            
            if chart_execution_result["success"]:
                # Combine generation output with chart execution results
                enhanced_output = generation_output.copy()
                enhanced_output["chart_execution"] = chart_execution_result["output"]
                
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
            "recommendations": recommendations,
            "generated_artifacts": generated_artifacts,
            "analysis_data": analysis_data,
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


class IntelligenceWorkflow:
    """
    Simplified workflow for integrating Intelligence Layer into DataFlow AI
    """
    
    def __init__(self, multi_mcp):
        self.intelligence_layer = IntelligenceLayer(multi_mcp)
    
    async def process_dataflow_request(self, user_query: str, analysis_results: Dict[str, Any],
                                     business_context: Dict[str, Any] = None) -> Dict[str, Any]:
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
            analysis_results, business_context
        )
        
        # Add workflow metadata
        intelligence_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_intelligence",
            "user_query": user_query,
            "processing_timestamp": "auto-generated",
            "intelligence_layer_status": intelligence_output.get("processing_summary", {}).get("status", "unknown")
        }
        
        return intelligence_output


# Example usage and testing function
async def test_intelligence_layer():
    """Test function to demonstrate Intelligence Layer capabilities"""
    
    # Mock multi_mcp for testing
    class MockMCP:
        pass
    
    mock_mcp = MockMCP()
    workflow = IntelligenceWorkflow(mock_mcp)
    
    # Example analysis data (would come from earlier DataFlow AI stages)
    sample_analysis = {
        "schema": {
            "columns": ["date", "sales", "region", "product", "customer_segment"],
            "types": {"date": "datetime", "sales": "numeric", "region": "categorical", 
                     "product": "categorical", "customer_segment": "categorical"}
        },
        "statistics": {
            "sales": {"mean": 15000, "std": 5000, "min": 1000, "max": 50000},
            "records_count": 10000,
            "date_range": {"start": "2023-01-01", "end": "2024-12-31"}
        },
        "patterns": {
            "trends": ["increasing_sales_q4", "regional_variance_high"],
            "anomalies": ["unusual_spike_november"],
            "correlations": [{"vars": ["sales", "region"], "strength": 0.7}]
        }
    }
    
    sample_context = {
        "domain": "retail_sales",
        "audience": ["executives", "sales_managers"],
        "objectives": ["performance_tracking", "trend_analysis", "forecasting"]
    }
    
    # Test the workflow
    user_query = "Analyze our sales performance and create a dashboard showing key trends and insights"
    
    try:
        result = await workflow.process_dataflow_request(
            user_query, sample_analysis, sample_context
        )
        print("✅ Intelligence Layer test completed successfully")
        return result
    except Exception as e:
        print(f"❌ Intelligence Layer test failed: {e}")
        return None


if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_intelligence_layer())
