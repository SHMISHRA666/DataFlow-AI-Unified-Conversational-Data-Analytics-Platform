# data_processing_flow.py - DataFlow AI Data Processing Layer Workflow

import asyncio
from agentLoop.agents import AgentRunner
from utils.utils import log_step, log_error
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json


class DataProcessingLayer:
    """
    Data Processing Layer for DataFlow AI that orchestrates the four core data processing agents:
    - DataIngestionAgent: Intelligent data loading and validation across multiple formats
    - DataCleaningAgent: Data quality enhancement and standardization
    - DataTransformationAgent: Feature engineering and normalization for analysis
    - DataAnalysisAgent: Statistical analysis and pattern discovery
    """
    
    def __init__(self, multi_mcp):
        self.multi_mcp = multi_mcp
        self.agent_runner = AgentRunner(multi_mcp)
    
    async def process_data_files(self, file_paths: List[str], 
                               processing_context: Dict[str, Any] = None,
                               user_query: str = "") -> Dict[str, Any]:
        """
        Complete data processing pipeline from raw files to analysis-ready data.
        
        Args:
            file_paths: List of file paths to process (CSV, JSON, Excel)
            processing_context: Processing preferences, quality thresholds, analysis goals
            user_query: Original user request for context-aware processing
            
        Returns:
            Complete data processing output ready for Intelligence Layer consumption
        """
        try:
            log_step("🔧 Starting Data Processing Layer pipeline", symbol="🚀")
            
            # Ensure processing context is available
            if processing_context is None:
                processing_context = {}
            
            # Add user query to context
            processing_context["original_query"] = user_query
            
            # Phase 1: Data Ingestion
            log_step("📥 Phase 1: Data Ingestion and Validation", symbol="1️⃣")
            ingestion_results = await self._run_data_ingestion(file_paths, processing_context, user_query)
            
            # Phase 2: Data Cleaning
            log_step("🧼 Phase 2: Data Quality Enhancement", symbol="2️⃣")
            cleaning_results = await self._run_data_cleaning(ingestion_results, processing_context, user_query)
            
            # Phase 3: Data Transformation
            log_step("🔄 Phase 3: Feature Engineering and Normalization", symbol="3️⃣")
            transformation_results = await self._run_data_transformation(cleaning_results, processing_context, user_query)
            
            # Phase 4: Data Analysis
            log_step("📊 Phase 4: Statistical Analysis and Pattern Discovery", symbol="4️⃣")
            analysis_results = await self._run_data_analysis(transformation_results, processing_context, user_query)
            
            # Prepare intelligence layer data
            intelligence_ready_data = self._prepare_intelligence_layer_data(
                analysis_results, transformation_results, processing_context
            )
            
            # Debug: Log the data_context and data_schema that was created
            data_context = intelligence_ready_data.get("data_context", {})
            data_schema = intelligence_ready_data.get("data_schema", {})
            available_columns = data_schema.get("available_columns", [])
            
            log_step(f"🔍 Data Processing Layer created data_context with keys: {list(data_context.keys())}")
            log_step(f"🔍 Data Processing Layer created data_schema with {len(available_columns)} columns")
            
            if available_columns:
                column_names = [col.get("name", "unknown") for col in available_columns[:5]]  # Show first 5
                log_step(f"🔍 Sample column names: {column_names}")
            
            if "df_csv_path" in data_context:
                log_step(f"🔍 df_csv_path: {data_context['df_csv_path']}")
            elif "df_excel_path" in data_context:
                log_step(f"🔍 df_excel_path: {data_context['df_excel_path']}")
            elif "df_json_path" in data_context:
                log_step(f"🔍 df_json_path: {data_context['df_json_path']}")
            
            # Combine all outputs
            processing_output = {
                "processing_summary": {
                    "status": "completed",
                    "operation": "data_processing_pipeline",
                    "processing_layer_version": "1.0",
                    "files_processed": len(file_paths),
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z"
                },
                "ingestion_results": ingestion_results,
                "cleaning_results": cleaning_results,
                "transformation_results": transformation_results,
                "analysis_results": analysis_results,
                "intelligence_layer_ready_data": intelligence_ready_data,
                "session_id": processing_context.get("session_id")  # ✅ Ensure session_id is included
            }
            
            log_step("✅ Data Processing Layer completed successfully", symbol="🎉")
            return processing_output
            
        except Exception as e:
            log_error(f"Data Processing Layer failed: {e}")
            return {
                "processing_summary": {
                    "status": "failed",
                    "operation": "data_processing_pipeline",
                    "error": str(e)
                }
            }
    
    async def _run_data_ingestion(self, file_paths: List[str], 
                                processing_context: Dict[str, Any], 
                                user_query: str = "") -> Dict[str, Any]:
        """Run the DataIngestionAgent for intelligent file loading and validation."""
        
        agent_input = {
            "files": file_paths,
            "instruction": f"Load and profile all provided datasets. User query context: {user_query}",
            "processing_context": processing_context,
            "task": "intelligent_data_ingestion",
            "reads": [],
            "writes": ["ingestion_results"]
        }
        
        result = await self.agent_runner.run_agent("DataIngestionAgent", agent_input)
        
        if result["success"]:
            log_step("✅ Data ingestion completed successfully")
            return result["output"]
        else:
            raise Exception(f"DataIngestionAgent failed: {result.get('error', 'Unknown error')}")
    
    async def _run_data_cleaning(self, ingestion_results: Dict[str, Any], 
                               processing_context: Dict[str, Any], 
                               user_query: str = "") -> Dict[str, Any]:
        """Run the DataCleaningAgent for data quality enhancement."""
        
        agent_input = {
            "inputs": {
                "ingestion_results": ingestion_results,
                "data_profile": ingestion_results.get("data_profile", {}),
                "processing_context": processing_context
            },
            "reads": ["ingestion_results"],
            "writes": ["cleaning_results"],
            "original_query": user_query,
            "task": "intelligent_data_cleaning"
        }
        
        result = await self.agent_runner.run_agent("DataCleaningAgent", agent_input)
        
        if result["success"]:
            log_step("✅ Data cleaning completed successfully")
            return result["output"]
        else:
            raise Exception(f"DataCleaningAgent failed: {result.get('error', 'Unknown error')}")
    
    async def _run_data_transformation(self, cleaning_results: Dict[str, Any], 
                                     processing_context: Dict[str, Any], 
                                     user_query: str = "") -> Dict[str, Any]:
        """Run the DataTransformationAgent for feature engineering and normalization."""
        
        agent_input = {
            "inputs": {
                "cleaning_results": cleaning_results,
                "cleaned_data_profile": cleaning_results.get("cleaned_data_profile", {}),
                "processing_context": processing_context
            },
            "reads": ["cleaning_results"],
            "writes": ["transformation_results"],
            "original_query": user_query,
            "task": "intelligent_data_transformation"
        }
        
        result = await self.agent_runner.run_agent("DataTransformationAgent", agent_input)
        
        if result["success"]:
            log_step("✅ Data transformation completed successfully")
            return result["output"]
        else:
            raise Exception(f"DataTransformationAgent failed: {result.get('error', 'Unknown error')}")
    
    async def _run_data_analysis(self, transformation_results: Dict[str, Any], 
                               processing_context: Dict[str, Any], 
                               user_query: str = "") -> Dict[str, Any]:
        """Run the DataAnalysisAgent for statistical analysis and pattern discovery."""
        
        # Get the actual data file path for the DataAnalysisAgent to read
        data_file_path = ""
        if transformation_results.get("transformed_data_profile", {}).get("file_path"):
            data_file_path = transformation_results.get("transformed_data_profile", {}).get("file_path")
        else:
            # Fall back to original files
            original_files = processing_context.get("original_files", [])
            if original_files:
                data_file_path = original_files[0]
        
        agent_input = {
            "inputs": {
                "transformation_results": transformation_results,
                "transformed_data_profile": transformation_results.get("transformed_data_profile", {}),
                "intelligence_layer_config": transformation_results.get("yaml_config", {}).get("intelligence_layer_config", {}),
                "processing_context": processing_context,
                "data_file_path": data_file_path,
                "data_format": transformation_results.get("transformed_data_profile", {}).get("format", "csv")
            },
            "reads": ["transformation_results"],
            "writes": ["analysis_results"],
            "original_query": user_query,
            "task": "intelligent_data_analysis"
        }
        
        result = await self.agent_runner.run_agent("DataAnalysisAgent", agent_input)
        
        if result["success"]:
            log_step("✅ Data analysis completed successfully")
            return result["output"]
        else:
            raise Exception(f"DataAnalysisAgent failed: {result.get('error', 'Unknown error')}")
    
    def _prepare_intelligence_layer_data(self, analysis_results: Dict[str, Any], 
                                       transformation_results: Dict[str, Any],
                                       processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data processing results for Intelligence Layer consumption.
        
        Converts data processing outputs into the format expected by the Intelligence Layer
        for visualization, reporting, and narrative generation.
        """
        
        try:
            # Get actual column names from the data file
            actual_columns = self._extract_actual_columns(processing_context)
            
            # Extract key data for Intelligence Layer - format as expected by Intelligence agents
            intelligence_data = {
                # Primary analysis results in the format expected by RecommendationAgent and GenerationAgent
                "key_findings": analysis_results.get("analysis_results", {}).get("key_findings", []),
                "statistical_analyses": analysis_results.get("analysis_results", {}).get("statistical_analyses", []),
                "data_segments": analysis_results.get("analysis_results", {}).get("data_segments", []),
                "summary_statistics": analysis_results.get("analysis_results", {}).get("summary_statistics", {}),
                
                # Visualization-ready data from DataAnalysisAgent
                "trend_analysis": analysis_results.get("intelligence_layer_ready_data", {}).get("trend_analysis", []),
                "top_performers": analysis_results.get("intelligence_layer_ready_data", {}).get("top_performers", []),
                "distributions": analysis_results.get("intelligence_layer_ready_data", {}).get("distributions", []),
                
                # Chart recommendations from transformation/analysis
                "chart_recommendations": analysis_results.get("yaml_config", {}).get("chart_recommendations", []),
                "narrative_insights": analysis_results.get("yaml_config", {}).get("narrative_insights", []),
                
                # Data schema and metadata for Intelligence Layer - CRITICAL: Use actual column names
                "data_schema": {
                    "available_columns": actual_columns,  # ✅ Real column names from data file
                    "total_records": transformation_results.get("transformed_data_profile", {}).get("final_shape", {}).get("rows", 0),
                    "total_features": transformation_results.get("transformed_data_profile", {}).get("final_shape", {}).get("columns", 0),
                    "feature_catalog": transformation_results.get("yaml_config", {}).get("feature_catalog", []),
                    "analytical_readiness": transformation_results.get("transformed_data_profile", {}).get("analytical_readiness", {}).get("analysis_ready_score", 0.0),
                    "primary_measures": self._identify_primary_measures(actual_columns),
                    "primary_dimensions": self._identify_primary_dimensions(actual_columns),
                    "time_columns": self._identify_time_columns(actual_columns)
                },
                
                # Quality and processing metadata
                "data_quality_summary": {
                    "ingestion_quality": analysis_results.get("analysis_results", {}).get("summary_statistics", {}).get("analysis_confidence_score", 0.0),
                    "transformation_confidence": transformation_results.get("yaml_config", {}).get("transformation_summary", {}).get("processing_confidence", 0.0),
                    "overall_quality": transformation_results.get("transformed_data_profile", {}).get("analytical_readiness", {}).get("analysis_ready_score", 0.0)
                },
                
                # Processing metadata
                "processing_metadata": {
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                    "user_query_context": processing_context.get("original_query", ""),
                    "data_processing_version": "1.0"
                },
                
                # Data context for chart execution (file paths, etc.)
                "data_context": self._build_data_context(transformation_results, processing_context)
            }
            
            return intelligence_data
            
        except Exception as e:
            log_error(f"Failed to prepare Intelligence Layer data: {e}")
            return {
                "analysis_data": {},
                "visualization_ready_data": {},
                "processing_metadata": {
                    "error": str(e),
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
    
    def _build_data_context(self, transformation_results: Dict[str, Any], 
                          processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build data_context for chart execution with proper file paths."""
        
        # Get the transformed data file path
        transformed_file = transformation_results.get("transformed_data_profile", {}).get("file_path", "")
        data_format = transformation_results.get("transformed_data_profile", {}).get("format", "csv")
        
        # Fall back to original file if no transformed file is available
        if not transformed_file:
            original_files = processing_context.get("original_files", [])
            if original_files:
                transformed_file = original_files[0]  # Use first original file
                # Infer format from file extension
                if transformed_file.endswith(('.xlsx', '.xls')):
                    data_format = "excel"
                elif transformed_file.endswith('.csv'):
                    data_format = "csv"
                elif transformed_file.endswith('.json'):
                    data_format = "json"
        
        # Ensure we have a valid file path
        if not transformed_file:
            # Last resort: try to find any file in the processing context
            original_files = processing_context.get("original_files", [])
            if original_files:
                transformed_file = original_files[0]
                data_format = "csv"  # Default to CSV
            else:
                # This should not happen in normal operation
                log_error("No file path available for data_context - this will cause chart execution to fail")
                return {
                    "session_id": processing_context.get("session_id", ""),
                    "ready_for_visualization": False,
                    "error": "No file path available"
                }
        
        # Build data context based on file format
        data_context = {
            "session_id": processing_context.get("session_id", ""),
            "ready_for_visualization": True,
            "source_files": processing_context.get("original_files", []),
            "data_format": data_format
        }
        
        # Set appropriate file path based on format
        if data_format.lower() == "csv" or transformed_file.endswith('.csv'):
            data_context["df_csv_path"] = transformed_file
            data_context["df_excel_path"] = None
        elif data_format.lower() in ["excel", "xlsx", "xls"] or transformed_file.endswith(('.xlsx', '.xls')):
            data_context["df_excel_path"] = transformed_file  
            data_context["df_csv_path"] = None
        elif data_format.lower() == "json" or transformed_file.endswith('.json'):
            data_context["df_json_path"] = transformed_file
            data_context["df_csv_path"] = None
            data_context["df_excel_path"] = None
        else:
            # Default to CSV for compatibility
            data_context["df_csv_path"] = transformed_file
            data_context["df_excel_path"] = None
            
        return data_context
    
    def _extract_actual_columns(self, processing_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actual column names and types from the data file."""
        try:
            import pandas as pd
            from pathlib import Path
            
            # Get the data file path
            original_files = processing_context.get("original_files", [])
            if not original_files:
                log_error("No original files found in processing context")
                return []
            
            data_file = original_files[0]
            data_file_path = Path(data_file)
            
            if not data_file_path.exists():
                log_error(f"Data file does not exist: {data_file_path}")
                return []
            
            # Read the data file to get actual column names and types
            try:
                if data_file.endswith('.csv'):
                    df = pd.read_csv(data_file, nrows=5)  # Read only first 5 rows for column info
                elif data_file.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(data_file, nrows=5)
                elif data_file.endswith('.json'):
                    df = pd.read_json(data_file, lines=True, nrows=5)
                else:
                    log_error(f"Unsupported file format: {data_file}")
                    return []
                
                # Extract column information
                columns = []
                for col in df.columns:
                    col_info = {
                        "name": col,
                        "type": self._infer_column_type(df[col]),
                        "description": f"Column: {col}",
                        "sample_values": df[col].dropna().head(3).astype(str).tolist(),
                        "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                        "unique_count": df[col].nunique()
                    }
                    columns.append(col_info)
                
                log_step(f"🔍 Extracted {len(columns)} actual columns from data file")
                return columns
                
            except Exception as e:
                log_error(f"Failed to read data file {data_file}: {e}")
                return []
                
        except Exception as e:
            log_error(f"Failed to extract actual columns: {e}")
            return []
    
    def _infer_column_type(self, series) -> str:
        """Infer column type from pandas series."""
        try:
            if pd.api.types.is_numeric_dtype(series):
                return "numerical"
            elif pd.api.types.is_datetime64_any_dtype(series):
                return "temporal"
            elif pd.api.types.is_bool_dtype(series):
                return "boolean"
            elif pd.api.types.is_categorical_dtype(series):
                return "categorical"
            else:
                return "text"
        except:
            return "text"
    
    def _identify_primary_measures(self, columns: List[Dict[str, Any]]) -> List[str]:
        """Identify primary measure columns (numerical columns suitable for aggregation)."""
        measures = []
        for col in columns:
            if col["type"] == "numerical":
                measures.append(col["name"])
        return measures
    
    def _identify_primary_dimensions(self, columns: List[Dict[str, Any]]) -> List[str]:
        """Identify primary dimension columns (categorical columns suitable for grouping)."""
        dimensions = []
        for col in columns:
            if col["type"] in ["categorical", "text"]:
                dimensions.append(col["name"])
        return dimensions
    
    def _identify_time_columns(self, columns: List[Dict[str, Any]]) -> List[str]:
        """Identify time-related columns."""
        time_cols = []
        for col in columns:
            if col["type"] == "temporal" or "date" in col["name"].lower() or "time" in col["name"].lower():
                time_cols.append(col["name"])
        return time_cols


class DataProcessingWorkflow:
    """
    Simplified workflow for integrating Data Processing Layer into DataFlow AI
    """
    
    def __init__(self, multi_mcp):
        self.data_processing_layer = DataProcessingLayer(multi_mcp)
    
    async def process_files_request(self, user_request: str, file_paths: List[str],
                                  processing_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for DataFlow AI data processing
        
        Args:
            user_request: User's data processing request
            file_paths: List of files to process (CSV, JSON, Excel)
            processing_context: Processing preferences and configuration
            
        Returns:
            Complete DataFlow AI data processing output ready for Intelligence Layer
        """
        
        # Ensure processing context is a dict
        if not processing_context:
            processing_context = {}
        
        # Add user request and original files to context
        processing_context["user_request"] = user_request
        processing_context["original_files"] = file_paths
        
        # Process through Data Processing Layer
        processing_output = await self.data_processing_layer.process_data_files(
            file_paths, processing_context, user_query=user_request
        )
        
        # Add workflow metadata
        processing_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_data_processing",
            "user_request": user_request,
            "files_processed": len(file_paths),
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_status": processing_output.get("processing_summary", {}).get("status", "unknown")
        }
        
        # Persist results per session and use case
        self._persist_result_per_session(processing_output, processing_context)

        return processing_output
    
    def _persist_result_per_session(self, data: Dict[str, Any], processing_context: Dict[str, Any]) -> None:
        """Persist processing results to session-specific file."""
        try:
            # Get session ID and create session-specific directory
            session_id = processing_context.get("session_id", str(int(__import__('time').time()))[-8:])
            
            # Create per-session directory structure
            session_dir = Path("generated_charts") / str(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create use-case specific filename based on request
            user_request = processing_context.get("user_request", "data_processing")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Sanitize user request for filename
            safe_request = "".join(c for c in user_request if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_request = safe_request.replace(' ', '_')[:50]  # Limit length
            
            # Create session and use-case specific filename
            filename = f"data_processing_{safe_request}_{timestamp}.json"
            output_file = session_dir / filename
            
            # Save the processing results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            log_step(f"💾 Data processing results saved to: {output_file}")
            
            # Also save a latest symlink for easy access
            latest_file = session_dir / "latest_data_processing.json"
            try:
                if latest_file.exists():
                    latest_file.unlink()
                # Create symlink to latest file (cross-platform compatible)
                with open(latest_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            except Exception:
                pass  # Symlink creation might fail on some systems
                
        except Exception as e:
            log_error(f"Failed to persist data processing results: {e}")
