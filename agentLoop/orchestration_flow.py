# orchestration_flow.py - DataFlow AI Orchestration Layer Workflow

import asyncio
from agentLoop.agents import AgentRunner
from utils.utils import log_step, log_error
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json


class OrchestrationLayer:
    """
    Orchestration Layer for DataFlow AI that manages the Discovery and Monitoring agents:
    - DiscoveryAgent: Discovers data sources, catalogs assets, identifies integration opportunities
    - MonitoringAgent: Monitors system health, data quality, performance, and compliance
    """
    
    def __init__(self, multi_mcp):
        self.multi_mcp = multi_mcp
        self.agent_runner = AgentRunner(multi_mcp)
    
    async def discover_and_catalog(self, organization_context: Dict[str, Any], 
                                 discovery_constraints: Dict[str, Any] = None,
                                 user_query: str = "") -> Dict[str, Any]:
        """
        Complete discovery and cataloging process.
        
        Args:
            organization_info: Organization details, infrastructure, requirements
            discovery_scope: Scope and constraints for discovery process
            
        Returns:
            Complete discovery output with sources, catalog, and recommendations
        """
        try:
            log_step("🔍 Starting data discovery and cataloging", symbol="🚀")
            
            # Run Discovery Agent
            discovery_results = await self._run_discovery(organization_context, discovery_constraints, user_query)
            
            log_step("✅ Discovery and cataloging completed", symbol="🎉")
            return {
                "discovery_results": discovery_results,
                "orchestration_summary": {
                    "status": "completed",
                    "operation": "discovery_and_catalog",
                    "orchestration_layer_version": "1.0"
                }
            }
            
        except Exception as e:
            log_error(f"Discovery and cataloging failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "orchestration_summary": {
                    "status": "failed",
                    "operation": "discovery_and_catalog",
                    "error": str(e)
                }
            }
    
    async def monitor_system(self, system_context: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
        """
        Complete system monitoring process.
        
        Args:
            monitoring_context: System metrics, logs, and monitoring configuration
            
        Returns:
            Complete monitoring output with health status, alerts, and recommendations
        """
        try:
            log_step("📊 Starting system monitoring", symbol="🚀")
            
            # Run Monitoring Agent
            monitoring_results = await self._run_monitoring(system_context, user_query)
            
            log_step("✅ System monitoring completed", symbol="🎉")
            return {
                "monitoring_results": monitoring_results,
                "orchestration_summary": {
                    "status": "completed",
                    "operation": "system_monitoring",
                    "orchestration_layer_version": "1.0"
                }
            }
            
        except Exception as e:
            log_error(f"System monitoring failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "orchestration_summary": {
                    "status": "failed",
                    "operation": "system_monitoring",
                    "error": str(e)
                }
            }
    
    async def comprehensive_orchestration(self, organization_context: Dict[str, Any],
                                        system_context: Dict[str, Any],
                                        discovery_constraints: Dict[str, Any] = None,
                                        user_query: str = "") -> Dict[str, Any]:
        """
        Run both discovery and monitoring in a coordinated workflow.
        
        Args:
            organization_info: Organization details for discovery
            monitoring_context: System state for monitoring
            discovery_scope: Optional discovery constraints
            
        Returns:
            Combined orchestration results
        """
        try:
            log_step("🎯 Starting comprehensive orchestration", symbol="🚀")
            
            # Run both agents in parallel for efficiency
            discovery_task = self._run_discovery(organization_context, discovery_constraints, user_query)
            monitoring_task = self._run_monitoring(system_context, user_query)
            
            discovery_results, monitoring_results = await asyncio.gather(
                discovery_task, monitoring_task, return_exceptions=True
            )
            
            # Handle results
            orchestration_output = {
                "orchestration_summary": {
                    "status": "completed",
                    "operation": "comprehensive_orchestration",
                    "orchestration_layer_version": "1.0"
                }
            }
            
            # Add discovery results
            if isinstance(discovery_results, Exception):
                log_error(f"Discovery failed: {discovery_results}")
                orchestration_output["discovery_error"] = str(discovery_results)
            else:
                orchestration_output["discovery_results"] = discovery_results
            
            # Add monitoring results
            if isinstance(monitoring_results, Exception):
                log_error(f"Monitoring failed: {monitoring_results}")
                orchestration_output["monitoring_error"] = str(monitoring_results)
            else:
                orchestration_output["monitoring_results"] = monitoring_results
            
            # Determine overall status
            if isinstance(discovery_results, Exception) and isinstance(monitoring_results, Exception):
                orchestration_output["orchestration_summary"]["status"] = "failed"
            elif isinstance(discovery_results, Exception) or isinstance(monitoring_results, Exception):
                orchestration_output["orchestration_summary"]["status"] = "partial_success"
            
            log_step("✅ Comprehensive orchestration completed", symbol="🎉")
            return orchestration_output
            
        except Exception as e:
            log_error(f"Comprehensive orchestration failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "orchestration_summary": {
                    "status": "failed",
                    "operation": "comprehensive_orchestration",
                    "error": str(e)
                }
            }
    
    async def _run_discovery(self, organization_context: Dict[str, Any], 
                           discovery_constraints: Dict[str, Any] = None,
                           user_query: str = "") -> Dict[str, Any]:
        """Run the Discovery Agent"""
        
        input_data = {
            "user_query": user_query,
            "organization_context": organization_context or {},
            "discovery_constraints": discovery_constraints or {},
            "task": "discover_and_catalog_data_sources",
            "objective": "Discover available data sources and create comprehensive data catalog"
        }
        
        result = await self.agent_runner.run_agent("DiscoveryAgent", input_data)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(f"DiscoveryAgent failed: {result.get('error', 'Unknown error')}")
    
    async def _run_monitoring(self, system_context: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
        """Run the Monitoring Agent"""
        
        input_data = {
            "user_query": user_query,
            "system_context": system_context or {},
            "metrics": (system_context or {}).get("system_metrics", {}),
            "monitoring_policies": (system_context or {}).get("monitoring_policies", {}),
            "task": "monitor_system_health_and_performance",
            "objective": "Monitor system health, data quality, and operational performance"
        }
        
        result = await self.agent_runner.run_agent("MonitoringAgent", input_data)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(f"MonitoringAgent failed: {result.get('error', 'Unknown error')}")


class OrchestrationWorkflow:
    """
    Simplified workflow for integrating Orchestration Layer into DataFlow AI
    """
    
    def __init__(self, multi_mcp):
        self.orchestration_layer = OrchestrationLayer(multi_mcp)
    
    async def process_discovery_request(self, user_request: str, 
                                      organization_context: Dict[str, Any],
                                      discovery_constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for DataFlow AI discovery processing
        
        Args:
            user_request: User's discovery request
            organization_context: Organization infrastructure and requirements
            discovery_constraints: Scope and limitations for discovery
            
        Returns:
            Complete DataFlow AI discovery output
        """
        
        # Ensure constraints is a dict
        if not discovery_constraints:
            discovery_constraints = {}
        
        # Add user request to context
        organization_context["user_request"] = user_request
        
        # Process through Orchestration Layer
        orchestration_output = await self.orchestration_layer.discover_and_catalog(
            organization_context, discovery_constraints, user_query=user_request
        )
        
        # Add workflow metadata
        orchestration_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_discovery",
            "user_request": user_request,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "orchestration_status": orchestration_output.get("orchestration_summary", {}).get("status", "unknown")
        }
        
        # Persist results per session and use case
        self._persist_result_per_session(orchestration_output, "discovery", user_request, organization_context)

        return orchestration_output
    
    async def process_monitoring_request(self, user_request: str,
                                       system_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for DataFlow AI monitoring processing
        
        Args:
            user_request: User's monitoring request
            system_context: Current system state and metrics
            
        Returns:
            Complete DataFlow AI monitoring output
        """
        
        # Add user request to context
        system_context["user_request"] = user_request
        system_context["monitoring_timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Process through Orchestration Layer
        orchestration_output = await self.orchestration_layer.monitor_system(system_context, user_query=user_request)
        
        # Add workflow metadata
        orchestration_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_monitoring",
            "user_request": user_request,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "orchestration_status": orchestration_output.get("orchestration_summary", {}).get("status", "unknown")
        }
        
        # Persist results per session and use case
        self._persist_result_per_session(orchestration_output, "monitoring", user_request, system_context)

        return orchestration_output
    
    async def process_full_orchestration(self, user_request: str,
                                       organization_context: Dict[str, Any],
                                       system_context: Dict[str, Any],
                                       discovery_constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process both discovery and monitoring in a single coordinated workflow
        
        Args:
            user_request: User's request for orchestration
            organization_context: Organization info for discovery
            system_context: System state for monitoring
            discovery_constraints: Optional discovery limitations
            
        Returns:
            Complete orchestration output
        """
        
        # Add user request to contexts
        organization_context["user_request"] = user_request
        system_context["user_request"] = user_request
        
        # Process through comprehensive orchestration
        orchestration_output = await self.orchestration_layer.comprehensive_orchestration(
            organization_context, system_context, discovery_constraints, user_query=user_request
        )
        
        # Add workflow metadata
        orchestration_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_full_orchestration",
            "user_request": user_request,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "orchestration_status": orchestration_output.get("orchestration_summary", {}).get("status", "unknown")
        }
        
        # Persist results per session and use case
        self._persist_result_per_session(orchestration_output, "full_orchestration", user_request, 
                                       {**organization_context, **system_context})

        return orchestration_output

    def _persist_result_per_session(self, data: Dict[str, Any], operation_type: str, 
                                   user_request: str, context: Dict[str, Any]) -> None:
        """
        Persist orchestration results in session-based directory structure like intelligence layer.
        
        Args:
            data: Result data to persist
            operation_type: Type of operation ("discovery", "monitoring", "full_orchestration")
            user_request: Original user request for filename generation
            context: Context containing session information
        """
        try:
            import re
            from pathlib import Path
            
            # Get session_id from context, generate one if not present
            session_id = context.get("session_id") or context.get("dag_context", {}).get("session_id") or "default_session"
            
            # Create session directory
            session_dir = Path("generated_charts") / str(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate clean filename from user request
            if user_request and user_request.strip():
                # Clean and truncate user request for filename
                clean_request = re.sub(r'[^\w\s-]', '', user_request.lower())
                clean_request = re.sub(r'\s+', '_', clean_request.strip())
                clean_request = clean_request[:50]  # Limit length
            else:
                clean_request = "orchestration_request"
            
            # Generate timestamped filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"orchestration_{operation_type}_{clean_request}_{timestamp}.json"
            
            # Write result file
            result_file = session_dir / filename
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # Create/update latest symlink
            latest_symlink = session_dir / f"latest_orchestration_{operation_type}.json"
            if latest_symlink.exists() or latest_symlink.is_symlink():
                latest_symlink.unlink()
            
            # Create relative symlink
            try:
                latest_symlink.symlink_to(filename)
            except (OSError, NotImplementedError):
                # Fallback: copy file for systems that don't support symlinks
                import shutil
                shutil.copy2(result_file, latest_symlink)
            
            log_step(f"💾 Orchestration {operation_type} results saved to: {result_file}")
            log_step(f"🔗 Latest symlink updated: {latest_symlink}")
            
        except Exception as e:
            log_error(f"Failed to persist orchestration results: {e}")
            # Fallback to simple file write in current directory
            try:
                fallback_file = f"orchestration_{operation_type}_result.json"
                with open(fallback_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                log_step(f"💾 Fallback save: {fallback_file}")
            except Exception as fallback_error:
                log_error(f"Even fallback persistence failed: {fallback_error}")
