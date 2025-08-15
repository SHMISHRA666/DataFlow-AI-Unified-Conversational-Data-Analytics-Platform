# orchestration_flow.py - DataFlow AI Orchestration Layer Workflow

import asyncio
from agentLoop.agents import AgentRunner
from utils.utils import log_step, log_error
from typing import Dict, Any, List


class OrchestrationLayer:
    """
    Orchestration Layer for DataFlow AI that manages the Discovery and Monitoring agents:
    - DiscoveryAgent: Discovers data sources, catalogs assets, identifies integration opportunities
    - MonitoringAgent: Monitors system health, data quality, performance, and compliance
    """
    
    def __init__(self, multi_mcp):
        self.multi_mcp = multi_mcp
        self.agent_runner = AgentRunner(multi_mcp)
    
    async def discover_and_catalog(self, organization_info: Dict[str, Any], 
                                 discovery_scope: Dict[str, Any] = None) -> Dict[str, Any]:
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
            discovery_results = await self._run_discovery(organization_info, discovery_scope)
            
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
    
    async def monitor_system(self, monitoring_context: Dict[str, Any]) -> Dict[str, Any]:
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
            monitoring_results = await self._run_monitoring(monitoring_context)
            
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
    
    async def comprehensive_orchestration(self, organization_info: Dict[str, Any],
                                        monitoring_context: Dict[str, Any],
                                        discovery_scope: Dict[str, Any] = None) -> Dict[str, Any]:
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
            discovery_task = self._run_discovery(organization_info, discovery_scope)
            monitoring_task = self._run_monitoring(monitoring_context)
            
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
    
    async def _run_discovery(self, organization_info: Dict[str, Any], 
                           discovery_scope: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the Discovery Agent"""
        
        input_data = {
            "organization_info": organization_info,
            "discovery_scope": discovery_scope or {},
            "task": "discover_and_catalog_data_sources",
            "objective": "Discover available data sources and create comprehensive data catalog"
        }
        
        result = await self.agent_runner.run_agent("DiscoveryAgent", input_data)
        
        if result["success"]:
            return result["output"]
        else:
            raise Exception(f"DiscoveryAgent failed: {result.get('error', 'Unknown error')}")
    
    async def _run_monitoring(self, monitoring_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the Monitoring Agent"""
        
        input_data = {
            "monitoring_context": monitoring_context,
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
        
        # Prepare discovery scope if not provided
        if not discovery_constraints:
            discovery_constraints = {
                "scope": "full_organization",
                "include_external": True,
                "security_level": "standard",
                "compliance_requirements": ["GDPR"]
            }
        
        # Add user request to context
        organization_context["user_request"] = user_request
        
        # Process through Orchestration Layer
        orchestration_output = await self.orchestration_layer.discover_and_catalog(
            organization_context, discovery_constraints
        )
        
        # Add workflow metadata
        orchestration_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_discovery",
            "user_request": user_request,
            "processing_timestamp": "auto-generated",
            "orchestration_status": orchestration_output.get("orchestration_summary", {}).get("status", "unknown")
        }
        
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
        system_context["monitoring_timestamp"] = "auto-generated"
        
        # Process through Orchestration Layer
        orchestration_output = await self.orchestration_layer.monitor_system(system_context)
        
        # Add workflow metadata
        orchestration_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_monitoring",
            "user_request": user_request,
            "processing_timestamp": "auto-generated",
            "orchestration_status": orchestration_output.get("orchestration_summary", {}).get("status", "unknown")
        }
        
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
            organization_context, system_context, discovery_constraints
        )
        
        # Add workflow metadata
        orchestration_output["workflow_info"] = {
            "workflow_type": "dataflow_ai_full_orchestration",
            "user_request": user_request,
            "processing_timestamp": "auto-generated",
            "orchestration_status": orchestration_output.get("orchestration_summary", {}).get("status", "unknown")
        }
        
        return orchestration_output


# Example usage and testing function
async def test_orchestration_layer():
    """Test function to demonstrate Orchestration Layer capabilities"""
    
    # Mock multi_mcp for testing
    class MockMCP:
        pass
    
    mock_mcp = MockMCP()
    workflow = OrchestrationWorkflow(mock_mcp)
    
    # Example organization context (would come from system configuration)
    sample_organization = {
        "name": "TechCorp Solutions",
        "infrastructure": {
            "databases": ["PostgreSQL", "MySQL", "MongoDB"],
            "cloud_platforms": ["AWS", "Azure"],
            "apis": ["REST", "GraphQL"],
            "file_systems": ["S3", "Azure Blob", "local storage"]
        },
        "compliance_requirements": ["GDPR", "SOX"],
        "security_level": "high",
        "departments": ["Sales", "Marketing", "Finance", "Operations"],
        "data_governance_maturity": "developing"
    }
    
    # Example monitoring context (would come from system monitoring)
    sample_monitoring = {
        "system_metrics": {
            "cpu_usage": 65.2,
            "memory_usage": 78.5,
            "disk_usage": 45.0,
            "network_throughput": 120.5
        },
        "pipeline_status": {
            "active_pipelines": 12,
            "failed_pipelines": 2,
            "average_execution_time": "15 minutes"
        },
        "data_quality": {
            "overall_score": 0.85,
            "issues_detected": 3,
            "anomalies_count": 1
        },
        "user_activity": {
            "active_users": 45,
            "queries_per_hour": 234,
            "peak_usage_time": "10:00-11:00"
        }
    }
    
    # Test discovery
    user_request = "Discover all available data sources in our organization and create a comprehensive data catalog"
    
    try:
        discovery_result = await workflow.process_discovery_request(
            user_request, sample_organization
        )
        print("✅ Discovery test completed successfully")
        
        # Test monitoring
        monitoring_request = "Monitor system health and provide performance recommendations"
        monitoring_result = await workflow.process_monitoring_request(
            monitoring_request, sample_monitoring
        )
        print("✅ Monitoring test completed successfully")
        
        # Test full orchestration
        full_request = "Perform comprehensive discovery and monitoring of our data ecosystem"
        full_result = await workflow.process_full_orchestration(
            full_request, sample_organization, sample_monitoring
        )
        print("✅ Full orchestration test completed successfully")
        
        return {
            "discovery": discovery_result,
            "monitoring": monitoring_result,
            "full_orchestration": full_result
        }
        
    except Exception as e:
        print(f"❌ Orchestration Layer test failed: {e}")
        return None


if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_orchestration_layer())
