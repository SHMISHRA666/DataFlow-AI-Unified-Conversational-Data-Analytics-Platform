"""
ConversationPlannerAgent - Classifies user queries as qualitative or quantitative
and routes to appropriate workflow approaches.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from utils.utils import log_step, log_error, load_file_type_config
from agentLoop.model_manager import ModelManager
from utils.json_parser import parse_llm_json
import json


@dataclass
class ConversationPlan:
    """Result of conversation planning analysis"""
    user_query: str
    primary_classification: str  # "qualitative" | "quantitative"
    secondary_classification: str  # "Report" | "Chart" | "None"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "primary_classification": self.primary_classification,
            "secondary_classification": self.secondary_classification,
        }


class ConversationPlannerAgent:
    """
    Agent that analyzes user queries and file uploads to determine the optimal
    processing approach (qualitative RAG vs quantitative analysis).
    """

    def __init__(self, model_name: str = "gemini"):
        self.model_manager = ModelManager(model_name)
        self.prompt_file = "prompts/conversation_planner_agent.txt"

    def _load_prompt(self) -> str:
        """Load the system prompt from file"""
        prompt_path = Path(self.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _analyze_file_types(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze uploaded files to understand their nature with fixed/flexible classification"""
        if not files:
            return {
                "file_types": [],
                "primary_data_nature": "none",
                "has_fixed_quantitative": False,
                "has_fixed_qualitative": False,
                "has_flexible_json": False
            }

        extensions = []
        
        # Fixed file type classifications (regardless of user intent)
        cfg = load_file_type_config()
        fixed_quantitative = cfg["fixed_quantitative"]
        fixed_qualitative = cfg["fixed_qualitative"]
        flexible_types = cfg["flexible_types"]

        has_fixed_quantitative = False
        has_fixed_qualitative = False
        has_flexible_json = False

        for file_info in files:
            file_path = file_info.get('path', file_info.get('name', ''))
            if isinstance(file_path, str):
                ext = Path(file_path).suffix.lower()
                if ext:
                    extensions.append(ext)
                    
                    if ext in fixed_quantitative:
                        has_fixed_quantitative = True
                    elif ext in fixed_qualitative:
                        has_fixed_qualitative = True
                    elif ext in flexible_types:
                        has_flexible_json = True

        # Determine primary nature based on fixed types first
        if has_fixed_quantitative and not has_fixed_qualitative:
            primary_nature = "structured"  # Only quantitative files
        elif has_fixed_qualitative and not has_fixed_quantitative:
            primary_nature = "unstructured"  # Only qualitative files
        elif has_fixed_quantitative and has_fixed_qualitative:
            primary_nature = "mixed"  # Both types present
        elif has_flexible_json:
            primary_nature = "flexible"  # Only JSON files (intent-dependent)
        else:
            primary_nature = "unknown"

        return {
            "file_types": list(set(extensions)),
            "primary_data_nature": primary_nature,
            "has_fixed_quantitative": has_fixed_quantitative,
            "has_fixed_qualitative": has_fixed_qualitative,
            "has_flexible_json": has_flexible_json
        }

    def _build_context(self, user_query: str, files: List[Dict[str, Any]], 
                      file_profiles: Optional[Dict[str, Any]] = None) -> str:
        """Build the context string for the prompt"""
        
        # Prepare files information
        files_info = []
        if files:
            for i, file_info in enumerate(files):
                file_name = file_info.get('name', f'file_{i+1}')
                file_path = file_info.get('path', '')
                file_size = file_info.get('size', 0)
                
                files_info.append({
                    "name": file_name,
                    "path": file_path,
                    "size": file_size,
                    "extension": Path(file_name).suffix.lower()
                })

        context = {
            "user_query": user_query,
            "files": files_info,
            "file_profiles": file_profiles or {}
        }

        return json.dumps(context, indent=2)

    async def classify_conversation(self, user_query: str, 
                                  files: List[Dict[str, Any]] = None,
                                  file_profiles: Optional[Dict[str, Any]] = None) -> ConversationPlan:
        """
        Main method to classify user conversation and determine processing approach.
        
        Args:
            user_query: The user's question or request
            files: List of uploaded file information
            file_profiles: Optional detailed file content analysis
            
        Returns:
            ConversationPlan object with classification and routing information
        """
        try:
            log_step("🧭 Analyzing conversation context", symbol="🔍")
            
            # Load system prompt
            system_prompt = self._load_prompt()
            
            # Build context for analysis
            context_data = self._build_context(user_query, files or [], file_profiles)
            
            # Format the full prompt safely without interpreting other braces in examples
            full_prompt = (
                system_prompt
                .replace("{user_query}", user_query)
                .replace("{files}", json.dumps(files or [], indent=2))
                .replace("{file_profiles}", json.dumps(file_profiles or {}, indent=2))
            )
            
            # Get model response
            log_step("🤖 Querying ConversationPlannerAgent", symbol="🧠")
            response = await self.model_manager.generate_text(full_prompt)
            
            # Parse JSON response
            parsed_response = parse_llm_json(response)
            
            # Validate required fields for simplified format
            if "primary_classification" not in parsed_response:
                raise ValueError("Model response missing required 'primary_classification' field")
            if "secondary_classification" not in parsed_response:
                raise ValueError("Model response missing required 'secondary_classification' field")
            
            # Create ConversationPlan object with simplified format
            plan = ConversationPlan(
                user_query=user_query,
                primary_classification=parsed_response.get("primary_classification"),
                secondary_classification=parsed_response.get("secondary_classification")
            )
            log_step(f"User query: " + user_query, symbol="💬")
            log_step(f"✅ Classification: {plan.primary_classification}" + 
                    (f" → {plan.secondary_classification}" if plan.secondary_classification != "None" else ""),
                    symbol="🎯")
            
            return plan
            
        except Exception as e:
            log_error(f"ConversationPlannerAgent failed: {e}")
            
            # Fallback classification should be INTENT-BASED, avoid file-type hardcoding
            uq = (user_query or "").strip().lower()
            # Simple intent heuristics when LLM fails
            qualitative_signals = [
                "summarize", "explain", "what does", "describe", "key points",
                "overview", "gist", "interpret"
            ]
            quantitative_signals = [
                "how many", "count", "average", "mean", "sum", "total", "trend",
                "top ", "bottom ", "compare", "distribution", "variance", "median",
                "chart", "graph", "plot", "visualize"
            ]
            if any(s in uq for s in qualitative_signals) and not any(s in uq for s in quantitative_signals):
                fallback_classification = "qualitative"
            else:
                # Default towards quantitative simple answer when uncertain
                fallback_classification = "quantitative"
            
            return ConversationPlan(
                user_query=user_query,
                primary_classification=fallback_classification,
                secondary_classification="None"
            )

    async def run_classification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run method compatible with the AgentRunner interface.
        
        Args:
            input_data: Dictionary containing user_query, files, file_profiles
            
        Returns:
            Dictionary with success status and simplified classification results
        """
        try:
            user_query = input_data.get("user_query", input_data.get("query", ""))
            files = input_data.get("files", [])
            file_profiles = input_data.get("file_profiles", {})
            
            # Handle different input formats
            if isinstance(files, str):
                # Single file path
                files = [{"path": files, "name": Path(files).name}]
            elif not isinstance(files, list):
                files = []
                
            # Run classification
            plan = await self.classify_conversation(user_query, files, file_profiles)
            
            return {
                "success": True,
                "output": plan.to_dict()
            }
            
        except Exception as e:
            log_error(f"ConversationPlannerAgent run failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": {
                    "user_query": input_data.get("user_query", input_data.get("query", "")),
                    "primary_classification": "qualitative",
                    "secondary_classification": "None"
                }
            }


# Factory function for AgentRunner compatibility
async def create_conversation_planner_agent(**kwargs) -> ConversationPlannerAgent:
    """Factory function to create ConversationPlannerAgent instance"""
    return ConversationPlannerAgent(**kwargs)


# For backwards compatibility
ConversationAgent = ConversationPlannerAgent
