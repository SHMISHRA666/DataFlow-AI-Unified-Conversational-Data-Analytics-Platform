# Agents package for Dataset Processing Workflow
# This file makes the agents directory a Python package

__version__ = "1.0.0"
__author__ = "Dataset Processing Workflow"

# Import all agent classes for easy access
from .agent_ingestion import DataIngestionAgent
from .agent_cleaning import DataCleaningAgent
from .agent_transformation import DataTransformationAgent
from .agent_analysis import DataAnalysisAgent

__all__ = [
    'DataIngestionAgent',
    'DataCleaningAgent', 
    'DataTransformationAgent',
    'DataAnalysisAgent'
]
