# DataFlow AI - Unified Conversational Data Analytics Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Overview

DataFlow AI is a comprehensive, agentic conversational data analytics platform that transforms raw data into actionable business intelligence through intelligent multi-agent workflows. The platform combines advanced LLM-based agents with traditional data processing to deliver end-to-end analytics solutions.

### 🎯 Key Features

- **🤖 Multi-Agent Architecture**: Specialized agents for different analytical tasks
- **📊 Universal Data Support**: CSV, JSON, Excel, PDF, HTML, and image processing
- **🧠 Intelligent Routing**: Automatic query classification and pipeline selection
- **📈 Real-Time Visualization**: Chart generation with multiple output formats
- **🔍 RAG-Powered Document Analysis**: Advanced document processing with OCR and multimodal capabilities
- **⚡ Session-Based Processing**: Organized output management with tracking
- **🔧 Production-Ready**: Safe code execution, error handling, and monitoring

## 🏗️ Architecture

DataFlow AI follows a layered architecture with specialized agents working in concert:

```
User Query → Conversation Planner → Route to Pipeline
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Qualitative                      Quantitative
                    ↓                               ↓
            RAG Processing              Data Processing Layer
                                                    ↓
                                          Intelligence Layer
                                                    ↓
                                          Chart Execution
                                                    ↓
                                          Export & Reporting
```

### Core Components

#### 1. **Conversation Planner Agent** 🧭
- **Purpose**: Intelligent query classification and routing
- **Capabilities**:
  - Automatic detection of qualitative vs quantitative queries
  - File type-based routing (CSV/Excel → quantitative, PDF/HTML → qualitative)
  - Secondary classification for quantitative queries (Report/Chart/None)
  - Context propagation to downstream agents

#### 2. **Data Processing Layer** 📊
Four specialized agents for comprehensive data handling:

- **DataIngestionAgent** 📥: Multi-format file loading and validation
- **DataCleaningAgent** 🧼: Intelligent quality enhancement and standardization
- **DataTransformationAgent** 🔄: Feature engineering and normalization
- **DataAnalysisAgent** 📈: Statistical analysis and pattern discovery

#### 3. **Intelligence Layer** 🧠
Three-phase processing for business intelligence:

- **RecommendationAgent** 🎯: KPI recommendations and visualization suggestions
- **GenerationAgent** 🔧: Production-ready code and dashboard generation
- **NarrativeAgent** 📝: Executive summaries and business reports

#### 4. **Chart Executor Agent** 🎨
- **Purpose**: Safe execution of generated visualization code
- **Capabilities**:
  - Sandboxed code execution with timeout protection
  - Multi-format output (PNG, SVG, HTML, PDF)
  - Automatic dependency management
  - Quality validation and error recovery

#### 5. **Orchestration Layer** ⚙️
Operational backbone for system management:

- **DiscoveryAgent** 🔍: Data source discovery and cataloging
- **MonitoringAgent** 📊: System health and performance monitoring

#### 6. **RAG Ingest System** 📚
Advanced document processing and retrieval:

- **Multi-Format Support**: PDF, JSON, HTML, images
- **OCR Integration**: Text extraction from images with preprocessing
- **Multimodal Processing**: Gemini integration for image captioning
- **FAISS Indexing**: Efficient vector search and retrieval
- **Chat Interface**: Conversational document interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google API Key (for Gemini integration)
- Optional: Ollama for local embeddings

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Dataflow_AI
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
# or using uv
uv sync
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```env
GOOGLE_API_KEY=your_gemini_api_key
EMBED_API_URL=http://localhost:11434/api/embeddings  # Optional: for local embeddings
DOCS_DIR=./documents  # For RAG ingest
FAISS_DIR=./faiss_index  # For RAG storage
```

### Basic Usage

#### Interactive Mode
```bash
python main.py
```

#### RAG Document Processing
```bash
# Ingest documents
python agentLoop/rag_ingest.py ingest

# Search documents
python agentLoop/rag_ingest.py search "your query"

# Chat with documents
python agentLoop/rag_ingest.py chat "your question"
```

#### Testing the Complete Pipeline
```bash
cd examples_NR
python test_complete_dataflow_pipeline.py
```

## 📋 Detailed Usage

### Data Processing Workflow

#### Supported File Formats
- **CSV**: Standard comma-separated values with intelligent encoding detection
- **JSON**: JavaScript Object Notation with structure adaptation
- **Excel**: .xlsx and .xls files with automatic sheet selection
- **PDF**: Document processing with OCR and image extraction
- **HTML**: Web content processing with screenshot capabilities
- **Images**: PNG, JPG, JPEG, SVG with caption generation

#### Processing Pipeline
```python
from agentLoop.flow import AgentLoop4

# Initialize the system
agent_loop = AgentLoop4(multi_mcp)

# Process files with query
file_paths = ["sales_data.csv", "inventory.xlsx"]
result = await agent_loop.run(
    query="Analyze sales performance trends",
    file_manifest=file_info,
    uploaded_files=file_paths
)
```

### Intelligence Layer Usage

#### Standalone Intelligence Processing
```python
from agentLoop.intelligence_flow import IntelligenceWorkflow

workflow = IntelligenceWorkflow(multi_mcp)
result = await workflow.process_dataflow_request(
    user_query="Create sales performance dashboard",
    analysis_data=your_data,
    business_context=context
)
```

#### Chart Generation
```python
from agentLoop.chart_executor import ChartExecutor

executor = ChartExecutor(output_directory="my_charts")
result = await executor.process_generation_output(generation_output)

# Access generated files
for chart in result["charts_created"]:
    print(f"Created: {chart['file_path']}")
```

### Orchestration Layer Usage

#### Discovery Operations
```python
from agentLoop.orchestration_flow import OrchestrationWorkflow

workflow = OrchestrationWorkflow(multi_mcp)
result = await workflow.process_discovery_request(
    user_request="Discover all data sources in our organization",
    organization_context=org_info,
    discovery_constraints=constraints
)
```

#### Monitoring Operations
```python
result = await workflow.process_monitoring_request(
    user_request="Monitor system health and performance",
    system_context=current_metrics
)
```

### RAG Ingest System

#### Document Processing
```python
from agentLoop.rag_ingest import process_documents, search, chat_with_gemini

# Process documents and build index
process_documents(rebuild_index=True)

# Search for relevant content
results = search("your query", k=5)

# Chat with documents
answer = chat_with_gemini("your question", context_chunks)
```

#### Supported Document Types
- **PDFs**: Text extraction with image processing
- **JSON**: Structured data processing
- **HTML**: Web content with screenshot fallback
- **Images**: OCR with Gemini multimodal fallback

## 📁 Output Structure

DataFlow AI creates organized, session-based outputs:

```
generated_charts/
├── {session_id}/                           # Unique session directory
│   ├── data_processing_analyze_sales_trends_20250109_143052.json
│   ├── intelligence_analyze_sales_trends_20250109_143123.json
│   ├── orchestration_discovery_data_sources_20250109_143200.json
│   ├── charts.yaml                         # Intelligence layer charts
│   ├── narrative_insights.json            # Intelligence layer narratives
│   ├── latest_data_processing.json        # Symlink to most recent
│   ├── latest_intelligence.json           # Symlink to most recent
│   ├── latest_orchestration_discovery.json # Symlink to most recent
│   ├── png/                               # Chart exports
│   │   ├── sales_chart.png
│   │   └── trend_analysis.png
│   ├── svg/                               # Vector graphics
│   │   ├── sales_chart.svg
│   │   └── trend_analysis.svg
│   └── html/                              # Interactive charts
│       ├── sales_chart.html
│       └── trend_analysis.html
└── {another_session_id}/
    └── ...
```

## 🔧 Configuration

### Agent Configuration (`config/agent_config.yaml`)

```yaml
# Data Processing Layer Agents
DataIngestionAgent:
  prompt_file: "prompts/data_ingestion_prompt.txt"
  model: "gemini"
  mcp_servers: []

DataCleaningAgent:
  prompt_file: "prompts/data_cleaning_prompt.txt"
  model: "gemini"
  mcp_servers: []

DataTransformationAgent:
  prompt_file: "prompts/data_transformation_prompt.txt"
  model: "gemini"
  mcp_servers: []

DataAnalysisAgent:
  prompt_file: "prompts/data_analysis_prompt.txt"
  model: "gemini"
  mcp_servers: []

# Intelligence Layer Agents
RecommendationAgent:
  prompt_file: "prompts/recommendation_prompt.txt"
  model: "gemini"
  mcp_servers: []

GenerationAgent:
  prompt_file: "prompts/generation_prompt.txt"
  model: "gemini"
  mcp_servers: []

NarrativeAgent:
  prompt_file: "prompts/narrative_prompt.txt"
  model: "gemini"
  mcp_servers: []

# Chart Execution
ChartExecutorAgent:
  prompt_file: "prompts/chart_executor_prompt.txt"
  model: "gemini"
  mcp_servers: []

# Orchestration Layer Agents
DiscoveryAgent:
  prompt_file: "prompts/discovery_prompt.txt"
  model: "gemini"
  mcp_servers: ["websearch"]

MonitoringAgent:
  prompt_file: "prompts/monitoring_prompt.txt"
  model: "gemini"
  mcp_servers: []

# Conversation Planning
ConversationPlannerAgent:
  prompt_file: "prompts/conversation_planner_agent.txt"
  model: "gemini"
  mcp_servers: []
```

### RAG Configuration

Environment variables for RAG ingest:
```env
DOCS_DIR=./documents                    # Document storage directory
FAISS_DIR=./faiss_index                # FAISS index storage
EMBED_API_URL=http://localhost:11434/api/embeddings  # Embedding service
EMBED_MODEL=nomic-embed-text           # Embedding model
CHUNK_SIZE_WORDS=256                   # Text chunk size
CHUNK_OVERLAP_WORDS=40                 # Chunk overlap
TOP_K=5                                # Search result count
```

## 🧪 Testing

### Complete Pipeline Test
```bash
cd examples_NR
python test_complete_dataflow_pipeline.py
```

### Individual Component Tests
```bash
cd examples_NR
python test_individual_flows.py
```

### RAG System Test
```bash
# Place test documents in documents/ directory
python agentLoop/rag_ingest.py ingest
python agentLoop/rag_ingest.py search "test query"
```

## 💡 Business Value

### For Data Analysts
- **Immediate Results**: Generated code automatically creates usable chart files
- **Multiple Formats**: PNG for reports, HTML for dashboards, SVG for presentations
- **Quality Assurance**: Automatic validation ensures charts render correctly
- **Time Savings**: No manual code execution or file management required

### For Business Users
- **Ready-to-Use Charts**: Actual image files can be directly inserted into presentations
- **Interactive Options**: HTML charts provide dynamic exploration capabilities
- **Professional Quality**: High-resolution outputs suitable for publication
- **Consistent Branding**: Standardized styling across all generated charts

### For IT Teams
- **Safe Execution**: Sandboxed environment prevents security issues
- **Resource Management**: Controlled resource usage and automatic cleanup
- **Monitoring**: Detailed execution logs and performance metrics
- **Scalability**: Efficient handling of multiple chart generation requests

### For Data Engineers
- **Automated Discovery**: Reduces manual effort in discovering and cataloging data sources
- **Proactive Monitoring**: Early detection of issues before they impact business operations
- **Integration Guidance**: Clear recommendations for data integration strategies
- **Performance Optimization**: Data-driven insights for system optimization

## 🔍 Use Cases

### 1. Sales Analysis Dashboard
- **Input**: Sales transaction data (CSV/Excel)
- **Output**: Regional performance dashboard, trend analysis, recommendations
- **Charts**: Bar charts, line graphs, heatmaps, pie charts

### 2. Financial Reporting
- **Input**: Financial metrics and KPIs
- **Output**: Executive reports, variance analysis, forecasting insights
- **Formats**: PDF reports, interactive dashboards, presentation slides

### 3. Document Analysis
- **Input**: PDF reports, research papers, HTML content
- **Output**: Summaries, key insights, Q&A responses
- **Features**: OCR, image captioning, semantic search

### 4. Operational Analytics
- **Input**: Process and performance data
- **Output**: Efficiency dashboards, bottleneck identification, optimization recommendations
- **Monitoring**: Real-time alerts, performance tracking, cost optimization

### 5. Compliance Assessment
- **Input**: Data sources with compliance requirements
- **Output**: Compliance status, risk assessment, remediation plan
- **Governance**: Automated monitoring, audit trails, policy enforcement

## 🛠️ Troubleshooting

### Common Issues

#### 1. MCP Server Connection Failures
- Check `config/mcp_server_config.yaml`
- Verify API keys in `.env` file
- Ensure required services are accessible

#### 2. File Not Found Errors
- Verify file paths are correct and files exist
- Check file permissions
- Ensure proper working directory

#### 3. Chart Execution Failures
- Verify Python visualization libraries are installed
- Check output directory write permissions
- Review execution logs for specific errors

#### 4. RAG Index Issues
- Ensure documents are in the correct directory
- Check FAISS index and metadata files exist
- Verify embedding service is running

#### 5. Session Directory Issues
- Check write permissions for `generated_charts/` directory
- Verify symlink support on your platform
- Check available disk space

### Debugging

Enable detailed logging:
```python
from utils.utils import log_step
log_step("Debug message", symbol="🐛")
```

### Getting Help

If issues persist:
1. Run individual component tests to isolate problems
2. Check generated test reports for detailed error information
3. Review session directory contents
4. Check system logs and console output for error details

## 🔮 Future Enhancements

### Planned Features
- **Machine Learning Integration**: Add ML model recommendations and AutoML capabilities
- **Real-time Analytics**: Support for streaming data and real-time insights
- **Multi-modal Analysis**: Enhanced support for text, image, and video data analysis
- **Collaborative Intelligence**: Multi-agent collaboration and consensus mechanisms
- **Feedback Learning**: Incorporate user feedback to improve recommendations

### Performance Optimizations
- **Parallel Execution**: Multiple charts generated simultaneously
- **Caching**: Reuse of common data preparations and computations
- **Format Selection**: Generate only requested output formats
- **Resource Pooling**: Shared execution environments for efficiency

### Advanced Features
- **PDF Output**: Direct PDF generation for reports
- **Custom Styling**: Brand-specific color schemes and layouts
- **Batch Processing**: Efficient handling of large chart sets
- **Cloud Storage**: Direct upload to S3, Azure Blob, etc.
- **Template Library**: Pre-built chart templates and themes

## 🤝 Contributing

When extending DataFlow AI:

1. Follow the established agent pattern
2. Maintain clear input/output specifications
3. Include comprehensive error handling
4. Add appropriate tests and documentation
5. Consider backwards compatibility

### Development Guidelines

- Use structured prompting following the 2504.02052v2.pdf guidelines
- Implement session-based output management
- Include YAML configuration support
- Add comprehensive error handling and logging
- Follow the established agent architecture patterns

## 🙏 Acknowledgments

- Built on top of Google's Gemini AI models
- Utilizes FAISS for efficient vector search
- Integrates with various MCP servers for enhanced capabilities
- Inspired by modern agentic AI frameworks and best practices

---

**DataFlow AI** - Transforming data into intelligence through conversational analytics.
