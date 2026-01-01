# DataFlow AI - Unified Conversational Data Analytics Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Overview

DataFlow AI is a comprehensive, agentic conversational data analytics platform that transforms raw data into actionable business intelligence through intelligent multi-agent workflows. The platform combines advanced LLM-based agents with traditional data processing to deliver end-to-end analytics solutions.

### 🎯 Key Features

- **🤖 Multi-Agent Architecture**: Specialized agents for different analytical tasks
- **📊 Universal Data Support**: CSV, JSON, Excel, PDF, HTML, and image processing
- **🧠 Intelligent Routing**: Automatic query classification and pipeline selection
- **📈 Real-Time Visualization**: Chart generation with multiple output formats (PNG, SVG, HTML, PDF)
- **🔍 RAG-Powered Document Analysis**: Advanced document processing with OCR and multimodal capabilities
- **⚡ Session-Based Processing**: Organized output management with tracking
- **🔧 Production-Ready**: Safe code execution, error handling, and monitoring
- **🌐 Web Interface**: Full-featured Flask web application with user authentication
- **📊 Plotly Integration**: Interactive dashboards with Chart Studio integration
- **🚀 Modern Package Management**: UV-based dependency management with `pyproject.toml`

## 🏗️ Architecture

DataFlow AI follows a **two-server, layered architecture** with specialized agents working in concert:

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Server (Port 5000)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Web UI     │  │  Auth System │  │ File Upload  │     │
│  │  (Flask)     │  │              │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
│                            │                                │
│                            ▼                                │
│                   HTTP API Request                          │
└────────────────────────────┼────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend API Server (Port 5001)                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │         Conversation Planner Agent                     │ │
│  │              (Query Classification)                   │ │
│  └───────────────┬──────────────────────────────────────┘ │
│                  │                                         │
│         ┌────────┴────────┐                              │
│         ▼                 ▼                               │
│  Qualitative         Quantitative                         │
│         │                 │                               │
│         ▼                 ▼                               │
│  RAG Processing    Data Processing Layer                  │
│                           │                               │
│                           ▼                               │
│                  Intelligence Layer                        │
│                           │                               │
│                           ▼                               │
│                  Chart Execution                          │
│                           │                               │
│                           ▼                               │
│                  Export & Reporting                       │
│                           │                               │
│                           ▼                               │
│              Structured JSON Response                      │
└───────────────────────────┼───────────────────────────────┘
                            │
                            ▼
                    Frontend displays results
```

### Agent Workflow

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
  - Query type-based routing to quantitative and qualitative analysis
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

#### 7. **Export Layer** 📤
Professional report generation and distribution:

- **Plotly Reports**: Interactive HTML dashboards with Chart Studio integration
- **Multi-Format Export**: PNG, SVG, HTML, PDF outputs
- **Report Agent**: Automated report generation with customization

#### 8. **Web Interface** 🌐
Full-featured Flask application for browser-based access:

- **Two-Server Architecture**: Separated frontend (port 5000) and backend (port 5001) for scalability
- **User Authentication**: Secure login and registration system with password hashing
- **File Upload**: Multi-file upload with format validation and session management
- **Real-Time Processing**: Asynchronous query processing via backend API
- **Results Visualization**: Embedded charts and downloadable artifacts
- **Session Management**: User-specific session tracking and history
- **CORS Support**: Cross-origin resource sharing enabled for API access
- **Artifact Serving**: Direct file serving from generated_charts directory

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

**Option A: Using UV (Recommended)**
```bash
# Install uv if not already installed
pip install uv

# Sync dependencies from pyproject.toml
uv sync

# Activate the virtual environment
# On Windows: .venv\Scripts\activate
# On Unix/Mac: source .venv/bin/activate
```

**Option B: Using pip**
```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows: .venv\Scripts\activate
# On Unix/Mac: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key

# Optional: Flask secret key for session management
FLASK_SECRET_KEY=your-secret-key-here

# Optional: Custom output directory
OUTPUTS_DIR=./generated_charts

# Optional: RAG Ingest Configuration
EMBED_API_URL=http://localhost:11434/api/embeddings  # For local embeddings
DOCS_DIR=./documents  # For RAG ingest
FAISS_DIR=./faiss_index  # For RAG storage

# Optional: Plotly Chart Studio Integration
PLOTLY_USERNAME=your_plotly_username
PLOTLY_API_KEY=your_plotly_api_key
```

### Basic Usage

#### Web Application (Recommended)

The platform uses a **two-server architecture** for separation of concerns:

1. **Backend API Server** (`main.py`) - Handles agentic framework processing
2. **Frontend Web Server** (`web_app.py`) - Handles UI, authentication, and file uploads

**Starting the Application:**

```bash
# Terminal 1: Start the Backend API Server (Port 5001)
python main.py

# Terminal 2: Start the Frontend Web Server (Port 5000)
python web_app.py

# Access the application at http://localhost:5000
# - Register a new user account
# - Login and upload files
# - Enter your analysis query
# - View results and download artifacts
```

**Note:** Both servers must be running simultaneously. The frontend communicates with the backend via HTTP API calls.

#### Backend API Usage

The backend server (`main.py`) exposes a REST API endpoint:

```bash
POST http://13.234.11.1:5000/run_agent_loop
Content-Type: application/json

{
  "session_id": "your-session-id",
  "question": "your analysis query"
}
```

The backend processes requests through the agentic framework and returns structured responses with artifacts and analysis results.

#### RAG Document Processing
```bash
# Ingest documents
python agentLoop/rag_ingest.py ingest

# Search documents
python agentLoop/rag_ingest.py search "your query"

# Chat with documents
python agentLoop/rag_ingest.py chat "your question"
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

### Backend API Usage

The backend server (`main.py`) provides a REST API endpoint for programmatic access:

#### Endpoint: `/run_agent_loop`

**Method**: `POST`

**URL**: `http://13.234.11.1:5000/run_agent_loop`

**Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "session_id": "uuid-string",
  "question": "Your analysis query"
}
```

**Response**:
```json
{
  "success": true,
  "session_id": "uuid-string",
  "output_directory": "generated_charts/{session_id}",
  "final_answer_text": "Analysis results...",
  "artifacts": {
    "session_dir": "generated_charts/{session_id}",
    "exists": true,
    "files": {
      "html": [...],
      "png": [...],
      "svg": [...],
      "pdf": [...],
      "json": [...],
      "yaml": [...]
    },
    "preferred_entry": {
      "relative": "generated_charts/{session_id}/plotly_index.html",
      "public_url": "http://localhost:5000/static/generated_charts/{session_id}/plotly_index.html"
    }
  },
  "classification": {
    "user_query": "Your query",
    "primary_classification": "quantitative|qualitative",
    "secondary_classification": "Report|Chart|None"
  }
}
```

**Error Responses**:
- `400`: Missing required parameters (session_id or question)
- `404`: Session manifest not found
- `500`: AgentLoop initialization failed or processing error
- `502`: Backend service unavailable (from frontend)

**Note**: The session_id must correspond to an existing session manifest in the `results/` directory, created during file upload.

### Web Interface Usage

#### Prerequisites
Ensure both servers are running:
- Backend API server: `python main.py` (port 5001)
- Frontend web server: `python web_app.py` (port 5000)

#### User Registration and Authentication
1. Navigate to `http://localhost:5000/register`
2. Create a new account with username and password
3. Login at `http://localhost:5000/login`
4. Password hashing ensures secure credential storage

#### File Upload and Analysis
1. **Upload Files**: Select one or multiple files (CSV, Excel, JSON, PDF, HTML)
   - Files are stored in `uploads/` directory with unique identifiers
   - Session manifest created in `results/` directory
2. **Enter Query**: Type your analysis question or request
3. **Submit**: Click "Submit Query" to start processing
   - Frontend forwards request to backend API
   - Backend processes through agentic framework
4. **View Results**: Results appear with:
   - Text answer/summary
   - Interactive charts (if generated)
   - Downloadable artifacts (PNG, SVG, HTML, PDF)
   - Session ID for tracking
   - Public URLs for artifact access

#### UI Integration
The platform provides a structured JSON payload for UI consumption:
```json
{
  "success": true,
  "session_id": "60437395",
  "output_directory": "generated_charts/60437395",
  "final_answer_text": "Analysis summary...",
  "artifacts": {
    "session_dir": "generated_charts/60437395",
    "exists": true,
    "files": {
      "html": [
        {
          "relative": "generated_charts/60437395/plotly_index.html",
          "public_url": "http://localhost:5000/static/generated_charts/60437395/plotly_index.html"
        }
      ],
      "png": [...],
      "svg": [...],
      "pdf": [...],
      "json": [...],
      "yaml": [...]
    },
    "preferred_entry": {
      "relative": "generated_charts/60437395/plotly_index.html",
      "public_url": "http://localhost:5000/static/generated_charts/60437395/plotly_index.html"
    }
  },
  "classification": {
    "user_query": "Analyze sales trends",
    "primary_classification": "quantitative",
    "secondary_classification": "Report"
  }
}
```

**Key Features:**
- Session-based artifact organization
- Public URLs for direct file access
- Classification metadata for query routing
- Preferred entry point (plotly_index.html) for dashboards

## 📁 Output Structure

DataFlow AI creates organized, session-based outputs:

```
generated_charts/
├── {session_id}/                           # Unique session directory
│   ├── data_processing_analyze_sales_trends_20250109_143052.json
│   ├── intelligence_analyze_sales_trends_20250109_143123.json
│   ├── orchestration_discovery_data_sources_20250109_143200.json
│   ├── charts.yaml                         # Intelligence layer charts metadata
│   ├── narrative_insights.json            # Business narratives and insights
│   ├── resolved_insights.json             # Resolved chart insights
│   ├── results_intelligence_layer.json    # Complete intelligence output
│   ├── rag_answer.json                    # RAG pipeline responses
│   ├── plotly_index.html                  # Main Plotly dashboard (preferred entry)
│   ├── latest_data_processing.json        # Symlink to most recent
│   ├── latest_intelligence.json           # Symlink to most recent
│   ├── latest_orchestration_discovery.json # Symlink to most recent
│   ├── png/                               # Chart exports (PNG format)
│   │   ├── sales_chart.png
│   │   ├── trend_analysis.png
│   │   └── performance_dashboard.png
│   ├── svg/                               # Vector graphics (SVG format)
│   │   ├── sales_chart.svg
│   │   ├── trend_analysis.svg
│   │   └── performance_dashboard.svg
│   ├── html/                              # Interactive charts (HTML format)
│   │   ├── sales_chart.html
│   │   ├── trend_analysis.html
│   │   └── performance_dashboard.html
│   └── pdf/                               # Report exports (PDF format)
│       └── analysis_report.pdf
└── {another_session_id}/
    └── ...
```

### Key Output Files

- **`plotly_index.html`**: Main dashboard with all charts integrated (recommended for viewing)
- **`results_intelligence_layer.json`**: Complete analysis results with all metadata
- **`narrative_insights.json`**: Executive summaries and business recommendations
- **`rag_answer.json`**: Direct answers from RAG pipeline for simple queries
- **`charts.yaml`**: Chart definitions and configurations

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

### Web Application Configuration

**Frontend Server** (`web_app.py`):
```python
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'html', 'htm', 'pdf'}
BACKEND_API_URL = "http://13.234.11.1:5000/run_agent_loop"  # Backend endpoint
```

**Backend Server** (`main.py`):
```python
# CORS enabled for cross-origin requests
# AgentLoop4 initialized on startup
# Output directory: generated_charts/
# Session manifests stored in: results/
```

**Environment Variables:**
```env
FLASK_SECRET_KEY=your-secret-key-here  # For session management
OUTPUTS_DIR=./generated_charts  # Optional: custom output directory
```

# Running the main application
```bash
# Terminal 1: Start backend server
python main.py

# Terminal 2: Start frontend server
python web_app.py

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

#### 1. Web Application Issues
- **Port Already in Use**: 
  - Frontend (5000): Change port in `web_app.py` or stop conflicting service
  - Backend (5001): Change port in `main.py` and update `BACKEND_API_URL` in `web_app.py`
- **Backend Connection Failed**: Ensure `main.py is running on port 5001 before starting `web_app.py`
- **CORS Errors**: Verify `flask-cors` is installed and CORS is enabled in `main.py`
- **File Upload Fails**: Check `UPLOAD_FOLDER` permissions and `MAX_CONTENT_LENGTH` setting
- **Session Timeout**: Increase timeout or check `app.secret_key` configuration
- **Login Issues**: Verify `users.json` exists and has proper permissions
- **Session Manifest Not Found**: Ensure file upload completed successfully and `results/` directory exists

#### 2. Plotly Chart Studio Integration
- **Charts Not Uploading**: Verify `PLOTLY_USERNAME` and `PLOTLY_API_KEY` in `.env`
- **URL Generation Fails**: Check internet connectivity and Chart Studio credentials
- **Index HTML Missing**: Ensure `export/plotly_v6.py` runs successfully

#### 4. File Not Found Errors
- Verify file paths are correct and files exist
- Check file permissions
- Ensure proper working directory

#### 5. Chart Execution Failures
- Verify Python visualization libraries are installed (matplotlib, seaborn, plotly)
- Check output directory write permissions
- Review execution logs for specific errors
- Ensure kaleido is installed for static image export

#### 6. RAG Index Issues
- Ensure documents are in the correct directory
- Check FAISS index and metadata files exist
- Verify embedding service is running (Ollama or Google Embeddings)

#### 7. Session Directory Issues
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

## 📂 Project Structure

```
Dataflow_AI/
├── agentLoop/                     # Core agent system
│   ├── agents.py                  # Data processing agents
│   ├── conversation_planner_agent.py
│   ├── intelligence_flow.py       # Intelligence layer workflow
│   ├── orchestration_flow.py      # Orchestration layer workflow
│   ├── chart_executor.py          # Chart generation and execution
│   ├── rag_ingest.py             # RAG document processing
│   ├── flow.py                    # Main agent loop (AgentLoop4)
│   ├── data_processing_flow.py    # Data processing workflow
│   ├── contextManager.py          # Execution context management
│   └── model_manager.py           # Model configuration management
├── action/                        # Code execution and sandboxing
│   ├── executor.py                # Safe code execution
│   └── execute_step.py            # Step-by-step execution
├── export/                        # Report generation and export
│   ├── plotly_v6.py              # Plotly dashboard generation
│   └── report_agent.py           # Automated reporting
├── config/                        # Configuration files
│   ├── agent_config.yaml         # Agent configurations
│   ├── file_types.yaml           # Supported file types
│   ├── models.json               # Model configurations
│   └── profiles.yaml             # User profiles
├── prompts/                       # Agent prompts
│   ├── conversation_planner_agent.txt
│   ├── data_*.txt                # Data processing prompts
│   ├── recommendation_prompt.txt
│   ├── generation_prompt.txt
│   ├── narrative_prompt.txt
│   ├── chart_executor_prompt.txt
│   ├── discovery_prompt.txt
│   └── monitoring_prompt.txt
├── templates/                     # Web UI templates
│   ├── index.html                 # Main dashboard
│   ├── login.html                 # Login page
│   └── register.html             # Registration page
├── static/                        # Web UI static assets
│   ├── css/                       # Stylesheets
│   └── js/                        # JavaScript files
├── utils/                         # Utility functions
│   ├── utils.py                   # Logging and helpers
│   └── json_parser.py            # JSON parsing utilities
├── generated_charts/              # Output directory (session-based)
├── uploads/                       # User file uploads
├── results/                       # Session manifests
├── main.py                        # Backend API server (port 5001)
├── web_app.py                    # Frontend web server (port 5000)
├── requirements.txt              # pip dependencies
├── pyproject.toml                # UV package configuration
└── README.md                     # This file
```

## 🔮 Future Enhancements

### Planned Features
- **Machine Learning Integration**: Add ML model recommendations and AutoML capabilities
- **Real-time Analytics**: Support for streaming data and real-time insights
- **Advanced Multi-modal Analysis**: Enhanced support for video data analysis
- **Collaborative Intelligence**: Multi-agent collaboration and consensus mechanisms
- **Feedback Learning**: Incorporate user feedback to improve recommendations
- **API Gateway**: RESTful API for programmatic access

### Performance Optimizations
- **Parallel Execution**: Multiple charts generated simultaneously
- **Caching**: Reuse of common data preparations and computations
- **Format Selection**: Generate only requested output formats
- **Resource Pooling**: Shared execution environments for efficiency
- **Distributed Processing**: Support for multi-node deployment

### Advanced Features
- **Custom Styling**: Brand-specific color schemes and layouts
- **Batch Processing**: Efficient handling of large chart sets
- **Cloud Storage**: Direct upload to S3, Azure Blob, GCS
- **Template Library**: Pre-built chart templates and themes
- **Webhook Integration**: Event-driven notifications and integrations
- **Dashboard Embedding**: iFrame support for external embedding

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

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.11+**: Primary programming language
- **Google Gemini AI**: Large language model for intelligent processing
- **FAISS**: Efficient vector search and similarity matching

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Visualization
- **Plotly 6.3.0**: Interactive charts and dashboards
- **Chart Studio**: Cloud-based chart hosting and sharing
- **Matplotlib**: Static plot generation
- **Seaborn**: Statistical data visualization
- **Kaleido**: Static image export for Plotly

### Web Framework
- **Flask 3.1+**: Web application framework
- **Flask-CORS**: Cross-origin resource sharing support
- **Jinja2**: Template engine for HTML rendering
- **Werkzeug**: WSGI utility library and security utilities

### Document Processing
- **PyMuPDF / pymupdf4llm**: PDF text extraction
- **Tesseract / pytesseract**: OCR for image text extraction
- **BeautifulSoup4**: HTML parsing
- **Trafilatura**: Web content extraction
- **Pillow**: Image processing

### Package Management
- **UV**: Modern, fast Python package manager
- **pip**: Traditional Python package installer

### Additional Libraries
- **LlamaIndex**: Document indexing and retrieval
- **Pydantic**: Data validation and settings management
- **Rich**: Terminal formatting and progress bars
- **TQDM**: Progress bars for long-running operations

## 🙏 Acknowledgments

- Built on top of **Google's Gemini AI** models for intelligent processing
- Utilizes **FAISS** (Facebook AI Similarity Search) for efficient vector search
- Integrates with **MCP servers** for enhanced agent capabilities
- Inspired by modern agentic AI frameworks and best practices
- Document processing powered by **PyMuPDF**, **Tesseract OCR**, and **Trafilatura**
- Interactive visualizations enabled by **Plotly** and **Chart Studio**

## 📄 Documentation

- **Main README**: This file
- **Architecture Diagrams**: 
  - `DataFlow_Drawn_Architecture.jpeg` - Visual architecture overview
  - `DataFlowAI_Architecture.drawio` - Editable Draw.io diagram
  - `DataFlowAI_Architecture.html` - Interactive HTML diagram
- **Prompt Templates**: Located in `prompts/` directory
- **Configuration Examples**: Located in `config/` directory
- **Sample Data**: Test datasets included in project root

## 📞 Support and Contact

For issues, questions, or contributions:
1. Review the troubleshooting section above
2. Review configuration files in `config/`
3. Consult the UI integration guide for web interface issues

---

**DataFlow AI** - Transforming data into intelligence through conversational analytics.

*Version 0.2.0 - Modern Agentic Data Analytics Platform*
