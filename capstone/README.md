# Dataset Processing Workflow

This tool processes and analyzes custom datasets in CSV, JSON, and Excel formats. It provides a complete workflow from data ingestion to analysis without requiring external API calls.

## Features

- **File Format Support**: CSV, JSON, Excel (.xlsx, .xls)
- **Data Processing Pipeline**: Ingestion → Cleaning → Transformation → Analysis
- **Flexible Configuration**: Interactive mode or command-line options
- **Multiple Output Formats**: CSV, JSON, Excel reports

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode (Recommended)

Run the workflow interactively to get guided setup:

```bash
python main.py --interactive
```

This will prompt you for:
- Dataset file path
- Number of rows to process (optional)
- Which workflow steps to skip

### Command Line Mode

Process a dataset directly from command line:

```bash
# Process all rows
python main.py --file your_dataset.csv

# Process specific number of rows
python main.py --file your_dataset.csv --limit 100

# Skip specific steps
python main.py --file your_dataset.csv --skip-cleaning --skip-transformation
```

### Available Options

- `--file`: Path to your dataset file
- `--limit`: Number of rows to process
- `--skip-ingestion`: Skip data loading step
- `--skip-cleaning`: Skip data cleaning step
- `--skip-transformation`: Skip data transformation step
- `--skip-analysis`: Skip analysis step
- `--output-dir`: Base directory for output files (default: "data")
- `--interactive`: Run in interactive mode

## Workflow Steps

### 1. Data Ingestion
- Loads dataset from specified file
- Validates file format and content
- Performs basic data quality checks
- Saves to `data/01_raw/raw_data.csv`

### 2. Data Cleaning
- Removes duplicate records
- Cleans text fields
- Removes empty rows and columns
- Saves to `data/02_cleaned/cleaned_data.csv`

### 3. Data Transformation
- Extracts years from date fields
- Normalizes assignee and inventor names
- Handles missing values
- Saves to `data/03_transformed/transformed_data.csv`

### 4. Data Analysis
- Innovation trend analysis
- Top assignees ranking
- Top inventors ranking
- Patent type distribution
- Generates multiple output formats

## Output Structure

```
data/
├── 01_raw/           # Raw loaded data
├── 02_cleaned/       # Cleaned data
├── 03_transformed/   # Transformed data
└── 04_analysis_results/  # Analysis reports
    ├── analysis_innovation_trend.csv
    ├── analysis_top_assignees.csv
    ├── analysis_top_inventors.csv
    ├── analysis_patent_types.csv
    ├── analysis_analysis_summary.xlsx
    └── analysis_analysis.json
```

## Example

1. Place your dataset file in the project directory
2. Run the workflow:
   ```bash
   python main.py --interactive
   ```
3. Follow the prompts to configure your analysis
4. Check the `data/` directory for results

## Supported File Formats

- **CSV**: Standard comma-separated values
- **JSON**: JavaScript Object Notation (handles various structures)
- **Excel**: .xlsx and .xls files (automatically selects best sheet)

## Data Requirements

The tool works best with datasets containing these columns (all optional):
- `patent_id`: Unique identifier
- `patent_title`: Patent title
- `patent_date`: Date information
- `patent_abstract`: Abstract text
- `patent_type`: Type classification
- `assignee_organization`: Organization name
- `inventor_name`: Inventor name

## Troubleshooting

- **File not found**: Ensure the file path is correct and the file exists
- **Unsupported format**: Use only .csv, .json, .xlsx, or .xls files
- **Memory issues**: Use the `--limit` option to process fewer rows
- **Excel errors**: Ensure `openpyxl` is installed: `pip install openpyxl`
