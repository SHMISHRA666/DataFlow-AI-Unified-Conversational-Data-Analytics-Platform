import pandas as pd
import os
import json
from typing import Dict, List, Any, Optional

class DataIngestionAgent:
    """Loads and validates datasets from local files in various formats."""

    def __init__(self):
        """Initialize the DataIngestionAgent."""
        self.supported_formats = ['.csv', '.json', '.xlsx', '.xls']
        print("📁 Data Ingestion Agent: Initialized for local file processing")

    def load_dataset(self, file_path: str, limit: Optional[int] = None, 
                    output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load a dataset from a local file.
        
        Args:
            file_path: Path to the dataset file
            limit: Maximum number of rows to load (None for all rows)
            output_path: Path to save the processed data (optional)
        
        Returns:
            DataFrame containing the loaded data
        """
        print("📁 Data Ingestion Agent: Starting dataset loading...")
        
        # Validate file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return pd.DataFrame()
        
        # Validate file format
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            print(f"❌ Unsupported file format: {file_ext}")
            print(f"Supported formats: {', '.join(self.supported_formats)}")
            return pd.DataFrame()
        
        print(f"📁 Loading {file_ext.upper()} file: {file_path}")
        
        try:
            # Load data based on file format
            if file_ext == '.csv':
                df = self._load_csv(file_path)
            elif file_ext == '.json':
                df = self._load_json(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            else:
                print(f"❌ Unexpected file format: {file_ext}")
                return pd.DataFrame()
            
            if df.empty:
                print("⚠️  No data loaded from file")
                return df
            
            # Apply row limit if specified
            if limit and limit > 0:
                if limit <= len(df):
                    df = df.head(limit)
                    print(f"📊 Limited to {limit} rows")
                else:
                    print(f"⚠️  Requested limit ({limit}) exceeds available rows ({len(df)})")
            
            # Basic data validation
            df = self._validate_data(df)
            
            # Save to output path if specified
            if output_path:
                self._save_data(df, output_path)
            
            print(f"✅ Data Ingestion Agent: Completed! Loaded {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return pd.DataFrame()

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ CSV loaded successfully with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try without specifying encoding
            df = pd.read_csv(file_path)
            print("✅ CSV loaded successfully (default encoding)")
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def _load_json(self, file_path: str) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of objects
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Single object or nested structure
                if any(isinstance(v, list) for v in data.values()):
                    # Find the list with the most items (likely the main data)
                    main_key = max(data.keys(), key=lambda k: len(data[k]) if isinstance(data[k], list) else 0)
                    df = pd.DataFrame(data[main_key])
                else:
                    # Single object
                    df = pd.DataFrame([data])
            else:
                print("❌ Unexpected JSON structure")
                return pd.DataFrame()
            
            print("✅ JSON loaded successfully")
            return df
            
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return pd.DataFrame()

    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load data from Excel file."""
        try:
            # Try to read all sheets and use the one with the most data
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                # Single sheet
                df = pd.read_excel(file_path, sheet_name=sheet_names[0])
            else:
                # Multiple sheets - find the one with most data
                max_rows = 0
                best_sheet = sheet_names[0]
                
                for sheet in sheet_names:
                    try:
                        temp_df = pd.read_excel(file_path, sheet_name=sheet, nrows=1000)
                        if len(temp_df) > max_rows:
                            max_rows = len(temp_df)
                            best_sheet = sheet
                    except:
                        continue
                
                df = pd.read_excel(file_path, sheet_name=best_sheet)
                print(f"📊 Using sheet: {best_sheet}")
            
            print("✅ Excel loaded successfully")
            return df
            
        except Exception as e:
            print(f"❌ Error loading Excel: {e}")
            return pd.DataFrame()

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data validation and cleaning."""
        print("🔍 Validating dataset...")
        
        # Remove completely empty rows and columns
        initial_rows, initial_cols = df.shape
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.shape != (initial_rows, initial_cols):
            print(f"🧹 Removed {initial_rows - df.shape[0]} empty rows and {initial_cols - df.shape[1]} empty columns")
        
        # Check for common data quality issues
        print(f"📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show column information
        print("\n📋 Column information:")
        for col in df.columns:
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            print(f"  • {col}: {non_null} non-null values, {null_count} nulls ({dtype})")
        
        return df

    def _save_data(self, df: pd.DataFrame, output_path: str):
        """Save data to the specified output path."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Determine file extension and save accordingly
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif output_path.endswith(('.xlsx', '.xls')):
                df.to_excel(output_path, index=False, engine='openpyxl')
            elif output_path.endswith('.json'):
                df.to_json(output_path, orient='records', indent=2, force_ascii=False)
            else:
                # Default to CSV
                output_path = output_path + '.csv' if '.' not in output_path else output_path.replace('.', '.csv.')
                df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"💾 Saved processed data to: {output_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save to {output_path}: {e}")

    def get_available_fields(self) -> List[str]:
        """Get list of available fields from the loaded dataset."""
        return [
            "patent_id",
            "patent_title", 
            "patent_date",
            "patent_year",
            "patent_abstract",
            "patent_type",
            "patent_earliest_application_date",
            "patent_term_extension",
            "assignee_organization",
            "inventor_name"
        ]

    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive information about the loaded dataset."""
        if df.empty:
            return {}
        
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "sample_data": df.head(3).to_dict('records')
        }
        
        return info


if __name__ == "__main__":
    # Example usage
    agent = DataIngestionAgent()
    
    # Example: Load a CSV file
    # df = agent.load_dataset(
    #     file_path="data/sample_dataset.csv",
    #     limit=100,
    #     output_path="data/01_raw/raw_data.csv"
    # )
    
    # if not df.empty:
    #     info = agent.get_dataset_info(df)
    #     print(f"\nDataset Info: {info['shape'][0]} rows × {info['shape'][1]} columns")
    #     print(f"Columns: {', '.join(info['columns'])}")
    #     print(f"Memory usage: {info['memory_usage'] / 1024:.2f} KB")
    
    print("📁 DataIngestionAgent ready for use!")
    print("Use load_dataset() method to load your dataset files.")