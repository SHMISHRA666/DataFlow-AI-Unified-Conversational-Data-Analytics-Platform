import pandas as pd
import re
import os

class DataCleaningAgent:
    """Cleans and prepares the raw patent data from a file."""

    def _clean_text(self, text):
        if not isinstance(text, str): 
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def process(self, input_path: str, output_path: str):
        print("🧼 Cleaning Agent: Starting...")
        
        # Read the input file based on extension
        file_ext = os.path.splitext(input_path)[1].lower()
        try:
            if file_ext == '.csv':
                df = pd.read_csv(input_path, encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(input_path, engine='openpyxl')
            elif file_ext == '.json':
                df = pd.read_json(input_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
        
        print(f"📊 Loaded {len(df)} records from {input_path}")
        print(f"Columns found: {list(df.columns)}")
        
        # Clean the data
        if 'patent_title' in df.columns:
            df['patent_title'] = df['patent_title'].apply(self._clean_text)
        if 'patent_abstract' in df.columns:
            df['patent_abstract'] = df['patent_abstract'].apply(self._clean_text)
        
        # Remove duplicates based on patent_id if available
        if 'patent_id' in df.columns:
            initial_count = len(df)
            df.drop_duplicates(subset=['patent_id'], inplace=True)
            final_count = len(df)
            print(f"🔄 Removed {initial_count - final_count} duplicate patent_id records")
        
        # Export in the same format as fetch_assignee_inventor
        self._export_files(df, output_path)
        print(f"✅ Cleaning Agent: Completed! Processed {len(df)} records")

    def _export_files(self, df: pd.DataFrame, output_prefix: str):
        """Export files in the same format as fetch_assignee_inventor - Copy.py"""
        csv_path = f"{output_prefix}.csv"
        xlsx_path = f"{output_prefix}.xlsx"
        json_path = f"{output_prefix}.json"

        df.to_csv(csv_path, index=False, encoding="utf-8")
        try:
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
        except Exception as exc:
            print(f"Excel export failed (install openpyxl?): {exc}")
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)

        print(f"📁 Exported files:")
        print(f"  CSV:  {csv_path}")
        print(f"  XLSX: {xlsx_path}")
        print(f"  JSON: {json_path}")