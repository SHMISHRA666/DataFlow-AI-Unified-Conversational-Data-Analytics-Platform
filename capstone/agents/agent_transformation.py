import pandas as pd
import re
import ast
import os

class DataTransformationAgent:
    """Normalizes data from a file for consistency and analysis."""

    def _normalize_assignee(self, assignee_list_str):
        try:
            if pd.isna(assignee_list_str) or assignee_list_str == "":
                return ['UNKNOWN']
            
            # Handle different formats
            if isinstance(assignee_list_str, list):
                assignee_list = assignee_list_str
            elif isinstance(assignee_list_str, str):
                # Try to parse as list string
                try:
                    assignee_list = ast.literal_eval(assignee_list_str)
                except (ValueError, SyntaxError):
                    # Treat as single string
                    assignee_list = [assignee_list_str]
            else:
                return ['UNKNOWN']
            
            if not assignee_list: 
                return ['UNKNOWN']
            
            normalized_list = []
            ASSIGNEE_MAP = {
                'UNKNOWN': 'UNKNOWN',
                'N/A': 'UNKNOWN',
                '': 'UNKNOWN'
            }
            
            for assignee in assignee_list:
                if not assignee or pd.isna(assignee):
                    continue
                assignee_upper = str(assignee).upper()
                if assignee_upper in ASSIGNEE_MAP:
                    normalized_list.append(ASSIGNEE_MAP[assignee_upper])
                    continue
                # Remove common suffixes
                assignee_upper = re.sub(r',?\s+(INC|LLC|CORP|LTD|CO|COMPANY)\.?$', '', assignee_upper)
                normalized_list.append(assignee_upper.strip())
            
            return list(set(normalized_list)) if normalized_list else ['UNKNOWN']
        except Exception:
            return ['UNKNOWN']

    def _extract_year_from_date(self, date_str):
        """Extract year from various date formats"""
        if pd.isna(date_str) or date_str == "":
            return None
        
        try:
            # Try pandas datetime parsing
            date_obj = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(date_obj):
                return int(date_obj.year)
        except:
            pass
        
        # Try regex extraction for common date formats
        year_patterns = [
            r'(\d{4})',  # YYYY
            r'(\d{2})/(\d{2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, str(date_str))
            if match:
                if len(match.groups()) == 1:
                    return int(match.group(1))
                elif len(match.groups()) == 3:
                    return int(match.group(3))  # Assume last group is year
        
        return None

    def _first_from_semicolon(self, value: str) -> str:
        """Return the first non-empty token from a semicolon-separated string."""
        if not isinstance(value, str) or value.strip() == "":
            return ""
        parts = [p.strip() for p in value.split(';') if p.strip()]
        return parts[0] if parts else ""

    def process(self, input_path: str, output_path: str):
        print("🔧 Transformation Agent: Starting...")
        
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
        
        # Transform the data
        initial_count = len(df)
        
        # Extract year from various date fields
        date_columns = ['patent_date', 'patent_earliest_application_date']
        for col in date_columns:
            if col in df.columns:
                df[f'{col}_year'] = df[col].apply(self._extract_year_from_date)
        
        # Normalize assignees
        if 'assignee_organization' in df.columns:
            df['assignees_normalized'] = df['assignee_organization'].apply(self._normalize_assignee)
            # Extract first assignee when semicolon-separated
            df['first_assignee_organization'] = df['assignee_organization'].apply(self._first_from_semicolon)
        
        # Normalize inventors
        if 'inventor_name' in df.columns:
            df['inventors_normalized'] = df['inventor_name'].apply(lambda x: [x.strip()] if pd.notna(x) and x != "" else ['UNKNOWN'])
            # Extract first inventor when semicolon-separated
            df['first_inventor_name'] = df['inventor_name'].apply(self._first_from_semicolon)
        
        # Fill missing values
        df.fillna({
            'patent_title': 'No Title Provided', 
            'patent_abstract': 'No Abstract Provided',
            'assignee_organization': 'UNKNOWN',
            'inventor_name': 'UNKNOWN',
            'first_assignee_organization': '',
            'first_inventor_name': ''
        }, inplace=True)
        
        # Remove rows with no patent_id
        if 'patent_id' in df.columns:
            df = df.dropna(subset=['patent_id'])
        
        final_count = len(df)
        print(f"🔄 Transformed data: {initial_count} → {final_count} records")
        
        # Export in the same format as fetch_assignee_inventor
        self._export_files(df, output_path)
        print(f"✅ Transformation Agent: Completed! Processed {len(df)} records")

    def _export_files(self, df: pd.DataFrame, prefix: str):
        """Export files in the same format as fetch_assignee_inventor - Copy.py"""
        csv_path = f"{prefix}.csv"
        xlsx_path = f"{prefix}.xlsx"
        json_path = f"{prefix}.json"

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