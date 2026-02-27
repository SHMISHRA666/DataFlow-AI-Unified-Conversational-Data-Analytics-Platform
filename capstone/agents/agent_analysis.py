import pandas as pd
import json
import os
import ast

class DataAnalysisAgent:
    """Performs analysis and generates multiple output files."""

    def analyze(self, input_path: str, output_base_path: str):
        print("📊 Analysis Agent: Starting...")
        
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
        
        # 1. Innovation Trend Analysis
        innovation_trend_df = self._analyze_innovation_trend(df)
        
        # 2. Top Assignees Analysis
        top_assignees_df = self._analyze_top_assignees(df)
        
        # 3. Top Inventors Analysis
        top_inventors_df = self._analyze_top_inventors(df)
        
        # 4. Patent Type Distribution
        patent_type_df = self._analyze_patent_types(df)
        
        # Create and Save Outputs
        self._save_analysis_results(
            output_base_path, 
            innovation_trend_df, 
            top_assignees_df, 
            top_inventors_df, 
            patent_type_df,
            df
        )
        
        print(f"✅ Analysis Agent: Completed! Generated comprehensive reports")

    def _analyze_innovation_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze patent filing trends over years"""
        print("📈 Analyzing innovation trends...")
        
        # Try different year columns
        year_columns = ['patent_year', 'patent_date_year', 'patent_earliest_application_date_year']
        year_col = None
        
        for col in year_columns:
            if col in df.columns:
                year_col = col
                break
        
        if year_col is None:
            # Create a dummy trend if no year data
            print("⚠️  No year data found, creating dummy trend")
            return pd.DataFrame({
                'year': ['No Year Data'],
                'patent_count': [len(df)]
            })
        
        # Filter out invalid years
        valid_years = df[year_col].dropna()
        if len(valid_years) == 0:
            return pd.DataFrame({
                'year': ['No Valid Year Data'],
                'patent_count': [len(df)]
            })
        
        trend_df = valid_years.value_counts().sort_index().reset_index()
        trend_df.columns = ['year', 'patent_count']
        return trend_df

    def _analyze_top_assignees(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze top assignees by patent count"""
        print("🏢 Analyzing top assignees...")
        
        # Try different assignee columns
        assignee_columns = ['assignees_normalized', 'assignee_organization']
        assignee_col = None
        
        for col in assignee_columns:
            if col in df.columns:
                assignee_col = col
                break
        
        if assignee_col is None:
            return pd.DataFrame({
                'assignee': ['No Assignee Data'],
                'patent_count': [len(df)]
            })
        
        try:
            # Handle different assignee formats
            if assignee_col == 'assignees_normalized':
                # Already normalized list format
                assignee_data = df[assignee_col]
            else:
                # Convert to list format
                assignee_data = df[assignee_col].apply(lambda x: [x] if pd.notna(x) and x != "" else [])
            
            # Explode and count
            assignee_exploded = df.explode(assignee_col)
            top_assignees = assignee_exploded[assignee_col].value_counts().nlargest(20).reset_index()
            top_assignees.columns = ['assignee', 'patent_count']
            return top_assignees
            
        except Exception as e:
            print(f"⚠️  Error analyzing assignees: {e}")
            return pd.DataFrame({
                'assignee': ['Error in Assignee Analysis'],
                'patent_count': [0]
            })

    def _analyze_top_inventors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze top inventors by patent count"""
        print("👨‍🔬 Analyzing top inventors...")
        
        # Try different inventor columns
        inventor_columns = ['inventors_normalized', 'inventor_name']
        inventor_col = None
        
        for col in inventor_columns:
            if col in df.columns:
                inventor_col = col
                break
        
        if inventor_col is None:
            return pd.DataFrame({
                'inventor': ['No Inventor Data'],
                'patent_count': [0]
            })
        
        try:
            # Handle different inventor formats
            if inventor_col == 'inventors_normalized':
                # Already normalized list format
                inventor_data = df[inventor_col]
            else:
                # Convert to list format
                inventor_data = df[inventor_col].apply(lambda x: [x] if pd.notna(x) and x != "" else [])
            
            # Explode and count
            inventor_exploded = df.explode(inventor_col)
            top_inventors = inventor_exploded[inventor_col].value_counts().nlargest(20).reset_index()
            top_inventors.columns = ['inventor', 'patent_count']
            return top_inventors
            
        except Exception as e:
            print(f"⚠️  Error analyzing inventors: {e}")
            return pd.DataFrame({
                'inventor': ['Error in Inventor Analysis'],
                'patent_count': [0]
            })

    def _analyze_patent_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze distribution of patent types"""
        print("📋 Analyzing patent types...")
        
        if 'patent_type' not in df.columns:
            return pd.DataFrame({
                'patent_type': ['No Type Data'],
                'count': [len(df)]
            })
        
        type_dist = df['patent_type'].value_counts().reset_index()
        type_dist.columns = ['patent_type', 'count']
        return type_dist

    def _save_analysis_results(self, output_base_path: str, innovation_trend_df: pd.DataFrame, 
                              top_assignees_df: pd.DataFrame, top_inventors_df: pd.DataFrame, 
                              patent_type_df: pd.DataFrame, original_df: pd.DataFrame):
        """Save all analysis results in the same format as fetch_assignee_inventor"""
        
        # 1. JSON Summary
        analysis_results = {
            "metadata": {
                "total_patents": len(original_df),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "columns_available": list(original_df.columns)
            },
            "innovation_trend": innovation_trend_df.to_dict('list'),
            "top_assignees": top_assignees_df.to_dict('list'),
            "top_inventors": top_inventors_df.to_dict('list'),
            "patent_types": patent_type_df.to_dict('list')
        }
        
        json_path = f"{output_base_path}_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=4, ensure_ascii=False)
        print(f"📁 JSON summary: {json_path}")
        
        # 2. Individual CSV files
        innovation_trend_df.to_csv(f"{output_base_path}_innovation_trend.csv", index=False, encoding='utf-8')
        top_assignees_df.to_csv(f"{output_base_path}_top_assignees.csv", index=False, encoding='utf-8')
        top_inventors_df.to_csv(f"{output_base_path}_top_inventors.csv", index=False, encoding='utf-8')
        patent_type_df.to_csv(f"{output_base_path}_patent_types.csv", index=False, encoding='utf-8')
        
        # 3. Excel summary with multiple sheets
        excel_path = f"{output_base_path}_analysis_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                innovation_trend_df.to_excel(writer, sheet_name='Innovation_Trend', index=False)
                top_assignees_df.to_excel(writer, sheet_name='Top_Assignees', index=False)
                top_inventors_df.to_excel(writer, sheet_name='Top_Inventor', index=False)
                patent_type_df.to_excel(writer, sheet_name='Patent_Types', index=False)
                # Add summary sheet
                summary_df = pd.DataFrame({
                    'Metric': ['Total Patents', 'Years Covered', 'Unique Assignees', 'Unique Inventors'],
                    'Value': [
                        len(original_df),
                        len(innovation_trend_df),
                        len(top_assignees_df),
                        len(top_inventors_df)
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"📁 Excel summary: {excel_path}")
        except Exception as exc:
            print(f"⚠️  Excel export failed: {exc}")
        
        print(f"📁 CSV reports: {output_base_path}_*.csv")