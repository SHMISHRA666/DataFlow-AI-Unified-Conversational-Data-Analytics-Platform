import pandas as pd
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from data_type_detector import DataTypeDetector, DataTypeProfile, DataDomain
import json
import os

class LLMCleaningAgent:
    """LLM-inspired data cleaning agent with domain-specific strategies."""
    
    def __init__(self):
        self.detector = DataTypeDetector()
        self.cleaning_strategies = self._initialize_cleaning_strategies()
        self.quality_metrics = {}
    
    def _initialize_cleaning_strategies(self) -> Dict[str, Any]:
        """Initialize LLM-inspired cleaning strategies for different domains."""
        return {
            "text_cleaning": {
                "remove_extra_whitespace": True,
                "standardize_case": True,
                "remove_special_characters": False,
                "normalize_unicode": True,
                "remove_html_tags": True,
                "standardize_quotes": True
            },
            "numeric_cleaning": {
                "handle_outliers": True,
                "impute_missing": True,
                "standardize_formats": True,
                "validate_ranges": True,
                "normalize_units": True
            },
            "date_cleaning": {
                "standardize_formats": True,
                "validate_dates": True,
                "handle_timezones": True,
                "extract_components": True,
                "fill_missing_dates": False
            },
            "categorical_cleaning": {
                "standardize_values": True,
                "handle_abbreviations": True,
                "merge_similar": True,
                "validate_categories": True,
                "create_unknown_category": True
            }
        }
    
    def process(self, input_path: str, output_path: str, domain_hint: Optional[str] = None):
        """Main processing method with LLM-based cleaning strategies."""
        print("🤖 LLM Cleaning Agent: Starting...")
        
        # Load data
        df = self._load_data(input_path)
        if df.empty:
            print("❌ No data loaded")
            return
        
        print(f"📊 Loaded {len(df)} records from {input_path}")
        print(f"Columns found: {list(df.columns)}")
        
        # Detect data type and domain
        if domain_hint:
            try:
                domain = DataDomain(domain_hint.lower())
                profile = self._create_profile_for_domain(domain, df)
            except ValueError:
                print(f"⚠️ Invalid domain hint: {domain_hint}, detecting automatically...")
                profile = self.detector.detect_data_type(df)
        else:
            profile = self.detector.detect_data_type(df)
        
        # Assess data quality
        quality_report = self._assess_data_quality(df, profile)
        print(f"📈 Data Quality Score: {quality_report['overall_score']:.2f}")
        
        # Apply domain-specific cleaning
        cleaned_df = self._apply_domain_cleaning(df, profile)
        
        # Apply general cleaning strategies
        cleaned_df = self._apply_general_cleaning(cleaned_df, profile)
        
        # Final quality assessment
        final_quality = self._assess_data_quality(cleaned_df, profile)
        print(f"📈 Final Quality Score: {final_quality['overall_score']:.2f}")
        
        # Export cleaned data
        self._export_files(cleaned_df, output_path)
        
        # Save quality report
        self._save_quality_report(quality_report, final_quality, output_path)
        
        print(f"✅ LLM Cleaning Agent: Completed! Processed {len(cleaned_df)} records")
    
    def _create_profile_for_domain(self, domain: DataDomain, df: pd.DataFrame) -> DataTypeProfile:
        """Create a profile for a specific domain."""
        column_analysis = self.detector._analyze_columns(df)
        return DataTypeProfile(
            domain=domain,
            confidence=1.0,
            key_columns=column_analysis["key_columns"],
            expected_patterns=self.detector.domain_patterns[domain],
            cleaning_rules=self.detector._get_cleaning_rules(domain),
            transformation_rules=self.detector._get_transformation_rules(domain)
        )
    
    def _load_data(self, input_path: str) -> pd.DataFrame:
        """Load data from various formats."""
        file_ext = os.path.splitext(input_path)[1].lower()
        try:
            if file_ext == '.csv':
                return pd.read_csv(input_path, encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(input_path, engine='openpyxl')
            elif file_ext == '.json':
                return pd.read_json(input_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return pd.DataFrame()
    
    def _assess_data_quality(self, df: pd.DataFrame, profile: DataTypeProfile) -> Dict[str, Any]:
        """Assess data quality using multiple metrics."""
        quality_metrics = {
            "completeness": self._calculate_completeness(df),
            "consistency": self._calculate_consistency(df, profile),
            "validity": self._calculate_validity(df, profile),
            "uniqueness": self._calculate_uniqueness(df, profile),
            "accuracy": self._calculate_accuracy(df, profile)
        }
        
        # Calculate overall score
        overall_score = np.mean(list(quality_metrics.values()))
        quality_metrics["overall_score"] = overall_score
        
        return quality_metrics
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        return completeness
    
    def _calculate_consistency(self, df: pd.DataFrame, profile: DataTypeProfile) -> float:
        """Calculate data consistency score."""
        consistency_scores = []
        
        # Check date format consistency
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    consistency_scores.append(1.0)
                except:
                    consistency_scores.append(0.0)
        
        # Check numeric format consistency
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].dtype in ['int64', 'float64']:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.5)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_validity(self, df: pd.DataFrame, profile: DataTypeProfile) -> float:
        """Calculate data validity score."""
        validity_scores = []
        
        # Check for valid email addresses
        email_columns = [col for col in df.columns if 'email' in col.lower()]
        for col in email_columns:
            if col in df.columns:
                email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                valid_emails = df[col].str.match(email_pattern, na=False).sum()
                total_emails = df[col].notna().sum()
                validity_scores.append(valid_emails / total_emails if total_emails > 0 else 1.0)
        
        # Check for valid numeric ranges
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].notna().sum() > 0:
                # Check for reasonable numeric ranges
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                valid_values = ((df[col] >= lower_bound) & (df[col] <= upper_bound)).sum()
                total_values = df[col].notna().sum()
                validity_scores.append(valid_values / total_values if total_values > 0 else 1.0)
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def _calculate_uniqueness(self, df: pd.DataFrame, profile: DataTypeProfile) -> float:
        """Calculate data uniqueness score."""
        if len(df) == 0:
            return 0.0
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        uniqueness = 1 - (duplicate_rows / len(df))
        
        # Check for ID column uniqueness
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for col in id_columns:
            if col in df.columns:
                unique_ids = df[col].nunique()
                total_ids = df[col].notna().sum()
                if total_ids > 0:
                    id_uniqueness = unique_ids / total_ids
                    uniqueness = min(uniqueness, id_uniqueness)
        
        return uniqueness
    
    def _calculate_accuracy(self, df: pd.DataFrame, profile: DataTypeProfile) -> float:
        """Calculate data accuracy score based on domain-specific rules."""
        accuracy_scores = []
        
        # Domain-specific accuracy checks
        if profile.domain == DataDomain.SALES:
            # Check for valid sales amounts
            amount_columns = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'price', 'revenue', 'sales'])]
            for col in amount_columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    positive_values = (df[col] >= 0).sum()
                    total_values = df[col].notna().sum()
                    accuracy_scores.append(positive_values / total_values if total_values > 0 else 1.0)
        
        elif profile.domain == DataDomain.MARKET:
            # Check for valid percentages
            percentage_columns = [col for col in df.columns if any(word in col.lower() for word in ['share', 'rate', 'percentage', 'percent'])]
            for col in percentage_columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    valid_percentages = ((df[col] >= 0) & (df[col] <= 100)).sum()
                    total_values = df[col].notna().sum()
                    accuracy_scores.append(valid_percentages / total_values if total_values > 0 else 1.0)
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _apply_domain_cleaning(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Apply domain-specific cleaning rules."""
        print(f"🧹 Applying {profile.domain.value} domain cleaning...")
        cleaned_df = df.copy()
        
        # Apply domain-specific cleaning rules
        if profile.domain == DataDomain.SALES:
            cleaned_df = self._clean_sales_data(cleaned_df)
        elif profile.domain == DataDomain.MARKET:
            cleaned_df = self._clean_market_data(cleaned_df)
        elif profile.domain == DataDomain.PRODUCT:
            cleaned_df = self._clean_product_data(cleaned_df)
        elif profile.domain == DataDomain.REGULATORY:
            cleaned_df = self._clean_regulatory_data(cleaned_df)
        elif profile.domain == DataDomain.PATENT:
            cleaned_df = self._clean_patent_data(cleaned_df)
        elif profile.domain == DataDomain.RD:
            cleaned_df = self._clean_rd_data(cleaned_df)
        
        return cleaned_df
    
    def _clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sales-specific data."""
        # Clean amount columns
        amount_columns = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'price', 'revenue', 'sales'])]
        for col in amount_columns:
            if col in df.columns:
                # Remove currency symbols and convert to numeric
                df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remove negative values (assuming sales should be positive)
                df[col] = df[col].abs()
        
        # Clean customer data
        customer_columns = [col for col in df.columns if 'customer' in col.lower()]
        for col in customer_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
    
    def _clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean market research data."""
        # Clean percentage columns
        percentage_columns = [col for col in df.columns if any(word in col.lower() for word in ['share', 'rate', 'percentage', 'percent'])]
        for col in percentage_columns:
            if col in df.columns:
                # Ensure percentages are between 0 and 100
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].clip(0, 100)
        
        # Clean rating columns
        rating_columns = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]
        for col in rating_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _clean_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean product data."""
        # Clean SKU columns
        sku_columns = [col for col in df.columns if 'sku' in col.lower()]
        for col in sku_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Clean category columns
        category_columns = [col for col in df.columns if 'category' in col.lower()]
        for col in category_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
    
    def _clean_regulatory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean regulatory compliance data."""
        # Clean status columns
        status_columns = [col for col in df.columns if 'status' in col.lower()]
        for col in status_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Clean date columns
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _clean_patent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean patent data."""
        # Clean patent numbers
        patent_columns = [col for col in df.columns if 'patent' in col.lower() and 'number' in col.lower()]
        for col in patent_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Clean assignee names
        assignee_columns = [col for col in df.columns if 'assignee' in col.lower()]
        for col in assignee_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
    
    def _clean_rd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean R&D data."""
        # Clean project codes
        project_columns = [col for col in df.columns if 'project' in col.lower() and 'code' in col.lower()]
        for col in project_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Clean phase names
        phase_columns = [col for col in df.columns if 'phase' in col.lower()]
        for col in phase_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
    
    def _apply_general_cleaning(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Apply general cleaning strategies."""
        print("🧹 Applying general cleaning strategies...")
        cleaned_df = df.copy()
        
        # Text cleaning
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = self._clean_text_column(cleaned_df[col])
        
        # Remove duplicates
        if profile.cleaning_rules.get("remove_duplicates", True):
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_count = initial_count - len(cleaned_df)
            if removed_count > 0:
                print(f"🔄 Removed {removed_count} duplicate records")
        
        # Handle missing values
        if profile.cleaning_rules.get("handle_missing_values", True):
            cleaned_df = self._handle_missing_values(cleaned_df, profile)
        
        return cleaned_df
    
    def _clean_text_column(self, series: pd.Series) -> pd.Series:
        """Clean text column using LLM-inspired strategies."""
        if series.dtype != 'object':
            return series
        
        # Remove extra whitespace
        series = series.astype(str).str.strip()
        series = series.str.replace(r'\s+', ' ', regex=True)
        
        # Remove HTML tags
        series = series.str.replace(r'<[^>]+>', '', regex=True)
        
        # Standardize quotes
        series = series.str.replace(r'[“”]', '"', regex=True)
        series = series.str.replace(r"[‘’]", "'", regex=True)
        
        # Handle NaN values
        series = series.replace('nan', np.nan)
        series = series.replace('', np.nan)
        
        return series
    
    def _handle_missing_values(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Handle missing values based on domain-specific rules."""
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    # For numeric columns, use median imputation
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == 'object':
                    # For categorical columns, use mode or 'Unknown'
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)
        
        return df
    
    def _export_files(self, df: pd.DataFrame, output_prefix: str):
        """Export files in multiple formats."""
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
    
    def _save_quality_report(self, initial_quality: Dict, final_quality: Dict, output_prefix: str):
        """Save data quality report."""
        quality_report = {
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "improvement": {
                "overall_score": final_quality["overall_score"] - initial_quality["overall_score"],
                "completeness": final_quality["completeness"] - initial_quality["completeness"],
                "consistency": final_quality["consistency"] - initial_quality["consistency"],
                "validity": final_quality["validity"] - initial_quality["validity"],
                "uniqueness": final_quality["uniqueness"] - initial_quality["uniqueness"],
                "accuracy": final_quality["accuracy"] - initial_quality["accuracy"]
            }
        }
        
        report_path = f"{output_prefix}_quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=4, ensure_ascii=False)
        print(f"📁 Quality report: {report_path}")
