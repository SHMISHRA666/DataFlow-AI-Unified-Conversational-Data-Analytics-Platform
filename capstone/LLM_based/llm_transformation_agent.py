import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from data_type_detector import DataTypeDetector, DataTypeProfile, DataDomain
import json
import os
from datetime import datetime, timedelta

class LLMTransformationAgent:
    """LLM-inspired data transformation agent with domain-specific strategies."""
    
    def __init__(self):
        self.detector = DataTypeDetector()
        self.transformation_strategies = self._initialize_transformation_strategies()
        self.feature_engineering_rules = self._initialize_feature_engineering()
    
    def _initialize_transformation_strategies(self) -> Dict[str, Any]:
        """Initialize LLM-inspired transformation strategies."""
        return {
            "date_transformations": {
                "extract_components": True,
                "create_time_features": True,
                "handle_timezones": True,
                "calculate_durations": True,
                "create_periods": True
            },
            "categorical_transformations": {
                "one_hot_encoding": True,
                "label_encoding": True,
                "target_encoding": False,
                "frequency_encoding": True,
                "ordinal_encoding": True
            },
            "numeric_transformations": {
                "scaling": True,
                "normalization": True,
                "log_transformation": True,
                "binning": True,
                "polynomial_features": False
            },
            "text_transformations": {
                "tokenization": True,
                "stemming": True,
                "lemmatization": True,
                "tf_idf": False,
                "word_embeddings": False
            }
        }
    
    def _initialize_feature_engineering(self) -> Dict[DataDomain, List[str]]:
        """Initialize domain-specific feature engineering rules."""
        return {
            DataDomain.SALES: [
                "customer_lifetime_value", "average_order_value", "purchase_frequency",
                "recency_score", "monetary_value", "rfm_segments", "seasonal_patterns",
                "growth_rates", "market_share", "customer_segments"
            ],
            DataDomain.MARKET: [
                "market_penetration", "growth_rates", "competitive_index",
                "brand_awareness", "customer_satisfaction", "market_share_trends",
                "demographic_segments", "psychographic_profiles", "behavioral_patterns"
            ],
            DataDomain.PRODUCT: [
                "popularity_score", "profit_margin", "inventory_turnover",
                "category_performance", "price_elasticity", "feature_importance",
                "quality_metrics", "innovation_index", "market_position"
            ],
            DataDomain.REGULATORY: [
                "compliance_score", "risk_level", "time_to_compliance",
                "violation_frequency", "audit_readiness", "regulatory_burden",
                "jurisdiction_complexity", "certification_status", "penalty_risk"
            ],
            DataDomain.PATENT: [
                "citation_impact", "innovation_index", "technology_classification",
                "patent_family_size", "forward_citations", "backward_citations",
                "legal_status", "commercial_potential", "competitive_advantage"
            ],
            DataDomain.RD: [
                "success_probability", "innovation_potential", "resource_efficiency",
                "time_to_market", "technical_feasibility", "market_readiness",
                "collaboration_index", "knowledge_transfer", "impact_assessment"
            ]
        }
    
    def process(self, input_path: str, output_path: str, domain_hint: Optional[str] = None):
        """Main processing method with LLM-based transformation strategies."""
        print("🔧 LLM Transformation Agent: Starting...")
        
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
        
        # Apply domain-specific transformations
        transformed_df = self._apply_domain_transformations(df, profile)
        
        # Apply general transformations
        transformed_df = self._apply_general_transformations(transformed_df, profile)
        
        # Apply feature engineering
        transformed_df = self._apply_feature_engineering(transformed_df, profile)
        
        # Create derived features
        transformed_df = self._create_derived_features(transformed_df, profile)
        
        # Final data validation
        transformed_df = self._validate_transformed_data(transformed_df, profile)
        
        # Export transformed data
        self._export_files(transformed_df, output_path)
        
        # Save transformation report
        self._save_transformation_report(df, transformed_df, profile, output_path)
        
        print(f"✅ LLM Transformation Agent: Completed! Processed {len(transformed_df)} records")
    
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
    
    def _apply_domain_transformations(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Apply domain-specific transformations."""
        print(f"🔧 Applying {profile.domain.value} domain transformations...")
        transformed_df = df.copy()
        
        # Apply domain-specific transformation rules
        if profile.domain == DataDomain.SALES:
            transformed_df = self._transform_sales_data(transformed_df)
        elif profile.domain == DataDomain.MARKET:
            transformed_df = self._transform_market_data(transformed_df)
        elif profile.domain == DataDomain.PRODUCT:
            transformed_df = self._transform_product_data(transformed_df)
        elif profile.domain == DataDomain.REGULATORY:
            transformed_df = self._transform_regulatory_data(transformed_df)
        elif profile.domain == DataDomain.PATENT:
            transformed_df = self._transform_patent_data(transformed_df)
        elif profile.domain == DataDomain.RD:
            transformed_df = self._transform_rd_data(transformed_df)
        
        return transformed_df
    
    def _transform_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform sales data."""
        # Extract year, month, quarter from date columns
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                df[f'{col}_year'] = pd.to_datetime(df[col], errors='coerce').dt.year
                df[f'{col}_month'] = pd.to_datetime(df[col], errors='coerce').dt.month
                df[f'{col}_quarter'] = pd.to_datetime(df[col], errors='coerce').dt.quarter
                df[f'{col}_day_of_week'] = pd.to_datetime(df[col], errors='coerce').dt.dayofweek
        
        # Create customer segments based on purchase behavior
        if 'amount' in df.columns or 'price' in df.columns:
            amount_col = 'amount' if 'amount' in df.columns else 'price'
            df['customer_segment'] = pd.cut(df[amount_col], 
                                          bins=[0, df[amount_col].quantile(0.33), 
                                                df[amount_col].quantile(0.67), df[amount_col].max()],
                                          labels=['Low', 'Medium', 'High'])
        
        # Calculate recency if we have customer and date data
        customer_cols = [col for col in df.columns if 'customer' in col.lower()]
        if customer_cols and date_columns:
            customer_col = customer_cols[0]
            date_col = date_columns[0]
            df['days_since_last_purchase'] = (pd.Timestamp.now() - pd.to_datetime(df[date_col], errors='coerce')).dt.days
        
        return df
    
    def _transform_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform market research data."""
        # Normalize market share data
        share_columns = [col for col in df.columns if 'share' in col.lower()]
        for col in share_columns:
            if col in df.columns:
                df[f'{col}_normalized'] = df[col] / df[col].sum() * 100
        
        # Create growth rate features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if 'rate' not in col.lower() and 'share' not in col.lower():
                df[f'{col}_growth_rate'] = df[col].pct_change()
        
        # Create competitive position features
        if len(share_columns) > 1:
            df['market_leader'] = df[share_columns[0]] == df[share_columns[0]].max()
            df['market_position'] = df[share_columns[0]].rank(ascending=False)
        
        return df
    
    def _transform_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform product data."""
        # Create product categories hierarchy
        category_columns = [col for col in df.columns if 'category' in col.lower()]
        for col in category_columns:
            if col in df.columns:
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
        
        # Create price tiers
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        for col in price_columns:
            if col in df.columns:
                df[f'{col}_tier'] = pd.cut(df[col], 
                                         bins=[0, df[col].quantile(0.33), 
                                               df[col].quantile(0.67), df[col].max()],
                                         labels=['Budget', 'Mid-range', 'Premium'])
        
        # Create product lifecycle features
        if 'launch_date' in df.columns:
            df['product_age_days'] = (pd.Timestamp.now() - pd.to_datetime(df['launch_date'], errors='coerce')).dt.days
            df['product_lifecycle_stage'] = pd.cut(df['product_age_days'], 
                                                 bins=[0, 365, 1095, 3650, float('inf')],
                                                 labels=['New', 'Growth', 'Mature', 'Decline'])
        
        return df
    
    def _transform_regulatory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform regulatory compliance data."""
        # Create compliance status features
        status_columns = [col for col in df.columns if 'status' in col.lower()]
        for col in status_columns:
            if col in df.columns:
                df[f'{col}_is_compliant'] = df[col].str.contains('compliant|approved|valid', case=False, na=False)
        
        # Calculate time to compliance
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if len(date_columns) >= 2:
            df['time_to_compliance_days'] = (pd.to_datetime(df[date_columns[1]], errors='coerce') - 
                                           pd.to_datetime(df[date_columns[0]], errors='coerce')).dt.days
        
        # Create risk level features
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        for col in score_columns:
            if col in df.columns:
                df[f'{col}_risk_level'] = pd.cut(df[col], 
                                               bins=[0, 30, 70, 100],
                                               labels=['Low', 'Medium', 'High'])
        
        return df
    
    def _transform_patent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform patent data."""
        # Extract year from filing dates
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                df[f'{col}_year'] = pd.to_datetime(df[col], errors='coerce').dt.year
        
        # Create patent family features
        if 'patent_id' in df.columns:
            df['patent_family_size'] = df.groupby('patent_id')['patent_id'].transform('count')
        
        # Create technology classification features
        class_columns = [col for col in df.columns if 'class' in col.lower()]
        for col in class_columns:
            if col in df.columns:
                df[f'{col}_main_class'] = df[col].astype(str).str.split('.').str[0]
        
        # Create citation impact features
        citation_columns = [col for col in df.columns if 'citation' in col.lower()]
        for col in citation_columns:
            if col in df.columns:
                df[f'{col}_impact_score'] = df[col] / df[col].mean() if df[col].mean() > 0 else 0
        
        return df
    
    def _transform_rd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform R&D data."""
        # Create project duration features
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if len(date_columns) >= 2:
            df['project_duration_days'] = (pd.to_datetime(df[date_columns[1]], errors='coerce') - 
                                         pd.to_datetime(df[date_columns[0]], errors='coerce')).dt.days
        
        # Create success probability features
        success_columns = [col for col in df.columns if 'success' in col.lower() or 'outcome' in col.lower()]
        for col in success_columns:
            if col in df.columns:
                df[f'{col}_probability'] = df[col].astype(str).str.contains('success|completed|achieved', case=False, na=False).astype(int)
        
        # Create resource efficiency features
        budget_columns = [col for col in df.columns if 'budget' in col.lower() or 'cost' in col.lower()]
        if budget_columns and 'project_duration_days' in df.columns:
            budget_col = budget_columns[0]
            df['daily_budget'] = df[budget_col] / df['project_duration_days'].replace(0, 1)
        
        return df
    
    def _apply_general_transformations(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Apply general transformation strategies."""
        print("🔧 Applying general transformations...")
        transformed_df = df.copy()
        
        # Date transformations
        if self.transformation_strategies["date_transformations"]["extract_components"]:
            transformed_df = self._extract_date_components(transformed_df)
        
        # Categorical transformations
        if self.transformation_strategies["categorical_transformations"]["label_encoding"]:
            transformed_df = self._apply_label_encoding(transformed_df)
        
        # Numeric transformations
        if self.transformation_strategies["numeric_transformations"]["scaling"]:
            transformed_df = self._apply_scaling(transformed_df)
        
        return transformed_df
    
    def _extract_date_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract components from date columns."""
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                try:
                    date_series = pd.to_datetime(df[col], errors='coerce')
                    df[f'{col}_year'] = date_series.dt.year
                    df[f'{col}_month'] = date_series.dt.month
                    df[f'{col}_day'] = date_series.dt.day
                    df[f'{col}_dayofweek'] = date_series.dt.dayofweek
                    df[f'{col}_quarter'] = date_series.dt.quarter
                    df[f'{col}_is_weekend'] = date_series.dt.dayofweek.isin([5, 6])
                except:
                    continue
        return df
    
    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns."""
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].nunique() < 50:  # Only encode if reasonable number of categories
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
        return df
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to numeric columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].std() > 0:  # Only scale if there's variation
                df[f'{col}_scaled'] = (df[col] - df[col].mean()) / df[col].std()
        return df
    
    def _apply_feature_engineering(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Apply domain-specific feature engineering."""
        print("🔧 Applying feature engineering...")
        engineered_df = df.copy()
        
        # Get domain-specific features
        domain_features = self.feature_engineering_rules.get(profile.domain, [])
        
        for feature in domain_features:
            if feature == "customer_lifetime_value" and profile.domain == DataDomain.SALES:
                engineered_df = self._create_customer_lifetime_value(engineered_df)
            elif feature == "market_penetration" and profile.domain == DataDomain.MARKET:
                engineered_df = self._create_market_penetration(engineered_df)
            elif feature == "compliance_score" and profile.domain == DataDomain.REGULATORY:
                engineered_df = self._create_compliance_score(engineered_df)
            # Add more feature engineering methods as needed
        
        return engineered_df
    
    def _create_customer_lifetime_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer lifetime value feature."""
        if 'amount' in df.columns and 'customer' in df.columns:
            customer_col = [col for col in df.columns if 'customer' in col.lower()][0]
            amount_col = 'amount' if 'amount' in df.columns else [col for col in df.columns if 'price' in col.lower()][0]
            df['customer_lifetime_value'] = df.groupby(customer_col)[amount_col].transform('sum')
        return df
    
    def _create_market_penetration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market penetration feature."""
        share_columns = [col for col in df.columns if 'share' in col.lower()]
        if share_columns:
            df['market_penetration'] = df[share_columns[0]] / df[share_columns[0]].sum() * 100
        return df
    
    def _create_compliance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create compliance score feature."""
        status_columns = [col for col in df.columns if 'status' in col.lower()]
        if status_columns:
            df['compliance_score'] = df[status_columns[0]].str.contains('compliant|approved', case=False, na=False).astype(int) * 100
        return df
    
    def _create_derived_features(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Create derived features based on domain rules."""
        print("🔧 Creating derived features...")
        derived_df = df.copy()
        
        # Create interaction features
        numeric_columns = derived_df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            col1, col2 = numeric_columns[0], numeric_columns[1]
            derived_df[f'{col1}_x_{col2}'] = derived_df[col1] * derived_df[col2]
            derived_df[f'{col1}_div_{col2}'] = derived_df[col1] / derived_df[col2].replace(0, 1)
        
        # Create ratio features
        for col in numeric_columns:
            if 'amount' in col.lower() or 'price' in col.lower():
                derived_df[f'{col}_ratio'] = derived_df[col] / derived_df[col].mean()
        
        return derived_df
    
    def _validate_transformed_data(self, df: pd.DataFrame, profile: DataTypeProfile) -> pd.DataFrame:
        """Validate transformed data."""
        print("🔧 Validating transformed data...")
        validated_df = df.copy()
        
        # Check for infinite values
        numeric_columns = validated_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            validated_df[col] = validated_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Check for extreme outliers
        for col in numeric_columns:
            if validated_df[col].std() > 0:
                z_scores = np.abs((validated_df[col] - validated_df[col].mean()) / validated_df[col].std())
                validated_df = validated_df[z_scores < 5]  # Remove extreme outliers
        
        return validated_df
    
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
    
    def _save_transformation_report(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame, 
                                  profile: DataTypeProfile, output_prefix: str):
        """Save transformation report."""
        transformation_report = {
            "domain": profile.domain.value,
            "confidence": profile.confidence,
            "original_shape": original_df.shape,
            "transformed_shape": transformed_df.shape,
            "new_features": list(set(transformed_df.columns) - set(original_df.columns)),
            "transformation_summary": {
                "date_features_created": len([col for col in transformed_df.columns if 'date' in col.lower() and col not in original_df.columns]),
                "categorical_features_encoded": len([col for col in transformed_df.columns if 'encoded' in col.lower()]),
                "scaled_features": len([col for col in transformed_df.columns if 'scaled' in col.lower()]),
                "derived_features": len([col for col in transformed_df.columns if col not in original_df.columns])
            }
        }
        
        report_path = f"{output_prefix}_transformation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(transformation_report, f, indent=4, ensure_ascii=False)
        print(f"📁 Transformation report: {report_path}")
