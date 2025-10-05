import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from data_type_detector import DataTypeDetector, DataTypeProfile, DataDomain
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class LLMAnalysisAgent:
    """LLM-inspired analysis agent with domain-specific analytics strategies."""
    
    def __init__(self):
        self.detector = DataTypeDetector()
        self.analysis_strategies = self._initialize_analysis_strategies()
        self.domain_metrics = self._initialize_domain_metrics()
    
    def _initialize_analysis_strategies(self) -> Dict[str, Any]:
        """Initialize LLM-inspired analysis strategies."""
        return {
            "temporal_analysis": {
                "trend_analysis": True,
                "seasonality_detection": True,
                "growth_metrics": True,
                "forecasting": False
            },
            "descriptive_analysis": {
                "summary_statistics": True,
                "distribution_analysis": True,
                "correlation_analysis": True,
                "outlier_detection": True
            },
            "segmentation_analysis": {
                "clustering": True,
                "rfm_analysis": True,
                "behavioral_segmentation": True,
                "demographic_analysis": True
            },
            "performance_analysis": {
                "kpi_calculation": True,
                "benchmarking": True,
                "efficiency_metrics": True,
                "roi_analysis": True
            }
        }
    
    def _initialize_domain_metrics(self) -> Dict[DataDomain, List[str]]:
        """Initialize domain-specific key metrics."""
        return {
            DataDomain.SALES: [
                "total_revenue", "average_order_value", "customer_lifetime_value",
                "sales_growth_rate", "conversion_rate", "retention_rate",
                "market_share", "sales_per_rep", "seasonal_variation"
            ],
            DataDomain.MARKET: [
                "market_penetration", "brand_awareness", "customer_satisfaction",
                "market_share", "growth_rate", "competitive_position",
                "demographic_insights", "behavioral_patterns", "trend_analysis"
            ],
            DataDomain.PRODUCT: [
                "product_performance", "inventory_turnover", "profit_margin",
                "customer_rating", "market_position", "feature_importance",
                "lifecycle_stage", "innovation_index", "quality_metrics"
            ],
            DataDomain.REGULATORY: [
                "compliance_rate", "risk_score", "audit_readiness",
                "violation_frequency", "time_to_compliance", "penalty_risk",
                "jurisdiction_coverage", "certification_status", "regulatory_burden"
            ],
            DataDomain.PATENT: [
                "innovation_trends", "citation_impact", "technology_classification",
                "patent_family_size", "forward_citations", "commercial_potential",
                "competitive_landscape", "legal_status", "filing_patterns"
            ],
            DataDomain.RD: [
                "success_rate", "innovation_potential", "resource_efficiency",
                "time_to_market", "technical_feasibility", "collaboration_index",
                "knowledge_transfer", "impact_assessment", "project_performance"
            ]
        }
    
    def analyze(self, input_path: str, output_base_path: str, domain_hint: Optional[str] = None):
        """Main analysis method with domain-specific strategies."""
        print("📊 LLM Analysis Agent: Starting...")
        
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
        
        # Perform domain-specific analysis
        analysis_results = self._perform_domain_analysis(df, profile)
        
        # Perform general analysis
        general_results = self._perform_general_analysis(df, profile)
        
        # Combine results
        combined_results = {**analysis_results, **general_results}
        
        # Generate insights
        insights = self._generate_insights(df, profile, combined_results)
        
        # Save all results
        self._save_analysis_results(combined_results, insights, profile, output_base_path)
        
        print(f"✅ LLM Analysis Agent: Completed! Generated comprehensive reports")
    
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
    
    def _perform_domain_analysis(self, df: pd.DataFrame, profile: DataTypeProfile) -> Dict[str, Any]:
        """Perform domain-specific analysis."""
        print(f"📊 Performing {profile.domain.value} domain analysis...")
        
        if profile.domain == DataDomain.SALES:
            return self._analyze_sales_data(df)
        elif profile.domain == DataDomain.MARKET:
            return self._analyze_market_data(df)
        elif profile.domain == DataDomain.PRODUCT:
            return self._analyze_product_data(df)
        elif profile.domain == DataDomain.REGULATORY:
            return self._analyze_regulatory_data(df)
        elif profile.domain == DataDomain.PATENT:
            return self._analyze_patent_data(df)
        elif profile.domain == DataDomain.RD:
            return self._analyze_rd_data(df)
        else:
            return self._analyze_generic_data(df)
    
    def _analyze_sales_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sales data."""
        results = {}
        
        # Revenue analysis
        revenue_cols = [col for col in df.columns if any(word in col.lower() for word in ['revenue', 'amount', 'sales', 'price'])]
        if revenue_cols:
            revenue_col = revenue_cols[0]
            results['revenue_analysis'] = {
                'total_revenue': df[revenue_col].sum(),
                'average_order_value': df[revenue_col].mean(),
                'median_order_value': df[revenue_col].median(),
                'revenue_growth': self._calculate_growth_rate(df, revenue_col)
            }
        
        # Customer analysis
        customer_cols = [col for col in df.columns if 'customer' in col.lower()]
        if customer_cols:
            customer_col = customer_cols[0]
            results['customer_analysis'] = {
                'total_customers': df[customer_col].nunique(),
                'repeat_customers': df[customer_col].value_counts().gt(1).sum(),
                'customer_retention_rate': df[customer_col].value_counts().gt(1).sum() / df[customer_col].nunique()
            }
        
        # Temporal analysis
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols and revenue_cols:
            results['temporal_analysis'] = self._analyze_temporal_patterns(df, date_cols[0], revenue_cols[0])
        
        return results
    
    def _analyze_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market research data."""
        results = {}
        
        # Market share analysis
        share_cols = [col for col in df.columns if 'share' in col.lower()]
        if share_cols:
            share_col = share_cols[0]
            results['market_share_analysis'] = {
                'total_market_share': df[share_col].sum(),
                'average_share': df[share_col].mean(),
                'market_leader': df[share_col].idxmax(),
                'market_concentration': self._calculate_herfindahl_index(df[share_col])
            }
        
        # Brand analysis
        brand_cols = [col for col in df.columns if 'brand' in col.lower()]
        if brand_cols:
            brand_col = brand_cols[0]
            results['brand_analysis'] = {
                'brand_count': df[brand_col].nunique(),
                'top_brands': df[brand_col].value_counts().head(10).to_dict(),
                'brand_diversity': df[brand_col].nunique() / len(df)
            }
        
        # Demographic analysis
        demo_cols = [col for col in df.columns if any(word in col.lower() for word in ['age', 'gender', 'income', 'education'])]
        if demo_cols:
            results['demographic_analysis'] = {}
            for col in demo_cols:
                results['demographic_analysis'][col] = df[col].value_counts().to_dict()
        
        return results
    
    def _analyze_product_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze product data."""
        results = {}
        
        # Product performance
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        if price_cols:
            price_col = price_cols[0]
            results['product_performance'] = {
                'average_price': df[price_col].mean(),
                'price_range': [df[price_col].min(), df[price_col].max()],
                'price_distribution': df[price_col].describe().to_dict()
            }
        
        # Category analysis
        category_cols = [col for col in df.columns if 'category' in col.lower()]
        if category_cols:
            category_col = category_cols[0]
            results['category_analysis'] = {
                'category_count': df[category_col].nunique(),
                'top_categories': df[category_col].value_counts().head(10).to_dict(),
                'category_diversity': df[category_col].nunique() / len(df)
            }
        
        # Inventory analysis
        inventory_cols = [col for col in df.columns if 'inventory' in col.lower() or 'stock' in col.lower()]
        if inventory_cols:
            inventory_col = inventory_cols[0]
            results['inventory_analysis'] = {
                'total_inventory': df[inventory_col].sum(),
                'average_inventory': df[inventory_col].mean(),
                'inventory_turnover': self._calculate_inventory_turnover(df, inventory_col)
            }
        
        return results
    
    def _analyze_regulatory_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regulatory compliance data."""
        results = {}
        
        # Compliance status
        status_cols = [col for col in df.columns if 'status' in col.lower()]
        if status_cols:
            status_col = status_cols[0]
            results['compliance_analysis'] = {
                'total_records': len(df),
                'compliant_records': df[status_col].str.contains('compliant|approved', case=False, na=False).sum(),
                'compliance_rate': df[status_col].str.contains('compliant|approved', case=False, na=False).mean(),
                'status_distribution': df[status_col].value_counts().to_dict()
            }
        
        # Risk analysis
        risk_cols = [col for col in df.columns if 'risk' in col.lower() or 'score' in col.lower()]
        if risk_cols:
            risk_col = risk_cols[0]
            results['risk_analysis'] = {
                'average_risk_score': df[risk_col].mean(),
                'high_risk_count': (df[risk_col] > df[risk_col].quantile(0.8)).sum(),
                'risk_distribution': df[risk_col].describe().to_dict()
            }
        
        # Jurisdiction analysis
        jurisdiction_cols = [col for col in df.columns if 'jurisdiction' in col.lower() or 'region' in col.lower()]
        if jurisdiction_cols:
            jurisdiction_col = jurisdiction_cols[0]
            results['jurisdiction_analysis'] = {
                'jurisdiction_count': df[jurisdiction_col].nunique(),
                'top_jurisdictions': df[jurisdiction_col].value_counts().head(10).to_dict()
            }
        
        return results
    
    def _analyze_patent_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patent data."""
        results = {}
        
        # Innovation trends
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            results['innovation_trends'] = self._analyze_innovation_trends(df, date_col)
        
        # Assignee analysis
        assignee_cols = [col for col in df.columns if 'assignee' in col.lower()]
        if assignee_cols:
            assignee_col = assignee_cols[0]
            results['assignee_analysis'] = {
                'total_assignees': df[assignee_col].nunique(),
                'top_assignees': df[assignee_col].value_counts().head(20).to_dict(),
                'assignee_concentration': self._calculate_herfindahl_index(df[assignee_col])
            }
        
        # Inventor analysis
        inventor_cols = [col for col in df.columns if 'inventor' in col.lower()]
        if inventor_cols:
            inventor_col = inventor_cols[0]
            results['inventor_analysis'] = {
                'total_inventors': df[inventor_col].nunique(),
                'top_inventors': df[inventor_col].value_counts().head(20).to_dict(),
                'inventor_productivity': df[inventor_col].value_counts().describe().to_dict()
            }
        
        # Patent type analysis
        type_cols = [col for col in df.columns if 'type' in col.lower()]
        if type_cols:
            type_col = type_cols[0]
            results['patent_type_analysis'] = {
                'type_distribution': df[type_col].value_counts().to_dict(),
                'type_diversity': df[type_col].nunique() / len(df)
            }
        
        return results
    
    def _analyze_rd_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze R&D data."""
        results = {}
        
        # Project performance
        success_cols = [col for col in df.columns if 'success' in col.lower() or 'outcome' in col.lower()]
        if success_cols:
            success_col = success_cols[0]
            results['project_performance'] = {
                'total_projects': len(df),
                'successful_projects': df[success_col].str.contains('success|completed|achieved', case=False, na=False).sum(),
                'success_rate': df[success_col].str.contains('success|completed|achieved', case=False, na=False).mean()
            }
        
        # Budget analysis
        budget_cols = [col for col in df.columns if 'budget' in col.lower() or 'cost' in col.lower()]
        if budget_cols:
            budget_col = budget_cols[0]
            results['budget_analysis'] = {
                'total_budget': df[budget_col].sum(),
                'average_budget': df[budget_col].mean(),
                'budget_distribution': df[budget_col].describe().to_dict()
            }
        
        # Technology analysis
        tech_cols = [col for col in df.columns if 'technology' in col.lower() or 'tech' in col.lower()]
        if tech_cols:
            tech_col = tech_cols[0]
            results['technology_analysis'] = {
                'technology_count': df[tech_col].nunique(),
                'top_technologies': df[tech_col].value_counts().head(10).to_dict(),
                'technology_diversity': df[tech_col].nunique() / len(df)
            }
        
        return results
    
    def _analyze_generic_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze generic data when domain is unknown."""
        results = {}
        
        # Basic statistics
        results['basic_statistics'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results['numeric_analysis'] = {}
            for col in numeric_cols:
                results['numeric_analysis'][col] = df[col].describe().to_dict()
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            results['categorical_analysis'] = {}
            for col in categorical_cols:
                results['categorical_analysis'][col] = {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head(10).to_dict()
                }
        
        return results
    
    def _perform_general_analysis(self, df: pd.DataFrame, profile: DataTypeProfile) -> Dict[str, Any]:
        """Perform general analysis applicable to all domains."""
        print("📊 Performing general analysis...")
        results = {}
        
        # Data quality analysis
        results['data_quality'] = self._assess_data_quality(df)
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            results['correlation_analysis'] = df[numeric_cols].corr().to_dict()
        
        # Outlier analysis
        if len(numeric_cols) > 0:
            results['outlier_analysis'] = self._detect_outliers(df, numeric_cols)
        
        # Distribution analysis
        results['distribution_analysis'] = self._analyze_distributions(df, numeric_cols)
        
        return results
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        return {
            'completeness': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'uniqueness': 1 - (df.duplicated().sum() / len(df)),
            'consistency': self._calculate_consistency_score(df),
            'validity': self._calculate_validity_score(df)
        }
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
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
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
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
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def _detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        outlier_results = {}
        
        for col in numeric_cols:
            if df[col].std() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_results[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(df) * 100,
                    'outlier_values': outliers[col].tolist() if len(outliers) < 100 else outliers[col].head(100).tolist()
                }
        
        return outlier_results
    
    def _analyze_distributions(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze distributions of numeric columns."""
        distribution_results = {}
        
        for col in numeric_cols:
            if df[col].std() > 0:
                distribution_results[col] = {
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'normality_test': self._test_normality(df[col])
                }
        
        return distribution_results
    
    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """Test normality of a series."""
        from scipy import stats
        
        try:
            statistic, p_value = stats.shapiro(series.dropna().head(5000))  # Limit sample size
            return {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            return {'error': 'Could not perform normality test'}
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Analyze temporal patterns in data."""
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[date_col, value_col])
            
            # Group by year and month
            df_temp['year'] = df_temp[date_col].dt.year
            df_temp['month'] = df_temp[date_col].dt.month
            
            yearly_trend = df_temp.groupby('year')[value_col].sum().to_dict()
            monthly_pattern = df_temp.groupby('month')[value_col].mean().to_dict()
            
            return {
                'yearly_trend': yearly_trend,
                'monthly_pattern': monthly_pattern,
                'seasonality_detected': len(set(monthly_pattern.values())) > 1
            }
        except Exception as e:
            return {'error': f'Could not analyze temporal patterns: {str(e)}'}
    
    def _analyze_innovation_trends(self, df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """Analyze innovation trends in patent data."""
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[date_col])
            
            # Group by year
            df_temp['year'] = df_temp[date_col].dt.year
            yearly_counts = df_temp.groupby('year').size().to_dict()
            
            # Calculate growth rate
            years = sorted(yearly_counts.keys())
            growth_rates = []
            for i in range(1, len(years)):
                if yearly_counts[years[i-1]] > 0:
                    growth_rate = (yearly_counts[years[i]] - yearly_counts[years[i-1]]) / yearly_counts[years[i-1]] * 100
                    growth_rates.append(growth_rate)
            
            return {
                'yearly_counts': yearly_counts,
                'average_growth_rate': np.mean(growth_rates) if growth_rates else 0,
                'peak_year': max(yearly_counts, key=yearly_counts.get) if yearly_counts else None,
                'total_patents': sum(yearly_counts.values())
            }
        except Exception as e:
            return {'error': f'Could not analyze innovation trends: {str(e)}'}
    
    def _calculate_growth_rate(self, df: pd.DataFrame, value_col: str) -> float:
        """Calculate growth rate for a value column."""
        try:
            if len(df) < 2:
                return 0
            first_value = df[value_col].iloc[0]
            last_value = df[value_col].iloc[-1]
            if first_value == 0:
                return 0
            return ((last_value - first_value) / first_value) * 100
        except:
            return 0
    
    def _calculate_herfindahl_index(self, series: pd.Series) -> float:
        """Calculate Herfindahl-Hirschman Index for market concentration."""
        try:
            proportions = series.value_counts(normalize=True)
            return (proportions ** 2).sum()
        except:
            return 0
    
    def _calculate_inventory_turnover(self, df: pd.DataFrame, inventory_col: str) -> float:
        """Calculate inventory turnover ratio."""
        try:
            # This is a simplified calculation - in practice, you'd need cost of goods sold
            return df[inventory_col].sum() / df[inventory_col].mean() if df[inventory_col].mean() > 0 else 0
        except:
            return 0
    
    def _generate_insights(self, df: pd.DataFrame, profile: DataTypeProfile, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from analysis results."""
        insights = {
            'domain': profile.domain.value,
            'confidence': profile.confidence,
            'key_findings': [],
            'recommendations': [],
            'data_quality_insights': [],
            'performance_insights': []
        }
        
        # Generate domain-specific insights
        if profile.domain == DataDomain.SALES:
            insights.update(self._generate_sales_insights(results))
        elif profile.domain == DataDomain.MARKET:
            insights.update(self._generate_market_insights(results))
        elif profile.domain == DataDomain.PRODUCT:
            insights.update(self._generate_product_insights(results))
        elif profile.domain == DataDomain.REGULATORY:
            insights.update(self._generate_regulatory_insights(results))
        elif profile.domain == DataDomain.PATENT:
            insights.update(self._generate_patent_insights(results))
        elif profile.domain == DataDomain.RD:
            insights.update(self._generate_rd_insights(results))
        
        return insights
    
    def _generate_sales_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sales-specific insights."""
        insights = {'key_findings': [], 'recommendations': []}
        
        if 'revenue_analysis' in results:
            revenue = results['revenue_analysis']
            insights['key_findings'].append(f"Total revenue: ${revenue['total_revenue']:,.2f}")
            insights['key_findings'].append(f"Average order value: ${revenue['average_order_value']:,.2f}")
            
            if revenue['revenue_growth'] > 0:
                insights['recommendations'].append("Revenue is growing - consider scaling successful strategies")
            else:
                insights['recommendations'].append("Revenue is declining - investigate and address issues")
        
        if 'customer_analysis' in results:
            customer = results['customer_analysis']
            insights['key_findings'].append(f"Customer retention rate: {customer['customer_retention_rate']:.1%}")
            
            if customer['customer_retention_rate'] < 0.5:
                insights['recommendations'].append("Low customer retention - focus on customer satisfaction and loyalty programs")
        
        return insights
    
    def _generate_market_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market-specific insights."""
        insights = {'key_findings': [], 'recommendations': []}
        
        if 'market_share_analysis' in results:
            market = results['market_share_analysis']
            insights['key_findings'].append(f"Market concentration (HHI): {market['market_concentration']:.3f}")
            
            if market['market_concentration'] > 0.25:
                insights['recommendations'].append("High market concentration - consider competitive strategies")
        
        return insights
    
    def _generate_product_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate product-specific insights."""
        insights = {'key_findings': [], 'recommendations': []}
        
        if 'product_performance' in results:
            product = results['product_performance']
            insights['key_findings'].append(f"Average price: ${product['average_price']:,.2f}")
            insights['recommendations'].append("Monitor price elasticity and adjust pricing strategy accordingly")
        
        return insights
    
    def _generate_regulatory_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate regulatory-specific insights."""
        insights = {'key_findings': [], 'recommendations': []}
        
        if 'compliance_analysis' in results:
            compliance = results['compliance_analysis']
            insights['key_findings'].append(f"Compliance rate: {compliance['compliance_rate']:.1%}")
            
            if compliance['compliance_rate'] < 0.9:
                insights['recommendations'].append("Low compliance rate - implement additional training and monitoring")
        
        return insights
    
    def _generate_patent_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate patent-specific insights."""
        insights = {'key_findings': [], 'recommendations': []}
        
        if 'innovation_trends' in results:
            trends = results['innovation_trends']
            if 'average_growth_rate' in trends:
                insights['key_findings'].append(f"Average innovation growth rate: {trends['average_growth_rate']:.1f}%")
                
                if trends['average_growth_rate'] > 10:
                    insights['recommendations'].append("Strong innovation growth - consider expanding R&D investment")
        
        return insights
    
    def _generate_rd_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate R&D-specific insights."""
        insights = {'key_findings': [], 'recommendations': []}
        
        if 'project_performance' in results:
            performance = results['project_performance']
            insights['key_findings'].append(f"Project success rate: {performance['success_rate']:.1%}")
            
            if performance['success_rate'] < 0.5:
                insights['recommendations'].append("Low project success rate - review project selection and management processes")
        
        return insights
    
    def _save_analysis_results(self, results: Dict[str, Any], insights: Dict[str, Any], 
                             profile: DataTypeProfile, output_base_path: str):
        """Save all analysis results."""
        
        # Create comprehensive results
        comprehensive_results = {
            "metadata": {
                "domain": profile.domain.value,
                "confidence": profile.confidence,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_records": results.get('basic_statistics', {}).get('total_records', 0)
            },
            "analysis_results": results,
            "insights": insights
        }
        
        # Save JSON summary
        json_path = f"{output_base_path}_llm_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=4, ensure_ascii=False, default=str)
        print(f"📁 JSON summary: {json_path}")
        
        # Save individual CSV files for key metrics
        self._save_metric_csvs(results, output_base_path)
        
        # Save Excel summary
        self._save_excel_summary(results, insights, output_base_path)
        
        print(f"📁 Analysis reports: {output_base_path}_*.csv, {output_base_path}_*.xlsx")
    
    def _save_metric_csvs(self, results: Dict[str, Any], output_base_path: str):
        """Save key metrics as CSV files."""
        # Save data quality metrics
        if 'data_quality' in results:
            quality_df = pd.DataFrame([results['data_quality']])
            quality_df.to_csv(f"{output_base_path}_data_quality.csv", index=False)
        
        # Save correlation matrix
        if 'correlation_analysis' in results:
            corr_df = pd.DataFrame(results['correlation_analysis'])
            corr_df.to_csv(f"{output_base_path}_correlations.csv")
        
        # Save outlier analysis
        if 'outlier_analysis' in results:
            outlier_data = []
            for col, data in results['outlier_analysis'].items():
                outlier_data.append({
                    'column': col,
                    'outlier_count': data['outlier_count'],
                    'outlier_percentage': data['outlier_percentage']
                })
            outlier_df = pd.DataFrame(outlier_data)
            outlier_df.to_csv(f"{output_base_path}_outliers.csv", index=False)
    
    def _save_excel_summary(self, results: Dict[str, Any], insights: Dict[str, Any], output_base_path: str):
        """Save comprehensive Excel summary."""
        excel_path = f"{output_base_path}_llm_analysis_summary.xlsx"
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for category, data in results.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            summary_data.append({
                                'Category': category,
                                'Metric': key,
                                'Value': str(value)[:100]  # Truncate long values
                            })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Insights sheet
                insights_data = []
                for category, items in insights.items():
                    if isinstance(items, list):
                        for item in items:
                            insights_data.append({
                                'Category': category,
                                'Insight': item
                            })
                
                insights_df = pd.DataFrame(insights_data)
                insights_df.to_excel(writer, sheet_name='Insights', index=False)
                
                # Data quality sheet
                if 'data_quality' in results:
                    quality_df = pd.DataFrame([results['data_quality']])
                    quality_df.to_excel(writer, sheet_name='Data_Quality', index=False)
                
            print(f"📁 Excel summary: {excel_path}")
        except Exception as exc:
            print(f"⚠️ Excel export failed: {exc}")
