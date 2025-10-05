import pandas as pd
import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class DataDomain(Enum):
    SALES = "sales"
    MARKET = "market"
    PRODUCT = "product"
    REGULATORY = "regulatory"
    PATENT = "patent"
    RD = "rd"
    FINANCIAL = "financial"
    CUSTOMER = "customer"
    OPERATIONS = "operations"
    UNKNOWN = "unknown"

@dataclass
class DataTypeProfile:
    domain: DataDomain
    confidence: float
    key_columns: List[str]
    expected_patterns: Dict[str, str]
    cleaning_rules: Dict[str, Any]
    transformation_rules: Dict[str, Any]

class DataTypeDetector:
    """LLM-inspired data type detection and classification system."""
    
    def __init__(self):
        self.domain_patterns = self._initialize_domain_patterns()
        self.column_patterns = self._initialize_column_patterns()
        self.data_quality_rules = self._initialize_quality_rules()
    
    def _initialize_domain_patterns(self) -> Dict[DataDomain, Dict[str, Any]]:
        """Initialize domain-specific patterns and rules."""
        return {
            DataDomain.SALES: {
                "keywords": ["revenue", "sales", "transaction", "order", "customer", "price", "amount", "quantity"],
                "patterns": [r"sales|revenue|transaction|order", r"customer|client", r"price|amount|cost"],
                "date_columns": ["sale_date", "transaction_date", "order_date", "created_at"],
                "numeric_columns": ["amount", "price", "quantity", "revenue", "sales_value"],
                "categorical_columns": ["product", "category", "region", "sales_rep", "customer_type"]
            },
            DataDomain.MARKET: {
                "keywords": ["market", "share", "competitor", "brand", "segment", "demographic", "survey"],
                "patterns": [r"market|share|competitor", r"brand|segment", r"demographic|survey"],
                "date_columns": ["survey_date", "market_date", "period", "quarter", "year"],
                "numeric_columns": ["market_share", "growth_rate", "penetration", "index", "score"],
                "categorical_columns": ["segment", "brand", "competitor", "region", "demographic"]
            },
            DataDomain.PRODUCT: {
                "keywords": ["product", "sku", "inventory", "category", "specification", "feature", "variant"],
                "patterns": [r"product|sku|inventory", r"category|specification", r"feature|variant"],
                "date_columns": ["launch_date", "created_date", "updated_date", "discontinued_date"],
                "numeric_columns": ["price", "cost", "weight", "dimension", "rating", "inventory_count"],
                "categorical_columns": ["category", "brand", "status", "type", "variant"]
            },
            DataDomain.REGULATORY: {
                "keywords": ["compliance", "regulation", "audit", "certification", "standard", "requirement"],
                "patterns": [r"compliance|regulation", r"audit|certification", r"standard|requirement"],
                "date_columns": ["compliance_date", "audit_date", "expiry_date", "effective_date"],
                "numeric_columns": ["score", "rating", "penalty", "fine", "compliance_rate"],
                "categorical_columns": ["status", "type", "category", "jurisdiction", "standard"]
            },
            DataDomain.PATENT: {
                "keywords": ["patent", "invention", "assignee", "inventor", "application", "filing", "claim"],
                "patterns": [r"patent|invention", r"assignee|inventor", r"application|filing"],
                "date_columns": ["filing_date", "grant_date", "expiry_date", "priority_date"],
                "numeric_columns": ["claim_count", "citation_count", "forward_citations"],
                "categorical_columns": ["type", "status", "assignee", "inventor", "class"]
            },
            DataDomain.RD: {
                "keywords": ["research", "development", "experiment", "trial", "innovation", "project", "study"],
                "patterns": [r"research|development", r"experiment|trial", r"innovation|project"],
                "date_columns": ["start_date", "end_date", "trial_date", "completion_date"],
                "numeric_columns": ["budget", "cost", "duration", "success_rate", "efficiency"],
                "categorical_columns": ["project_type", "status", "phase", "team", "technology"]
            },
            DataDomain.FINANCIAL: {
                "keywords": ["financial", "accounting", "budget", "expense", "profit", "loss", "asset"],
                "patterns": [r"financial|accounting", r"budget|expense", r"profit|loss|asset"],
                "date_columns": ["fiscal_year", "period", "reporting_date", "transaction_date"],
                "numeric_columns": ["amount", "balance", "value", "rate", "percentage"],
                "categorical_columns": ["account", "category", "type", "status", "department"]
            },
            DataDomain.CUSTOMER: {
                "keywords": ["customer", "client", "user", "profile", "behavior", "satisfaction", "feedback"],
                "patterns": [r"customer|client|user", r"profile|behavior", r"satisfaction|feedback"],
                "date_columns": ["registration_date", "last_activity", "purchase_date", "interaction_date"],
                "numeric_columns": ["age", "score", "rating", "lifetime_value", "frequency"],
                "categorical_columns": ["segment", "type", "status", "source", "preference"]
            }
        }
    
    def _initialize_column_patterns(self) -> Dict[str, List[str]]:
        """Initialize column name patterns for different data types."""
        return {
            "date_columns": [
                r".*date.*", r".*time.*", r".*created.*", r".*updated.*", r".*timestamp.*",
                r".*period.*", r".*year.*", r".*month.*", r".*day.*", r".*fiscal.*"
            ],
            "numeric_columns": [
                r".*amount.*", r".*price.*", r".*cost.*", r".*value.*", r".*count.*",
                r".*number.*", r".*quantity.*", r".*rate.*", r".*score.*", r".*rating.*",
                r".*percentage.*", r".*ratio.*", r".*index.*", r".*total.*", r".*sum.*"
            ],
            "categorical_columns": [
                r".*type.*", r".*category.*", r".*status.*", r".*class.*", r".*group.*",
                r".*segment.*", r".*brand.*", r".*name.*", r".*description.*", r".*label.*"
            ],
            "id_columns": [
                r".*id.*", r".*key.*", r".*code.*", r".*number.*", r".*reference.*",
                r".*identifier.*", r".*uuid.*", r".*guid.*"
            ]
        }
    
    def _initialize_quality_rules(self) -> Dict[str, Any]:
        """Initialize data quality rules for different domains."""
        return {
            "completeness": {
                "threshold": 0.95,
                "critical_columns": ["id", "date", "amount", "customer", "product"]
            },
            "consistency": {
                "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"],
                "numeric_precision": 2,
                "text_encoding": "utf-8"
            },
            "validity": {
                "email_pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "phone_pattern": r"^[\+]?[1-9][\d]{0,15}$",
                "url_pattern": r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
            }
        }
    
    def detect_data_type(self, df: pd.DataFrame) -> DataTypeProfile:
        """Detect the data type and domain of the dataset."""
        print("🔍 Detecting data type and domain...")
        
        # Analyze column names and content
        column_analysis = self._analyze_columns(df)
        content_analysis = self._analyze_content(df)
        
        # Calculate domain confidence scores
        domain_scores = self._calculate_domain_scores(column_analysis, content_analysis)
        
        # Select best domain
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        domain = best_domain[0]
        confidence = best_domain[1]
        
        # Generate data type profile
        profile = DataTypeProfile(
            domain=domain,
            confidence=confidence,
            key_columns=column_analysis["key_columns"],
            expected_patterns=self.domain_patterns[domain],
            cleaning_rules=self._get_cleaning_rules(domain),
            transformation_rules=self._get_transformation_rules(domain)
        )
        
        print(f"📊 Detected domain: {domain.value} (confidence: {confidence:.2f})")
        return profile
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column names and types."""
        analysis = {
            "columns": list(df.columns),
            "key_columns": [],
            "date_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "id_columns": []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for ID columns
            if any(re.search(pattern, col_lower) for pattern in self.column_patterns["id_columns"]):
                analysis["id_columns"].append(col)
                analysis["key_columns"].append(col)
            
            # Check for date columns
            elif any(re.search(pattern, col_lower) for pattern in self.column_patterns["date_columns"]):
                analysis["date_columns"].append(col)
            
            # Check for numeric columns
            elif any(re.search(pattern, col_lower) for pattern in self.column_patterns["numeric_columns"]):
                analysis["numeric_columns"].append(col)
            
            # Check for categorical columns
            elif any(re.search(pattern, col_lower) for pattern in self.column_patterns["categorical_columns"]):
                analysis["categorical_columns"].append(col)
        
        return analysis
    
    def _analyze_content(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze actual data content for domain detection."""
        content_analysis = {
            "text_content": "",
            "numeric_stats": {},
            "categorical_stats": {}
        }
        
        # Sample text content for keyword analysis
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            sample_text = " ".join(df[text_columns].astype(str).head(100).values.flatten())
            content_analysis["text_content"] = sample_text.lower()
        
        # Analyze numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            content_analysis["numeric_stats"][col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
            }
        
        return content_analysis
    
    def _calculate_domain_scores(self, column_analysis: Dict, content_analysis: Dict) -> Dict[DataDomain, float]:
        """Calculate confidence scores for each domain."""
        scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0.0
            
            # Score based on column names
            column_score = 0
            for col in column_analysis["columns"]:
                col_lower = col.lower()
                for keyword in patterns["keywords"]:
                    if keyword in col_lower:
                        column_score += 1
            
            # Score based on content keywords
            content_score = 0
            if content_analysis["text_content"]:
                for keyword in patterns["keywords"]:
                    content_score += content_analysis["text_content"].count(keyword)
            
            # Normalize scores
            total_columns = len(column_analysis["columns"])
            column_score = column_score / total_columns if total_columns > 0 else 0
            content_score = content_score / 100  # Normalize by sample size
            
            # Weighted combination
            scores[domain] = (column_score * 0.6) + (content_score * 0.4)
        
        return scores
    
    def _get_cleaning_rules(self, domain: DataDomain) -> Dict[str, Any]:
        """Get domain-specific cleaning rules."""
        base_rules = {
            "remove_duplicates": True,
            "handle_missing_values": True,
            "standardize_text": True,
            "validate_data_types": True,
            "remove_outliers": False
        }
        
        domain_specific = {
            DataDomain.SALES: {
                **base_rules,
                "remove_outliers": True,
                "validate_amounts": True,
                "standardize_currency": True,
                "validate_dates": True
            },
            DataDomain.MARKET: {
                **base_rules,
                "validate_percentages": True,
                "standardize_ratings": True,
                "validate_survey_data": True
            },
            DataDomain.PRODUCT: {
                **base_rules,
                "validate_skus": True,
                "standardize_categories": True,
                "validate_specifications": True
            },
            DataDomain.REGULATORY: {
                **base_rules,
                "validate_compliance_dates": True,
                "standardize_status_values": True,
                "validate_certification_codes": True
            },
            DataDomain.PATENT: {
                **base_rules,
                "validate_patent_numbers": True,
                "standardize_assignee_names": True,
                "validate_dates": True
            },
            DataDomain.RD: {
                **base_rules,
                "validate_project_codes": True,
                "standardize_phase_names": True,
                "validate_budget_data": True
            }
        }
        
        return domain_specific.get(domain, base_rules)
    
    def _get_transformation_rules(self, domain: DataDomain) -> Dict[str, Any]:
        """Get domain-specific transformation rules."""
        base_rules = {
            "extract_dates": True,
            "normalize_categories": True,
            "create_derived_features": True,
            "standardize_encodings": True
        }
        
        domain_specific = {
            DataDomain.SALES: {
                **base_rules,
                "calculate_metrics": ["total_sales", "average_order_value", "customer_lifetime_value"],
                "create_time_features": ["year", "quarter", "month", "day_of_week"],
                "normalize_geographic_data": True
            },
            DataDomain.MARKET: {
                **base_rules,
                "calculate_metrics": ["market_share", "growth_rate", "penetration_index"],
                "create_segmentation_features": True,
                "normalize_ratings": True
            },
            DataDomain.PRODUCT: {
                **base_rules,
                "calculate_metrics": ["inventory_turnover", "profit_margin", "popularity_score"],
                "create_categorization_features": True,
                "normalize_specifications": True
            },
            DataDomain.REGULATORY: {
                **base_rules,
                "calculate_metrics": ["compliance_score", "risk_level", "time_to_compliance"],
                "create_status_tracking_features": True,
                "normalize_jurisdiction_data": True
            },
            DataDomain.PATENT: {
                **base_rules,
                "calculate_metrics": ["citation_impact", "innovation_index", "patent_family_size"],
                "create_technology_classification": True,
                "normalize_assignee_data": True
            },
            DataDomain.RD: {
                **base_rules,
                "calculate_metrics": ["success_rate", "efficiency_index", "innovation_potential"],
                "create_project_tracking_features": True,
                "normalize_technology_categories": True
            }
        }
        
        return domain_specific.get(domain, base_rules)
