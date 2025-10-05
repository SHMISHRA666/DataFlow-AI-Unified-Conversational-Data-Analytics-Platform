import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

class QualityDimension(Enum):
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"

@dataclass
class QualityMetric:
    dimension: QualityDimension
    score: float
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class QualityReport:
    overall_score: float
    metrics: List[QualityMetric]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

class DataQualityFramework:
    """Comprehensive data quality assessment framework with LLM-inspired strategies."""
    
    def __init__(self):
        self.quality_rules = self._initialize_quality_rules()
        self.validation_patterns = self._initialize_validation_patterns()
        self.domain_quality_standards = self._initialize_domain_standards()
    
    def _initialize_quality_rules(self) -> Dict[QualityDimension, Dict[str, Any]]:
        """Initialize quality assessment rules for each dimension."""
        return {
            QualityDimension.COMPLETENESS: {
                "thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "acceptable": 0.70,
                    "poor": 0.50
                },
                "critical_columns": ["id", "date", "amount", "customer", "product"],
                "weighted_scoring": True
            },
            QualityDimension.CONSISTENCY: {
                "thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "acceptable": 0.70,
                    "poor": 0.50
                },
                "date_format_consistency": True,
                "encoding_consistency": True,
                "case_consistency": True
            },
            QualityDimension.VALIDITY: {
                "thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "acceptable": 0.70,
                    "poor": 0.50
                },
                "email_validation": True,
                "phone_validation": True,
                "url_validation": True,
                "numeric_range_validation": True
            },
            QualityDimension.UNIQUENESS: {
                "thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "acceptable": 0.70,
                    "poor": 0.50
                },
                "duplicate_row_threshold": 0.05,
                "id_uniqueness_required": True
            },
            QualityDimension.ACCURACY: {
                "thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "acceptable": 0.70,
                    "poor": 0.50
                },
                "outlier_detection": True,
                "business_rule_validation": True,
                "cross_field_validation": True
            },
            QualityDimension.TIMELINESS: {
                "thresholds": {
                    "excellent": 0.95,
                    "good": 0.85,
                    "acceptable": 0.70,
                    "poor": 0.50
                },
                "data_freshness_days": 30,
                "update_frequency_validation": True
            }
        }
    
    def _initialize_validation_patterns(self) -> Dict[str, str]:
        """Initialize validation patterns for different data types."""
        return {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^[\+]?[1-9][\d]{0,15}$",
            "url": r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$",
            "date_iso": r"^\d{4}-\d{2}-\d{2}$",
            "date_us": r"^\d{2}/\d{2}/\d{4}$",
            "date_eu": r"^\d{2}/\d{2}/\d{4}$",
            "postal_code_us": r"^\d{5}(-\d{4})?$",
            "postal_code_uk": r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$",
            "credit_card": r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$",
            "ssn": r"^\d{3}-\d{2}-\d{4}$"
        }
    
    def _initialize_domain_standards(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific quality standards."""
        return {
            "sales": {
                "required_columns": ["customer_id", "amount", "date", "product_id"],
                "amount_validation": {"min": 0, "max": 1000000},
                "date_freshness_days": 7,
                "duplicate_tolerance": 0.01
            },
            "market": {
                "required_columns": ["survey_id", "response_date", "score"],
                "score_validation": {"min": 0, "max": 100},
                "date_freshness_days": 30,
                "duplicate_tolerance": 0.05
            },
            "product": {
                "required_columns": ["product_id", "name", "category", "price"],
                "price_validation": {"min": 0, "max": 10000},
                "date_freshness_days": 90,
                "duplicate_tolerance": 0.02
            },
            "regulatory": {
                "required_columns": ["compliance_id", "status", "date", "jurisdiction"],
                "status_values": ["compliant", "non-compliant", "pending", "exempt"],
                "date_freshness_days": 1,
                "duplicate_tolerance": 0.0
            },
            "patent": {
                "required_columns": ["patent_id", "title", "filing_date", "assignee"],
                "date_validation": {"min_year": 1900, "max_year": 2030},
                "date_freshness_days": 365,
                "duplicate_tolerance": 0.0
            },
            "rd": {
                "required_columns": ["project_id", "start_date", "status", "budget"],
                "budget_validation": {"min": 0, "max": 10000000},
                "date_freshness_days": 7,
                "duplicate_tolerance": 0.01
            }
        }
    
    def assess_quality(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityReport:
        """Comprehensive data quality assessment."""
        print("🔍 Data Quality Framework: Starting assessment...")
        
        metrics = []
        
        # Assess each quality dimension
        for dimension in QualityDimension:
            metric = self._assess_dimension(df, dimension, domain)
            metrics.append(metric)
        
        # Calculate overall score
        overall_score = np.mean([metric.score for metric in metrics])
        
        # Generate summary and recommendations
        summary = self._generate_quality_summary(metrics, overall_score)
        recommendations = self._generate_recommendations(metrics, domain)
        
        report = QualityReport(
            overall_score=overall_score,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"📊 Overall Quality Score: {overall_score:.2f}")
        return report
    
    def _assess_dimension(self, df: pd.DataFrame, dimension: QualityDimension, domain: Optional[str] = None) -> QualityMetric:
        """Assess a specific quality dimension."""
        if dimension == QualityDimension.COMPLETENESS:
            return self._assess_completeness(df, domain)
        elif dimension == QualityDimension.CONSISTENCY:
            return self._assess_consistency(df, domain)
        elif dimension == QualityDimension.VALIDITY:
            return self._assess_validity(df, domain)
        elif dimension == QualityDimension.UNIQUENESS:
            return self._assess_uniqueness(df, domain)
        elif dimension == QualityDimension.ACCURACY:
            return self._assess_accuracy(df, domain)
        elif dimension == QualityDimension.TIMELINESS:
            return self._assess_timeliness(df, domain)
    
    def _assess_completeness(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityMetric:
        """Assess data completeness."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Check critical columns if domain is specified
        critical_missing = 0
        if domain and domain in self.domain_quality_standards:
            required_columns = self.domain_quality_standards[domain]["required_columns"]
            for col in required_columns:
                if col in df.columns and df[col].isnull().any():
                    critical_missing += df[col].isnull().sum()
        
        details = {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
            "critical_missing": critical_missing,
            "column_completeness": df.notna().mean().to_dict()
        }
        
        recommendations = []
        if completeness_score < 0.95:
            recommendations.append("Address missing values in critical columns")
        if critical_missing > 0:
            recommendations.append("Fill missing values in required columns")
        
        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            details=details,
            recommendations=recommendations
        )
    
    def _assess_consistency(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityMetric:
        """Assess data consistency."""
        consistency_scores = []
        
        # Date format consistency
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        date_consistency = 0
        if date_columns:
            date_consistency_scores = []
            for col in date_columns:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    date_consistency_scores.append(1.0)
                except:
                    date_consistency_scores.append(0.0)
            date_consistency = np.mean(date_consistency_scores) if date_consistency_scores else 1.0
        consistency_scores.append(date_consistency)
        
        # Case consistency in text columns
        text_columns = df.select_dtypes(include=['object']).columns
        case_consistency = 0
        if len(text_columns) > 0:
            case_scores = []
            for col in text_columns:
                if df[col].notna().sum() > 0:
                    # Check if all values follow same case pattern
                    sample_values = df[col].dropna().head(100)
                    if len(sample_values) > 0:
                        is_upper = sample_values.str.isupper().all()
                        is_lower = sample_values.str.islower().all()
                        is_title = sample_values.str.istitle().all()
                        case_scores.append(1.0 if (is_upper or is_lower or is_title) else 0.5)
            case_consistency = np.mean(case_scores) if case_scores else 1.0
        consistency_scores.append(case_consistency)
        
        # Encoding consistency
        encoding_consistency = 1.0  # Assume UTF-8 for now
        consistency_scores.append(encoding_consistency)
        
        overall_consistency = np.mean(consistency_scores)
        
        details = {
            "date_consistency": date_consistency,
            "case_consistency": case_consistency,
            "encoding_consistency": encoding_consistency,
            "inconsistent_columns": self._find_inconsistent_columns(df)
        }
        
        recommendations = []
        if date_consistency < 0.9:
            recommendations.append("Standardize date formats across all date columns")
        if case_consistency < 0.8:
            recommendations.append("Standardize text case (upper/lower/title) in text columns")
        
        return QualityMetric(
            dimension=QualityDimension.CONSISTENCY,
            score=overall_consistency,
            details=details,
            recommendations=recommendations
        )
    
    def _assess_validity(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityMetric:
        """Assess data validity."""
        validity_scores = []
        
        # Email validation
        email_columns = [col for col in df.columns if 'email' in col.lower()]
        email_validity = 0
        if email_columns:
            email_scores = []
            for col in email_columns:
                if df[col].notna().sum() > 0:
                    valid_emails = df[col].str.match(self.validation_patterns["email"], na=False).sum()
                    total_emails = df[col].notna().sum()
                    email_scores.append(valid_emails / total_emails if total_emails > 0 else 1.0)
            email_validity = np.mean(email_scores) if email_scores else 1.0
        validity_scores.append(email_validity)
        
        # Phone validation
        phone_columns = [col for col in df.columns if 'phone' in col.lower()]
        phone_validity = 0
        if phone_columns:
            phone_scores = []
            for col in phone_columns:
                if df[col].notna().sum() > 0:
                    valid_phones = df[col].str.match(self.validation_patterns["phone"], na=False).sum()
                    total_phones = df[col].notna().sum()
                    phone_scores.append(valid_phones / total_phones if total_phones > 0 else 1.0)
            phone_validity = np.mean(phone_scores) if phone_scores else 1.0
        validity_scores.append(phone_validity)
        
        # Numeric range validation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_validity = 0
        if len(numeric_columns) > 0:
            numeric_scores = []
            for col in numeric_columns:
                if df[col].notna().sum() > 0:
                    # Check for reasonable numeric ranges
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    valid_values = ((df[col] >= lower_bound) & (df[col] <= upper_bound)).sum()
                    total_values = df[col].notna().sum()
                    numeric_scores.append(valid_values / total_values if total_values > 0 else 1.0)
            numeric_validity = np.mean(numeric_scores) if numeric_scores else 1.0
        validity_scores.append(numeric_validity)
        
        # Domain-specific validation
        domain_validity = 1.0
        if domain and domain in self.domain_quality_standards:
            domain_validity = self._validate_domain_rules(df, domain)
        validity_scores.append(domain_validity)
        
        overall_validity = np.mean(validity_scores)
        
        details = {
            "email_validity": email_validity,
            "phone_validity": phone_validity,
            "numeric_validity": numeric_validity,
            "domain_validity": domain_validity,
            "invalid_values": self._find_invalid_values(df)
        }
        
        recommendations = []
        if email_validity < 0.9:
            recommendations.append("Clean and validate email addresses")
        if phone_validity < 0.9:
            recommendations.append("Standardize phone number formats")
        if numeric_validity < 0.8:
            recommendations.append("Review and correct numeric outliers")
        
        return QualityMetric(
            dimension=QualityDimension.VALIDITY,
            score=overall_validity,
            details=details,
            recommendations=recommendations
        )
    
    def _assess_uniqueness(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityMetric:
        """Assess data uniqueness."""
        if len(df) == 0:
            return QualityMetric(
                dimension=QualityDimension.UNIQUENESS,
                score=0.0,
                details={"error": "Empty dataset"},
                recommendations=["Provide data for uniqueness assessment"]
            )
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        row_uniqueness = 1 - (duplicate_rows / len(df))
        
        # Check for ID column uniqueness
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        id_uniqueness = 1.0
        if id_columns:
            id_scores = []
            for col in id_columns:
                unique_ids = df[col].nunique()
                total_ids = df[col].notna().sum()
                if total_ids > 0:
                    id_scores.append(unique_ids / total_ids)
            id_uniqueness = np.mean(id_scores) if id_scores else 1.0
        
        # Domain-specific uniqueness requirements
        domain_uniqueness = 1.0
        if domain and domain in self.domain_quality_standards:
            duplicate_tolerance = self.domain_quality_standards[domain]["duplicate_tolerance"]
            if duplicate_rows / len(df) > duplicate_tolerance:
                domain_uniqueness = 0.5
        
        overall_uniqueness = np.mean([row_uniqueness, id_uniqueness, domain_uniqueness])
        
        details = {
            "duplicate_rows": duplicate_rows,
            "row_uniqueness": row_uniqueness,
            "id_uniqueness": id_uniqueness,
            "domain_uniqueness": domain_uniqueness,
            "duplicate_percentage": (duplicate_rows / len(df)) * 100
        }
        
        recommendations = []
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicate rows")
        if id_uniqueness < 0.95:
            recommendations.append("Ensure ID columns contain unique values")
        
        return QualityMetric(
            dimension=QualityDimension.UNIQUENESS,
            score=overall_uniqueness,
            details=details,
            recommendations=recommendations
        )
    
    def _assess_accuracy(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityMetric:
        """Assess data accuracy."""
        accuracy_scores = []
        
        # Outlier detection
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_accuracy = 1.0
        if len(numeric_columns) > 0:
            outlier_scores = []
            for col in numeric_columns:
                if df[col].std() > 0:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    total_values = df[col].notna().sum()
                    outlier_scores.append(1 - (outliers / total_values) if total_values > 0 else 1.0)
            outlier_accuracy = np.mean(outlier_scores) if outlier_scores else 1.0
        accuracy_scores.append(outlier_accuracy)
        
        # Business rule validation
        business_rule_accuracy = self._validate_business_rules(df, domain)
        accuracy_scores.append(business_rule_accuracy)
        
        # Cross-field validation
        cross_field_accuracy = self._validate_cross_fields(df, domain)
        accuracy_scores.append(cross_field_accuracy)
        
        overall_accuracy = np.mean(accuracy_scores)
        
        details = {
            "outlier_accuracy": outlier_accuracy,
            "business_rule_accuracy": business_rule_accuracy,
            "cross_field_accuracy": cross_field_accuracy,
            "outliers_detected": self._count_outliers(df, numeric_columns)
        }
        
        recommendations = []
        if outlier_accuracy < 0.9:
            recommendations.append("Review and validate outlier values")
        if business_rule_accuracy < 0.8:
            recommendations.append("Validate data against business rules")
        
        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=overall_accuracy,
            details=details,
            recommendations=recommendations
        )
    
    def _assess_timeliness(self, df: pd.DataFrame, domain: Optional[str] = None) -> QualityMetric:
        """Assess data timeliness."""
        timeliness_scores = []
        
        # Data freshness
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        freshness_score = 1.0
        if date_columns:
            freshness_scores = []
            for col in date_columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    if dates.notna().any():
                        latest_date = dates.max()
                        days_old = (pd.Timestamp.now() - latest_date).days
                        # Score based on how recent the data is
                        if domain and domain in self.domain_quality_standards:
                            max_days = self.domain_quality_standards[domain]["date_freshness_days"]
                        else:
                            max_days = 30
                        freshness = max(0, 1 - (days_old / max_days))
                        freshness_scores.append(freshness)
            freshness_score = np.mean(freshness_scores) if freshness_scores else 1.0
        timeliness_scores.append(freshness_score)
        
        # Update frequency (if we can detect it)
        update_frequency_score = 1.0  # Placeholder - would need historical data
        timeliness_scores.append(update_frequency_score)
        
        overall_timeliness = np.mean(timeliness_scores)
        
        details = {
            "freshness_score": freshness_score,
            "update_frequency_score": update_frequency_score,
            "latest_date": self._get_latest_date(df, date_columns),
            "data_age_days": self._calculate_data_age(df, date_columns)
        }
        
        recommendations = []
        if freshness_score < 0.7:
            recommendations.append("Update data to ensure freshness")
        
        return QualityMetric(
            dimension=QualityDimension.TIMELINESS,
            score=overall_timeliness,
            details=details,
            recommendations=recommendations
        )
    
    def _validate_domain_rules(self, df: pd.DataFrame, domain: str) -> float:
        """Validate domain-specific business rules."""
        if domain not in self.domain_quality_standards:
            return 1.0
        
        rules = self.domain_quality_standards[domain]
        validation_scores = []
        
        # Validate required columns
        required_columns = rules.get("required_columns", [])
        for col in required_columns:
            if col in df.columns:
                completeness = df[col].notna().mean()
                validation_scores.append(completeness)
        
        # Validate numeric ranges
        amount_validation = rules.get("amount_validation", {})
        if amount_validation and "amount" in df.columns:
            min_val = amount_validation.get("min", 0)
            max_val = amount_validation.get("max", float('inf'))
            valid_amounts = ((df["amount"] >= min_val) & (df["amount"] <= max_val)).mean()
            validation_scores.append(valid_amounts)
        
        # Validate categorical values
        status_values = rules.get("status_values", [])
        if status_values and "status" in df.columns:
            valid_statuses = df["status"].isin(status_values).mean()
            validation_scores.append(valid_statuses)
        
        return np.mean(validation_scores) if validation_scores else 1.0
    
    def _validate_business_rules(self, df: pd.DataFrame, domain: Optional[str] = None) -> float:
        """Validate general business rules."""
        rule_scores = []
        
        # Rule: IDs should be positive
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for col in id_columns:
            if df[col].dtype in ['int64', 'float64']:
                positive_ids = (df[col] > 0).mean()
                rule_scores.append(positive_ids)
        
        # Rule: Amounts should be non-negative
        amount_columns = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'price', 'cost', 'revenue'])]
        for col in amount_columns:
            if df[col].dtype in ['int64', 'float64']:
                non_negative_amounts = (df[col] >= 0).mean()
                rule_scores.append(non_negative_amounts)
        
        # Rule: Dates should be reasonable
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                if dates.notna().any():
                    reasonable_dates = ((dates >= pd.Timestamp('1900-01-01')) & 
                                      (dates <= pd.Timestamp('2030-12-31'))).mean()
                    rule_scores.append(reasonable_dates)
            except:
                pass
        
        return np.mean(rule_scores) if rule_scores else 1.0
    
    def _validate_cross_fields(self, df: pd.DataFrame, domain: Optional[str] = None) -> float:
        """Validate cross-field relationships."""
        cross_field_scores = []
        
        # Rule: Start date should be before end date
        if 'start_date' in df.columns and 'end_date' in df.columns:
            try:
                start_dates = pd.to_datetime(df['start_date'], errors='coerce')
                end_dates = pd.to_datetime(df['end_date'], errors='coerce')
                valid_dates = (start_dates <= end_dates).mean()
                cross_field_scores.append(valid_dates)
            except:
                pass
        
        # Rule: Total should equal sum of parts
        if 'total' in df.columns and 'subtotal' in df.columns and 'tax' in df.columns:
            try:
                valid_totals = (df['total'] == df['subtotal'] + df['tax']).mean()
                cross_field_scores.append(valid_totals)
            except:
                pass
        
        return np.mean(cross_field_scores) if cross_field_scores else 1.0
    
    def _find_inconsistent_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns with consistency issues."""
        inconsistent_columns = []
        
        # Check date format consistency
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                pd.to_datetime(df[col], errors='coerce')
            except:
                inconsistent_columns.append(col)
        
        return inconsistent_columns
    
    def _find_invalid_values(self, df: pd.DataFrame) -> Dict[str, List[Any]]:
        """Find invalid values in the dataset."""
        invalid_values = {}
        
        # Check email columns
        email_columns = [col for col in df.columns if 'email' in col.lower()]
        for col in email_columns:
            if df[col].notna().sum() > 0:
                invalid_emails = df[~df[col].str.match(self.validation_patterns["email"], na=False)][col].tolist()
                if invalid_emails:
                    invalid_values[col] = invalid_emails[:10]  # Limit to first 10
        
        return invalid_values
    
    def _count_outliers(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, int]:
        """Count outliers in numeric columns."""
        outlier_counts = {}
        
        for col in numeric_columns:
            if df[col].std() > 0:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
        
        return outlier_counts
    
    def _get_latest_date(self, df: pd.DataFrame, date_columns: List[str]) -> Optional[str]:
        """Get the latest date from date columns."""
        latest_dates = []
        
        for col in date_columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                if dates.notna().any():
                    latest_dates.append(dates.max())
            except:
                pass
        
        if latest_dates:
            return max(latest_dates).strftime('%Y-%m-%d')
        return None
    
    def _calculate_data_age(self, df: pd.DataFrame, date_columns: List[str]) -> Optional[int]:
        """Calculate data age in days."""
        latest_date = self._get_latest_date(df, date_columns)
        if latest_date:
            try:
                latest_timestamp = pd.Timestamp(latest_date)
                return (pd.Timestamp.now() - latest_timestamp).days
            except:
                pass
        return None
    
    def _generate_quality_summary(self, metrics: List[QualityMetric], overall_score: float) -> Dict[str, Any]:
        """Generate quality summary."""
        dimension_scores = {metric.dimension.value: metric.score for metric in metrics}
        
        # Determine quality grade
        if overall_score >= 0.95:
            grade = "Excellent"
        elif overall_score >= 0.85:
            grade = "Good"
        elif overall_score >= 0.70:
            grade = "Acceptable"
        else:
            grade = "Poor"
        
        return {
            "overall_score": overall_score,
            "quality_grade": grade,
            "dimension_scores": dimension_scores,
            "total_metrics": len(metrics),
            "critical_issues": len([m for m in metrics if m.score < 0.7])
        }
    
    def _generate_recommendations(self, metrics: List[QualityMetric], domain: Optional[str] = None) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Collect recommendations from all metrics
        for metric in metrics:
            recommendations.extend(metric.recommendations)
        
        # Add domain-specific recommendations
        if domain and domain in self.domain_quality_standards:
            domain_rules = self.domain_quality_standards[domain]
            if "required_columns" in domain_rules:
                recommendations.append(f"Ensure all required columns are present for {domain} data")
        
        # Add general recommendations based on overall score
        if overall_score < 0.7:
            recommendations.append("Implement comprehensive data quality monitoring")
            recommendations.append("Establish data quality standards and governance")
        
        return list(set(recommendations))  # Remove duplicates
    
    def save_quality_report(self, report: QualityReport, output_path: str):
        """Save quality report to file."""
        report_dict = {
            "overall_score": report.overall_score,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "timestamp": report.timestamp,
            "metrics": [
                {
                    "dimension": metric.dimension.value,
                    "score": metric.score,
                    "details": metric.details,
                    "recommendations": metric.recommendations
                }
                for metric in report.metrics
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=4, ensure_ascii=False, default=str)
        
        print(f"📁 Quality report saved: {output_path}")
