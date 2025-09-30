"""
Advanced filtering operators for EconoVault API query parameters.
Provides comprehensive filtering capabilities with operators like eq, ne, gt, lt, in, nin, like, regex.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Query
from sqlalchemy import and_, or_
import logging

logger = logging.getLogger(__name__)


class FilterOperator(str, Enum):
    """Available filter operators"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "nin"
    LIKE = "like"
    NOT_LIKE = "nlike"
    REGEX = "regex"
    NOT_REGEX = "nregex"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"
    NOT_BETWEEN = "nbetween"


class FilterCondition(BaseModel):
    """Single filter condition"""
    field: str = Field(..., description="Field name to filter on")
    operator: FilterOperator = Field(..., description="Filter operator")
    value: Optional[Any] = Field(None, description="Value to filter by")
    values: Optional[List[Any]] = Field(None, description="Multiple values for in/not_in operators")
    
    @validator('value', 'values')
    def validate_value_requirement(cls, v, values):
        """Validate that value is provided when required"""
        operator = values.get('operator')
        if operator in [FilterOperator.IN, FilterOperator.NOT_IN] and not v:
            raise ValueError(f"Operator '{operator}' requires 'values' field")
        elif operator not in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL] and not v:
            if operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
                if not values.get('values'):
                    raise ValueError(f"Operator '{operator}' requires 'values' field")
            else:
                if not v and operator not in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
                    raise ValueError(f"Operator '{operator}' requires 'value' field")
        return v


class FilterGroup(BaseModel):
    """Group of filter conditions with logical operator"""
    operator: str = Field("and", pattern="^(and|or)$", description="Logical operator for conditions")
    conditions: List[Union[FilterCondition, 'FilterGroup']] = Field(..., description="List of conditions or nested groups")
    
    class Config:
        arbitrary_types_allowed = True


class AdvancedFilterParams(BaseModel):
    """Advanced filtering parameters"""
    filter: Optional[str] = Field(None, description="JSON-encoded filter conditions")
    sort: Optional[str] = Field(None, description="JSON-encoded sort conditions")
    fields: Optional[str] = Field(None, description="JSON-encoded field selection")
    
    @validator('filter', 'sort', 'fields')
    def validate_json_fields(cls, v):
        """Validate that JSON fields are properly formatted"""
        if v is not None:
            try:
                import json
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for field")
        return v


class FilterParser:
    """Parse and validate filter expressions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_filter_string(self, filter_str: str) -> FilterGroup:
        """Parse filter string into FilterGroup"""
        try:
            import json
            filter_data = json.loads(filter_str)
            return self._parse_filter_data(filter_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in filter parameter: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid filter format: {str(e)}")
    
    def _parse_filter_data(self, data: Dict[str, Any]) -> FilterGroup:
        """Parse filter data dictionary"""
        if "conditions" not in data:
            raise ValueError("Filter must contain 'conditions' field")
        
        operator = data.get("operator", "and")
        conditions = []
        
        for condition_data in data["conditions"]:
            if "conditions" in condition_data:
                # Nested group
                conditions.append(self._parse_filter_data(condition_data))
            else:
                # Single condition
                conditions.append(FilterCondition(**condition_data))
        
        return FilterGroup(operator=operator, conditions=conditions)


class FilterValidator:
    """Validate filter conditions against allowed fields and operators"""
    
    def __init__(self, allowed_fields: Dict[str, type], allowed_operators: Optional[Dict[str, List[FilterOperator]]] = None):
        self.allowed_fields = allowed_fields
        self.allowed_operators = allowed_operators or {}
    
    def validate_filter_group(self, filter_group: FilterGroup) -> None:
        """Validate filter group against allowed fields and operators"""
        for condition in filter_group.conditions:
            if isinstance(condition, FilterGroup):
                self.validate_filter_group(condition)
            else:
                self.validate_condition(condition)
    
    def validate_condition(self, condition: FilterCondition) -> None:
        """Validate single filter condition"""
        # Check if field is allowed
        if condition.field not in self.allowed_fields:
            raise ValueError(f"Field '{condition.field}' is not allowed for filtering")
        
        # Check if operator is allowed for this field
        field_type = self.allowed_fields[condition.field]
        allowed_ops = self.allowed_operators.get(condition.field, self._get_default_operators(field_type))
        
        if condition.operator not in allowed_ops:
            raise ValueError(f"Operator '{condition.operator}' is not allowed for field '{condition.field}'")
        
        # Validate value type
        self._validate_value_type(condition, field_type)
    
    def _get_default_operators(self, field_type: type) -> List[FilterOperator]:
        """Get default operators for field type"""
        if field_type is str:
            return [
                FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                FilterOperator.LIKE, FilterOperator.NOT_LIKE,
                FilterOperator.REGEX, FilterOperator.NOT_REGEX,
                FilterOperator.IN, FilterOperator.NOT_IN,
                FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
            ]
        elif field_type in (int, float, datetime, date):
            return [
                FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                FilterOperator.GREATER_THAN, FilterOperator.GREATER_THAN_OR_EQUAL,
                FilterOperator.LESS_THAN, FilterOperator.LESS_THAN_OR_EQUAL,
                FilterOperator.BETWEEN, FilterOperator.NOT_BETWEEN,
                FilterOperator.IN, FilterOperator.NOT_IN,
                FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
            ]
        elif field_type is bool:
            return [
                FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
            ]
        else:
            return [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS]
    
    def _validate_value_type(self, condition: FilterCondition, field_type: type) -> None:
        """Validate that value matches expected field type"""
        if condition.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return
        
        value = condition.value or condition.values
        if value is None:
            raise ValueError(f"Value is required for operator '{condition.operator}'")
        
        if condition.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(value, list):
                raise ValueError(f"Operator '{condition.operator}' requires a list of values")
            for item in value:
                self._validate_single_value(item, field_type)
        elif condition.operator in [FilterOperator.BETWEEN, FilterOperator.NOT_BETWEEN]:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Operator '{condition.operator}' requires exactly 2 values")
            for item in value:
                self._validate_single_value(item, field_type)
        else:
            self._validate_single_value(value, field_type)
    
    def _validate_single_value(self, value: Any, field_type: type) -> None:
        """Validate single value against field type"""
        try:
            if field_type is str and not isinstance(value, str):
                raise ValueError(f"Expected string, got {type(value).__name__}")
            elif field_type is int and not isinstance(value, int):
                raise ValueError(f"Expected integer, got {type(value).__name__}")
            elif field_type is float and not isinstance(value, (int, float)):
                raise ValueError(f"Expected numeric, got {type(value).__name__}")
            elif field_type is bool and not isinstance(value, bool):
                raise ValueError(f"Expected boolean, got {type(value).__name__}")
            elif field_type == datetime and not isinstance(value, (str, datetime)):
                raise ValueError(f"Expected datetime or ISO string, got {type(value).__name__}")
            elif field_type == date and not isinstance(value, (str, date)):
                raise ValueError(f"Expected date or ISO string, got {type(value).__name__}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid value type: {str(e)}")


class FilterExecutor:
    """Execute filter conditions against data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_filter_group(self, data: List[Dict[str, Any]], filter_group: FilterGroup) -> List[Dict[str, Any]]:
        """Apply filter group to data"""
        results = []
        
        for item in data:
            if self._evaluate_filter_group(item, filter_group):
                results.append(item)
        
        return results
    
    def _evaluate_filter_group(self, item: Dict[str, Any], filter_group: FilterGroup) -> bool:
        """Evaluate filter group against single item"""
        results = []
        
        for condition in filter_group.conditions:
            if isinstance(condition, FilterGroup):
                results.append(self._evaluate_filter_group(item, condition))
            else:
                results.append(self._evaluate_condition(item, condition))
        
        if filter_group.operator == "and":
            return all(results)
        else:  # or
            return any(results)
    
    def _evaluate_condition(self, item: Dict[str, Any], condition: FilterCondition) -> bool:
        """Evaluate single condition against item"""
        field_value = item.get(condition.field)
        
        try:
            if condition.operator == FilterOperator.EQUALS:
                return field_value == condition.value
            elif condition.operator == FilterOperator.NOT_EQUALS:
                return field_value != condition.value
            elif condition.operator == FilterOperator.GREATER_THAN:
                return field_value is not None and condition.value is not None and field_value > condition.value
            elif condition.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                return field_value is not None and condition.value is not None and field_value >= condition.value
            elif condition.operator == FilterOperator.LESS_THAN:
                return field_value is not None and condition.value is not None and field_value < condition.value
            elif condition.operator == FilterOperator.LESS_THAN_OR_EQUAL:
                return field_value is not None and condition.value is not None and field_value <= condition.value
            elif condition.operator == FilterOperator.IN:
                return field_value is not None and condition.values is not None and field_value in condition.values
            elif condition.operator == FilterOperator.NOT_IN:
                return field_value is not None and condition.values is not None and field_value not in condition.values
            elif condition.operator == FilterOperator.LIKE:
                return self._like_match(str(field_value), str(condition.value))
            elif condition.operator == FilterOperator.NOT_LIKE:
                return not self._like_match(str(field_value), str(condition.value))
            elif condition.operator == FilterOperator.REGEX:
                return condition.value is not None and bool(re.search(condition.value, str(field_value)))
            elif condition.operator == FilterOperator.NOT_REGEX:
                return condition.value is not None and not bool(re.search(condition.value, str(field_value)))
            elif condition.operator == FilterOperator.IS_NULL:
                return field_value is None
            elif condition.operator == FilterOperator.IS_NOT_NULL:
                return field_value is not None
            elif condition.operator == FilterOperator.BETWEEN:
                return (condition.value is not None and len(condition.value) == 2 and 
                        field_value is not None and 
                        condition.value[0] <= field_value <= condition.value[1])
            elif condition.operator == FilterOperator.NOT_BETWEEN:
                return (condition.value is not None and len(condition.value) == 2 and 
                        field_value is not None and 
                        not (condition.value[0] <= field_value <= condition.value[1]))
            else:
                self.logger.warning(f"Unknown operator: {condition.operator}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition}: {str(e)}")
            return False
    
    def _like_match(self, text: str, pattern: str) -> bool:
        """Implement SQL LIKE pattern matching"""
        # Convert SQL LIKE pattern to regex
        regex_pattern = pattern.replace("%", ".*").replace("_", ".")
        return bool(re.search(f"^{regex_pattern}$", text, re.IGNORECASE))
    
    def apply_to_sqlalchemy_query(self, query: Query, filter_group: FilterGroup) -> Query:
        """Apply filter to SQLAlchemy query"""
        filter_expression = self._build_sqlalchemy_filter(filter_group)
        if filter_expression is not None:
            return query.filter(filter_expression)
        return query
    
    def _build_sqlalchemy_filter(self, filter_group: FilterGroup):
        """Build SQLAlchemy filter expression"""
        expressions = []
        
        for condition in filter_group.conditions:
            if isinstance(condition, FilterGroup):
                nested_expr = self._build_sqlalchemy_filter(condition)
                if nested_expr is not None:
                    expressions.append(nested_expr)
            else:
                expr = self._build_sqlalchemy_condition(condition)
                if expr is not None:
                    expressions.append(expr)
        
        if not expressions:
            return None
        
        if filter_group.operator == "and":
            return and_(*expressions)
        else:  # or
            return or_(*expressions)
    
    def _build_sqlalchemy_condition(self, condition: FilterCondition):
        """Build SQLAlchemy condition expression"""
        # Column import removed as it's unused
        
        # This would need to be adapted based on your SQLAlchemy models
        # For now, return None to indicate unsupported
        self.logger.warning(f"SQLAlchemy condition building not fully implemented for {condition}")
        return None


class EconomicDataFilterConfig:
    """Configuration for economic data filtering"""
    
    # Allowed fields for filtering
    INDICATOR_FIELDS = {
        "series_id": str,
        "title": str,
        "source": str,
        "indicator_type": str,
        "frequency": str,
        "seasonal_adjustment": str,
        "geography_level": str,
        "units": str,
        "start_date": date,
        "end_date": date,
        "registration_key_required": bool,
        "last_updated": datetime
    }
    
    DATA_POINT_FIELDS = {
        "date": date,
        "value": float,
        "period": str,
        "period_name": str
    }
    
    # Allowed operators per field type
    FIELD_OPERATORS = {
        str: [
            FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
            FilterOperator.LIKE, FilterOperator.NOT_LIKE,
            FilterOperator.REGEX, FilterOperator.NOT_REGEX,
            FilterOperator.IN, FilterOperator.NOT_IN,
            FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
        ],
        date: [
            FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
            FilterOperator.GREATER_THAN, FilterOperator.GREATER_THAN_OR_EQUAL,
            FilterOperator.LESS_THAN, FilterOperator.LESS_THAN_OR_EQUAL,
            FilterOperator.BETWEEN, FilterOperator.NOT_BETWEEN,
            FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
        ],
        float: [
            FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
            FilterOperator.GREATER_THAN, FilterOperator.GREATER_THAN_OR_EQUAL,
            FilterOperator.LESS_THAN, FilterOperator.LESS_THAN_OR_EQUAL,
            FilterOperator.BETWEEN, FilterOperator.NOT_BETWEEN,
            FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
        ],
        bool: [
            FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
            FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL
        ]
    }


# Utility functions for common filter operations
def create_indicator_filter(source: Optional[str] = None, indicator_type: Optional[str] = None,
                           title_pattern: Optional[str] = None) -> Optional[FilterGroup]:
    """Create common indicator filter conditions"""
    conditions = []
    
    if source:
        conditions.append(FilterCondition(
            field="source",
            operator=FilterOperator.EQUALS,
            value=source.upper(),
            values=None
        ))
    
    if indicator_type:
        conditions.append(FilterCondition(
            field="indicator_type",
            operator=FilterOperator.EQUALS,
            value=indicator_type.upper(),
            values=None
        ))
    
    if title_pattern:
        conditions.append(FilterCondition(
            field="title",
            operator=FilterOperator.LIKE,
            value=f"%{title_pattern}%",
            values=None
        ))
    
    if conditions:
        return FilterGroup(operator="and", conditions=conditions)
    
    return None


def create_date_range_filter(start_date: Optional[date] = None, end_date: Optional[date] = None,
                            field: str = "date") -> Optional[FilterGroup]:
    """Create date range filter conditions"""
    conditions = []
    
    if start_date:
        conditions.append(FilterCondition(
            field=field,
            operator=FilterOperator.GREATER_THAN_OR_EQUAL,
            value=start_date.isoformat(),
            values=None
        ))
    
    if end_date:
        conditions.append(FilterCondition(
            field=field,
            operator=FilterOperator.LESS_THAN_OR_EQUAL,
            value=end_date.isoformat(),
            values=None
        ))
    
    if conditions:
        return FilterGroup(operator="and", conditions=conditions)
    
    return None


def parse_filter_query(filter_query: str, allowed_fields: Dict[str, type]) -> FilterGroup:
    """Parse filter query string and validate against allowed fields"""
    parser = FilterParser()
    validator = FilterValidator(allowed_fields)
    
    # Parse filter
    filter_group = parser.parse_filter_string(filter_query)
    
    # Validate filter
    validator.validate_filter_group(filter_group)
    
    return filter_group