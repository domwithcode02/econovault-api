"""
Cursor-based pagination for EconoVault API.
Provides efficient pagination using cursors instead of offset-based approach.
"""

from __future__ import annotations
import base64
import json
import time
from typing import Optional, Dict, Any, List, Generic, TypeVar, Union, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field, field_serializer
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
import logging

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Cursor(BaseModel):
    """Cursor for pagination"""
    value: Union[str, int, datetime]
    direction: str = Field(default="next", pattern=r"^(next|prev)$")
    
    @field_serializer('value')
    def serialize_value(self, value: Union[str, int, datetime], _info) -> Union[str, int]:
        """Serialize datetime values to ISO format"""
        if isinstance(value, datetime):
            return value.isoformat()
        return value


class PageInfo(BaseModel):
    """Page information for cursor-based pagination"""
    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None
    total_count: Optional[int] = None


class CursorPaginationParams(BaseModel):
    """Parameters for cursor-based pagination"""
    first: Optional[int] = Field(None, ge=1, le=100, description="Number of items to fetch (forward pagination)")
    after: Optional[str] = Field(None, description="Cursor to fetch items after")
    last: Optional[int] = Field(None, ge=1, le=100, description="Number of items to fetch (backward pagination)")
    before: Optional[str] = Field(None, description="Cursor to fetch items before")
    
    class Config:
        # Allow forward references to be resolved
        arbitrary_types_allowed = True


# Rebuild the model to resolve any forward references
CursorPaginationParams.model_rebuild()


def get_pagination_params(
    first: Optional[int] = None,
    after: Optional[str] = None,
    last: Optional[int] = None,
    before: Optional[str] = None
) -> CursorPaginationParams:
    """Dependency function for cursor-based pagination parameters"""
    return CursorPaginationParams(first=first, after=after, last=last, before=before)


class CursorPaginationResponse(BaseModel, Generic[T]):
    """Response for cursor-based pagination"""
    edges: List[Dict[str, Any]] = Field(description="List of edges containing nodes and cursors")
    page_info: PageInfo
    total_count: Optional[int] = None


class CursorEncoder:
    """Encode and decode cursors"""
    
    @staticmethod
    def encode_cursor(value: Union[str, int, datetime], direction: str = "next") -> str:
        """Encode cursor value to base64 string"""
        cursor_data = {
            "value": value,
            "direction": direction,
            "timestamp": int(time.time())
        }
        
        # Convert to JSON and encode to base64
        json_str = json.dumps(cursor_data, default=str)
        encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
        return encoded
    
    @staticmethod
    def decode_cursor(cursor_str: str) -> Optional[Cursor]:
        """Decode cursor from base64 string"""
        try:
            # Decode from base64
            decoded_bytes = base64.urlsafe_b64decode(cursor_str.encode())
            cursor_data = json.loads(decoded_bytes.decode())
            
            # Parse datetime if needed
            if isinstance(cursor_data["value"], str) and "T" in cursor_data["value"]:
                cursor_data["value"] = datetime.fromisoformat(cursor_data["value"])
            
            return Cursor(**cursor_data)
            
        except Exception as e:
            logger.warning(f"Failed to decode cursor: {e}")
            return None


class CursorPaginator:
    """Generic cursor-based paginator"""
    
    def __init__(self, default_page_size: int = 20, max_page_size: int = 100):
        self.default_page_size = default_page_size
        self.max_page_size = max_page_size
        self.encoder = CursorEncoder()
    
    def paginate_query(
        self,
        query,
        cursor_field: str,
        first: Optional[int] = None,
        after: Optional[str] = None,
        last: Optional[int] = None,
        before: Optional[str] = None,
        order_by: str = "asc"
    ) -> Dict[str, Any]:
        """
        Paginate a SQLAlchemy query using cursors
        
        Args:
            query: SQLAlchemy query object
            cursor_field: Field name to use for cursor positioning
            first: Number of items to fetch (forward pagination)
            after: Cursor to fetch items after
            last: Number of items to fetch (backward pagination)
            before: Cursor to fetch items before
            order_by: Sort order ("asc" or "desc")
        
        Returns:
            Dictionary with paginated results and page info
        """
        
        # Determine pagination direction and parameters
        if first is not None:
            # Forward pagination
            limit = min(first, self.max_page_size)
            direction = "forward"
            cursor = self.encoder.decode_cursor(after) if after else None
        elif last is not None:
            # Backward pagination
            limit = min(last, self.max_page_size)
            direction = "backward"
            cursor = self.encoder.decode_cursor(before) if before else None
        else:
            # Default to forward pagination
            limit = self.default_page_size
            direction = "forward"
            cursor = None
        
        # Apply cursor filter if provided
        if cursor:
            if direction == "forward":
                if order_by == "asc":
                    query = query.filter(getattr(query.column_descriptions[0]['entity'], cursor_field) > cursor.value)
                else:
                    query = query.filter(getattr(query.column_descriptions[0]['entity'], cursor_field) < cursor.value)
            else:  # backward
                if order_by == "asc":
                    query = query.filter(getattr(query.column_descriptions[0]['entity'], cursor_field) < cursor.value)
                else:
                    query = query.filter(getattr(query.column_descriptions[0]['entity'], cursor_field) > cursor.value)
        
        # Apply ordering
        if order_by == "asc":
            query = query.order_by(asc(cursor_field))
        else:
            query = query.order_by(desc(cursor_field))
        
        # Fetch one extra item to determine if there are more pages
        query = query.limit(limit + 1)
        
        # Execute query
        items = query.all()
        
        # Check if there are more items
        has_more = len(items) > limit
        if has_more:
            items = items[:limit]  # Remove the extra item
        
        # Reverse items for backward pagination (to maintain correct order)
        if direction == "backward":
            items = list(reversed(items))
        
        # Generate cursors for items
        edges = []
        for item in items:
            cursor_value = getattr(item, cursor_field)
            cursor = self.encoder.encode_cursor(cursor_value)
            edges.append({
                "node": item,
                "cursor": cursor
            })
        
        # Determine page info
        has_next_page = has_more if direction == "forward" else (cursor is not None)
        has_previous_page = has_more if direction == "backward" else (cursor is not None)
        
        start_cursor = edges[0]["cursor"] if edges else None
        end_cursor = edges[-1]["cursor"] if edges else None
        
        return {
            "edges": edges,
            "page_info": {
                "has_next_page": has_next_page,
                "has_previous_page": has_previous_page,
                "start_cursor": start_cursor,
                "end_cursor": end_cursor
            },
            "items": items  # For backward compatibility
        }
    
    def create_pagination_response(
        self,
        edges: List[Dict[str, Any]],
        page_info: Dict[str, Any],
        total_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create standardized pagination response"""
        return {
            "data": {
                "edges": edges,
                "page_info": page_info,
                "total_count": total_count
            }
        }


class EconomicDataCursorPaginator(CursorPaginator):
    """Specialized cursor paginator for economic data"""
    
    def __init__(self):
        super().__init__(default_page_size=50, max_page_size=500)
    
    def paginate_indicators(
        self,
        db: Session,
        cursor_field: str = "series_id",
        first: Optional[int] = None,
        after: Optional[str] = None,
        last: Optional[int] = None,
        before: Optional[str] = None,
        source: Optional[str] = None,
        indicator_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Paginate economic indicators with filtering"""
        
        from database import EconomicIndicator
        
        # Build base query
        query = db.query(EconomicIndicator)
        
        # Apply filters
        if source:
            query = query.filter(EconomicIndicator.source == source)
        if indicator_type:
            query = query.filter(EconomicIndicator.indicator_type == indicator_type)
        
        # Paginate
        result = self.paginate_query(
            query=query,
            cursor_field=cursor_field,
            first=first,
            after=after,
            last=last,
            before=before,
            order_by="asc"
        )
        
        # Get total count for pagination info
        total_count = query.count()
        result["page_info"]["total_count"] = total_count
        
        return result
    
    def paginate_data_points(
        self,
        db: Session,
        series_id: str,
        cursor_field: str = "date",
        first: Optional[int] = None,
        after: Optional[str] = None,
        last: Optional[int] = None,
        before: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Paginate data points for a specific indicator"""
        
        from database import DataPoint, EconomicIndicator
        
        # Build base query
        query = db.query(DataPoint).join(EconomicIndicator).filter(
            EconomicIndicator.series_id == series_id
        )
        
        # Apply date filters
        if start_date:
            query = query.filter(DataPoint.date >= start_date)
        if end_date:
            query = query.filter(DataPoint.date <= end_date)
        
        # Paginate
        result = self.paginate_query(
            query=query,
            cursor_field=cursor_field,
            first=first,
            after=after,
            last=last,
            before=before,
            order_by="desc"  # Most recent data first
        )
        
        return result


# Utility functions for cursor-based pagination
def create_cursor_from_value(value: Union[str, int, datetime], direction: str = "next") -> str:
    """Create a cursor from a value"""
    encoder = CursorEncoder()
    return encoder.encode_cursor(value, direction)


def decode_cursor_value(cursor_str: str) -> Optional[Union[str, int, datetime]]:
    """Extract value from cursor"""
    encoder = CursorEncoder()
    cursor = encoder.decode_cursor(cursor_str)
    return cursor.value if cursor else None


def validate_cursor_pagination_params(first: Optional[int], after: Optional[str], 
                                    last: Optional[int], before: Optional[str]) -> Dict[str, Any]:
    """Validate cursor pagination parameters"""
    errors = []
    
    # Check for conflicting pagination directions
    if first is not None and last is not None:
        errors.append("Cannot use both 'first' and 'last' pagination parameters")
    
    # Validate cursor formats
    if after:
        cursor = CursorEncoder().decode_cursor(after)
        if not cursor:
            errors.append("Invalid 'after' cursor format")
    
    if before:
        cursor = CursorEncoder().decode_cursor(before)
        if not cursor:
            errors.append("Invalid 'before' cursor format")
    
    # Validate page sizes
    if first is not None and (first < 1 or first > 500):
        errors.append("'first' must be between 1 and 500")
    
    if last is not None and (last < 1 or last > 500):
        errors.append("'last' must be between 1 and 500")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }