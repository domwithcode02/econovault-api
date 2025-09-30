# EconoVaultAPI Development Guidelines

## Build/Lint/Test Commands

### Testing
```bash
# Run single test file
cd backend && pytest tests/test_specific.py -v

# Run tests with coverage
cd backend && pytest tests/ --cov=. --cov-report=term -v

# Run specific test function
cd backend && pytest tests/test_file.py::test_function_name -v
```

### Code Quality
```bash
# Linting
cd backend && ruff check .

# Format checking
cd backend && black --check .

# Import sorting
cd backend && isort --check-only .

# Type checking
cd backend && mypy .
```

### Security Scanning
```bash
# Dependency vulnerabilities
safety check

# Code security issues
bandit -r backend/

# SAST scanning
semgrep --config=auto backend/
```

## Code Style Guidelines

### Python Standards
- **Python 3.11+** with comprehensive type hints
- **FastAPI patterns** with Pydantic models for validation
- **Async/await** for I/O operations
- **Security-first** approach with input validation

### Import Organization
```python
# Standard library imports
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union

# Third-party imports
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime

# Local imports
from config import get_config
from models import EconomicIndicatorType
from security import get_current_user
```

### Naming Conventions
- **Classes**: PascalCase (`EconomicIndicator`, `DataPoint`)
- **Functions/Methods**: snake_case (`get_indicator_data`, `validate_frequency`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Variables**: snake_case (`indicator_data`, `user_id`)

### Error Handling
```python
# Use custom exceptions with proper HTTP status codes
raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail=f"Indicator {indicator_id} not found"
)

# Always include audit logging for security events
audit_logger.log_access_attempt(user_id, resource, success=False)
```

### GDPR Compliance
- Always implement data deletion endpoints
- Include consent management in user models
- Provide data export functionality
- Maintain audit logs with hash chains