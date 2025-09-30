# EconoVault API

> **⚠️ IMPORTANT**: This API requires external API keys from BLS, BEA, and FRED to function. See [DEPLOYMENT_SETUP.md](DEPLOYMENT_SETUP.md) for setup instructions.

A minimal, GDPR-compliant economic data API built with FastAPI, designed for production deployment with strict file count and complexity constraints.

## Features

### Core Functionality
- **Economic Indicators**: Access to BLS, BEA, and Federal Reserve economic data
- **Time Series Data**: Historical and real-time economic data points
- **Data Normalization**: Consistent data format across different sources
- **Streaming**: Real-time data updates via Server-Sent Events (SSE)

### GDPR Compliance
- **Data Deletion**: Right to erasure (Article 17) with soft/hard/anonymization options
- **Consent Management**: Granular consent tracking and withdrawal
- **Data Export**: Right to data portability (Article 20) in multiple formats
- **Audit Logging**: Comprehensive audit trail for compliance reporting

### Security
- **API Key Authentication**: Secure API key-based authentication
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Data Encryption**: PII encryption at rest and in transit
- **Audit Logging**: Security event tracking and monitoring

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Development Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

```bash
docker build -t econovault-api .
docker run -p 8000:8000 econovault-api
```

## API Endpoints

### Economic Data

#### Get Indicators
```http
GET /v1/indicators
```

Returns list of available economic indicators with filtering options.

**Query Parameters:**
- `source`: Data source (BLS, BEA, FRED, etc.)
- `indicator_type`: Type of indicator (GDP, CPI, UNEMPLOYMENT, etc.)
- `frequency`: Data frequency (MONTHLY, QUARTERLY, etc.)
- `geography_level`: Geographic level (NATIONAL, STATE, etc.)
- `limit`: Maximum number of results (1-1000)

#### Get Specific Indicator
```http
GET /v1/indicators/{series_id}
```

Returns detailed information about a specific economic indicator.

#### Get Indicator Data
```http
GET /v1/indicators/{series_id}/data
```

Returns time series data for a specific indicator.

**Query Parameters:**
- `start_date`: Start date for data range (YYYY-MM-DD)
- `end_date`: End date for data range (YYYY-MM-DD)
- `limit`: Maximum number of data points (1-10000)
- `format`: Response format (json, csv)

#### Stream Indicator Data
```http
GET /v1/indicators/{series_id}/stream
```

Streams real-time updates using Server-Sent Events.

**Query Parameters:**
- `update_interval`: Update interval in seconds (1-60)

#### Stream Market Data
```http
GET /v1/market-data/{symbol}/stream
```

Streams real-time market data (EURUSD, GBPUSD, USDJPY).

#### Stream Economic Calendar
```http
GET /v1/economic-calendar/stream
```

Streams economic calendar events and announcements.

### GDPR Compliance

#### Data Deletion
```http
DELETE /v1/users/{user_id}/data
```

Request deletion of user data with different deletion types.

**Request Body:**
```json
{
  "user_id": "user123",
  "deletion_type": "soft",
  "reason": "User requested deletion",
  "verification_token": "secure_token"
}
```

**Deletion Types:**
- `soft`: Mark as deleted, retain for legal requirements
- `hard`: Complete removal where legally permissible
- `anonymize`: Remove identifying data while preserving patterns

#### Consent Management
```http
POST /v1/users/{user_id}/consent/{consent_type}
```

Update user consent status.

**Consent Types:**
- `marketing`: Marketing communications
- `analytics`: Analytics and tracking
- `functional`: Essential functionality
- `advertising`: Advertising and personalization

**Request Body:**
```json
{
  "consent_type": "analytics",
  "status": "granted",
  "consent_version": "1.0"
}
```

#### Data Export
```http
POST /v1/users/{user_id}/export
```

Request export of user data in portable format.

**Request Body:**
```json
{
  "user_id": "user123",
  "data_categories": ["profile", "preferences"],
  "format": "json",
  "verification_token": "secure_token"
}
```

**Response Formats:**
- `json`: JSON format
- `csv`: CSV format
- `xml`: XML format

## Authentication

The API supports two authentication methods:

### API Key Authentication
Include your API key in the request header:
```http
X-API-Key: pk_your_api_key_here
```

### JWT Bearer Token
Include your JWT token in the Authorization header:
```http
Authorization: Bearer your_jwt_token_here
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:
- **Per minute**: 60 requests (default)
- **Per hour**: 3600 requests (default)

Rate limits are tracked per API key and reset automatically.

## Data Sources

### Bureau of Labor Statistics (BLS)
- Consumer Price Index (CPI)
- Unemployment Rate
- Employment Statistics
- Producer Price Index (PPI)

### Bureau of Economic Analysis (BEA)
- Gross Domestic Product (GDP)
- Personal Income
- Consumer Spending

### Federal Reserve Economic Data (FRED)
- Interest Rates
- Monetary Policy Data
- Economic Indicators

## GDPR Compliance Features

### Data Minimization
- Only essential data is collected and stored
- PII is encrypted at rest and in transit
- Data retention policies are enforced

### Consent Management
- Granular consent tracking by category
- Easy consent withdrawal process
- Consent expiration handling

### Right to Access
- Users can access their data through the API
- Data export in multiple formats
- Complete audit trail of data access

### Right to Erasure
- Soft deletion for legal compliance
- Hard deletion where legally permissible
- Data anonymization options

### Audit Logging
- Comprehensive audit trail
- Tamper-evident logging with hash chains
- Compliance reporting capabilities

## Security Features

### Data Encryption
- AES-256 encryption for sensitive data
- TLS 1.3 for data in transit
- Secure key management

### Input Validation
- Strict input validation and sanitization
- SQL injection prevention
- XSS protection

### Error Handling
- Secure error messages (no sensitive data exposure)
- Comprehensive logging for debugging
- Graceful degradation

## Configuration

### Environment Variables

```bash
# Database (Render PostgreSQL)
DATABASE_URL=postgresql://user:password@econovault-db.render.com/econovault

# Redis (for rate limiting and caching)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
MASTER_ENCRYPTION_KEY=your-encryption-key-here

# External Data Source API Keys (Required for Production)
# These must be obtained from respective providers and set as environment variables
# See DEPLOYMENT_SETUP.md for detailed instructions
BLS_API_KEY=your-bls-api-key
BEA_API_KEY=your-bea-api-key
FRED_API_KEY=your-fred-api-key
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing

Run the test suite:

```bash
# Basic tests
python -m pytest test_main.py -v

# Comprehensive tests
python -m pytest test_comprehensive.py -v
```

## Production Deployment

### Render Web Services

1. Connect your GitHub repository to Render
2. Configure Render Web Service
3. Set up environment variables
4. Connect to Render PostgreSQL database
5. Configure automatic deployment

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location /stream/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        chunked_transfer_encoding off;
    }
}
```

## Monitoring and Observability

### Health Checks
- `/health` endpoint for service health
- Database connectivity checks
- External API availability monitoring

### Metrics
- Request/response times
- Error rates
- Rate limiting statistics
- Data freshness metrics

### Logging
- Structured JSON logging
- GDPR-compliant audit logs
- Security event logging

## Compliance and Legal

### GDPR Compliance
- Data processing agreements
- Privacy policy integration
- Data retention policies
- Breach notification procedures

### Data Sources
- Proper attribution to data providers
- Compliance with terms of service
- Rate limiting respect for external APIs

## Support and Maintenance

### Regular Maintenance
- Database optimization
- Log rotation and cleanup
- Security updates
- Dependency updates

### Monitoring
- Performance monitoring
- Error tracking
- Security monitoring
- Compliance auditing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For support or questions, please contact: support@econovault.com