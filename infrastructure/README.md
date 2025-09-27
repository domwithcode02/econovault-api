# EconoVault API Infrastructure

This directory contains the complete Render infrastructure configuration for the EconoVault API, including SSL/TLS certificates, DNS routing, web service configuration, auto scaling policies, and metrics integration.

## Architecture Overview

The infrastructure is built using Render and includes the following components:

### 1. SSL/TLS Certificates (Automatic)
- **Automatic SSL certificate management** by Render
- Support for both public and private certificates
- DNS validation with custom domain integration
- Certificate expiration monitoring with Render alarms
- Automated renewal and management

### 2. DNS Routing (`dns-routing.yaml`)
- Public and private hosted zones
- Multiple routing policies (latency-based, geolocation, weighted, failover)
- Health checks and DNS failover
- DNS query logging and firewall protection
- Support for blue-green deployments

### 3. Web Service Configuration (`web-service.yaml`)
- Production-ready Render web service configuration
- Multi-container patterns with sidecars
- Secrets management with Render environment variables
- Environment-specific configurations
- Security hardening (non-root user, read-only filesystem)

### 4. Auto Scaling Policies (`auto-scaling-policies.yaml`)
- Target tracking scaling (CPU, memory, request count)
- Step scaling for granular control
- Scheduled scaling for business hours
- Predictive scaling using machine learning
- Multi-metric composite scaling policies

### 5. Metrics Integration (`metrics-integration.yaml`)
- Centralized logging with Render Metrics
- Advanced log routing and filtering
- Log metrics and alarms
- Multi-service log aggregation
- Cost optimization with log classes

## Quick Start

### Prerequisites

- Render CLI configured with appropriate permissions
- Docker installed (for local testing)
- Python 3.8+ (for application development)

### Deployment

1. **Configure Render CLI:**
   ```bash
   render login
   # Enter your Render credentials
   ```

2. **Make the deployment script executable:**
   ```bash
   chmod +x deploy-infrastructure.sh
   ```

3. **Deploy the infrastructure:**
   ```bash
   ./deploy-infrastructure.sh deploy
   ```

4. **Check deployment status:**
   ```bash
   ./deploy-infrastructure.sh status
   ```

### Update Infrastructure

To update existing infrastructure:
```bash
./deploy-infrastructure.sh update
```

### Delete Infrastructure

To remove all infrastructure components:
```bash
./deploy-infrastructure.sh delete
```

## Configuration

### Environment-Specific Settings

The infrastructure supports three environments:

- **Development**: Lower resource allocation, shorter log retention
- **Staging**: Production-like configuration with reduced capacity
- **Production**: Full production configuration with high availability

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Environment` | Deployment environment | `production` |
| `DomainName` | API domain name | `api.econovault.com` |
| `MinInstances` | Minimum instance count | `2` |
| `MaxInstances` | Maximum instance count | `20` |
| `TargetCPUUtilization` | CPU scaling threshold | `60` |
| `LogRetentionDays` | Render log retention | `30` |

## Security Features

### SSL/TLS Security
- TLS 1.2+ enforcement
- Strong cipher suites
- Perfect Forward Secrecy (PFS)
- Certificate transparency monitoring

### Network Security
- Render-managed network isolation
- Service permissions with least privilege
- DNS firewall for domain filtering
- Render network monitoring

### Container Security
- Non-root container execution
- Read-only root filesystem
- Linux capability dropping
- Image scanning with Render

### Data Security
- Encryption at rest with Render
- Secrets management with Render environment variables
- Parameter encryption with Render
- Audit logging with Render Logs

## Monitoring and Alerting

### Render Metrics
- Web service metrics (CPU, memory, instance count)
- Application metrics (request count, latency, errors)
- Custom business metrics
- Log-based metrics

### Alarms
- High CPU/memory utilization
- Service health checks
- Certificate expiration
- Error rate thresholds
- Performance degradation

### Dashboards
- Application performance dashboard
- Infrastructure health dashboard
- Cost monitoring dashboard
- Security monitoring dashboard

## Cost Optimization

### Resource Right-Sizing
- CPU/memory optimization based on actual usage
- Render Spot instances for non-critical workloads
- Scheduled scaling for off-hours

### Log Management
- Environment-based log retention policies
- Infrequent Access log class for non-production
- Render archival for long-term storage

### Auto Scaling
- Predictive scaling to prevent over-provisioning
- Multi-metric scaling for efficiency
- Scheduled scaling for known patterns

## High Availability

### Multi-Region Deployment
- Web services distributed across regions
- Load balancer with health checks
- Database with multi-region configuration

### Failover Capabilities
- Render health checks and failover
- Auto scaling for capacity management
- Circuit breakers for deployment safety

### Backup and Recovery
- Automated backups for Render PostgreSQL
- Render disk volume versioning for data protection
- Disaster recovery procedures

## Compliance

### GDPR Compliance
- Data encryption in transit and at rest
- Audit logging with hash chains
- Data retention policies
- Right to be forgotten implementation

### Financial Services Compliance
- PCI DSS requirements for payment processing
- SOC 2 Type II controls
- Data residency requirements
- Regulatory reporting capabilities

## Troubleshooting

### Common Issues

1. **Certificate Validation Failed**
   - Check DNS records are properly configured
   - Verify domain ownership
   - Review validation method settings

2. **Web Service Fails to Start**
   - Check Render logs for errors
   - Verify web service configuration
   - Review service permissions

3. **Auto Scaling Not Working**
   - Check Render metrics
   - Verify scalable target registration
   - Review scaling policies

4. **DNS Resolution Issues**
   - Check Render health checks
   - Verify domain configuration
   - Review DNS propagation

### Debug Commands

```bash
# Check service status
render service status econovault-api

# View web service events
render service events econovault-api

# Check auto scaling policies
render service scaling econovault-api

# View Render logs
render logs econovault-api --follow
```

## Development

### Local Testing

1. **Build and test Docker image:**
   ```bash
   docker build -t econovault-api .
   docker run -p 8000:8000 econovault-api
   ```

2. **Run infrastructure validation:**
   ```bash
   render validate infrastructure.yaml
   ```

### Adding New Components

1. Create a new Render configuration file
2. Add it to the main configuration as a nested service
3. Update the deployment script if needed
4. Test in development environment first

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Render service events
3. Check Render logs
4. Contact the development team

## License

This infrastructure configuration is part of the EconoVault API project and follows the same licensing terms.