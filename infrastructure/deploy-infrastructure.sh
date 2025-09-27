#!/bin/bash
# EconoVault API Infrastructure Deployment Script
# This script deploys the complete AWS infrastructure for the EconoVault API

set -euo pipefail

# Configuration
STACK_NAME="EconoVault-Infrastructure"
REGION="us-west-1"
ENVIRONMENT="production"
APPLICATION_NAME="econovault-api"
DOMAIN_NAME="api.econovault.com"
VPC_ID="vpc-0b70d62bed88c8d98"
ALB_DNS="EconoVault-ALB-1841275054.us-west-1.elb.amazonaws.com"
DB_HOST="econovault-db.cx8k6c0mwva4.us-west-1.rds.amazonaws.com"
IMAGE_URI="123456789012.dkr.ecr.us-west-1.amazonaws.com/econovault-api:latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check AWS CLI and permissions
check_aws_cli() {
    log "Checking AWS CLI configuration..."
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if AWS CLI is configured
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    # Check region
    CURRENT_REGION=$(aws configure get region)
    if [[ "$CURRENT_REGION" != "$REGION" ]]; then
        warning "Current AWS region is $CURRENT_REGION, but deployment region is $REGION"
        log "Setting region to $REGION..."
        export AWS_DEFAULT_REGION=$REGION
    fi
    
    success "AWS CLI configuration check passed"
}

# Validate prerequisites
validate_prerequisites() {
    log "Validating deployment prerequisites..."
    
    # Check if required parameters are set
    if [[ -z "$VPC_ID" ]]; then
        error "VPC_ID is not set. Please update the configuration."
        exit 1
    fi
    
    if [[ -z "$ALB_DNS" ]]; then
        error "ALB_DNS is not set. Please update the configuration."
        exit 1
    fi
    
    if [[ -z "$DB_HOST" ]]; then
        error "DB_HOST is not set. Please update the configuration."
        exit 1
    fi
    
    # Check if CloudFormation templates exist
    if [[ ! -f "ssl-certificate.yaml" ]]; then
        error "ssl-certificate.yaml not found in current directory"
        exit 1
    fi
    
    if [[ ! -f "route53-dns.yaml" ]]; then
        error "route53-dns.yaml not found in current directory"
        exit 1
    fi
    
    if [[ ! -f "ecs-task-definitions.yaml" ]]; then
        error "ecs-task-definitions.yaml not found in current directory"
        exit 1
    fi
    
    if [[ ! -f "auto-scaling-policies.yaml" ]]; then
        error "auto-scaling-policies.yaml not found in current directory"
        exit 1
    fi
    
    if [[ ! -f "cloudwatch-logs.yaml" ]]; then
        error "cloudwatch-logs.yaml not found in current directory"
        exit 1
    fi
    
    if [[ ! -f "master-infrastructure.yaml" ]]; then
        error "master-infrastructure.yaml not found in current directory"
        exit 1
    fi
    
    success "Prerequisites validation passed"
}

# Package and upload templates to S3 (if needed)
package_templates() {
    log "Packaging CloudFormation templates..."
    
    # Create S3 bucket for templates if it doesn't exist
    TEMPLATE_BUCKET="econovault-cf-templates-${REGION}-${ACCOUNT_ID}"
    
    if ! aws s3 ls "s3://${TEMPLATE_BUCKET}" &> /dev/null; then
        log "Creating S3 bucket for templates: ${TEMPLATE_BUCKET}"
        aws s3 mb "s3://${TEMPLATE_BUCKET}" --region $REGION
    fi
    
    # Upload templates to S3
    log "Uploading templates to S3..."
    aws s3 cp ssl-certificate.yaml "s3://${TEMPLATE_BUCKET}/ssl-certificate.yaml"
    aws s3 cp route53-dns.yaml "s3://${TEMPLATE_BUCKET}/route53-dns.yaml"
    aws s3 cp ecs-task-definitions.yaml "s3://${TEMPLATE_BUCKET}/ecs-task-definitions.yaml"
    aws s3 cp auto-scaling-policies.yaml "s3://${TEMPLATE_BUCKET}/auto-scaling-policies.yaml"
    aws s3 cp cloudwatch-logs.yaml "s3://${TEMPLATE_BUCKET}/cloudwatch-logs.yaml"
    aws s3 cp master-infrastructure.yaml "s3://${TEMPLATE_BUCKET}/master-infrastructure.yaml"
    
    success "Templates uploaded to S3: s3://${TEMPLATE_BUCKET}/"
    
    # Update template URLs in master template
    log "Updating template URLs in master template..."
    sed -i.bak "s|./ssl-certificate.yaml|https://${TEMPLATE_BUCKET}.s3.${REGION}.amazonaws.com/ssl-certificate.yaml|g" master-infrastructure.yaml
    sed -i.bak "s|./route53-dns.yaml|https://${TEMPLATE_BUCKET}.s3.${REGION}.amazonaws.com/route53-dns.yaml|g" master-infrastructure.yaml
    sed -i.bak "s|./ecs-task-definitions.yaml|https://${TEMPLATE_BUCKET}.s3.${REGION}.amazonaws.com/ecs-task-definitions.yaml|g" master-infrastructure.yaml
    sed -i.bak "s|./auto-scaling-policies.yaml|https://${TEMPLATE_BUCKET}.s3.${REGION}.amazonaws.com/auto-scaling-policies.yaml|g" master-infrastructure.yaml
    sed -i.bak "s|./cloudwatch-logs.yaml|https://${TEMPLATE_BUCKET}.s3.${REGION}.amazonaws.com/cloudwatch-logs.yaml|g" master-infrastructure.yaml
    
    success "Template URLs updated in master template"
}

# Deploy individual components
deploy_ssl_certificate() {
    log "Deploying SSL Certificate..."
    
    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}-ssl-certificate" \
        --template-body file://ssl-certificate.yaml \
        --parameters \
            ParameterKey=DomainName,ParameterValue="${DOMAIN_NAME}" \
            ParameterKey=ValidationMethod,ParameterValue="DNS" \
            ParameterKey=CertificateType,ParameterValue="PUBLIC" \
        --tags \
            Key=Application,Value="${APPLICATION_NAME}" \
            Key=Environment,Value="${ENVIRONMENT}" \
            Key=Component,Value="SSL-Certificate" \
        --region $REGION \
        --capabilities CAPABILITY_IAM
    
    log "Waiting for SSL Certificate stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}-ssl-certificate" \
        --region $REGION
    
    success "SSL Certificate deployed successfully"
}

deploy_route53_dns() {
    log "Deploying Route 53 DNS..."
    
    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}-route53-dns" \
        --template-body file://route53-dns.yaml \
        --parameters \
            ParameterKey=DomainName,ParameterValue="econovault.com" \
            ParameterKey=SubdomainPrefix,ParameterValue="api" \
            ParameterKey=ApplicationLoadBalancerDNS,ParameterValue="${ALB_DNS}" \
            ParameterKey=Environment,ParameterValue="${ENVIRONMENT}" \
            ParameterKey=VPCId,ParameterValue="${VPC_ID}" \
            ParameterKey=EnableHealthChecks,ParameterValue="true" \
            ParameterKey=HealthCheckPath,ParameterValue="/health" \
        --tags \
            Key=Application,Value="${APPLICATION_NAME}" \
            Key=Environment,Value="${ENVIRONMENT}" \
            Key=Component,Value="DNS-Routing" \
        --region $REGION \
        --capabilities CAPABILITY_IAM
    
    log "Waiting for Route 53 DNS stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}-route53-dns" \
        --region $REGION
    
    success "Route 53 DNS deployed successfully"
}

deploy_ecs_task_definitions() {
    log "Deploying ECS Task Definitions..."
    
    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}-ecs-task-definitions" \
        --template-body file://ecs-task-definitions.yaml \
        --parameters \
            ParameterKey=ApplicationName,ParameterValue="${APPLICATION_NAME}" \
            ParameterKey=Environment,ParameterValue="${ENVIRONMENT}" \
            ParameterKey=ImageUri,ParameterValue="${IMAGE_URI}" \
            ParameterKey=ContainerPort,ParameterValue="8000" \
            ParameterKey=TaskCPU,ParameterValue="1024" \
            ParameterKey=TaskMemory,ParameterValue="2048" \
            ParameterKey=ContainerCPU,ParameterValue="512" \
            ParameterKey=ContainerMemory,ParameterValue="1024" \
            ParameterKey=LogRetentionDays,ParameterValue="30" \
            ParameterKey=HealthCheckPath,ParameterValue="/health" \
            ParameterKey=DatabaseHost,ParameterValue="${DB_HOST}" \
            ParameterKey=RedisEndpoint,ParameterValue="redis-endpoint:6379" \
        --tags \
            Key=Application,Value="${APPLICATION_NAME}" \
            Key=Environment,Value="${ENVIRONMENT}" \
            Key=Component,Value="ECS-TaskDefinitions" \
        --region $REGION \
        --capabilities CAPABILITY_IAM
    
    log "Waiting for ECS Task Definitions stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}-ecs-task-definitions" \
        --region $REGION
    
    success "ECS Task Definitions deployed successfully"
}

deploy_auto_scaling() {
    log "Deploying Auto Scaling Policies..."
    
    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}-auto-scaling" \
        --template-body file://auto-scaling-policies.yaml \
        --parameters \
            ParameterKey=ApplicationName,ParameterValue="${APPLICATION_NAME}" \
            ParameterKey=Environment,ParameterValue="${ENVIRONMENT}" \
            ParameterKey=ECSCluster,ParameterValue="${ECSCluster}" \
            ParameterKey=ECSService,ParameterValue="${ECSService}" \
            ParameterKey=MinCapacity,ParameterValue="2" \
            ParameterKey=MaxCapacity,ParameterValue="20" \
            ParameterKey=TargetCPUUtilization,ParameterValue="60" \
            ParameterKey=TargetMemoryUtilization,ParameterValue="70" \
            ParameterKey=ScaleOutCooldown,ParameterValue="60" \
            ParameterKey=ScaleInCooldown,ParameterValue="300" \
            ParameterKey=EnablePredictiveScaling,ParameterValue="true" \
            ParameterKey=EnableScheduledScaling,ParameterValue="true" \
            ParameterKey=BusinessHoursStart,ParameterValue="09:00" \
            ParameterKey=BusinessHoursEnd,ParameterValue="17:00" \
            ParameterKey=TimeZone,ParameterValue="America/Los_Angeles" \
        --tags \
            Key=Application,Value="${APPLICATION_NAME}" \
            Key=Environment,Value="${ENVIRONMENT}" \
            Key=Component,Value="AutoScaling" \
        --region $REGION \
        --capabilities CAPABILITY_IAM
    
    log "Waiting for Auto Scaling stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}-auto-scaling" \
        --region $REGION
    
    success "Auto Scaling Policies deployed successfully"
}

deploy_cloudwatch_logs() {
    log "Deploying CloudWatch Logs Integration..."
    
    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}-cloudwatch-logs" \
        --template-body file://cloudwatch-logs.yaml \
        --parameters \
            ParameterKey=ApplicationName,ParameterValue="${APPLICATION_NAME}" \
            ParameterKey=Environment,ParameterValue="${ENVIRONMENT}" \
            ParameterKey=LogRetentionDays,ParameterValue="30" \
            ParameterKey=EnableContainerInsights,ParameterValue="true" \
            ParameterKey=EnableQueryLogging,ParameterValue="true" \
            ParameterKey=EnableFireLens,ParameterValue="true" \
            ParameterKey=EnableLogMetrics,ParameterValue="true" \
            ParameterKey=EnableLogAlarms,ParameterValue="true" \
        --tags \
            Key=Application,Value="${APPLICATION_NAME}" \
            Key=Environment,Value="${ENVIRONMENT}" \
            Key=Component,Value="CloudWatch-Logs" \
        --region $REGION \
        --capabilities CAPABILITY_IAM
    
    log "Waiting for CloudWatch Logs stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}-cloudwatch-logs" \
        --region $REGION
    
    success "CloudWatch Logs Integration deployed successfully"
}

# Deploy master stack
deploy_master_stack() {
    log "Deploying Master Infrastructure Stack..."
    
    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}" \
        --template-body file://master-infrastructure.yaml \
        --parameters \
            ParameterKey=ApplicationName,ParameterValue="${APPLICATION_NAME}" \
            ParameterKey=Environment,ParameterValue="${ENVIRONMENT}" \
            ParameterKey=DomainName,ParameterValue="${DOMAIN_NAME}" \
            ParameterKey=CertificateValidationMethod,ParameterValue="DNS" \
            ParameterKey=ImageUri,ParameterValue="${IMAGE_URI}" \
            ParameterKey=ECSCluster,ParameterValue="${ECSCluster}" \
            ParameterKey=ECSService,ParameterValue="${ECSService}" \
            ParameterKey=ApplicationLoadBalancerDNS,ParameterValue="${ALB_DNS}" \
            ParameterKey=VPCId,ParameterValue="${VPC_ID}" \
            ParameterKey=DatabaseHost,ParameterValue="${DB_HOST}" \
            ParameterKey=RedisEndpoint,ParameterValue="redis-endpoint:6379" \
        --tags \
            Key=Application,Value="${APPLICATION_NAME}" \
            Key=Environment,Value="${ENVIRONMENT}" \
            Key=Project,Value="EconoVault" \
            Key=ManagedBy,Value="CloudFormation" \
        --region $REGION \
        --capabilities CAPABILITY_IAM
    
    log "Waiting for Master Infrastructure stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}" \
        --region $REGION
    
    success "Master Infrastructure Stack deployed successfully"
}

# Update existing stacks
update_stacks() {
    log "Updating existing infrastructure stacks..."
    
    # Update each component stack
    for component in ssl-certificate route53-dns ecs-task-definitions auto-scaling cloudwatch-logs; do
        log "Updating ${component} stack..."
        
        if aws cloudformation describe-stacks --stack-name "${STACK_NAME}-${component}" --region $REGION &> /dev/null; then
            aws cloudformation update-stack \
                --stack-name "${STACK_NAME}-${component}" \
                --template-body "file://${component}.yaml" \
                --region $REGION \
                --capabilities CAPABILITY_IAM || warning "No updates needed for ${component}"
        else
            warning "Stack ${STACK_NAME}-${component} does not exist, skipping update"
        fi
    done
    
    success "All stacks updated successfully"
}

# Delete stacks (cleanup)
delete_stacks() {
    log "Deleting infrastructure stacks..."
    
    # Delete in reverse order to handle dependencies
    for component in cloudwatch-logs auto-scaling ecs-task-definitions route53-dns ssl-certificate; do
        log "Deleting ${component} stack..."
        
        if aws cloudformation describe-stacks --stack-name "${STACK_NAME}-${component}" --region $REGION &> /dev/null; then
            aws cloudformation delete-stack \
                --stack-name "${STACK_NAME}-${component}" \
                --region $REGION
            
            log "Waiting for ${component} stack deletion to complete..."
            aws cloudformation wait stack-delete-complete \
                --stack-name "${STACK_NAME}-${component}" \
                --region $REGION
            
            success "${component} stack deleted successfully"
        else
            warning "Stack ${STACK_NAME}-${component} does not exist, skipping deletion"
        fi
    done
    
    success "All stacks deleted successfully"
}

# Show stack status
show_status() {
    log "Checking infrastructure stack status..."
    
    echo "=== Infrastructure Stack Status ==="
    for component in ssl-certificate route53-dns ecs-task-definitions auto-scaling cloudwatch-logs; do
        if aws cloudformation describe-stacks --stack-name "${STACK_NAME}-${component}" --region $REGION &> /dev/null; then
            STATUS=$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}-${component}" --region $REGION --query 'Stacks[0].StackStatus' --output text)
            echo "${component}: ${STATUS}"
        else
            echo "${component}: NOT_FOUND"
        fi
    done
    
    echo ""
    echo "=== Key Resources ==="
    
    # Get certificate ARN
    if CERT_ARN=$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}-ssl-certificate" --region $REGION --query 'Stacks[0].Outputs[?OutputKey==`CertificateArn`].OutputValue' --output text 2>/dev/null); then
        echo "SSL Certificate ARN: ${CERT_ARN}"
    fi
    
    # Get API endpoint
    if API_ENDPOINT=$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}-route53-dns" --region $REGION --query 'Stacks[0].Outputs[?OutputKey==`APIEndpoint`].OutputValue' --output text 2>/dev/null); then
        echo "API Endpoint: ${API_ENDPOINT}"
    fi
    
    # Get task definition ARN
    if TASK_DEF_ARN=$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}-ecs-task-definitions" --region $REGION --query 'Stacks[0].Outputs[?OutputKey==`TaskDefinitionArn`].OutputValue' --output text 2>/dev/null); then
        echo "Task Definition ARN: ${TASK_DEF_ARN}"
    fi
}

# Main deployment function
main() {
    log "Starting EconoVault API Infrastructure Deployment"
    log "Region: $REGION"
    log "Environment: $ENVIRONMENT"
    log "Application: $APPLICATION_NAME"
    
    # Get AWS Account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log "AWS Account ID: $ACCOUNT_ID"
    
    # Check prerequisites
    check_aws_cli
    validate_prerequisites
    
    # Parse command line arguments
    case "${1:-deploy}" in
        deploy)
            package_templates
            deploy_ssl_certificate
            deploy_route53_dns
            deploy_ecs_task_definitions
            deploy_auto_scaling
            deploy_cloudwatch_logs
            deploy_master_stack
            show_status
            ;;
        update)
            update_stacks
            show_status
            ;;
        delete)
            delete_stacks
            ;;
        status)
            show_status
            ;;
        *)
            echo "Usage: $0 {deploy|update|delete|status}"
            echo "  deploy  - Deploy all infrastructure components"
            echo "  update  - Update existing infrastructure"
            echo "  delete  - Delete all infrastructure"
            echo "  status  - Show infrastructure status"
            exit 1
            ;;
    esac
    
    success "Infrastructure deployment script completed successfully!"
}

# Run main function with all arguments
main "$@"