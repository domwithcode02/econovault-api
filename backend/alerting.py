"""
Operational Alerting System for EconoVault API

This module provides comprehensive alerting capabilities including:
- Circuit breaker state change notifications
- API error rate monitoring
- Health check failure alerts
- Integration with Slack, PagerDuty, and email
"""

import asyncio
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from slack_sdk.web.async_client import AsyncWebClient
import pybreaker

logger = logging.getLogger(__name__)


class AlertConfig(BaseModel):
    """Alerting system configuration"""
    slack_token: Optional[str] = None
    slack_channel: str = "#alerts"
    pagerduty_token: Optional[str] = None
    pagerduty_routing_key: Optional[str] = None
    email_enabled: bool = False
    email_from: Optional[str] = None
    email_to: List[str] = Field(default_factory=list)
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True


class AlertData(BaseModel):
    """Alert data structure"""
    alert_type: str
    severity: str  # critical, warning, info
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service: str = "econovault-api"
    environment: str = "production"


class CircuitBreakerAlertListener(pybreaker.CircuitBreakerListener):
    """Circuit breaker listener for alerting"""
    
    def __init__(self, alerting_service: 'AlertingService', breaker_name: str):
        self.alerting_service = alerting_service
        self.breaker_name = breaker_name
        self.logger = logging.getLogger(__name__)
    
    def state_change(self, cb, old_state, new_state):
        """Handle circuit breaker state changes"""
        self.logger.warning(f"Circuit Breaker {self.breaker_name}: {old_state} → {new_state}")
        
        # Determine severity based on state change
        if new_state == "open":
            severity = "critical"
            message = f"🚨 Circuit breaker '{self.breaker_name}' is now OPEN"
        elif new_state == "half-open":
            severity = "warning"
            message = f"⚠️ Circuit breaker '{self.breaker_name}' is HALF-OPEN"
        else:  # closed
            severity = "info"
            message = f"✅ Circuit breaker '{self.breaker_name}' is CLOSED"
        
        # Send alert
        alert_data = AlertData(
            alert_type="circuit_breaker_state_change",
            severity=severity,
            message=message,
            details={
                "breaker_name": self.breaker_name,
                "old_state": old_state,
                "new_state": new_state,
                "failure_count": cb.fail_counter,
                "success_count": cb.success_counter
            }
        )
        
        # Run in background to avoid blocking
        asyncio.create_task(self.alerting_service.send_alert(alert_data))
    
    def failure(self, cb, exc):
        """Handle circuit breaker failures"""
        self.logger.error(f"Circuit breaker {self.breaker_name} failure: {exc}")
        
        # Send alert for critical failures
        if cb.fail_counter >= cb.fail_max * 0.8:  # 80% of failure threshold
            alert_data = AlertData(
                alert_type="circuit_breaker_failure",
                severity="warning",
                message=f"Circuit breaker '{self.breaker_name}' approaching failure threshold",
                details={
                    "breaker_name": self.breaker_name,
                    "failure_count": cb.fail_counter,
                    "max_failures": cb.fail_max,
                    "exception": str(exc)
                }
            )
            asyncio.create_task(self.alerting_service.send_alert(alert_data))


class AlertingService:
    """Comprehensive alerting service for operational monitoring"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.slack_client = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize Slack client if configured
        if config.slack_token:
            self.slack_client = AsyncWebClient(token=config.slack_token)
    
    async def send_alert(self, alert_data: AlertData):
        """Send alert through all configured channels"""
        self.logger.info(f"Sending alert: {alert_data.alert_type} - {alert_data.message}")
        
        # Send to all configured channels
        tasks = []
        
        if self.slack_client:
            tasks.append(self._send_slack_alert(alert_data))
        
        if self.config.email_enabled and self.config.email_to:
            tasks.append(self._send_email_alert(alert_data))
        
        # Wait for all alerts to be sent
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to send alert via channel {i}: {result}")
    
    async def _send_slack_alert(self, alert_data: AlertData):
        """Send alert to Slack"""
        try:
            if not self.slack_client:
                self.logger.warning("Slack client not initialized, skipping Slack alert")
                return
                
            color_map = {
                'critical': 'danger',
                'warning': 'warning',
                'info': 'good'
            }
            
            attachment = {
                "color": color_map.get(alert_data.severity, 'warning'),
                "title": f"🚨 {alert_data.alert_type.replace('_', ' ').title()}",
                "text": alert_data.message,
                "fields": [
                    {"title": "Severity", "value": alert_data.severity, "short": True},
                    {"title": "Service", "value": alert_data.service, "short": True},
                    {"title": "Environment", "value": alert_data.environment, "short": True},
                    {"title": "Time", "value": alert_data.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
                ],
                "footer": "EconoVault API Alerting System",
                "ts": int(alert_data.timestamp.timestamp())
            }
            
            if alert_data.details:
                for key, value in alert_data.details.items():
                    attachment['fields'].append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
            
            await self.slack_client.chat_postMessage(
                channel=self.config.slack_channel,
                attachments=[attachment]
            )
            
            self.logger.info(f"Slack alert sent successfully for {alert_data.alert_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            raise
    
    async def _send_email_alert(self, alert_data: AlertData):
        """Send alert via email"""
        try:
            if not self.config.email_from:
                self.logger.warning("Email from address not configured, skipping email alert")
                return
                
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[{alert_data.severity.upper()}] {alert_data.alert_type}: {alert_data.message}"
            
            body = f"""
Alert Type: {alert_data.alert_type}
Severity: {alert_data.severity}
Service: {alert_data.service}
Environment: {alert_data.environment}
Time: {alert_data.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Message: {alert_data.message}

Details:
{self._format_details(alert_data.details)}

---
This alert was generated by the EconoVault API monitoring system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent successfully for {alert_data.alert_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            raise
    
    def _format_details(self, details: Dict[str, Any]) -> str:
        """Format details dictionary for email"""
        if not details:
            return "No additional details available."
        
        formatted = []
        for key, value in details.items():
            formatted.append(f"{key.replace('_', ' ').title()}: {value}")
        return '\n'.join(formatted)
    
    async def send_health_alert(self, health_status: Dict[str, Any]):
        """Send health check failure alert"""
        failed_checks = health_status.get('failed_checks', [])
        
        if not failed_checks:
            return
        
        severity = "critical" if len(failed_checks) > 2 else "warning"
        
        alert_data = AlertData(
            alert_type="health_check_failure",
            severity=severity,
            message=f"🚨 Health check failed: {', '.join(failed_checks)}",
            details={
                "failed_checks": failed_checks,
                "status": health_status.get('status', 'unknown'),
                "timestamp": health_status.get('timestamp', 'unknown'),
                "check_details": health_status.get('details', {})
            }
        )
        
        await self.send_alert(alert_data)
    
    async def send_api_error_alert(self, method: str, endpoint: str, status_code: int, error_message: str, user_id: Optional[str] = None):
        """Send API error alert"""
        severity = "critical" if status_code >= 500 else "warning"
        
        alert_data = AlertData(
            alert_type="api_error",
            severity=severity,
            message=f"API Error {status_code} on {method} {endpoint}",
            details={
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "error_message": error_message,
                "user_id": user_id or "anonymous"
            }
        )
        
        await self.send_alert(alert_data)
    
    async def send_data_freshness_alert(self, indicator: str, freshness_hours: float, threshold_hours: float = 24.0):
        """Send data freshness alert"""
        if freshness_hours > threshold_hours:
            alert_data = AlertData(
                alert_type="data_freshness_warning",
                severity="warning",
                message=f"Data for {indicator} is {freshness_hours:.1f} hours old (threshold: {threshold_hours}h)",
                details={
                    "indicator": indicator,
                    "freshness_hours": freshness_hours,
                    "threshold_hours": threshold_hours,
                    "data_source": "BLS"
                }
            )
            
            await self.send_alert(alert_data)


# Global alerting service instance
alerting_service: Optional[AlertingService] = None


def initialize_alerting(config: AlertConfig):
    """Initialize the global alerting service"""
    global alerting_service
    alerting_service = AlertingService(config)
    logger.info("Alerting service initialized")


def get_alerting_service() -> AlertingService:
    """Get the alerting service instance"""
    if alerting_service is None:
        raise RuntimeError("Alerting service not initialized")
    return alerting_service