"""
Comprehensive Monitoring and Alerting System for EconoVault API

This module provides production-ready monitoring, metrics collection,
and alerting for the EconoVault API service.
"""

import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from functools import wraps
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None
import os

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str]
    unit: str = ""
    description: str = ""


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    tags: Dict[str, str]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    response_time_ms: float
    error_rate: float
    availability: float
    details: Dict[str, Any]


class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_old_metrics():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics cleanup: {e}")
        
        self._cleanup_task = threading.Thread(target=cleanup_old_metrics, daemon=True)
        self._cleanup_task.start()
    
    def _cleanup_metrics(self):
        """Remove old metrics data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            for name, metrics_list in self.metrics.items():
                self.metrics[name] = [
                    metric for metric in metrics_list
                    if metric.timestamp > cutoff_time
                ]
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        if labels is None:
            labels = {}
        
        with self._lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True)}"
            self.counters[key] += value
            
            # Store as metric for historical data
            metric = Metric(
                name=name,
                value=self.counters[key],
                metric_type=MetricType.COUNTER,
                timestamp=datetime.utcnow(),
                labels=labels
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        if labels is None:
            labels = {}
        
        with self._lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True)}"
            self.gauges[key] = value
            
            # Store as metric for historical data
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=datetime.utcnow(),
                labels=labels
            )
            self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        if labels is None:
            labels = {}
        
        with self._lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True)}"
            self.histograms[key].append(value)
            
            # Keep only last 1000 values for histograms
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            # Store as metric for historical data
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=datetime.utcnow(),
                labels=labels
            )
            self.metrics[name].append(metric)
    
    def get_metrics(self, name: Optional[str] = None, labels: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Get metrics by name and optional labels"""
        with self._lock:
            if name is None:
                # Return all metrics
                all_metrics = []
                for metrics_list in self.metrics.values():
                    all_metrics.extend(metrics_list)
                return all_metrics
            
            metrics = self.metrics.get(name, [])
            
            if labels:
                # Filter by labels
                filtered_metrics = []
                for metric in metrics:
                    if all(metric.labels.get(k) == v for k, v in labels.items()):
                        filtered_metrics.append(metric)
                return filtered_metrics
            
            return metrics
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current metric values"""
        with self._lock:
            current_values = {}
            
            # Counters
            for key, value in self.counters.items():
                name, labels_str = key.split(":", 1)
                labels = json.loads(labels_str) if labels_str != "{}" else {}
                if name not in current_values:
                    current_values[name] = {}
                current_values[name][labels_str] = value
            
            # Gauges
            for key, value in self.gauges.items():
                name, labels_str = key.split(":", 1)
                labels = json.loads(labels_str) if labels_str != "{}" else {}
                if name not in current_values:
                    current_values[name] = {}
                current_values[name][labels_str] = value
            
            # Histograms (with statistics)
            for key, values in self.histograms.items():
                name, labels_str = key.split(":", 1)
                if values:
                    if name not in current_values:
                        current_values[name] = {}
                    current_values[name][labels_str] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
            
            return current_values


class AlertManager:
    """Centralized alert management system"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self.alert_rules: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._alert_counter = 0
    
    def add_alert_handler(self, level: AlertLevel, handler: Callable):
        """Add an alert handler for a specific level"""
        self.alert_handlers[level].append(handler)
    
    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str = "unknown",
        tags: Optional[Dict[str, str]] = None
    ) -> Alert:
        """Create a new alert"""
        if tags is None:
            tags = {}
        
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(time.time())}"
            
            alert = Alert(
                alert_id=alert_id,
                level=level,
                title=title,
                message=message,
                timestamp=datetime.utcnow(),
                source=source,
                tags=tags
            )
            
            self.alerts.append(alert)
            
            # Trigger alert handlers
            for handler in self.alert_handlers[level]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            logger.warning(f"Alert created: {level.value.upper()} - {title}")
            return alert
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active (unresolved) alerts"""
        with self._lock:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if level:
                active_alerts = [alert for alert in active_alerts if alert.level == level]
            
            return active_alerts
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified number of hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            return [alert for alert in self.alerts if alert.timestamp > cutoff_time]


class HealthMonitor:
    """Service health monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self._lock = threading.Lock()
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    async def run_health_checks(self) -> Dict[str, ServiceHealth]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                
                # Run health check (support async functions)
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Create service health record
                health = ServiceHealth(
                    service_name=name,
                    status=result.get("status", "unknown"),
                    last_check=datetime.utcnow(),
                    response_time_ms=response_time,
                    error_rate=result.get("error_rate", 0.0),
                    availability=result.get("availability", 100.0),
                    details=result.get("details", {})
                )
                
                with self._lock:
                    self.service_health[name] = health
                
                results[name] = health
                
                # Record metrics
                self.metrics.set_gauge(f"health_status_{name}", 1 if health.status == "healthy" else 0)
                self.metrics.record_histogram(f"health_response_time_{name}", response_time)
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                
                # Create error health record
                health = ServiceHealth(
                    service_name=name,
                    status="unhealthy",
                    last_check=datetime.utcnow(),
                    response_time_ms=0,
                    error_rate=100.0,
                    availability=0.0,
                    details={"error": str(e)}
                )
                
                with self._lock:
                    self.service_health[name] = health
                
                results[name] = health
                
                # Record error metrics
                self.metrics.set_gauge(f"health_status_{name}", 0)
        
        return results
    
    def get_service_health(self, service_name: Optional[str] = None) -> Union[ServiceHealth, Dict[str, ServiceHealth], None]:
        """Get health status for a specific service or all services"""
        with self._lock:
            if service_name:
                return self.service_health.get(service_name)
            else:
                return self.service_health.copy()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        with self._lock:
            if not self.service_health:
                return {
                    "status": "unknown",
                    "services_count": 0,
                    "healthy_services": 0,
                    "unhealthy_services": 0,
                    "overall_availability": 0.0
                }
            
            total_services = len(self.service_health)
            healthy_services = sum(1 for h in self.service_health.values() if h.status == "healthy")
            degraded_services = sum(1 for h in self.service_health.values() if h.status == "degraded")
            unhealthy_services = sum(1 for h in self.service_health.values() if h.status == "unhealthy")
            
            # Calculate overall availability
            availabilities = [h.availability for h in self.service_health.values()]
            overall_availability = sum(availabilities) / len(availabilities) if availabilities else 0.0
            
            # Determine overall status
            # Degraded services (like Redis fallback mode) should not make system unhealthy
            if unhealthy_services == 0:
                if degraded_services == 0:
                    status = "healthy"
                else:
                    status = "degraded"
            elif unhealthy_services < total_services / 2:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return {
                "status": status,
                "services_count": total_services,
                "healthy_services": healthy_services,
                "degraded_services": degraded_services,
                "unhealthy_services": unhealthy_services,
                "overall_availability": overall_availability
            }


class MonitoringSystem:
    """Complete monitoring and alerting system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = MetricsCollector(
            retention_hours=self.config.get("metrics_retention_hours", 24)
        )
        self.alerts = AlertManager()
        self.health = HealthMonitor(self.metrics)
        self._monitoring_task = None
        self._shutdown = False
        
        # Register default alert handlers
        self._setup_default_handlers()
        
        # Register default health checks
        self._setup_default_health_checks()
    
    def _setup_default_handlers(self):
        """Setup default alert handlers"""
        
        def log_handler(alert: Alert):
            """Log alerts"""
            logger.log(
                getattr(logging, alert.level.value.upper()),
                f"ALERT [{alert.level.value.upper()}] {alert.title}: {alert.message}"
            )
        
        def console_handler(alert: Alert):
            """Print alerts to console"""
            print(f"🚨 [{alert.level.value.upper()}] {alert.title}: {alert.message}")
        
        # Add handlers for all levels
        for level in AlertLevel:
            self.alerts.add_alert_handler(level, log_handler)
            self.alerts.add_alert_handler(level, console_handler)
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        def system_health_check():
            """Check system resources"""
            try:
                if psutil is None:
                    return {
                        "status": "degraded",
                        "error_rate": 0.0,
                        "availability": 100.0,
                        "details": {"message": "psutil not available, limited system monitoring"}
                    }
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                
                # Determine status
                if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                    status = "unhealthy"
                elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 70:
                    status = "degraded"
                else:
                    status = "healthy"
                
                return {
                    "status": status,
                    "error_rate": 0.0,
                    "availability": 100.0,
                    "details": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_percent": disk_percent,
                        "memory_available_mb": memory.available / (1024 * 1024),
                        "disk_available_gb": disk.free / (1024 * 1024 * 1024)
                    }
                }
                
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error_rate": 100.0,
                    "availability": 0.0,
                    "details": {"error": str(e)}
                }
        
        self.health.register_health_check("system", system_health_check)
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self._monitoring_task is not None:
            logger.warning("Monitoring is already running")
            return
        
        async def monitoring_loop():
            while not self._shutdown:
                try:
                    # Run health checks
                    await self.health.run_health_checks()
                    
                    # Check for alerts based on metrics
                    await self._check_alert_rules()
                    
                    # Wait for next interval
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(interval_seconds)
        
        self._monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info(f"Monitoring started with {interval_seconds}s interval")
    
    async def _check_alert_rules(self):
        """Check alert rules and trigger alerts if needed"""
        # This would contain custom alert rule logic
        # For now, just check system health
        overall_health = self.health.get_overall_health()
        
        if overall_health["status"] == "unhealthy":
            self.alerts.create_alert(
                level=AlertLevel.CRITICAL,
                title="System Health Critical",
                message=f"System is unhealthy with {overall_health['unhealthy_services']} unhealthy services",
                source="monitoring_system",
                tags={"type": "system_health", "status": "unhealthy"}
            )
        elif overall_health["status"] == "degraded" and overall_health["unhealthy_services"] > 0:
            # Only alert for degraded status if there are actual unhealthy services
            # (not just degraded services like Redis fallback mode)
            self.alerts.create_alert(
                level=AlertLevel.WARNING,
                title="System Health Degraded",
                message=f"System is degraded with {overall_health['unhealthy_services']} unhealthy services",
                source="monitoring_system",
                tags={"type": "system_health", "status": "degraded"}
            )
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._shutdown = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        logger.info("Monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall monitoring status"""
        return {
            "monitoring_active": self._monitoring_task is not None,
            "metrics_collected": len(self.metrics.metrics),
            "active_alerts": len(self.alerts.get_active_alerts()),
            "health_checks": len(self.health.health_checks),
            "overall_health": self.health.get_overall_health()
        }


# Global monitoring system instance
monitoring_system = None


def init_monitoring(config: Optional[Dict[str, Any]] = None) -> MonitoringSystem:
    """Initialize global monitoring system"""
    global monitoring_system
    
    if monitoring_system is None:
        monitoring_system = MonitoringSystem(config)
        logger.info("Global monitoring system initialized")
    
    return monitoring_system


def get_monitoring() -> MonitoringSystem:
    """Get global monitoring system instance"""
    if monitoring_system is None:
        raise RuntimeError("Monitoring system not initialized. Call init_monitoring() first.")
    
    return monitoring_system


# Decorator for automatic function monitoring
def monitor_function(metric_name: Optional[str] = None, track_time: bool = True, track_errors: bool = True):
    """Decorator to automatically monitor function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metrics
                if metric_name:
                    get_monitoring().metrics.increment_counter(
                        f"{metric_name}_success_total",
                        labels={"function": func.__name__}
                    )
                
                # Record execution time
                if track_time:
                    execution_time = time.time() - start_time
                    get_monitoring().metrics.record_histogram(
                        f"{metric_name or func.__name__}_duration_seconds",
                        execution_time
                    )
                
                return result
                
            except Exception as e:
                # Record error metrics
                if track_errors and metric_name:
                    get_monitoring().metrics.increment_counter(
                        f"{metric_name}_errors_total",
                        labels={"function": func.__name__, "error_type": type(e).__name__}
                    )
                
                # Create alert for errors
                get_monitoring().alerts.create_alert(
                    level=AlertLevel.ERROR,
                    title=f"Function Error: {func.__name__}",
                    message=f"Error in {func.__name__}: {str(e)}",
                    source="function_monitoring",
                    tags={"function": func.__name__, "error_type": type(e).__name__}
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success metrics
                if metric_name:
                    get_monitoring().metrics.increment_counter(
                        f"{metric_name}_success_total",
                        labels={"function": func.__name__}
                    )
                
                # Record execution time
                if track_time:
                    execution_time = time.time() - start_time
                    get_monitoring().metrics.record_histogram(
                        f"{metric_name or func.__name__}_duration_seconds",
                        execution_time
                    )
                
                return result
                
            except Exception as e:
                # Record error metrics
                if track_errors and metric_name:
                    get_monitoring().metrics.increment_counter(
                        f"{metric_name}_errors_total",
                        labels={"function": func.__name__, "error_type": type(e).__name__}
                    )
                
                # Create alert for errors
                get_monitoring().alerts.create_alert(
                    level=AlertLevel.ERROR,
                    title=f"Function Error: {func.__name__}",
                    message=f"Error in {func.__name__}: {str(e)}",
                    source="function_monitoring",
                    tags={"function": func.__name__, "error_type": type(e).__name__}
                )
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize monitoring
    monitoring = init_monitoring({
        "metrics_retention_hours": 24,
        "alert_webhook_url": "https://hooks.slack.com/your-webhook-url"
    })
    
    # Add custom alert handler
    def slack_alert_handler(alert: Alert):
        """Send alerts to Slack"""
        # This would integrate with Slack API
        print(f"🚨 Slack Alert: {alert.title} - {alert.message}")
    
    monitoring.alerts.add_alert_handler(AlertLevel.ERROR, slack_alert_handler)
    monitoring.alerts.add_alert_handler(AlertLevel.CRITICAL, slack_alert_handler)
    
    # Add custom health check
    def database_health_check():
        """Check database connectivity"""
        # This would check actual database connection
        return {
            "status": "healthy",
            "error_rate": 0.0,
            "availability": 100.0,
            "details": {"connection_time_ms": 25.3}
        }
    
    monitoring.health.register_health_check("database", database_health_check)
    
    # Start monitoring
    monitoring.start_monitoring(interval_seconds=30)
    
    # Example of using the monitoring decorator
    @monitor_function(metric_name="api_request", track_time=True, track_errors=True)
    async def example_api_endpoint():
        """Example API endpoint with monitoring"""
        # Simulate some work
        await asyncio.sleep(0.1)
        return {"status": "success"}
    
    # Run example
    async def main():
        try:
            result = await example_api_endpoint()
            print(f"API Result: {result}")
            
            # Wait a bit to see monitoring in action
            await asyncio.sleep(2)
            
            # Get monitoring status
            status = monitoring.get_status()
            print(f"Monitoring Status: {json.dumps(status, indent=2)}")
            
            # Get current metrics
            current_metrics = monitoring.metrics.get_current_values()
            print(f"Current Metrics: {json.dumps(current_metrics, indent=2)}")
            
        finally:
            monitoring.stop_monitoring()
    
    # Run the example
    asyncio.run(main())