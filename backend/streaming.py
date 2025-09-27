"""
EconoVault Real-time Streaming Module

This module implements Server-Sent Events (SSE) streaming for real-time
economic data updates with proper connection management and error handling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
import pandas as pd

from bls_client import BLSClient, BLSAPIException
from database import get_db, DatabaseManager
from models import EconomicIndicatorType

# Configure logging
logger = logging.getLogger(__name__)

# Connection management
active_connections: Set[str] = set()
connection_metadata: Dict[str, Dict[str, Any]] = {}

# Rate limiting for streaming
STREAM_RATE_LIMIT = 60  # seconds between updates
MAX_CONNECTIONS_PER_IP = 5
MAX_TOTAL_CONNECTIONS = 100
CONNECTION_TIMEOUT = 3600  # 1 hour


class StreamingError(Exception):
    """Custom exception for streaming errors"""
    pass


class ConnectionManager:
    """Manages SSE connections with rate limiting and cleanup"""
    
    def __init__(self):
        self.active_connections: Set[str] = set()
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_connection(self, connection_id: str, client_ip: str, series_id: str, metadata: Dict[str, Any] | None = None) -> bool:
        """Add a new connection with rate limiting"""
        
        # Check total connections
        if len(self.active_connections) >= MAX_TOTAL_CONNECTIONS:
            self.logger.warning(f"Maximum connections reached: {MAX_TOTAL_CONNECTIONS}")
            return False
        
        # Check connections per IP
        ip_connections = sum(1 for conn in self.connection_metadata.values() 
                           if conn.get('client_ip') == client_ip)
        if ip_connections >= MAX_CONNECTIONS_PER_IP:
            self.logger.warning(f"Maximum connections per IP reached for {client_ip}: {MAX_CONNECTIONS_PER_IP}")
            return False
        
        # Add connection
        self.active_connections.add(connection_id)
        self.connection_metadata[connection_id] = {
            'client_ip': client_ip,
            'series_id': series_id,
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'update_count': 0,
            'error_count': 0,
            **(metadata or {})
        }
        
        self.logger.info(f"Added connection {connection_id} for series {series_id} from {client_ip}")
        return True
    
    def remove_connection(self, connection_id: str) -> None:
        """Remove a connection"""
        if connection_id in self.active_connections:
            self.active_connections.remove(connection_id)
            metadata = self.connection_metadata.pop(connection_id, {})
            self.logger.info(f"Removed connection {connection_id} for series {metadata.get('series_id')}")
    
    def update_activity(self, connection_id: str) -> None:
        """Update last activity timestamp"""
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]['last_activity'] = datetime.utcnow()
            self.connection_metadata[connection_id]['update_count'] += 1
    
    def record_error(self, connection_id: str, error: str) -> None:
        """Record an error for a connection"""
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]['error_count'] += 1
            self.connection_metadata[connection_id]['last_error'] = error
    
    def cleanup_expired_connections(self) -> int:
        """Clean up expired connections"""
        now = datetime.utcnow()
        expired_connections = []
        
        for connection_id, metadata in self.connection_metadata.items():
            last_activity = metadata.get('last_activity', now)
            if (now - last_activity).total_seconds() > CONNECTION_TIMEOUT:
                expired_connections.append(connection_id)
        
        for connection_id in expired_connections:
            self.remove_connection(connection_id)
        
        if expired_connections:
            self.logger.info(f"Cleaned up {len(expired_connections)} expired connections")
        
        return len(expired_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        now = datetime.utcnow()
        
        stats = {
            'total_connections': len(self.active_connections),
            'connections_by_series': {},
            'connections_by_ip': {},
            'avg_connection_duration': 0,
            'total_updates': 0,
            'total_errors': 0
        }
        
        total_duration = 0
        
        for connection_id, metadata in self.connection_metadata.items():
            series_id = metadata.get('series_id', 'unknown')
            client_ip = metadata.get('client_ip', 'unknown')
            connected_at = metadata.get('connected_at', now)
            update_count = metadata.get('update_count', 0)
            error_count = metadata.get('error_count', 0)
            
            # Count by series
            stats['connections_by_series'][series_id] = stats['connections_by_series'].get(series_id, 0) + 1
            
            # Count by IP
            stats['connections_by_ip'][client_ip] = stats['connections_by_ip'].get(client_ip, 0) + 1
            
            # Connection duration
            duration = (now - connected_at).total_seconds()
            total_duration += duration
            
            # Update totals
            stats['total_updates'] += update_count
            stats['total_errors'] += error_count
        
        # Calculate average duration
        if self.connection_metadata:
            stats['avg_connection_duration'] = total_duration / len(self.connection_metadata)
        
        return stats


# Global connection manager
connection_manager = ConnectionManager()


class RealTimeDataStreamer:
    """Real-time data streaming with BLS API integration"""
    
    def __init__(self, bls_client: BLSClient):
        self.bls_client = bls_client
        self.logger = logging.getLogger(__name__)
        self.last_data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def stream_indicator_data(
        self, 
        series_id: str, 
        request: Request,
        update_interval: int = STREAM_RATE_LIMIT,
        max_events: int = 100
    ) -> StreamingResponse:
        """Stream real-time updates for an economic indicator"""
        
        # Validate series ID
        if not self._validate_series_id(series_id):
            raise HTTPException(status_code=404, detail=f"Indicator {series_id} not found")
        
        # Get client IP for connection management
        client_ip = self._get_client_ip(request)
        connection_id = f"{client_ip}:{series_id}:{int(time.time())}"
        
        # Add connection with rate limiting
        if not connection_manager.add_connection(connection_id, client_ip, series_id):
            raise HTTPException(status_code=429, detail="Connection limit exceeded")
        
        self.logger.info(f"Starting stream for {series_id} from {client_ip}")
        
        async def generate_updates():
            """Generate real-time updates"""
            event_count = 0
            last_data = None
            
            try:
                while event_count < max_events:
                    # Check if client is still connected
                    if await request.is_disconnected():
                        self.logger.info(f"Client disconnected from stream {series_id}")
                        break
                    
                    try:
                        # Get latest data from BLS API
                        current_data = await self._fetch_latest_data(series_id)
                        
                        if current_data:
                            current_point = {
                                "date": current_data.get('date', ''),
                                "value": current_data.get('value'),
                                "period": current_data.get('period', ''),
                                "period_name": current_data.get('period_name', '')
                            }
                            
                            # Only send update if data has changed
                            if last_data is None or current_point != last_data:
                                # Create update event
                                update_data = {
                                    "series_id": series_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "data_point": current_point,
                                    "event_type": "data_update",
                                    "sequence": event_count
                                }
                                
                                # Format as SSE event
                                sse_data = f"data: {json.dumps(update_data)}\n\n"
                                yield sse_data.encode('utf-8')
                                
                                last_data = current_point
                                connection_manager.update_activity(connection_id)
                                self.logger.info(f"Stream update for {series_id}: {current_point}")
                            else:
                                self.logger.debug(f"No data change for {series_id}, skipping update")
                        else:
                            self.logger.warning(f"No data returned for {series_id}")
                        
                    except HTTPException as e:
                        # Send error event to client
                        error_data = {
                            "error": "API Error",
                            "message": e.detail,
                            "timestamp": datetime.utcnow().isoformat(),
                            "event_type": "error",
                            "sequence": event_count
                        }
                        yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                        connection_manager.record_error(connection_id, e.detail)
                        self.logger.error(f"API error in stream for {series_id}: {e.detail}")
                        
                        # Wait longer before retrying after an error
                        await asyncio.sleep(update_interval * 2)
                        continue
                        
                    except Exception as e:
                        # Send generic error event
                        error_data = {
                            "error": "Stream error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                            "event_type": "error",
                            "sequence": event_count
                        }
                        yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
                        connection_manager.record_error(connection_id, str(e))
                        self.logger.error(f"Unexpected error in stream for {series_id}: {str(e)}")
                        
                        # Wait longer before retrying after an error
                        await asyncio.sleep(update_interval * 2)
                        continue
                    
                    # Send periodic heartbeat
                    if event_count % 10 == 0:
                        yield b": heartbeat\n\n"
                    
                    event_count += 1
                    await asyncio.sleep(update_interval)
                    
            except asyncio.CancelledError:
                self.logger.info(f"Stream cancelled for {series_id}")
                raise
            except Exception as e:
                self.logger.error(f"Fatal error in stream {series_id}: {str(e)}")
                # Send final error event
                error_data = {
                    "error": "Fatal stream error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
            finally:
                connection_manager.remove_connection(connection_id)
                self.logger.info(f"Stream ended for {series_id} after {event_count} events")
        
        return StreamingResponse(
            generate_updates(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def _fetch_latest_data(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Fetch latest data for a series with caching"""
        try:
            # Check cache first
            cache_key = f"latest_{series_id}"
            now = time.time()
            
            if (cache_key in self.last_data_cache and 
                now - self.last_data_cache[cache_key].get('timestamp', 0) < self.cache_timeout):
                return self.last_data_cache[cache_key]['data']
            
            # Fetch from BLS API
            data_df = self.bls_client.get_latest_data(series_id)
            
            if data_df.empty:
                return None
            
            # Extract latest point
            latest_point = {
                'date': data_df.iloc[0]['date'].strftime('%Y-%m-%d'),
                'value': float(data_df.iloc[0]['value']) if pd.notna(data_df.iloc[0]['value']) else None,
                'period': data_df.iloc[0]['period'],
                'period_name': data_df.iloc[0]['period_name']
            }
            
            # Cache the result
            self.last_data_cache[cache_key] = {
                'data': latest_point,
                'timestamp': now
            }
            
            return latest_point
            
        except BLSAPIException as e:
            self.logger.error(f"BLS API error fetching latest data for {series_id}: {str(e)}")
            raise HTTPException(status_code=503, detail="BLS API service unavailable")
        except Exception as e:
            self.logger.error(f"Error fetching latest data for {series_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _validate_series_id(self, series_id: str) -> bool:
        """Validate series ID against known indicators"""
        # This should be expanded to check against database
        valid_series = [
            "CUUR0000SA0", "LNS14000000", "CES0000000001", "LNS11300000",
            "LNS12000000", "LNS13000000"
        ]
        return series_id in valid_series
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers first
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"


class StreamingMetrics:
    """Metrics collection for streaming operations"""
    
    def __init__(self):
        self.metrics = {
            'total_connections': 0,
            'total_updates': 0,
            'total_errors': 0,
            'average_connection_duration': 0,
            'peak_connections': 0,
            'start_time': datetime.utcnow()
        }
        self.connection_durations = []
    
    def record_connection(self, duration: float) -> None:
        """Record a completed connection"""
        self.metrics['total_connections'] += 1
        self.connection_durations.append(duration)
        
        # Update average duration
        if self.connection_durations:
            self.metrics['average_connection_duration'] = sum(self.connection_durations) / len(self.connection_durations)
    
    def record_update(self) -> None:
        """Record a data update"""
        self.metrics['total_updates'] += 1
    
    def record_error(self) -> None:
        """Record an error"""
        self.metrics['total_errors'] += 1
    
    def update_peak_connections(self, current_connections: int) -> None:
        """Update peak connections count"""
        if current_connections > self.metrics['peak_connections']:
            self.metrics['peak_connections'] = current_connections
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        
        return {
            **self.metrics,
            'uptime_seconds': uptime,
            'current_connections': len(connection_manager.active_connections),
            'error_rate': self.metrics['total_errors'] / max(self.metrics['total_updates'], 1)
        }


# Global metrics collector
streaming_metrics = StreamingMetrics()


# Utility functions
def cleanup_expired_connections() -> int:
    """Clean up expired connections (can be called periodically)"""
    return connection_manager.cleanup_expired_connections()


def get_streaming_stats() -> Dict[str, Any]:
    """Get comprehensive streaming statistics"""
    connection_stats = connection_manager.get_connection_stats()
    metrics = streaming_metrics.get_metrics()
    
    return {
        'connections': connection_stats,
        'metrics': metrics,
        'system': {
            'active_connections': len(connection_manager.active_connections),
            'max_connections': MAX_TOTAL_CONNECTIONS,
            'max_connections_per_ip': MAX_CONNECTIONS_PER_IP,
            'connection_timeout': CONNECTION_TIMEOUT
        }
    }


# Background task for periodic cleanup
async def periodic_cleanup():
    """Periodic cleanup of expired connections"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            cleaned = cleanup_expired_connections()
            if cleaned > 0:
                logger.info(f"Periodic cleanup removed {cleaned} expired connections")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {str(e)}")


# Initialize streamer (will be set by main application)
streamer: Optional[RealTimeDataStreamer] = None


def initialize_streaming(bls_client: BLSClient):
    """Initialize streaming with BLS client"""
    global streamer
    streamer = RealTimeDataStreamer(bls_client)
    logger.info("Real-time streaming initialized")