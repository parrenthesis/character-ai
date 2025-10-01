"""
ELK stack integration for log analysis.

This module provides Elasticsearch, Logstash, and Kibana integration
for advanced log analysis and visualization of the Character AI.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from ..core.log_aggregation import LogEntry, LogLevel, LogSource
from ..core.logging import get_logger

logger = get_logger(__name__)


class ElasticsearchConfig(BaseModel):
    """Elasticsearch configuration."""

    host: str = "localhost"
    port: int = 9200
    scheme: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    index_prefix: str = "icp-logs"
    max_retries: int = 3
    timeout: int = 30


class LogstashConfig(BaseModel):
    """Logstash configuration."""

    host: str = "localhost"
    port: int = 5044
    pipeline_id: Optional[str] = None
    batch_size: int = 100
    flush_interval: int = 5


class KibanaConfig(BaseModel):
    """Kibana configuration."""

    host: str = "localhost"
    port: int = 5601
    scheme: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None


class ELKIntegration:
    """ELK stack integration for log analysis."""

    def __init__(
        self,
        elasticsearch: ElasticsearchConfig,
        logstash: LogstashConfig,
        kibana: KibanaConfig,
    ):
        self.elasticsearch = elasticsearch
        self.logstash = logstash
        self.kibana = kibana

        # Build URLs
        self.es_url = (
            f"{elasticsearch.scheme}://{elasticsearch.host}:{elasticsearch.port}"
        )
        self.kibana_url = f"{kibana.scheme}://{kibana.host}:{kibana.port}"

        # HTTP client configuration
        self.auth: Optional[tuple[str, str]] = None
        if elasticsearch.username and elasticsearch.password:
            self.auth = (elasticsearch.username, elasticsearch.password)

        logger.info(
            "ELKIntegration initialized", es_url=self.es_url, kibana_url=self.kibana_url

        )

    async def create_index_template(self) -> bool:
        """Create Elasticsearch index template for CAI logs."""
        template = {
            "index_patterns": [f"{self.elasticsearch.index_prefix}-*"],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index.refresh_interval": "5s",
                },
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "message": {"type": "text", "analyzer": "standard"},
                        "logger_name": {"type": "keyword"},
                        "request_id": {"type": "keyword"},
                        "trace_id": {"type": "keyword"},
                        "device_id": {"type": "keyword"},
                        "character_id": {"type": "keyword"},
                        "component": {"type": "keyword"},
                        "duration_ms": {"type": "float"},
                        "status_code": {"type": "integer"},
                        "error_type": {"type": "keyword"},
                        "confidence": {"type": "float"},
                        "metadata": {"type": "object"},
                    }
                },
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.elasticsearch.timeout) as client:
                response = await client.put(
                    f"{self.es_url}/_index_template/icp-logs-template",
                    json=template,
                    auth=self.auth,  # type: ignore
                )
                response.raise_for_status()
                logger.info("Index template created successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to create index template: {e}")
            return False

    async def index_log_entry(self, entry: LogEntry) -> bool:
        """Index a log entry to Elasticsearch."""
        index_name = (
            f"{self.elasticsearch.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}"
        )

        document = {
            "timestamp": entry.timestamp.isoformat(),
            "level": entry.level.value,
            "source": entry.source.value,
            "message": entry.message,
            "logger_name": entry.logger_name,
            "request_id": entry.request_id,
            "trace_id": entry.trace_id,
            "device_id": entry.device_id,
            "character_id": entry.character_id,
            "component": entry.component,
            "duration_ms": entry.duration_ms,
            "status_code": entry.status_code,
            "error_type": entry.error_type,
            "confidence": entry.confidence,
            "metadata": entry.metadata,
        }

        try:
            async with httpx.AsyncClient(timeout=self.elasticsearch.timeout) as client:
                response = await client.post(
                    f"{self.es_url}/{index_name}/_doc", json=document, auth=self.auth  # type: ignore
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to index log entry: {e}")
            return False

    async def bulk_index_logs(self, entries: List[LogEntry]) -> bool:
        """Bulk index multiple log entries."""
        if not entries:
            return True

        index_name = (
            f"{self.elasticsearch.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}"
        )
        bulk_data: List[Dict[str, Any]] = []

        for entry in entries:
            # Index action
            bulk_data.append({"index": {"_index": index_name, "_id": entry.id}})

            # Document
            document = {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,
                "source": entry.source.value,
                "message": entry.message,
                "logger_name": entry.logger_name,
                "request_id": entry.request_id,
                "trace_id": entry.trace_id,
                "device_id": entry.device_id,
                "character_id": entry.character_id,
                "component": entry.component,
                "duration_ms": entry.duration_ms,
                "status_code": entry.status_code,
                "error_type": entry.error_type,
                "confidence": entry.confidence,
                "metadata": entry.metadata,
            }
            bulk_data.append(document)

        try:
            async with httpx.AsyncClient(timeout=self.elasticsearch.timeout) as client:
                response = await client.post(
                    f"{self.es_url}/_bulk",
                    data="\n".join(json.dumps(item) for item in bulk_data) + "\n",  # type: ignore
                    headers={"Content-Type": "application/x-ndjson"},
                    auth=self.auth,  # type: ignore
                )
                response.raise_for_status()
                logger.info(f"Bulk indexed {len(entries)} log entries")
                return True
        except Exception as e:
            logger.error(f"Failed to bulk index logs: {e}")
            return False

    async def search_logs(
        self,
        query: str = "*",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        source: Optional[LogSource] = None,
        size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search logs in Elasticsearch."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()

        # Build query
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat(),
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": size,
        }

        # Add text search
        if query != "*":
            es_query["query"]["bool"]["must"].append(  # type: ignore
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["message", "logger_name", "component"],
                    }
                }
            )

        # Add level filter
        if level:
            es_query["query"]["bool"]["must"].append({"term": {"level": level.value}})  # type: ignore

        # Add source filter
        if source:
            es_query["query"]["bool"]["must"].append({"term": {"source": source.value}})  # type: ignore


        try:
            index_pattern = f"{self.elasticsearch.index_prefix}-*"
            async with httpx.AsyncClient(timeout=self.elasticsearch.timeout) as client:
                response = await client.post(
                    f"{self.es_url}/{index_pattern}/_search",
                    json=es_query,
                    auth=self.auth,  # type: ignore
                )
                response.raise_for_status()

                result = response.json()
                hits = result.get("hits", {}).get("hits", [])
                return [hit["_source"] for hit in hits]
        except Exception as e:
            logger.error(f"Failed to search logs: {e}")
            return []

    async def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get log statistics from Elasticsearch."""
        start_time = datetime.now() - timedelta(hours=hours)
        end_time = datetime.now()

        query = {
            "query": {
                "range": {
                    "timestamp": {
                        "gte": start_time.isoformat(),
                        "lte": end_time.isoformat(),
                    }
                }
            },
            "aggs": {
                "by_level": {"terms": {"field": "level"}},
                "by_source": {"terms": {"field": "source"}},
                "by_hour": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "hour",
                    }
                },
                "error_rate": {"filter": {"terms": {"level": ["ERROR", "CRITICAL"]}}},
            },
            "size": 0,
        }

        try:
            index_pattern = f"{self.elasticsearch.index_prefix}-*"
            async with httpx.AsyncClient(timeout=self.elasticsearch.timeout) as client:
                response = await client.post(
                    f"{self.es_url}/{index_pattern}/_search", json=query, auth=self.auth  # type: ignore

                )
                response.raise_for_status()

                result = response.json()
                aggs = result.get("aggregations", {})

                return {
                    "total_logs": result.get("hits", {})
                    .get("total", {})
                    .get("value", 0),
                    "by_level": {
                        bucket["key"]: bucket["doc_count"]
                        for bucket in aggs.get("by_level", {}).get("buckets", [])
                    },
                    "by_source": {
                        bucket["key"]: bucket["doc_count"]
                        for bucket in aggs.get("by_source", {}).get("buckets", [])
                    },
                    "by_hour": [
                        {
                            "timestamp": bucket["key_as_string"],
                            "count": bucket["doc_count"],
                        }
                        for bucket in aggs.get("by_hour", {}).get("buckets", [])
                    ],
                    "error_count": aggs.get("error_rate", {}).get("doc_count", 0),
                }
        except Exception as e:
            logger.error(f"Failed to get log statistics: {e}")
            return {}

    def create_kibana_dashboard(self) -> Dict[str, Any]:
        """Create Kibana dashboard configuration."""
        return {
            "version": "8.0.0",
            "objects": [
                {
                    "id": "icp-logs-dashboard",
                    "type": "dashboard",
                    "attributes": {
                        "title": "CAI Logs Dashboard",
                        "description": "Character AI Log Analysis",
                        "panelsJSON": json.dumps(
                            [
                                {
                                    "version": "8.0.0",
                                    "gridData": {
                                        "x": 0,
                                        "y": 0,
                                        "w": 24,
                                        "h": 15,
                                        "i": "1",
                                    },
                                    "panelIndex": "1",
                                    "embeddableConfig": {
                                        "title": "Log Volume Over Time",
                                        "vis": {
                                            "type": "histogram",
                                            "params": {
                                                "grid": {
                                                    "categoryLines": False,
                                                    "style": {"color": "#eee"},
                                                },
                                                "categoryAxes": [
                                                    {
                                                        "id": "CategoryAxis-1",
                                                        "type": "category",
                                                        "position": "bottom",
                                                        "show": True,
                                                        "style": {},
                                                        "scale": {"type": "linear"},
                                                    }
                                                ],
                                                "valueAxes": [
                                                    {
                                                        "id": "ValueAxis-1",
                                                        "name": "LeftAxis-1",
                                                        "type": "value",
                                                        "position": "left",
                                                        "show": True,
                                                        "style": {},
                                                    }
                                                ],
                                                "seriesParams": [
                                                    {
                                                        "data": {"id": "1"},
                                                        "type": "histogram",
                                                        "mode": "stacked",
                                                        "show": True,
                                                        "valueAxis": "ValueAxis-1",
                                                    }
                                                ],
                                            },
                                        },
                                    },
                                }
                            ]
                        ),
                    },
                }
            ],
        }

    async def export_kibana_dashboard(self, output_path: Path) -> bool:
        """Export Kibana dashboard configuration."""
        try:
            dashboard_config = self.create_kibana_dashboard()
            with open(output_path, "w") as f:
                json.dump(dashboard_config, f, indent=2)
            logger.info("Kibana dashboard exported", output_path=str(output_path))
            return True
        except Exception as e:
            logger.error(f"Failed to export Kibana dashboard: {e}")
            return False

    async def setup_elk_integration(self, output_dir: Path) -> bool:
        """Set up ELK integration with configuration files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create index template
            await self.create_index_template()

            # Export Kibana dashboard
            kibana_dashboard_path = output_dir / "kibana-dashboard.json"
            await self.export_kibana_dashboard(kibana_dashboard_path)

            # Create Logstash configuration
            logstash_config = {
                "input": {"beats": {"port": self.logstash.port}},
                "filter": {
                    "if": "[fields][service] == 'icp'",
                    "then": [
                        {
                            "grok": {
                                "match": {
                                    "message": "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{DATA:logger_name} %{GREEDYDATA:message}"
                                }
                            }
                        },
                        {"date": {"match": ["timestamp", "ISO8601"]}},
                    ],
                },
                "output": {
                    "elasticsearch": {
                        "hosts": [
                            f"{self.elasticsearch.host}:{self.elasticsearch.port}"
                        ],
                        "index": f"{self.elasticsearch.index_prefix}-%{{+YYYY.MM.dd}}",
                    }
                },
            }

            logstash_config_path = output_dir / "logstash.conf"
            with open(logstash_config_path, "w") as f:
                json.dump(logstash_config, f, indent=2)

            logger.info("ELK integration setup completed", output_dir=str(output_dir))
            return True

        except Exception as e:
            logger.error(f"Failed to setup ELK integration: {e}")
            return False


def create_elk_config() -> tuple[ElasticsearchConfig, LogstashConfig, KibanaConfig]:
    """Create default ELK configuration."""
    return (ElasticsearchConfig(), LogstashConfig(), KibanaConfig())


async def setup_elk_integration(
    elasticsearch: ElasticsearchConfig,
    logstash: LogstashConfig,
    kibana: KibanaConfig,
    output_dir: Path = Path("monitoring/elk"),
) -> bool:
    """Set up ELK integration with configuration files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create ELK integration
        elk = ELKIntegration(elasticsearch, logstash, kibana)

        # Create index template
        await elk.create_index_template()

        # Export Kibana dashboard
        kibana_dashboard_path = output_dir / "kibana-dashboard.json"
        await elk.export_kibana_dashboard(kibana_dashboard_path)

        # Create Logstash configuration
        logstash_config = {
            "input": {"beats": {"port": logstash.port}},
            "filter": {
                "if": "[fields][service] == 'icp'",
                "then": [
                    {
                        "grok": {
                            "match": {
                                "message": "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{DATA:logger_name} %{GREEDYDATA:message}"
                            }
                        }
                    },
                    {"date": {"match": ["timestamp", "ISO8601"]}},
                ],
            },
            "output": {
                "elasticsearch": {
                    "hosts": [f"{elasticsearch.host}:{elasticsearch.port}"],
                    "index": f"{elasticsearch.index_prefix}-%{{+YYYY.MM.dd}}",
                }
            },
        }

        logstash_config_path = output_dir / "logstash.conf"
        with open(logstash_config_path, "w") as f:
            json.dump(logstash_config, f, indent=2)

        logger.info("ELK integration setup completed", output_dir=str(output_dir))
        return True

    except Exception as e:
        logger.error(f"Failed to setup ELK integration: {e}")
        return False
