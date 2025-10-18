"""
Cost monitoring system for cloud LLM usage.

Tracks token usage and costs for OpenAI, Anthropic, and other cloud LLM providers
during character creation and other AI operations.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.persistence.base_manager import BaseDataManager
from ...core.persistence.json_manager import JSONRepository

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers for cost monitoring."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class CostEntry:
    """Individual cost entry for an LLM operation."""

    provider: str
    model: str
    operation: str  # e.g., "character_generation", "voice_processing"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str
    character_name: Optional[str] = None
    franchise: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert CostEntry to dictionary for JSON serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
            "character_name": self.character_name,
            "franchise": self.franchise,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEntry":
        """Create CostEntry from dictionary."""
        return cls(
            provider=data["provider"],
            model=data["model"],
            operation=data["operation"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            total_tokens=data["total_tokens"],
            cost_usd=data["cost_usd"],
            timestamp=data["timestamp"],
            character_name=data.get("character_name"),
            franchise=data.get("franchise"),
            metadata=data.get("metadata"),
        )


class LLMCostService(BaseDataManager):
    """Monitor and track LLM usage costs across different providers."""

    def __init__(self, cost_data_dir: Path = Path.cwd() / "data/cost_monitoring"):
        """Initialize cost monitoring system."""
        super().__init__(cost_data_dir, "LLMCostService")

        # Cost tracking files
        self.cost_entries_file = self.storage_path / "cost_entries.json"
        self.monthly_summary_file = self.storage_path / "monthly_summary.json"

        # Don't load data during instantiation - load when actually needed
        self.cost_entries: List[CostEntry] = []
        self.monthly_summary: Dict[str, Any] = {}

        # Provider cost rates (per 1K tokens)
        self.cost_rates = {
            LLMProvider.OPENAI: {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
                "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            },
            LLMProvider.ANTHROPIC: {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            },
            LLMProvider.GOOGLE: {
                "gemini-pro": {"input": 0.0005, "output": 0.0015},
                "gemini-pro-vision": {"input": 0.0005, "output": 0.0015},
            },
        }

    async def initialize(self) -> None:
        """Initialize the cost monitor."""
        await super().initialize()

    async def _load_data(self) -> None:
        """Load cost monitoring data."""
        # Load cost entries
        cost_entries_data = JSONRepository.load_json(self.cost_entries_file, {})
        if isinstance(cost_entries_data, dict) and "entries" in cost_entries_data:
            self.cost_entries = [
                CostEntry.from_dict(entry) for entry in cost_entries_data["entries"]
            ]
        else:
            self.cost_entries = []

        # Load monthly summary
        self.monthly_summary = JSONRepository.load_json(self.monthly_summary_file, {})

        logger.info(f"Loaded {len(self.cost_entries)} cost entries and monthly summary")

    async def _save_data(self) -> None:
        """Save cost monitoring data."""
        # Save cost entries
        cost_entries_data = {
            "entries": [entry.to_dict() for entry in self.cost_entries]
        }
        success = JSONRepository.save_json(self.cost_entries_file, cost_entries_data)
        if not success:
            logger.error("Failed to save cost entries")

        # Save monthly summary
        success = JSONRepository.save_json(
            self.monthly_summary_file, self.monthly_summary
        )
        if not success:
            logger.error("Failed to save monthly summary")

    def calculate_cost(
        self, provider: LLMProvider, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a given provider, model, and token usage."""
        try:
            provider_rates = self.cost_rates.get(provider)
            if not provider_rates:
                logger.warning(f"No cost rates found for provider: {provider}")
                return 0.0

            model_rates = provider_rates.get(model)
            if not model_rates:
                logger.warning(f"No cost rates found for model: {model}")
                return 0.0

            input_cost = (input_tokens / 1000) * model_rates["input"]
            output_cost = (output_tokens / 1000) * model_rates["output"]

            return input_cost + output_cost

        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0

    def record_usage(
        self,
        provider: LLMProvider,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        character_name: Optional[str] = None,
        franchise: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostEntry:
        """Record LLM usage and calculate cost."""
        try:
            # Calculate cost
            cost_usd = self.calculate_cost(provider, model, input_tokens, output_tokens)

            # Create cost entry
            cost_entry = CostEntry(
                provider=provider.value,
                model=model,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost_usd,
                timestamp=datetime.now(timezone.utc).isoformat(),
                character_name=character_name,
                franchise=franchise,
                metadata=metadata or {},
            )

            # Add to entries
            self.cost_entries.append(cost_entry)

            # Update monthly summary
            self._update_monthly_summary(cost_entry)

            # Note: Auto-save will be handled by BaseDataManager

            logger.info(
                f"Recorded {operation} usage: {cost_entry.total_tokens} tokens, "
                f"${cost_usd:.4f}"
            )
            return cost_entry

        except Exception as e:
            logger.error(f"Error recording usage: {e}")
            raise

    def _update_monthly_summary(self, cost_entry: CostEntry) -> None:
        """Update monthly cost summary."""
        try:
            # Get current month key
            timestamp = datetime.fromisoformat(
                cost_entry.timestamp.replace("Z", "+00:00")
            )
            month_key = timestamp.strftime("%Y-%m")

            if month_key not in self.monthly_summary:
                self.monthly_summary[month_key] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "operations": {},
                    "providers": {},
                    "franchises": {},
                }

            month_data = self.monthly_summary[month_key]

            # Update totals
            month_data["total_cost"] += cost_entry.cost_usd
            month_data["total_tokens"] += cost_entry.total_tokens

            # Update operation stats
            if cost_entry.operation not in month_data["operations"]:
                month_data["operations"][cost_entry.operation] = {
                    "count": 0,
                    "cost": 0.0,
                }
            month_data["operations"][cost_entry.operation]["count"] += 1
            month_data["operations"][cost_entry.operation][
                "cost"
            ] += cost_entry.cost_usd

            # Update provider stats
            if cost_entry.provider not in month_data["providers"]:
                month_data["providers"][cost_entry.provider] = {
                    "cost": 0.0,
                    "tokens": 0,
                }
            month_data["providers"][cost_entry.provider]["cost"] += cost_entry.cost_usd
            month_data["providers"][cost_entry.provider][
                "tokens"
            ] += cost_entry.total_tokens

            # Update franchise stats
            if cost_entry.franchise:
                if cost_entry.franchise not in month_data["franchises"]:
                    month_data["franchises"][cost_entry.franchise] = {
                        "cost": 0.0,
                        "tokens": 0,
                    }
                month_data["franchises"][cost_entry.franchise][
                    "cost"
                ] += cost_entry.cost_usd
                month_data["franchises"][cost_entry.franchise][
                    "tokens"
                ] += cost_entry.total_tokens

        except Exception as e:
            logger.error(f"Error updating monthly summary: {e}")

    def get_cost_summary(self, month: Optional[str] = None) -> Dict[str, Any]:
        """Get cost summary for a specific month or all time."""
        if month:
            return self.monthly_summary.get(month, {})  # type: ignore

        # Return all-time summary
        total_cost = sum(entry.cost_usd for entry in self.cost_entries)
        total_tokens = sum(entry.total_tokens for entry in self.cost_entries)

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_entries": len(self.cost_entries),
            "monthly_breakdown": self.monthly_summary,
        }

    def get_franchise_costs(
        self, franchise: str, month: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost breakdown for a specific franchise."""
        franchise_entries = [
            entry for entry in self.cost_entries if entry.franchise == franchise
        ]

        if month:
            # Filter by month
            franchise_entries = [
                entry
                for entry in franchise_entries
                if entry.timestamp.startswith(month)
            ]

        total_cost = sum(entry.cost_usd for entry in franchise_entries)
        total_tokens = sum(entry.total_tokens for entry in franchise_entries)

        # Group by operation
        operations = {}
        for entry in franchise_entries:
            operation = entry.operation
            if operation not in operations:
                operations[operation] = {"count": 0, "cost": 0.0, "tokens": 0}
            operations[operation]["count"] += 1
            operations[operation]["cost"] += entry.cost_usd
            operations[operation]["tokens"] += entry.total_tokens

        return {
            "franchise": franchise,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_entries": len(franchise_entries),
            "operations": operations,
        }

    def get_provider_costs(
        self, provider: str, month: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost breakdown for a specific provider."""
        provider_entries = [
            entry for entry in self.cost_entries if entry.provider == provider
        ]

        if month:
            # Filter by month
            provider_entries = [
                entry for entry in provider_entries if entry.timestamp.startswith(month)
            ]

        total_cost = sum(entry.cost_usd for entry in provider_entries)
        total_tokens = sum(entry.total_tokens for entry in provider_entries)

        # Group by model
        models = {}
        for entry in provider_entries:
            model = entry.model
            if model not in models:
                models[model] = {"count": 0, "cost": 0.0, "tokens": 0}
            models[model]["count"] += 1
            models[model]["cost"] += entry.cost_usd
            models[model]["tokens"] += entry.total_tokens

        return {
            "provider": provider,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_entries": len(provider_entries),
            "models": models,
        }

    def export_cost_data(self, output_file: Optional[Path] = None) -> Path:
        """Export cost data to JSON file."""
        # Create directory if it doesn't exist
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True, exist_ok=True)

        if not output_file:
            output_file = (
                self.storage_path
                / f"cost_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "cost_entries": [entry.to_dict() for entry in self.cost_entries],
            "monthly_summary": self.monthly_summary,
            "cost_rates": {
                provider.value: rates for provider, rates in self.cost_rates.items()
            },
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported cost data to {output_file}")
        return output_file
