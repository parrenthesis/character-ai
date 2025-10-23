"""
Resilience patterns for memory system components.

Implements circuit breaker pattern and retry logic for database operations
and LLM calls to ensure system stability under failure conditions.
"""

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreaker:
    """Circuit breaker for resilient service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """Initialize circuit breaker with failure detection settings."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class RetryConfig:
    """Configuration for retry logic."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
    ):
        """Initialize retry configuration."""
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter


def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig,
    expected_exceptions: tuple = (Exception,),
) -> T:
    """Execute function with retry logic and exponential backoff."""
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return func()

        except expected_exceptions as e:
            last_exception = e

            if attempt == config.max_attempts - 1:
                logger.error(f"All {config.max_attempts} attempts failed")
                raise e

            # Calculate delay
            delay = config.base_delay
            if config.exponential_backoff:
                delay = min(delay * (2**attempt), config.max_delay)

            if config.jitter:
                import random

                delay *= 0.5 + random.random() * 0.5  # Add ±25% jitter

            logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
            )
            time.sleep(delay)

    # This should never be reached, but just in case
    raise last_exception  # type: ignore


class ResilientMemoryOperations:
    """Resilient operations for memory system components."""

    def __init__(self) -> None:
        """Initialize with circuit breakers for different operations."""
        # Database operations circuit breaker
        self.db_circuit = CircuitBreaker(
            failure_threshold=3, recovery_timeout=30.0, expected_exception=Exception
        )

        # LLM operations circuit breaker
        self.llm_circuit = CircuitBreaker(
            failure_threshold=2, recovery_timeout=60.0, expected_exception=Exception
        )

        # Retry configuration
        self.retry_config = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0)

    def resilient_db_operation(self, operation: Callable[[], T]) -> T:
        """Execute database operation with circuit breaker and retry."""

        def wrapped_operation() -> T:
            return self.db_circuit.call(operation)

        return retry_with_backoff(
            wrapped_operation, self.retry_config, expected_exceptions=(Exception,)
        )

    def resilient_llm_operation(self, operation: Callable[[], T]) -> T:
        """Execute LLM operation with circuit breaker and retry."""

        def wrapped_operation() -> T:
            return self.llm_circuit.call(operation)

        return retry_with_backoff(
            wrapped_operation, self.retry_config, expected_exceptions=(Exception,)
        )

    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get resilience system statistics."""
        return {
            "database_circuit": self.db_circuit.get_state(),
            "llm_circuit": self.llm_circuit.get_state(),
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay,
                "exponential_backoff": self.retry_config.exponential_backoff,
                "jitter": self.retry_config.jitter,
            },
        }
