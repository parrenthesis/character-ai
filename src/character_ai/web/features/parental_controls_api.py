"""
Parental controls API endpoints.

Provides endpoints for parental control management, child safety monitoring,
and usage reporting for child-focused deployments.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...features.parental_controls import (
    ChildAgeGroup,
    ParentalControlService,
    create_parental_control_manager,
)
from ..security_deps import get_security_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/parental-controls", tags=["Parental Controls"])


# Pydantic models for API
class CreateProfileRequest(BaseModel):
    """Request model for creating a parental control profile."""

    child_id: str = Field(..., description="Unique identifier for the child")
    child_name: str = Field(..., description="Display name for the child")
    age_group: str = Field(
        ..., description="Age group: toddler, preschool, elementary, teen"
    )
    parent_id: str = Field(..., description="Parent/guardian identifier")


class UpdateProfileRequest(BaseModel):
    """Request model for updating a parental control profile."""

    content_filter_level: Optional[str] = Field(
        None, description="Content filter level"
    )
    safety_level: Optional[str] = Field(None, description="Safety level")
    monitoring_enabled: Optional[bool] = Field(None, description="Enable monitoring")
    alert_parents: Optional[bool] = Field(
        None, description="Alert parents on violations"
    )
    time_limits: Optional[List[Dict[str, Any]]] = Field(
        None, description="Time limit configurations"
    )


class ContentSafetyRequest(BaseModel):
    """Request model for content safety checking."""

    content: str = Field(..., description="Content to check for safety")
    character_id: Optional[str] = Field(None, description="Character ID if applicable")


class TimeLimitRequest(BaseModel):
    """Request model for time limit checking."""

    child_id: str = Field(..., description="Child identifier")


class UsageReportResponse(BaseModel):
    """Response model for usage reports."""

    child_id: str = Field(..., description="Child identifier")
    child_name: str = Field(..., description="Child display name")
    age_group: str = Field(..., description="Child age group")
    usage_stats: Dict[str, Any] = Field(..., description="Usage statistics")
    recent_alerts: List[Dict[str, Any]] = Field(..., description="Recent safety alerts")

    profile_settings: Dict[str, Any] = Field(..., description="Profile settings")


class ParentDashboardResponse(BaseModel):
    """Response model for parent dashboard."""

    parent_id: str = Field(..., description="Parent identifier")
    children: List[UsageReportResponse] = Field(..., description="Children's data")
    total_alerts: int = Field(..., description="Total unresolved alerts")
    active_sessions: int = Field(..., description="Active monitoring sessions")


class SafetyAlertResponse(BaseModel):
    """Response model for safety alerts."""

    alert_id: str = Field(..., description="Alert identifier")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    timestamp: float = Field(..., description="Alert timestamp")
    child_id: str = Field(..., description="Child identifier")
    character_id: Optional[str] = Field(None, description="Character identifier")
    resolved: bool = Field(..., description="Whether alert is resolved")


class ContentSafetyResponse(BaseModel):
    """Response model for content safety checking."""

    is_safe: bool = Field(..., description="Whether content is safe")
    safety_level: str = Field(..., description="Safety classification level")
    confidence: float = Field(..., description="Confidence score")
    blocked_categories: List[str] = Field(..., description="Blocked content categories")

    alert_created: bool = Field(..., description="Whether a safety alert was created")


class SessionRequest(BaseModel):
    """Request model for session management."""

    child_id: str = Field(..., description="Child identifier")
    device_id: str = Field(..., description="Device identifier")


class InteractionRequest(BaseModel):
    """Request model for recording interactions."""

    child_id: str = Field(..., description="Child identifier")
    character_id: str = Field(..., description="Character identifier")
    content: str = Field(..., description="Interaction content")


def get_parental_control_manager() -> ParentalControlService:
    """Dependency to get parental control manager instance."""
    return create_parental_control_manager()


@router.post("/profiles", response_model=Dict[str, str])
async def create_profile(
    request: CreateProfileRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, str]:
    """Create a new parental control profile for a child."""
    try:
        # Validate age group
        try:
            age_group = ChildAgeGroup(request.age_group)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid age group: {request.age_group}"
            )

        # Create profile
        profile = manager.create_profile(
            child_id=request.child_id,
            parent_id=request.parent_id,
            child_name=request.child_name,
            age_group=age_group,
        )

        return {
            "status": "success",
            "message": f"Parental control profile created for {request.child_name}",
            "child_id": profile.child_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create parental control profile: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create profile: {str(e)}"
        )


@router.get("/profiles/{child_id}", response_model=Dict[str, Any])
async def get_profile(
    child_id: str,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, Any]:
    """Get parental control profile for a child."""
    try:
        # manager is now injected via dependency

        profile = manager.get_profile(child_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        return profile.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get parental control profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")


@router.put("/profiles/{child_id}", response_model=Dict[str, str])
async def update_profile(
    child_id: str,
    request: UpdateProfileRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, str]:
    """Update parental control profile for a child."""
    try:
        # manager is now injected via dependency

        # Prepare updates
        updates: Dict[str, Any] = {}
        if request.content_filter_level:
            updates["content_filter"] = {"level": request.content_filter_level}
        if request.safety_level:
            updates["safety_level"] = request.safety_level
        if request.monitoring_enabled is not None:
            updates["monitoring_enabled"] = request.monitoring_enabled
        if request.alert_parents is not None:
            updates["alert_parents"] = request.alert_parents
        if request.time_limits:
            updates["time_limits"] = request.time_limits

        # Update profile
        success = manager.update_profile(child_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail="Profile not found")

        return {"status": "success", "message": f"Profile updated for child {child_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update parental control profile: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update profile: {str(e)}"
        )


@router.post("/content-safety", response_model=ContentSafetyResponse)
async def check_content_safety(
    child_id: str,
    request: ContentSafetyRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> ContentSafetyResponse:
    """Check if content is safe for a child."""
    try:
        # manager is now injected via dependency

        # Check content safety
        is_safe, safety_result, alert = manager.check_content_safety(
            child_id=child_id,
            text=request.content,
            character_id=request.character_id or "unknown",
        )

        return ContentSafetyResponse(
            is_safe=is_safe,
            safety_level=safety_result.level.value,
            confidence=safety_result.confidence,
            blocked_categories=safety_result.categories,
            alert_created=alert is not None,
        )

    except Exception as e:
        logger.error(f"Failed to check content safety: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check content safety: {str(e)}"
        )


@router.post("/time-limits", response_model=Dict[str, Any])
async def check_time_limits(
    request: TimeLimitRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, Any]:
    """Check if child has exceeded time limits."""
    try:
        # manager is now injected via dependency

        # Check time limits
        can_continue, alert = manager.check_time_limits(request.child_id)

        return {
            "can_continue": can_continue,
            "alert_created": alert is not None,
            "alert": (
                {
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity,
                    "message": alert.message,
                }
                if alert
                else None
            ),
        }

    except Exception as e:
        logger.error(f"Failed to check time limits: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check time limits: {str(e)}"
        )


@router.post("/sessions/start", response_model=Dict[str, str])
async def start_session(
    request: SessionRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, str]:
    """Start a monitoring session for a child."""
    try:
        # manager is now injected via dependency

        # Start session
        success = manager.start_session(request.child_id, request.device_id)
        if not success:
            return {
                "status": "blocked",
                "message": "Session blocked due to time limits or other restrictions",
            }

        return {
            "status": "success",
            "message": f"Monitoring session started for child {request.child_id}",
        }

    except Exception as e:
        logger.error(f"Failed to start monitoring session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start session: {str(e)}"
        )


@router.post("/sessions/end", response_model=Dict[str, str])
async def end_session(
    child_id: str,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, str]:
    """End a monitoring session for a child."""
    try:
        # manager is now injected via dependency

        # End session
        manager.end_session(child_id)

        return {
            "status": "success",
            "message": f"Monitoring session ended for child {child_id}",
        }

    except Exception as e:
        logger.error(f"Failed to end monitoring session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.post("/interactions", response_model=Dict[str, Any])
async def record_interaction(
    request: InteractionRequest,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, Any]:
    """Record a child's interaction with a character."""
    try:
        # manager is now injected via dependency

        # Record interaction
        is_safe, alert = manager.record_interaction(
            child_id=request.child_id,
            character_id=request.character_id,
            content=request.content,
        )

        return {
            "is_safe": is_safe,
            "alert_created": alert is not None,
            "alert": (
                {
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity,
                    "message": alert.message,
                }
                if alert
                else None
            ),
        }

    except Exception as e:
        logger.error(f"Failed to record interaction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record interaction: {str(e)}"
        )


@router.get("/usage-reports/{child_id}", response_model=UsageReportResponse)
async def get_usage_report(
    child_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to include"),
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> UsageReportResponse:
    """Get usage report for a child."""
    try:
        # manager is now injected via dependency

        # Get usage report
        report = manager.get_usage_report(child_id)
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])

        return UsageReportResponse(**report)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get usage report: {str(e)}"
        )


@router.get("/dashboard/{parent_id}", response_model=ParentDashboardResponse)
async def get_parent_dashboard(
    parent_id: str,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> ParentDashboardResponse:
    """Get parent dashboard with all children's data."""
    try:
        # manager is now injected via dependency

        # Get parent dashboard
        dashboard = manager.get_parent_dashboard(parent_id)

        return ParentDashboardResponse(**dashboard)

    except Exception as e:
        logger.error(f"Failed to get parent dashboard: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get parent dashboard: {str(e)}"
        )


@router.get("/alerts", response_model=List[SafetyAlertResponse])
async def get_safety_alerts(
    child_id: Optional[str] = Query(None, description="Filter by child ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> List[SafetyAlertResponse]:
    """Get safety alerts with optional filtering."""
    try:
        # manager is now injected via dependency

        # Filter alerts
        alerts = manager.alerts
        if child_id:
            alerts = [alert for alert in alerts if alert.child_id == child_id]
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.acknowledged == resolved]

        # Convert to response format
        alert_responses = []
        for alert in alerts:
            alert_responses.append(
                SafetyAlertResponse(
                    alert_id=str(id(alert)),  # Use object id as alert_id
                    alert_type=alert.alert_type.value,
                    severity=alert.severity,
                    message=alert.message,
                    timestamp=alert.timestamp,
                    child_id=alert.child_id or "unknown",
                    character_id=alert.metadata.get("character_id", "unknown"),
                    resolved=alert.acknowledged,
                )
            )

        return alert_responses

    except Exception as e:
        logger.error(f"Failed to get safety alerts: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get safety alerts: {str(e)}"
        )


@router.put("/alerts/{alert_id}/resolve", response_model=Dict[str, str])
async def resolve_alert(
    alert_id: str,
    current_device: str = Query(..., description="Device ID"),
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, str]:
    """Mark a safety alert as resolved."""
    try:
        # manager is now injected via dependency

        # Find and resolve alert
        alert_found = False
        for alert in manager.alerts:
            if str(id(alert)) == alert_id:
                alert.acknowledged = True
                alert_found = True
                break

        if not alert_found:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Save alerts
        manager._save_alerts()

        return {"status": "success", "message": f"Alert {alert_id} marked as resolved"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to resolve alert: {str(e)}"
        )


@router.get("/health")
async def parental_controls_health_check(
    security: Any = Depends(get_security_middleware),
    manager: ParentalControlService = Depends(get_parental_control_manager),
) -> Dict[str, Any]:
    """Health check for parental controls system."""
    try:
        # manager is now injected via dependency

        return {
            "status": "healthy",
            "total_profiles": len(manager.profiles),
            "total_alerts": len(manager.alerts),
            "active_sessions": len(manager.active_sessions),
            "storage_path": str(manager.storage_path),
        }

    except Exception as e:
        logger.error(f"Parental controls health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
