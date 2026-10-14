import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends

from schemas.models import ProcessTriggerRequest, DetectorFrameRequest, ClassifyIntentRequest
from engines.orchestrator import get_orchestrator
from providers.openai_provider import get_openai_provider
from core.auth import get_current_user, get_optional_current_user

MAX_FRAME_B64_LEN = 2_000_000  # ~1.5 MB image max
logger = logging.getLogger(__name__)
router = APIRouter(tags=["ai"])


@router.post("/v1/process-trigger", response_model=dict)
async def process_trigger(
    request: ProcessTriggerRequest,
    current_user: Optional[dict] = Depends(get_optional_current_user),
):
    """Process confusion trigger with screen capture"""
    try:
        user_id = current_user["id"] if current_user else request.user_id
        orchestrator = get_orchestrator()
        result = await orchestrator.process_confusion_trigger(
            screen_capture=request.screen_capture,
            session_id=request.session_id,
            user_query=request.user_query,
            use_web_search=True,
            user_id=user_id,
        )
        return {
            "session_id": result["session_id"],
            "text_response": result["text_response"],
            "audio_base64": result.get("audio_base64"),
            "screen_analysis": result.get("screen_analysis"),
            "sources_used": result.get("sources_used", []),
            "success": result.get("success", True)
        }
    except Exception as e:
        logger.error(f"Error in process-trigger: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/ask", response_model=dict)
async def ask_question(
    question: str,
    session_id: Optional[str] = None,
    current_user: Optional[dict] = Depends(get_optional_current_user),
):
    """Direct question endpoint"""
    try:
        if not session_id:
            session_id = str(uuid.uuid4())

        user_id = current_user["id"] if current_user else None
        orchestrator = get_orchestrator()
        result = await orchestrator.process_direct_question(
            question=question,
            session_id=session_id,
            user_id=user_id,
        )
        return {
            "session_id": result["session_id"],
            "text_response": result["text_response"],
            "audio_base64": result.get("audio_base64"),
            "sources_used": result.get("sources_used", []),
            "success": result.get("success", True)
        }
    except Exception as e:
        logger.error(f"Error in ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/detector/frame", response_model=dict)
async def detect_frame(
    request: DetectorFrameRequest,
    current_user: dict = Depends(get_current_user),
):
    """Analyze one camera frame and return distraction/confusion states. Requires authentication."""
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    if len(request.image_base64) > MAX_FRAME_B64_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Frame payload too large (max {MAX_FRAME_B64_LEN} characters)",
        )
    try:
        from engines.combined_detector_engine import get_combined_detector_engine
        detector = get_combined_detector_engine()
        user_id = current_user.get("id", "user")
        session_key = f"{user_id}:{request.session_id}" if request.session_id else user_id
        result = await detector.detect_from_base64_async(request.image_base64, session_id=session_key)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in detector/frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/utils/classify-intent", response_model=dict)
async def classify_intent(request: ClassifyIntentRequest):
    """Semantic yes/no classification: does `text` match `intent`?"""
    try:
        provider = get_openai_provider()
        match = await provider.classify_intent(request.text, request.intent)
        return {"match": match}
    except Exception as e:
        logger.error(f"Error in classify-intent: {e}")
        return {"match": False}
