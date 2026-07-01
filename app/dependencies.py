from fastapi import HTTPException, Request, status
from app.services.model_service import ModelService
def get_model_service(request: Request) -> ModelService: 
    service: ModelService | None =getattr(
        request.app.state, 
        "model_service",
        None,
    )
    if service is None: 
        raise HTTPException (
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service has not been initialize"
        )
    if not service.is_ready: 
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=service.loading_error or "Model is not ready"
        )
    return service