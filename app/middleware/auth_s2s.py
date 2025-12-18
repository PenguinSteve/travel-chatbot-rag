import os
from fastapi import Header, HTTPException, status
from app.config.settings import settings

# Lấy key từ biến môi trường (đảm bảo file .env của bạn đã có PYTHON_INTERNAL_API_KEY)
INTERNAL_API_KEY = os.getenv("PYTHON_INTERNAL_API_KEY")

async def verify_internal_api_key(
    x_api_key: str = Header(..., alias="Internal-API-Key") 
):
    if not settings.INTERNAL_API_KEY:
        # Fallback an toàn nếu chưa cấu hình env
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Server Internal API Key configuration is missing."
        )

    if x_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Internal API Key"
        )
    
    return True