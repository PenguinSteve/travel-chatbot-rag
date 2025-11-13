from fastapi import Header, HTTPException, Depends
from app.core.security import decode_access_token   


def get_user_payload_optional(
    authorization: str | None = Header(None)
) -> dict | None:

    if authorization is None:
        # Không có header -> người dùng khách
        return None
        
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401, 
            detail="Invalid Authorization header format. Expected 'Bearer <token>'."
        )
    
    token = parts[1]
    
    # Giải mã token
    payload = decode_access_token(token)

    # Token không hợp lệ
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token."
        )
    
    # payload sẽ là dict (nếu thành công) hoặc None (nếu thất bại)
    return payload

def get_current_user_payload_strict(
    user_payload: dict | None = Depends(get_user_payload_optional)
) -> dict:
    if user_payload is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. This feature is for logged-in users only."
        )
    return user_payload