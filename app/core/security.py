from jose import JWTError, jwt
from app.config.settings import settings

# Lấy các giá trị này từ file .env hoặc settings.py của bạn
# ĐÂY LÀ VÍ DỤ, BẠN PHẢI THAY BẰNG GIÁ TRỊ CỦA BẠN
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM

def decode_access_token(token: str) -> dict | None:
    try:
        # Giải mã token
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload
    except JWTError:
        print(f"JWTError: Could not validate credentials for token.")
        return None
    except Exception as e:
        print(f"An error occurred during token decoding: {e}")
        return None