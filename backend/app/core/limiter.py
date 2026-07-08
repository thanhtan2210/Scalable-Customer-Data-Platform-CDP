from slowapi import Limiter
from slowapi.util import get_remote_address

def rate_limit_key_func(request) -> str:
    # Key function: prioritize X-API-Key header if available, fallback to IP address
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    return f"ip:{get_remote_address(request)}"

limiter = Limiter(key_func=rate_limit_key_func)
