import sys
from slowapi import Limiter
from slowapi.util import get_remote_address


def rate_limit_key_func(request) -> str:
    # Key function: prioritize X-API-Key header if available, fallback to IP address
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    return f"ip:{get_remote_address(request)}"


# Disable rate limit if running under pytest
is_testing = "pytest" in sys.modules
limiter = Limiter(key_func=rate_limit_key_func, enabled=not is_testing)
