import time
import logging
import json
from fastapi import Request, Response

logger = logging.getLogger("api.access")

async def logging_middleware(request: Request, call_next):
    start = time.monotonic()

    # Log request (không log body để tránh log sensitive data)
    logger.info(json.dumps({
        "type": "request",
        "method": request.method,
        "path": request.url.path,
        "client": request.client.host if request.client else "unknown"
    }))

    response: Response = await call_next(request)

    latency_ms = round((time.monotonic() - start) * 1000, 2)

    logger.info(json.dumps({
        "type": "response",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms
    }))

    # Thêm latency vào response header
    response.headers["X-Response-Time-Ms"] = str(latency_ms)

    return response
