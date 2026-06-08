import mlflow
import threading
from typing import Any, Dict
from datetime import datetime, timedelta

class ModelCache:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._last_loaded: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self.ttl = timedelta(minutes=10)

    def get_model(self, model_uri: str) -> Any:
        with self._lock:
            # Simple TTL-based cache / Manual reload logic can be added
            if model_uri in self._models:
                if datetime.utcnow() - self._last_loaded[model_uri] < self.ttl:
                    return self._models[model_uri]
            
            print(f"📥 Loading model from registry: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            self._models[model_uri] = model
            self._last_loaded[model_uri] = datetime.utcnow()
            return model

model_cache = ModelCache()
