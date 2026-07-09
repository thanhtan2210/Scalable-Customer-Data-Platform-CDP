import mlflow
import threading
import logging
from typing import Any, Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("cdp.serving.model_loader")

class ModelCache:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._last_loaded: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self.ttl = timedelta(minutes=10)

    def _detect_model_type(self, model) -> str:
        # MTL model là Pipeline chứa MTLChurnModel
        # Standard model là Pipeline chứa sklearn estimator
        try:
            try:
                from backend.app.core.training.mtl_trainer import MTLChurnModel
            except ImportError:
                from app.core.training.mtl_trainer import MTLChurnModel
                
            # Kiểm tra step cuối của pipeline
            last_step = model.steps[-1][1] if hasattr(model, 'steps') else model
            if isinstance(last_step, MTLChurnModel):
                return "mtl_sklearn"
            return "sklearn"
        except Exception:
            return "sklearn"

    def _is_expired(self, entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        loaded_at = entry.get("loaded_at")
        if loaded_at is None or loaded_at == 0:
            return False
        import time
        return (time.time() - loaded_at) > self.ttl.total_seconds()

    def get_or_load(self, cache_key: str, model_uri: str) -> Tuple[Any, str]:
        """Tải mô hình sử dụng Double-Checked Locking để tránh race condition."""
        # 1. Check ngoài lock (fast path)
        if cache_key in self._models:
            entry = self._models[cache_key]
            if not self._is_expired(entry):
                if isinstance(entry, dict):
                    return entry["model"], entry["model_type"]
                return entry, "sklearn"

        # 2. Acquire lock (slow path)
        with self._lock:
            # Check lại trong lock (tránh double-load)
            if cache_key in self._models:
                entry = self._models[cache_key]
                if not self._is_expired(entry):
                    if isinstance(entry, dict):
                        return entry["model"], entry["model_type"]
                    return entry, "sklearn"

            logger.info(f"📥 Loading model from registry: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            model_type = self._detect_model_type(model)
            
            import time
            self._models[cache_key] = {
                "model": model,
                "model_type": model_type,
                "loaded_at": time.time(),
                "model_uri": model_uri
            }
            self._last_loaded[cache_key] = datetime.utcnow()
            return model, model_type

    def get_model(self, cache_key: str):
        """Trả về (model, model_type) tuple."""
        if cache_key in self._models:
            entry = self._models[cache_key]
            if not self._is_expired(entry):
                if isinstance(entry, dict):
                    return (entry["model"],
                            entry["model_type"])
                else:
                    # Backward compat với cache cũ
                    return (entry, "sklearn")
        return None, None

    def load_model(self, model_uri: str, cache_key: str):
        return self.get_or_load(cache_key, model_uri)

    def invalidate(self,
                   dataset_id: str = None,
                   model_uri: str = None) -> int:
        """
        Xóa cache entries.
        Trả về số entries đã xóa.
        """
        with self._lock:
            if not dataset_id and not model_uri:
                # Xóa toàn bộ cache
                count = len(self._models)
                self._models.clear()
                self._last_loaded.clear()
                return count

            keys_to_delete = []
            for key in list(self._models.keys()):
                entry = self._models[key]
                if isinstance(entry, dict):
                    cached_uri = entry.get("model_uri", "")
                else:
                    cached_uri = ""

                if model_uri and model_uri in key:
                    keys_to_delete.append(key)
                elif dataset_id and dataset_id in key:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._models[key]
                self._last_loaded.pop(key, None)

            return len(keys_to_delete)

model_cache = ModelCache()
