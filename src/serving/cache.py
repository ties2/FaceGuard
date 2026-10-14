from typing import Any, Protocol


class ModelCache(Protocol):
    """
    Abstraction for Model Caching layer.
    """

    def get(self, model_name: str, version: str) -> Any: ...
    def set(self, model_name: str, version: str, model: Any) -> None: ...
    def contains(self, model_name: str, version: str) -> bool: ...
    def remove(self, model_name: str, version: str) -> None: ...
    def invalidate(self, model_name: str, version: str) -> None: ...
    def clear(self) -> None: ...


class InMemoryModelCache:
    """
    In-memory implementation of ModelCache.
    """

    def __init__(self):
        # dictionary structure with key (model_name,version)
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, model_name: str, version: str) -> Any:
        return self._cache.get((model_name, version))

    def set(self, model_name: str, version: str, model: Any) -> None:
        self._cache[(model_name, version)] = model

    def contains(self, model_name: str, version: str) -> bool:
        return (model_name, version) in self._cache

    def remove(self, model_name: str, version: str) -> None:
        if self.contains(model_name, version):
            del self._cache[(model_name, version)]

    def invalidate(self, model_name: str, version: str) -> None:
        """
        Alias for remove. Used specifically during model promotion
        to explicitly state the domain intent of invalidating old cache.
        """
        self.remove(model_name, version)

    def clear(self) -> None:
        self._cache.clear()


# one sample (Singleton) for use in whole application
model_cache = InMemoryModelCache()
