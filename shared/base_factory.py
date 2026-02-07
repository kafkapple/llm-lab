"""
Base Factory with Registry Pattern.

Extracted from common patterns in:
- llm_tasks/src/tasks/factory.py (TaskFactory)
- rag/src/embeddings/embedding_factory.py
- rag/src/retriever/retriever_factory.py
- llava_emotion/src/models/model_factory.py
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Type, TypeVar

T = TypeVar("T")


class BaseFactory(ABC):
    """Registry-based Factory ABC.

    Subclass and populate ``_registry`` to create concrete factories.

    Example::

        class TaskFactory(BaseFactory):
            _registry = {
                "emotion": EmotionTask,
                "summary": SummaryTask,
            }

        task = TaskFactory.create("emotion", config=cfg)
    """

    _registry: ClassVar[Dict[str, Type]] = {}

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> Any:
        """Instantiate a registered class by *key*.

        Args:
            key: Lookup key in the registry.
            **kwargs: Passed to the constructor.

        Raises:
            ValueError: If *key* is not registered.
        """
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown key '{key}' for {cls.__name__}. "
                f"Available: [{available}]"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def register(cls, key: str, klass: Type) -> None:
        """Dynamically add an entry to the registry."""
        cls._registry[key] = klass

    @classmethod
    def available_keys(cls) -> list[str]:
        """Return sorted list of registered keys."""
        return sorted(cls._registry)
