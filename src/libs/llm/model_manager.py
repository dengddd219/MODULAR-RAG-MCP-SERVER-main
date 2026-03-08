"""Model Manager for managing multiple LLM configurations and switching.

This module provides functionality to:
- Manage multiple LLM model configurations
- Switch between models dynamically
- Track which model is currently active
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.model_evaluator import ModelEvaluator, MetricsContext

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single LLM model.
    
    Attributes:
        model_id: Unique identifier (e.g., "openai-gpt-4o-mini")
        provider: Provider name (e.g., "openai")
        model_name: Model name (e.g., "gpt-4o-mini")
        display_name: Human-readable name for UI
        description: Description of the model
        is_small_model: Whether this is a small/fast model
        config_override: Optional config overrides for this model
    """
    model_id: str
    provider: str
    model_name: str
    display_name: str
    description: str = ""
    is_small_model: bool = False
    config_override: Dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """Manager for multiple LLM models with switching and evaluation.
    
    This class provides functionality to:
    - Register multiple model configurations
    - Switch between models
    - Track performance metrics
    - Create LLM instances with evaluation
    
    Example:
        >>> manager = ModelManager(settings)
        >>> manager.register_model(ModelConfig(
        ...     model_id="openai-mini",
        ...     provider="openai",
        ...     model_name="gpt-4o-mini",
        ...     display_name="GPT-4o Mini",
        ...     is_small_model=True,
        ... ))
        >>> llm = manager.get_llm("openai-mini")
        >>> with manager.track_call("openai-mini") as metrics:
        ...     response = llm.chat(messages)
    """
    
    def __init__(
        self,
        settings: Settings,
        evaluator: Optional[ModelEvaluator] = None,
    ) -> None:
        """Initialize ModelManager.
        
        Args:
            settings: Application settings.
            evaluator: Optional ModelEvaluator instance.
        """
        self.settings = settings
        self.evaluator = evaluator or ModelEvaluator()
        
        self._models: Dict[str, ModelConfig] = {}
        self._llm_cache: Dict[str, BaseLLM] = {}
        self._current_model_id: Optional[str] = None
        
        # Auto-register default model from settings
        self._register_default_model()
    
    def _register_default_model(self) -> None:
        """Register the default model from settings."""
        try:
            provider = self.settings.llm.provider
            model_name = self.settings.llm.model
            
            model_id = f"{provider}-{model_name}".replace("/", "-").replace(":", "-")
            display_name = f"{provider.title()} {model_name}"
            
            config = ModelConfig(
                model_id=model_id,
                provider=provider,
                model_name=model_name,
                display_name=display_name,
                description=f"Default model from settings.yaml",
            )
            
            self.register_model(config)
            self.set_current_model(model_id)
        except Exception as e:
            logger.warning(f"Failed to register default model: {e}")
    
    def register_model(self, config: ModelConfig) -> None:
        """Register a model configuration.
        
        Args:
            config: ModelConfig instance.
        """
        self._models[config.model_id] = config
        logger.info(f"Registered model: {config.model_id} ({config.display_name})")
    
    def list_models(self) -> List[ModelConfig]:
        """List all registered models.
        
        Returns:
            List of ModelConfig instances.
        """
        return list(self._models.values())
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID.
        
        Args:
            model_id: Model identifier.
            
        Returns:
            ModelConfig if found, None otherwise.
        """
        return self._models.get(model_id)
    
    def set_current_model(self, model_id: str) -> None:
        """Set the current active model.
        
        Args:
            model_id: Model identifier.
            
        Raises:
            ValueError: If model_id is not registered.
        """
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not registered")
        
        self._current_model_id = model_id
        logger.info(f"Switched to model: {model_id}")
    
    def get_current_model_id(self) -> Optional[str]:
        """Get the current active model ID.
        
        Returns:
            Current model ID or None.
        """
        return self._current_model_id
    
    def get_llm(
        self,
        model_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> BaseLLM:
        """Get LLM instance for a model.
        
        Args:
            model_id: Model identifier. If None, uses current model.
            use_cache: Whether to use cached LLM instances.
            
        Returns:
            BaseLLM instance.
            
        Raises:
            ValueError: If model_id is not registered.
        """
        effective_model_id = model_id or self._current_model_id
        if effective_model_id is None:
            raise ValueError("No model selected and no model_id provided")
        
        if effective_model_id not in self._models:
            raise ValueError(f"Model '{effective_model_id}' not registered")
        
        # Check cache
        if use_cache and effective_model_id in self._llm_cache:
            return self._llm_cache[effective_model_id]
        
        # Create new LLM instance
        config = self._models[effective_model_id]
        
        # Create a temporary settings object with overrides
        temp_settings = self._create_settings_with_overrides(config)
        
        # Create LLM via factory
        llm = LLMFactory.create(temp_settings, **config.config_override)
        
        # Cache it
        if use_cache:
            self._llm_cache[effective_model_id] = llm
        
        return llm
    
    def _create_settings_with_overrides(self, config: ModelConfig) -> Settings:
        """Create a settings object with model-specific overrides.
        
        Args:
            config: ModelConfig instance.
            
        Returns:
            Settings object with overrides applied.
        """
        # Create a copy-like settings object
        # We'll modify the llm section temporarily
        class TempSettings:
            def __init__(self, original: Settings, override: ModelConfig):
                self._original = original
                self._override = override
                
                # Copy all other attributes
                for attr in dir(original):
                    if not attr.startswith("_") and attr != "llm":
                        setattr(self, attr, getattr(original, attr))
            
            @property
            def llm(self):
                class TempLLM:
                    def __init__(self, original_llm: Any, override: ModelConfig):
                        self._original = original_llm
                        self._override = override
                    
                    def __getattr__(self, name: str):
                        if name == "provider":
                            return self._override.provider
                        elif name == "model":
                            return self._override.model_name
                        else:
                            return getattr(self._original, name)
                
                return TempLLM(self._original.llm, self._override)
        
        return TempSettings(self.settings, config)
    
    def track_call(
        self,
        model_id: Optional[str] = None,
        query: str = "",
    ) -> MetricsContext:
        """Create a metrics tracking context for a model call.
        
        Args:
            model_id: Model identifier. If None, uses current model.
            query: Input query string.
            
        Returns:
            MetricsContext manager.
        """
        effective_model_id = model_id or self._current_model_id
        if effective_model_id is None:
            raise ValueError("No model selected and no model_id provided")
        
        config = self._models.get(effective_model_id)
        if config is None:
            raise ValueError(f"Model '{effective_model_id}' not registered")
        
        return self.evaluator.track_call(
            model_id=effective_model_id,
            provider=config.provider,
            model_name=config.model_name,
            query=query,
        )
    
    def get_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for models.
        
        Args:
            model_id: Optional filter by model_id.
            
        Returns:
            Dictionary mapping model_id to ModelStats.
        """
        return self.evaluator.get_stats(model_id=model_id)
    
    def clear_cache(self) -> None:
        """Clear the LLM instance cache."""
        self._llm_cache.clear()
        logger.info("Cleared LLM cache")

