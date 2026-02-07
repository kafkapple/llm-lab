"""
Hydra / OmegaConf configuration helpers.

Extracted from common patterns in llm_tasks and rag projects,
both of which use Hydra for config management.
"""

from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


def load_hydra_config(
    config_path: str = "./configs",
    config_name: str = "config",
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Programmatically compose a Hydra config without the decorator.

    Useful for notebooks or scripts that cannot use ``@hydra.main``.

    Args:
        config_path: Relative or absolute path to the config directory.
        config_name: Name of the root config file (without ``.yaml``).
        overrides: Optional list of Hydra-style overrides.

    Returns:
        Composed ``DictConfig``.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import os

    abs_config_path = os.path.abspath(config_path)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=abs_config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def get_nested_config(cfg: DictConfig, dotpath: str, default: Any = None) -> Any:
    """Safely retrieve a nested value using dot notation.

    Args:
        cfg: OmegaConf DictConfig.
        dotpath: Dot-separated key path, e.g. ``"model.quantization.precision"``.
        default: Value returned when the path does not exist.

    Returns:
        The resolved value or *default*.
    """
    try:
        return OmegaConf.select(cfg, dotpath, default=default)
    except Exception:
        return default
