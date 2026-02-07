"""
Common model loading utilities.

Extracted from shared patterns across:
- llm_tasks: 8-bit quantization via BitsAndBytes
- rag: 8-bit quantization + device_map="auto"
- llava_emotion: 4-bit NF4 quantization via BitsAndBytes
"""

from typing import Any, Dict, Optional

import torch


def get_quantization_config(
    bits: int = 4,
    compute_dtype: torch.dtype = torch.float16,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
) -> Any:
    """Build a BitsAndBytesConfig for the requested precision.

    Args:
        bits: 4 or 8.
        compute_dtype: Compute dtype for 4-bit (ignored for 8-bit).
        use_double_quant: Enable double quantization (4-bit only).
        quant_type: Quantization type, e.g. ``"nf4"`` (4-bit only).

    Returns:
        A ``BitsAndBytesConfig`` instance.
    """
    from transformers import BitsAndBytesConfig

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type=quant_type,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}. Use 4 or 8.")


def load_model_with_quantization(
    model_name_or_path: str,
    bits: int = 8,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load a HuggingFace causal LM with quantization.

    Combines the loading patterns used across the mono-repo projects.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        bits: Quantization bits (4 or 8).
        device_map: Device placement strategy.
        torch_dtype: Model dtype.
        trust_remote_code: Whether to trust remote code.
        extra_kwargs: Additional kwargs passed to ``from_pretrained``.

    Returns:
        The loaded model.
    """
    from transformers import AutoModelForCausalLM

    quant_config = get_quantization_config(bits=bits)

    kwargs: Dict[str, Any] = {
        "quantization_config": quant_config,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
