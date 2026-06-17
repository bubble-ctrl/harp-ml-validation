"""
shared_model.py — Singleton model loader with 4-bit quantization support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODEL_NAME, USE_4BIT, FALLBACK_DTYPE, DEVICE_MAP

_MODEL = None
_TOKENIZER = None

def get_model():
    """Load model once and cache it. Uses 4-bit if enabled."""
    global _MODEL, _TOKENIZER

    if _MODEL is None:
        print(f"[Shared] Loading {MODEL_NAME}...")

        # 4-bit quantization config (only if enabled)
        if USE_4BIT:
            print("[Shared] Using 4-bit quantization (bitsandbytes).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            quantization_config = bnb_config
            torch_dtype = None  # quantization handles dtype
        else:
            print(f"[Shared] Using full precision ({FALLBACK_DTYPE}).")
            quantization_config = None
            torch_dtype = FALLBACK_DTYPE

        # Load tokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token

        # Load model
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=DEVICE_MAP,
            trust_remote_code=True,
        )

        print("[Shared] Model ready.")

    return _MODEL, _TOKENIZER


# Quick test if run directly
if __name__ == "__main__":
    model, tokenizer = get_model()
    print(f"Model loaded on {next(model.parameters()).device}")
    print(f"Vocabulary size: {len(tokenizer)}")