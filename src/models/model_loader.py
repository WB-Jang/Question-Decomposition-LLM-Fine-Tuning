"""
Model loading and preparation utilities
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Tuple, Optional


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_8bit: bool = False,
    device_map={"": 0},
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model and tokenizer with quantization
    
    Args:
        model_name: Hugging Face model identifier
        use_4bit: Use 4-bit quantization
        use_8bit: Use 8-bit quantization
        device_map: 0 번 GPU 에 모든 레이어를 로드
        trust_remote_code: Trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Configure quantization
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        max_memory={0: "7500MiB", "cpu": "30GiB"}
    )

    # Disable cache for training
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def prepare_model_for_lora(
    model: AutoModelForCausalLM,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> AutoModelForCausalLM:
    """
    Prepare model for LoRA fine-tuning
    
    Args:
        model: Base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA
        bias: Bias configuration
        task_type: Task type
        
    Returns:
        PEFT model with LoRA
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_and_prepare_model(
    model_name: str,
    use_4bit: bool = True,
    use_8bit: bool = False,
    device_map: Optional[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with tokenizer and prepare for LoRA training
    
    Returns:
        Tuple of (peft_model, tokenizer)
    """
    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        device_map=device_map,
    )
    
    # Prepare for LoRA
    model = prepare_model_for_lora(
        model=model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    return model, tokenizer
