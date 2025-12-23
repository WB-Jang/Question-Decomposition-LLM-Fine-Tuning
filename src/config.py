"""
Configuration management for training
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    use_4bit: bool = True
    use_8bit: bool = False
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    evaluation_strategy: str = "steps"
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    max_seq_length: int = 512
    optim: str = "paged_adamw_32bit"
    group_by_length: bool = True
    report_to: str = "wandb"


@dataclass
class DataConfig:
    """Data configuration"""
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    max_samples: Optional[int] = None
    test_size: float = 0.1


@dataclass
class Config:
    """Main configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            seed=config_dict.get("seed", 42)
        )
