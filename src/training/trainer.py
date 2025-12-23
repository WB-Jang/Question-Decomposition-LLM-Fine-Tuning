"""
Training utilities and trainer
"""
import os
from typing import Optional
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import DatasetDict


class QuestionDecompositionTrainer:
    """Custom trainer for question decomposition task"""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./outputs",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_grad_norm: float = 0.3,
        weight_decay: float = 0.001,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine",
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100,
        eval_strategy: str = "steps",
        save_total_limit: int = 3,
        fp16: bool = False,
        bf16: bool = True,
        optim: str = "paged_adamw_32bit",
        group_by_length: bool = True,
        report_to: str = "wandb",
    ):
        """
        Initialize trainer
        
        Args:
            model: PEFT model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory
            ... (other training arguments)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy=evaluation_strategy,
            save_total_limit=save_total_limit,
            fp16=fp16,
            bf16=bf16,
            optim=optim,
            group_by_length=group_by_length,
            report_to=report_to,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )
        
        # Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )
    
    def train(self):
        """Start training"""
        print("Starting training...")
        self.trainer.train()
        print("Training completed!")
        
        # Save final model
        self.save_model()
    
    def evaluate(self):
        """Evaluate model"""
        if self.eval_dataset is None:
            print("No evaluation dataset provided!")
            return
        
        print("Evaluating model...")
        metrics = self.trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save model and tokenizer"""
        if output_dir is None:
            output_dir = os.path.join(self.training_args.output_dir, "final_model")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")


def create_trainer(
    model,
    tokenizer,
    dataset: DatasetDict,
    training_config: dict,
) -> QuestionDecompositionTrainer:
    """
    Create trainer from config
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        dataset: DatasetDict with train and eval splits
        training_config: Training configuration dictionary
        
    Returns:
        QuestionDecompositionTrainer instance
    """
    return QuestionDecompositionTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        **training_config
    )
