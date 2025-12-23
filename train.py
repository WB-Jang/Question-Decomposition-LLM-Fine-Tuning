"""
Main training script for Question Decomposition LLM Fine-Tuning with LoRA
"""
import os
import argparse
from dataclasses import asdict

from src.config import Config
from src.data import QuestionDecompositionDataset, get_tokenize_function
from src.models import load_and_prepare_model
from src.training import QuestionDecompositionTrainer
from src.utils import set_seed, get_device, print_gpu_utilization


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Question Decomposition LLM with LoRA")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to training data (JSON or JSONL)"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data (JSON or JSONL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and models"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Print device info
    device = get_device()
    print_gpu_utilization()
    
    # Load configuration
    config = Config()
    config.model.model_name = args.model_name
    config.model.use_4bit = args.use_4bit
    config.lora.r = args.lora_r
    config.lora.lora_alpha = args.lora_alpha
    config.training.output_dir = args.output_dir
    config.training.num_train_epochs = args.num_epochs
    config.training.per_device_train_batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.max_seq_length = args.max_seq_length
    config.data.train_data_path = args.train_data
    config.data.eval_data_path = args.eval_data
    config.seed = args.seed
    
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(f"Model: {config.model.model_name}")
    print(f"LoRA rank: {config.lora.r}, alpha: {config.lora.lora_alpha}")
    print(f"4-bit quantization: {config.model.use_4bit}")
    print(f"Training epochs: {config.training.num_train_epochs}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print("=" * 80)
    
    # Load and prepare model
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_and_prepare_model(
        model_name=config.model.model_name,
        use_4bit=config.model.use_4bit,
        use_8bit=config.model.use_8bit,
        lora_r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
    )
    
    print_gpu_utilization()
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_loader = QuestionDecompositionDataset(
        train_path=config.data.train_data_path,
        eval_path=config.data.eval_data_path
    )
    dataset = dataset_loader.load_dataset(
        test_size=config.data.test_size,
        max_samples=config.data.max_samples
    )
    
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Eval dataset size: {len(dataset['eval'])}")
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenize_function = get_tokenize_function(tokenizer, config.training.max_seq_length)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    training_config = asdict(config.training)
    trainer = QuestionDecompositionTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        **training_config
    )
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Evaluate model
    print("\n" + "=" * 80)
    print("Evaluating model...")
    print("=" * 80)
    trainer.evaluate()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Model saved to: {config.training.output_dir}/final_model")
    print("=" * 80)
    
    print_gpu_utilization()


if __name__ == "__main__":
    main()
