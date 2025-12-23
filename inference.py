"""
Inference script for trained model
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_name: str, adapter_path: str, device: str = "cuda"):
    """Load base model with LoRA adapter"""
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge LoRA weights for faster inference
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_prompt(question: str) -> str:
    """Create prompt for inference"""
    return f"""### Instruction:
복잡한 질문을 단순한 여러 개의 하위 질문으로 분해하세요.

### Question:
{question}

### Sub-questions:
"""


def decompose_question(
    model,
    tokenizer,
    question: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Decompose a complex question into sub-questions"""
    prompt = create_prompt(question)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to decompose"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    args = parser.parse_args()
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, tokenizer = load_model(args.base_model, args.adapter_path, device)
    print("Model loaded successfully!\n")
    
    if args.interactive:
        # Interactive mode
        print("=" * 80)
        print("Interactive Mode - Type 'quit' to exit")
        print("=" * 80)
        
        while True:
            question = input("\n질문을 입력하세요 (종료하려면 'quit' 입력): ")
            if question.lower() == 'quit':
                break
            
            if not question.strip():
                continue
            
            print("\n처리 중...")
            result = decompose_question(
                model, tokenizer, question,
                max_length=args.max_length,
                temperature=args.temperature
            )
            print("\n" + "=" * 80)
            print(result)
            print("=" * 80)
    
    elif args.question:
        # Single question mode
        print(f"Question: {args.question}\n")
        result = decompose_question(
            model, tokenizer, args.question,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print("\n" + "=" * 80)
        print(result)
        print("=" * 80)
    
    else:
        print("Please provide --question or use --interactive mode")


if __name__ == "__main__":
    main()
