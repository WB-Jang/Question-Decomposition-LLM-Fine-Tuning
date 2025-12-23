"""
Data loading and preprocessing utilities
"""
import json
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd


class QuestionDecompositionDataset:
    """Question decomposition dataset handler"""
    
    def __init__(self, train_path: Optional[str] = None, eval_path: Optional[str] = None):
        self.train_path = train_path
        self.eval_path = eval_path
        
    def load_from_json(self, file_path: str) -> List[Dict]:
        """Load data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def load_from_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def create_prompt(self, question: str, sub_questions: Optional[List[str]] = None) -> str:
        """Create training prompt for question decomposition"""
        prompt = f"""### Instruction:
복잡한 질문을 단순한 여러 개의 하위 질문으로 분해하세요.

### Question:
{question}

### Sub-questions:
"""
        if sub_questions:
            for i, sub_q in enumerate(sub_questions, 1):
                prompt += f"{i}. {sub_q}\n"
        
        return prompt
    
    def format_data(self, examples: List[Dict]) -> List[Dict]:
        """Format data for training"""
        formatted_data = []
        for example in examples:
            question = example.get("question", "")
            sub_questions = example.get("sub_questions", [])
            
            text = self.create_prompt(question, sub_questions)
            formatted_data.append({"text": text})
        
        return formatted_data
    
    def load_dataset(self, test_size: float = 0.1, max_samples: Optional[int] = None) -> DatasetDict:
        """Load and prepare dataset"""
        # Load training data
        if self.train_path:
            if self.train_path.endswith('.json'):
                train_data = self.load_from_json(self.train_path)
            elif self.train_path.endswith('.jsonl'):
                train_data = self.load_from_jsonl(self.train_path)
            else:
                raise ValueError("Unsupported file format. Use .json or .jsonl")
        else:
            # Create sample data for demonstration
            train_data = self._create_sample_data()
        
        # Limit samples if specified
        if max_samples:
            train_data = train_data[:max_samples]
        
        # Format data
        formatted_data = self.format_data(train_data)
        
        # Create dataset
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        # Split into train and eval
        if self.eval_path:
            if self.eval_path.endswith('.json'):
                eval_data = self.load_from_json(self.eval_path)
            elif self.eval_path.endswith('.jsonl'):
                eval_data = self.load_from_jsonl(self.eval_path)
            formatted_eval = self.format_data(eval_data)
            eval_dataset = Dataset.from_pandas(pd.DataFrame(formatted_eval))
            
            return DatasetDict({
                "train": dataset,
                "eval": eval_dataset
            })
        else:
            # Split automatically
            split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
            return DatasetDict({
                "train": split_dataset["train"],
                "eval": split_dataset["test"]
            })
    
    def _create_sample_data(self) -> List[Dict]:
        """Create sample data for demonstration"""
        return [
            {
                "question": "한국의 수도 서울에서 가장 유명한 관광지의 역사와 그곳을 방문하는 가장 좋은 시기는 언제인가요?",
                "sub_questions": [
                    "한국의 수도는 어디인가요?",
                    "서울에서 가장 유명한 관광지는 어디인가요?",
                    "그 관광지의 역사는 어떻게 되나요?",
                    "그곳을 방문하기 가장 좋은 시기는 언제인가요?"
                ]
            },
            {
                "question": "기계 학습에서 가장 널리 사용되는 알고리즘의 장단점과 실제 적용 사례는 무엇인가요?",
                "sub_questions": [
                    "기계 학습에서 가장 널리 사용되는 알고리즘은 무엇인가요?",
                    "그 알고리즘의 장점은 무엇인가요?",
                    "그 알고리즘의 단점은 무엇인가요?",
                    "실제로 어떤 분야에 적용되고 있나요?"
                ]
            }
        ]


def get_tokenize_function(tokenizer, max_length: int = 512):
    """Create tokenization function"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    return tokenize_function
