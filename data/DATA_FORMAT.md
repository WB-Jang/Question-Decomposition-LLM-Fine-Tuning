# Data format documentation

## Training Data Format

Question Decomposition 학습 데이터는 두 가지 형식을 지원합니다:

### 1. JSON Format (.json)

```json
[
  {
    "question": "복잡한 질문",
    "sub_questions": [
      "하위 질문 1",
      "하위 질문 2",
      "하위 질문 3"
    ]
  },
  {
    "question": "또 다른 복잡한 질문",
    "sub_questions": [
      "하위 질문 1",
      "하위 질문 2"
    ]
  }
]
```

### 2. JSONL Format (.jsonl)

각 줄이 하나의 JSON 객체:

```jsonl
{"question": "복잡한 질문", "sub_questions": ["하위 질문 1", "하위 질문 2", "하위 질문 3"]}
{"question": "또 다른 복잡한 질문", "sub_questions": ["하위 질문 1", "하위 질문 2"]}
```

## Field Descriptions

- `question` (string, required): 분해할 복잡한 질문
- `sub_questions` (list of strings, required): 하위 질문 리스트

## Best Practices

### Good Examples

✅ **질문이 명확하고 하위 질문이 논리적인 순서로 분해됨**
```json
{
  "question": "한국의 GDP 성장률이 세계 경제에 미치는 영향과 주요 산업 분야는?",
  "sub_questions": [
    "한국의 현재 GDP 성장률은?",
    "한국의 GDP가 세계 경제에 미치는 영향은?",
    "한국의 주요 산업 분야는 무엇인가?"
  ]
}
```

✅ **하위 질문들이 독립적이면서도 원래 질문에 답하기 위해 필요함**
```json
{
  "question": "기계 학습 알고리즘 중 가장 정확도가 높은 것과 그 학습 시간은?",
  "sub_questions": [
    "어떤 기계 학습 알고리즘들이 있나?",
    "각 알고리즘의 정확도는 어떻게 되나?",
    "가장 정확도가 높은 알고리즘은?",
    "그 알고리즘의 학습 시간은?"
  ]
}
```

### Bad Examples

❌ **하위 질문이 너무 추상적이거나 원래 질문과 관련 없음**
```json
{
  "question": "서울의 인구와 면적은?",
  "sub_questions": [
    "서울은 어디에 있나?",  // 불필요
    "한국의 수도는?",  // 원래 질문과 무관
    "서울의 인구는?",
    "서울의 면적은?"
  ]
}
```

❌ **하위 질문이 원래 질문을 단순히 반복**
```json
{
  "question": "AI의 장점과 단점은?",
  "sub_questions": [
    "AI의 장점과 단점은?"  // 분해되지 않음
  ]
}
```

## Data Collection Tips

1. **다양한 도메인**: 다양한 주제의 질문 수집
2. **질문 복잡도**: 2-5개의 하위 질문으로 분해 가능한 정도
3. **논리적 순서**: 하위 질문이 논리적 순서를 따르도록
4. **명확성**: 각 하위 질문이 명확하고 답변 가능하도록
5. **독립성**: 각 하위 질문이 독립적으로 답변 가능하도록

## Data Size Recommendations

- **최소**: 100개 샘플 (proof of concept)
- **권장**: 1,000-10,000개 샘플 (실용적 성능)
- **최적**: 10,000개 이상 샘플 (고품질 모델)

## Data Split

기본 설정:
- Train: 90%
- Eval: 10%

별도의 eval 데이터를 제공하지 않으면 자동으로 분할됩니다.
