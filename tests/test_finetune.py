# tests/test_finetune.py

import logging
import torch
from src.data.dataset import FashionDataset
from src.training.trainer import FashionCLIPLoRATrainer
from src.training.evaluator import FashionCLIPEvaluator

logging.basicConfig(level=logging.INFO)
print("=== LoRA Fine-tuning Test ===")
print()

# ----------------------------------------
# 1. 트레이너 초기화
# ----------------------------------------
trainer = FashionCLIPLoRATrainer(
    device="cuda",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    temperature=0.07,
)
print()

# ----------------------------------------
# 2. 파인튜닝 실행 (소규모 테스트)
# ----------------------------------------
print("=== Fine-tuning 시작 ===")
history = trainer.train(
    data_dir="data/raw",
    save_dir="models/lora",
    epochs=15,
    batch_size=32,
    lr=1e-4,
    max_samples=7000,  # 테스트용 5000장
)

print()
print("=== 학습 손실 ===")
for i, loss in enumerate(history["train_loss"], 1):
    print(f"  Epoch {i}: {loss:.4f}")
print()

# ----------------------------------------
# 3. 평가: Zero-shot vs Fine-tuned 비교
# ----------------------------------------
print("=== 평가 시작 ===")

from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

dataset = FashionDataset(
    data_dir="data/raw",
    split="all",
    max_samples=500,
)
evaluator = FashionCLIPEvaluator(device="cuda")

# Zero-shot 평가
print("Zero-shot 평가 중...")
zero_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to("cuda")
zero_proc = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
zero_metrics = evaluator.evaluate(
    zero_model, zero_proc, dataset,
    max_samples=500,
)
print(f"Zero-shot Recall@1: {zero_metrics['recall@1']:.4f}")

# Fine-tuned 평가
print("Fine-tuned 평가 중...")
ft_metrics = evaluator.evaluate(
    trainer.model, trainer.processor, dataset,
    max_samples=500,
)
print(f"Fine-tuned Recall@1: {ft_metrics['recall@1']:.4f}")

# 비교 출력
evaluator.compare(zero_metrics, ft_metrics)

print()
print("=== Test Passed! ===")