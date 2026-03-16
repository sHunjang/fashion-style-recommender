# src/training/trainer.py

"""
CLIP LoRA 파인튜닝 트레이너

구조:
① CLIP 모델 로드 (Frozen)
② LoRA 어댑터 추가 (학습 가능)
③ Contrastive Loss 계산
④ 학습 루프
⑤ 모델 저장

핵심 포인트:
- 전체 파라미터의 ~2%만 학습!
- 기존 CLIP 지식 보존!
- 패션 도메인 특화!
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


# ----------------------------------------
# Contrastive Loss
# ----------------------------------------
class ContrastiveLoss(nn.Module):
    """
    CLIP Contrastive Loss

    수식:
    similarity = image_emb @ text_emb.T / temperature
    L_image = CrossEntropy(similarity, labels)
    L_text  = CrossEntropy(similarity.T, labels)
    Loss    = (L_image + L_text) / 2

    핵심:
    - 대각선 (짝) → 유사도 최대화
    - 비대각선 (비짝) → 유사도 최소화
    - temperature: 분포 날카롭게 조절
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_embeddings: (N, D) L2 정규화된 이미지 임베딩
            text_embeddings:  (N, D) L2 정규화된 텍스트 임베딩

        Returns:
            loss: 스칼라 손실값
        """
        N = image_embeddings.shape[0]

        # L2 정규화
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings  = F.normalize(text_embeddings,  dim=-1)

        # 유사도 행렬 (N, N)
        # [i, j] = i번째 이미지 & j번째 텍스트 유사도
        logits = (
            image_embeddings @ text_embeddings.T
        ) / self.temperature

        # 정답 레이블: 대각선 (0,0), (1,1), ..., (N-1,N-1)
        labels = torch.arange(N, device=logits.device)

        # 이미지 → 텍스트 방향 손실
        loss_image = F.cross_entropy(logits, labels)

        # 텍스트 → 이미지 방향 손실
        loss_text = F.cross_entropy(logits.T, labels)

        return (loss_image + loss_text) / 2


# ----------------------------------------
# Fashion Dataset for Training
# ----------------------------------------
class FashionTrainDataset(Dataset):
    """
    파인튜닝용 패션 데이터셋

    각 샘플: (이미지, 텍스트) 쌍
    → Contrastive Learning 학습용
    """

    def __init__(
        self,
        data_dir: str,
        processor: CLIPProcessor,
        max_samples: int = None,
    ):
        from src.data.dataset import FashionDataset
        self.processor = processor

        base = FashionDataset(
            data_dir=data_dir,
            split="train",
            max_samples=max_samples,
        )

        # 유효한 샘플만 수집
        self.samples = []
        for meta in base.id_to_meta.values():
            if Path(meta["image_path"]).exists():
                self.samples.append(meta)

        logger.info(
            f"[TrainDataset] {len(self.samples):,}개 로드 완료!"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        # 이미지 로드
        image = Image.open(
            meta["image_path"]
        ).convert("RGB")

        # 텍스트
        text = meta["text"]

        # CLIP 전처리
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids":    inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


# ----------------------------------------
# LoRA CLIP Trainer
# ----------------------------------------
class FashionCLIPLoRATrainer:
    """
    CLIP LoRA 파인튜닝 트레이너

    학습 전략:
    ① Vision Encoder에 LoRA 적용
    ② Text Encoder에 LoRA 적용
    ③ Contrastive Loss로 학습
    ④ 패션 도메인 특화!

    파라미터:
    전체: ~150M
    학습: ~3M (2%!)
    """

    def __init__(
        self,
        model_name:  str   = "openai/clip-vit-base-patch32",
        device:      str   = "cuda",
        lora_r:      int   = 8,
        lora_alpha:  int   = 16,
        lora_dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        """
        Args:
            lora_r:      LoRA 랭크 (낮을수록 파라미터 적음)
            lora_alpha:  LoRA 스케일링 (보통 r*2)
            lora_dropout: LoRA 드롭아웃
            temperature: Contrastive Loss 온도
        """
        self.device = torch.device(device)

        # ① CLIP 모델 로드
        logger.info(f"[Trainer] 모델 로딩: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(
            model_name
        )

        # ② LoRA 설정
        # Vision Encoder의 attention 레이어에 적용
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # CLIP의 attention projection 레이어들
            target_modules=[
                "q_proj", "v_proj",
                # Vision + Text 양쪽 모두!
            ],
            bias="none",
        )

        # ③ LoRA 어댑터 추가
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

        # 파라미터 수 출력
        self.model.print_trainable_parameters()

        # ④ Loss & Optimizer
        self.criterion = ContrastiveLoss(temperature)

        logger.info("[Trainer] 초기화 완료!")

    def train(
        self,
        data_dir:    str,
        save_dir:    str,
        epochs:      int   = 5,
        batch_size:  int   = 32,
        lr:          float = 1e-4,
        max_samples: int   = None,
        eval_every:  int   = 1,
    ) -> dict:
        """
        파인튜닝 실행

        Args:
            data_dir:    데이터 경로
            save_dir:    모델 저장 경로
            epochs:      학습 에폭
            batch_size:  배치 크기
            lr:          학습률
            max_samples: 최대 샘플 수 (None=전체)
            eval_every:  평가 주기 (에폭 단위)

        Returns:
            history: 학습 기록
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 데이터셋 & 로더
        dataset = FashionTrainDataset(
            data_dir=data_dir,
            processor=self.processor,
            max_samples=max_samples,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters()
            ),
            lr=lr,
            weight_decay=0.01,
        )

        # LR Scheduler (Cosine)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.1,
        )

        # 학습 기록
        history = {
            "train_loss": [],
            "eval_loss":  [],
        }

        logger.info(
            f"[Trainer] 학습 시작! "
            f"epochs={epochs} | "
            f"batch={batch_size} | "
            f"lr={lr} | "
            f"samples={len(dataset):,}"
        )

        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # ----------------------------------------
            # Train
            # ----------------------------------------
            self.model.train()
            epoch_loss = 0.0
            start = time.time()

            for batch in tqdm(
                loader,
                desc=f"Epoch {epoch}/{epochs}",
            ):
                # 배치를 GPU로
                pixel_values  = batch["pixel_values"].to(self.device)
                input_ids     = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # 임베딩 추출
                # Transformers 5.x: pooler_output 사용
                image_emb = outputs.vision_model_output.pooler_output
                text_emb  = outputs.text_model_output.pooler_output

                # Projection (512d)
                image_emb = outputs.image_embeds
                text_emb  = outputs.text_embeds

                # Contrastive Loss
                loss = self.criterion(image_emb, text_emb)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping (안정적 학습!)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )

                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            avg_loss   = epoch_loss / len(loader)
            epoch_time = time.time() - start
            history["train_loss"].append(avg_loss)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            # ----------------------------------------
            # 최적 모델 저장
            # ----------------------------------------
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(str(save_path / "best"))
                logger.info(
                    f"  ✅ Best model 저장! "
                    f"loss={best_loss:.4f}"
                )

        # 최종 모델 저장
        self.save(str(save_path / "final"))
        logger.info(
            f"[Trainer] 학습 완료! "
            f"Best loss: {best_loss:.4f}"
        )

        return history

    def save(self, save_dir: str) -> None:
        """LoRA 어댑터 저장"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        logger.info(f"[Trainer] 모델 저장: {save_dir}")

    def load(self, save_dir: str) -> None:
        """LoRA 어댑터 로드"""
        from peft import PeftModel
        base = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model = PeftModel.from_pretrained(
            base, save_dir
        )
        self.model.to(self.device)
        logger.info(f"[Trainer] 모델 로드: {save_dir}")