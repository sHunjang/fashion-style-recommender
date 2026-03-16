# src/training/evaluator.py

"""
파인튜닝 전/후 성능 비교 평가

평가 지표:
① Recall@K: 상위 K개 중 정답 비율
② MRR: Mean Reciprocal Rank
③ 유사도 분포 비교 (대각선 vs 비대각선)

핵심 포인트:
Zero-shot vs Fine-tuned 성능 비교
→ 논문의 핵심 실험 결과!
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FashionCLIPEvaluator:
    """
    CLIP 파인튜닝 평가기

    평가 지표:
    ① Recall@1, @5, @10
       상위 K개 검색 결과 중 정답 비율
    ② MRR (Mean Reciprocal Rank)
       정답이 몇 번째에 나왔는지 역수 평균
    ③ 유사도 분포
       대각선(짝) vs 비대각선(비짝) 유사도
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

    def evaluate(
        self,
        model,
        processor,
        dataset,
        batch_size: int = 64,
        max_samples: int = 1000,
    ) -> dict:
        """
        모델 평가 실행

        Args:
            model:       CLIPModel (Zero-shot or Fine-tuned)
            processor:   CLIPProcessor
            dataset:     FashionDataset
            max_samples: 평가할 샘플 수

        Returns:
            metrics: {
                "recall@1":  0.xx,
                "recall@5":  0.xx,
                "recall@10": 0.xx,
                "mrr":       0.xx,
                "diag_sim":  0.xx,  ← 짝 유사도
                "off_sim":   0.xx,  ← 비짝 유사도
                "gap":       0.xx,  ← 차이 (클수록 좋음!)
            }
        """
        model.eval()
        all_image_embs = []
        all_text_embs  = []

        # 샘플 수집
        samples = list(dataset.id_to_meta.values())
        if max_samples:
            samples = samples[:max_samples]

        logger.info(
            f"[Evaluator] 평가 시작: {len(samples)}개"
        )

        # 배치 처리
        for i in tqdm(
            range(0, len(samples), batch_size),
            desc="Evaluating"
        ):
            batch = samples[i:i + batch_size]

            images = []
            texts  = []
            for meta in batch:
                try:
                    img = Image.open(
                        meta["image_path"]
                    ).convert("RGB")
                    images.append(img)
                    texts.append(meta["text"])
                except Exception:
                    continue

            if not images:
                continue

            inputs = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                image_emb = F.normalize(
                    outputs.image_embeds, dim=-1
                )
                text_emb = F.normalize(
                    outputs.text_embeds, dim=-1
                )

            all_image_embs.append(
                image_emb.cpu().numpy()
            )
            all_text_embs.append(
                text_emb.cpu().numpy()
            )

        # 전체 임베딩 합치기
        image_embs = np.vstack(all_image_embs)
        text_embs  = np.vstack(all_text_embs)
        N = len(image_embs)

        # 유사도 행렬 (N, N)
        sim_matrix = image_embs @ text_embs.T

        # ----------------------------------------
        # Recall@K 계산
        # ----------------------------------------
        recalls = {}
        for k in [1, 5, 10]:
            correct = 0
            for i in range(N):
                # i번째 이미지의 Top-K 텍스트
                top_k = np.argsort(
                    sim_matrix[i]
                )[::-1][:k]
                if i in top_k:
                    correct += 1
            recalls[f"recall@{k}"] = correct / N

        # ----------------------------------------
        # MRR 계산
        # ----------------------------------------
        mrr = 0.0
        for i in range(N):
            ranked = np.argsort(
                sim_matrix[i]
            )[::-1]
            rank = np.where(ranked == i)[0][0] + 1
            mrr += 1.0 / rank
        mrr /= N

        # ----------------------------------------
        # 유사도 분포
        # ----------------------------------------
        diag     = np.diag(sim_matrix)
        mask     = ~np.eye(N, dtype=bool)
        off_diag = sim_matrix[mask]

        metrics = {
            **recalls,
            "mrr":      round(float(mrr), 4),
            "diag_sim": round(float(diag.mean()), 4),
            "off_sim":  round(float(off_diag.mean()), 4),
            "gap":      round(
                float(diag.mean() - off_diag.mean()), 4
            ),
            "n_samples": N,
        }

        return metrics

    def compare(
        self,
        zero_shot_metrics: dict,
        finetuned_metrics: dict,
    ) -> None:
        """Zero-shot vs Fine-tuned 성능 비교 출력"""
        print("\n" + "=" * 55)
        print("📊 Zero-shot vs Fine-tuned 성능 비교")
        print("=" * 55)
        print(f"{'지표':<15} {'Zero-shot':>12} {'Fine-tuned':>12} {'개선':>10}")
        print("-" * 55)

        for key in [
            "recall@1", "recall@5", "recall@10",
            "mrr", "diag_sim", "off_sim", "gap"
        ]:
            z = zero_shot_metrics[key]
            f = finetuned_metrics[key]
            diff = f - z
            arrow = "↑" if diff > 0 else "↓"
            print(
                f"{key:<15} {z:>12.4f} "
                f"{f:>12.4f} "
                f"{arrow}{abs(diff):>8.4f}"
            )
        print("=" * 55)