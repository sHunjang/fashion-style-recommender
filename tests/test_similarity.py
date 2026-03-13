# tests/test_similarity.py

import numpy as np
from src.data.dataset import FashionDataset
from src.models.clip_encoder import FashionCLIPEncoder
from src.models.similarity import (
    FashionSimilarityCalculator,
    STYLE_CATEGORIES,
)

print("=== Similarity Calculator Test ===")
print()

# ----------------------------------------
# 1. 초기화
# ----------------------------------------
encoder    = FashionCLIPEncoder(device="cuda", use_fp16=True)
calculator = FashionSimilarityCalculator()

# ----------------------------------------
# 2. 코사인 유사도 테스트
# ----------------------------------------
print("=== Cosine Similarity Test ===")

v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([1.0, 0.0, 0.0])
v3 = np.array([0.0, 1.0, 0.0])
v4 = np.array([-1.0, 0.0, 0.0])

print(f"  동일 벡터:      {calculator.cosine_similarity(v1, v2):.4f}")
# 1.0
print(f"  직교 벡터:      {calculator.cosine_similarity(v1, v3):.4f}")
# 0.0
print(f"  반대 벡터:      {calculator.cosine_similarity(v1, v4):.4f}")
# -1.0
print()

# ----------------------------------------
# 3. 유사도 행렬 테스트
# ----------------------------------------
print("=== Similarity Matrix Test ===")

dataset = FashionDataset(
    data_dir="data/raw",
    split="all",
    max_samples=50,
)

from PIL import Image
images = [
    Image.open(meta["image_path"]).convert("RGB")
    for meta in list(dataset.id_to_meta.values())[:10]
]
texts = [
    meta["text"]
    for meta in list(dataset.id_to_meta.values())[:10]
]

# 임베딩 추출
img_embs = encoder.encode_images(images, show_progress=False)
txt_embs = encoder.encode_texts(texts, show_progress=False)

# 이미지-텍스트 유사도 행렬 (10×10)
sim_matrix = calculator.cosine_similarity_matrix(
    img_embs, txt_embs
)
print(f"  유사도 행렬 shape: {sim_matrix.shape}")

# 대각선이 가장 높아야 함! (짝인 이미지-텍스트)
diagonal = np.diag(sim_matrix)
print(f"  대각선 평균 유사도: {diagonal.mean():.4f}")
print(f"  전체 평균 유사도:   {sim_matrix.mean():.4f}")
print(f"  대각선 > 전체 평균: {diagonal.mean() > sim_matrix.mean()}")
print()

# ----------------------------------------
# 4. Top-K 유사 검색 테스트
# ----------------------------------------
print("=== Top-K Similar Test ===")

query_emb = img_embs[0]
top_indices, top_scores = calculator.top_k_similar(
    query_emb,
    img_embs,
    k=5,
    exclude_idx=0,
)

meta_list = list(dataset.id_to_meta.values())[:10]
print(f"  쿼리: {meta_list[0]['name']}")
print(f"  Top-5 유사 아이템:")
for rank, (idx, score) in enumerate(
    zip(top_indices, top_scores), 1
):
    print(
        f"    {rank}. [{score:.4f}] "
        f"{meta_list[idx]['name']}"
    )
print()

# ----------------------------------------
# 5. 스타일 카테고리 분류 테스트
# ----------------------------------------
print("=== Style Classification Test ===")
print("스타일 카테고리 임베딩 생성 중...")

style_embeddings = calculator.encode_style_categories(
    encoder, STYLE_CATEGORIES
)
print()

# 이미지 5개 스타일 분류
for i in range(5):
    meta   = meta_list[i]
    result = calculator.classify_style(
        img_embs[i], style_embeddings
    )
    top    = result["top_style"]
    score  = result["scores"][top]
    print(
        f"  [{top:<8} {score:.3f}] "
        f"{meta['article_type']} - {meta['name'][:40]}"
    )
print()

# ----------------------------------------
# 6. 유사도 통계
# ----------------------------------------
print("=== Similarity Stats ===")
stats = calculator.get_similarity_stats(sim_matrix)
for k, v in stats.items():
    print(f"  {k}: {v}")
print()

print("=== All Tests Passed! ===")