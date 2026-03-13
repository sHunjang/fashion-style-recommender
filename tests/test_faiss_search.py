# tests/test_faiss_search.py

import time
import logging
from PIL import Image
from src.data.dataset import FashionDataset
from src.models.clip_encoder import FashionCLIPEncoder
from src.search.indexer import FashionFAISSIndexer
from src.search.retriever import FashionRetriever

logging.basicConfig(level=logging.INFO)
print("=== FAISS Search Test ===")
print()

# ----------------------------------------
# 1. 초기화
# ----------------------------------------
encoder = FashionCLIPEncoder(
    device="cuda",
    use_fp16=True,
)
indexer = FashionFAISSIndexer(
    embed_dim=512,
    nlist=50,
    # 샘플 수가 적으므로 nlist 작게!
    # 실제: 44000장 → nlist=200
    nprobe=10,
)
retriever = FashionRetriever(encoder, indexer)

# ----------------------------------------
# 2. 인덱스 구축
# ----------------------------------------
print("=== Build Index ===")
dataset = FashionDataset(
    data_dir="data/raw",
    split="all",
    max_samples=500,
    # 테스트용 500장
    # 실제: max_samples=None (전체!)
)

retriever.build_index_from_dataset(
    dataset,
    batch_size=64,
)
print()

# ----------------------------------------
# 3. 인덱스 저장 / 로드 테스트
# ----------------------------------------
print("=== Save / Load Index ===")

retriever.save_index("data/index")
print("저장 완료!")

# 새 인덱서로 로드 테스트
new_indexer  = FashionFAISSIndexer(embed_dim=512)
new_retriever = FashionRetriever(encoder, new_indexer)
new_retriever.load_index("data/index")
print("로드 완료!")
print()

# ----------------------------------------
# 4. 이미지 검색 테스트
# ----------------------------------------
print("=== Image Search Test ===")

query_image = Image.open(
    "data/raw/images/15970.jpg"
).convert("RGB")

result = new_retriever.search_by_image(
    query_image, top_k=5
)

print(f"검색 시간: {result['search_time_ms']}ms")
print(f"결과 수:   {result['total_results']}개")
print()
print("Top-5 유사 스타일:")
for r in result["results"]:
    meta = r["metadata"]
    print(
        f"  {r['rank']}. [{r['score']:.4f}] "
        f"{meta.get('article_type', 'N/A')} | "
        f"{meta.get('name', 'N/A')[:40]}"
    )
print()

# ----------------------------------------
# 5. 텍스트 검색 테스트
# ----------------------------------------
print("=== Text Search Test ===")

queries = [
    "casual blue shirt for men",
    "elegant formal dress for women",
    "sport shoes for running",
]

for query in queries:
    result = new_retriever.search_by_text(
        query, top_k=3
    )
    print(f"쿼리: '{query}'")
    print(f"검색 시간: {result['search_time_ms']}ms")
    for r in result["results"]:
        meta = r["metadata"]
        print(
            f"  {r['rank']}. [{r['score']:.4f}] "
            f"{meta.get('article_type', 'N/A')} | "
            f"{meta.get('name', 'N/A')[:35]}"
        )
    print()

# ----------------------------------------
# 6. 검색 속도 벤치마크
# ----------------------------------------
print("=== Speed Benchmark ===")

N_QUERIES = 100
start = time.time()

for _ in range(N_QUERIES):
    new_retriever.search_by_text(
        "casual blue shirt", top_k=10
    )

elapsed = time.time() - start
print(f"쿼리 수:      {N_QUERIES}개")
print(f"총 시간:      {elapsed:.2f}s")
print(f"쿼리당 시간:  {elapsed/N_QUERIES*1000:.1f}ms")
print(f"초당 쿼리:    {N_QUERIES/elapsed:.0f} QPS")
print()

# 7. 통계
print("=== Index Stats ===")
stats = new_retriever.get_stats()
print(f"인덱스 벡터 수: {stats['index']['total']:,}")
print(f"embed_dim:      {stats['index']['embed_dim']}")
print(f"nlist:          {stats['index']['nlist']}")
print(f"nprobe:         {stats['index']['nprobe']}")
print()
print("=== All Tests Passed! ===")