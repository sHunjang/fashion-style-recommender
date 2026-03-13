# tests/test_api.py

"""
FastAPI 서버 테스트
서버 실행 후 테스트!

실행 순서:
1. uvicorn src.api.main:app --port 8000
2. python tests/test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    print("=== Health Check ===")
    res = requests.get(f"{BASE_URL}/health")
    data = res.json()
    print(f"Status:       {data['status']}")
    print(f"Model loaded: {data['model_loaded']}")
    print(f"Index loaded: {data['index_loaded']}")
    print(f"Index size:   {data['index_size']:,}")
    print(f"Device:       {data['device']}")
    print()


def test_text_search():
    print("=== Text Search ===")
    queries = [
        "casual blue shirt for men",
        "elegant formal dress for women",
        "sport shoes for running",
    ]

    for query in queries:
        res = requests.post(
            f"{BASE_URL}/search/text",
            json={"text": query, "top_k": 3}
        )
        data = res.json()
        print(f"쿼리: '{query}'")
        print(f"검색 시간: {data['search_time_ms']}ms")
        for r in data["results"]:
            meta = r["metadata"]
            print(
                f"  {r['rank']}. [{r['score']:.4f}] "
                f"{meta.get('article_type')} | "
                f"{meta.get('name', '')[:35]}"
            )
        print()


def test_image_search():
    print("=== Image Search ===")
    with open("data/raw/images/15970.jpg", "rb") as f:
        res = requests.post(
            f"{BASE_URL}/search/image",
            files={"file": ("15970.jpg", f, "image/jpeg")},
            params={"top_k": 5}
        )
    data = res.json()
    print(f"검색 시간: {data['search_time_ms']}ms")
    for r in data["results"]:
        meta = r["metadata"]
        print(
            f"  {r['rank']}. [{r['score']:.4f}] "
            f"{meta.get('article_type')} | "
            f"{meta.get('name', '')[:35]}"
        )
    print()


def test_similar_by_id():
    print("=== Similar by ID ===")
    res = requests.get(
        f"{BASE_URL}/search/similar/15970",
        params={"top_k": 3}
    )
    data = res.json()
    print(f"검색 시간: {data['search_time_ms']}ms")
    for r in data["results"]:
        meta = r["metadata"]
        print(
            f"  {r['rank']}. [{r['score']:.4f}] "
            f"{meta.get('name', '')[:40]}"
        )
    print()


if __name__ == "__main__":
    print("=== API Server Test ===")
    print()
    test_health()
    test_text_search()
    test_image_search()
    test_similar_by_id()
    print("=== All Tests Passed! ===")