# 👗 Fashion Style Recommender

> CLIP 기반 패션 스타일 유사도 측정 및 추천 시스템

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Demo-FFD21E)](https://huggingface.co/spaces/sngdmtdkw-02/fashion-style-recommender)

---

## 📌 프로젝트 개요

패션 이미지에서 스타일을 의미론적으로 이해하고
유사한 스타일의 옷을 추천하는 AI 시스템

석사 논문 연계 프로젝트:
**"패션 스타일 유사도 측정 및 추천"**

### 핵심 기능
- 📸 이미지 업로드 → 유사 스타일 검색
- 📝 텍스트 설명 → Zero-shot 이미지 검색
- 🏷️ 스타일 카테고리 자동 분류 (casual, formal, sport, street, vintage)
- 🚀 REST API 서버 제공 (FastAPI)
- 🌐 웹 데모 배포 (HuggingFace Spaces)

---

## 🌐 데모

**HuggingFace Spaces**: [fashion-style-recommender](https://huggingface.co/spaces/sngdmtdkw-02/fashion-style-recommender)

**HuggingFace Dataset**: [fashion-product-images](https://huggingface.co/datasets/sngdmtdkw-02/fashion-product-images)

---

## 🏗️ 시스템 아키텍처

```
이미지 입력              텍스트 입력
    ↓                        ↓
[Image Encoder]        [Text Encoder]
(CLIP ViT-B/32)        (CLIP Transformer)
    ↓                        ↓
    └──── 512d L2정규화 임베딩 ────┘
                 ↓
         [FAISS IVF Index]
         (벡터 유사도 검색, O(√N))
                 ↓
       Top-K 유사 스타일 추천
                 ↓
    [FastAPI REST API / Gradio UI]
```

---

## 🛠️ 기술 스택

### AI/ML
| 기술 | 버전 | 용도 |
|------|------|------|
| PyTorch | 2.10.0+cu128 | 딥러닝 프레임워크 |
| CLIP | ViT-B/32 | 이미지-텍스트 임베딩 |
| FAISS | 1.13.2 | 벡터 유사도 검색 |
| Transformers | 5.3.0 | CLIP 모델 로드 |

### 백엔드 / 프론트엔드
| 기술 | 버전 | 용도 |
|------|------|------|
| FastAPI | 0.135.1 | REST API 서버 |
| Gradio | 6.9.0 | 웹 데모 UI |
| Uvicorn | 최신 | ASGI 서버 |
| Pydantic | 2.x | 데이터 검증 |

### 인프라 / 환경
| 기술 | 용도 |
|------|------|
| HuggingFace Spaces | 무료 배포 (CPU Basic) |
| HuggingFace Dataset | 이미지 데이터 호스팅 |
| GitHub | 버전 관리 (Git Flow) |
| conda | 환경 관리 |

### 개발 환경
- OS: Windows
- GPU: RTX 5060 (Blackwell)
- CUDA: 12.8
- Python: 3.10

---

## 📐 자료구조

| 자료구조 | 적용 위치 | 시간복잡도 | 선택 이유 |
|---------|-----------|-----------|---------|
| numpy array | 임베딩 저장 | O(1) 접근 | 연속 메모리, 행렬 연산 최적화 |
| FAISS IVF Index | 유사도 검색 | O(√N) | Brute-force O(N) 대비 고속 검색 |
| Hash Map (dict) | 메타데이터 조회 | O(1) | 이미지 ID → 정보 즉시 조회 |
| Priority Queue | Top-K 정렬 | O(K log N) | 전체 정렬 불필요 |
| DataFrame | 메타데이터 관리 | - | 필터링/통계 처리 |
| Set | 중복 감지 | O(1) | 데이터 검증 |

---

## 📊 데이터셋

- **출처**: [Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- **규모**: 44,424장 이미지
- **카테고리**: Apparel(21,397), Accessories(11,274), Footwear(9,219)
- **주요 컬럼**: gender, masterCategory, subCategory, articleType, baseColour, season, usage, productDisplayName
- **텍스트 템플릿**: `"a photo of {colour} {article} for {gender}"`

---

## 🚀 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/sHunjang/fashion-style-recommender
cd fashion-style-recommender

# conda 환경 생성
conda create -n fashion-cv python=3.10 -y
conda activate fashion-cv

# 패키지 설치
pip install -r requirements.txt

# API 서버 실행
uvicorn src.api.main:app --reload --port 8000

# 웹 데모 실행
python app/demo.py
```

### API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | /api/v1/health | 서버 상태 확인 |
| POST | /api/v1/search/image | 이미지로 유사 스타일 검색 |
| POST | /api/v1/search/text | 텍스트로 Zero-shot 검색 |
| GET | /api/v1/search/similar/{id} | ID로 유사 상품 추천 |

**Swagger UI**: `http://localhost:8000/docs`

---

## 🌿 브랜치 전략 (Git Flow)

```
main        → 배포 브랜치 (v1.0.0 태그)
develop     → 개발 통합 브랜치
feature/*   → 기능 개발
release/*   → 배포 준비
hotfix/*    → 긴급 수정
```

### 커밋 히스토리

| 브랜치 | 내용 |
|--------|------|
| feature/project-setup | 폴더 구조, 환경 세팅, README |
| feature/data-pipeline | Dataset, Preprocessor, Downloader |
| feature/clip-encoder | CLIPEncoder (GPU fp16), SimilarityCalculator |
| feature/faiss-search | FAISSIndexer, FashionRetriever |
| feature/api-server | FastAPI REST API |
| feature/web-demo | Gradio UI + HuggingFace Spaces 배포 |
| release/v1.0.0 | main 배포 |

---

## 📈 성능

| 지표 | 값 |
|------|-----|
| 이미지 인코딩 속도 | 0.9ms/장 (GPU fp16) |
| 텍스트 검색 속도 | 6.4ms/쿼리 |
| 초당 쿼리 처리 | 156 QPS |
| GPU 메모리 사용 | 296MB |
| 이미지-텍스트 유사도 (대각선) | 0.3038 |
| 이미지-텍스트 유사도 (전체 평균) | 0.2196 |

---

## 🔧 이슈 해결 기록

| 이슈 | 원인 | 해결 |
|------|------|------|
| faiss-gpu 설치 실패 | RTX 5060 Blackwell CUDA 12.8 미지원 | faiss-cpu로 대체 |
| Transformers 5.x CLIP API 변경 | get_image_features() 반환타입 변경 | `.pooler_output` 속성 사용 |
| Windows cp949 인코딩 오류 | config.yaml 한글 주석 | `encoding="utf-8"` 명시 |
| app.py vs app/ 패키지 충돌 | Python 모듈명 충돌 | `__init__.py` 추가 |
| HF Upload Rate Limit | 파일마다 개별 커밋 | `upload_large_folder` 사용 |
| Spaces 데이터 경로 불일치 | snapshot_download 경로 구조 차이 | `ignore_patterns`로 분리 저장 |

---

## 📁 프로젝트 구조

```
fashion-style-recommender/
├── src/
│   ├── data/
│   │   ├── dataset.py        # FashionDataset (Hash Map, DataFrame)
│   │   ├── preprocessor.py   # FashionPreprocessor (train/inference 모드)
│   │   └── downloader.py     # FashionDataDownloader (무결성 검증)
│   ├── models/
│   │   ├── clip_encoder.py   # FashionCLIPEncoder (GPU fp16)
│   │   └── similarity.py     # FashionSimilarityCalculator (N×M 행렬)
│   ├── search/
│   │   ├── indexer.py        # FashionFAISSIndexer (IVF Index)
│   │   └── retriever.py      # FashionRetriever (통합 파이프라인)
│   └── api/
│       ├── main.py           # FastAPI 앱 (lifespan)
│       ├── routes.py         # 엔드포인트 정의
│       └── schemas.py        # Pydantic 스키마
├── app/
│   └── demo.py               # Gradio 웹 데모 (로컬용)
├── app.py                    # HuggingFace Spaces 진입점
├── tests/                    # 각 모듈 테스트
├── configs/
│   └── config.yaml           # 설정 파일
├── requirements.txt
├── environment.yaml
└── pyproject.toml
```

---

## 📝 석사 논문 연계

**논문 주제**: 패션 스타일 유사도 측정 및 추천

**핵심 기여**:
- CLIP 기반 의미론적 스타일 임베딩
- Zero-shot 스타일 카테고리 분류
- FAISS IVF 기반 실시간 유사도 검색