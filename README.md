# 👗 Fashion Style Recommender

> CLIP 기반 패션 스타일 유사도 측정 및 추천 시스템

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---

## 📌 프로젝트 개요

패션 이미지에서 스타일을 의미론적으로 이해하고
유사한 스타일의 옷을 추천하는 AI 시스템

석사 논문 연계 프로젝트:
**"패션 스타일 유사도 측정 및 추천"**

### 핵심 기능
- 이미지 업로드 → 유사 스타일 검색
- 텍스트 설명 → Zero-shot 이미지 검색
- 스타일 카테고리 자동 분류
- REST API 서버 제공

---

## 🏗️ 시스템 아키텍처
```
이미지 입력              텍스트 입력
    ↓                        ↓
[Image Encoder]        [Text Encoder]
(CLIP ViT-B/32)        (CLIP Transformer)
    ↓                        ↓
    └──── 512d 임베딩 벡터 ────┘
                 ↓
         [FAISS IVF Index]
         (벡터 유사도 검색)
                 ↓
       Top-K 유사 스타일 추천
```

---

## 🛠️ 기술 스택

### AI/ML
| 기술 | 버전 | 용도 |
|------|------|------|
| PyTorch | 2.0+ | 딥러닝 프레임워크 |
| CLIP | ViT-B/32 | 이미지-텍스트 임베딩 |
| FAISS | 1.7.4 | 벡터 유사도 검색 |
| Transformers | 4.35+ | 텍스트 인코더 |

### 백엔드 / 프론트엔드
| 기술 | 버전 | 용도 |
|------|------|------|
| FastAPI | 0.104+ | REST API 서버 |
| Gradio | 4.0+ | 웹 데모 UI |
| Uvicorn | 0.24+ | ASGI 서버 |

### 인프라
| 기술 | 용도 |
|------|------|
| Docker | 컨테이너화 |
| HuggingFace Spaces | 무료 배포 |
| GitHub Actions | CI/CD |

---

## 📐 자료구조

| 자료구조 | 적용 위치 | 시간복잡도 | 선택 이유 |
|---------|-----------|-----------|---------|
| numpy array | 임베딩 저장 | O(1) 접근 | 연속 메모리, 행렬 연산 최적화 |
| FAISS IVF Index | 유사도 검색 | O(√N) | Brute-force O(N) 대비 고속 검색 |
| Hash Map (dict) | 메타데이터 조회 | O(1) | 이미지 ID → 정보 즉시 조회 |
| Priority Queue | Top-K 정렬 | O(K log N) | 전체 정렬 불필요 |

---

## 📊 데이터셋
- **DeepFashion2**: 80만 장 패션 이미지
- **Polyvore**: 코디 조합 데이터

---

## 🚀 빠른 시작
```bash
# 저장소 클론
git clone https://github.com/username/fashion-style-recommender
cd fashion-style-recommender

# conda 환경 생성
conda create -n fashion-cv python=3.10 -y
conda activate fashion-cv

# 패키지 설치
pip install -r requirements.txt

# API 서버 실행
uvicorn src.api.main:app --reload

# 웹 데모 실행
python app/demo.py
```

---

## 🌿 브랜치 전략 (Git Flow)
```
main        → 배포 브랜치 (안정적)
develop     → 개발 통합 브랜치
feature/*   → 기능 개발
release/*   → 배포 준비
hotfix/*    → 긴급 수정
```

---

## 📈 성능

| 지표 | 값 |
|------|-----|
| Top-1 정확도 | 업데이트 예정 |
| Top-5 정확도 | 업데이트 예정 |
| 검색 속도 | 업데이트 예정 |

---

## 📝 석사 논문 연계

**논문 주제**: 패션 스타일 유사도 측정 및 추천
**핵심 기여**: CLIP 기반 의미론적 스타일 임베딩

---

## 📁 프로젝트 구조
```
fashion-style-recommender/
├── src/
│   ├── data/        # 데이터 처리
│   ├── models/      # CLIP 인코더
│   ├── search/      # FAISS 검색
│   ├── api/         # FastAPI 서버
│   └── utils/       # 유틸리티
├── app/             # Gradio 데모
├── configs/         # 설정 파일
├── tests/           # 테스트
└── notebooks/       # 실험 노트북
```