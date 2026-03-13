# src/api/main.py

"""
FastAPI 메인 애플리케이션

서버 시작 순서:
① CLIP 모델 로드 (GPU)
② FAISS 인덱스 로드 or 구축
③ API 서버 시작
④ 요청 처리

실행:
uvicorn src.api.main:app --reload --port 8000
"""

import logging
import os
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.data.dataset import FashionDataset
from src.models.clip_encoder import FashionCLIPEncoder
from src.search.indexer import FashionFAISSIndexer
from src.search.retriever import FashionRetriever

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# 설정 로드
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# 전역 retriever 인스턴스
# 서버 시작 시 한 번만 로드!
# 매 요청마다 로드하면 너무 느림!
retriever: FashionRetriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작/종료 이벤트 처리

    시작 시:
    - CLIP 모델 로드
    - FAISS 인덱스 로드 or 구축

    종료 시:
    - 리소스 정리
    """
    global retriever

    # ----------------------------------------
    # 서버 시작 시 실행
    # ----------------------------------------
    logger.info("서버 시작 중...")

    # CLIP 인코더 로드
    encoder = FashionCLIPEncoder(
        model_name=config["model"]["clip_model"],
        device=config["model"]["device"],
        use_fp16=True,
    )

    # FAISS 인덱서 초기화
    indexer = FashionFAISSIndexer(
        embed_dim=config["model"]["embed_dim"],
        nlist=config["search"]["nlist"],
        nprobe=config["search"]["nprobe"],
    )

    retriever = FashionRetriever(encoder, indexer)

    # 인덱스 로드 or 구축
    index_dir = config["data"]["index_dir"]

    if Path(index_dir + "/fashion.index").exists():
        # 기존 인덱스 로드
        logger.info("기존 인덱스 로드 중...")
        retriever.load_index(index_dir)
        logger.info(
            f"인덱스 로드 완료! "
            f"{indexer.total:,}개 벡터"
        )
    else:
        # 인덱스 구축
        logger.info("인덱스 구축 중...")
        dataset = FashionDataset(
            data_dir=config["data"]["data_dir"],
            split="all",
            max_samples=config["data"]["max_images"],
        )
        retriever.build_index_from_dataset(
            dataset,
            batch_size=config["model"]["batch_size"],
        )
        retriever.save_index(index_dir)
        logger.info("인덱스 구축 및 저장 완료!")

    logger.info("서버 준비 완료! 🚀")

    yield
    # ----------------------------------------
    # 서버 종료 시 실행
    # ----------------------------------------
    logger.info("서버 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="Fashion Style Recommender API",
    description="""
    CLIP 기반 패션 스타일 유사도 검색 시스템

    ## 기능
    - 이미지 업로드 → 유사 스타일 검색
    - 텍스트 설명 → Zero-shot 이미지 검색
    - 이미지 ID → 유사 상품 추천

    ## 기술 스택
    - CLIP (ViT-B/32): 이미지/텍스트 임베딩
    - FAISS (IVF): 벡터 유사도 검색
    - FastAPI: REST API 서버
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(
    router,
    prefix="/api/v1",
    tags=["Fashion Search"],
)

# 정적 파일 (이미지 서빙)
if Path("data/raw/images").exists():
    app.mount(
        "/images",
        StaticFiles(directory="data/raw/images"),
        name="images",
    )