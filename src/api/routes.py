# src/api/routes.py

"""
FastAPI 라우터 정의

엔드포인트:
GET  /health          → 서버 상태 확인
POST /search/image    → 이미지로 검색
POST /search/text     → 텍스트로 검색
GET  /search/similar/{image_id} → ID로 유사 검색
"""

import io
import logging

from fastapi import APIRouter, File, HTTPException
from fastapi import Query, UploadFile
from PIL import Image

from src.api.schemas import (
    HealthResponse,
    ImageSearchResponse,
    TextSearchRequest,
    TextSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_retriever():
    """
    retriever 인스턴스 반환
    main.py에서 주입!
    """
    from src.api.main import retriever
    return retriever


# ----------------------------------------
# 서버 상태 확인
# ----------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="서버 상태 확인",
)
async def health_check():
    """서버 및 모델 로드 상태 확인"""
    ret = get_retriever()

    if ret is None:
        return HealthResponse(
            status       = "loading",
            model_loaded = False,
            index_loaded = False,
            index_size   = 0,
            device       = "unknown",
            embed_dim    = 0,
        )

    stats = ret.get_stats()
    return HealthResponse(
        status       = "healthy",
        model_loaded = True,
        index_loaded = stats["index"]["is_trained"],
        index_size   = stats["index"]["total"],
        device       = stats["encoder"]["device"],
        embed_dim    = stats["encoder"]["embed_dim"],
    )


# ----------------------------------------
# 이미지로 유사 스타일 검색
# ----------------------------------------
@router.post(
    "/search/image",
    response_model=ImageSearchResponse,
    summary="이미지로 유사 스타일 검색",
)
async def search_by_image(
    file:  UploadFile = File(...),
    top_k: int = Query(
        default=10, ge=1, le=50,
        description="반환할 결과 수"
    ),
):
    """
    이미지 업로드 → 유사 스타일 Top-K 반환

    - 지원 형식: JPG, PNG, WEBP
    - 최대 파일 크기: 10MB
    """
    ret = get_retriever()
    if ret is None:
        raise HTTPException(
            status_code=503,
            detail="서버 초기화 중입니다. 잠시 후 재시도!"
        )

    # 파일 형식 확인
    if file.content_type not in [
        "image/jpeg", "image/png",
        "image/webp", "image/jpg"
    ]:
        raise HTTPException(
            status_code=400,
            detail="JPG, PNG, WEBP 형식만 지원합니다!"
        )

    # 이미지 읽기
    try:
        contents = await file.read()
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"이미지 읽기 실패: {str(e)}"
        )

    # 검색 실행
    try:
        result = ret.search_by_image(image, top_k)
    except Exception as e:
        logger.error(f"검색 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"검색 중 오류 발생: {str(e)}"
        )

    return ImageSearchResponse(**result)


# ----------------------------------------
# 텍스트로 유사 스타일 검색
# ----------------------------------------
@router.post(
    "/search/text",
    response_model=TextSearchResponse,
    summary="텍스트로 유사 스타일 검색",
)
async def search_by_text(
    request: TextSearchRequest,
):
    """
    텍스트 설명 → 유사 스타일 Top-K 반환

    Zero-shot 검색!
    학습 없이 텍스트 설명만으로 검색 가능!

    예시:
    - "casual blue denim jacket for men"
    - "elegant red dress for women"
    - "white sport shoes for running"
    """
    ret = get_retriever()
    if ret is None:
        raise HTTPException(
            status_code=503,
            detail="서버 초기화 중입니다. 잠시 후 재시도!"
        )

    try:
        result = ret.search_by_text(
            request.text, request.top_k
        )
    except Exception as e:
        logger.error(f"검색 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"검색 중 오류 발생: {str(e)}"
        )

    return TextSearchResponse(**result)


# ----------------------------------------
# 이미지 ID로 유사 스타일 검색
# ----------------------------------------
@router.get(
    "/search/similar/{image_id}",
    response_model=ImageSearchResponse,
    summary="이미지 ID로 유사 스타일 검색",
)
async def search_by_id(
    image_id: int,
    top_k: int = Query(
        default=10, ge=1, le=50
    ),
):
    """
    DB에 있는 이미지 ID로 유사 스타일 검색
    "이 상품과 비슷한 상품" 기능!
    """
    ret = get_retriever()
    if ret is None:
        raise HTTPException(
            status_code=503,
            detail="서버 초기화 중입니다!"
        )

    # ID → 이미지 경로 조회
    # Hash Map O(1) 조회
    meta = None
    for idx, m in ret.indexer.idx_to_meta.items():
        if m.get("id") == image_id:
            meta = m
            break

    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"이미지 ID {image_id} 없음!"
        )

    try:
        image = Image.open(
            meta["image_path"]
        ).convert("RGB")
        result = ret.search_by_image(image, top_k + 1)

        # 자기 자신 제외
        result["results"] = [
            r for r in result["results"]
            if r["image_id"] != image_id
        ][:top_k]
        result["total_results"] = len(result["results"])

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    return ImageSearchResponse(**result)