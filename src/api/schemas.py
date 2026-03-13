# src/api/schemas.py

"""
FastAPI Pydantic 스키마 정의

역할:
- API 요청/응답 데이터 구조 정의
- 자동 유효성 검사
- API 문서 자동 생성 (Swagger UI)

실무 포인트:
- 입력 검증을 Pydantic이 자동으로!
- 잘못된 요청 → 422 에러 자동 반환
- /docs 에서 Swagger UI로 API 테스트 가능!
"""

from typing import Optional
from pydantic import BaseModel, Field


# ----------------------------------------
# 검색 결과 메타데이터
# ----------------------------------------
class FashionMetadata(BaseModel):
    id:           Optional[int]    = None
    image_path:   Optional[str]    = None
    gender:       Optional[str]    = None
    category:     Optional[str]    = None
    sub_category: Optional[str]    = None
    article_type: Optional[str]    = None
    colour:       Optional[str]    = None
    season:       Optional[str]    = None
    usage:        Optional[str]    = None
    name:         Optional[str]    = None
    text:         Optional[str]    = None


# ----------------------------------------
# 단일 검색 결과
# ----------------------------------------
class SearchResult(BaseModel):
    rank:     int
    score:    float
    image_id: int
    metadata: FashionMetadata


# ----------------------------------------
# 이미지 검색 응답
# ----------------------------------------
class ImageSearchResponse(BaseModel):
    query_type:     str
    search_time_ms: float
    total_results:  int
    results:        list[SearchResult]


# ----------------------------------------
# 텍스트 검색 요청
# ----------------------------------------
class TextSearchRequest(BaseModel):
    text:  str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="검색할 스타일 설명",
        examples=["casual blue denim jacket for men"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="반환할 결과 수 (1~50)",
    )


# ----------------------------------------
# 텍스트 검색 응답
# ----------------------------------------
class TextSearchResponse(BaseModel):
    query_type:     str
    query:          str
    search_time_ms: float
    total_results:  int
    results:        list[SearchResult]


# ----------------------------------------
# 서버 상태 응답
# ----------------------------------------
class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    index_loaded:  bool
    index_size:    int
    device:        str
    embed_dim:     int