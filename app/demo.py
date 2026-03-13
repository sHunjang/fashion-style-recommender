# app/demo.py

"""
Gradio 웹 데모
HuggingFace Spaces 배포용

기능:
① 이미지 업로드 → 유사 스타일 검색
② 텍스트 입력  → Zero-shot 이미지 검색
③ 검색 결과 갤러리로 시각화

실행:
python app/demo.py
"""

import io
import logging
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import yaml
from PIL import Image

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import FashionDataset
from src.models.clip_encoder import FashionCLIPEncoder
from src.search.indexer import FashionFAISSIndexer
from src.search.retriever import FashionRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------
# 설정 로드
# ----------------------------------------
with open(ROOT / "configs/config.yaml",
          encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ----------------------------------------
# 모델 & 인덱스 로드 (전역 1회)
# ----------------------------------------
logger.info("모델 로딩 중...")

encoder = FashionCLIPEncoder(
    model_name=config["model"]["clip_model"],
    device=config["model"]["device"],
    use_fp16=True,
)
indexer = FashionFAISSIndexer(
    embed_dim=config["model"]["embed_dim"],
    nlist=config["search"]["nlist"],
    nprobe=config["search"]["nprobe"],
)
retriever = FashionRetriever(encoder, indexer)

# 인덱스 로드 or 구축
index_dir = str(ROOT / config["data"]["index_dir"])
index_path = Path(index_dir) / "fashion.index"

if index_path.exists():
    logger.info("기존 인덱스 로드 중...")
    retriever.load_index(index_dir)
else:
    logger.info("인덱스 구축 중...")
    dataset = FashionDataset(
        data_dir=str(ROOT / config["data"]["data_dir"]),
        split="all",
        max_samples=config["data"]["max_images"],
    )
    retriever.build_index_from_dataset(dataset)
    retriever.save_index(index_dir)

logger.info("준비 완료!")

# ----------------------------------------
# 검색 결과 → 이미지 갤러리 변환
# ----------------------------------------
def results_to_gallery(results: list) -> list:
    """
    검색 결과 → Gradio 갤러리 형식 변환

    Returns:
        [(PIL Image, caption), ...]
    """
    gallery = []
    for r in results:
        meta = r["metadata"]
        img_path = meta.get("image_path", "")

        try:
            img = Image.open(img_path).convert("RGB")
            # 갤러리용 리사이즈
            img = img.resize((224, 224))
        except Exception:
            # 이미지 로드 실패 시 빈 이미지
            img = Image.new("RGB", (224, 224), (200, 200, 200))

        # 캡션 생성
        caption = (
            f"#{r['rank']} | "
            f"유사도: {r['score']:.3f}\n"
            f"{meta.get('article_type', '')} | "
            f"{meta.get('colour', '')}\n"
            f"{meta.get('name', '')[:30]}"
        )
        gallery.append((img, caption))

    return gallery


# ----------------------------------------
# 이미지 검색 함수
# ----------------------------------------
def search_by_image(
    image: Image.Image,
    top_k: int,
) -> tuple:
    """
    이미지 업로드 → 유사 스타일 검색

    Returns:
        (갤러리, 검색 정보 텍스트)
    """
    if image is None:
        return [], "이미지를 업로드해주세요!"

    try:
        result  = retriever.search_by_image(image, top_k)
        gallery = results_to_gallery(result["results"])
        info    = (
            f"검색 완료! | "
            f"검색 시간: {result['search_time_ms']}ms | "
            f"결과: {result['total_results']}개"
        )
        return gallery, info

    except Exception as e:
        logger.error(f"검색 오류: {e}")
        return [], f"검색 오류: {str(e)}"


# ----------------------------------------
# 텍스트 검색 함수
# ----------------------------------------
def search_by_text(
    text: str,
    top_k: int,
) -> tuple:
    """
    텍스트 입력 → 유사 스타일 검색

    Returns:
        (갤러리, 검색 정보 텍스트)
    """
    if not text or not text.strip():
        return [], "검색어를 입력해주세요!"

    try:
        result  = retriever.search_by_text(text, top_k)
        gallery = results_to_gallery(result["results"])
        info    = (
            f"검색 완료! | "
            f"검색 시간: {result['search_time_ms']}ms | "
            f"결과: {result['total_results']}개"
        )
        return gallery, info

    except Exception as e:
        logger.error(f"검색 오류: {e}")
        return [], f"검색 오류: {str(e)}"


# ----------------------------------------
# 텍스트 예시 목록
# ----------------------------------------
TEXT_EXAMPLES = [
    "casual blue shirt for men",
    "elegant formal dress for women",
    "white sport shoes for running",
    "black leather handbag",
    "navy blue jeans casual",
    "floral summer dress",
    "formal black suit for men",
    "red sneakers streetwear",
]

# ----------------------------------------
# Gradio UI 구성
# ----------------------------------------
with gr.Blocks(
    title="Fashion Style Recommender",
) as demo:

    # 헤더
    gr.Markdown("""
    # 👗 Fashion Style Recommender
    **CLIP 기반 패션 스타일 유사도 검색 시스템**

    - 📸 **이미지 검색**: 패션 이미지를 업로드하면 유사한 스타일을 찾아드립니다
    - 📝 **텍스트 검색**: 원하는 스타일을 텍스트로 설명하면 관련 상품을 찾아드립니다
    ---
    """)

    # ----------------------------------------
    # Tab 1: 이미지 검색
    # ----------------------------------------
    with gr.Tab("📸 이미지로 검색"):
        gr.Markdown("### 패션 이미지를 업로드하세요!")

        with gr.Row():
            # 입력
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="pil",
                    label="검색할 이미지",
                    height=300,
                )
                img_top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=6,
                    step=1,
                    label="결과 수",
                )
                img_btn = gr.Button(
                    "🔍 유사 스타일 검색",
                    variant="primary",
                    size="lg",
                )
                img_info = gr.Textbox(
                    label="검색 정보",
                    interactive=False,
                )

            # 출력
            with gr.Column(scale=2):
                img_gallery = gr.Gallery(
                    label="유사 스타일 결과",
                    columns=3,
                    rows=2,
                    height=500,
                    object_fit="contain",
                )

        # 이미지 예시
        gr.Markdown("#### 예시 이미지")
        img_examples = gr.Examples(
            examples=[
                [str(ROOT / "data/raw/images/15970.jpg"), 6],
                [str(ROOT / "data/raw/images/39386.jpg"), 6],
                [str(ROOT / "data/raw/images/59263.jpg"), 6],
            ],
            inputs=[img_input, img_top_k],
            label="클릭해서 테스트!",
        )

        # 이벤트 연결
        img_btn.click(
            fn=search_by_image,
            inputs=[img_input, img_top_k],
            outputs=[img_gallery, img_info],
        )

    # ----------------------------------------
    # Tab 2: 텍스트 검색
    # ----------------------------------------
    with gr.Tab("📝 텍스트로 검색"):
        gr.Markdown("### 원하는 스타일을 텍스트로 설명하세요!")

        with gr.Row():
            # 입력
            with gr.Column(scale=1):
                txt_input = gr.Textbox(
                    label="스타일 설명 (영어)",
                    placeholder="casual blue shirt for men",
                    lines=2,
                )
                txt_top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=6,
                    step=1,
                    label="결과 수",
                )
                txt_btn = gr.Button(
                    "🔍 스타일 검색",
                    variant="primary",
                    size="lg",
                )
                txt_info = gr.Textbox(
                    label="검색 정보",
                    interactive=False,
                )

            # 출력
            with gr.Column(scale=2):
                txt_gallery = gr.Gallery(
                    label="검색 결과",
                    columns=3,
                    rows=2,
                    height=500,
                    object_fit="contain",
                )

        # 텍스트 예시
        gr.Markdown("#### 예시 검색어 (클릭해서 테스트!)")
        gr.Examples(
            examples=[[t, 6] for t in TEXT_EXAMPLES],
            inputs=[txt_input, txt_top_k],
            label="",
        )

        # 이벤트 연결
        txt_btn.click(
            fn=search_by_text,
            inputs=[txt_input, txt_top_k],
            outputs=[txt_gallery, txt_info],
        )

    # ----------------------------------------
    # 푸터
    # ----------------------------------------
    gr.Markdown("""
    ---
    **기술 스택**: CLIP (ViT-B/32) | FAISS IVF | FastAPI | Gradio

    **데이터**: Fashion Product Images (Kaggle) - 44,424장
    """)


# ----------------------------------------
# 실행
# ----------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        # share=True: 외부 공유 링크 생성
        theme=gr.themes.Soft()
        # HuggingFace Spaces 배포 시 False
    )