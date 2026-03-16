# app.py 전체 수정본

import os
import sys
import logging
import yaml
from pathlib import Path

import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download, HfApi

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data.dataset import FashionDataset
from src.models.clip_encoder import FashionCLIPEncoder
from src.search.indexer import FashionFAISSIndexer
from src.search.retriever import FashionRetriever

# ----------------------------------------
# 설정
# ----------------------------------------
with open(ROOT / "configs/config.yaml",
          encoding="utf-8") as f:
    config = yaml.safe_load(f)

DEVICE    = "cpu"
USE_FP16  = False
DATA_DIR  = ROOT / "data/raw"
INDEX_DIR = ROOT / "data/index"
INDEX_FT_DIR = ROOT / "data/index_finetuned"
LORA_DIR  = ROOT / "models/lora/best"

# ----------------------------------------
# 데이터 다운로드
# ----------------------------------------
def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "images").mkdir(parents=True, exist_ok=True)

    images_exist = any((DATA_DIR / "images").iterdir()) \
        if (DATA_DIR / "images").exists() else False

    if (DATA_DIR / "styles.csv").exists() and images_exist:
        logger.info("데이터 이미 존재! 스킵")
        return

    logger.info("HF Dataset 다운로드 중...")
    hf_hub_download(
        repo_id="sngdmtdkw-02/fashion-product-images",
        filename="styles.csv",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
    )
    snapshot_download(
        repo_id="sngdmtdkw-02/fashion-product-images",
        repo_type="dataset",
        local_dir=str(DATA_DIR / "images"),
        ignore_patterns=["*.csv", "*.md", ".gitattributes"],
    )
    logger.info("다운로드 완료!")

# ----------------------------------------
# 모델 & 인덱스 로드
# ----------------------------------------
download_data()

# Zero-shot 인코더
logger.info("Zero-shot 모델 로드 중...")
encoder_zero = FashionCLIPEncoder(
    model_name=config["model"]["clip_model"],
    device=DEVICE,
    use_fp16=USE_FP16,
)

# Fine-tuned 인코더 (LoRA)
logger.info("Fine-tuned 모델 로드 중...")
encoder_ft = FashionCLIPEncoder(
    model_name=config["model"]["clip_model"],
    device=DEVICE,
    use_fp16=USE_FP16,
    lora_path=str(LORA_DIR) if LORA_DIR.exists() else None,
)

def build_or_load_index(encoder, index_dir, label):
    """인덱스 로드 or 구축"""
    indexer = FashionFAISSIndexer(
        embed_dim=config["model"]["embed_dim"],
        nlist=50,
        nprobe=10,
    )
    retriever = FashionRetriever(encoder, indexer)
    index_path = Path(index_dir) / "fashion.index"

    if index_path.exists():
        logger.info(f"[{label}] 인덱스 로드 중...")
        retriever.load_index(str(index_dir))
    else:
        logger.info(f"[{label}] 인덱스 구축 중...")
        dataset = FashionDataset(
            data_dir=str(DATA_DIR),
            split="all",
            max_samples=5000,
        )
        retriever.build_index_from_dataset(
            dataset, batch_size=32
        )
        retriever.save_index(str(index_dir))

    return retriever

# 두 retriever 구축
retriever_zero = build_or_load_index(
    encoder_zero, str(INDEX_DIR), "Zero-shot"
)
retriever_ft = build_or_load_index(
    encoder_ft, str(INDEX_FT_DIR), "Fine-tuned"
)

logger.info("준비 완료!")

# ----------------------------------------
# 검색 함수
# ----------------------------------------
def results_to_gallery(results):
    gallery = []
    for r in results:
        meta     = r["metadata"]
        img_path = meta.get("image_path", "")
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
        except Exception:
            img = Image.new("RGB", (224, 224), (200, 200, 200))
        caption = (
            f"#{r['rank']} | 유사도: {r['score']:.3f}\n"
            f"{meta.get('article_type', '')} | "
            f"{meta.get('colour', '')}\n"
            f"{meta.get('name', '')[:30]}"
        )
        gallery.append((img, caption))
    return gallery


def search_image_zero(image, top_k):
    if image is None:
        return [], "이미지를 업로드해주세요!"
    try:
        result  = retriever_zero.search_by_image(image, int(top_k))
        gallery = results_to_gallery(result["results"])
        info    = f"검색 완료! | {result['search_time_ms']}ms | {result['total_results']}개"
        return gallery, info
    except Exception as e:
        return [], f"오류: {str(e)}"


def search_image_ft(image, top_k):
    if image is None:
        return [], "이미지를 업로드해주세요!"
    try:
        result  = retriever_ft.search_by_image(image, int(top_k))
        gallery = results_to_gallery(result["results"])
        info    = f"검색 완료! | {result['search_time_ms']}ms | {result['total_results']}개"
        return gallery, info
    except Exception as e:
        return [], f"오류: {str(e)}"


def search_text_zero(text, top_k):
    if not text or not text.strip():
        return [], "검색어를 입력해주세요!"
    try:
        result  = retriever_zero.search_by_text(text, int(top_k))
        gallery = results_to_gallery(result["results"])
        info    = f"검색 완료! | {result['search_time_ms']}ms | {result['total_results']}개"
        return gallery, info
    except Exception as e:
        return [], f"오류: {str(e)}"


def search_text_ft(text, top_k):
    if not text or not text.strip():
        return [], "검색어를 입력해주세요!"
    try:
        result  = retriever_ft.search_by_text(text, int(top_k))
        gallery = results_to_gallery(result["results"])
        info    = f"검색 완료! | {result['search_time_ms']}ms | {result['total_results']}개"
        return gallery, info
    except Exception as e:
        return [], f"오류: {str(e)}"


TEXT_EXAMPLES = [
    "casual blue shirt for men",
    "elegant formal dress for women",
    "white sport shoes for running",
    "black leather handbag",
    "navy blue jeans casual",
]

# ----------------------------------------
# Gradio UI
# ----------------------------------------
with gr.Blocks(title="Fashion Style Recommender") as demo:

    gr.Markdown("""
    # 👗 Fashion Style Recommender
    **CLIP 기반 패션 스타일 유사도 검색 시스템**

    | 모드 | 설명 | Image-Text 유사도 | Recall@10 |
    |------|------|:-----------------:|:---------:|
    | Zero-shot | 사전학습 CLIP | 0.20 | 77.6% |
    | **Fine-tuned** | LoRA 파인튜닝 | **0.75** | **98.0%** |
    ---
    """)

    # ----------------------------------------
    # Tab 1: Zero-shot vs Fine-tuned 이미지 비교
    # ----------------------------------------
    with gr.Tab("📸 이미지 검색 비교"):
        gr.Markdown("### 같은 이미지로 Zero-shot vs Fine-tuned 결과 비교!")
        with gr.Row():
            img_input = gr.Image(
                type="pil", label="검색할 이미지", height=280
            )
            img_top_k = gr.Slider(
                1, 20, value=6, step=1, label="결과 수"
            )

        with gr.Row():
            img_btn_zero = gr.Button(
                "🔍 Zero-shot 검색", variant="secondary"
            )
            img_btn_ft = gr.Button(
                "⚡ Fine-tuned 검색", variant="primary"
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Zero-shot 결과")
                img_gallery_zero = gr.Gallery(
                    columns=3, rows=2, height=400,
                    object_fit="contain",
                )
                img_info_zero = gr.Textbox(
                    label="검색 정보", interactive=False
                )
            with gr.Column():
                gr.Markdown("#### ⚡ Fine-tuned 결과")
                img_gallery_ft = gr.Gallery(
                    columns=3, rows=2, height=400,
                    object_fit="contain",
                )
                img_info_ft = gr.Textbox(
                    label="검색 정보", interactive=False
                )

        img_btn_zero.click(
            fn=search_image_zero,
            inputs=[img_input, img_top_k],
            outputs=[img_gallery_zero, img_info_zero],
        )
        img_btn_ft.click(
            fn=search_image_ft,
            inputs=[img_input, img_top_k],
            outputs=[img_gallery_ft, img_info_ft],
        )

    # ----------------------------------------
    # Tab 2: Zero-shot vs Fine-tuned 텍스트 비교
    # ----------------------------------------
    with gr.Tab("📝 텍스트 검색 비교"):
        gr.Markdown("### 같은 텍스트로 Zero-shot vs Fine-tuned 결과 비교!")
        with gr.Row():
            txt_input = gr.Textbox(
                label="스타일 설명 (영어)",
                placeholder="casual blue shirt for men",
                lines=2,
            )
            txt_top_k = gr.Slider(
                1, 20, value=6, step=1, label="결과 수"
            )

        with gr.Row():
            txt_btn_zero = gr.Button(
                "🔍 Zero-shot 검색", variant="secondary"
            )
            txt_btn_ft = gr.Button(
                "⚡ Fine-tuned 검색", variant="primary"
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Zero-shot 결과")
                txt_gallery_zero = gr.Gallery(
                    columns=3, rows=2, height=400,
                    object_fit="contain",
                )
                txt_info_zero = gr.Textbox(
                    label="검색 정보", interactive=False
                )
            with gr.Column():
                gr.Markdown("#### ⚡ Fine-tuned 결과")
                txt_gallery_ft = gr.Gallery(
                    columns=3, rows=2, height=400,
                    object_fit="contain",
                )
                txt_info_ft = gr.Textbox(
                    label="검색 정보", interactive=False
                )

        gr.Examples(
            examples=[[t, 6] for t in TEXT_EXAMPLES],
            inputs=[txt_input, txt_top_k],
        )
        txt_btn_zero.click(
            fn=search_text_zero,
            inputs=[txt_input, txt_top_k],
            outputs=[txt_gallery_zero, txt_info_zero],
        )
        txt_btn_ft.click(
            fn=search_text_ft,
            inputs=[txt_input, txt_top_k],
            outputs=[txt_gallery_ft, txt_info_ft],
        )

    gr.Markdown("""
    ---
    **기술 스택**: CLIP (ViT-B/32) | LoRA | FAISS IVF | Gradio

    **Fine-tuning**: lora_r=16, epochs=15, samples=7,000
    """)


if __name__ == "__main__":
    demo.launch()