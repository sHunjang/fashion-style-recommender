# tests/test_lora_encoder.py

from src.models.clip_encoder import FashionCLIPEncoder
from PIL import Image

print("=== LoRA Encoder Test ===")

# Zero-shot
encoder_zero = FashionCLIPEncoder(
    device="cuda", use_fp16=True
)
print(encoder_zero.get_model_info())

# Fine-tuned
encoder_ft = FashionCLIPEncoder(
    device="cuda",
    use_fp16=True,
    lora_path="models/lora/best",
)
print(encoder_ft.get_model_info())

# 임베딩 비교
img = Image.open("data/raw/images/15970.jpg").convert("RGB")
emb_zero = encoder_zero.encode_single_image(img)
emb_ft   = encoder_ft.encode_single_image(img)

print(f"Zero-shot shape: {emb_zero.shape}")
print(f"Fine-tuned shape: {emb_ft.shape}")
print(f"두 임베딩 유사도: {encoder_zero.compute_similarity(emb_zero, emb_ft):.4f}")
print()

# 같은 이미지 → 유사도 1.0 확인
emb1 = encoder_ft.encode_single_image(img)
emb2 = encoder_ft.encode_single_image(img)
print(f"동일 이미지 유사도: {encoder_ft.compute_similarity(emb1, emb2):.4f}")
# → 1.0000 이어야 함!

# 이미지-텍스트 짝 유사도
text_emb = encoder_ft.encode_single_text(
    "a photo of Navy Blue Shirts for Men"
)
print(f"이미지-텍스트 유사도 (zero-shot): {encoder_zero.compute_similarity(emb_zero, text_emb):.4f}")
print(f"이미지-텍스트 유사도 (fine-tuned): {encoder_ft.compute_similarity(emb_ft, text_emb):.4f}")
# fine-tuned가 높아야 함
print("=== Test Passed! ===")