# src/detection/detector.py

"""
YOLOv8 기반 패션 아이템 탐지기

역할:
- 이미지에서 패션 아이템 위치 탐지
- Bounding Box + 클래스 + 신뢰도 반환
- 탐지된 아이템 크롭 (CLIP 검색용)

실무 포인트:
- YOLOv8 파인튜닝으로 패션 도메인 특화
- NMS로 중복 박스 제거 (자동 처리!)
- 탐지된 크롭 → CLIP 임베딩 → FAISS 검색
  → PROJECT 1과 통합!

자료구조:
- List: 탐지 결과 순서 관리
- Dict: 탐지 결과 포맷
  {class, confidence, bbox, crop}
- numpy array: 이미지 처리
"""

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# 패션 클래스 정의
FASHION_CLASSES = [
    "Exposed",
    "Female",
    "Flats",
    "High Heel",
    "Hoodie",
    "Jacket",
    "Long Pants",
    "Long Shirt",
    "Long Skirt",
    "Male",
    "Shirt",
    "Short",
    "Short Skirt",
    "SleevelessShirt",
    "Slipper",
    "Sport Shoes",
]

# 검색에 유용한 아이템 클래스
# (성별/노출 클래스 제외)
SEARCHABLE_CLASSES = [
    "Flats",
    "High Heel",
    "Hoodie",
    "Jacket",
    "Long Pants",
    "Long Shirt",
    "Long Skirt",
    "Shirt",
    "Short",
    "Short Skirt",
    "SleevelessShirt",
    "Slipper",
    "Sport Shoes",
]

# 클래스별 색상 (시각화용)
CLASS_COLORS = {
    "Shirt":          (255, 100, 100),
    "Long Shirt":     (255, 150, 100),
    "SleevelessShirt":(255, 200, 100),
    "Hoodie":         (255, 100, 150),
    "Jacket":         (200, 100, 255),
    "Long Pants":     (100, 100, 255),
    "Short":          (100, 150, 255),
    "Long Skirt":     (100, 255, 150),
    "Short Skirt":    (100, 255, 200),
    "Flats":          (255, 255, 100),
    "High Heel":      (255, 200, 100),
    "Slipper":        (200, 255, 100),
    "Sport Shoes":    (100, 255, 100),
    "Female":         (255, 100, 255),
    "Male":           (100, 255, 255),
    "Exposed":        (200, 200, 200),
}


class FashionDetector:
    """
    YOLOv8 기반 패션 아이템 탐지기

    파이프라인:
    이미지 입력
        ↓
    YOLOv8 추론
        ↓
    NMS (중복 박스 제거) ← 자동!
        ↓
    결과: [
        {
            class_name:  "Shirt",
            confidence:  0.95,
            bbox:        [x1, y1, x2, y2],
            crop:        PIL Image,
        },
        ...
    ]
        ↓
    각 crop → CLIP 임베딩 → FAISS 검색
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ):
        """
        Args:
            model_path:     YOLOv8 모델 경로
                            'yolov8n.pt'      → 기본 COCO 모델
                            'models/yolo/...' → 파인튜닝 모델
            confidence:     최소 신뢰도 (낮을수록 더 많이 탐지)
            iou_threshold:  NMS IoU 임계값
            device:         'cuda' | 'cpu'
        """
        self.confidence    = confidence
        self.iou_threshold = iou_threshold
        self.device        = device

        logger.info(
            f"[FashionDetector] 모델 로딩: {model_path}"
        )
        self.model = YOLO(model_path)
        logger.info(
            f"[FashionDetector] 로드 완료! | "
            f"confidence={confidence} | "
            f"iou={iou_threshold}"
        )

    def detect(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        only_searchable: bool = True,
    ) -> list:
        """
        이미지에서 패션 아이템 탐지

        Args:
            image:           PIL Image, 경로, numpy array
            only_searchable: True → 검색 가능한 클래스만
                             (성별/노출 클래스 제외)

        Returns:
            detections: [
                {
                    "rank":       1,
                    "class_id":   10,
                    "class_name": "Shirt",
                    "confidence": 0.95,
                    "bbox":       [x1, y1, x2, y2],
                    "crop":       PIL Image (크롭된 이미지),
                },
                ...
            ]
            신뢰도 내림차순 정렬!

        자료구조:
        - List: 탐지 결과 순서 관리
        - Dict: 각 탐지 결과 포맷
        """
        # 이미지 로드
        pil_image = self._load_image(image)
        img_array = np.array(pil_image)

        # YOLOv8 추론
        results = self.model.predict(
            source=img_array,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                # 클래스명
                if hasattr(result, 'names'):
                    class_name = result.names[class_id]
                else:
                    class_name = FASHION_CLASSES[class_id] \
                        if class_id < len(FASHION_CLASSES) \
                        else f"class_{class_id}"

                # 검색 가능 클래스 필터링
                if only_searchable and \
                   class_name not in SEARCHABLE_CLASSES:
                    continue

                # 크롭 이미지 추출
                crop = self._crop_image(
                    pil_image, x1, y1, x2, y2
                )

                detections.append({
                    "class_id":   class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox":       [x1, y1, x2, y2],
                    "crop":       crop,
                })

        # 신뢰도 내림차순 정렬
        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        # rank 추가
        for i, det in enumerate(detections, 1):
            det["rank"] = i

        logger.info(
            f"[FashionDetector] 탐지 완료: "
            f"{len(detections)}개 아이템"
        )

        return detections

    def detect_batch(
        self,
        images: list,
        only_searchable: bool = True,
    ) -> list:
        """
        배치 이미지 탐지
        여러 이미지를 한 번에 처리!

        Args:
            images: PIL Image 리스트

        Returns:
            batch_detections: 이미지별 탐지 결과 리스트
        """
        batch_results = []
        for image in images:
            detections = self.detect(
                image,
                only_searchable=only_searchable
            )
            batch_results.append(detections)
        return batch_results

    def visualize(
        self,
        image: Union[Image.Image, str, Path],
        detections: list,
        show_confidence: bool = True,
    ) -> Image.Image:
        """
        탐지 결과 시각화
        Bounding Box + 클래스명 + 신뢰도 표시

        Args:
            image:      원본 이미지
            detections: detect() 반환값
            show_confidence: 신뢰도 표시 여부

        Returns:
            visualized: PIL Image (박스 그려진 이미지)
        """
        pil_image = self._load_image(image)
        img_array = np.array(pil_image).copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name     = det["class_name"]
            confidence     = det["confidence"]

            # 클래스별 색상
            color = CLASS_COLORS.get(
                class_name, (100, 255, 100)
            )

            # Bounding Box 그리기
            cv2.rectangle(
                img_array,
                (x1, y1), (x2, y2),
                color, 2
            )

            # 레이블 텍스트
            label = (
                f"{class_name} {confidence:.2f}"
                if show_confidence
                else class_name
            )

            # 텍스트 배경
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 2
            )
            cv2.rectangle(
                img_array,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                color, -1
            )

            # 텍스트
            cv2.putText(
                img_array,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

        return Image.fromarray(img_array)

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "model":         str(self.model.model),
            "confidence":    self.confidence,
            "iou_threshold": self.iou_threshold,
            "device":        self.device,
            "classes":       FASHION_CLASSES,
            "n_classes":     len(FASHION_CLASSES),
        }

    def _load_image(
        self,
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> Image.Image:
        """이미지 로드 헬퍼"""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(
                f"지원하지 않는 이미지 타입: {type(image)}"
            )

    def _crop_image(
        self,
        image: Image.Image,
        x1: int, y1: int,
        x2: int, y2: int,
        padding: float = 0.05,
    ) -> Image.Image:
        """
        Bounding Box 크롭 + 패딩

        패딩 추가 이유:
        박스 경계가 너무 딱 맞으면
        CLIP 임베딩 품질 저하!
        약간 여유 있게 크롭!
        """
        w = x2 - x1
        h = y2 - y1

        # 패딩 추가
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        img_w, img_h = image.size

        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(img_w, x2 + pad_x)
        y2_pad = min(img_h, y2 + pad_y)

        return image.crop(
            (x1_pad, y1_pad, x2_pad, y2_pad)
        )