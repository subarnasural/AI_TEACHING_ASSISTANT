import io
from typing import Any, Dict

import cv2
import numpy as np
import pytesseract
from PIL import Image


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary_img > 0))
    if len(coords) == 0:
        return binary_img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _preprocess_for_ocr(image_bytes: bytes) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale for small/low-resolution text.
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    # Noise removal + adaptive thresholding for uneven lighting.
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return _deskew(thresh)


def _ocr_with_confidence(processed_img: np.ndarray) -> Dict[str, Any]:
    data = pytesseract.image_to_data(
        processed_img,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",
    )
    words = []
    confidences = []

    for txt, conf in zip(data["text"], data["conf"]):
        cleaned = (txt or "").strip()
        if not cleaned:
            continue
        words.append(cleaned)
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = -1
        if conf_val >= 0:
            confidences.append(conf_val)

    text = " ".join(words).strip()
    confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    return {"text": text, "confidence": round(confidence, 2)}


def extract_text_from_image(image_file) -> Dict[str, Any]:
    """
    OCR with preprocessing, confidence score, and retry fallback.
    Returns dict: {text, confidence, method, error}
    """
    try:
        image_bytes = image_file.getvalue()
        processed = _preprocess_for_ocr(image_bytes)

        primary = _ocr_with_confidence(processed)

        # Fallback retry with different page segmentation for sparse/irregular layouts.
        if not primary["text"] or primary["confidence"] < 35:
            fallback_data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 11",
            )
            words = [w.strip() for w in fallback_data["text"] if (w or "").strip()]
            text = " ".join(words).strip()
            if text and len(text) > len(primary["text"]):
                return {
                    "text": text,
                    "confidence": primary["confidence"],
                    "method": "fallback_psm11",
                    "error": None,
                }

        return {
            "text": primary["text"],
            "confidence": primary["confidence"],
            "method": "preprocessed_psm6",
            "error": None,
        }
    except Exception as exc:
        return {
            "text": "",
            "confidence": 0.0,
            "method": "error",
            "error": str(exc),
        }
