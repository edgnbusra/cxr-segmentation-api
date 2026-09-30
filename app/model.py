"""
"Mutfak": Modeli yükleme, ön işleme, tahmin ve son işleme.
FastAPI'den tamamen bağımsızdır; test_model.py de bu modülü kullanır.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "ianpan/chest-x-ray-basic"
# Kodunu okuyup incelediğimiz sürüm. Model sahibi kodu değiştirse bile biz bu sürümü kullanırız.
MODEL_REVISION = "fec189cd739afa79402a78ce41b46b4d35d784ab"

CLASS_NAMES = {1: "sag_akciger", 2: "sol_akciger", 3: "kalp"}
# BGR formatında renkler (OpenCV BGR kullanır): mavi, yeşil, kırmızı
CLASS_COLORS = {1: (255, 144, 30), 2: (80, 200, 60), 3: (60, 60, 230)}


@dataclass
class Prediction:
    image: np.ndarray       # orijinal gri görüntü (H, W)
    mask: np.ndarray        # etiket maskesi (H, W), değerler 0..3
    inference_ms: float


class SegmentationModel:
    def __init__(self) -> None:
        self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        # İlk çalıştırmada Hugging Face'ten indirir, sonra ~/.cache/huggingface önbelleğinden okur
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
        self.model.eval()

    def predict(self, img: np.ndarray) -> Prediction:
        """img: gri tonlamalı görüntü (H, W), uint8."""
        h, w = img.shape
        x = self.model.preprocess(img)                  # (320, 320)
        x = torch.from_numpy(x).float()[None, None]     # (1, 1, 320, 320) -> batch ve kanal boyutu

        t0 = time.perf_counter()
        with torch.inference_mode():                    # gradyan yok -> daha hızlı, daha az bellek
            out = self.model(x)
        inference_ms = (time.perf_counter() - t0) * 1000

        # (1, 4, 320, 320) olasılık -> her piksel için en olası sınıf (320, 320)
        mask = out["mask"].argmax(1)[0].numpy().astype(np.uint8)
        # NEAREST: etiketler büyütülürken 1.5 gibi anlamsız ara değerler oluşmasın
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return Prediction(image=img, mask=mask, inference_ms=inference_ms)


def decode_image(data: bytes) -> np.ndarray | None:
    """Ham dosya baytlarını (PNG/JPG...) gri görüntüye çevirir. Geçersizse None."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def colorize(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, bgr in CLASS_COLORS.items():
        color[mask == cls] = bgr
    return color


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    return cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1 - alpha, colorize(mask), alpha, 0)


def calculate_ctr(mask: np.ndarray) -> float | None:
    """Kardiyotorasik oran = kalbin en geniş yeri / akciğerlerin en geniş yeri (yatay)."""
    lung_x = np.where((mask == 1) | (mask == 2))[1]
    heart_x = np.where(mask == 3)[1]
    if lung_x.size == 0 or heart_x.size == 0:
        return None
    return float((heart_x.max() - heart_x.min()) / (lung_x.max() - lung_x.min()))


def area_ratios(mask: np.ndarray) -> dict[str, float]:
    """Her sınıfın görüntüdeki alan oranı (0-1)."""
    return {name: float((mask == cls).mean()) for cls, name in CLASS_NAMES.items()}
