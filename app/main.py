"""
"Garson": HTTP isteklerini alır, modele iletir, cevabı döndürür.
Çalıştırma:  .venv/Scripts/uvicorn app.main:app --port 8000
Test arayüzü: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from enum import Enum

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.model import (
    MODEL_ID,
    Prediction,
    SegmentationModel,
    area_ratios,
    calculate_ctr,
    colorize,
    decode_image,
    overlay,
)

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

segmenter = SegmentationModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sunucu açılırken BİR KEZ çalışır: model her istekte değil, burada yüklenir
    segmenter.load()
    yield
    # Sunucu kapanırken buraya düşer (temizlenecek kaynak yok)


app = FastAPI(
    title="Göğüs Röntgeni Segmentasyon API",
    description=(
        f"Hugging Face modeli `{MODEL_ID}` ile sağ akciğer, sol akciğer ve kalp segmentasyonu.\n\n"
        "**Uyarı:** Yalnızca araştırma/eğitim amaçlıdır, klinik kullanım için onaylı değildir."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class OutputType(str, Enum):
    mask = "mask"        # renkli maske
    overlay = "overlay"  # röntgen + yarı saydam maske
    raw = "raw"          # piksel değerleri 0-3 olan ham etiket maskesi (başka programlar için)


def run_prediction(file: UploadFile) -> Prediction:
    """Dosyayı doğrula, görüntüye çevir ve modelden geçir."""
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Dosya 10 MB sınırını aşıyor.")
    img = decode_image(data)
    if img is None:
        raise HTTPException(status_code=400, detail="Dosya okunamadı. PNG veya JPG bir görüntü gönderin.")
    return segmenter.predict(img)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": segmenter.is_loaded}


# Not: Endpoint'ler "async def" değil düz "def". Model CPU'yu meşgul eden, senkron bir iştir;
# FastAPI düz "def" fonksiyonları ayrı bir iş parçacığında çalıştırır, böylece sunucu kilitlenmez.
@app.post(
    "/predict",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Segmentasyon maskesi (PNG)"}},
)
def predict(
    file: UploadFile = File(..., description="Göğüs röntgeni (PNG/JPG)"),
    output: OutputType = Query(OutputType.overlay, description="Döndürülecek görüntü türü"),
):
    pred = run_prediction(file)
    if output == OutputType.mask:
        result = colorize(pred.mask)
    elif output == OutputType.overlay:
        result = overlay(pred.image, pred.mask)
    else:
        result = pred.mask

    ok, png = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Sonuç görüntüsü oluşturulamadı.")
    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={"X-Inference-Ms": f"{pred.inference_ms:.0f}"},
    )


@app.post("/analyze")
def analyze(file: UploadFile = File(..., description="Göğüs röntgeni (PNG/JPG)")):
    pred = run_prediction(file)
    ctr = calculate_ctr(pred.mask)
    return {
        "filename": file.filename,
        "image_size": {"height": pred.image.shape[0], "width": pred.image.shape[1]},
        "area_ratios": {k: round(v, 4) for k, v in area_ratios(pred.mask).items()},
        "cardiothoracic_ratio": round(ctr, 3) if ctr is not None else None,
        "ctr_interpretation": None if ctr is None else ("normal (<0.50)" if ctr < 0.5 else "artmış (>=0.50)"),
        "inference_ms": round(pred.inference_ms, 1),
        "disclaimer": "Araştırma/eğitim amaçlıdır, klinik tanı için kullanılamaz.",
    }
