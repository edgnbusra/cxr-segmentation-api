"""
Modeli FastAPI'den bağımsız olarak test et (app/model.py'deki "mutfağı" doğrudan kullanır).
Çalıştırma:  .venv/Scripts/python test_model.py samples/chest_xray_1.jpg
"""
import sys
import time
from pathlib import Path

import cv2

from app.model import SegmentationModel, area_ratios, calculate_ctr, colorize, overlay


def main(image_path: str) -> None:
    t0 = time.perf_counter()
    segmenter = SegmentationModel()
    segmenter.load()
    print(f"Model yüklendi: {time.perf_counter() - t0:.1f} sn")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Görüntü okunamadı: {image_path}")
    pred = segmenter.predict(img)
    print(f"Inference: {pred.inference_ms:.0f} ms")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), colorize(pred.mask))
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay(pred.image, pred.mask))

    for name, ratio in area_ratios(pred.mask).items():
        print(f"  {name:12s}: görüntünün %{100 * ratio:.1f}'i")
    ctr = calculate_ctr(pred.mask)
    print(f"  CTR         : {ctr:.2f}  (normal < 0.50)" if ctr is not None else "  CTR         : hesaplanamadı")
    print(f"Kaydedildi: outputs/{stem}_mask.png, outputs/{stem}_overlay.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/chest_xray_1.jpg")
