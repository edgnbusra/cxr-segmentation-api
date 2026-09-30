"""
README için demo GIF'i oluşturur. Görüntüleri doğrudan çalışan API'den alır.
Önce sunucuyu başlat:  .venv/Scripts/uvicorn app.main:app --port 8001
Sonra:                 .venv/Scripts/python make_gif.py --url http://localhost:8001
"""
import argparse
import io
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

SAMPLES = ["samples/chest_xray_1.jpg", "samples/chest_xray_2.jpg"]
WIDTH = 512
BAR_H = 64
LEGEND = [("Sag akciger", (30, 144, 255)), ("Sol akciger", (60, 200, 80)), ("Kalp", (230, 60, 60))]


def call_api(url: str, path: str) -> tuple[Image.Image, Image.Image, dict, str]:
    with open(path, "rb") as f:
        data = f.read()
    files = {"file": (Path(path).name, data, "image/jpeg")}
    r = requests.post(f"{url}/predict", params={"output": "overlay"}, files=files, timeout=60)
    r.raise_for_status()
    overlay = Image.open(io.BytesIO(r.content)).convert("RGB")
    ms = r.headers.get("X-Inference-Ms", "?")
    stats = requests.post(f"{url}/analyze", files=files, timeout=60).json()
    original = Image.open(io.BytesIO(data)).convert("RGB")
    return original, overlay, stats, ms


def resize(img: Image.Image) -> Image.Image:
    h = round(img.height * WIDTH / img.width)
    return img.resize((WIDTH, h), Image.LANCZOS)


def with_caption(img: Image.Image, line1: str, line2: str, show_legend: bool, height: int) -> Image.Image:
    # Tüm kareler aynı yükseklikte olsun diye görüntüyü ortalayıp alta bilgi şeridi ekliyoruz
    canvas = Image.new("RGB", (WIDTH, height + BAR_H), (18, 18, 24))
    canvas.paste(img, (0, (height - img.height) // 2))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    y0 = height + 8
    d.text((12, y0), line1, fill=(240, 240, 240), font=font)
    d.text((12, y0 + 26), line2, fill=(170, 170, 185), font=small)
    if show_legend:
        x = WIDTH - 12
        for name, rgb in reversed(LEGEND):
            w = d.textlength(name, font=small)
            x -= w
            d.text((x, y0 + 26), name, fill=(220, 220, 220), font=small)
            x -= 16
            d.rectangle([x, y0 + 29, x + 11, y0 + 40], fill=rgb)
            x -= 14
    return canvas


def main(url: str, out: str) -> None:
    results = [call_api(url, p) for p in SAMPLES]
    height = max(resize(o).height for o, *_ in results)
    frames, durations = [], []

    for path, (original, overlay, stats, ms) in zip(SAMPLES, results):
        original, overlay = resize(original), resize(overlay)
        name = Path(path).name
        ctr = stats["cardiothoracic_ratio"]

        # 1) Orijinal röntgen
        frames.append(with_caption(original, f"Girdi: {name}", "POST /predict  ->  isleniyor...", False, height))
        durations.append(1200)
        # 2) Maske yavaşça beliriyor
        for i in range(1, 9):
            blended = Image.blend(original, overlay, i / 8)
            frames.append(with_caption(blended, f"Girdi: {name}", f"Inference: {ms} ms (CPU)", True, height))
            durations.append(90)
        # 3) Sonuç
        frames.append(with_caption(overlay, f"Kardiyotorasik oran (CTR): {ctr:.2f}  |  {ms} ms",
                                   "ianpan/chest-x-ray-basic", True, height))
        durations.append(2500)

    Path(out).parent.mkdir(exist_ok=True)
    frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{out}: {len(frames)} kare, {Path(out).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--out", default="assets/demo.gif")
    args = parser.parse_args()
    main(args.url, args.out)
