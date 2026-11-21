"""Brand images: YouTube banner (2560x1440, everything important inside the 1546x423 safe area) and a profile picture.

    .venv\Scripts\python tools\brand.py [--name "Channel Name"] [--tagline "..."]
Outputs go to brand/. Without --name the banner carries only the tagline, so it survives a rename.
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "brand"
SLATE, SLATE2, STEEL, YELLOW, RED = "#121922", "#1b2531", "#8794a3", "#f2c14e", "#d4675a"
S = 2   # supersampling


def font(name, size):
    return ImageFont.truetype(str(ROOT / "fonts" / name), size * S)


def wave_points(x0, x1, y, spike_x, spike_h, seed=7):
    rnd = random.Random(seed)
    pts, v = [], 0.0
    for x in range(x0, x1 + 1, 4):
        v = 0.82 * v + rnd.uniform(-1, 1) * 5           # calm, slightly alive
        g = math.exp(-((x - spike_x) / 26) ** 2)         # one sharp peak: the moment
        pts.append((x * S, (y + v * (1 - g) - spike_h * g) * S))
    return pts


def mark(img, cx, cy, h, dot=True):
    """Play triangle with softly rounded corners (blur + threshold), optional red 'recording' dot top-right."""
    w = h * 0.9
    tri = [(cx - w * 0.42, cy - h / 2), (cx + w * 0.58, cy), (cx - w * 0.42, cy + h / 2)]
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).polygon([(x * S, y * S) for x, y in tri], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(h * 0.06 * S)).point(lambda v: 255 if v > 150 else 0)
    m = m.filter(ImageFilter.GaussianBlur(0.5 * S))
    img.paste(Image.new("RGB", img.size, YELLOW), (0, 0), m)
    if dot:
        d = h * 0.1
        x, y = cx + w * 0.5, cy - h * 0.42
        ImageDraw.Draw(img).ellipse([(x - d) * S, (y - d) * S, (x + d) * S, (y + d) * S], fill=RED)


def banner(name, tagline, sub):
    Wb, Hb = 2560, 1440
    img = Image.new("RGB", (Wb * S, Hb * S), SLATE)
    glow = Image.new("RGB", img.size, SLATE)
    ImageDraw.Draw(glow).ellipse([700 * S, 420 * S, 1860 * S, 1020 * S], fill=SLATE2)
    img = Image.blend(img, glow.filter(ImageFilter.GaussianBlur(160 * S)), 1.0)
    d = ImageDraw.Draw(img)
    sx0, sy0, sx1, sy1 = (Wb - 1546) // 2, (Hb - 423) // 2, (Wb + 1546) // 2, (Hb + 423) // 2   # YouTube safe area
    base_y = sy0 + 300
    spike_x = sx1 - 60
    d.line(wave_points(0, Wb, base_y, spike_x, 210), fill=YELLOW, width=5 * S, joint="curve")
    tip = (spike_x, base_y - 210)
    rr = 15
    d.ellipse([(tip[0] - rr) * S, (tip[1] - rr) * S, (tip[0] + rr) * S, (tip[1] + rr) * S], fill=RED)
    mark(img, sx0 + 130, sy0 + 160, 170, dot=False)
    d = ImageDraw.Draw(img)
    tx = sx0 + 270
    if name:
        f1, f2 = font("ArchivoBlack-Regular.ttf", 96), font("ui-Archivo.ttf", 40)
        d.text((tx * S, (sy0 + 70) * S), name, font=f1, fill="#f4f6f8")
        d.text((tx * S, (sy0 + 190) * S), tagline, font=f2, fill=STEEL)
    else:
        f1, f2 = font("ArchivoBlack-Regular.ttf", 64), font("ui-Archivo.ttf", 38)
        d.text((tx * S, (sy0 + 92) * S), tagline, font=f1, fill="#f4f6f8")
        d.text((tx * S, (sy0 + 186) * S), sub, font=f2, fill=STEEL)
    img = img.resize((Wb, Hb), Image.LANCZOS)
    return img, (sx0, sy0, sx1, sy1)


def profile():
    P = 800
    img = Image.new("RGB", (P * S, P * S), SLATE)
    mark(img, P / 2 - 20, P / 2 + 10, 380)
    return img.resize((P, P), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--tagline", default="The internet's best moments.")
    ap.add_argument("--sub", default="Cut to the second that matters. New clips every day.")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    img, (x0, y0, x1, y1) = banner(a.name, a.tagline, a.sub)
    img.save(OUT / "youtube_banner_2560x1440.png", optimize=True)
    # what each device actually shows, for checking
    img.crop((x0, y0, x1, y1)).save(OUT / "check_phone_1546x423.png")
    img.crop((0, y0, 2560, y1)).save(OUT / "check_desktop_2560x423.png")
    profile().save(OUT / "profile_800x800.png", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
