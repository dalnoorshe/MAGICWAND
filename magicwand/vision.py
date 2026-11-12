"""MAGICWAND — vision.

The research agents can look at pictures. Two ways an image gets here:

  1. RENDERED   — we draw the candlestick chart ourselves from the same bars the
                  strategy trades. Deterministic, reproducible, no third-party
                  site, and the agent cannot see a different chart than the
                  strategy saw.
  2. CAPTURED   — a real browser screenshot of a public page, if a headless
                  browser is available. Best-effort; we do not log in anywhere.

Why render rather than scrape a trading site: a screenshot of somebody else's
page depends on their layout, their bot detection, their login wall, and their
terms. A chart drawn from our own data has none of those failure modes and is
exactly reproducible — which matters when you are grading a claim later.

The image is UNTRUSTED INPUT to the model, in the strongest sense: a chart can
contain rendered text, and rendered text can contain instructions. Everything
the vision agent concludes goes through the same schema validation and the same
veto-only rule as every other reader. An image can never open a trade.
"""

from __future__ import annotations

import base64
import io
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .common import Bar, iso, utcnow


class VisionError(RuntimeError):
    pass


@dataclass
class Image:
    data: bytes
    mime: str
    source: str          # "rendered-chart" | "browser-screenshot" | "file"
    note: str = ""
    width: int = 0
    height: int = 0

    @property
    def data_url(self) -> str:
        b64 = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime};base64,{b64}"

    @property
    def kb(self) -> float:
        return len(self.data) / 1024.0


# --------------------------------------------------------------------------
# Chart rendering (PIL only, no matplotlib)
# --------------------------------------------------------------------------

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
)


def _font(size: int):
    """A real font at a readable size. The PIL bitmap default is 8px — far too
    small for a vision model to read axis labels, which is the whole point of
    the image."""
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _require_pil():
    try:
        from PIL import Image as PILImage  # noqa: F401
        from PIL import ImageDraw, ImageFont
        return PILImage, ImageDraw, ImageFont
    except ImportError as e:
        raise VisionError("Pillow is required to render charts") from e


def render_chart(bars: list[Bar], title: str = "SOL-USD 1h",
                 width: int = 1280, height: int = 720,
                 max_bars: int = 120, ma_fast: int = 12,
                 ma_slow: int = 48) -> Image:
    """Draw a candlestick chart with moving averages and volume.

    Pure PIL so it works anywhere Python does. Colours are chosen for a dark
    background, which is also what the model sees, so thin wicks stay legible.
    """
    PILImage, ImageDraw, ImageFont = _require_pil()
    if not bars:
        raise VisionError("no bars to chart")
    bars = bars[-max_bars:]

    bg = (13, 16, 22)
    grid = (32, 38, 50)
    up = (53, 208, 127)
    down = (255, 93, 108)
    txt = (198, 205, 218)
    dim = (112, 122, 140)
    ma1 = (110, 168, 254)
    ma2 = (185, 140, 255)

    im = PILImage.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(im)
    font = _font(15)          # axis price labels
    font_small = _font(14)    # header/footer
    font_title = _font(19)    # title
    pad_l, pad_r, pad_t = 12, 104, 46
    vol_h = int(height * 0.15)
    price_h = height - pad_t - vol_h - 34
    plot_w = width - pad_l - pad_r
    n = len(bars)
    step = plot_w / max(1, n)

    hi = max(b.high for b in bars)
    lo = min(b.low for b in bars)
    span = max(hi - lo, 1e-9)
    hi += span * 0.04
    lo -= span * 0.04
    span = hi - lo

    def y_of(price: float) -> float:
        return pad_t + (hi - price) / span * price_h

    def x_of(i: int) -> float:
        return pad_l + (i + 0.5) * step

    # grid + price labels
    for k in range(5):
        price = lo + span * k / 4
        y = y_of(price)
        d.line([(pad_l, y), (width - pad_r, y)], fill=grid, width=1)
        d.text((width - pad_r + 8, y - 8), f"{price:,.2f}", fill=dim, font=font)

    # moving averages
    closes = [b.close for b in bars]

    def sma_window(period: int) -> list[Optional[float]]:
        out: list[Optional[float]] = []
        for i in range(len(closes)):
            if i + 1 < period:
                out.append(None)
            else:
                w = closes[i + 1 - period: i + 1]
                out.append(sum(w) / period)
        return out

    for period, colour in ((ma_fast, ma1), (ma_slow, ma2)):
        series = sma_window(max(2, period))
        pts = [(x_of(i), y_of(v)) for i, v in enumerate(series) if v is not None]
        if len(pts) > 1:
            d.line(pts, fill=colour, width=2)

    # candles
    for i, b in enumerate(bars):
        cx = x_of(i)
        colour = up if b.close >= b.open else down
        d.line([(cx, y_of(b.high)), (cx, y_of(b.low))], fill=colour, width=1)
        w = max(1.0, step * 0.62)
        top, bot = y_of(max(b.open, b.close)), y_of(min(b.open, b.close))
        if bot - top < 1:
            bot = top + 1
        d.rectangle([cx - w / 2, top, cx + w / 2, bot], fill=colour)

    # volume
    vmax = max((b.volume for b in bars), default=1.0) or 1.0
    vol_top = height - vol_h - 18
    for i, b in enumerate(bars):
        cx = x_of(i)
        h = (b.volume / vmax) * vol_h
        colour = up if b.close >= b.open else down
        d.rectangle([cx - step * 0.3, vol_top + vol_h - h,
                     cx + step * 0.3, vol_top + vol_h], fill=colour)

    # header + axis labels
    last = bars[-1]
    d.text((pad_l, 10), f"{title}    last {last.close:,.2f}",
           fill=txt, font=font_title)
    d.text((pad_l, 34),
           f"{bars[0].ts:%Y-%m-%d %H:%M} → {last.ts:%Y-%m-%d %H:%M} UTC  ·  "
           f"{n} candles  ·  MA{ma_fast} blue  ·  MA{ma_slow} purple",
           fill=dim, font=font_small)

    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return Image(data=buf.getvalue(), mime="image/png", source="rendered-chart",
                 note=f"{n} candles, {title}", width=width, height=height)


# --------------------------------------------------------------------------
# Real browser screenshot (best effort)
# --------------------------------------------------------------------------

_BROWSERS = ("chromium", "chromium-browser", "google-chrome",
             "google-chrome-stable", "firefox")


def browser_available() -> Optional[str]:
    for b in _BROWSERS:
        from shutil import which
        if which(b):
            return b
    return None


def capture_page(url: str, width: int = 1280, height: int = 900,
                 timeout: int = 60) -> Image:
    """Screenshot a PUBLIC page. Best effort — raises VisionError if it fails.

    We never log in, never use an account, and never attempt to defeat bot
    detection. If a site does not want to be captured, we do not capture it.
    """
    browser = browser_available()
    if browser is None:
        raise VisionError("no headless browser available on this machine")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "shot.png"
        if browser == "firefox":
            cmd = [browser, "--headless", "--no-remote",
                   f"--window-size={width},{height}",
                   f"--screenshot={out}", url]
        else:
            cmd = [browser, "--headless=new", "--no-sandbox", "--disable-gpu",
                   f"--window-size={width},{height}", f"--screenshot={out}", url]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as e:
            raise VisionError(f"browser capture failed: {e}") from e
        if not out.exists():
            msg = (proc.stderr or b"").decode("utf-8", "ignore")[:300]
            raise VisionError(f"browser produced no screenshot: {msg}")
        data = out.read_bytes()
        if len(data) < 1000:
            raise VisionError("screenshot was suspiciously small")
        return Image(data=data, mime="image/png", source="browser-screenshot",
                     note=url, width=width, height=height)


# --------------------------------------------------------------------------
# Persistence: keep the exact image a claim was based on
# --------------------------------------------------------------------------

def save(image: Image, directory: str | Path, name: str = "") -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ext = "png" if image.mime == "image/png" else "jpg"
    fn = name or f"img_{datetime.now():%Y%m%dT%H%M%S}"
    p = d / f"{fn}.{ext}"
    p.write_bytes(image.data)
    return p


def load(path: str | Path) -> Image:
    p = Path(path)
    if not p.exists():
        raise VisionError(f"image not found: {p}")
    data = p.read_bytes()
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return Image(data=data, mime=mime, source="file", note=str(p))
