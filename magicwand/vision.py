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


def find_levels(bars: list[Bar], k: int = 3, max_levels: int = 3,
                tolerance: float = 0.004) -> dict:
    """Observed swing highs and lows — the levels a trader would mark.

    These are FACTS about the window that was drawn: a pivot high is a bar whose
    high exceeds the k bars either side. They are not projections: a level is only
    drawn where price actually turned, and it is never extended into the future.

    Returns {"resistance": [prices], "support": [prices]}.
    """
    highs: list[tuple[float, int]] = []
    lows: list[tuple[float, int]] = []
    n = len(bars)
    for i in range(k, n - k):
        window = bars[i - k: i + k + 1]
        if bars[i].high >= max(b.high for b in window):
            highs.append((bars[i].high, i))
        if bars[i].low <= min(b.low for b in window):
            lows.append((bars[i].low, i))

    def thin(points: list[tuple[float, int]]) -> list[tuple[float, int]]:
        out: list[tuple[float, int]] = []
        for price, idx in sorted(points, reverse=True):
            if all(abs(price - p) / max(p, 1e-9) > tolerance for p, _ in out):
                out.append((price, idx))
        return out[:max_levels]

    return {"resistance": thin(highs), "support": sorted(thin(lows))}


def render_chart(bars: list[Bar], title: str = "SOL-USD 1h",
                 width: int = 1280, height: int = 720,
                 max_bars: int = 120, ma_fast: int = 12,
                 ma_slow: int = 48, levels: bool = True,
                 position: dict | None = None) -> Image:
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

    # ---- observed levels: drawn only where price actually turned ----
    if levels:
        lv = find_levels(bars)
        used_label_y: list[float] = []
        for side, colour, mark in (("resistance", (255, 93, 108), "R"),
                                   ("support", (53, 208, 127), "S")):
            for price, idx in lv[side]:
                y = y_of(price)
                if not (pad_t < y < pad_t + price_h):
                    continue
                x0 = max(pad_l, x_of(idx) - 2 * step)
                x1 = min(width - pad_r, x_of(idx) + 8 * step)
                d.line([(x0, y), (x1, y)], fill=colour, width=1)
                # label to the right of the dash, nudged if it would overlap
                ly = y - 7
                while any(abs(ly - u) < 13 for u in used_label_y):
                    ly += 13
                if pad_t < ly < pad_t + price_h - 10:
                    d.text((min(x1 + 6, width - pad_r - 64), ly),
                           f"{mark} {price:,.2f}", fill=colour, font=font_small)
                    used_label_y.append(ly)

    # last traded price: a dotted reference, not a forecast
    y_last = y_of(bars[-1].close)
    if pad_t < y_last < pad_t + price_h:
        x = pad_l
        while x < width - pad_r:
            d.line([(x, y_last), (min(x + 6, width - pad_r), y_last)],
                   fill=(198, 205, 218), width=1)
            x += 12

    # an open position: entry, stop, target — the only lines that matter to risk
    if position:
        for key, colour, label in (
                ("entry_price", (110, 168, 254), "entry"),
                ("stop", (255, 93, 108), "stop"),
                ("target", (53, 208, 127), "target")):
            val = position.get(key)
            if not val:
                continue
            y = y_of(float(val))
            if pad_t < y < pad_t + price_h:
                d.line([(pad_l, y), (width - pad_r, y)], fill=colour, width=1)
                d.text((pad_l + 6, y - 16), f"{label} {float(val):,.2f}",
                       fill=colour, font=font_small)

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
           f"{n} candles  ·  MA{ma_fast} blue  ·  MA{ma_slow} purple  ·  "
           f"R/S observed levels  ·  NO projection",
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
