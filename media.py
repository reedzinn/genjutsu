"""Video preparation only. This module never pretends to generate AI output."""
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

FPS = 16


def normalize(image: Image.Image, size: tuple[int, int], color="black") -> Image.Image:
    return ImageOps.pad(ImageOps.exif_transpose(image).convert("RGB"), size,
                        method=Image.Resampling.LANCZOS, color=color)


def read_clip(path, size=(384, 224), max_frames=17):
    if max_frames not in (17, 33, 49, 81):
        raise ValueError("Escolha 17, 33, 49 ou 81 quadros.")
    if any(v < 64 or v % 16 for v in size):
        raise ValueError("As dimensões precisam ser múltiplas de 16, com pelo menos 64 pixels.")
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 100 * 1024**2:
        raise ValueError("Envie um vídeo de até 100 MB.")
    cap = cv2.VideoCapture(str(path))
    try:
        rate = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not math.isfinite(rate) or rate <= 0 or count < 2:
            raise ValueError("Não foi possível ler este vídeo. Experimente MP4/H.264.")
        # Last sample must be within the source. VACE requires 4*n + 1 frames.
        available = math.floor((count - 1) / rate * FPS + 1e-6) + 1
        n = min(max_frames, available)
        n = ((n - 1) // 4) * 4 + 1
        if n < 5:
            raise ValueError("O vídeo precisa ter pelo menos cerca de meio segundo.")
        frames = []
        for i in range(n):
            source_index = min(int(round(i * rate / FPS)), int(count) - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Não foi possível ler o quadro {source_index}.")
            frames.append(normalize(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), size))
        return frames
    finally:
        cap.release()


def parse_box(value):
    try:
        box = tuple(float(x.strip()) for x in value.split(","))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Região: use quatro números, por exemplo 0.25,0.2,0.75,0.85.") from exc
    if (len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError("Região inválida. Use esquerda,topo,direita,base entre 0 e 1.")
    return box


def box_masks(size, count, start, end=None):
    start = parse_box(start)
    end = parse_box(end) if end else start
    width, height = size
    masks = []
    for i in range(count):
        fraction = i / max(count - 1, 1)
        box = [a + fraction * (b - a) for a, b in zip(start, end)]
        x0, y0 = int(box[0] * width), int(box[1] * height)
        x1, y1 = math.ceil(box[2] * width), math.ceil(box[3] * height)
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
        masks.append(mask)
    return masks


def mask_preview(frame, mask):
    overlay = Image.new("RGB", frame.size, "#ff5252")
    return Image.composite(Image.blend(frame, overlay, .45), frame, mask)


def save_video(frames, path, fps=FPS):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1,
                            ffmpeg_log_level="error") as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) != len(frames):
            raise RuntimeError("O vídeo salvo não contém a sequência esperada.")
    finally:
        cap.release()
    return str(path)

