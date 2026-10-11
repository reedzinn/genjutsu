"""Real VACE inference, gated on CUDA. No paid API or simulated generation."""
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

from media import FPS, box_masks, normalize, read_clip, save_video

MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
NEGATIVE = "blurry, low quality, distorted anatomy, extra limbs, text, watermark, flickering"


def require_gpu():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("O motor não está instalado. Use o notebook em uma sessão com GPU NVIDIA.") from exc
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        raise RuntimeError("Geração indisponível: este protótipo requer GPU NVIDIA/CUDA. "
                           "A RX 580/DirectML não foi validada. Nenhuma geração foi simulada.")
    if torch.cuda.get_device_capability()[0] < 7:
        raise RuntimeError("A GPU é antiga para o caminho de quantização deste protótipo.")
    return torch


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_text(torch, prompt, dtype):
    """Encode first, release the text model, then load the video generator.

    4-bit text encoding reduces memory; this is an experimental configuration,
    not a claim about a minimum GPU size or equivalence to the full model.
    """
    from diffusers import WanVACEPipeline, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL, subfolder="text_encoder", quantization_config=quant,
        device_map={"": 0}, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    pipe = WanVACEPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=None,
                           transformer=None, scheduler=UniPCMultistepScheduler(flow_shift=3.0))
    try:
        with torch.inference_mode():
            positive, negative = pipe.encode_prompt(
                prompt=prompt, negative_prompt=NEGATIVE, device=torch.device("cuda"),
                do_classifier_free_guidance=True, max_sequence_length=256)
        return positive.cpu(), negative.cpu()
    finally:
        del pipe, text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def prepare_pose(frames, report):
    from controlnet_aux import OpenposeDetector
    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    try:
        result = []
        for i, frame in enumerate(frames):
            report(f"Extraindo pose: {i + 1}/{len(frames)}")
            pose = detector(frame, include_body=True, include_hand=False, include_face=False)
            if pose.convert("L").getbbox() is None:
                raise ValueError(f"Nenhuma pose detectada no quadro {i + 1}. Use um vídeo com uma pessoa visível.")
            result.append(pose.resize(frame.size, Image.Resampling.BILINEAR).convert("RGB"))
        return result
    finally:
        del detector
        gc.collect()


def generate(video, reference, mode, prompt, size=(384, 224), max_frames=17,
             steps=25, seed=42, start_box="0.25,0.2,0.75,0.85", end_box="",
             output_root="outputs", report=lambda message: None):
    if mode not in ("motion", "swap"):
        raise ValueError("Modo desconhecido.")
    if not prompt or not prompt.strip():
        raise ValueError("Descreva a aparência e a ação desejadas.")
    if not 10 <= int(steps) <= 50 or not 0 <= int(seed) < 2**32:
        raise ValueError("Passos ou semente fora do intervalo permitido.")
    # Validate inputs before downloading any model or touching a GPU.
    frames = read_clip(video, size=size, max_frames=int(max_frames))
    with Image.open(reference) as im:
        ref = normalize(im, size, color="white")
    masks = box_masks(size, len(frames), start_box, end_box) if mode == "swap" else [
        Image.new("L", size, 0) for _ in frames]
    torch = require_gpu()
    run = Path(output_root).resolve() / uuid.uuid4().hex
    run.mkdir(parents=True)
    metadata = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "model": MODEL, "mode": mode, "prompt": prompt, "negative_prompt": NEGATIVE,
                "seed": int(seed), "steps": int(steps), "fps": FPS,
                "size": list(size), "frames": len(frames), "start_box": start_box if mode == "swap" else None,
                "end_box": end_box if mode == "swap" else None,
                "input_sha256": _hash(video), "reference_sha256": _hash(reference),
                "gpu": torch.cuda.get_device_name(), "text_quantization": "nf4",
                "audio": "omitted", "paid_inference_api_calls": 0}
    for package in ("torch", "diffusers", "transformers", "bitsandbytes", "controlnet-aux"):
        try:
            metadata[package] = version(package)
        except PackageNotFoundError:
            pass
    manifest = run / "run.json"
    started = time.monotonic()
    pipe = None
    try:
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        report("Preparando sinais de controle")
        controls = prepare_pose(frames, report) if mode == "motion" else frames
        save_video(controls, run / "controle.mp4")
        save_video([m.convert("RGB") for m in masks], run / "mascara.mp4")
        ref.save(run / "referencia.png")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        report("Carregando o codificador de texto; o primeiro uso baixa os modelos")
        positive, negative = encode_text(torch, prompt, dtype)
        from diffusers import AutoencoderKLWan, WanVACEPipeline, UniPCMultistepScheduler
        report("Carregando o gerador VACE")
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVACEPipeline.from_pretrained(MODEL, vae=vae, text_encoder=None,
                                               tokenizer=None, torch_dtype=dtype)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()

        def on_step(pipeline, step, timestep, kwargs):
            report(f"Gerando: etapa {step + 1}/{steps}")
            return kwargs

        with torch.inference_mode():
            output = pipe(video=controls, mask=masks, reference_images=[ref],
                          prompt_embeds=positive, negative_prompt_embeds=negative,
                          height=size[1], width=size[0], num_frames=len(frames),
                          num_inference_steps=int(steps), guidance_scale=5.0,
                          generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                          output_type="np", callback_on_step_end=on_step).frames[0]
        if len(output) != len(frames):
            raise RuntimeError("O modelo retornou uma quantidade inesperada de quadros.")
        if not np.isfinite(output).all():
            raise RuntimeError("A geração apresentou erro numérico. Nenhum vídeo inválido será entregue.")
        output = [Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype(np.uint8)) for frame in output]
        report("Salvando o vídeo gerado")
        result = save_video(output, run / "resultado.mp4")
        metadata["status"] = "completed"
        return result, str(manifest)
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 2)
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()

