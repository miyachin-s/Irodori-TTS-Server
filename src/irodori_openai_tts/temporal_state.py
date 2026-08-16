from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from irodori_tts.codec import patchify_latent
from irodori_tts.model import patch_sequence_with_mask
from irodori_tts.speaker_inversion import save_speaker_inversion_safetensors


def cache_temporal_speaker_state_from_files(
    *,
    reference_paths: list[Path],
    output_path: Path,
    runtime: Any,
) -> dict[str, int | str]:
    if len(reference_paths) != 3:
        raise ValueError("exactly three reference waveforms are required")
    if output_path.exists():
        raise FileExistsError(output_path)

    latent_pieces = []
    total_frames = 0
    expected_dim = int(runtime.model_cfg.latent_dim)
    for reference_path in reference_paths:
        latent = runtime.codec.encode_file(reference_path)
        if (
            latent.ndim != 3
            or int(latent.shape[0]) != 1
            or int(latent.shape[1]) <= 0
            or int(latent.shape[2]) != expected_dim
        ):
            raise RuntimeError(
                "unexpected encoded reference latent shape: "
                f"{tuple(latent.shape)} for {reference_path}"
            )
        latent_pieces.append(latent.detach().to(device="cpu", dtype=torch.float32))
        total_frames += int(latent.shape[1])

    runtime_dtype = next(runtime.model.parameters()).dtype
    with torch.inference_mode():
        reference = torch.cat(latent_pieces, dim=1)
        patched = patchify_latent(
            reference,
            int(runtime.model_cfg.latent_patch_size),
        ).to(device=runtime.model_device, dtype=runtime_dtype)
        mask = torch.ones(
            (1, int(patched.shape[1])),
            dtype=torch.bool,
            device=runtime.model_device,
        )
        speaker_input, speaker_mask = patch_sequence_with_mask(
            seq=patched,
            mask=mask,
            patch_size=int(runtime.model_cfg.speaker_patch_size),
        )
        state = runtime.model.speaker_encoder(speaker_input, speaker_mask)
        state = runtime.model.speaker_norm(state)
        state, speaker_mask = runtime.model._prepend_masked_mean_token(state, speaker_mask)
        if state.ndim != 3 or int(state.shape[0]) != 1:
            raise RuntimeError(f"unexpected speaker state shape: {tuple(state.shape)}")
        if not bool(speaker_mask.all()) or int(speaker_mask.shape[1]) != int(state.shape[1]):
            raise RuntimeError("unexpected temporal speaker mask")
        stored = state[0].detach().to(device="cpu", dtype=torch.float32).contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.speaker.safetensors"
    )
    try:
        save_speaker_inversion_safetensors(
            temporary_path,
            {"speaker_embedding": stored},
            dtype=torch.float32,
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "reference_count": len(reference_paths),
        "reference_frames": total_frames,
        "speaker_tokens": int(stored.shape[0]),
        "speaker_dim": int(stored.shape[1]),
        "bytes": output_path.stat().st_size,
        "output_path": str(output_path),
    }
