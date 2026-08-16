from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from irodori_openai_tts.temporal_state import cache_temporal_speaker_state_from_files


class FakeCodec:
    def __init__(self, latents):
        self.latents = latents
        self.paths = []

    def encode_file(self, path):
        self.paths.append(path)
        return self.latents[path.name]


class FakeModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.weight

    def speaker_encoder(self, state, mask):
        assert bool(mask.all())
        return state.repeat_interleave(2, dim=-1)

    def speaker_norm(self, state):
        return state + 1.0

    @staticmethod
    def _prepend_masked_mean_token(state, mask):
        mean = state.mean(dim=1, keepdim=True)
        return (
            torch.cat((mean, state), dim=1),
            torch.cat((torch.ones((1, 1), dtype=torch.bool), mask), dim=1),
        )


def test_caches_three_ordered_references_as_temporal_state(tmp_path):
    reference_paths = [tmp_path / f"reference-{index}.wav" for index in range(1, 4)]
    for path in reference_paths:
        path.write_bytes(b"wav")
    latents = {
        path.name: torch.full((1, index + 1, 4), float(index))
        for index, path in enumerate(reference_paths, start=1)
    }
    codec = FakeCodec(latents)
    runtime = SimpleNamespace(
        codec=codec,
        model=FakeModel(),
        model_device=torch.device("cpu"),
        model_cfg=SimpleNamespace(
            latent_dim=4,
            latent_patch_size=1,
            speaker_patch_size=1,
        ),
    )
    output_path = tmp_path / "designed.speaker.safetensors"

    result = cache_temporal_speaker_state_from_files(
        reference_paths=reference_paths,
        output_path=output_path,
        runtime=runtime,
    )

    stored = load_file(output_path)["speaker_embedding"]
    assert codec.paths == reference_paths
    assert result["reference_frames"] == 9
    assert result["speaker_tokens"] == 10
    assert result["speaker_dim"] == 8
    assert stored.shape == (10, 8)
    assert torch.equal(stored[1, :4], torch.full((4,), 2.0))
    assert torch.equal(stored[-1, :4], torch.full((4,), 4.0))
