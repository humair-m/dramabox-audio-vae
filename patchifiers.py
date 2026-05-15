from typing import Optional, Tuple, Protocol
import einops
import torch

from .types import AudioLatentShape, VideoLatentShape

class Patchifier(Protocol):
    """
    Protocol for patchifiers that convert latent tensors into patches and assemble them back.
    """

    def patchify(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        ...
        """
        Convert latent tensors into flattened patch tokens.
        Args:
            latents: Latent tensor to patchify.
        Returns:
            Flattened patch tokens tensor.
        """

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: AudioLatentShape | VideoLatentShape,
    ) -> torch.Tensor:
        """
        Converts latent tensors between spatio-temporal formats and flattened sequence representations.
        Args:
            latents: Patch tokens that must be rearranged back into the latent grid constructed by `patchify`.
            output_shape: Shape of the output tensor. Note that output_shape is either AudioLatentShape or
            VideoLatentShape.
        Returns:
            Dense latent tensor restored from the flattened representation.
        """

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        ...
        """
        Returns the patch size as a tuple of (temporal, height, width) dimensions
        """

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ...
        """
        Compute metadata describing where each latent patch resides within the
        grid specified by `output_shape`.
        Args:
            output_shape: Target grid layout for the patches.
            device: Target device for the returned tensor.
        Returns:
            Tensor containing patch coordinate metadata such as spatial or temporal intervals.
        """


class AudioPatchifier(Patchifier):
    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        """
        Patchifier tailored for spectrogram/audio latents.
        Args:
            patch_size: Number of mel bins combined into a single patch. This
                controls the resolution along the frequency axis.
            sample_rate: Original waveform sampling rate. Used to map latent
                indices back to seconds so downstream consumers can align audio
                and video cues.
            hop_length: Window hop length used for the spectrogram. Determines
                how many real-time samples separate two consecutive latent frames.
            audio_latent_downsample_factor: Ratio between spectrogram frames and
                latent frames; compensates for additional downsampling inside the
                VAE encoder.
            is_causal: When True, timing is shifted to account for causal
                receptive fields so timestamps do not peek into the future.
            shift: Integer offset applied to the latent indices. Enables
                constructing overlapping windows from the same latent sequence.
        """
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        return tgt_shape.frames

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")

        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)

        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor

        if self.is_causal:
            # Frame offset for causal alignment.
            # The "+1" ensures the timestamp corresponds to the first sample that is fully available.
            causal_offset = 1
            audio_mel_frame = (audio_mel_frame + causal_offset - self.audio_latent_downsample_factor).clip(min=0)

        return audio_mel_frame * self.hop_length / self.sample_rate

    def _compute_audio_timings(
        self,
        batch_size: int,
        num_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        resolved_device = device
        if resolved_device is None:
            resolved_device = torch.device("cpu")

        start_timings = self._get_audio_latent_time_in_sec(
            self.shift,
            num_steps + self.shift,
            torch.float32,
            resolved_device,
        )
        start_timings = start_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1,
            num_steps + self.shift + 1,
            torch.float32,
            resolved_device,
        )
        end_timings = end_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        return torch.stack([start_timings, end_timings], dim=-1)

    def patchify(
        self,
        audio_latents: torch.Tensor,
    ) -> torch.Tensor:
        audio_latents = einops.rearrange(
            audio_latents,
            "b c t f -> b t (c f)",
        )

        return audio_latents

    def unpatchify(
        self,
        audio_latents: torch.Tensor,
        output_shape: AudioLatentShape,
    ) -> torch.Tensor:
        audio_latents = einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )

        return audio_latents

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not isinstance(output_shape, AudioLatentShape):
            raise ValueError("AudioPatchifier expects AudioLatentShape when computing coordinates")

        return self._compute_audio_timings(output_shape.batch, output_shape.frames, device)
