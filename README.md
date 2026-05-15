# Audio VAE & Vocoder (DramaBox)

This repository contains the standalone **Audio VAE and Vocoder** components, extracted from the `ltx_core` architecture and configured to directly run the `dramabox-audio-vae-vocoder` model. 

It acts as a fully independent module to encode audio into latent representations and decode them back to high-fidelity audio (with a Bandwidth Extension / BWE generator to upscale the output to 24kHz).

## Setup & Installation

You need Python 3.10+ and standard PyTorch dependencies.

```bash
# Install PyTorch (refer to pytorch.org for instructions based on your hardware)
pip install torch torchaudio

# Install additional dependencies
pip install einops safetensors huggingface_hub
```

## Usage

This module provides three main network architectures with their defaults matched against the `dramabox` safetensors structure:
- `AudioEncoder`: Encodes audio log-mel spectrograms to latents.
- `AudioDecoder`: Decodes latents to intermediate audio features.
- `VocoderWithBWE`: Converts intermediate audio features to high-fidelity 24kHz audio waveforms.

### Inference Script

We provide an `inference.py` script that demonstrates how to load the `dramabox-audio-vae-vocoder` safetensors directly from Hugging Face and run an encoding/decoding pass.

```bash
python inference.py --input test.wav --output reconstructed.wav
```
*(If the `--input` file is not found, the script will automatically generate and test on a dummy 440Hz sine wave.)*

### Direct Module Usage

You can also import the modules in your own code:

```python
import torch
import torchaudio
from audio_vae import AudioEncoder, AudioDecoder, encode_audio, decode_audio
from audio_vae.types import Audio
from inference import build_dramabox_vocoder, load_dramabox_weights
from huggingface_hub import hf_hub_download

# 1. Initialize components
encoder = AudioEncoder()
decoder = AudioDecoder()
vocoder = build_dramabox_vocoder()

# 2. Download and load weights
model_path = hf_hub_download(repo_id="zuhri025/dramabox-audio-vae-vocoder", filename="dramabox-audiovae-vocoder.safetensors")
load_dramabox_weights(model_path, encoder, decoder, vocoder)

# 3. Load Audio
waveform, sample_rate = torchaudio.load("my_audio.wav")
audio = Audio(waveform=waveform.unsqueeze(0), sampling_rate=sample_rate)

# 4. Encode & Decode
with torch.no_grad():
    latents = encode_audio(audio, encoder)
    reconstructed_audio = decode_audio(latents, decoder, vocoder)

# 5. Save Output
torchaudio.save("output.wav", reconstructed_audio.waveform, reconstructed_audio.sampling_rate)
```
