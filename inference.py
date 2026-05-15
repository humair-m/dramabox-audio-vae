import argparse
import os

import torch
import torchaudio
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from audio_vae import (
    AudioEncoder,
    AudioDecoder,
    encode_audio,
    decode_audio,
)
from audio_vae.vocoder import Vocoder, VocoderWithBWE, MelSTFT
from audio_vae.types import Audio

def build_dramabox_vocoder() -> VocoderWithBWE:
    """Builds the VocoderWithBWE matching the dramabox safetensors structure."""
    
    # 16kHz vocoder (6 upsample layers)
    vocoder_16k = Vocoder(
        resblock="AMP1",
        upsample_rates=[5, 2, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1536,
        output_sampling_rate=16000,
        activation="snakebeta",
        use_tanh_at_final=True,
        apply_final_activation=True,
        use_bias_at_final=True,
    )
    
    # 48kHz BWE generator (5 upsample layers)
    bwe_48k = Vocoder(
        resblock="AMP1",
        upsample_rates=[6, 5, 2, 2, 2],
        upsample_kernel_sizes=[12, 11, 4, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=512,
        output_sampling_rate=48000,
        activation="snakebeta",
        use_tanh_at_final=True,
        apply_final_activation=False, # BWE adds residual, so no final activation
        use_bias_at_final=True,
    )
    
    mel_stft = MelSTFT(
        filter_length=512,
        hop_length=80,
        win_length=512,
        n_mel_channels=64,
    )
    
    return VocoderWithBWE(
        vocoder=vocoder_16k,
        bwe_generator=bwe_48k,
        mel_stft=mel_stft,
        input_sampling_rate=16000,
        output_sampling_rate=48000,
        hop_length=80,
    )

def load_dramabox_weights(safetensors_path: str, encoder, decoder, vocoder):
    state_dict = load_file(safetensors_path)
    
    encoder_dict = {}
    decoder_dict = {}
    vocoder_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith("audio_vae.encoder."):
            encoder_dict[k.replace("audio_vae.encoder.", "")] = v
        elif k.startswith("audio_vae.decoder."):
            decoder_dict[k.replace("audio_vae.decoder.", "")] = v
        elif k.startswith("audio_vae.per_channel_statistics."):
            # Per channel stats belong to both encoder and decoder
            stat_k = k.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")
            encoder_dict[stat_k] = v
            decoder_dict[stat_k] = v
        elif k.startswith("vocoder."):
            vocoder_dict[k.replace("vocoder.", "")] = v
            
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)
    vocoder.load_state_dict(vocoder_dict)
    print("Weights loaded successfully!")


def main():
    parser = argparse.ArgumentParser(description="Test DramaBox Audio VAE")
    parser.add_argument("--input", type=str, help="Path to input audio file", default="test.wav")
    parser.add_argument("--output", type=str, help="Path to output reconstructed audio", default="reconstructed.wav")
    args = parser.parse_args()
    
    print("Initializing models...")
    encoder = AudioEncoder()
    decoder = AudioDecoder()
    vocoder = build_dramabox_vocoder()
    
    model_path = hf_hub_download(repo_id="zuhri025/dramabox-audio-vae-vocoder", filename="dramabox-audiovae-vocoder.safetensors")
    print(f"Loading weights from {model_path}...")
    load_dramabox_weights(model_path, encoder, decoder, vocoder)
    
    encoder.eval()
    decoder.eval()
    vocoder.eval()
    
    # Create dummy audio if input doesn't exist
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found, generating a dummy audio signal for testing...")
        sample_rate = 16000
        t = torch.linspace(0, 1, sample_rate * 2) # 2 seconds
        waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0).unsqueeze(0) # 1 channel
        waveform = waveform.expand(1, 2, -1) # Stereo
        audio = Audio(waveform=waveform, sampling_rate=sample_rate)
    else:
        print(f"Loading audio from {args.input}...")
        waveform, sample_rate = torchaudio.load(args.input)
        audio = Audio(waveform=waveform.unsqueeze(0), sampling_rate=sample_rate)
        
    print("Encoding audio to latents...")
    with torch.no_grad():
        latents = encode_audio(audio, encoder)
        print(f"Latents shape: {latents.shape}")
        
        print("Decoding latents to audio...")
        reconstructed_audio = decode_audio(latents, decoder, vocoder)
        
    print(f"Saving reconstructed audio to {args.output}...")
    torchaudio.save(args.output, reconstructed_audio.waveform, reconstructed_audio.sampling_rate)
    print("Done!")

if __name__ == "__main__":
    main()
