# OmniVoice Cherry Training GUI

The training GUI wraps the existing OmniVoice fine-tuning pipeline with a local Gradio interface.

## Start

```bash
python -m omnivoice.cli.train_gui --ip 0.0.0.0 --port 8002
```

Open `http://localhost:8002`.

## Dataset workflow

1. Create a project name.
2. Upload one or more audio files.
3. Add transcripts directly using:

```text
001.wav|ข้อความที่พูดในไฟล์|th
002.wav|Another sentence|en
```

For one uploaded file, a plain transcript can be pasted without the filename.

Alternatively upload a CSV, TSV, or JSONL manifest. Supported columns/keys are:

- `filename`, `audio`, or `audio_path`
- `text` or `transcript`
- optional `language_id`

4. Choose the development split percentage and click **Prepare Dataset**.

The GUI copies audio into the project workspace and writes OmniVoice-compatible `train.jsonl` and `dev.jsonl` manifests.

## Fine-tuning

The Fine-tune tab exposes:

- GPU IDs, including multi-GPU values such as `0,1`
- CPU mode using `cpu`
- SDPA or flex attention configs
- base checkpoint and audio tokenizer
- steps, learning rate, batch tokens, gradient accumulation, precision, save/eval/log intervals
- token cache reuse or forced re-tokenization
- live logs and process stop controls

Before training starts, the user must confirm that they own or have permission to use every uploaded voice.

Generated data is stored under:

```text
data/gui_projects/<project>/
├── audio/
├── manifests/
├── tokens/
├── config/
├── checkpoints/
└── training.log
```

## Notes

- Training still requires the same CUDA/PyTorch and model dependencies as the normal OmniVoice fine-tuning scripts.
- SDPA is the safer default for GPUs that do not support `flex_attention` well.
- Uploaded audio should be clean, correctly transcribed, and consistently recorded for better results.
- Do not use the tool for unauthorized impersonation, fraud, scams, or deceptive content.
