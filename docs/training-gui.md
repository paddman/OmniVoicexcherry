# OmniVoice Cherry Training GUI

The Cherry training studio wraps OmniVoice dataset preparation, audio quality checks, Thai text normalization, preflight diagnostics, tokenization, and fine-tuning in a local Gradio interface.

## Start

```bash
omnivoice-train-gui --ip 127.0.0.1 --port 8002
```

From a source checkout, the equivalent command is:

```bash
python -m omnivoice.cli.train_gui --ip 127.0.0.1 --port 8002
```

Do not expose the interface to an untrusted network without adding authentication and network controls. Binding to `0.0.0.0` makes the service reachable from other machines allowed by the host firewall.

## 1. Prepare and inspect a dataset

1. Create a project name.
2. Upload one or more audio files.
3. Add transcripts directly:

```text
001.wav|ข้อความที่พูดในไฟล์|th|session-01
002.wav|Another sentence|en|session-02
```

For one uploaded file, a plain transcript can be pasted without the filename. A CSV, TSV, or JSONL manifest can also be uploaded. Supported fields are:

- `filename`, `audio`, or `audio_path`
- `text` or `transcript`
- optional `language_id`
- optional `speaker_id`
- optional `session_id`, `source_id`, or `recording_session`

The studio keeps recording sessions together during the train/dev split when at least two session groups are available. This reduces evaluation leakage from nearly identical clips recorded in the same session.

### Audio quality report

Each uploaded file is scanned for:

- duration, sample rate, channel count, peak, RMS, and DC offset
- clipping and high silence ratio
- empty or missing files
- exact duplicate file content
- decoder support in the local SoundFile/libsndfile installation

The report is stored at:

```text
data/gui_projects/<project>/manifests/quality_report.json
```

Quality warnings do not automatically prove that a clip is unusable. Listen to flagged clips before deleting them. Decoder support varies by operating system, so MP3/M4A files may require FFmpeg even when training can otherwise handle them.

### Thai text normalization

When enabled for `language_id` values beginning with `th`, the studio normalizes common TTS patterns including:

- Thai digits and Arabic digits
- dates and times
- baht/satang amounts and percentages
- telephone numbers and IPv4 addresses
- selected Thai abbreviations

Both `raw_text` and `normalized_text` are retained in the JSONL manifest. The `text` field used for training contains the normalized value. Keeping the raw transcript makes normalization changes auditable instead of quietly rewriting the source of truth, a habit software occasionally adopts when it wants future debugging to become folklore.

Preparing a new dataset replaces only the project's `audio`, `manifests`, and token cache. Existing versioned training runs remain untouched.

## 2. Run preflight diagnostics

Use the **ตรวจระบบก่อนเทรน** button or run:

```bash
omnivoice-preflight \
  --gpu-ids 0 \
  --workspace data/gui_projects/cherry-voice \
  --checkpoint k2-fsa/OmniVoice \
  --tokenizer eustlb/higgs-audio-v2-tokenizer
```

Preflight checks Python, writable workspace, free disk, FFmpeg, Accelerate, PyTorch/CUDA visibility, GPU IDs, and local model paths. Remote model identifiers are reported as warnings because network access, authentication, and licensing cannot be proven by a local diagnostic.

## 3. Fine-tune

LoRA is enabled by default. The GUI exposes:

- GPU IDs or CPU smoke-test mode
- SDPA or flex attention
- precision, steps, learning rate, batch tokens, and gradient accumulation
- LoRA rank, alpha, and dropout
- save/evaluation/log intervals
- checkpoint retention, defaulting to the latest three checkpoints
- token cache reuse based on a dataset fingerprint
- process logs and stop controls

Each training attempt gets an independent run directory:

```text
data/gui_projects/<project>/
├── audio/
├── manifests/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── dataset.json
│   └── quality_report.json
├── tokens/
│   └── dataset.sha256
└── runs/
    └── run-YYYYMMDD-HHMMSS/
        ├── config/
        │   ├── train_config.json
        │   └── data_config.json
        ├── checkpoints/
        ├── preflight.json
        ├── run.json
        └── training.log
```

`run.json` records the dataset fingerprint, base checkpoint, tokenizer, LoRA mode, config hash, source revision when available, command plan, and current status. A repeated run name receives a numeric suffix instead of overwriting previous results.

## Token cache behavior

Audio tokens are shared at project level. The cache is reused only when the current train/dev manifest fingerprint matches `tokens/dataset.sha256`. Changing transcripts, split membership, or files invalidates the token cache automatically. The **Tokenize ใหม่** option forces regeneration.

## License and consent

The repository source license, pretrained model weights, audio tokenizer, datasets, and uploaded voices may all have different terms. Before commercial deployment:

1. confirm permission for every recorded voice and transcript;
2. review the current license and usage terms of the exact checkpoint revision;
3. review the audio tokenizer and all training datasets separately;
4. retain consent, provenance, and withdrawal records outside the local checkbox;
5. add authentication, tenant isolation, quotas, and audit logging before exposing the studio as a service.

The GUI consent checkbox records an operator acknowledgement for the current action. It is not a substitute for a signed consent process or legal review.
