from pathlib import Path

import numpy as np
import soundfile as sf

from omnivoice.data.quality import scan_audio_file, scan_dataset


def _write_tone(path: Path, amplitude: float = 0.1, seconds: float = 2.0) -> None:
    sample_rate = 24_000
    t = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    audio = amplitude * np.sin(2 * np.pi * 220 * t)
    sf.write(path, audio, sample_rate)


def test_scan_clean_audio(tmp_path):
    path = tmp_path / "clean.wav"
    _write_tone(path)
    result = scan_audio_file(path)
    assert result.usable
    assert result.status == "ok"
    assert result.sample_rate == 24_000
    assert result.channels == 1
    assert result.duration_seconds == 2.0
    assert result.rms_dbfs is not None


def test_scan_detects_clipping(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.ones(24_000, dtype=np.float32), 24_000, subtype="FLOAT")
    result = scan_audio_file(path)
    assert result.usable
    assert result.status == "warning"
    assert any("clipping" in issue for issue in result.issues)


def test_scan_dataset_detects_duplicates(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_tone(first)
    second.write_bytes(first.read_bytes())
    summary = scan_dataset([first, second])
    assert summary.duplicate_files == 1
    assert summary.results[1].duplicate_of == "first.wav"
