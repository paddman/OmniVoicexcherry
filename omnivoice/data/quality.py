"""Audio dataset quality checks used by the Cherry training studio.

The scanner is intentionally dependency-light: NumPy and SoundFile are already
runtime dependencies of OmniVoice.  It reports suspicious files but only marks
missing or empty files as fatal, so compressed formats unsupported by the local
libsndfile build are not silently discarded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioQualityThresholds:
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 30.0
    min_sample_rate: int = 16_000
    clipping_amplitude: float = 0.999
    max_clipping_ratio: float = 0.001
    silence_amplitude: float = 10 ** (-50 / 20)
    max_silence_ratio: float = 0.45
    min_rms_dbfs: float = -42.0
    max_rms_dbfs: float = -3.0
    max_dc_offset: float = 0.02


@dataclass
class AudioQualityResult:
    path: str
    filename: str
    byte_size: int = 0
    sha256: str = ""
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    frames: int | None = None
    peak: float | None = None
    rms_dbfs: float | None = None
    clipping_ratio: float | None = None
    silence_ratio: float | None = None
    dc_offset: float | None = None
    issues: list[str] = field(default_factory=list)
    fatal_issues: list[str] = field(default_factory=list)
    duplicate_of: str | None = None

    @property
    def readable(self) -> bool:
        return self.duration_seconds is not None

    @property
    def usable(self) -> bool:
        return not self.fatal_issues

    @property
    def status(self) -> str:
        if self.fatal_issues:
            return "fatal"
        if self.issues:
            return "warning"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(readable=self.readable, usable=self.usable, status=self.status)
        return data


@dataclass
class DatasetQualitySummary:
    results: list[AudioQualityResult]

    @property
    def total_files(self) -> int:
        return len(self.results)

    @property
    def usable_files(self) -> int:
        return sum(result.usable for result in self.results)

    @property
    def warning_files(self) -> int:
        return sum(bool(result.issues) and not result.fatal_issues for result in self.results)

    @property
    def fatal_files(self) -> int:
        return sum(bool(result.fatal_issues) for result in self.results)

    @property
    def duplicate_files(self) -> int:
        return sum(result.duplicate_of is not None for result in self.results)

    @property
    def total_duration_seconds(self) -> float:
        return sum(result.duration_seconds or 0.0 for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_files": self.total_files,
                "usable_files": self.usable_files,
                "warning_files": self.warning_files,
                "fatal_files": self.fatal_files,
                "duplicate_files": self.duplicate_files,
                "total_duration_seconds": round(self.total_duration_seconds, 3),
            },
            "files": [result.to_dict() for result in self.results],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def to_markdown(self, max_rows: int = 30) -> str:
        hours, remainder = divmod(int(self.total_duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        lines = [
            "### Dataset Quality Report",
            "",
            f"- ไฟล์ทั้งหมด: **{self.total_files}**",
            f"- ใช้งานได้: **{self.usable_files}**",
            f"- มีคำเตือน: **{self.warning_files}**",
            f"- ใช้งานไม่ได้: **{self.fatal_files}**",
            f"- ไฟล์ซ้ำ: **{self.duplicate_files}**",
            f"- ระยะเวลารวมที่อ่านได้: **{duration}**",
        ]
        suspicious = [result for result in self.results if result.issues or result.fatal_issues]
        if not suspicious:
            lines.extend(["", "ไม่พบปัญหาตามเกณฑ์พื้นฐาน"])
            return "\n".join(lines)

        lines.extend(["", "| ไฟล์ | สถานะ | รายละเอียด |", "|---|---|---|"])
        for result in suspicious[:max_rows]:
            details = result.fatal_issues + result.issues
            lines.append(
                f"| `{result.filename}` | {result.status} | "
                + "; ".join(details).replace("|", "\\|")
                + " |"
            )
        if len(suspicious) > max_rows:
            lines.append(f"\nแสดง {max_rows} จาก {len(suspicious)} ไฟล์ที่ควรตรวจสอบ")
        return "\n".join(lines)


def _hash_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _append_threshold_issues(result: AudioQualityResult, thresholds: AudioQualityThresholds) -> None:
    duration = result.duration_seconds
    if duration is not None:
        if duration < thresholds.min_duration_seconds:
            result.issues.append(
                f"เสียงสั้นกว่า {thresholds.min_duration_seconds:g} วินาที ({duration:.2f}s)"
            )
        if duration > thresholds.max_duration_seconds:
            result.issues.append(
                f"เสียงยาวกว่า {thresholds.max_duration_seconds:g} วินาที ({duration:.2f}s)"
            )
    if result.sample_rate is not None and result.sample_rate < thresholds.min_sample_rate:
        result.issues.append(
            f"sample rate ต่ำกว่า {thresholds.min_sample_rate} Hz ({result.sample_rate} Hz)"
        )
    if result.channels is not None and result.channels > 1:
        result.issues.append(f"มี {result.channels} channels ควรตรวจว่าเป็นเสียง mono ที่สม่ำเสมอ")
    if result.clipping_ratio is not None and result.clipping_ratio > thresholds.max_clipping_ratio:
        result.issues.append(f"มี clipping {result.clipping_ratio * 100:.3f}%")
    if result.silence_ratio is not None and result.silence_ratio > thresholds.max_silence_ratio:
        result.issues.append(f"ช่วงเงียบสูง {result.silence_ratio * 100:.1f}%")
    if result.rms_dbfs is not None:
        if result.rms_dbfs < thresholds.min_rms_dbfs:
            result.issues.append(f"เสียงเบามาก RMS {result.rms_dbfs:.1f} dBFS")
        elif result.rms_dbfs > thresholds.max_rms_dbfs:
            result.issues.append(f"เสียงดังมาก RMS {result.rms_dbfs:.1f} dBFS")
    if result.dc_offset is not None and result.dc_offset > thresholds.max_dc_offset:
        result.issues.append(f"DC offset สูง {result.dc_offset:.4f}")


def scan_audio_file(
    path: str | Path,
    thresholds: AudioQualityThresholds | None = None,
    block_frames: int = 65_536,
) -> AudioQualityResult:
    """Inspect one audio file without loading the entire recording into RAM."""
    thresholds = thresholds or AudioQualityThresholds()
    source = Path(path).expanduser()
    result = AudioQualityResult(path=str(source), filename=source.name)

    if not source.exists() or not source.is_file():
        result.fatal_issues.append("ไม่พบไฟล์")
        return result
    result.byte_size = source.stat().st_size
    if result.byte_size == 0:
        result.fatal_issues.append("ไฟล์ว่าง")
        return result
    try:
        result.sha256 = _hash_file(source)
    except OSError as exc:
        result.fatal_issues.append(f"อ่านไฟล์ไม่ได้: {exc}")
        return result

    try:
        with sf.SoundFile(source) as audio:
            result.sample_rate = int(audio.samplerate)
            result.channels = int(audio.channels)
            result.frames = int(len(audio))
            if result.frames <= 0 or result.sample_rate <= 0:
                result.fatal_issues.append("ไม่มี audio frame")
                return result
            result.duration_seconds = result.frames / result.sample_rate

            sample_count = 0
            sum_squares = 0.0
            clipping_count = 0
            silence_count = 0
            peak = 0.0
            channel_sums = np.zeros(result.channels, dtype=np.float64)

            for block in audio.blocks(
                blocksize=block_frames,
                dtype="float32",
                always_2d=True,
            ):
                if not np.isfinite(block).all():
                    result.issues.append("พบ NaN หรือ Infinity ในตัวอย่างเสียง")
                    block = np.nan_to_num(block, copy=False)
                values = block.astype(np.float64, copy=False)
                absolute = np.abs(values)
                sample_count += values.size
                sum_squares += float(np.square(values).sum())
                clipping_count += int((absolute >= thresholds.clipping_amplitude).sum())
                silence_count += int((absolute <= thresholds.silence_amplitude).sum())
                peak = max(peak, float(absolute.max(initial=0.0)))
                channel_sums += values.sum(axis=0)

            if sample_count == 0:
                result.fatal_issues.append("ไม่มีตัวอย่างเสียงที่อ่านได้")
                return result
            rms = math.sqrt(sum_squares / sample_count)
            result.peak = peak
            result.rms_dbfs = 20 * math.log10(max(rms, 1e-12))
            result.clipping_ratio = clipping_count / sample_count
            result.silence_ratio = silence_count / sample_count
            frames_read = sample_count / max(result.channels, 1)
            result.dc_offset = float(np.max(np.abs(channel_sums / max(frames_read, 1))))
    except (OSError, RuntimeError, ValueError) as exc:
        # libsndfile support varies by platform. Keep the file but make the gap visible.
        result.issues.append(f"ตัวตรวจคุณภาพถอดรหัสไฟล์ไม่ได้: {exc}")
        return result

    _append_threshold_issues(result, thresholds)
    return result


def scan_dataset(
    paths: Iterable[str | Path],
    thresholds: AudioQualityThresholds | None = None,
) -> DatasetQualitySummary:
    results = [scan_audio_file(path, thresholds=thresholds) for path in paths]
    first_by_hash: dict[str, AudioQualityResult] = {}
    for result in results:
        if not result.sha256:
            continue
        first = first_by_hash.get(result.sha256)
        if first is None:
            first_by_hash[result.sha256] = result
            continue
        result.duplicate_of = first.filename
        result.issues.append(f"เนื้อหาไฟล์ซ้ำกับ {first.filename}")
    return DatasetQualitySummary(results)


__all__ = [
    "AudioQualityResult",
    "AudioQualityThresholds",
    "DatasetQualitySummary",
    "scan_audio_file",
    "scan_dataset",
]
