"""Gradio studio for preparing, checking, and fine-tuning OmniVoice voices."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import uuid4

import gradio as gr

from omnivoice.data.quality import DatasetQualitySummary, scan_dataset
from omnivoice.utils.preflight import parse_gpu_ids, run_preflight
from omnivoice.utils.thai_text import normalize_thai_text

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-._")
    if not value:
        raise gr.Error("กรุณาตั้งชื่อโปรเจกต์")
    return value[:80]


def optional_safe_name(value: str, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-._")
    return (cleaned or default)[:100]


def upload_path(value: Any) -> Path:
    if isinstance(value, str):
        return Path(value)
    for attr in ("path", "name"):
        found = getattr(value, attr, None)
        if found:
            return Path(found)
    raise gr.Error("อ่านไฟล์อัปโหลดไม่สำเร็จ")


def parse_manifest(value: Any | None) -> dict[str, dict[str, str]]:
    if not value:
        return {}
    path = upload_path(value)
    rows: dict[str, dict[str, str]] = {}
    try:
        if path.suffix.lower() == ".jsonl":
            source = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        elif path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open(encoding="utf-8-sig", newline="") as handle:
                source = list(csv.DictReader(handle, delimiter=delimiter))
        else:
            raise gr.Error("Manifest ต้องเป็น CSV, TSV หรือ JSONL")
        for row in source:
            filename = row.get("filename") or row.get("audio") or row.get("audio_path")
            text = row.get("text") or row.get("transcript")
            if filename and text:
                rows[Path(str(filename)).name] = {
                    "text": str(text).strip(),
                    "language_id": str(row.get("language_id") or "").strip(),
                    "speaker_id": str(row.get("speaker_id") or "").strip(),
                    "session_id": str(
                        row.get("session_id")
                        or row.get("source_id")
                        or row.get("recording_session")
                        or ""
                    ).strip(),
                }
    except (OSError, csv.Error, json.JSONDecodeError, AttributeError) as exc:
        raise gr.Error(f"อ่าน manifest ไม่สำเร็จ: {exc}") from exc
    return rows


def parse_transcripts(raw: str, count: int) -> dict[str, dict[str, str]]:
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if not lines:
        return {}
    if count == 1 and len(lines) == 1 and "|" not in lines[0] and "\t" not in lines[0]:
        return {
            "__single__": {
                "text": lines[0],
                "language_id": "",
                "speaker_id": "",
                "session_id": "",
            }
        }
    result: dict[str, dict[str, str]] = {}
    for number, line in enumerate(lines, 1):
        delimiter = "|" if "|" in line else "\t" if "\t" in line else None
        if delimiter is None:
            raise gr.Error(
                f"Transcript บรรทัด {number} ต้องเป็น filename|ข้อความ|language_id|session_id"
            )
        parts = [part.strip() for part in line.split(delimiter, 3)]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise gr.Error(f"Transcript บรรทัด {number} ไม่สมบูรณ์")
        result[Path(parts[0]).name] = {
            "text": parts[1],
            "language_id": parts[2] if len(parts) > 2 else "",
            "speaker_id": "",
            "session_id": parts[3] if len(parts) > 3 else "",
        }
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest_fingerprint(*paths: Path) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def split_records(
    records: list[dict[str, Any]],
    dev_percent: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records while keeping recording sessions together when possible."""
    shuffled = list(records)
    random.Random(int(seed)).shuffle(shuffled)
    if len(shuffled) < 2 or float(dev_percent) <= 0:
        return shuffled, []

    target = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * float(dev_percent) / 100)),
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    for row in shuffled:
        session_id = str(row.get("session_id") or "").strip()
        if session_id:
            grouped.setdefault(session_id, []).append(row)
        else:
            ungrouped.append(row)

    if len(grouped) >= 2 and not ungrouped:
        keys = list(grouped)
        random.Random(int(seed)).shuffle(keys)
        dev_rows: list[dict[str, Any]] = []
        selected: set[str] = set()
        for key in keys[:-1]:
            if len(dev_rows) >= target:
                break
            selected.add(key)
            dev_rows.extend(grouped[key])
        train_rows = [
            row for key in keys if key not in selected for row in grouped[key]
        ]
        if train_rows and dev_rows:
            return train_rows, dev_rows

    return shuffled[target:], shuffled[:target]


def _commit_dataset(staging: Path, project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    for name in ("audio", "manifests"):
        target = project / name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staging / name), str(target))
    if (project / "tokens").exists():
        shutil.rmtree(project / "tokens")
    shutil.rmtree(staging, ignore_errors=True)


def prepare_dataset(
    root: Path,
    project_name: str,
    uploads: list[Any] | None,
    manifest: Any | None,
    transcripts: str,
    default_language: str,
    dev_percent: float,
    seed: int,
    normalize_thai: bool,
    reject_unusable: bool,
):
    if not uploads:
        raise gr.Error("กรุณาอัปโหลดไฟล์เสียง")

    project = root / safe_name(project_name)
    project.mkdir(parents=True, exist_ok=True)
    staging = project / f".dataset-staging-{uuid4().hex[:10]}"
    audio_dir = staging / "audio"
    manifest_dir = staging / "manifests"
    audio_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    try:
        mapping = parse_manifest(manifest)
        mapping.update(parse_transcripts(transcripts, len(uploads)))
        records: list[dict[str, Any]] = []
        missing: list[str] = []

        for index, item in enumerate(uploads, 1):
            source = upload_path(item)
            if source.suffix.lower() not in AUDIO_EXTENSIONS:
                raise gr.Error(f"ไม่รองรับไฟล์ {source.name}")
            metadata = mapping.get(source.name) or (
                mapping.get("__single__") if len(uploads) == 1 else None
            )
            if not metadata or not metadata["text"]:
                missing.append(source.name)
                continue

            destination = audio_dir / re.sub(r"[^A-Za-z0-9._-]+", "_", source.name)
            suffix = 2
            while destination.exists():
                destination = audio_dir / f"{source.stem}_{suffix}{source.suffix}"
                suffix += 1
            shutil.copy2(source, destination)

            language_id = metadata["language_id"] or (default_language or "").strip()
            raw_text = metadata["text"].strip()
            normalized_text = (
                normalize_thai_text(raw_text)
                if normalize_thai and language_id.lower().startswith("th")
                else raw_text
            )
            records.append(
                {
                    "id": f"{safe_name(project_name)}_{index:05d}",
                    "audio_path": str((project / "audio" / destination.name).resolve()),
                    "_scan_path": str(destination.resolve()),
                    "text": normalized_text,
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "language_id": language_id,
                    "speaker_id": metadata.get("speaker_id", ""),
                    "session_id": metadata.get("session_id", ""),
                    "source_filename": source.name,
                    "source_sha256": _file_sha256(destination),
                }
            )

        if missing:
            raise gr.Error("ยังไม่มี transcript สำหรับ: " + ", ".join(missing[:10]))
        if not records:
            raise gr.Error("ไม่มีไฟล์ที่ใช้สร้าง dataset ได้")

        quality: DatasetQualitySummary = scan_dataset(
            row["_scan_path"] for row in records
        )
        quality_by_path = {
            str(Path(result.path).resolve()): result for result in quality.results
        }
        retained: list[dict[str, Any]] = []
        rejected = 0
        for row in records:
            result = quality_by_path.get(str(Path(row["_scan_path"]).resolve()))
            if result is not None:
                result.path = row["audio_path"]
                row["quality_status"] = result.status
                row["quality_issues"] = result.fatal_issues + result.issues
                if reject_unusable and not result.usable:
                    rejected += 1
                    continue
            row.pop("_scan_path", None)
            retained.append(row)
        records = retained
        if not records:
            raise gr.Error("ทุกไฟล์ถูกปฏิเสธโดยตัวตรวจคุณภาพ")

        train_rows, dev_rows = split_records(records, dev_percent, int(seed))
        train_manifest = manifest_dir / "train.jsonl"
        dev_manifest = manifest_dir / "dev.jsonl"
        _write_jsonl(train_manifest, train_rows)
        if dev_rows:
            _write_jsonl(dev_manifest, dev_rows)
        quality.write_json(manifest_dir / "quality_report.json")

        fingerprint = _manifest_fingerprint(train_manifest, dev_manifest)
        dataset_meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "project": safe_name(project_name),
            "normalize_thai": bool(normalize_thai),
            "reject_unusable": bool(reject_unusable),
            "train_samples": len(train_rows),
            "dev_samples": len(dev_rows),
            "quality": quality.to_dict()["summary"],
        }
        (manifest_dir / "dataset.json").write_text(
            json.dumps(dataset_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _commit_dataset(staging, project)

        preview_lines = []
        for row in records[:20]:
            text = row["text"][:120]
            if row["raw_text"] != row["text"]:
                text = f"{row['raw_text'][:60]} → {text}"
            preview_lines.append(
                f"{Path(row['audio_path']).name} | {text} | "
                f"{row['language_id'] or '-'} | {row.get('quality_status', '-')}"
            )
        status = (
            f"เตรียมสำเร็จ **{len(train_rows)} train / {len(dev_rows)} dev**"
            f"  \nDataset fingerprint: `{fingerprint[:16]}`  \n`{project}`"
        )
        if rejected:
            status += f"  \nปฏิเสธไฟล์ที่ใช้ไม่ได้: **{rejected}**"
        return status, "\n".join(preview_lines), quality.to_markdown()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@dataclass(frozen=True)
class CommandStep:
    stage: str
    command: list[str]


def _update_run_manifest(run_dir: Path, status: str, message: str | None = None) -> None:
    path = run_dir / "run.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data["status"] = status
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if message:
            data["message"] = message
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


class Trainer:
    def __init__(self):
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.lines: list[str] = []
        self.state = "idle"
        self.stop_requested = False
        self.run_dir: Path | None = None

    def snapshot(self):
        with self.lock:
            return self.state, "\n".join(self.lines[-800:])

    def active(self):
        return self.state in {"preparing", "tokenizing", "training", "stopping"}

    def append(self, line: str):
        with self.lock:
            self.lines.append(line.rstrip())
            self.lines = self.lines[-4000:]

    def start(
        self,
        steps: list[CommandStep],
        env: dict[str, str],
        run_dir: Path,
    ):
        with self.lock:
            if self.active():
                raise gr.Error("มีงานเทรนกำลังทำงานอยู่แล้ว")
            self.lines, self.state, self.stop_requested = [], "preparing", False
            self.run_dir = run_dir
        _update_run_manifest(run_dir, "preparing")
        threading.Thread(
            target=self.run,
            args=(steps, env, run_dir),
            daemon=True,
        ).start()

    def run(
        self,
        steps: list[CommandStep],
        env: dict[str, str],
        run_dir: Path,
    ):
        log_path = run_dir / "training.log"
        try:
            for step in steps:
                with self.lock:
                    if self.stop_requested:
                        self.state = "stopped"
                        _update_run_manifest(run_dir, "stopped")
                        return
                    self.state = step.stage
                _update_run_manifest(run_dir, step.stage)
                self.append("$ " + shlex.join(step.command))
                kwargs: dict[str, Any] = dict(
                    cwd=str(repo_root()),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if os.name != "nt":
                    kwargs["start_new_session"] = True
                self.process = subprocess.Popen(step.command, **kwargs)
                assert self.process.stdout is not None
                with log_path.open("a", encoding="utf-8") as log:
                    for line in self.process.stdout:
                        self.append(line)
                        log.write(line)
                        log.flush()
                code = self.process.wait()
                self.process = None
                if code:
                    with self.lock:
                        self.state = "stopped" if self.stop_requested else "failed"
                    message = f"Process exited with code {code}"
                    self.append(message)
                    _update_run_manifest(run_dir, self.state, message)
                    return
            with self.lock:
                self.state = "completed"
            self.append("Training completed successfully")
            _update_run_manifest(run_dir, "completed")
        except Exception as exc:
            message = f"ERROR: {type(exc).__name__}: {exc}"
            self.append(message)
            with self.lock:
                self.state = "failed"
                self.process = None
            _update_run_manifest(run_dir, "failed", message)

    def stop(self):
        with self.lock:
            if not self.active():
                return self.snapshot()
            self.stop_requested, self.state, process = True, "stopping", self.process
            run_dir = self.run_dir
        if run_dir:
            _update_run_manifest(run_dir, "stopping")
        if process and process.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.append("Stop requested")
        return self.snapshot()


TRAINER = Trainer()


def _allocate_run_dir(project: Path, requested_name: str) -> Path:
    default = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    base = optional_safe_name(requested_name, default)
    runs = project / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    candidate = runs / base
    suffix = 2
    while candidate.exists():
        candidate = runs / f"{base}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_plan(
    root: Path,
    project_name: str,
    run_name: str,
    consent: bool,
    gpu_ids: str,
    tokenizer: str,
    checkpoint: str,
    attention: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    min_free_gb: float,
    steps: int,
    learning_rate: float,
    batch_tokens: int,
    grad_accum: int,
    precision: str,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    keep_last: int,
    seed: int,
    retokenize: bool,
):
    if not consent:
        raise gr.Error("ต้องยืนยันสิทธิ์การใช้เสียงก่อน")
    if int(steps) <= 0 or int(batch_tokens) <= 0 or int(grad_accum) <= 0:
        raise gr.Error("Steps, batch tokens และ gradient accumulation ต้องมากกว่า 0")
    if int(save_steps) <= 0 or int(eval_steps) <= 0 or int(logging_steps) <= 0:
        raise gr.Error("ช่วงบันทึก ประเมิน และ log ต้องมากกว่า 0")
    if int(keep_last) == 0 or int(keep_last) < -1:
        raise gr.Error("Keep checkpoints ต้องเป็น -1 หรือจำนวนเต็มบวก")

    project = root / safe_name(project_name)
    train_manifest = project / "manifests/train.jsonl"
    dev_manifest = project / "manifests/dev.jsonl"
    if not train_manifest.exists():
        raise gr.Error("กรุณาเตรียม Dataset ก่อน")

    preflight = run_preflight(
        gpu_ids,
        project,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        min_free_gb=float(min_free_gb),
    )
    if not preflight.ok:
        errors = "\n".join(
            f"- {check.name}: {check.message}" for check in preflight.errors
        )
        raise gr.Error("Preflight ไม่ผ่าน\n" + errors)

    run_dir = _allocate_run_dir(project, run_name)
    config_dir = run_dir / "config"
    output_dir = run_dir / "checkpoints"
    token_dir = project / "tokens"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight.write_json(run_dir / "preflight.json")

    if use_lora:
        base_name = "train_config_finetune_lora.json"
    elif attention == "sdpa":
        base_name = "train_config_finetune_sdpa.json"
    else:
        base_name = "train_config_finetune.json"
    base_path = repo_root() / "examples/config" / base_name
    if not base_path.exists():
        raise gr.Error(f"ไม่พบ config ต้นแบบ {base_path}")
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config.update(
        init_from_checkpoint=checkpoint.strip(),
        use_lora=bool(use_lora),
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        attn_implementation=attention,
        steps=int(steps),
        learning_rate=float(learning_rate),
        batch_tokens=int(batch_tokens),
        gradient_accumulation_steps=int(grad_accum),
        mixed_precision=precision,
        save_steps=int(save_steps),
        eval_steps=int(eval_steps),
        logging_steps=int(logging_steps),
        keep_last_n_checkpoints=int(keep_last),
        seed=int(seed),
    )
    train_config = config_dir / "train_config.json"
    train_config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    data = {
        "train": [
            {"manifest_path": [str((token_dir / "train/data.lst").resolve())]}
        ],
        "dev": [],
    }
    if dev_manifest.exists():
        data["dev"] = [
            {"manifest_path": [str((token_dir / "dev/data.lst").resolve())]}
        ]
    data_config = config_dir / "data_config.json"
    data_config.write_text(json.dumps(data, indent=2), encoding="utf-8")

    fingerprint = _manifest_fingerprint(train_manifest, dev_manifest)
    token_fingerprint = token_dir / "dataset.sha256"
    cached_fingerprint = (
        token_fingerprint.read_text(encoding="utf-8").strip()
        if token_fingerprint.exists()
        else ""
    )
    needs_tokenization = bool(retokenize) or cached_fingerprint != fingerprint

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
    use_cpu, requested_gpu_ids = parse_gpu_ids(gpu_ids)
    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        accelerate_gpu_ids = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(item) for item in requested_gpu_ids
        )
        accelerate_gpu_ids = ",".join(
            str(index) for index in range(len(requested_gpu_ids))
        )

    command_steps: list[CommandStep] = []
    if needs_tokenization:
        if token_dir.exists():
            shutil.rmtree(token_dir)
        for split, manifest_path in (
            ("train", train_manifest),
            ("dev", dev_manifest),
        ):
            if manifest_path.exists():
                command_steps.append(
                    CommandStep(
                        "tokenizing",
                        [
                            sys.executable,
                            "-m",
                            "omnivoice.scripts.extract_audio_tokens",
                            "--input_jsonl",
                            str(manifest_path.resolve()),
                            "--tar_output_pattern",
                            str(
                                (
                                    token_dir
                                    / split
                                    / "audios/shard-%06d.tar"
                                ).resolve()
                            ),
                            "--jsonl_output_pattern",
                            str(
                                (
                                    token_dir
                                    / split
                                    / "txts/shard-%06d.jsonl"
                                ).resolve()
                            ),
                            "--tokenizer_path",
                            tokenizer.strip(),
                            "--nj_per_gpu",
                            "3",
                            "--shuffle",
                            "True",
                        ],
                    )
                )
        marker_code = (
            "from pathlib import Path; "
            f"p=Path({str(token_fingerprint)!r}); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            f"p.write_text({fingerprint!r}, encoding='utf-8')"
        )
        command_steps.append(
            CommandStep("preparing", [sys.executable, "-c", marker_code])
        )

    accelerate = (
        [shutil.which("accelerate")]
        if shutil.which("accelerate")
        else [sys.executable, "-m", "accelerate.commands.launch"]
    )
    if use_cpu:
        accelerate += ["--cpu", "--num_processes", "1"]
    else:
        accelerate += [
            "--gpu_ids",
            accelerate_gpu_ids,
            "--num_processes",
            str(len(requested_gpu_ids)),
        ]
    train_command = accelerate + [
        "-m",
        "omnivoice.cli.train",
        "--train_config",
        str(train_config.resolve()),
        "--data_config",
        str(data_config.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
    ]
    command_steps.append(CommandStep("training", train_command))

    run_manifest = {
        "run_id": run_dir.name,
        "project": project.name,
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "git_revision": _git_revision(),
        "base_checkpoint": checkpoint.strip(),
        "audio_tokenizer": tokenizer.strip(),
        "use_lora": bool(use_lora),
        "config_sha256": _file_sha256(train_config),
        "token_cache_reused": not needs_tokenization,
        "commands": [asdict(step) for step in command_steps],
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return command_steps, env, run_dir


def start_training(root: Path, *values):
    steps, env, run_dir = make_plan(root, *values)
    TRAINER.start(steps, env, run_dir)
    while TRAINER.active():
        state, logs = TRAINER.snapshot()
        yield f"สถานะ: **{state}**  \nRun: `{run_dir}`", logs
        time.sleep(1)
    state, logs = TRAINER.snapshot()
    yield f"สถานะ: **{state}**  \nRun: `{run_dir}`", logs


def check_environment(
    root: Path,
    project_name: str,
    gpu_ids: str,
    checkpoint: str,
    tokenizer: str,
    min_free_gb: float,
) -> str:
    project = root / safe_name(project_name)
    return run_preflight(
        gpu_ids,
        project,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        min_free_gb=float(min_free_gb),
    ).to_markdown()


def build_demo(root: Path):
    with gr.Blocks(title="OmniVoice Cherry Trainer") as demo:
        gr.Markdown(
            "# 🍒 OmniVoice Cherry — Voice Training Studio\n"
            "ตรวจข้อมูล สร้าง dataset และ fine-tune แบบ versioned runs"
        )
        gr.Markdown(
            "> ใช้เฉพาะเสียงที่ตนเองเป็นเจ้าของหรือได้รับอนุญาต "
            "และตรวจ license ของ model weights กับ tokenizer ก่อนนำไปใช้เชิงพาณิชย์"
        )
        with gr.Tab("1. Dataset"):
            with gr.Row():
                project = gr.Textbox(value="cherry-voice", label="ชื่อโปรเจกต์")
                language = gr.Textbox(value="th", label="Language ID")
                dev_percent = gr.Slider(
                    0,
                    30,
                    value=10,
                    step=1,
                    label="Dev split (%)",
                )
                dataset_seed = gr.Number(value=42, precision=0, label="Split seed")
            audio = gr.File(
                file_count="multiple",
                file_types=["audio"],
                type="filepath",
                label="ไฟล์เสียง",
            )
            manifest = gr.File(
                file_types=[".csv", ".tsv", ".jsonl"],
                type="filepath",
                label="Manifest (ไม่บังคับ)",
            )
            transcript = gr.Textbox(
                lines=10,
                label="Transcript",
                placeholder="filename.wav|ข้อความ|th|session-01",
            )
            with gr.Row():
                thai_normalize = gr.Checkbox(
                    value=True, label="Normalize ข้อความภาษาไทย"
                )
                reject_unusable = gr.Checkbox(
                    value=True, label="ตัดไฟล์ว่างหรืออ่านไม่ได้"
                )
            prepare = gr.Button("ตรวจและเตรียม Dataset", variant="primary")
            dataset_status = gr.Markdown()
            preview = gr.Textbox(
                lines=10,
                interactive=False,
                label="ตัวอย่าง transcript",
            )
            quality_report = gr.Markdown()
            prepare.click(
                lambda *args: prepare_dataset(root, *args),
                [
                    project,
                    audio,
                    manifest,
                    transcript,
                    language,
                    dev_percent,
                    dataset_seed,
                    thai_normalize,
                    reject_unusable,
                ],
                [dataset_status, preview, quality_report],
            )

        with gr.Tab("2. Fine-tune"):
            with gr.Row():
                run_name = gr.Textbox(label="Run name (เว้นว่างให้ระบบตั้ง)")
                consent = gr.Checkbox(
                    label="ฉันยืนยันว่ามีสิทธิ์ใช้เสียงทั้งหมด"
                )
            with gr.Row():
                gpu = gr.Textbox(value="0", label="GPU IDs")
                attention = gr.Radio(
                    ["flex_attention", "sdpa"],
                    value="sdpa",
                    label="Attention",
                )
                precision = gr.Dropdown(
                    ["bf16", "fp16", "no"],
                    value="bf16",
                    label="Precision",
                )
                min_disk = gr.Number(value=20, label="พื้นที่ว่างขั้นต่ำ (GB)")
            with gr.Row():
                tokenizer = gr.Textbox(
                    value="eustlb/higgs-audio-v2-tokenizer",
                    label="Audio tokenizer",
                )
                checkpoint = gr.Textbox(
                    value="k2-fsa/OmniVoice",
                    label="Base checkpoint",
                )
            with gr.Accordion("LoRA", open=True):
                use_lora = gr.Checkbox(value=True, label="ใช้ LoRA (แนะนำ)")
                with gr.Row():
                    lora_r = gr.Number(value=16, precision=0, label="LoRA rank")
                    lora_alpha = gr.Number(
                        value=32,
                        precision=0,
                        label="LoRA alpha",
                    )
                    lora_dropout = gr.Number(
                        value=0.05,
                        label="LoRA dropout",
                    )
            with gr.Accordion("Training parameters", open=True):
                with gr.Row():
                    steps = gr.Number(value=5000, precision=0, label="Steps")
                    lr = gr.Number(value=1e-4, label="Learning rate")
                    batch = gr.Number(
                        value=8192,
                        precision=0,
                        label="Batch tokens",
                    )
                    grad = gr.Number(
                        value=1,
                        precision=0,
                        label="Gradient accumulation",
                    )
                with gr.Row():
                    save = gr.Number(
                        value=500,
                        precision=0,
                        label="Save every",
                    )
                    evaluate = gr.Number(
                        value=500,
                        precision=0,
                        label="Evaluate every",
                    )
                    logging = gr.Number(
                        value=50,
                        precision=0,
                        label="Log every",
                    )
                    keep_last = gr.Number(
                        value=3,
                        precision=0,
                        label="Keep latest checkpoints (-1 = all)",
                    )
                    train_seed = gr.Number(
                        value=42,
                        precision=0,
                        label="Seed",
                    )
                retokenize = gr.Checkbox(
                    label="Tokenize ใหม่แม้ dataset ไม่เปลี่ยน"
                )

            preflight_button = gr.Button("ตรวจระบบก่อนเทรน")
            preflight_output = gr.Markdown()
            preflight_button.click(
                lambda *args: check_environment(root, *args),
                [project, gpu, checkpoint, tokenizer, min_disk],
                preflight_output,
            )

            start = gr.Button("เริ่ม Fine-tune", variant="primary")
            stop = gr.Button("หยุด", variant="stop")
            status = gr.Markdown("สถานะ: **idle**")
            logs = gr.Textbox(
                lines=24,
                interactive=False,
                label="Training log",
            )
            start.click(
                lambda *args: start_training(root, *args),
                [
                    project,
                    run_name,
                    consent,
                    gpu,
                    tokenizer,
                    checkpoint,
                    attention,
                    use_lora,
                    lora_r,
                    lora_alpha,
                    lora_dropout,
                    min_disk,
                    steps,
                    lr,
                    batch,
                    grad,
                    precision,
                    save,
                    evaluate,
                    logging,
                    keep_last,
                    train_seed,
                    retokenize,
                ],
                [status, logs],
            )
            stop.click(
                lambda: (
                    f"สถานะ: **{TRAINER.stop()[0]}**",
                    TRAINER.snapshot()[1],
                ),
                [],
                [status, logs],
                queue=False,
            )
        gr.Markdown(
            "ผลลัพธ์อยู่ใน `data/gui_projects/<project>/runs/<run-id>/` "
            "ส่วน token cache อยู่ระดับโปรเจกต์และจะใช้ซ้ำเฉพาะ dataset fingerprint เดิม"
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--root-dir",
        default=str(repo_root() / "data/gui_projects"),
    )
    args = parser.parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    build_demo(root).queue(default_concurrency_limit=4).launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
