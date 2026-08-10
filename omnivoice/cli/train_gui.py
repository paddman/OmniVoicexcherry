"""Gradio GUI for preparing datasets and fine-tuning OmniVoice."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-._")
    if not value:
        raise gr.Error("กรุณาตั้งชื่อโปรเจกต์")
    return value[:80]


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
            source = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
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
                }
    except (OSError, csv.Error, json.JSONDecodeError) as exc:
        raise gr.Error(f"อ่าน manifest ไม่สำเร็จ: {exc}") from exc
    return rows


def parse_transcripts(raw: str, count: int) -> dict[str, dict[str, str]]:
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if not lines:
        return {}
    if count == 1 and len(lines) == 1 and "|" not in lines[0] and "\t" not in lines[0]:
        return {"__single__": {"text": lines[0], "language_id": ""}}
    result: dict[str, dict[str, str]] = {}
    for number, line in enumerate(lines, 1):
        delimiter = "|" if "|" in line else "\t" if "\t" in line else None
        if delimiter is None:
            raise gr.Error(f"Transcript บรรทัด {number} ต้องเป็น filename|ข้อความ|language_id")
        parts = [part.strip() for part in line.split(delimiter, 2)]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise gr.Error(f"Transcript บรรทัด {number} ไม่สมบูรณ์")
        result[Path(parts[0]).name] = {
            "text": parts[1],
            "language_id": parts[2] if len(parts) > 2 else "",
        }
    return result


def prepare_dataset(root: Path, project_name: str, uploads: list[Any] | None, manifest: Any | None,
                    transcripts: str, default_language: str, dev_percent: float, seed: int):
    if not uploads:
        raise gr.Error("กรุณาอัปโหลดไฟล์เสียง")
    project = root / safe_name(project_name)
    for stale in (project / "audio", project / "manifests", project / "tokens"):
        if stale.exists():
            shutil.rmtree(stale)
    audio_dir = project / "audio"
    manifest_dir = project / "manifests"
    audio_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    mapping = parse_manifest(manifest)
    mapping.update(parse_transcripts(transcripts, len(uploads)))
    records: list[dict[str, str]] = []
    missing: list[str] = []
    for index, item in enumerate(uploads, 1):
        source = upload_path(item)
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            raise gr.Error(f"ไม่รองรับไฟล์ {source.name}")
        metadata = mapping.get(source.name) or (mapping.get("__single__") if len(uploads) == 1 else None)
        if not metadata or not metadata["text"]:
            missing.append(source.name)
            continue
        destination = audio_dir / re.sub(r"[^A-Za-z0-9._-]+", "_", source.name)
        suffix = 2
        while destination.exists():
            destination = audio_dir / f"{source.stem}_{suffix}{source.suffix}"
            suffix += 1
        shutil.copy2(source, destination)
        records.append({
            "id": f"{safe_name(project_name)}_{index:05d}",
            "audio_path": str(destination.resolve()),
            "text": metadata["text"],
            "language_id": metadata["language_id"] or (default_language or "").strip(),
        })
    if missing:
        raise gr.Error("ยังไม่มี transcript สำหรับ: " + ", ".join(missing[:10]))

    random.Random(int(seed)).shuffle(records)
    dev_count = 0 if len(records) < 2 else min(len(records) - 1, max(0, round(len(records) * float(dev_percent) / 100)))
    if dev_percent and len(records) > 1:
        dev_count = max(1, dev_count)
    dev_rows, train_rows = records[:dev_count], records[dev_count:]
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        path = manifest_dir / f"{name}.jsonl"
        if rows:
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    preview = "\n".join(f"{Path(row['audio_path']).name} | {row['text'][:100]} | {row['language_id'] or '-'}" for row in records[:20])
    return f"เตรียมสำเร็จ **{len(train_rows)} train / {len(dev_rows)} dev**  \n`{project}`", preview


class Trainer:
    def __init__(self):
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.lines: list[str] = []
        self.state = "idle"
        self.stop_requested = False

    def snapshot(self):
        with self.lock:
            return self.state, "\n".join(self.lines[-800:])

    def active(self):
        return self.state in {"preparing", "tokenizing", "training", "stopping"}

    def append(self, line: str):
        with self.lock:
            self.lines.append(line.rstrip())
            self.lines = self.lines[-4000:]

    def start(self, commands: list[list[str]], env: dict[str, str], project: Path):
        with self.lock:
            if self.active():
                raise gr.Error("มีงานเทรนกำลังทำงานอยู่แล้ว")
            self.lines, self.state, self.stop_requested = [], "preparing", False
        threading.Thread(target=self.run, args=(commands, env, project), daemon=True).start()

    def run(self, commands: list[list[str]], env: dict[str, str], project: Path):
        log_path = project / "training.log"
        try:
            for index, command in enumerate(commands):
                with self.lock:
                    if self.stop_requested:
                        self.state = "stopped"
                        return
                    self.state = "tokenizing" if index < len(commands) - 1 else "training"
                self.append("$ " + " ".join(command))
                kwargs: dict[str, Any] = dict(cwd=str(repo_root()), env=env, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                if os.name != "nt":
                    kwargs["start_new_session"] = True
                self.process = subprocess.Popen(command, **kwargs)
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
                    self.append(f"Process exited with code {code}")
                    return
            with self.lock:
                self.state = "completed"
            self.append("Training completed successfully")
        except Exception as exc:
            self.append(f"ERROR: {type(exc).__name__}: {exc}")
            with self.lock:
                self.state = "failed"
                self.process = None

    def stop(self):
        with self.lock:
            self.stop_requested, self.state, process = True, "stopping", self.process
        if process and process.poll() is None:
            try:
                process.terminate() if os.name == "nt" else os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.append("Stop requested")
        return self.snapshot()


TRAINER = Trainer()


def make_plan(root: Path, project_name: str, consent: bool, gpu_ids: str, tokenizer: str,
              checkpoint: str, attention: str, steps: int, learning_rate: float,
              batch_tokens: int, grad_accum: int, precision: str, save_steps: int,
              eval_steps: int, logging_steps: int, seed: int, retokenize: bool):
    if not consent:
        raise gr.Error("ต้องยืนยันสิทธิ์การใช้เสียงก่อน")
    project = root / safe_name(project_name)
    train_manifest, dev_manifest = project / "manifests/train.jsonl", project / "manifests/dev.jsonl"
    if not train_manifest.exists():
        raise gr.Error("กรุณาเตรียม Dataset ก่อน")
    config_dir, token_dir, output_dir = project / "config", project / "tokens", project / "checkpoints"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = "train_config_finetune_sdpa.json" if attention == "sdpa" else "train_config_finetune.json"
    config = json.loads((repo_root() / "examples/config" / base_name).read_text(encoding="utf-8"))
    config.update(init_from_checkpoint=checkpoint.strip(), steps=int(steps), learning_rate=float(learning_rate),
                  batch_tokens=int(batch_tokens), gradient_accumulation_steps=int(grad_accum),
                  mixed_precision=precision, save_steps=int(save_steps), eval_steps=int(eval_steps),
                  logging_steps=int(logging_steps), seed=int(seed))
    train_config = config_dir / "train_config.json"
    train_config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    data = {"train": [{"manifest_path": [str((token_dir / 'train/data.lst').resolve())]}], "dev": []}
    if dev_manifest.exists():
        data["dev"] = [{"manifest_path": [str((token_dir / 'dev/data.lst').resolve())]}]
    data_config = config_dir / "data_config.json"
    data_config.write_text(json.dumps(data, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
    gpu_ids = gpu_ids.strip()
    gpu_list = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    use_cpu = gpu_ids.lower() in {"", "cpu", "none"}
    if not use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    commands: list[list[str]] = []
    if retokenize or not (token_dir / "train/data.lst").exists():
        for split, manifest in (("train", train_manifest), ("dev", dev_manifest)):
            if manifest.exists():
                commands.append([sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
                                 "--input_jsonl", str(manifest.resolve()),
                                 "--tar_output_pattern", str((token_dir / split / "audios/shard-%06d.tar").resolve()),
                                 "--jsonl_output_pattern", str((token_dir / split / "txts/shard-%06d.jsonl").resolve()),
                                 "--tokenizer_path", tokenizer.strip(), "--nj_per_gpu", "3", "--shuffle", "True"])
    accelerate = [shutil.which("accelerate")] if shutil.which("accelerate") else [sys.executable, "-m", "accelerate.commands.launch"]
    accelerate += ["--cpu", "--num_processes", "1"] if use_cpu else ["--gpu_ids", gpu_ids, "--num_processes", str(len(gpu_list))]
    commands.append(accelerate + ["-m", "omnivoice.cli.train", "--train_config", str(train_config.resolve()),
                                  "--data_config", str(data_config.resolve()), "--output_dir", str(output_dir.resolve())])
    return commands, env, project


def start_training(root: Path, *values):
    commands, env, project = make_plan(root, *values)
    TRAINER.start(commands, env, project)
    while TRAINER.active():
        state, logs = TRAINER.snapshot()
        yield f"สถานะ: **{state}**", logs
        time.sleep(1)
    state, logs = TRAINER.snapshot()
    yield f"สถานะ: **{state}**", logs


def build_demo(root: Path):
    with gr.Blocks(title="OmniVoice Cherry Trainer") as demo:
        gr.Markdown("# 🍒 OmniVoice Cherry — Voice Training Studio\nอัปโหลดเสียง สร้าง dataset และ fine-tune ผ่าน GUI")
        gr.Markdown("> ใช้เฉพาะเสียงของตนเองหรือเสียงที่ได้รับอนุญาต ห้ามใช้เพื่อปลอมตัว หลอกลวง หรือทำให้ผู้อื่นเสียหาย")
        with gr.Tab("1. Dataset"):
            with gr.Row():
                project = gr.Textbox(value="cherry-voice", label="ชื่อโปรเจกต์")
                language = gr.Textbox(value="th", label="Language ID")
                dev_percent = gr.Slider(0, 30, value=10, step=1, label="Dev split (%)")
                dataset_seed = gr.Number(value=42, precision=0, label="Split seed")
            audio = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="ไฟล์เสียง")
            manifest = gr.File(file_types=[".csv", ".tsv", ".jsonl"], type="filepath", label="Manifest (ไม่บังคับ)")
            transcript = gr.Textbox(lines=10, label="Transcript", placeholder="filename.wav|ข้อความ|th")
            prepare = gr.Button("เตรียม Dataset", variant="primary")
            dataset_status, preview = gr.Markdown(), gr.Textbox(lines=10, interactive=False, label="ตัวอย่าง")
            prepare.click(lambda *args: prepare_dataset(root, *args),
                          [project, audio, manifest, transcript, language, dev_percent, dataset_seed],
                          [dataset_status, preview])
        with gr.Tab("2. Fine-tune"):
            consent = gr.Checkbox(label="ฉันยืนยันว่ามีสิทธิ์ใช้เสียงทั้งหมด")
            with gr.Row():
                gpu = gr.Textbox(value="0", label="GPU IDs")
                attention = gr.Radio(["flex_attention", "sdpa"], value="sdpa", label="Attention")
                precision = gr.Dropdown(["bf16", "fp16", "no"], value="bf16", label="Precision")
            with gr.Row():
                tokenizer = gr.Textbox(value="eustlb/higgs-audio-v2-tokenizer", label="Audio tokenizer")
                checkpoint = gr.Textbox(value="k2-fsa/OmniVoice", label="Base checkpoint")
            with gr.Accordion("Training parameters", open=True):
                with gr.Row():
                    steps = gr.Number(value=5000, precision=0, label="Steps")
                    lr = gr.Number(value=1e-5, label="Learning rate")
                    batch = gr.Number(value=8192, precision=0, label="Batch tokens")
                    grad = gr.Number(value=1, precision=0, label="Gradient accumulation")
                with gr.Row():
                    save = gr.Number(value=500, precision=0, label="Save every")
                    evaluate = gr.Number(value=500, precision=0, label="Evaluate every")
                    logging = gr.Number(value=50, precision=0, label="Log every")
                    train_seed = gr.Number(value=42, precision=0, label="Seed")
                retokenize = gr.Checkbox(label="Tokenize ใหม่")
            start, stop = gr.Button("เริ่ม Fine-tune", variant="primary"), gr.Button("หยุด", variant="stop")
            status, logs = gr.Markdown("สถานะ: **idle**"), gr.Textbox(lines=24, interactive=False, label="Training log")
            start.click(lambda *args: start_training(root, *args),
                        [project, consent, gpu, tokenizer, checkpoint, attention, steps, lr, batch, grad,
                         precision, save, evaluate, logging, train_seed, retokenize], [status, logs])
            stop.click(lambda: (f"สถานะ: **{TRAINER.stop()[0]}**", TRAINER.snapshot()[1]), [], [status, logs], queue=False)
        gr.Markdown("ผลลัพธ์อยู่ใน `data/gui_projects/<project>/checkpoints` พร้อม config และ training.log")
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--root-dir", default=str(repo_root() / "data/gui_projects"))
    args = parser.parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    build_demo(root).queue(default_concurrency_limit=4).launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
