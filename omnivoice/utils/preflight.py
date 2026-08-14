"""Environment diagnostics for OmniVoice training jobs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    level: str  # ok, warning, error
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class PreflightReport:
    checks: list[PreflightCheck]

    @property
    def ok(self) -> bool:
        return not any(check.level == "error" for check in self.checks)

    @property
    def errors(self) -> list[PreflightCheck]:
        return [check for check in self.checks if check.level == "error"]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [check for check in self.checks if check.level == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_markdown(self) -> str:
        icons = {"ok": "✅", "warning": "⚠️", "error": "❌"}
        lines = ["### Training Preflight", ""]
        lines.extend(
            f"- {icons.get(check.level, '•')} **{check.name}:** {check.message}"
            for check in self.checks
        )
        lines.append("")
        lines.append("**พร้อมเริ่มงาน**" if self.ok else "**ยังไม่พร้อมเริ่มงาน กรุณาแก้รายการ ❌ ก่อน**")
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_gpu_ids(value: str | None) -> tuple[bool, list[int]]:
    raw = (value or "").strip().lower()
    if raw in {"", "cpu", "none"}:
        return True, []
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items or any(not item.isdigit() for item in items):
        raise ValueError("GPU IDs ต้องเป็นเลขคั่นด้วย comma เช่น 0 หรือ 0,1 หรือใช้ cpu")
    ids = [int(item) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("GPU IDs มีค่าซ้ำ")
    return False, ids


def _looks_like_local_path(value: str) -> bool:
    value = value.strip()
    return (
        value.startswith((".", "/", "~"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
    )


def _check_reference(checks: list[PreflightCheck], label: str, value: str) -> None:
    value = value.strip()
    if not value:
        checks.append(PreflightCheck(label, "error", "ไม่ได้ระบุค่า"))
        return
    path = Path(value).expanduser()
    if path.exists():
        checks.append(PreflightCheck(label, "ok", f"พบ local path `{path}`"))
    elif _looks_like_local_path(value):
        checks.append(PreflightCheck(label, "error", f"ไม่พบ local path `{path}`"))
    else:
        checks.append(
            PreflightCheck(
                label,
                "warning",
                f"ใช้ remote reference `{value}` ต้องตรวจสิทธิ์เข้าถึงและ license แยกต่างหาก",
            )
        )


def run_preflight(
    gpu_ids: str | None,
    workspace: str | Path,
    checkpoint: str = "k2-fsa/OmniVoice",
    tokenizer: str = "eustlb/higgs-audio-v2-tokenizer",
    min_free_gb: float = 20.0,
) -> PreflightReport:
    checks: list[PreflightCheck] = []

    if sys.version_info >= (3, 10):
        checks.append(PreflightCheck("Python", "ok", sys.version.split()[0]))
    else:
        checks.append(PreflightCheck("Python", "error", f"ต้องใช้ Python 3.10+ แต่พบ {sys.version.split()[0]}"))

    workspace_path = Path(workspace).expanduser().resolve()
    try:
        workspace_path.mkdir(parents=True, exist_ok=True)
        probe = workspace_path / ".omnivoice-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(PreflightCheck("Workspace", "ok", f"เขียนได้ที่ `{workspace_path}`"))
    except OSError as exc:
        checks.append(PreflightCheck("Workspace", "error", f"เขียนไม่ได้: {exc}"))

    try:
        free_gb = shutil.disk_usage(workspace_path).free / (1024**3)
        level = "ok" if free_gb >= min_free_gb else "error"
        checks.append(
            PreflightCheck(
                "Disk",
                level,
                f"ว่าง {free_gb:.1f} GB (ขั้นต่ำที่ตั้งไว้ {min_free_gb:.1f} GB)",
            )
        )
    except OSError as exc:
        checks.append(PreflightCheck("Disk", "error", f"ตรวจพื้นที่ไม่ได้: {exc}"))

    checks.append(
        PreflightCheck(
            "FFmpeg",
            "ok" if shutil.which("ffmpeg") else "warning",
            "พบคำสั่ง ffmpeg" if shutil.which("ffmpeg") else "ไม่พบ ffmpeg ไฟล์ MP3/M4A บางชนิดอาจเปิดไม่ได้",
        )
    )
    checks.append(
        PreflightCheck(
            "Accelerate",
            "ok" if importlib.util.find_spec("accelerate") else "error",
            "พร้อมใช้งาน" if importlib.util.find_spec("accelerate") else "ไม่พบแพ็กเกจ accelerate",
        )
    )

    try:
        use_cpu, requested_ids = parse_gpu_ids(gpu_ids)
    except ValueError as exc:
        checks.append(PreflightCheck("GPU IDs", "error", str(exc)))
        use_cpu, requested_ids = True, []

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        checks.append(PreflightCheck("PyTorch", "error", "ไม่พบแพ็กเกจ torch"))
    else:
        try:
            import torch

            checks.append(PreflightCheck("PyTorch", "ok", str(torch.__version__)))
            if use_cpu:
                checks.append(PreflightCheck("Compute", "warning", "เลือก CPU ซึ่งเหมาะกับ smoke test มากกว่าการฝึกจริง"))
            elif not torch.cuda.is_available():
                checks.append(PreflightCheck("CUDA", "error", "ระบุ GPU แต่ PyTorch มองไม่เห็น CUDA"))
            else:
                device_count = torch.cuda.device_count()
                invalid = [device for device in requested_ids if device >= device_count]
                if invalid:
                    checks.append(
                        PreflightCheck(
                            "GPU IDs",
                            "error",
                            f"ขอ GPU {invalid} แต่ระบบมี index 0 ถึง {device_count - 1}",
                        )
                    )
                else:
                    descriptions: list[str] = []
                    for device in requested_ids:
                        properties = torch.cuda.get_device_properties(device)
                        total_gb = properties.total_memory / (1024**3)
                        free_text = ""
                        try:
                            with torch.cuda.device(device):
                                free_bytes, _ = torch.cuda.mem_get_info()
                            free_text = f", free {free_bytes / (1024**3):.1f} GB"
                        except (RuntimeError, AttributeError):
                            pass
                        descriptions.append(f"{device}: {properties.name} ({total_gb:.1f} GB{free_text})")
                    checks.append(PreflightCheck("CUDA", "ok", "; ".join(descriptions)))
        except Exception as exc:  # defensive: broken CUDA/PyTorch installs can fail during inspection
            checks.append(PreflightCheck("PyTorch", "error", f"ตรวจสอบไม่ได้: {type(exc).__name__}: {exc}"))

    _check_reference(checks, "Base checkpoint", checkpoint)
    _check_reference(checks, "Audio tokenizer", tokenizer)
    return PreflightReport(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an OmniVoice training environment")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--workspace", default="data/gui_projects")
    parser.add_argument("--checkpoint", default="k2-fsa/OmniVoice")
    parser.add_argument("--tokenizer", default="eustlb/higgs-audio-v2-tokenizer")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as a non-zero exit")
    args = parser.parse_args()

    report = run_preflight(
        args.gpu_ids,
        args.workspace,
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        min_free_gb=args.min_free_gb,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.as_json else report.to_markdown())
    if not report.ok or (args.strict and report.warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["PreflightCheck", "PreflightReport", "parse_gpu_ids", "run_preflight"]
