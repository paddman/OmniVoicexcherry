"""Public OmniVoice API with lazy model imports.

Keeping model imports lazy lets lightweight utilities such as dataset checks and
preflight diagnostics run before the full model stack has been loaded.
"""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any
import warnings

warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message="invalid escape sequence",
    module="pydub.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.algorithms.ddp_comm_hooks",
)

try:
    __version__ = version("omnivoice")
except PackageNotFoundError:
    __version__ = "0.0.0"

_MODEL_EXPORTS = {
    "OmniVoice",
    "OmniVoiceConfig",
    "OmniVoiceGenerationConfig",
    "VoiceClonePrompt",
}

if TYPE_CHECKING:
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceConfig,
        OmniVoiceGenerationConfig,
        VoiceClonePrompt,
    )


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        module = import_module("omnivoice.models.omnivoice")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MODEL_EXPORTS)


__all__ = [
    "OmniVoice",
    "OmniVoiceConfig",
    "OmniVoiceGenerationConfig",
    "VoiceClonePrompt",
]
