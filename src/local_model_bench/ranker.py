from __future__ import annotations

import re
from typing import Any

from .models import HardwareInfo, ModelRecommendation


class ModelRanker:
    def __init__(self, hardware: HardwareInfo) -> None:
        self.hardware = hardware

    def rank(self, file: Any) -> ModelRecommendation:
        quant = self.extract_quant(file.filename)
        size_b = self.extract_param_size(file.repo_id + "/" + file.filename)
        fit, fit_score, fit_reason = self.fit(size_b, quant)
        popularity = min(15.0, file.downloads / 20_000) + min(5.0, file.likes / 100)
        quant_score = 18 if quant.startswith(("Q4", "Q5", "UD-Q4", "UD-Q5")) else 12
        size_score = self.size_score(size_b)
        score = round(min(100.0, fit_score + popularity + quant_score + size_score), 1)
        return ModelRecommendation(
            model_name=file.repo_id.split("/")[-1],
            repo_id=file.repo_id,
            pull_name=self.pull_name(file.repo_id, quant),
            quant=quant,
            size_label=self.size_label(size_b),
            fit=fit,
            score=score,
            recommended_uses=self.uses(size_b),
            reason=f"{fit}: {self.size_label(size_b)}, {quant or 'unknown quant'}; {fit_reason}",
            downloads=file.downloads,
            likes=file.likes,
            updated_at=file.updated_at,
            trending_page=getattr(file, "trending_page", 0),
        )

    def fit(self, size_b: float, quant: str = "") -> tuple[str, float, str]:
        if size_b <= 0:
            return "Conditional", 18.0, "parameter size is unknown, manual check needed"

        estimated_memory = self.estimated_memory_gb(size_b, quant)
        estimated_storage = max(0.1, estimated_memory * 1.05)
        if self.hardware.preferred_drive_free_gb and self.hardware.preferred_drive_free_gb < estimated_storage:
            return (
                "Not recommended",
                4.0,
                f"needs about {estimated_storage:.1f}GB free storage, only {self.hardware.preferred_drive_free_gb:.1f}GB detected",
            )

        usable_vram = self.hardware.vram_gb * 0.88 if self.hardware.vram_gb else 0.0
        usable_ram = self.hardware.ram_gb * 0.72 if self.hardware.ram_gb else 0.0
        memory_note = (
            f"estimated {estimated_memory:.1f}GB model memory vs "
            f"{self.hardware.vram_gb:.1f}GB VRAM / {self.hardware.ram_gb:.1f}GB RAM"
        )

        if usable_vram and estimated_memory <= usable_vram:
            return "Ready", 45.0, f"{memory_note}; expected to fit mostly in VRAM"
        if usable_ram and estimated_memory <= usable_ram and size_b <= 22:
            return "Conditional", 34.0, f"{memory_note}; likely needs RAM/CPU offload"
        if usable_ram and estimated_memory <= usable_ram and size_b <= 32:
            return "Conditional", 25.0, f"{memory_note}; large model, expect slower RAM/CPU offload"
        return "Not recommended", 5.0, f"{memory_note}; too large for this hardware profile"

    def estimated_memory_gb(self, size_b: float, quant: str = "") -> float:
        bits = self.quant_bits(quant)
        return round((size_b * bits / 8.0) * 1.18 + 0.8, 1)

    def quant_bits(self, quant: str) -> float:
        normalized = quant.upper()
        if "Q2" in normalized:
            return 2.75
        if "Q3" in normalized:
            return 3.5
        if "Q4" in normalized or "IQ4" in normalized:
            return 4.5
        if "Q5" in normalized or "IQ5" in normalized:
            return 5.5
        if "Q6" in normalized:
            return 6.5
        if "Q8" in normalized:
            return 8.5
        return 4.8

    def size_score(self, size_b: float) -> float:
        if 7 <= size_b <= 14:
            return 20.0
        if 15 <= size_b <= 22:
            return 14.0
        if 23 <= size_b <= 32:
            return 8.0
        if 3 <= size_b < 7:
            return 12.0
        return 4.0

    def uses(self, size_b: float) -> list[str]:
        if size_b <= 10:
            return ["fast_draft", "low_resource_fast", "korean_summary"]
        if size_b <= 16:
            return ["coding_assistant", "code_review", "korean_summary"]
        if size_b <= 24:
            return ["long_analysis", "high_quality_slow", "code_review"]
        return ["high_quality_slow", "long_analysis"]

    def extract_quant(self, text: str) -> str:
        base = text.rsplit("/", 1)[-1].replace(".gguf", "")
        for pattern in [r"(UD-Q\d_[A-Z_]+)", r"(Q\d_[A-Z_]+)", r"(IQ\d_[A-Za-z0-9_]+)"]:
            match = re.search(pattern, base, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def pull_name(self, repo_id: str, quant: str) -> str:
        if self.is_ollama_pull_quant(quant):
            return f"hf.co/{repo_id}:{quant}"
        return f"hf.co/{repo_id}:latest"

    def is_ollama_pull_quant(self, quant: str) -> bool:
        normalized = quant.upper()
        if not normalized:
            return False
        if normalized.startswith("UD-"):
            normalized = normalized.removeprefix("UD-")
        if normalized in {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"}:
            return True
        if re.fullmatch(r"Q[2-6]_K(?:_[SML])?", normalized):
            return True
        return False

    def extract_param_size(self, text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", text)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+(?:\.\d+)?)[-_ ]?[bB][-_ ]", text)
        if match:
            return float(match.group(1))
        return 0.0

    def size_label(self, size_b: float) -> str:
        return f"{size_b:g}B" if size_b > 0 else "Unknown"
