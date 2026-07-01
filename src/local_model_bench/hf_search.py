from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from dataclasses import replace

from .models import HardwareInfo, ModelRecommendation
from .ranker import ModelRanker


@dataclass(frozen=True)
class HfModelFile:
    repo_id: str
    filename: str
    downloads: int = 0
    likes: int = 0
    updated_at: str = ""
    trending_page: int = 0
    tags: tuple[str, ...] = ()
    card_text: str = ""


class HuggingFaceSearch:
    def __init__(self, hardware: HardwareInfo) -> None:
        self.hardware = hardware
        self.ranker = ModelRanker(hardware)

    def recommendations(
        self,
        limit: int = 20,
        *,
        min_params_b: float = 0.0,
        max_params_b: float = 32.0,
        task: str = "text-generation",
        library: str = "gguf",
        app: str = "",
        language: str = "",
        license: str = "",
        provider: str = "",
        other: str = "",
        keyword: str = "",
        page_start: int = 1,
        page_end: int = 3,
    ) -> list[ModelRecommendation]:
        page_start, page_end = self._normalize_page_range(page_start, page_end)
        files = self._search_files(
            task=task,
            library=library,
            app=app,
            language=language,
            license=license,
            provider=provider,
            other=other,
            keyword=keyword,
            search_limit=page_end * 30,
        )
        files = [file for file in files if self._in_page_range(file, page_start, page_end)]
        files = [file for file in files if self._in_parameter_range(file, min_params_b, max_params_b)]
        ranked = [self.ranker.rank(file) for file in files]
        ranked = [self._with_thinking_metadata(item, next((file for file in files if file.repo_id == item.repo_id), None)) for item in ranked]
        if max_params_b <= 32:
            ranked = [item for item in ranked if item.fit != "Not recommended"]
        ranked = self._filter_by_keyword(ranked, keyword)
        ranked.sort(key=lambda item: self._page_score_sort_key(item))
        ranked = self._dedupe_repositories(ranked)
        selected = self._ordered_trending_pages(ranked, limit, page_start, page_end)
        return [self._with_readme_thinking_metadata(item) for item in selected]

    def _normalize_page_range(self, page_start: int, page_end: int) -> tuple[int, int]:
        try:
            start = int(page_start)
            end = int(page_end)
        except (TypeError, ValueError):
            return 1, 3
        start = max(1, min(20, start))
        end = max(1, min(20, end))
        if start > end:
            start, end = end, start
        return start, end

    def _in_page_range(self, file: HfModelFile, page_start: int, page_end: int) -> bool:
        if file.trending_page <= 0:
            return True
        return page_start <= file.trending_page <= page_end

    def _in_parameter_range(self, file: HfModelFile, min_params_b: float, max_params_b: float) -> bool:
        size_b = self.ranker.extract_param_size(file.repo_id + "/" + file.filename)
        if size_b <= 0:
            return min_params_b <= 0
        upper = float("inf") if max_params_b >= 501 else max_params_b
        return min_params_b <= size_b <= upper

    def lookup_thinking_metadata(self, model_name: str) -> dict[str, str]:
        repo_id = self._repo_from_model_name(model_name)
        if repo_id:
            return self._thinking_metadata(repo_id, "", (), self._fetch_readme(repo_id))
        query = self._search_query_from_model_name(model_name)
        if not query:
            return {"thinking_support": "unknown", "thinking_evidence": ""}
        try:
            repos = self._search_models_rest(query, limit=5)
        except Exception:
            return {"thinking_support": "unknown", "thinking_evidence": ""}
        for repo in repos:
            repo_id = str(repo.get("modelId", ""))
            if not repo_id:
                continue
            tags = tuple(str(tag) for tag in repo.get("tags", []) or [])
            siblings = repo.get("siblings", []) or []
            filename = " ".join(str(item.get("rfilename", "")) for item in siblings[:8])
            metadata = self._thinking_metadata(repo_id, filename, tags, self._fetch_readme(repo_id))
            if metadata["thinking_support"] != "unknown":
                return metadata
        return {"thinking_support": "unknown", "thinking_evidence": ""}

    def _dedupe_repositories(self, recs: list[ModelRecommendation]) -> list[ModelRecommendation]:
        seen: set[str] = set()
        unique: list[ModelRecommendation] = []
        for rec in recs:
            key = rec.repo_id.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(rec)
        return unique

    def _page_score_sort_key(self, rec: ModelRecommendation) -> tuple[int, float, str]:
        page = rec.trending_page if rec.trending_page > 0 else 999
        return page, -float(rec.score or 0), rec.repo_id.casefold()

    def _ordered_trending_pages(
        self,
        recs: list[ModelRecommendation],
        limit: int,
        page_start: int = 1,
        page_end: int = 3,
    ) -> list[ModelRecommendation]:
        return [
            rec
            for rec in recs
            if rec.trending_page <= 0 or page_start <= rec.trending_page <= page_end
        ][:limit]

    def _search_files(
        self,
        *,
        task: str = "text-generation",
        library: str = "gguf",
        app: str = "",
        language: str = "",
        license: str = "",
        provider: str = "",
        other: str = "",
        keyword: str = "",
        search_limit: int = 90,
    ) -> list[HfModelFile]:
        if self._repo_id_from_keyword(keyword):
            files: list[HfModelFile] = []
            repo_id = self._repo_id_from_keyword(keyword)
            if repo_id:
                files.extend(self._search_exact_repo(repo_id))
            try:
                files.extend(
                    self._search_with_web_pages(
                        task=task,
                        library=library,
                        app=app,
                        language=language,
                        license=license,
                        provider=provider,
                        other=other,
                        keyword=keyword,
                        page_start=1,
                        page_end=max(1, search_limit // 30),
                    )
                )
            except Exception:
                files.extend(
                    self._search_with_rest(
                        task=task,
                        library=library,
                        app=app,
                        language=language,
                        license=license,
                        provider=provider,
                        other=other,
                        keyword=keyword,
                        search_limit=search_limit,
                    )
                )
            return self._dedupe_files(files)
        try:
            return self._search_with_web_pages(
                task=task,
                library=library,
                app=app,
                language=language,
                license=license,
                provider=provider,
                other=other,
                keyword=keyword,
                page_start=1,
                page_end=max(1, search_limit // 30),
            )
        except Exception:
            return self._search_with_rest(
                task=task,
                library=library,
                app=app,
                language=language,
                license=license,
                provider=provider,
                other=other,
                keyword=keyword,
                search_limit=search_limit,
            )

    def _search_with_web_pages(
        self,
        *,
        task: str,
        library: str,
        app: str,
        language: str,
        license: str,
        provider: str,
        other: str,
        keyword: str,
        page_start: int,
        page_end: int,
    ) -> list[HfModelFile]:
        files: list[HfModelFile] = []
        for page in range(page_start, page_end + 1):
            repo_ids = self._fetch_model_page_repo_ids(
                page,
                task=task,
                library=library,
                app=app,
                language=language,
                license=license,
                provider=provider,
                other=other,
                keyword=keyword,
            )
            for repo_id in repo_ids:
                for file in self._search_exact_repo(repo_id):
                    if self._repo_id_matches_keyword(repo_id, keyword):
                        files.append(replace(file, trending_page=page))
        return self._dedupe_files(files)

    def _fetch_model_page_repo_ids(
        self,
        page: int,
        *,
        task: str,
        library: str,
        app: str,
        language: str,
        license: str,
        provider: str,
        other: str,
        keyword: str,
    ) -> list[str]:
        query_items: list[tuple[str, str]] = [("sort", "trending")]
        if page > 1:
            query_items.append(("p", str(page - 1)))
        if task:
            query_items.append(("pipeline_tag", task))
        if library:
            query_items.append(("library", library))
        if app:
            query_items.append(("apps", app))
        if language:
            query_items.append(("language", language))
        if license:
            query_items.append(("license", license))
        if provider:
            query_items.append(("inference_provider", provider))
        if other == "ungated":
            query_items.append(("gated", "false"))
        query = urllib.parse.urlencode(query_items)
        with urllib.request.urlopen(f"https://huggingface.co/models?{query}", timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
        return self._parse_model_page_repo_ids(html)

    def _parse_model_page_repo_ids(self, html: str) -> list[str]:
        repo_ids: list[str] = []
        seen: set[str] = set()
        ignored = {"models", "datasets", "spaces", "docs", "pricing", "login", "join", "organizations", "settings", "inference"}
        for match in re.finditer(r'href="\/([^"?#]+)"', html):
            repo_id = match.group(1).strip("/")
            if repo_id.count("/") != 1:
                continue
            owner = repo_id.split("/", 1)[0]
            if owner in ignored:
                continue
            key = repo_id.casefold()
            if key in seen:
                continue
            seen.add(key)
            repo_ids.append(repo_id)
        return repo_ids

    def _repo_id_matches_keyword(self, repo_id: str, keyword: str) -> bool:
        terms = [term.casefold() for term in re.split(r"\s+", keyword.strip()) if term.strip()]
        if not terms:
            return True
        text = repo_id.casefold()
        return all(term in text for term in terms)

    def _search_with_library(
        self,
        *,
        task: str,
        library: str,
        app: str,
        language: str,
        license: str,
        provider: str,
        other: str,
        keyword: str,
        search_limit: int,
    ) -> list[HfModelFile]:
        from huggingface_hub import HfApi

        api = HfApi()
        repos = api.list_models(
            search=keyword.strip() or None,
            filter=self._filter_tags(library=library, language=language, license=license),
            apps=app or None,
            gated=False if other == "ungated" else None,
            inference_provider=provider or None,
            pipeline_tag=task or None,
            sort="trendingScore",
            direction=-1,
            limit=search_limit,
            full=True,
        )
        files: list[HfModelFile] = []
        for index, repo in enumerate(repos):
            repo_id = getattr(repo, "modelId", "")
            siblings = getattr(repo, "siblings", []) or []
            for sibling in siblings:
                filename = getattr(sibling, "rfilename", "")
                if self._is_candidate_file(filename):
                    files.append(
                        HfModelFile(
                            repo_id=repo_id,
                            filename=filename,
                            downloads=int(getattr(repo, "downloads", 0) or 0),
                            likes=int(getattr(repo, "likes", 0) or 0),
                            updated_at=str(getattr(repo, "lastModified", "") or ""),
                            trending_page=index // 30 + 1,
                            tags=tuple(str(tag) for tag in (getattr(repo, "tags", []) or [])),
                        )
                    )
        return files

    def _search_with_rest(
        self,
        *,
        task: str,
        library: str,
        app: str,
        language: str,
        license: str,
        provider: str,
        other: str,
        keyword: str,
        search_limit: int,
    ) -> list[HfModelFile]:
        data = self._search_models_rest(
            keyword.strip() or "GGUF",
            limit=search_limit,
            sort="trendingScore",
            task=task,
            library=library,
            app=app,
            language=language,
            license=license,
            provider=provider,
            other=other,
        )
        files: list[HfModelFile] = []
        for index, repo in enumerate(data):
            repo_id = repo.get("modelId", "")
            for sibling in repo.get("siblings", []) or []:
                filename = sibling.get("rfilename", "")
                if self._is_candidate_file(filename):
                    files.append(
                        HfModelFile(
                            repo_id=repo_id,
                            filename=filename,
                            downloads=int(repo.get("downloads", 0) or 0),
                            likes=int(repo.get("likes", 0) or 0),
                            updated_at=str(repo.get("lastModified", "") or ""),
                            trending_page=index // 30 + 1,
                            tags=tuple(str(tag) for tag in repo.get("tags", []) or []),
                        )
                    )
        return files

    def _search_exact_repo(self, repo_id: str) -> list[HfModelFile]:
        try:
            with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe='/')}?blobs=false",
                timeout=30,
            ) as response:
                repo = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        repo_id = str(repo.get("modelId", repo_id) or repo_id)
        files: list[HfModelFile] = []
        for sibling in repo.get("siblings", []) or []:
            filename = sibling.get("rfilename", "")
            if self._is_candidate_file(filename):
                files.append(
                    HfModelFile(
                        repo_id=repo_id,
                        filename=filename,
                        downloads=int(repo.get("downloads", 0) or 0),
                        likes=int(repo.get("likes", 0) or 0),
                        updated_at=str(repo.get("lastModified", "") or ""),
                        trending_page=0,
                        tags=tuple(str(tag) for tag in repo.get("tags", []) or []),
                    )
                )
        return files

    def _dedupe_files(self, files: list[HfModelFile]) -> list[HfModelFile]:
        seen: set[tuple[str, str]] = set()
        unique: list[HfModelFile] = []
        for file in files:
            key = (file.repo_id.casefold(), file.filename.casefold())
            if key in seen:
                continue
            seen.add(key)
            unique.append(file)
        return unique

    def _repo_id_from_keyword(self, keyword: str) -> str:
        text = keyword.strip()
        if not text:
            return ""
        text = re.sub(r"^https?://huggingface\.co/", "", text, flags=re.IGNORECASE)
        text = text.removeprefix("hf.co/")
        text = text.split("?", 1)[0].split("#", 1)[0].strip("/")
        parts = [part for part in text.split("/") if part]
        candidate = "/".join(parts[:2])
        if len(parts) >= 2 and candidate and not any(char.isspace() for char in candidate):
            return candidate
        return ""

    def _filter_by_keyword(self, recs: list[ModelRecommendation], keyword: str) -> list[ModelRecommendation]:
        keyword = re.sub(r"^https?://huggingface\.co/", "", keyword.strip(), flags=re.IGNORECASE)
        keyword = keyword.removeprefix("hf.co/")
        terms = [term.casefold() for term in re.split(r"\s+", keyword) if term.strip()]
        if not terms:
            return recs
        filtered: list[ModelRecommendation] = []
        for rec in recs:
            haystack = " ".join(
                [
                    rec.model_name,
                    rec.repo_id,
                    rec.pull_name,
                    rec.quant,
                    rec.size_label,
                    rec.fit,
                    rec.reason,
                    " ".join(rec.recommended_uses),
                ]
            ).casefold()
            if all(term in haystack for term in terms):
                filtered.append(rec)
        return filtered

    def _search_models_rest(
        self,
        search: str,
        *,
        limit: int,
        sort: str = "downloads",
        task: str = "",
        library: str = "gguf",
        app: str = "",
        language: str = "",
        license: str = "",
        provider: str = "",
        other: str = "",
    ) -> list[dict]:
        query_items: list[tuple[str, str]] = [
            ("search", search),
            ("sort", sort),
            ("direction", "-1"),
            ("limit", str(limit)),
            ("full", "true"),
        ]
        for tag in self._filter_tags(library=library, language=language, license=license):
            query_items.append(("filter", tag))
        if task:
            query_items.append(("pipeline_tag", task))
        if app:
            query_items.append(("apps", app))
        if provider:
            query_items.append(("inference_provider", provider))
        if other == "ungated":
            query_items.append(("gated", "false"))
        query = urllib.parse.urlencode(query_items)
        with urllib.request.urlopen(f"https://huggingface.co/api/models?{query}", timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _filter_tags(self, *, library: str, language: str, license: str) -> list[str]:
        tags = [value for value in [library, language, license] if value]
        return tags or ["gguf"]

    def _is_candidate_file(self, filename: str) -> bool:
        lowered = filename.lower()
        if not lowered.endswith(".gguf"):
            return False
        quant = self._extract_quant(filename)
        if not quant:
            return False
        return any(key in quant.upper() for key in ["Q4", "Q5", "IQ3", "Q3"])

    def _rank(self, file: HfModelFile) -> ModelRecommendation:
        return self.ranker.rank(file)

    def _with_thinking_metadata(self, rec: ModelRecommendation, file: HfModelFile | None) -> ModelRecommendation:
        if file is None:
            metadata = self._thinking_metadata(rec.repo_id, rec.pull_name, (), "")
        else:
            metadata = self._thinking_metadata(file.repo_id, file.filename, file.tags, file.card_text)
        return replace(rec, **metadata)

    def _with_readme_thinking_metadata(self, rec: ModelRecommendation) -> ModelRecommendation:
        if rec.thinking_support == "yes":
            return rec
        metadata = self._thinking_metadata(rec.repo_id, rec.pull_name, (), self._fetch_readme(rec.repo_id))
        if metadata["thinking_support"] == "unknown":
            return rec
        return replace(rec, **metadata)

    def _thinking_metadata(
        self,
        repo_id: str,
        filename: str,
        tags: tuple[str, ...],
        card_text: str,
    ) -> dict[str, str]:
        text = " ".join([repo_id, filename, " ".join(tags), card_text[:60_000]]).lower()
        negative = ["non-thinking", "non thinking", "no thinking", "without thinking"]
        if any(marker in text for marker in negative):
            return {"thinking_support": "unknown", "thinking_evidence": "negative mention in HF metadata"}
        explicit = ["<think>", "thinking mode", "thinking budget", "reasoning model", "reasoning-mode", "chain-of-thought"]
        named = ["deepseek-r1", "qwq", "qwen3", "reasoner", "reasoning", "thinking"]
        for marker in explicit + named:
            if marker in text:
                return {"thinking_support": "yes", "thinking_evidence": f"HF metadata: {marker}"}
        size_b = self.ranker.extract_param_size(repo_id + "/" + filename)
        if size_b >= 27:
            return {"thinking_support": "likely", "thinking_evidence": "27B+ model; verify by benchmark"}
        return {"thinking_support": "unknown", "thinking_evidence": ""}

    def _fetch_readme(self, repo_id: str) -> str:
        url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                return response.read(80_000).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _repo_from_model_name(self, model_name: str) -> str:
        text = model_name.strip()
        if text.startswith("hf.co/"):
            return text.removeprefix("hf.co/").split(":", 1)[0]
        if "/" in text and ":" in text:
            return text.split(":", 1)[0]
        return ""

    def _search_query_from_model_name(self, model_name: str) -> str:
        text = model_name.strip().replace("hf.co/", "")
        text = text.split(":", 1)[0].rsplit("/", 1)[-1]
        text = re.sub(r"[-_]+", " ", text)
        parts = [part for part in text.split() if part.lower() not in {"gguf", "latest"}]
        return " ".join(parts[:4])

    def _extract_quant(self, text: str) -> str:
        base = text.rsplit("/", 1)[-1].replace(".gguf", "")
        patterns = [
            r"(UD-Q\d_[A-Z_]+)",
            r"(Q\d_[A-Z_]+)",
            r"(IQ\d_[A-Za-z0-9_]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, base, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""
