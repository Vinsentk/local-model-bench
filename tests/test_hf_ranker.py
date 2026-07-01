from local_model_bench.hf_search import HfModelFile, HuggingFaceSearch
from local_model_bench.models import HardwareInfo


def test_ranker_prefers_9b_q4() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    rec = search._rank(HfModelFile("org/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", downloads=100000))
    assert rec.fit == "Ready"
    assert rec.quant == "Q4_K_M"
    assert rec.score > 70
    assert "coding_assistant" in rec.recommended_uses or "fast_draft" in rec.recommended_uses


def test_ranker_uses_latest_for_non_ollama_quant_tag() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    rec = search._rank(HfModelFile("org/Model-9B-GGUF", "Model-9B-Q4_XS.gguf", downloads=1000))

    assert rec.quant == "Q4_XS"
    assert rec.pull_name == "hf.co/org/Model-9B-GGUF:latest"


def test_ranker_uses_latest_for_q4_nl_tag() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    rec = search._rank(HfModelFile("org/Model-9B-GGUF", "Model-9B-Q4_NL.gguf", downloads=1000))

    assert rec.quant == "Q4_NL"
    assert rec.pull_name == "hf.co/org/Model-9B-GGUF:latest"


def test_ranker_rejects_70b() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    rec = search._rank(HfModelFile("org/Llama-70B-GGUF", "Llama-70B-Q4_K_M.gguf"))
    assert rec.fit == "Not recommended"
    assert "too large" in rec.reason


def test_ranker_marks_27b_as_conditional_for_ram_offload() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    rec = search._rank(HfModelFile("org/Coder-27B-GGUF", "Coder-27B-Q4_K_M.gguf"))

    assert rec.fit == "Conditional"
    assert "large model" in rec.reason


def test_ranker_rejects_when_storage_is_too_low() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 2)
    search = HuggingFaceSearch(hardware)
    rec = search._rank(HfModelFile("org/Coder-14B-GGUF", "Coder-14B-Q4_K_M.gguf"))

    assert rec.fit == "Not recommended"
    assert "free storage" in rec.reason


def test_hf_metadata_detects_thinking_support() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    metadata = search._thinking_metadata(
        "org/Some-GGUF",
        "Some-Q4_K_M.gguf",
        ("text-generation",),
        "This model supports thinking mode with <think> tags.",
    )
    assert metadata["thinking_support"] == "yes"
    assert "thinking mode" in metadata["thinking_evidence"] or "<think>" in metadata["thinking_evidence"]


def test_hf_metadata_marks_27b_as_likely() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    metadata = search._thinking_metadata("org/Model-27B-GGUF", "Model-27B-Q4_K_M.gguf", (), "")
    assert metadata["thinking_support"] == "likely"


def test_recommendations_filter_by_parameter_range() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    search._search_files = lambda **_filters: [  # type: ignore[method-assign]
        HfModelFile("org/Small-9B-GGUF", "Small-9B-Q4_K_M.gguf", downloads=1000, trending_page=1),
        HfModelFile("org/Mid-27B-GGUF", "Mid-27B-Q4_K_M.gguf", downloads=1000, trending_page=1),
    ]
    search._fetch_readme = lambda _repo_id: ""  # type: ignore[method-assign]

    recs = search.recommendations(min_params_b=24, max_params_b=32)

    assert [rec.repo_id for rec in recs] == ["org/Mid-27B-GGUF"]


def test_recommendations_filter_by_keyword_after_other_filters() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    search._search_files = lambda **_filters: [  # type: ignore[method-assign]
        HfModelFile("empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF", "Qwythos-9B-Q4_K_M.gguf", downloads=1000, trending_page=1),
        HfModelFile("org/Other-9B-GGUF", "Other-9B-Q4_K_M.gguf", downloads=1000, trending_page=1),
    ]
    search._fetch_readme = lambda _repo_id: ""  # type: ignore[method-assign]

    recs = search.recommendations(min_params_b=6, max_params_b=12, keyword="Mythos")

    assert [rec.repo_id for rec in recs] == ["empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF"]


def test_recommendations_filter_by_trending_page_range() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    search._search_files = lambda **_filters: [  # type: ignore[method-assign]
        HfModelFile("org/Page1-9B-GGUF", "Page1-9B-Q4_K_M.gguf", downloads=1000, trending_page=1),
        HfModelFile("org/Page2-9B-GGUF", "Page2-9B-Q4_K_M.gguf", downloads=1000, trending_page=2),
        HfModelFile("org/Page3-9B-GGUF", "Page3-9B-Q4_K_M.gguf", downloads=1000, trending_page=3),
        HfModelFile("org/Page4-9B-GGUF", "Page4-9B-Q4_K_M.gguf", downloads=1000, trending_page=4),
    ]
    search._fetch_readme = lambda _repo_id: ""  # type: ignore[method-assign]

    recs = search.recommendations(page_start=2, page_end=3)

    assert {rec.repo_id for rec in recs} == {"org/Page2-9B-GGUF", "org/Page3-9B-GGUF"}


def test_recommendations_pass_hf_filter_values() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    seen = {}

    def fake_search_files(**filters):
        seen.update(filters)
        return [HfModelFile("org/Small-9B-GGUF", "Small-9B-Q4_K_M.gguf", downloads=1000, trending_page=1)]

    search._search_files = fake_search_files  # type: ignore[method-assign]
    search._fetch_readme = lambda _repo_id: ""  # type: ignore[method-assign]

    search.recommendations(
        task="text-generation",
        library="gguf",
        app="ollama",
        language="ko",
        license="apache-2.0",
        provider="groq",
        other="ungated",
    )

    assert seen == {
        "task": "text-generation",
        "library": "gguf",
        "app": "ollama",
        "language": "ko",
        "license": "apache-2.0",
        "provider": "groq",
        "other": "ungated",
        "keyword": "",
        "search_limit": 90,
    }


def test_recommendations_page_end_controls_search_limit() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    seen = {}

    def fake_search_files(**filters):
        seen.update(filters)
        return [HfModelFile("org/Page5-9B-GGUF", "Page5-9B-Q4_K_M.gguf", downloads=1000, trending_page=5)]

    search._search_files = fake_search_files  # type: ignore[method-assign]
    search._fetch_readme = lambda _repo_id: ""  # type: ignore[method-assign]

    recs = search.recommendations(page_start=4, page_end=5)

    assert seen["search_limit"] == 150
    assert [rec.repo_id for rec in recs] == ["org/Page5-9B-GGUF"]


def test_recommendations_keep_page_one_before_later_pages() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)

    def fake_fetch_page(page, **_filters):
        return {
            1: ["org/Page1-9B-GGUF"],
            2: ["org/Page2-9B-GGUF"],
        }.get(page, [])

    def fake_exact_repo(repo_id):
        downloads = 1 if "Page1" in repo_id else 100_000
        return [HfModelFile(repo_id, f"{repo_id.rsplit('/', 1)[-1]}-Q4_K_M.gguf", downloads=downloads)]

    search._fetch_model_page_repo_ids = fake_fetch_page  # type: ignore[method-assign]
    search._search_exact_repo = fake_exact_repo  # type: ignore[method-assign]
    search._fetch_readme = lambda _repo_id: ""  # type: ignore[method-assign]

    recs = search.recommendations(page_start=1, page_end=2)

    assert [rec.trending_page for rec in recs] == [1, 2]
    assert [rec.repo_id for rec in recs] == ["org/Page1-9B-GGUF", "org/Page2-9B-GGUF"]


def test_parse_model_page_repo_ids_uses_huggingface_model_links_only() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)
    html = """
    <a href="/inference/models">Inference</a>
    <a href="/org/Model-A-GGUF">A</a>
    <a href="/org/Model-A-GGUF?x=1">A duplicate</a>
    <a href="/datasets/org/data">Data</a>
    <a href="/another/Model-B">B</a>
    """

    assert search._parse_model_page_repo_ids(html) == ["org/Model-A-GGUF", "another/Model-B"]


def test_hf_filter_tags_default_to_gguf() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "RTX", 16.0, "E:\\", 1000)
    search = HuggingFaceSearch(hardware)

    assert search._filter_tags(library="", language="", license="") == ["gguf"]
    assert search._filter_tags(library="gguf", language="ko", license="apache-2.0") == [
        "gguf",
        "ko",
        "apache-2.0",
    ]
