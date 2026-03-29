#!/usr/bin/env python3
"""Report Agent Pipeline — publication-quality scientific report generation.

Receives the manifest produced by the captain pipeline and orchestrates a
multi-agent GroupChat to produce a rich, scientifically interpreted report
with web-sourced literature context, biologics domain knowledge, and
enhanced visualisations.

Architecture:
    DeepResearchAgent  →  research context (literature, regulatory)
    ChromaDB RAG       →  biologics knowledge base (local sentence-transformers embeddings)
    Visualisation Agent →  publication-quality figures
    Interpretation Agent →  scientific narrative synthesis
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from tools import parse_json_tolerant

logger = logging.getLogger("report_pipeline")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_RESEARCH_TURNS = 3
DEFAULT_MAX_GROUPCHAT_ROUNDS = 25
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
REPORT_FIGURES_SUBDIR = "report_figures"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file, returning empty dict on failure."""
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except Exception as exc:
        logger.warning("Failed to load JSON %s: %s", path, exc)
        return {}


def _collect_plots(artifact_list: List[str]) -> List[str]:
    """Filter artifact paths to only PNG/SVG image files."""
    exts = {".png", ".svg", ".jpg", ".jpeg"}
    return [p for p in artifact_list if Path(p).suffix.lower() in exts]


def _build_research_message(
    domain_hints: Dict[str, Any],
    findings: List[str],
    context_text: str,
) -> str:
    """Build a research task message for the DeepResearchAgent."""
    domains = []
    if domain_hints.get("chromatography"):
        domains.append("chromatography (SEC, IEX, HIC, HPLC)")
    if domain_hints.get("mass_spectrometry"):
        domains.append("mass spectrometry (intact mass, LC-MS)")
    if not domains:
        domains.append("general biologics characterisation")

    domain_str = " and ".join(domains)
    findings_str = "\n".join(f"- {f}" for f in findings[:10]) if findings else "No specific findings yet."

    return (
        f"Research the following biologics analytical topic for a scientific report.\n\n"
        f"**Domain**: {domain_str}\n\n"
        f"**Key findings from data analysis**:\n{findings_str}\n\n"
        f"**Additional context**:\n{context_text[:2000] if context_text else 'None provided.'}\n\n"
        f"Please provide:\n"
        f"1. Relevant scientific background for interpreting these results\n"
        f"2. Expected reference ranges or typical values for the metrics mentioned\n"
        f"3. Regulatory context (ICH guidelines, pharmacopoeia) if applicable\n"
        f"4. Recent literature on best practices for this type of analysis\n\n"
        f"Return your findings as structured JSON with keys: "
        f"'background', 'reference_ranges', 'regulatory_context', 'literature_notes', 'citations'"
    )


def _extract_manifest_data(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse manifest items into a structured list for the report agents."""
    items = manifest.get("items", [])
    parsed = []
    for item in items:
        entry: Dict[str, Any] = {
            "file": item.get("file", "unknown"),
            "file_name": Path(item.get("file", "unknown")).stem,
        }

        # Cleaning data
        cleaning = item.get("cleaning", {})
        entry["cleaning_summary"] = cleaning.get("summary", {})
        entry["cleaned_path"] = cleaning.get("cleaned_path", "")

        # Analysis data
        analysis = item.get("analysis", {})
        entry["analysis_summary_path"] = analysis.get("analysis_summary_path", "")
        entry["analysis_artifacts"] = analysis.get("artifacts", [])
        entry["domain_hints"] = analysis.get("domain_hints", {})
        entry["plots"] = _collect_plots(analysis.get("artifacts", []))

        # Load analysis summary if path exists
        if entry["analysis_summary_path"] and Path(entry["analysis_summary_path"]).exists():
            entry["analysis_summary"] = _load_json(entry["analysis_summary_path"])
        else:
            entry["analysis_summary"] = {}

        # Fallback: if analysis_summary is empty, scan artifacts for JSON
        if not entry["analysis_summary"] and entry["analysis_artifacts"]:
            for artifact_path in entry["analysis_artifacts"]:
                if artifact_path.endswith(".json") and Path(artifact_path).exists():
                    loaded = _load_json(artifact_path)
                    if loaded and ("findings" in loaded or "per_group" in loaded):
                        entry["analysis_summary"] = loaded
                        logger.info(
                            "Loaded analysis summary from artifact: %s", artifact_path
                        )
                        break
            # If still empty but plots exist, create minimal summary
            if not entry["analysis_summary"] and entry["plots"]:
                entry["analysis_summary"] = {
                    "findings": [
                        f"Analysis produced {len(entry['plots'])} plots"
                    ],
                    "plots": entry["plots"],
                }

        # Cross-validation
        entry["cross_validation"] = item.get("cross_validation", {})

        # Existing report (if any)
        entry["existing_report_path"] = item.get("report_path", "")

        parsed.append(entry)

    return parsed


# ──────────────────────────────────────────────────────────────────────
# Research Agent setup
# ──────────────────────────────────────────────────────────────────────


def _create_research_agent(llm_config: Dict[str, Any]) -> Any:
    """Create a DeepResearchAgent for literature/regulatory research.

    Falls back to WebSurferAgent (crawl4ai) if DeepResearchAgent is
    unavailable, and finally to None if neither works.
    """
    try:
        from autogen.agents.experimental import DeepResearchAgent
        from autogen import LLMConfig

        # Convert dict config to LLMConfig object (AG2 >= 0.11: positional args)
        config_list = llm_config.get("config_list", [{}])
        agent_llm_config = LLMConfig(
            *config_list,
            temperature=llm_config.get("temperature", 0.3),
        )

        agent = DeepResearchAgent(
            name="ResearchAgent",
            llm_config=agent_llm_config,
        )
        logger.info("Created DeepResearchAgent for web research")
        return agent

    except ImportError:
        logger.warning("DeepResearchAgent unavailable, trying WebSurferAgent")

    try:
        from autogen.agents.experimental import WebSurferAgent
        from autogen import LLMConfig

        config_list = llm_config.get("config_list", [{}])
        agent_llm_config = LLMConfig(
            *config_list,
            temperature=llm_config.get("temperature", 0.3),
        )

        agent = WebSurferAgent(
            name="ResearchAgent",
            web_tool="crawl4ai",
            llm_config=agent_llm_config,
        )
        logger.info("Created WebSurferAgent (crawl4ai) for web research")
        return agent

    except ImportError:
        logger.warning("No web research agent available — reports will lack literature context")
        return None


# ──────────────────────────────────────────────────────────────────────
# Patch: limit DeepResearchTool decomposition chat turns.
#
# AG2's DeepResearchTool creates DecompositionAgent ↔ DecompositionCritic
# with NO max_turns and a termination condition that requires the exact
# prefix "Subquestions answered:".  Qwen3.5 never produces this prefix,
# causing an infinite loop (450+ exchanges observed in run 859801).
#
# We monkey-patch the static method that creates the decomposition chat
# to enforce max_turns=10, which is more than enough for subquestion
# negotiation.
# ──────────────────────────────────────────────────────────────────────
try:
    from autogen.tools.experimental.deep_research.deep_research import DeepResearchTool as _DRT
    import functools as _ft

    _orig_get_split = _DRT._get_split_question_and_answer_subquestions

    @staticmethod
    def _patched_get_split(llm_config, max_web_steps):
        inner_fn = _orig_get_split(llm_config, max_web_steps)

        @_ft.wraps(inner_fn)
        def wrapper(question, llm_config=llm_config, max_web_steps=max_web_steps):
            from autogen.agentchat import ConversableAgent as _CA
            _real_initiate = _CA.initiate_chat

            def _limited_initiate(self, recipient, *args, **kwargs):
                if self.name == "DecompositionCritic":
                    kwargs.setdefault("max_turns", 10)
                return _real_initiate(self, recipient, *args, **kwargs)

            _CA.initiate_chat = _limited_initiate
            try:
                return inner_fn(question, llm_config=llm_config,
                                max_web_steps=max_web_steps)
            finally:
                _CA.initiate_chat = _real_initiate

        return wrapper

    _DRT._get_split_question_and_answer_subquestions = _patched_get_split
    logger.info("Patched DeepResearchTool decomposition to enforce max_turns=10")
except Exception:
    pass


def _run_research(
    agent: Any,
    message: str,
    max_turns: int = DEFAULT_MAX_RESEARCH_TURNS,
) -> Dict[str, Any]:
    """Execute a research task and return structured results."""
    if agent is None:
        return {"error": "no_research_agent", "background": "", "citations": []}

    try:
        result = agent.run(
            message=message,
            tools=agent.tools,
            max_turns=max_turns,
            user_input=False,
            summary_method="reflection_with_llm",
        )
        result.process()

        # Extract the summary text from the run result
        summary_text = ""
        if hasattr(result, "summary"):
            summary_text = result.summary
        elif hasattr(result, "chat_history") and result.chat_history:
            summary_text = result.chat_history[-1].get("content", "")

        # Try to parse as JSON
        try:
            parsed = json.loads(summary_text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        return {
            "background": summary_text,
            "reference_ranges": "",
            "regulatory_context": "",
            "literature_notes": "",
            "citations": [],
        }

    except Exception as exc:
        logger.warning("Research agent failed (non-fatal): %s", exc)
        return {"error": str(exc), "background": "", "citations": []}


# ──────────────────────────────────────────────────────────────────────
# Knowledge base (local ChromaDB RAG — no OpenAI dependency)
# ──────────────────────────────────────────────────────────────────────


class _KnowledgeBase:
    """Local RAG over the biologics knowledge base.

    Uses chromadb (in-memory) + sentence-transformers all-MiniLM-L6-v2
    for embeddings. Built at startup from all *.md files in knowledge_base/.
    No OpenAI or external API calls required — fully compatible with
    local vLLM deployments.
    """

    def __init__(self, kb_dir: Path) -> None:
        import chromadb
        from chromadb.utils import embedding_functions as _ef

        ef = _ef.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.Client()
        self._col = client.get_or_create_collection(
            "biologics_knowledge", embedding_function=ef
        )
        self._index(kb_dir)

    def _index(self, kb_dir: Path) -> None:
        docs, ids, metas = [], [], []
        for f in sorted(kb_dir.glob("*.md")):
            try:
                text = f.read_text("utf-8").strip()
            except Exception as exc:
                logger.warning("Could not read KB file %s: %s", f, exc)
                continue
            for j, chunk in enumerate(self._chunk(text)):
                docs.append(chunk)
                ids.append(f"{f.stem}__{j}")
                metas.append({"source": f.name})
        if docs:
            self._col.add(documents=docs, ids=ids, metadatas=metas)
        logger.info("Knowledge base indexed: %d chunks from %s", len(docs), kb_dir)

    @staticmethod
    def _chunk(text: str, size: int = 600, overlap: int = 80) -> list:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start : start + size])
            start += size - overlap
        return chunks

    def query(self, text: str, n_results: int = 5) -> list:
        result = self._col.query(query_texts=[text], n_results=n_results)
        return result["documents"][0] if result["documents"] else []


def _build_knowledge_base() -> Optional[_KnowledgeBase]:
    """Build a local ChromaDB knowledge base from markdown files.

    Replaces DocAgent (which requires OpenAI embeddings incompatible with
    local vLLM). Uses sentence-transformers all-MiniLM-L6-v2 locally.
    Returns None if the knowledge base directory is missing or empty.
    """
    if not KNOWLEDGE_BASE_DIR.exists() or not any(KNOWLEDGE_BASE_DIR.glob("*.md")):
        logger.info("Knowledge base %s missing or empty — skipping.", KNOWLEDGE_BASE_DIR)
        return None
    try:
        return _KnowledgeBase(KNOWLEDGE_BASE_DIR)
    except Exception as exc:
        logger.warning("Knowledge base build failed: %s", exc)
        return None


def _query_knowledge(
    kb: Optional[_KnowledgeBase],
    query: str,
    llm_config: Dict[str, Any],
) -> str:
    """Query the local knowledge base and synthesise an answer via vLLM.

    Retrieves top-5 relevant chunks via ChromaDB similarity search, then
    calls vLLM directly (same pattern as _run_validation_direct) to
    synthesise a concise answer grounded in the retrieved context.
    Returns empty string if knowledge base is empty or any step fails.
    """
    if kb is None:
        return ""
    chunks = kb.query(query, n_results=5)
    if not chunks:
        return ""

    context = "\n\n---\n\n".join(chunks)
    system = (
        "You are a biologics analytical science expert. "
        "Answer the question using ONLY the provided reference material. "
        "Be concise and cite specific values or ranges where available."
    )
    prompt = f"## Reference Material\n\n{context}\n\n## Question\n\n{query}"

    try:
        import copy
        import autogen

        cfg = copy.deepcopy(llm_config)
        for entry in cfg.get("config_list", []):
            entry["timeout"] = 120
        client = autogen.OpenAIWrapper(config_list=cfg["config_list"])
        response = client.create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return autogen.OpenAIWrapper.extract_text_or_completion_object(response)[0]
    except Exception as exc:
        logger.warning("Knowledge base query failed: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────
# Visualisation & Interpretation Agents (GroupChat-based)
# ──────────────────────────────────────────────────────────────────────


def _create_groupchat_agents(
    llm_config: Dict[str, Any],
    work_dir: Path,
    figure_dir: Optional[Path] = None,
) -> tuple:
    """Create the Visualisation and Interpretation agents for the GroupChat.

    Returns (vis_agent, interp_agent, user_proxy, groupchat, manager).

    If *figure_dir* is provided, a figure-manifest message is injected into
    the GroupChat after the VisualisationAgent's code has been executed so
    that the InterpretationAgent knows exactly which figures exist on disk
    and can reference them by filename.
    """
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

    from prompts import VISUALISATION_AGENT_PROMPT, INTERPRETATION_AGENT_PROMPT

    # User proxy for code execution
    user_proxy = UserProxyAgent(
        name="Computer_terminal",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": str(work_dir),
            "use_docker": False,
            "last_n_messages": 3,
            "timeout": 120,
        },
        max_consecutive_auto_reply=5,
        default_auto_reply="",
        is_termination_msg=lambda msg: "REPORT_COMPLETE" in msg.get("content", ""),
    )

    vis_agent = AssistantAgent(
        name="VisualisationAgent",
        system_message=VISUALISATION_AGENT_PROMPT,
        llm_config=llm_config,
    )

    interp_agent = AssistantAgent(
        name="InterpretationAgent",
        system_message=INTERPRETATION_AGENT_PROMPT,
        llm_config=llm_config,
    )

    # Custom speaker selection: vis_agent → interp_agent → done
    # Deterministic turn limit prevents vis→terminal loop from starving
    # the InterpretationAgent (see pipeline stabilisation plan).
    _MAX_VIS_TURNS = 5
    _figure_manifest_injected = False

    # Directories to scan for generated figures
    _scan_dirs = [d for d in [figure_dir, work_dir] if d is not None]

    def _speaker_selection(last_speaker, groupchat):
        """Route messages through the defined workflow."""
        nonlocal _figure_manifest_injected

        vis_turns = sum(
            1 for m in groupchat.messages if m.get("name") == vis_agent.name
        )

        if last_speaker == user_proxy:
            # ── Figure manifest injection ──
            # After code execution by VisualisationAgent, scan for generated
            # figures and inject a manifest so InterpretationAgent knows
            # which files are available and can reference them by filename.
            if not _figure_manifest_injected:
                vis_spoke = any(
                    m.get("name") == vis_agent.name for m in groupchat.messages
                )
                if vis_spoke:
                    generated: List[str] = []
                    for scan_dir in _scan_dirs:
                        if scan_dir and scan_dir.exists():
                            generated.extend(
                                p.name for p in sorted(scan_dir.rglob("*.png"))
                                if p.stat().st_size >= 5000
                            )
                    # Deduplicate preserving order
                    seen_names: set = set()
                    unique_names: List[str] = []
                    for n in generated:
                        if n not in seen_names:
                            seen_names.add(n)
                            unique_names.append(n)
                    if unique_names:
                        manifest_lines = [
                            "## Generated Figures\n",
                            "The following figures have been created. Use these "
                            "EXACT filenames when referencing figures in the report. "
                            "The figure number matches the numeric prefix in the "
                            "filename (e.g. fig1_* = Figure 1, 01_* = Figure 1).\n",
                        ]
                        for fn in unique_names:
                            manifest_lines.append(f"- `{fn}`")
                        groupchat.messages.append({
                            "role": "assistant",
                            "name": "FigureManifest",
                            "content": "\n".join(manifest_lines),
                        })
                        _figure_manifest_injected = True
                        logger.debug(
                            "Per-file figure manifest injected: %d figures",
                            len(unique_names),
                        )

            # After code execution: if vis has had enough turns, hand off
            if vis_turns >= _MAX_VIS_TURNS:
                logger.info(
                    "VisualisationAgent reached %d turns — handing off to "
                    "InterpretationAgent",
                    vis_turns,
                )
                return interp_agent
            # Otherwise return to the agent that requested execution
            for msg in reversed(groupchat.messages):
                if msg.get("name") == vis_agent.name:
                    return vis_agent
                if msg.get("name") == interp_agent.name:
                    return interp_agent
            return vis_agent

        if last_speaker == vis_agent:
            last_msg = groupchat.messages[-1].get("content", "")
            if "```python" in last_msg and vis_turns < _MAX_VIS_TURNS:
                return user_proxy
            # Vis done (no code or max turns reached), move to interpretation
            return interp_agent

        if last_speaker == interp_agent:
            last_msg = groupchat.messages[-1].get("content", "")
            if "```python" in last_msg:
                return user_proxy
            # Check for termination
            if "REPORT_COMPLETE" in last_msg:
                return None
            return interp_agent

        return vis_agent

    groupchat = GroupChat(
        agents=[user_proxy, vis_agent, interp_agent],
        messages=[],
        max_round=DEFAULT_MAX_GROUPCHAT_ROUNDS,
        speaker_selection_method=_speaker_selection,
        allow_repeat_speaker=True,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    return vis_agent, interp_agent, user_proxy, groupchat, manager


# ──────────────────────────────────────────────────────────────────────
# Report Assembly
# ──────────────────────────────────────────────────────────────────────


def _extract_report_markdown(messages: List[Dict[str, Any]]) -> str:
    """Extract the final report markdown from GroupChat messages."""
    for msg in reversed(messages):
        content = msg.get("content", "")
        if not content:
            continue

        # Try to extract JSON with report_markdown key (WP7: use robust parser)
        if "report_markdown" in content:
            parsed = parse_json_tolerant(content)
            if isinstance(parsed, dict) and parsed.get("report_markdown"):
                return parsed["report_markdown"]

            # Try fenced JSON blocks as fallback
            fenced = re.findall(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            for block in fenced:
                try:
                    parsed_block = json.loads(block.strip())
                    if isinstance(parsed_block, dict) and parsed_block.get("report_markdown"):
                        return parsed_block["report_markdown"]
                except json.JSONDecodeError:
                    continue

        # Look for markdown content directly (starts with # heading)
        if content.strip().startswith("#") and len(content) > 200:
            cleaned = content.replace("REPORT_COMPLETE", "").strip()
            if cleaned:
                return cleaned

    return ""


def _build_figure_number_map(
    all_figures: List[str],
) -> Dict[int, str]:
    """Build a figure-number → path mapping using filename prefixes first,
    falling back to ordinal position only for unmatched figures.

    Recognises these naming conventions (case-insensitive):
      fig1_description.png   →  Figure 1
      fig_1_description.png  →  Figure 1
      figure1_description.png→  Figure 1
      01_description.png     →  Figure 1
      001_description.png    →  Figure 1

    Any figures whose filenames do NOT contain a recognisable numeric prefix
    are assigned to the next available figure number by ordinal position.
    """
    fig_map: Dict[int, str] = {}
    unmatched: List[str] = []

    _prefix_re = re.compile(
        r'^(?:fig(?:ure)?[_\-]?)?(\d{1,3})[_\-]',
        re.IGNORECASE,
    )

    for fig_path in all_figures:
        stem = Path(fig_path).stem
        m = _prefix_re.match(stem)
        if m:
            num = int(m.group(1))
            if num not in fig_map:
                fig_map[num] = fig_path
            else:
                # Number already claimed — treat as unmatched
                unmatched.append(fig_path)
        else:
            unmatched.append(fig_path)

    # Assign unmatched figures to the next free ordinal slots
    next_num = 1
    for fig_path in unmatched:
        while next_num in fig_map:
            next_num += 1
        fig_map[next_num] = fig_path
        next_num += 1

    return fig_map


def _embed_inline_figures(
    report_md: str,
    all_figures: List[str],
) -> tuple:
    """Match inline 'Figure N' references to actual figure files and embed them.

    Returns (updated_report_md, set_of_embedded_figure_paths).

    Matching strategy (in priority order):
      1. Explicit filename reference on the same line (e.g. ``fig1_uv.png``)
      2. Filename numeric-prefix match  (fig1_* / 01_* → Figure 1)
      3. Ordinal-position fallback (Nth figure in list)

    Figures are inserted as ![...](path) immediately after the paragraph that
    references them.  Unmatched figures are NOT embedded here (caller handles).
    """
    if not all_figures:
        return report_md, set()

    # ── Build smart mapping: figure number → path ──
    fig_map = _build_figure_number_map(all_figures)

    # Also build a quick lookup by filename (lowercased) for explicit refs
    path_by_name: Dict[str, str] = {
        Path(fp).name.lower(): fp for fp in all_figures
    }

    # ── Walk lines and embed after paragraphs ──
    embedded: set = set()
    lines = report_md.split("\n")
    result_lines: List[str] = []
    fig_pattern = re.compile(r'(?:Figure|Fig\.?)\s+(\d+)', re.IGNORECASE)
    filename_pattern = re.compile(r'[`"\']?([\w\-]+\.png)[`"\']?', re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i]
        result_lines.append(line)

        # Check if this line references a figure
        matches = fig_pattern.findall(line)
        if matches:
            is_para_end = (
                i + 1 >= len(lines)
                or lines[i + 1].strip() == ""
                or lines[i + 1].startswith("#")
            )
            if is_para_end:
                for fig_num_str in matches:
                    fig_num = int(fig_num_str)

                    # Strategy 1: explicit filename on the same line
                    fig_path = None
                    fname_match = filename_pattern.search(line)
                    if fname_match:
                        candidate = fname_match.group(1).lower()
                        if candidate in path_by_name:
                            fig_path = path_by_name[candidate]

                    # Strategy 2: filename-prefix mapping
                    if fig_path is None and fig_num in fig_map:
                        fig_path = fig_map[fig_num]

                    if fig_path and fig_path not in embedded:
                        fig_name = Path(fig_path).stem.replace("_", " ").title()
                        encoded_path = quote(fig_path, safe="/:")
                        result_lines.append("")
                        result_lines.append(
                            f"![Figure {fig_num}: {fig_name}]({encoded_path})"
                        )
                        embedded.add(fig_path)
        i += 1

    return "\n".join(result_lines), embedded


def _select_figures(
    all_figures: List[str],
    figure_selection: str = "all",
    max_figures: int = 10,
    png_min_bytes: int = 5000,
) -> List[str]:
    """Apply figure selection strategy (WP6) and return filtered list."""
    if figure_selection == "ranked":
        all_figures = [
            f for f in all_figures
            if Path(f).exists() and Path(f).stat().st_size >= png_min_bytes
        ]
        seen_stems: set = set()
        ranked: List[str] = []
        for fig in sorted(
            all_figures,
            key=lambda p: Path(p).stat().st_size if Path(p).exists() else 0,
            reverse=True,
        ):
            stem = Path(fig).stem
            if stem not in seen_stems:
                seen_stems.add(stem)
                ranked.append(fig)
        return ranked[:max_figures]
    elif figure_selection == "top_n":
        return all_figures[:max_figures]
    return all_figures


def _classify_figure_theme(stem: str) -> str:
    """Classify a figure into a thematic group based on its filename stem."""
    sl = stem.lower()
    _THEME_KEYWORDS = {
        "Chromatography & Elution": [
            "elution", "chromatogram", "overlay", "gradient", "uv_280",
            "uv280", "mau", "peak", "column", "profile",
        ],
        "Statistical Comparisons": [
            "boxplot", "violin", "box", "comparison", "kruskal", "anova",
            "significance", "bar", "grouped",
        ],
        "Correlation & Heatmaps": [
            "heatmap", "correlation", "heat", "spearman", "pearson", "matrix",
        ],
        "Outlier & Anomaly Detection": [
            "outlier", "anomal", "deviation", "detection",
        ],
        "Mass Spectrometry": [
            "mass", "mz", "dalton", "kda", "charge", "deconvol", "spectrum",
            "tic", "response",
        ],
        "Trends & Distributions": [
            "trend", "kde", "density", "distribution", "line", "scatter",
            "hexbin",
        ],
        "Summary & Dashboard": [
            "dashboard", "summary", "overview",
        ],
    }
    for theme, keywords in _THEME_KEYWORDS.items():
        if any(kw in sl for kw in keywords):
            return theme
    return "Other"


def _build_themed_figure_section(figures: List[str]) -> str:
    """Build an 'Additional Figures' markdown section grouped by theme."""
    from collections import OrderedDict

    themed: Dict[str, List[str]] = OrderedDict()
    for fig_path in figures:
        stem = Path(fig_path).stem
        theme = _classify_figure_theme(stem)
        themed.setdefault(theme, []).append(fig_path)

    section = "\n\n## Additional Figures\n"
    fig_counter = 1
    for theme, paths in themed.items():
        section += f"\n### {theme}\n\n"
        for fig_path in paths:
            fig_name = Path(fig_path).stem.replace("_", " ").title()
            abs_fig_path = Path(fig_path).resolve()
            encoded_path = quote(str(abs_fig_path), safe="/:")
            section += f"**Figure {fig_counter}**: {fig_name}\n\n"
            section += f"![{fig_name}]({encoded_path})\n\n"
            fig_counter += 1

    return section


def _log_figure_alignment(report_md: str) -> None:
    """Log a summary of all figure references in the assembled report.

    For each ![Figure N: desc](path) found, logs the figure number,
    alt text, file path, and whether the file exists on disk.
    """
    ref_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    refs_found = ref_pattern.findall(report_md)
    if not refs_found:
        return
    for alt_text, path_str in refs_found:
        # Decode URL-encoded path for existence check
        from urllib.parse import unquote
        decoded = unquote(path_str)
        exists = Path(decoded).exists() if not decoded.startswith(("http", "data:")) else True
        status = "OK" if exists else "MISSING"
        logger.info(
            "Figure alignment: [%s] alt='%s' path='%s'",
            status, alt_text[:60], Path(decoded).name if not decoded.startswith("http") else decoded[:60],
        )
    logger.info("Figure alignment: %d total references in assembled report", len(refs_found))


def _assemble_report_with_figures(
    report_md: str,
    figure_dir: Path,
    original_plots: List[str],
    figure_selection: str = "all",
    max_figures: int = 10,
    png_min_bytes: int = 5000,
) -> str:
    """Embed figure references into the report markdown.

    First attempts to embed figures inline next to their "Figure N" references.
    Any remaining figures are appended in a Figures section at the end, grouped
    by analytical theme.

    Figure ordering: report-stage figures (from figure_dir) come FIRST so that
    their numeric prefixes (fig1_*, 01_*) align with the InterpretationAgent's
    "Figure N" numbering, followed by original analysis-stage figures.

    figure_selection controls strategy:
      - "all": include all figures (baseline behaviour)
      - "ranked": rank by file size, deduplicate by stem, cap at max_figures
      - "top_n": take first max_figures figures in order
    """
    _fig_count = (
        len(list(figure_dir.glob("*.png"))) if figure_dir.exists() else 0
    )
    logger.debug(
        "Figure assembly: figure_dir=%s exists=%s, figures_in_dir=%d, "
        "original_plots=%d",
        figure_dir, figure_dir.exists(), _fig_count, len(original_plots),
    )
    # Collect all figures — report-stage figures FIRST so that their
    # numeric filename prefixes (fig1_*, 01_*) align with the
    # InterpretationAgent's "Figure N" numbering.
    all_figures: List[str] = []
    if figure_dir.exists():
        for ext in ("*.png", "*.svg", "*.jpg"):
            all_figures.extend(str(p) for p in sorted(figure_dir.glob(ext)))
    all_figures.extend(original_plots)

    # Deduplicate while preserving order (report figures take precedence)
    seen: set = set()
    deduped: List[str] = []
    for f in all_figures:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    all_figures = deduped

    # Build a combined lookup by stem across ALL figures (report + analysis)
    all_by_stem: Dict[str, str] = {}
    for fp in all_figures:
        all_by_stem.setdefault(Path(fp).stem.lower(), fp)

    # If report already has image markdown references, absolutize paths,
    # validate existence, and attempt to fix broken refs against all figures.
    if "![" in report_md:
        # Build a lookup of actual figures in figure_dir by stem
        actual_by_stem: Dict[str, str] = {}
        if figure_dir.exists():
            for ext in ("*.png", "*.svg", "*.jpg"):
                for p in figure_dir.glob(ext):
                    actual_by_stem[p.stem.lower()] = str(p)

        referenced_paths: set = set()

        # Filename-prefix regex for matching figN_ / figure_N_ / 01_ patterns
        _prefix_re = re.compile(
            r'^(?:fig(?:ure)?[_\-]?)?(\d{1,3})[_\-]', re.IGNORECASE,
        )

        def _fix_ref(m):
            alt_text = m.group(1)
            path_str = m.group(2)
            if path_str.startswith(("http://", "https://", "data:")):
                referenced_paths.add(path_str)
                return m.group(0)
            p = Path(path_str)
            if not p.is_absolute():
                p = p.resolve()
            # If the resolved path exists, absolutize and keep
            if p.exists():
                referenced_paths.add(str(p))
                return f"![{alt_text}]({quote(str(p), safe='/:=')})"
            # Path doesn't exist — try to match stem against all known figures
            stem = p.stem.lower()
            if stem in all_by_stem:
                replacement = all_by_stem[stem]
                referenced_paths.add(replacement)
                return f"![{alt_text}]({quote(replacement, safe='/:=')})"
            if stem in actual_by_stem:
                replacement = actual_by_stem[stem]
                referenced_paths.add(replacement)
                return f"![{alt_text}]({quote(replacement, safe='/:=')})"
            # Try matching by figure number from alt text (e.g. "Figure 3")
            fig_match = re.search(r'(\d+)', alt_text)
            if fig_match:
                fig_num = int(fig_match.group(1))
                # Search ALL known figures for a matching numeric prefix
                for fstem, fpath in all_by_stem.items():
                    pm = _prefix_re.match(fstem)
                    if pm and int(pm.group(1)) == fig_num:
                        referenced_paths.add(fpath)
                        return f"![{alt_text}]({quote(fpath, safe='/:=')})"
            # No match found — leave the broken ref (will be logged)
            logger.warning("Figure ref not resolved: %s", path_str)
            return m.group(0)

        report_md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _fix_ref, report_md)

        # Append any figures from figure_dir that weren't referenced
        unreferenced = [
            fp for fp in all_figures if fp not in referenced_paths
        ]
        if figure_dir.exists():
            for ext in ("*.png", "*.svg", "*.jpg"):
                for p in figure_dir.glob(ext):
                    if str(p) not in referenced_paths and str(p) not in unreferenced:
                        unreferenced.append(str(p))

        unreferenced = _select_figures(
            unreferenced, figure_selection, max_figures, png_min_bytes
        )
        if unreferenced:
            report_md += _build_themed_figure_section(unreferenced)

        _log_figure_alignment(report_md)
        return report_md

    # Apply figure selection strategy (WP6)
    all_figures = _select_figures(all_figures, figure_selection, max_figures, png_min_bytes)

    if not all_figures:
        return report_md

    # Phase 1: Embed figures inline next to their "Figure N" references
    report_md, embedded = _embed_inline_figures(report_md, all_figures)

    # Phase 2: Append any remaining (unreferenced) figures at the end
    remaining = [f for f in all_figures if f not in embedded]
    if remaining:
        report_md += _build_themed_figure_section(remaining)

    _log_figure_alignment(report_md)

    return report_md


# ──────────────────────────────────────────────────────────────────────
# PDF Generation
# ──────────────────────────────────────────────────────────────────────


def _markdown_to_pdf(
    markdown_text: str,
    output_path: Path,
    title: str = "Analysis Report",
    plot_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Convert markdown report to PDF using WeasyPrint.

    Falls back to a basic approach if WeasyPrint is unavailable.
    """
    try:
        from pdf_writer import markdown_to_pdf

        markdown_to_pdf(
            markdown_text,
            output_path,
            title=title,
            plot_paths=plot_paths or [],
        )
        return str(output_path)
    except ImportError:
        logger.info("pdf_writer not available — skipping PDF generation")
        return None
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline Class
# ──────────────────────────────────────────────────────────────────────


class ReportPipeline:
    """Orchestrates multi-agent report generation from captain pipeline output.

    Usage:
        pipeline = ReportPipeline(llm_config=llm_config)
        result = pipeline.run(manifest)
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        output_dir: Optional[Path] = None,
        max_research_turns: int = DEFAULT_MAX_RESEARCH_TURNS,
        max_groupchat_rounds: int = DEFAULT_MAX_GROUPCHAT_ROUNDS,
        figure_selection: str = "all",
        max_report_figures: int = 10,
        run_config: Optional[Any] = None,
    ):
        self.llm_config = llm_config
        self.output_dir = output_dir
        self.max_research_turns = max_research_turns
        self.max_groupchat_rounds = max_groupchat_rounds
        self.figure_selection = figure_selection
        self.max_report_figures = max_report_figures
        self._run_config = run_config

        # BS-4: Payload budget defaults (overridden by run_config when provided)
        self._budget_global = 40000
        self._budget_per_file = 24000
        self._budget_report_excerpt = 10000
        if run_config is not None:
            self._budget_global = getattr(run_config, "payload_budget_global", self._budget_global)
            self._budget_per_file = getattr(run_config, "payload_budget_per_file", self._budget_per_file)
            self._budget_report_excerpt = getattr(run_config, "payload_budget_report_excerpt", self._budget_report_excerpt)

        # Agents are created lazily
        self._research_agent = None
        self._kb: Optional[_KnowledgeBase] = None
        self._research_agent_initialised = False
        self._kb_initialised = False

        # Circuit breaker: disable research agent after consecutive failures
        self._research_consecutive_failures = 0
        self._research_max_failures = 3  # disable after 3 consecutive failures
        self._research_disabled = False

    def _get_research_agent(self):
        if not self._research_agent_initialised:
            self._research_agent = _create_research_agent(self.llm_config)
            self._research_agent_initialised = True
        return self._research_agent

    def _get_knowledge_base(self) -> Optional[_KnowledgeBase]:
        if not self._kb_initialised:
            self._kb = _build_knowledge_base()
            self._kb_initialised = True
        return self._kb

    def run(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full report pipeline on a captain pipeline manifest.

        Returns a dict with report paths and metadata.
        """
        output_root = self.output_dir or Path(manifest.get("output_root", "outputs"))
        reports_dir = output_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        items = _extract_manifest_data(manifest)
        if not items:
            logger.warning("No items found in manifest — nothing to report on.")
            return {"error": "empty_manifest", "reports": []}

        results = []

        # ── Per-file reports ──
        for item in items:
            logger.info("Generating report for: %s", item["file_name"])
            try:
                report_result = self._generate_file_report(item, reports_dir)
                results.append(report_result)
            except Exception as exc:
                logger.error("Report generation failed for %s: %s", item["file_name"], exc)
                results.append({
                    "file": item["file"],
                    "error": str(exc),
                    "report_path": None,
                    "pdf_path": None,
                })

        # ── Global summary report (if multiple files with findings) ──
        global_report = None
        _items_with_findings = [
            item for item in items
            if item.get("analysis_summary", {}).get("findings")
        ]
        if len(_items_with_findings) > 1:
            try:
                global_report = self._generate_global_report(
                    _items_with_findings, results, reports_dir
                )
            except Exception as exc:
                logger.error("Global report generation failed: %s", exc)
        elif len(items) > 1:
            logger.warning(
                "Skipping global report: only %d of %d items have analysis findings",
                len(_items_with_findings), len(items),
            )

        # Clean up reports directory — keep only report files and figures
        self._cleanup_reports_dir(reports_dir)

        return {
            "reports": results,
            "global_report": global_report,
            "reports_dir": str(reports_dir),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_file_report(
        self,
        item: Dict[str, Any],
        reports_dir: Path,
    ) -> Dict[str, Any]:
        """Generate a full report for a single file."""
        file_name = item["file_name"]
        analysis_summary = item.get("analysis_summary", {})
        findings = analysis_summary.get("findings", [])
        domain_hints = item.get("domain_hints", {})

        # ── Phase 1: Research ──
        logger.info("[%s] Phase 1: Web research", file_name)
        if self._research_disabled:
            logger.info(
                "[%s] Research agent disabled (circuit breaker: %d consecutive failures)",
                file_name, self._research_consecutive_failures,
            )
            research_results = {"error": "circuit_breaker_open", "background": "", "citations": []}
        else:
            research_msg = _build_research_message(
                domain_hints=domain_hints,
                findings=findings,
                context_text=json.dumps(analysis_summary.get("per_run_per_stage", {}), default=str)[:1500],
            )
            research_results = _run_research(
                self._get_research_agent(),
                research_msg,
                max_turns=self.max_research_turns,
            )
            # Circuit breaker: track consecutive failures
            if research_results.get("error"):
                self._research_consecutive_failures += 1
                if self._research_consecutive_failures >= self._research_max_failures:
                    self._research_disabled = True
                    logger.warning(
                        "Research agent circuit breaker OPEN after %d consecutive failures. "
                        "Disabling web research for remaining files.",
                        self._research_consecutive_failures,
                    )
            else:
                self._research_consecutive_failures = 0  # reset on success

        # ── Phase 2: Knowledge base queries ──
        logger.info("[%s] Phase 2: Knowledge base queries", file_name)
        knowledge_context = self._query_knowledge_for_item(item)

        # ── Phase 3: Visualisation & Interpretation (GroupChat) ──
        logger.info("[%s] Phase 3: GroupChat (visualisation + interpretation)", file_name)
        exec_dir = reports_dir.parent / "exec_workdir"
        work_dir = exec_dir / f"report_{file_name}"
        work_dir.mkdir(parents=True, exist_ok=True)
        figure_dir = reports_dir / REPORT_FIGURES_SUBDIR / file_name
        figure_dir.mkdir(parents=True, exist_ok=True)

        groupchat_payload = self._build_groupchat_payload(
            item=item,
            research_results=research_results,
            knowledge_context=knowledge_context,
            figure_dir=figure_dir,
        )

        report_md = self._run_groupchat(groupchat_payload, work_dir, figure_dir=figure_dir)

        # ── Phase 3b: Copy figures from exec_workdir to report figure dir ──
        # The VisualisationAgent's code executes with CWD=work_dir and may
        # write figures relative to it instead of using the absolute
        # figure_dir path.  Deterministically copy any valid PNGs back.
        import shutil
        _exec_figs = sorted(work_dir.rglob("*.png")) if work_dir.exists() else []
        for _efig in _exec_figs:
            if _efig.stat().st_size >= 5000:
                _dest = figure_dir / _efig.name
                if not _dest.exists():
                    shutil.copy2(_efig, _dest)
                    logger.debug("Copied report figure: %s -> %s", _efig.name, _dest)
        if _exec_figs:
            _n_copied = len(list(figure_dir.glob("*.png")))
            logger.info(
                "[%s] Figure copy-back: %d candidates in exec_workdir, "
                "%d files now in %s",
                file_name, len(_exec_figs), _n_copied, REPORT_FIGURES_SUBDIR,
            )

        # ── Phase 4: Assembly ──
        logger.info("[%s] Phase 4: Report assembly", file_name)
        if not report_md.strip():
            # Check if a higher-quality report already exists (e.g. from
            # the captain pipeline's _run_report_stage direct LLM call).
            # Preserve it rather than overwriting with the fallback template.
            _existing = reports_dir / f"{file_name}__report.md"
            if _existing.exists() and _existing.stat().st_size > 1000:
                logger.info(
                    "[%s] GroupChat produced empty report — preserving "
                    "existing report (%d bytes) over fallback",
                    file_name, _existing.stat().st_size,
                )
                report_md = _existing.read_text("utf-8")
            else:
                logger.warning(
                    "[%s] GroupChat produced empty report — using "
                    "deterministic fallback",
                    file_name,
                )
                report_md = self._fallback_report(item, research_results, knowledge_context)

        report_md = _assemble_report_with_figures(
            report_md, figure_dir, item.get("plots", []),
            figure_selection=self.figure_selection,
            max_figures=self.max_report_figures,
        )

        # Write outputs
        md_path = reports_dir / f"{file_name}__report.md"
        md_path.write_text(report_md, encoding="utf-8")

        pdf_path = _markdown_to_pdf(
            report_md,
            reports_dir / f"{file_name}__report.pdf",
            title=file_name,
            plot_paths=item.get("plots", []),
        )

        return {
            "file": item["file"],
            "file_name": file_name,
            "report_path": str(md_path),
            "pdf_path": pdf_path,
            "research_results": research_results,
            "sections_generated": report_md.count("\n## "),
        }

    def _query_knowledge_for_item(self, item: Dict[str, Any]) -> str:
        """Query the knowledge agent for domain-specific context."""
        domain_hints = item.get("domain_hints", {})
        findings = item.get("analysis_summary", {}).get("findings", [])

        queries = []
        if domain_hints.get("chromatography"):
            queries.append(
                "What are typical acceptance criteria and reference ranges for "
                "SEC monomer percentage, aggregate levels, and fragment levels "
                "in biologics characterisation?"
            )
        if domain_hints.get("mass_spectrometry"):
            queries.append(
                "What are expected mass accuracy tolerances and charge state "
                "distributions for intact mass analysis of monoclonal antibodies?"
            )
        if not queries:
            queries.append(
                "What are the key quality attributes and typical analytical "
                "methods used in biologics characterisation?"
            )

        knowledge_parts = []
        kb = self._get_knowledge_base()
        for q in queries:
            response = _query_knowledge(kb, q, self.llm_config)
            if response:
                knowledge_parts.append(response)

        return "\n\n".join(knowledge_parts)

    def _build_groupchat_payload(
        self,
        item: Dict[str, Any],
        research_results: Dict[str, Any],
        knowledge_context: str,
        figure_dir: Path,
    ) -> str:
        """Build the initial message for the GroupChat agents."""
        file_name = item["file_name"]
        analysis_summary = item.get("analysis_summary", {})
        cleaning_summary = item.get("cleaning_summary", {})
        cross_validation = item.get("cross_validation", {})
        plots = item.get("plots", [])
        cleaned_path = item.get("cleaned_path", "")

        payload = {
            "task": "report_generation",
            "file_name": file_name,
            "cleaned_data_path": cleaned_path,
            "analysis_summary_path": item.get("analysis_summary_path", ""),
            "figure_output_dir": str(figure_dir),
            "existing_plots": plots,
            "analysis_summary": analysis_summary,
            "cleaning_summary": cleaning_summary,
            "cross_validation": cross_validation,
            "research_context": research_results,
            "knowledge_context": knowledge_context,
            "domain_hints": item.get("domain_hints", {}),
        }

        # Build a human-readable listing of existing analysis figures so
        # both VisualisationAgent and InterpretationAgent are aware of them.
        existing_figures_listing = ""
        if plots:
            fig_lines = [
                "## Existing Analysis Figures",
                "",
                "The following figures were generated during the analysis stage. "
                "VisualisationAgent should enhance these rather than recreating "
                "equivalent plots. InterpretationAgent should reference and "
                "discuss these alongside any new figures.",
                "",
            ]
            for p in plots:
                name = Path(p).name
                stem = Path(p).stem.replace("_", " ").replace("-", " ").title()
                fig_lines.append(f"- `{name}` — {stem}")
            existing_figures_listing = "\n".join(fig_lines) + "\n\n"

        return (
            f"Generate a publication-quality scientific report for the analysis of **{file_name}**.\n\n"
            f"## Available Data\n\n"
            f"```json\n{json.dumps(payload, indent=2, default=str)[:self._budget_per_file]}\n```\n\n"
            f"{existing_figures_listing}"
            f"## Workflow\n\n"
            f"1. **VisualisationAgent**: Review existing analysis figures listed above. "
            f"Create enhanced or supplementary figures from the cleaned data at "
            f"`{cleaned_path}`. Save NEW figures to `{figure_dir}` using the naming "
            f"convention `fig{{N}}_{{description}}.png` (e.g. fig1_uv_by_column.png). "
            f"Do NOT recreate plots that already exist in the analysis figures — instead, "
            f"focus on creating new visualisations that add analytical value (e.g. "
            f"cross-group comparisons, statistical overlays, summary dashboards). "
            f"Use a professional scientific style (seaborn 'whitegrid', publication fonts).\n\n"
            f"2. **InterpretationAgent**: Write the full report using the analysis findings, "
            f"research context, knowledge base context, existing analysis figures, AND new "
            f"figures from VisualisationAgent. Reference figures by their exact filename. "
            f"The report must contain: Title, Executive Summary, Introduction, Materials & "
            f"Methods, Results, Discussion, Conclusions & Recommendations, and References.\n\n"
            f"When the report is complete, include 'REPORT_COMPLETE' at the end."
        )

    def _run_groupchat(
        self,
        payload_message: str,
        work_dir: Path,
        figure_dir: Optional[Path] = None,
    ) -> str:
        """Execute the GroupChat and extract the report markdown."""
        try:
            vis_agent, interp_agent, user_proxy, groupchat, manager = (
                _create_groupchat_agents(self.llm_config, work_dir, figure_dir=figure_dir)
            )

            user_proxy.initiate_chat(
                manager,
                message=payload_message,
                clear_history=True,
            )

            report_md = _extract_report_markdown(groupchat.messages)
            if not report_md:
                agent_names = [m.get("name", "?") for m in groupchat.messages]
                interp_spoke = any(n == "InterpretationAgent" for n in agent_names)
                logger.warning(
                    "Per-file report extraction returned empty. "
                    "Rounds used: %d/%d, InterpretationAgent spoke: %s, "
                    "speakers: %s",
                    len(groupchat.messages),
                    groupchat.max_round,
                    interp_spoke,
                    agent_names[-10:],
                )
            return report_md

        except Exception as exc:
            logger.error("GroupChat execution failed: %s", exc)
            return ""

    def _generate_global_report(
        self,
        items: List[Dict[str, Any]],
        file_results: List[Dict[str, Any]],
        reports_dir: Path,
    ) -> Dict[str, Any]:
        """Generate a cross-file comparison report with full amalgamation.

        Passes enriched data (all findings, analysis summaries, individual report
        content, figures, cross-validation details) and uses a GroupChat with both
        VisualisationAgent and InterpretationAgent for cross-file comparison figures.
        """
        # ── Build enriched file summaries ──
        file_summaries = []
        all_cleaned_paths = []
        all_figure_paths = []
        all_analysis_summary_paths = []

        for item, result in zip(items, file_results):
            analysis = item.get("analysis_summary", {})
            findings = analysis.get("findings", [])
            cross_val = item.get("cross_validation", {})
            cleaned_path = item.get("cleaned_path", "")

            # Read individual report content (truncated for context window)
            _excerpt_budget = self._budget_report_excerpt
            individual_report_content = ""
            report_path = result.get("report_path")
            if report_path and Path(report_path).exists():
                try:
                    full_content = Path(report_path).read_text("utf-8")
                    individual_report_content = full_content[:_excerpt_budget]
                    if len(full_content) > _excerpt_budget:
                        individual_report_content += "\n\n[... truncated ...]"
                except Exception:
                    pass

            file_summaries.append({
                "file": item["file_name"],
                "finding_count": len(findings),
                "top_findings": findings[:10],
                "all_findings_count": len(findings),
                "per_group_keys": list(analysis.get("per_group", {}).keys())[:20],
                "cross_validation": {
                    "consistent": cross_val.get("consistent"),
                    "verified_claims": cross_val.get("verified_claims", [])[:5],
                    "gaps": cross_val.get("gaps", []),
                },
                "report_generated": report_path is not None,
                "individual_report_excerpt": individual_report_content,
                "cleaned_path": cleaned_path,
                "analysis_summary_path": item.get("analysis_summary_path", ""),
            })

            if cleaned_path and Path(cleaned_path).exists():
                all_cleaned_paths.append(cleaned_path)
            if item.get("analysis_summary_path"):
                all_analysis_summary_paths.append(item["analysis_summary_path"])
            all_figure_paths.extend(item.get("plots", []))

        # ── Set up GroupChat with VisualisationAgent ──
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
        from prompts import GLOBAL_INTERPRETATION_PROMPT, VISUALISATION_AGENT_PROMPT

        exec_dir = reports_dir.parent / "exec_workdir"
        work_dir = exec_dir / "report_global"
        work_dir.mkdir(parents=True, exist_ok=True)

        global_figure_dir = reports_dir / "global_figures"
        global_figure_dir.mkdir(parents=True, exist_ok=True)

        user_proxy = UserProxyAgent(
            name="Computer_terminal",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": str(work_dir),
                "use_docker": False,
                "timeout": 120,
            },
            max_consecutive_auto_reply=5,
            default_auto_reply="",
            is_termination_msg=lambda msg: "REPORT_COMPLETE" in msg.get("content", ""),
        )

        vis_agent = AssistantAgent(
            name="VisualisationAgent",
            system_message=VISUALISATION_AGENT_PROMPT,
            llm_config=self.llm_config,
        )

        interp_agent = AssistantAgent(
            name="GlobalInterpretationAgent",
            system_message=GLOBAL_INTERPRETATION_PROMPT,
            llm_config=self.llm_config,
        )

        # Speaker selection: vis_agent creates cross-file figures, then interp writes
        # BS-1: Track whether we've injected the figure manifest yet
        _figure_manifest_injected = False

        _MAX_VIS_TURNS_GLOBAL = 5

        def _global_speaker_selection(last_speaker, groupchat):
            nonlocal _figure_manifest_injected
            vis_turns = sum(
                1 for m in groupchat.messages
                if m.get("name") == vis_agent.name
            )

            if last_speaker == user_proxy:
                # BS-1: After user_proxy executes VisAgent code, scan for
                # generated figures and inject a manifest message so the
                # InterpretationAgent knows which global figures exist.
                if not _figure_manifest_injected:
                    # Check if any vis_agent code has been executed
                    vis_spoke = any(
                        m.get("name") == vis_agent.name
                        for m in groupchat.messages
                    )
                    if vis_spoke and global_figure_dir.exists():
                        generated = sorted(
                            p.name for p in global_figure_dir.iterdir()
                            if p.suffix.lower() in (".png", ".svg", ".jpg")
                        )
                        if generated:
                            manifest_msg = (
                                "## Generated Global Figures\n\n"
                                "The following figures have been created in "
                                f"`{global_figure_dir}`:\n\n"
                                + "\n".join(f"- `{fn}`" for fn in generated)
                                + "\n\nUse these exact filenames when referencing "
                                "figures in the report."
                            )
                            groupchat.messages.append({
                                "role": "assistant",
                                "name": "FigureManifest",
                                "content": manifest_msg,
                            })
                            _figure_manifest_injected = True

                # Force handoff after max vis turns
                if vis_turns >= _MAX_VIS_TURNS_GLOBAL:
                    logger.info(
                        "Global VisualisationAgent reached %d turns — "
                        "handing off to GlobalInterpretationAgent",
                        vis_turns,
                    )
                    return interp_agent
                for msg in reversed(groupchat.messages):
                    if msg.get("name") == vis_agent.name:
                        return vis_agent
                    if msg.get("name") == interp_agent.name:
                        return interp_agent
                return vis_agent
            if last_speaker == vis_agent:
                last_msg = groupchat.messages[-1].get("content", "")
                if "```python" in last_msg and vis_turns < _MAX_VIS_TURNS_GLOBAL:
                    return user_proxy
                return interp_agent
            if last_speaker == interp_agent:
                last_msg = groupchat.messages[-1].get("content", "")
                if "```python" in last_msg:
                    return user_proxy
                if "REPORT_COMPLETE" in last_msg:
                    return None
                return interp_agent
            return vis_agent

        groupchat = GroupChat(
            agents=[user_proxy, vis_agent, interp_agent],
            messages=[],
            max_round=20,
            speaker_selection_method=_global_speaker_selection,
            allow_repeat_speaker=True,
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config,
        )

        # ── Build enriched payload message ──
        # NOTE (BS-1): Do NOT pass per-file analysis figures as
        # "existing_figure_paths" — they confuse the InterpretationAgent
        # into referencing per-file plots instead of global figures.
        # The VisualisationAgent creates new figures in global_figure_dir.

        # BS-5: Pre-read column metadata from each cleaned parquet file
        # so the VisualisationAgent knows exact column names for merging.
        dataset_columns = {}
        for cp in all_cleaned_paths:
            try:
                import pandas as pd
                cols = pd.read_parquet(cp, columns=[]).columns.tolist()
                dataset_columns[Path(cp).stem] = cols
            except Exception as exc:
                logger.warning("Could not read columns from %s: %s", cp, exc)

        payload = {
            "task": "global_report_generation",
            "dataset_count": len(items),
            "file_summaries": file_summaries,
            "cleaned_data_paths": all_cleaned_paths,
            "analysis_summary_paths": all_analysis_summary_paths,
            "figure_output_dir": str(global_figure_dir),
            "dataset_columns": dataset_columns,
        }

        summary_msg = (
            f"Generate a **cross-file comparison report** synthesising the analyses of "
            f"{len(items)} datasets.\n\n"
            f"## Available Data\n\n"
            f"```json\n{json.dumps(payload, indent=2, default=str)[:self._budget_global]}\n```\n\n"
            f"## Dataset Column Reference (BS-5)\n\n"
            f"Use the `dataset_columns` field above to determine exact column names "
            f"when loading parquet files. Do NOT assume column names — read them from "
            f"this metadata to avoid 'Unknown' labels or KeyError failures.\n\n"
            f"## Workflow\n\n"
            f"1. **VisualisationAgent**: Create cross-file comparison figures from the "
            f"cleaned data files. Save figures to `{global_figure_dir}`. Create at least: "
            f"a comparative boxplot/violin of key metrics across datasets, and a "
            f"deviation heatmap showing runs×stages across all files.\n\n"
            f"2. **GlobalInterpretationAgent**: Write the full cross-file report using "
            f"all analysis findings, individual report excerpts, and new cross-comparison "
            f"figures. The report must be SELF-CONTAINED — summarise each dataset's key "
            f"findings before the cross-file analysis.\n\n"
            f"When the report is complete, include 'REPORT_COMPLETE' at the end."
        )

        try:
            user_proxy.initiate_chat(
                manager,
                message=summary_msg,
                clear_history=True,
            )
            report_md = _extract_report_markdown(groupchat.messages)
        except Exception as exc:
            logger.error("Global report GroupChat failed: %s", exc)
            report_md = ""

        # Fallback: extract from direct chat if GroupChat extraction failed
        if not report_md:
            for msg in reversed(
                interp_agent.chat_messages.get(user_proxy, [])
                + groupchat.messages
            ):
                content = msg.get("content", "")
                if content and len(content) > 200 and "#" in content:
                    report_md = content.replace("REPORT_COMPLETE", "").strip()
                    break

        # ── Copy global figures from exec_workdir to global_figure_dir ──
        # VisualisationAgent code executes in work_dir; figures may land
        # there instead of global_figure_dir if the LLM used relative paths.
        import shutil as _shutil_g
        _exec_global_figs = sorted(work_dir.rglob("*.png")) if work_dir.exists() else []
        for _efig in _exec_global_figs:
            if _efig.stat().st_size >= 5000:
                _dest = global_figure_dir / _efig.name
                if not _dest.exists():
                    _shutil_g.copy2(_efig, _dest)
                    logger.debug("Copied global figure: %s -> %s", _efig.name, _dest)
        if _exec_global_figs:
            _n_global = len(list(global_figure_dir.glob("*.png")))
            logger.info(
                "Global figure copy-back: %d candidates in exec_workdir, "
                "%d files now in global_figures/",
                len(_exec_global_figs), _n_global,
            )

        if report_md:
            # Assemble with global figures
            # BS-1: Pass empty original_plots — the figure_dir scan in
            # _assemble_report_with_figures will pick up global figures.
            # Passing all_figure_paths here would mix in per-file analysis
            # plots that don't belong in the global report.
            report_md = _assemble_report_with_figures(
                report_md, global_figure_dir, [],
                figure_selection=self.figure_selection,
                max_figures=self.max_report_figures,
            )

            global_md_path = reports_dir / "global_report.md"
            global_md_path.write_text(report_md, encoding="utf-8")

            # Use global figures for PDF, not per-file analysis plots
            _global_plot_paths = [
                str(p) for p in sorted(global_figure_dir.glob("*.png"))
            ]
            global_pdf_path = _markdown_to_pdf(
                report_md,
                reports_dir / "global_report.pdf",
                title="Cross-File Comparison Report",
                plot_paths=_global_plot_paths,
            )

            return {
                "report_path": str(global_md_path),
                "pdf_path": global_pdf_path,
            }

        return {"error": "empty_global_report"}

    def _cleanup_reports_dir(self, reports_dir: Path) -> None:
        """Remove stray directories from reports/ — keep only report files and figures."""
        import shutil
        for child in sorted(reports_dir.iterdir()):
            if child.is_dir() and child.name not in (REPORT_FIGURES_SUBDIR, "global_figures"):
                shutil.rmtree(child, ignore_errors=True)
                logger.info("Cleaned up stray directory from reports: %s", child.name)

    @staticmethod
    def _fallback_report(
        item: Dict[str, Any],
        research_results: Dict[str, Any],
        knowledge_context: str,
    ) -> str:
        """Deterministic fallback report when the GroupChat produces nothing."""
        file_name = item.get("file_name", "Unknown")
        analysis = item.get("analysis_summary", {})
        cleaning = item.get("cleaning_summary", {})
        cross_val = item.get("cross_validation", {})
        findings = analysis.get("findings", [])

        sections = [f"# Analysis Report: {file_name}\n"]
        sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        # Executive Summary
        sections.append("## 1. Executive Summary\n")
        sections.append(
            f"This report presents the automated analysis of **{file_name}**. "
            f"A total of {len(findings)} analytical findings were identified.\n"
        )

        # Data Overview
        sections.append("## 2. Data Overview\n")
        rows_b = cleaning.get("rows_before", "N/A")
        rows_a = cleaning.get("rows_after", "N/A")
        sections.append(f"- Rows before cleaning: {rows_b}\n- Rows after cleaning: {rows_a}\n")

        # Findings
        sections.append("## 3. Results\n")
        if findings:
            for i, f in enumerate(findings, 1):
                sections.append(f"{i}. {f}\n")
        else:
            sections.append("No specific findings were recorded.\n")

        # Research Context
        background = research_results.get("background", "")
        if background:
            sections.append("## 4. Literature Context\n")
            sections.append(f"{background[:2000]}\n")

        # Cross-Validation
        if cross_val:
            sections.append("## 5. Cross-Validation\n")
            consistent = cross_val.get("consistent", "N/A")
            sections.append(f"- Consistency check: {'Passed' if consistent else 'Issues found'}\n")
            verified = cross_val.get("verified_claims", [])
            if verified:
                sections.append("- Verified claims:\n")
                for claim in verified[:5]:
                    sections.append(f"  - {claim}\n")

        # Plots
        plots = item.get("plots", [])
        if plots:
            sections.append("## 6. Figures\n")
            for i, p in enumerate(plots, 1):
                name = Path(p).stem.replace("_", " ").title()
                sections.append(f"**Figure {i}**: {name}\n\n![{name}]({p})\n\n")

        return "\n".join(sections)


# ──────────────────────────────────────────────────────────────────────
# Knowledge Base Setup
# ──────────────────────────────────────────────────────────────────────


def setup_knowledge_base(target_dir: Optional[Path] = None) -> Path:
    """Bootstrap the biologics knowledge base with public reference documents.

    Downloads ICH guidelines and open-access method primers into the
    knowledge_base directory for use by the local ChromaDB knowledge base.

    Returns the path to the knowledge base directory.
    """
    kb_dir = target_dir or KNOWLEDGE_BASE_DIR
    kb_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder references — these would be replaced with actual
    # download URLs for ICH Q6B, Q2(R2), and relevant open-access papers
    references = [
        {
            "name": "biologics_characterisation_primer.md",
            "content": (
                "# Biologics Characterisation — Quick Reference\n\n"
                "## Size Exclusion Chromatography (SEC)\n"
                "- Purpose: Quantify monomer, aggregate, and fragment content\n"
                "- Typical monomer %: >95% for mAbs\n"
                "- HMW aggregates: typically <5%\n"
                "- LMW fragments: typically <2%\n\n"
                "## Ion Exchange Chromatography (IEX)\n"
                "- Purpose: Charge variant analysis\n"
                "- Acidic variants: deamidation, sialylation\n"
                "- Basic variants: C-terminal lysine, succinimide\n"
                "- Main peak: typically 50-80%\n\n"
                "## Intact Mass Analysis\n"
                "- Purpose: Confirm molecular weight, detect modifications\n"
                "- Mass accuracy: typically <50 ppm for mAbs\n"
                "- Expected mass: ~148 kDa for IgG1\n"
                "- Glycoform resolution: G0F, G1F, G2F\n\n"
                "## CE-SDS (Capillary Electrophoresis)\n"
                "- Non-reduced: intact IgG, fragments\n"
                "- Reduced: HC (~50 kDa) and LC (~25 kDa)\n"
                "- Purity: typically >95%\n\n"
                "## ICH Q6B Key Points\n"
                "- Specifications for biotechnological/biological products\n"
                "- Requires characterisation of physicochemical properties\n"
                "- Identity, purity, potency testing\n"
                "- Stability indicating methods\n\n"
                "## Common Quality Attributes\n"
                "- Appearance (colour, clarity)\n"
                "- pH\n"
                "- Osmolality\n"
                "- Protein concentration (A280)\n"
                "- Subvisible particles\n"
                "- Endotoxin levels\n"
                "- Bioburden\n"
            ),
        },
        {
            "name": "analytical_method_reference_ranges.md",
            "content": (
                "# Analytical Method Reference Ranges\n\n"
                "## SEC (Size Exclusion Chromatography)\n"
                "| Attribute | Typical Range | Alert Limit |\n"
                "|-----------|--------------|-------------|\n"
                "| Monomer % | 95-100% | <95% |\n"
                "| HMW Aggregates | 0-5% | >5% |\n"
                "| LMW Fragments | 0-2% | >2% |\n"
                "| Recovery | 90-110% | <90% |\n\n"
                "## IEX (Ion Exchange Chromatography)\n"
                "| Attribute | Typical Range | Alert Limit |\n"
                "|-----------|--------------|-------------|\n"
                "| Main Peak | 50-80% | <40% |\n"
                "| Acidic Variants | 10-30% | >40% |\n"
                "| Basic Variants | 5-20% | >30% |\n\n"
                "## Intact Mass\n"
                "| Attribute | Typical Range | Alert Limit |\n"
                "|-----------|--------------|-------------|\n"
                "| Mass Accuracy | <50 ppm | >100 ppm |\n"
                "| S/N Ratio | >10 | <5 |\n\n"
                "## CE-SDS\n"
                "| Attribute | Typical Range | Alert Limit |\n"
                "|-----------|--------------|-------------|\n"
                "| Purity (NR) | >95% | <90% |\n"
                "| HC + LC (R) | >95% | <90% |\n\n"
                "## General\n"
                "| Attribute | Typical Range |\n"
                "|-----------|-------------|\n"
                "| pH | 5.0-7.0 |\n"
                "| Protein Conc | ±10% of target |\n"
                "| Osmolality | 250-350 mOsm/kg |\n"
            ),
        },
    ]

    for ref in references:
        ref_path = kb_dir / ref["name"]
        if not ref_path.exists():
            ref_path.write_text(ref["content"], encoding="utf-8")
            logger.info("Created knowledge base reference: %s", ref_path.name)

    return kb_dir


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def run_report_pipeline(
    manifest: Dict[str, Any],
    llm_config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    max_research_turns: int = DEFAULT_MAX_RESEARCH_TURNS,
    max_groupchat_rounds: int = DEFAULT_MAX_GROUPCHAT_ROUNDS,
    figure_selection: str = "all",
    max_report_figures: int = 10,
) -> Dict[str, Any]:
    """Convenience function to run the report pipeline.

    Args:
        manifest: Output dict from run_captain_pipeline()
        llm_config: LLM configuration dict (same format as captain pipeline)
        output_dir: Override output directory (defaults to manifest's output_root)
        max_research_turns: Max turns for the DeepResearchAgent
        max_groupchat_rounds: Max rounds for the GroupChat
        figure_selection: Figure selection strategy ("all", "ranked", "top_n")
        max_report_figures: Max figures when using ranked/top_n

    Returns:
        Dict with report paths and metadata
    """
    # Ensure knowledge base is bootstrapped
    setup_knowledge_base()

    pipeline = ReportPipeline(
        llm_config=llm_config,
        output_dir=output_dir,
        max_research_turns=max_research_turns,
        max_groupchat_rounds=max_groupchat_rounds,
        figure_selection=figure_selection,
        max_report_figures=max_report_figures,
    )
    return pipeline.run(manifest)
