#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

from captain_pipeline import run_captain_pipeline, ServerManager, PipelineMode
from report_pipeline import run_report_pipeline


def _build_llm_config(model: str, temperature: float, api_key: str | None, base_url: str | None) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Provide --api-key or set the env var.")
    config = {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
            }
        ],
        "temperature": temperature,
    }
    if base_url:
        config["config_list"][0]["base_url"] = base_url
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CaptainAgent pipeline on an input directory")
    parser.add_argument("input_dir", help="Folder (or file) containing raw data inputs")
    parser.add_argument("output_dir", help="Folder to write pipeline outputs")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.5-27B"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--metadata-db",
        default=os.environ.get("METADATA_DB_DIR"),
        help="Path to vectorized metadata DB (e.g., Database/metadata). Optional.",
    )
    parser.add_argument(
        "--context-path",
        default=os.environ.get("CONTEXT_PATH"),
        help=(
            "Path to a context file (.md or .txt). "
            "Markdown files may contain ```yaml blocks to configure "
            "pipeline stages, agent selection, and quality thresholds. "
            "Auto-discovered from CWD (context.md > context.txt) if not provided."
        ),
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--pipeline-mode",
        choices=["single_pass", "stage_validation", "full_closed_loop"],
        default=os.environ.get("PIPELINE_MODE", "full_closed_loop"),
        help=(
            "Pipeline execution mode for ablation study. "
            "single_pass: no validation/VLM review. "
            "stage_validation: validation + retry (default). "
            "full_closed_loop: validation + VLM review feeds back into reruns."
        ),
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        default=False,
        help="Skip the report generation pipeline after captain pipeline completes.",
    )
    parser.add_argument(
        "--report-only",
        type=str,
        default=None,
        metavar="MANIFEST_PATH",
        help=(
            "Run only the report pipeline on a previously saved manifest JSON. "
            "Skips the captain pipeline entirely."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    llm_config = _build_llm_config(args.model, args.temperature, args.api_key, args.base_url)

    # Create ServerManager if vLLM server PID and port are available
    server_manager = None
    vllm_pid = os.environ.get("VLLM_PID")
    vllm_port = os.environ.get("VLLM_PORT")
    hf_token = os.environ.get("HUGGINGFACE_API_KEY", "")
    if vllm_pid and vllm_port:
        logs_dir = Path(args.output_dir) / "debug" / "vllm_logs"
        server_manager = ServerManager(port=int(vllm_port), hf_token=hf_token, logs_dir=logs_dir)
        server_manager.adopt(pid=int(vllm_pid), model_name="Qwen/Qwen3.5-27B")
        logging.info("ServerManager created: port=%s, adopted PID=%s", vllm_port, vllm_pid)

    # ── Report-only mode: skip captain pipeline ──
    if args.report_only:
        manifest_file = Path(args.report_only)
        if not manifest_file.exists():
            logging.error("Manifest file not found: %s", manifest_file)
            return 1
        import json
        manifest = json.loads(manifest_file.read_text("utf-8"))
        logging.info("Loaded manifest from %s — running report pipeline only", manifest_file)

        report_result = run_report_pipeline(
            manifest=manifest,
            llm_config=llm_config,
            output_dir=Path(args.output_dir),
        )
        logging.info("Report pipeline complete. Reports at: %s", report_result.get("reports_dir"))
        return 0

    # ── Captain pipeline ──
    manifest = run_captain_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        llm_config=llm_config,
        metadata_db_dir=Path(args.metadata_db) if args.metadata_db else None,
        context_path=Path(args.context_path) if args.context_path else None,
        server_manager=server_manager,
        pipeline_mode=PipelineMode(args.pipeline_mode),
        skip_global_report=not args.skip_report,
    )

    manifest_path = manifest.get("manifest_path")
    if manifest_path:
        logging.info("Captain pipeline complete. Manifest written to %s", manifest_path)
    else:
        logging.info("Captain pipeline complete.")

    # ── Report pipeline (auto-triggered unless --skip-report) ──
    if not args.skip_report:
        logging.info("Starting report pipeline...")
        try:
            _rc = manifest.get("run_config", {})
            report_result = run_report_pipeline(
                manifest=manifest,
                llm_config=llm_config,
                output_dir=Path(args.output_dir),
                figure_selection=_rc.get("figure_selection", "all"),
                max_report_figures=_rc.get("max_report_figures", 10),
            )
            logging.info(
                "Report pipeline complete. Reports at: %s",
                report_result.get("reports_dir"),
            )
        except Exception as exc:
            logging.error("Report pipeline failed (non-fatal): %s", exc)
    else:
        logging.info("Report pipeline skipped (--skip-report).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
