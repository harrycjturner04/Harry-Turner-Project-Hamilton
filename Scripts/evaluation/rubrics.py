"""Evaluation rubric definitions and judge prompt templates."""

from typing import Any, Dict, List

# ── Rubric Definition ────────────────────────────────────────────────────────

RUBRIC_V1 = {
    "version": "v1",
    "criteria": [
        {
            "name": "analytical_depth",
            "weight": 0.20,
            "description": (
                "Depth and thoroughness of analysis beyond "
                "surface-level statistics"
            ),
            "anchors": {
                0: "No analysis; only restates raw numbers",
                3: (
                    "Basic descriptive statistics only; no stratification "
                    "or subgroup analysis"
                ),
                5: (
                    "Moderate depth; some per-group breakdown but missing "
                    "key analytical dimensions"
                ),
                7: (
                    "Good depth; per-run/per-stage analysis with identified "
                    "patterns and anomalies"
                ),
                10: (
                    "Exceptional; multi-dimensional analysis with interaction "
                    "effects, root cause hypotheses, and mechanistic reasoning"
                ),
            },
        },
        {
            "name": "statistical_reasoning",
            "weight": 0.15,
            "description": (
                "Correctness and appropriateness of statistical "
                "methods and claims"
            ),
            "anchors": {
                0: (
                    "Statistically incorrect claims; fundamentally "
                    "flawed methodology"
                ),
                3: (
                    "Basic statistics present but inappropriate tests "
                    "or missing assumptions checks"
                ),
                5: (
                    "Correct basic statistics; some p-values but without "
                    "effect sizes or confidence intervals"
                ),
                7: (
                    "Appropriate tests with reported effect sizes; "
                    "acknowledges limitations"
                ),
                10: (
                    "Rigorous statistical methodology; correct test "
                    "selection, effect sizes, confidence intervals, "
                    "multiple comparison corrections where needed"
                ),
            },
        },
        {
            "name": "figure_quality",
            "weight": 0.10,
            "description": (
                "Quality, interpretability, and relevance of figures "
                "referenced in the report"
            ),
            "anchors": {
                0: "No figures or figures unrelated to findings",
                3: (
                    "Figures present but poorly labeled, redundant, "
                    "or not referenced in text"
                ),
                5: (
                    "Adequate figures; labels present but could be more "
                    "informative; some redundancy"
                ),
                7: (
                    "Good figures covering diverse analytical questions; "
                    "clear labels and legends; well-referenced in narrative"
                ),
                10: (
                    "Publication-quality figures; every figure answers a "
                    "distinct question; clear quantitative annotations; "
                    "figure captions explain what the reader should observe"
                ),
            },
        },
        {
            "name": "claim_evidence_consistency",
            "weight": 0.20,
            "description": (
                "Consistency between stated claims and "
                "supporting evidence/data"
            ),
            "anchors": {
                0: (
                    "Claims contradict the data or have no "
                    "supporting evidence"
                ),
                3: (
                    "Some claims supported but key findings lack data "
                    "backing; some fabricated numbers"
                ),
                5: (
                    "Most claims have evidence but some gaps; minor "
                    "inconsistencies between text and figures"
                ),
                7: (
                    "Strong evidence-claim alignment; cross-validation "
                    "results integrated; minor gaps only"
                ),
                10: (
                    "Every claim traceable to specific computed values; "
                    "cross-validation confirms all key metrics; "
                    "no unsupported assertions"
                ),
            },
        },
        {
            "name": "report_clarity",
            "weight": 0.10,
            "description": (
                "Clarity, organization, and scientific "
                "communication quality"
            ),
            "anchors": {
                0: "Incoherent; no logical structure",
                3: (
                    "Some structure but poor flow; mixing of results "
                    "and methods; jargon without definition"
                ),
                5: (
                    "Readable with standard sections; adequate but "
                    "not engaging scientific prose"
                ),
                7: (
                    "Well-organized IMRAD structure; clear writing; "
                    "appropriate use of domain terminology"
                ),
                10: (
                    "Publication-ready scientific communication; "
                    "compelling narrative arc; excellent executive "
                    "summary; clear limitations section"
                ),
            },
        },
        {
            "name": "evidence_based_reasoning",
            "weight": 0.15,
            "description": (
                "Use of evidence-based reasoning and citations "
                "to support interpretations"
            ),
            "anchors": {
                0: (
                    "No references to literature or standards; "
                    "purely speculative interpretations"
                ),
                3: (
                    "Generic references (e.g., 'according to ICH') "
                    "without specific application"
                ),
                5: (
                    "Some specific regulatory citations; attempts to "
                    "connect findings to domain knowledge"
                ),
                7: (
                    "Good integration of regulatory standards and domain "
                    "knowledge; specific acceptance criteria cited "
                    "and applied"
                ),
                10: (
                    "Deep integration of biologics domain knowledge; "
                    "specific regulatory guideline sections applied to "
                    "findings; comparison with published benchmarks"
                ),
            },
        },
        {
            "name": "domain_correctness",
            "weight": 0.10,
            "description": (
                "Scientific accuracy in biologics/chromatography/"
                "MS domain"
            ),
            "anchors": {
                0: (
                    "Fundamentally incorrect domain claims; "
                    "misidentifies analytical techniques"
                ),
                3: (
                    "Some correct domain knowledge but significant "
                    "errors in interpretation"
                ),
                5: (
                    "Generally correct but superficial domain "
                    "application; no errors but no deep insight"
                ),
                7: (
                    "Correct domain application; appropriate "
                    "interpretation of UV280, conductivity, MS metrics; "
                    "understands chromatography stages"
                ),
                10: (
                    "Expert-level domain correctness; understands column "
                    "chemistry, binding mechanisms, regulatory significance "
                    "of metrics; identifies domain-specific failure modes"
                ),
            },
        },
    ],
}

GLOBAL_RUBRIC_V1 = {
    "version": "global_v1",
    "criteria": [
        {
            "name": "cross_dataset_synthesis",
            "weight": 0.25,
            "description": (
                "Quality of integration across datasets — does the report "
                "connect findings rather than concatenating per-dataset summaries"
            ),
            "anchors": {
                0: "No cross-dataset analysis; merely lists individual results",
                3: (
                    "Mentions multiple datasets but does not draw connections "
                    "or identify shared patterns"
                ),
                5: (
                    "Some cross-dataset comparisons present but analysis "
                    "remains surface-level; obvious connections noted"
                ),
                7: (
                    "Strong synthesis; identifies shared trends, divergences, "
                    "and interactions across datasets with supporting evidence"
                ),
                10: (
                    "Exceptional integration; reveals non-obvious relationships "
                    "between datasets, builds a unified mechanistic narrative, "
                    "and quantifies cross-dataset effect sizes"
                ),
            },
        },
        {
            "name": "narrative_coherence",
            "weight": 0.20,
            "description": (
                "Logical flow and unified scientific storytelling across "
                "the entire report"
            ),
            "anchors": {
                0: "Incoherent; sections are disjointed with no logical thread",
                3: (
                    "Some structure but reads as disconnected dataset summaries "
                    "stitched together; no overarching narrative"
                ),
                5: (
                    "Adequate structure with introduction and conclusion; "
                    "transitions between sections are mechanical"
                ),
                7: (
                    "Well-structured narrative arc; each section builds on "
                    "the previous; clear thematic progression"
                ),
                10: (
                    "Publication-ready scientific communication; compelling "
                    "narrative that guides the reader from question through "
                    "evidence to insight; seamless transitions"
                ),
            },
        },
        {
            "name": "conclusion_scoping",
            "weight": 0.15,
            "description": (
                "Appropriateness of conclusions relative to the breadth "
                "and strength of the evidence presented"
            ),
            "anchors": {
                0: (
                    "Conclusions completely unsupported or wildly over-claimed "
                    "relative to evidence"
                ),
                3: (
                    "Significant over-generalisation; draws broad conclusions "
                    "from narrow evidence without caveats"
                ),
                5: (
                    "Conclusions generally reasonable but missing important "
                    "limitations or confidence qualifications"
                ),
                7: (
                    "Well-scoped conclusions; clearly distinguishes strong "
                    "findings from tentative observations; acknowledges "
                    "limitations"
                ),
                10: (
                    "Exemplary scoping; conclusions precisely calibrated to "
                    "evidence strength; explicit confidence levels; clear "
                    "separation of findings, interpretations, and speculation"
                ),
            },
        },
        {
            "name": "actionable_insights",
            "weight": 0.15,
            "description": (
                "Quality of executive summary and practical takeaways "
                "for a biologics audience"
            ),
            "anchors": {
                0: "No actionable content; purely descriptive with no recommendations",
                3: (
                    "Vague recommendations without specificity; "
                    "not tailored to biologics context"
                ),
                5: (
                    "Some useful recommendations but lacking prioritisation "
                    "or concrete next steps"
                ),
                7: (
                    "Clear, prioritised recommendations with domain-specific "
                    "context; identifies key risks and opportunities"
                ),
                10: (
                    "Expert-level executive summary; actionable items ranked "
                    "by impact and feasibility; directly maps findings to "
                    "process decisions, regulatory considerations, or "
                    "further investigation priorities"
                ),
            },
        },
        {
            "name": "contradiction_identification",
            "weight": 0.10,
            "description": (
                "Identification and handling of inconsistencies or "
                "contradictions between datasets"
            ),
            "anchors": {
                0: (
                    "Contradictions between datasets ignored or not noticed"
                ),
                3: (
                    "Some inconsistencies noted but not explained or resolved"
                ),
                5: (
                    "Contradictions identified and flagged; basic attempt "
                    "to explain discrepancies"
                ),
                7: (
                    "Systematic identification of cross-dataset tensions; "
                    "plausible explanations offered with evidence"
                ),
                10: (
                    "Thorough treatment of all inconsistencies; uses "
                    "contradictions as analytical leverage to deepen "
                    "understanding; proposes testable hypotheses"
                ),
            },
        },
        {
            "name": "completeness",
            "weight": 0.15,
            "description": (
                "Coverage of all datasets and major findings from the "
                "individual analyses"
            ),
            "anchors": {
                0: "Covers only one dataset; ignores most pipeline outputs",
                3: (
                    "Covers some datasets but omits major findings or "
                    "entire analytical dimensions"
                ),
                5: (
                    "Most datasets represented; some important findings "
                    "missing or under-represented"
                ),
                7: (
                    "All datasets covered with their key findings; minor "
                    "omissions only in secondary details"
                ),
                10: (
                    "Comprehensive coverage of every dataset; all major "
                    "and significant minor findings represented with "
                    "appropriate emphasis and proportionality"
                ),
            },
        },
    ],
}

RUBRICS = {"v1": RUBRIC_V1, "global_v1": GLOBAL_RUBRIC_V1}


def get_rubric(version: str = "v1") -> Dict[str, Any]:
    """Return the rubric definition for a given version."""
    if version not in RUBRICS:
        raise ValueError(
            f"Unknown rubric version {version!r}. "
            f"Available: {list(RUBRICS.keys())}"
        )
    return RUBRICS[version]


def get_criterion_names(version: str = "v1") -> List[str]:
    """Return ordered list of criterion names."""
    rubric = get_rubric(version)
    return [c["name"] for c in rubric["criteria"]]


def get_criterion_weights(version: str = "v1") -> Dict[str, float]:
    """Return dict of criterion_name -> weight."""
    rubric = get_rubric(version)
    return {c["name"]: c["weight"] for c in rubric["criteria"]}


# ── Prompt Templates ─────────────────────────────────────────────────────────

def _format_rubric_for_prompt(version: str = "v1") -> str:
    """Format the rubric as text for inclusion in the judge prompt."""
    rubric = get_rubric(version)
    lines = []
    for c in rubric["criteria"]:
        lines.append(f"### {c['name']} (weight: {c['weight']})")
        lines.append(c["description"])
        lines.append("Anchor points:")
        for score, desc in sorted(c["anchors"].items()):
            lines.append(f"  {score}: {desc}")
        lines.append("")
    return "\n".join(lines)


JUDGE_SYSTEM_PROMPT = """You are an expert scientific evaluator assessing the quality of automated \
biologics data analysis outputs. You evaluate reports generated by an AI pipeline that analyzes \
chromatography and mass spectrometry data.

You must evaluate STRICTLY according to the rubric provided. For each criterion:
1. Assign an integer score from 0 to 10 using the provided anchors as reference points.
2. Provide a 2-3 sentence explanation citing SPECIFIC evidence from the report.
3. Be calibrated: a score of 5 means "adequate but unremarkable", 7 means "good", 9+ is rare.

IMPORTANT RULES:
- Base scores ONLY on evidence present in the provided text. Do not hallucinate content.
- If the report makes claims without supporting data, that is a deficiency in claim_evidence_consistency.
- If statistical methods are mentioned but details are missing, penalize statistical_reasoning.
- A report that correctly identifies limitations deserves credit in report_clarity.

## EVALUATION RUBRIC

{rubric_text}

OUTPUT FORMAT (strict JSON, no markdown fences):
{{
    "scores": {{
        "analytical_depth": <int 0-10>,
        "statistical_reasoning": <int 0-10>,
        "figure_quality": <int 0-10>,
        "claim_evidence_consistency": <int 0-10>,
        "report_clarity": <int 0-10>,
        "evidence_based_reasoning": <int 0-10>,
        "domain_correctness": <int 0-10>
    }},
    "explanations": {{
        "analytical_depth": "<2-3 sentences citing specific evidence>",
        "statistical_reasoning": "<2-3 sentences citing specific evidence>",
        "figure_quality": "<2-3 sentences citing specific evidence>",
        "claim_evidence_consistency": "<2-3 sentences citing specific evidence>",
        "report_clarity": "<2-3 sentences citing specific evidence>",
        "evidence_based_reasoning": "<2-3 sentences citing specific evidence>",
        "domain_correctness": "<2-3 sentences citing specific evidence>"
    }},
    "overall_assessment": "<3-5 sentence summary of key strengths and weaknesses>"
}}"""


JUDGE_USER_PROMPT = """Evaluate the following scientific analysis report according to the rubric.

DATASET: {dataset_name}
PIPELINE CONFIGURATION: {run_config_summary}

--- REPORT START ---
{report_content}
--- REPORT END ---

--- ANALYSIS SUMMARY (JSON excerpt, for context only) ---
{analysis_summary_excerpt}
--- ANALYSIS SUMMARY END ---

--- CROSS-VALIDATION RESULTS (for context only) ---
{cross_validation_excerpt}
--- CROSS-VALIDATION END ---

Evaluate this report against each criterion in the rubric. Remember:
- Score 5 = adequate, 7 = good, 9+ = exceptional (rare)
- Cite specific evidence from the report for each score
- Output strict JSON only"""


JUDGE_USER_PROMPT_WITH_VISION = """Evaluate the following scientific analysis report according to the rubric.

DATASET: {dataset_name}
PIPELINE CONFIGURATION: {run_config_summary}

--- REPORT START ---
{report_content}
--- REPORT END ---

--- ANALYSIS SUMMARY (JSON excerpt, for context only) ---
{analysis_summary_excerpt}
--- ANALYSIS SUMMARY END ---

--- CROSS-VALIDATION RESULTS (for context only) ---
{cross_validation_excerpt}
--- CROSS-VALIDATION END ---

The figures generated by the pipeline are attached as images. When scoring \
figure_quality, evaluate the actual figures for readability, labeling, \
information density, and relevance to the analysis claims.

Evaluate this report against each criterion in the rubric. Remember:
- Score 5 = adequate, 7 = good, 9+ = exceptional (rare)
- Cite specific evidence from the report for each score
- Output strict JSON only"""


# ── Pairwise Comparison Prompts ──────────────────────────────────────────────

PAIRWISE_SYSTEM_PROMPT = """You are an expert scientific evaluator comparing two automated analysis \
reports for the same dataset. You must determine which analysis is stronger and explain why.

For each criterion, indicate which report is stronger (A or B) or if they are tied.
Then provide an overall winner with confidence level.

Be specific: cite concrete differences in analytical depth, statistical rigor, and domain correctness.

OUTPUT FORMAT (strict JSON, no markdown fences):
{{
    "criterion_preferences": {{
        "analytical_depth": {{"winner": "A"|"B"|"tie", "reason": "..."}},
        "statistical_reasoning": {{"winner": "A"|"B"|"tie", "reason": "..."}},
        "figure_quality": {{"winner": "A"|"B"|"tie", "reason": "..."}},
        "claim_evidence_consistency": {{"winner": "A"|"B"|"tie", "reason": "..."}},
        "report_clarity": {{"winner": "A"|"B"|"tie", "reason": "..."}},
        "evidence_based_reasoning": {{"winner": "A"|"B"|"tie", "reason": "..."}},
        "domain_correctness": {{"winner": "A"|"B"|"tie", "reason": "..."}}
    }},
    "overall_winner": "A"|"B"|"tie",
    "confidence": <float 0.0-1.0>,
    "explanation": "<3-5 sentence justification>"
}}"""


# ── Global Report Evaluation Prompts ─────────────────────────────────────────

GLOBAL_JUDGE_SYSTEM_PROMPT = """You are an expert scientific evaluator assessing the quality of a global \
synthesis report produced by an automated biologics data analysis pipeline. This report integrates \
findings from multiple individual dataset analyses (e.g. chromatography, mass spectrometry) into a \
single overarching scientific narrative.

You must evaluate STRICTLY according to the rubric provided. For each criterion:
1. Assign an integer score from 0 to 10 using the provided anchors as reference points.
2. Provide a 2-3 sentence explanation citing SPECIFIC evidence from the report.
3. Be calibrated: a score of 5 means "adequate but unremarkable", 7 means "good", 9+ is rare.

IMPORTANT RULES:
- This is a GLOBAL report — it should synthesise across datasets, not merely repeat per-dataset findings.
- Reward cross-dataset comparisons, unified narratives, and appropriately scoped conclusions.
- Penalise reports that read as concatenated per-dataset summaries with no integration.
- A report that correctly identifies contradictions between datasets deserves credit.
- Base scores ONLY on evidence present in the provided text. Do not hallucinate content.

## EVALUATION RUBRIC

{rubric_text}

OUTPUT FORMAT (strict JSON, no markdown fences):
{{{{
    "scores": {{{{
        "cross_dataset_synthesis": <int 0-10>,
        "narrative_coherence": <int 0-10>,
        "conclusion_scoping": <int 0-10>,
        "actionable_insights": <int 0-10>,
        "contradiction_identification": <int 0-10>,
        "completeness": <int 0-10>
    }}}},
    "explanations": {{{{
        "cross_dataset_synthesis": "<2-3 sentences citing specific evidence>",
        "narrative_coherence": "<2-3 sentences citing specific evidence>",
        "conclusion_scoping": "<2-3 sentences citing specific evidence>",
        "actionable_insights": "<2-3 sentences citing specific evidence>",
        "contradiction_identification": "<2-3 sentences citing specific evidence>",
        "completeness": "<2-3 sentences citing specific evidence>"
    }}}},
    "overall_assessment": "<3-5 sentence summary of key strengths and weaknesses>"
}}}}"""


GLOBAL_JUDGE_USER_PROMPT = """Evaluate the following global synthesis report according to the rubric.

PIPELINE CONFIGURATION: {run_config_summary}
DATASETS ANALYSED: {dataset_names}

--- GLOBAL REPORT START ---
{report_content}
--- GLOBAL REPORT END ---

--- PER-DATASET SUMMARIES (for reference — these are the individual analyses the global report should synthesise) ---
{per_dataset_summaries}
--- PER-DATASET SUMMARIES END ---

Evaluate this global report against each criterion in the rubric. Remember:
- This report should SYNTHESISE across datasets, not just summarise them individually
- Score 5 = adequate, 7 = good, 9+ = exceptional (rare)
- Cite specific evidence from the report for each score
- Output strict JSON only"""


PAIRWISE_USER_PROMPT = """Compare these two analysis reports for the dataset: {dataset_name}

--- REPORT {label_a} ---
{report_a_content}
--- END REPORT {label_a} ---

--- REPORT {label_b} ---
{report_b_content}
--- END REPORT {label_b} ---

Which analysis is stronger overall? Evaluate each criterion separately, \
then give an overall verdict."""


def build_judge_system_prompt(rubric_version: str = "v1") -> str:
    """Build the complete judge system prompt with rubric text."""
    rubric_text = _format_rubric_for_prompt(rubric_version)
    if rubric_version.startswith("global"):
        return GLOBAL_JUDGE_SYSTEM_PROMPT.format(rubric_text=rubric_text)
    return JUDGE_SYSTEM_PROMPT.format(rubric_text=rubric_text)
