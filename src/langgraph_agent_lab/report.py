"""Report generation helper.

TODO(student): implement report rendering using MetricsReport data
and the template in reports/lab_report_template.md.
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from .metrics import MetricsReport


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "local workspace"


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    TODO(student): Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    scenario_rows = "\n".join(
        "| {scenario} | {expected} | {actual} | {success} | {retries} | {interrupts} |".format(
            scenario=item.scenario_id,
            expected=item.expected_route,
            actual=item.actual_route or "",
            success="yes" if item.success else "no",
            retries=item.retry_count,
            interrupts=item.interrupt_count,
        )
        for item in metrics.scenario_metrics
    )
    failures = [item for item in metrics.scenario_metrics if not item.success]
    failure_summary = (
        "\n".join(
            f"- {item.scenario_id}: route={item.actual_route}, errors={item.errors}"
            for item in failures
        )
        if failures
        else "- No scenario failures in the latest run."
    )

    commit = _git_commit()
    report_date = date.today().isoformat()

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: Tran Van Quang
- Repo/commit: {commit}
- Date: {report_date}

## 2. Architecture

The workflow is a LangGraph `StateGraph` with a single typed `AgentState`. It starts at
`intake`, classifies the support request, then uses conditional routing for simple answers,
tool lookups, clarification, risky human approval, retry handling, and dead-letter fallback.
Every path terminates through `finalize`, which appends the final audit event.

The retry path is bounded by `attempt < max_attempts`. Risky requests pass through
`risky_action -> approval` before any tool/action execution.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| query, route, risk_level | overwrite | current request and routing decision |
| attempt, max_attempts | overwrite | bounded retry control |
| evaluation_result | overwrite | conditional gate after tool evaluation |
| pending_question, proposed_action, approval | overwrite | HITL and clarification state |
| final_answer | overwrite | final user-facing response |
| messages, tool_results, errors, events | append | audit trail and grading evidence |

## 4. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total interrupts/approvals | {metrics.total_interrupts} |
| Resume success | {metrics.resume_success} |

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
{scenario_rows}

## 6. Failure analysis

1. Retry or tool failure: transient tool errors are marked by `evaluate` as `needs_retry`.
   The graph loops through `retry` only until `max_attempts`, then sends the ticket to
   `dead_letter` with an explanatory final answer.
2. Risky action without approval: destructive or customer-impacting requests are routed to
   `risky_action` and require an approval decision before tool execution. Rejection routes to
   clarification instead of executing the action.

Latest failed scenarios:

{failure_summary}

## 7. Persistence / recovery evidence

The CLI passes a stable `thread_id` for each scenario and compiles the graph with a checkpointer.
`memory` is used by default for local runs; `sqlite` is implemented through `SqliteSaver` for
durable checkpointing when `langgraph-checkpoint-sqlite` is installed and configured.

## 8. Extension work

Completed extension: SQLite checkpointer support plus a generated Markdown report from metrics.
The approval node can also use LangGraph `interrupt()` when `LANGGRAPH_INTERRUPT=true`.

## 9. Improvement plan

With one more day, the next production step would be replacing the mock tool with real support
system adapters, adding LLM-as-judge evaluation for tool quality, and recording checkpoint
history evidence in the report.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
