# Day 08 Lab Report

## 1. Team / student

- Name: Tran Van Quang
- Repo/commit: 6d8252d
- Date: 2026-06-29

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
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total interrupts/approvals | 2 |
| Resume success | False |

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 |
| S02_tool | tool | tool | yes | 0 | 0 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |
| S04_risky | risky | risky | yes | 0 | 1 |
| S05_error | error | error | yes | 2 | 0 |
| S06_delete | risky | risky | yes | 0 | 1 |
| S07_dead_letter | error | error | yes | 1 | 0 |

## 6. Failure analysis

1. Retry or tool failure: transient tool errors are marked by `evaluate` as `needs_retry`.
   The graph loops through `retry` only until `max_attempts`, then sends the ticket to
   `dead_letter` with an explanatory final answer.
2. Risky action without approval: destructive or customer-impacting requests are routed to
   `risky_action` and require an approval decision before tool execution. Rejection routes to
   clarification instead of executing the action.

Latest failed scenarios:

- No scenario failures in the latest run.

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
