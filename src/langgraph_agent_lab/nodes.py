"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .state import AgentState, ApprovalDecision, make_event


class Classification(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The workflow route for the support ticket."
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Risk level of the requested action."
    )
    rationale: str = Field(description="Short reason for the classification.")


def _content_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _offline_classify(query: str) -> Classification:
    """Local fallback for development when no provider key is configured."""
    text = query.lower()
    risky_terms = ("refund", "delete", "cancel", "send confirmation", "email", "chargeback")
    tool_terms = ("lookup", "order", "status", "tracking", "search", "find")
    error_terms = (
        "timeout",
        "failure",
        "failed",
        "crash",
        "unavailable",
        "cannot recover",
        "system",
    )
    vague_terms = ("fix it", "help", "problem", "issue", "not working")

    if any(term in text for term in risky_terms):
        return Classification(route="risky", risk_level="high", rationale="side-effect action")
    if any(term in text for term in tool_terms):
        return Classification(route="tool", risk_level="low", rationale="requires lookup")
    if any(term in text for term in vague_terms) and len(text.split()) <= 5:
        return Classification(
            route="missing_info",
            risk_level="low",
            rationale="underspecified request",
        )
    if any(term in text for term in error_terms):
        return Classification(route="error", risk_level="medium", rationale="system failure")
    return Classification(route="simple", risk_level="low", rationale="general support question")


def _invoke_classification_llm(query: str) -> Classification:
    from .llm import get_llm

    prompt = (
        "Classify this support-ticket query into exactly one route.\n"
        "Routes:\n"
        "- risky: side effects such as refunds, deletes, cancellations, emails, account changes\n"
        "- tool: lookups such as order status, tracking, search, or account data retrieval\n"
        "- missing_info: too vague or incomplete to act safely\n"
        "- error: system failures, timeouts, crashes, service unavailable\n"
        "- simple: answerable support question with no tool or side effect\n"
        "Priority: risky > tool > missing_info > error > simple.\n"
        f"Query: {query}"
    )
    structured = get_llm(temperature=0.0).with_structured_output(Classification)
    result = structured.invoke(prompt)
    if isinstance(result, Classification):
        return result
    if isinstance(result, dict):
        return Classification.model_validate(result)
    return Classification.model_validate_json(_content_text(result))


def _generate_answer_with_llm(state: AgentState) -> str:
    from .llm import get_llm

    prompt = (
        "You are a support agent. Write a concise, helpful final response grounded only in "
        "the provided workflow context. Do not invent tool results.\n\n"
        f"User query: {state.get('query', '')}\n"
        f"Route: {state.get('route', '')}\n"
        f"Tool results: {state.get('tool_results', [])}\n"
        f"Approval: {state.get('approval')}\n"
        f"Errors: {state.get('errors', [])}\n"
    )
    return _content_text(get_llm(temperature=0.2).invoke(prompt)).strip()


def _offline_answer(state: AgentState) -> str:
    route = state.get("route", "simple")
    query = state.get("query", "")
    latest_tool = (state.get("tool_results") or [""])[-1]
    if route == "tool":
        return f"I checked the available support data for your request. {latest_tool}"
    if route == "risky":
        return f"The approved support action has been prepared and processed. {latest_tool}"
    if route == "error":
        return f"The transient failure has been retried and recovered. {latest_tool}"
    return f"Here is the support guidance for your request: {query}"


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    try:
        classification = _invoke_classification_llm(query)
        source = "llm"
    except Exception as exc:
        classification = _offline_classify(query)
        source = f"offline_fallback:{exc.__class__.__name__}"

    return {
        "route": classification.route,
        "risk_level": "high" if classification.route == "risky" else classification.risk_level,
        "messages": [f"classify:{classification.route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                "query classified",
                route=classification.route,
                risk_level=classification.risk_level,
                source=source,
                rationale=classification.rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0) or 0)
    query = state.get("query", "")
    if route == "error" and attempt < 2:
        result = f"ERROR transient tool failure on attempt {attempt}: backend timeout"
        event_type = "failed"
    elif route == "risky":
        action = state.get("proposed_action") or query
        result = f"SUCCESS risky action executed after approval: {action}"
        event_type = "completed"
    else:
        result = f"SUCCESS mock support lookup/action for query: {query}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, "tool executed", attempt=attempt, result=result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                evaluation_result=evaluation_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    try:
        answer = _generate_answer_with_llm(state)
        source = "llm"
    except Exception as exc:
        answer = _offline_answer(state)
        source = f"offline_fallback:{exc.__class__.__name__}"
    return {
        "final_answer": answer,
        "messages": [f"answer:{answer[:60]}"],
        "events": [make_event("answer", "completed", "final answer generated", source=source)],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    pending_question = (
        "Could you share the account, order, error message, or exact action you want help with?"
    )
    final_answer = f"I need a little more information before I can safely help with: {query}"
    return {
        "pending_question": pending_question,
        "final_answer": final_answer,
        "messages": [f"clarify:{pending_question}"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = f"Review and approve this customer-impacting action before execution: {query}"
    return {
        "proposed_action": proposed_action,
        "messages": [f"risky_action:{proposed_action}"],
        "events": [make_event("risky_action", "completed", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return approval decision and audit event.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        payload = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "query": state.get("query"),
                "instruction": "Approve or reject this risky action.",
            }
        )
        approved = bool(payload.get("approved", False)) if isinstance(payload, dict) else False
        comment = str(payload.get("comment", "")) if isinstance(payload, dict) else ""
        reviewer = (
            str(payload.get("reviewer", "human-reviewer"))
            if isinstance(payload, dict)
            else "human-reviewer"
        )
    else:
        approved = True
        comment = "Mock approval for lab automation."
        reviewer = "mock-reviewer"

    decision = ApprovalDecision(approved=approved, reviewer=reviewer, comment=comment).model_dump()
    return {
        "approval": decision,
        "events": [
            make_event("approval", "completed", "approval decision recorded", approved=approved)
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0) or 0) + 1
    latest = (state.get("tool_results") or ["no tool result yet"])[-1]
    error = f"attempt {attempt}: retry scheduled after {latest}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "completed", "retry/fallback evaluated", attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 3) or 3)
    answer = (
        "I could not complete the request after the configured retry limit. "
        "The ticket has been moved to dead letter handling for manual support review."
    )
    return {
        "final_answer": answer,
        "errors": [f"dead_letter after {attempt}/{max_attempts} attempts"],
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route"),
                attempt=state.get("attempt", 0),
            )
        ]
    }
