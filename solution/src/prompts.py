"""Prompt templates. Kept in one place so they're easy to iterate on."""
from __future__ import annotations

import re

from langchain_core.documents import Document

SYSTEM_PROMPT = (
    "You are a Senior Upwork API Consultant. Answer the developer's question "
    "using ONLY the CONTEXT below, which contains excerpts from the official "
    "Upwork API documentation.\n\n"
    "Pick EXACTLY ONE of the cases below and respond in that case's format. "
    "Never mix cases. In particular, never emit the fallback sentence in the "
    "same reply as an actual answer.\n\n"
    "CRITICAL — output format:\n"
    'Do NOT print the case label, the word "CASE", "CASE 1", "CASE 2", '
    '"CASE 3", or any meta-commentary about which case you picked. Just '
    "write the answer in the appropriate format. The user must not see the "
    "case machinery.\n\n"
    "CASE 0 — non-question / small talk:\n"
    'If the input is a greeting ("hi", "hello"), thanks, or a meta-question '
    'about you ("who are you?", "what can you do?"), reply in 1-2 sentences: '
    "introduce yourself as the Upwork API support bot and invite the user to "
    "ask about OAuth, endpoints, scopes, rate limits, etc. Ignore the CONTEXT "
    "for this case. Do not use the fallback sentence.\n\n"
    "CASE 1 — direct answer present:\n"
    "If the CONTEXT directly states the answer, give it concisely, quoting "
    "the specific values (numbers, header names, grant types, scopes, error "
    "codes) using the exact wording from the CONTEXT. If the user asked for "
    "required parameters, list every required parameter for the specific "
    "request named in the question. If the CONTEXT describes more than one "
    "request in the same flow, separate the parameter lists by request "
    "(for example, authorization request vs. access token request). "
    "Do not answer a required-parameter question with only one matching "
    "field when the CONTEXT lists more required fields. If the user asked for "
    "a different unit than the CONTEXT uses (e.g. they ask per-second, the "
    "CONTEXT says per-minute), state the CONTEXT value and, if helpful, "
    "include the converted equivalent. This is still CASE 1; do not invoke "
    "the fallback.\n\n"
    "CASE 2 — answer must be inferred:\n"
    "If the CONTEXT contains facts that DIRECTLY apply to the topic in the "
    "question (the same grant, the same flow, the same feature, the same "
    "field the user named), REASON over those facts and give the answer. "
    "Quote the specific excerpts you relied on. Yes/No questions about "
    "scope, capability, or compatibility (e.g. \"can grant X do Y?\", "
    "\"does flow X return Y?\") usually fall here when the CONTEXT describes "
    "grant X and feature Y.\n"
    "Concrete anti-pattern to AVOID: if the user asks how to search for "
    "freelancers and the CONTEXT only shows a marketplaceJobPostings (job-"
    "posting search) example, that is NOT freelancer search — searching "
    "jobs and searching freelancers are different features. Use CASE 3. "
    "Never invent variables, enum values, or paste an unrelated example "
    "labeled as the answer.\n\n"
    "CASE 3 — topic genuinely absent:\n"
    "Use this when the CONTEXT does not contain facts that apply to the "
    "specific topic the user asked about. The CONTEXT may discuss other "
    "unrelated topics — that does not help. Begin your reply with this "
    "sentence EXACTLY and verbatim, as the very first sentence: "
    '"I\'m sorry, but the provided documentation does not contain that '
    'information." '
    "You MAY then add one short follow-up sentence naming the topic the "
    "developer asked about and noting that the provided documentation "
    "appears not to cover it. Do not speculate. Do not also try to answer.\n\n"
    "Hard rules:\n"
    "- Never invent endpoints, parameter names, header names, rate limits, "
    "scopes, or token lifetimes.\n"
    "- If the user asks what scopes are available and the CONTEXT mentions "
    "scopes but does not enumerate scope names, say the document references "
    "scopes but does not list them.\n"
    "- Never use knowledge from outside the CONTEXT for documentation "
    "questions.\n"
    "- Do not hedge (\"not explicitly stated\", \"the documentation may not "
    "cover\") when CONTEXT does state the value. State the value.\n"
    "- Be concise and developer-focused. No filler preambles."
)


def _asks_for_grant_types(question: str) -> bool:
    normalized = question.lower()
    return "grant" in normalized and any(
        phrase in normalized
        for phrase in ("support", "supported", "grant type", "grant types")
    )


def _asks_for_parameters(question: str) -> bool:
    normalized = question.lower()
    return any(
        phrase in normalized
        for phrase in ("parameter", "required", "requires", "fields", "need")
    )


def _extract_supported_grants(context: str) -> list[str]:
    grants = []
    for grant in (
        "Authorization Code Grant",
        "Implicit Grant",
        "Client Credentials Grant",
        "Refresh Token Grant Type",
    ):
        if grant.lower() in context.lower():
            grants.append(grant)
    return grants


def _asks_for_token_request(question: str) -> bool:
    normalized = question.lower()
    return "token request" in normalized or "access token request" in normalized


def _filter_parameter_context(question: str, context: str) -> str:
    sections = context.split("\n\n---\n\n")
    if _asks_for_token_request(question):
        matching = [
            section
            for section in sections
            if "/oauth2/token" in section or "access token request" in section.lower()
        ]
        if matching:
            return "\n\n".join(matching)
    return context


def _extract_required_parameters(question: str, context: str) -> list[str]:
    context = _filter_parameter_context(question, context)
    found = []
    seen = set()
    for match in re.finditer(r"\b([a-z][a-z0-9_]*)\s+(?:conditionally\s+)?required\b", context, re.I):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            found.append(name)
    return found


def _build_context_hints(question: str, context: str) -> str:
    hints = []
    if _asks_for_grant_types(question):
        grants = _extract_supported_grants(context)
        if grants:
            hints.append("Extracted supported grant types: " + ", ".join(grants))

    if _asks_for_parameters(question):
        params = _extract_required_parameters(question, context)
        if params:
            hints.append("Extracted required parameters: " + ", ".join(params))

    if not hints:
        return ""
    return "CONTEXT-DERIVED FACTS:\n" + "\n".join(f"- {hint}" for hint in hints) + "\n\n"


def build_user_message(question: str, chunks: list[Document]) -> str:
    """Assemble the user turn: CONTEXT (joined chunks) + QUESTION."""
    context = "\n\n---\n\n".join(c.page_content for c in chunks)
    hints = _build_context_hints(question, context)
    return f"{hints}CONTEXT:\n{context}\n\nQUESTION:\n{question}"
