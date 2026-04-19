from __future__ import annotations

import re

_LEADING_COMMENT_RE = re.compile(r"\A(?:\s+|--[^\n]*(?:\n|\Z)|/\*.*?\*/)+", re.DOTALL)
_MULTI_STATEMENT_RE = re.compile(r";\s*\S")
_FIRST_TOKEN_RE = re.compile(r"\A([a-zA-Z_][a-zA-Z0-9_]*)")
_MUTATION_KEYWORD_RE = re.compile(
    r"\b(?:insert|update|delete|merge|upsert|replace|alter|drop|create|truncate|grant|revoke|"
    r"comment|vacuum|analyze|attach|detach|copy|call|exec|execute|do|pragma)\b",
    re.IGNORECASE,
)
_ALLOWED_FIRST_TOKENS = {"select", "with"}


def _strip_leading_comments(sql: str) -> str:
    text = sql
    while True:
        match = _LEADING_COMMENT_RE.match(text)
        if match is None:
            return text.lstrip()
        text = text[match.end() :]


def _normalize_sql(sql: str) -> str:
    normalized = _strip_leading_comments(sql).strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def is_read_only_sql(sql: str) -> bool:
    normalized = _normalize_sql(sql)
    if not normalized:
        return False
    if _MULTI_STATEMENT_RE.search(normalized):
        return False

    first = _FIRST_TOKEN_RE.match(normalized)
    if first is None:
        return False
    if first.group(1).lower() not in _ALLOWED_FIRST_TOKENS:
        return False

    return _MUTATION_KEYWORD_RE.search(normalized) is None


def validate_raw_sql_statement(*, sql: str, allow_raw_sql: bool) -> None:
    if not allow_raw_sql:
        raise ValueError(
            "Raw sql= execution is disabled by default. "
            "Set allow_raw_sql=True and pass a read-only SELECT/WITH statement."
        )
    if not is_read_only_sql(sql):
        raise ValueError(
            "Raw sql= only supports single-statement read-only SELECT/WITH queries. "
            "Mutating or multi-statement SQL is blocked."
        )
