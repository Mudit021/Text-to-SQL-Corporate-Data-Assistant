"""Local, privacy-first MCA Text-to-SQL assistant.

Security model (in order of enforcement):
1. SQLite is opened through a ``file:...?mode=ro`` URI, so the OS/SQLite layer
   refuses writes even if every layer above it is compromised.
2. Every connection also sets ``PRAGMA query_only=ON`` as defence in depth.
3. A sqlglot guard accepts exactly one SELECT / CTE / set-operation statement,
   rejects DML and DDL, and wraps the statement in an outer LIMIT 15.
4. The raw user question is never interpolated into a SQL string. Only generated
   SQL is executed, and only after validation.
5. Results are capped by rows and characters, then summarized deterministically
   so a 0.5B model cannot invent numbers in the final answer.

Model design:
The default Falcon-H1 0.5B model is approximately 430 MB in Ollama. Models this
small cannot reliably follow LangChain's multi-step ReAct agent format, so this
app uses a simpler LangChain chain: generate one SQL query from the schema,
execute it through the guarded SQLDatabase, retry once with the SQLite error,
then summarize the returned rows deterministically.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import sqlglot
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sqlalchemy import create_engine, inspect as sqlalchemy_inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
from sqlglot import exp
from sqlglot.errors import ParseError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_VERSION = "2026-10-29-v3"
MAX_ROWS = 15                    # Non-negotiable hard cap on every SQL result.
MAX_STRING_LENGTH = 300          # Truncate individual text cells.
MAX_RESULT_CHARS = 12_000        # Secondary cap so 15 wide rows fit the context.
OLLAMA_NUM_CTX = 4096            # Modest context; Falcon-H1 needs only a short prompt.
OLLAMA_NUM_PREDICT = 220         # SQL responses are short.
MAX_SQL_ATTEMPTS = 3             # Initial generation plus two automatic repairs.

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "falcon-h1-0.5b:q6k")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Prefer the database currently used by the user; fall back to the mock DB.
_DB_CANDIDATES = ("company_database.sqlite", "company_database.db", "mca_master.db")
DEFAULT_DB_PATH = next(
    (
        Path(__file__).resolve().with_name(name)
        for name in _DB_CANDIDATES
        if Path(__file__).resolve().with_name(name).is_file()
    ),
    Path(__file__).resolve().with_name(_DB_CANDIDATES[0]),
)

# The prompt is schema-aware. Hints and examples are generated from the actual
# tables at startup, so the same app works with the HR database and the mock MCA
# database without hard-coded table names.
SQL_SYSTEM_PROMPT = """You translate business questions into SQLite SELECT queries.
Schema:
{schema}

Rules:
- Return exactly one SQLite SELECT statement. No markdown and no explanation.
- Only use the tables and columns in the schema. Never modify data.
- Use LOWER(column) = LOWER('value') for text filters.
- Never wrap a number in LOWER(), and never apply LOWER only to the literal.
- For highest/largest/top, use ORDER BY the numeric column DESC and LIMIT the requested number.
- Every non-aggregate selected column must appear in GROUP BY.
- When the question says "by X" or "per X", include X in SELECT and GROUP BY.
- For lists and groupings, include identifying fields such as a name, title, or date, not only numeric IDs.
- Never return only department_id when a human-readable department_name is available.
- Use LIMIT 15 for lists. Counts, totals, and averages do not need LIMIT.
- Aggregates are always allowed: COUNT, SUM, AVG, MIN, MAX, and GROUP BY are
  computed from existing rows and do not require a stored aggregate column.
- Never invent a column. Use only fields that exist in the schema.
- Dates are ISO text in YYYY-MM-DD format.
- SQLite dialect only: never use TOP, DATE_SUB, CURDATE, DATEADD, NOW(), ILIKE
  or PostgreSQL casts. Use LIMIT, date('now'), date('now','-1 year'), and LIKE.
- Relative dates must use date('now', ...); never hardcode a year for "last year" or "today".
- If a requested column belongs to a different table, use that table or JOIN
  the tables with their shared key.
- Age and tenure are not stored. Compute them from date_of_birth or hire_date
  with julianday(), for example
  AVG((julianday('now') - julianday(date_of_birth)) / 365.25).
  Never average the year part of a date.
- A department filter uses departments.department_name, never job_title.
- A skill filter uses employee_skills.skill; proficiency is the skill level.
- Percentages use 100.0 * numerator / NULLIF(denominator, 0).
- "no X", "without X", and "never X" mean a missing related row: use
  LEFT JOIN ... WHERE key IS NULL or NOT EXISTS. Do not test a numeric column = 0.
- "last month" for payroll means pay_month = (SELECT MAX(pay_month) FROM payroll).
- Use exact stored values from the value lists below in WHERE comparisons.
- Use a table alias or a bare column name. Never write alias.table.column.
- Use only the guidance and examples below when their tables/columns exist in the schema.

Schema-specific guidance:
{domain_hints}

Examples:
{examples}

Write SQL for the user's question.
"""

SQL_HUMAN_PROMPT = """Question: {question}

{repair_context}SQL:"""


# ---------------------------------------------------------------------------
# SQL safety guard
# ---------------------------------------------------------------------------

class UnsafeQueryError(ValueError):
    """Raised when generated SQL is not a single read-only SELECT."""


_TOP_LEVEL_READ_ONLY_TYPES = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_NESTED_WRITE_TYPES = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)


def guard_and_cap_sql(raw_sql: str, max_rows: int = MAX_ROWS) -> str:
    """Validate one generated SQL statement and enforce the row limit.

    This runs before every execution. The returned SQL is a subquery wrapper,
    which caps results even when the model writes ``LIMIT 10000`` or omits LIMIT.
    """
    if not isinstance(raw_sql, str) or not raw_sql.strip():
        raise UnsafeQueryError("Empty SQL is not allowed.")

    query = raw_sql.strip()

    # Some local models wrap SQL in a markdown fence despite instructions.
    # Removing the fence is safe; the content is still parsed and validated.
    if query.startswith("```"):
        query = re.sub(r"^```\s*(?:sql)?\s*", "", query, flags=re.IGNORECASE)
        if query.endswith("```"):
            query = query[:-3]
        query = query.strip()

    if not query:
        raise UnsafeQueryError("Empty SQL is not allowed.")

    try:
        statements = [stmt for stmt in sqlglot.parse(query, read="sqlite") if stmt is not None]
    except ParseError as exc:
        raise UnsafeQueryError(f"SQL syntax error: {exc}") from exc

    if len(statements) != 1:
        raise UnsafeQueryError("Exactly one SQL statement is allowed.")
    statement = statements[0]

    # Allow SELECT, set operations, and SELECTs with WITH/CTE prefixes only.
    if not isinstance(statement, _TOP_LEVEL_READ_ONLY_TYPES):
        raise UnsafeQueryError("Only read-only SELECT queries are allowed.")

    # Defence in depth: reject write nodes anywhere in the AST.
    for node_type in _NESTED_WRITE_TYPES:
        if statement.find(node_type) is not None:
            raise UnsafeQueryError("Data-changing SQL is blocked.")

    # Render from the validated AST to remove trailing semicolons and comments.
    safe_query = statement.sql(dialect="sqlite")

    limit = int(max_rows)
    if limit < 1:
        raise ValueError("max_rows must be at least 1.")

    return (
        "SELECT * FROM (\n"
        f"{safe_query}\n"
        f") AS _mca_capped_result\n"
        f"LIMIT {limit}"
    )


# ---------------------------------------------------------------------------
# Read-only database and LangChain SQLDatabase wrapper
# ---------------------------------------------------------------------------

def resolve_db_path(raw_path: str) -> Path:
    """Resolve relative paths against this project directory, not the CWD."""
    if not raw_path or not raw_path.strip():
        raise ValueError("Database path is empty.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def create_readonly_engine(db_path: Path) -> Engine:
    """Create a SQLAlchemy engine whose underlying SQLite connections are RO.

    ``mode=ro`` is SQLite's strict read-only file mode. ``PRAGMA query_only``
    is an additional per-connection guard. NullPool avoids connection reuse
    across Streamlit reruns or threads.
    """
    db_path = db_path.expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    # Percent-encode spaces and other URL-sensitive characters. Keep / and : so
    # Windows drive letters and POSIX paths both work with SQLite's URI parser.
    sqlite_uri = f"file:{quote(db_path.as_posix(), safe='/:')}?mode=ro"

    def connect_readonly() -> sqlite3.Connection:
        connection = sqlite3.connect(
            sqlite_uri,
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only = ON")
        return connection

    engine = create_engine(
        "sqlite://",
        creator=connect_readonly,
        poolclass=NullPool,
    )

    # Fail closed if the per-connection guard did not take effect.
    with engine.connect() as connection:
        query_only = connection.exec_driver_sql("PRAGMA query_only").scalar()
        if query_only != 1:
            engine.dispose()
            raise RuntimeError("Could not enable SQLite query_only mode.")

    return engine


class GuardedSQLDatabase(SQLDatabase):
    """LangChain SQLDatabase that validates and caps every executed query."""

    def __init__(self, *args: Any, max_rows: int = MAX_ROWS, **kwargs: Any) -> None:
        self.max_rows = int(max_rows)
        super().__init__(*args, **kwargs)

    def run(
        self,
        command: Any,
        fetch: str = "all",
        include_columns: bool = False,
        *,
        parameters: dict[str, Any] | None = None,
        execution_options: dict[str, Any] | None = None,
    ) -> Any:
        """Guard the SQL, then execute it through SQLDatabase's normal path."""
        guarded = guard_and_cap_sql(str(command), max_rows=self.max_rows)
        result = super().run(
            guarded,
            fetch=fetch,  # type: ignore[arg-type]
            include_columns=include_columns,
            parameters=parameters,
            execution_options=execution_options,
        )

        # Fifteen rows can still be wide. A second cap keeps pathological text
        # values from consuming the local model's small context window.
        if isinstance(result, str) and len(result) > MAX_RESULT_CHARS:
            result = (
                result[:MAX_RESULT_CHARS]
                + "\n... [result truncated to fit the model context]"
            )
        return result

    def run_no_throw(
        self,
        command: str,
        fetch: str = "all",
        include_columns: bool = False,
        *,
        parameters: dict[str, Any] | None = None,
        execution_options: dict[str, Any] | None = None,
    ) -> Any:
        """Return errors as strings. Used by tests and optional integrations."""
        try:
            return self.run(
                command,
                fetch=fetch,
                include_columns=include_columns,
                parameters=parameters,
                execution_options=execution_options,
            )
        except UnsafeQueryError as exc:
            return f"Error: {exc} Rewrite as one read-only SELECT statement."
        except SQLAlchemyError as exc:
            return f"Error: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Small-model SQL pipeline
# ---------------------------------------------------------------------------

def normalize_sqlite_dialect(text: str) -> str:
    """Fix the most common MySQL/SQL Server output mistakes from tiny models.

    This only rewrites read-only SELECT output before validation; the sqlglot
    guard still parses and approves the final statement.
    """
    # SELECT TOP 5 ... -> SELECT ... LIMIT 5 (SQLite has no TOP clause).
    top_match = re.search(r"(?i)\bSELECT\s+TOP\s+(\d+)\s+", text)
    if top_match:
        limit = int(top_match.group(1))
        text = text[: top_match.start()] + "SELECT " + text[top_match.end() :]
        if not re.search(r"\bLIMIT\b", text, flags=re.IGNORECASE):
            semicolon = re.search(r";\s*$", text)
            if semicolon:
                text = text[: semicolon.start()] + f" LIMIT {limit}" + text[semicolon.start() :]
            else:
                text = text + f" LIMIT {limit}"

    unit_map = {
        "year": "year", "years": "year",
        "month": "month", "months": "month",
        "week": "week", "weeks": "week",
        "day": "day", "days": "day",
        "hour": "hour", "hours": "hour",
        "minute": "minute", "minutes": "minute",
    }
    base_pattern = (
        r"(?:CURDATE\s*\(\s*\)|NOW\s*\(\s*\)|CURRENT_DATE|"
        r"[A-Za-z_][\w.]*|'[^']+'|\d{4}-\d{2}-\d{2})"
    )

    def _norm_base(base: str) -> str:
        if re.fullmatch(r"(?i)(?:CURDATE|NOW)\s*\(\s*\)|CURRENT_DATE", base):
            return "date('now')"
        return base

    def _shift(base: str, value: str, unit: str, sign: str) -> str:
        unit = unit_map.get(unit.lower(), unit.lower())
        base = _norm_base(base)
        if value in {"0", "+0", "-0"}:
            return base
        if sign == "sub":
            if not value.startswith("-"):
                value = f"-{value}"
        elif not value.startswith(("-", "+")):
            value = f"+{value}"
        if base == "date('now')":
            return f"date('now','{value} {unit}')"
        return f"date({base},'{value} {unit}')"

    def _date_sub(match: re.Match[str]) -> str:
        return _shift(match.group(1), match.group(2), match.group(3), "sub")

    def _date_add(match: re.Match[str]) -> str:
        return _shift(match.group(1), match.group(2), match.group(3), "add")

    def _dateadd(match: re.Match[str]) -> str:
        unit, value, base = match.group(1), match.group(2), match.group(3)
        return _shift(base, value, unit, "add" if not value.startswith("-") else "sub")

    text = re.sub(
        rf"(?i)\bDATE_SUB\s*\(\s*({base_pattern})\s*,\s*"
        r"INTERVAL\s+'?(-?\d+)'?\s+(\w+)\s*\)",
        _date_sub,
        text,
    )
    text = re.sub(
        rf"(?i)\bDATE_ADD\s*\(\s*({base_pattern})\s*,\s*"
        r"INTERVAL\s+'?(\d+)'?\s+(\w+)\s*\)",
        _date_add,
        text,
    )
    text = re.sub(
        r"(?i)\bDATEADD\s*\(\s*(\w+)\s*,\s*'?(-?\d+)'?\s*,\s*"
        rf"({base_pattern})\s*\)",
        _dateadd,
        text,
    )
    text = re.sub(
        r"(?i)\bDATEDIFF\s*\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)",
        r"(julianday(\1) - julianday(\2))",
        text,
    )
    text = re.sub(
        rf"(?i)\bDATE_FORMAT\s*\(\s*({base_pattern})\s*,\s*'([^']*)'\s*\)",
        r"strftime('\2', \1)",
        text,
    )
    extract_map = {"year": "%Y", "month": "%m", "day": "%d", "hour": "%H", "minute": "%M"}
    for unit, fmt in extract_map.items():
        text = re.sub(
            rf"(?i)\bEXTRACT\s*\(\s*{unit}\s+FROM\s+([^)]+)\)",
            rf"strftime('{fmt}', \1)",
            text,
        )
    # MySQL YEAR(expr)/MONTH(expr)/DAY(expr) -> SQLite strftime.
    for function, fmt in (("YEAR", "%Y"), ("MONTH", "%m"), ("DAY", "%d")):
        text = re.sub(
            rf"(?i)\b{function}\s*\(\s*([^()]+)\s*\)",
            rf"CAST(strftime('{fmt}', \1) AS INTEGER)",
            text,
        )
    text = re.sub(r"(?i)\bISNULL\s*\(", "IFNULL(", text)
    text = re.sub(r"(?i)\bCURDATE\s*\(\s*\)", "date('now')", text)
    text = re.sub(r"(?i)\bNOW\s*\(\s*\)", "datetime('now')", text)
    text = re.sub(r"(?i)\bILIKE\b", "LIKE", text)
    return text


def _strip_trailing_prose(text: str) -> str:
    """Remove explanations that tiny models append after their SQL.

    qwen/Falcon models often emit ``... GROUP BY 1;  Explanation: - We count ...``.
    The prose makes sqlglot fail with a tokenizing error even though the SQL is
    usable, so cut at the first real statement terminator or prose marker.
    """
    in_quote: str | None = None
    cut = len(text)
    for index, char in enumerate(text):
        if in_quote:
            if char == in_quote:
                in_quote = None
        elif char in "'\"`":
            in_quote = char
        elif char == ";":
            cut = index + 1
            break
    text = text[:cut].strip()

    # Inline or block prose after the query (no semicolon was generated).
    text = re.split(
        r"(?im)^\s*(?:Explanation|Explain|Note|Why|This query|The query|The above|Here|We use|We count|We join)\b.*$",
        text,
        maxsplit=1,
    )[0]
    text = re.split(
        r"(?is)\s+(?:Explanation|Note)\s*:",
        text,
        maxsplit=1,
    )[0]

    # A prose bullet list is not SQL; stop before the first bullet line.
    lines: list[str] = []
    for line in text.splitlines():
        if lines and re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def clean_generated_sql(raw_output: Any) -> str:
    """Extract SQL from a small model's response without executing anything."""
    text = str(raw_output).strip()

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        text = re.sub(r"^```\s*(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    text = re.sub(r"^\s*(?:SQLQuery|SQL|Query)\s*:\s*", "", text, flags=re.IGNORECASE)

    # Prefer SQL that starts a line. If the model prefixed prose on the same
    # line, require a SELECT ... FROM pattern instead of cutting at prose that
    # merely contains the word "with" or "select".
    match = re.search(r"(?im)^\s*(?:SELECT|WITH)\b", text)
    if not match:
        match = re.search(r"(?is)\bSELECT\b.*?\bFROM\b", text)
    if match and match.start() > 0:
        text = text[match.start():]

    text = _strip_trailing_prose(text)
    return normalize_sqlite_dialect(text.strip())


def _short_error(exc: Exception, limit: int = 700) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:limit]


@dataclass
class AnswerResult:
    answer: str
    queries: list[str]
    rows: list[dict[str, Any]]


class SQLPipelineError(RuntimeError):
    """Raised when both the initial SQL and the repair attempt fail."""

    def __init__(self, message: str, queries: list[str]) -> None:
        super().__init__(message)
        self.queries = queries


def parse_result_rows(raw_result: Any) -> list[dict[str, Any]]:
    """Convert SQLDatabase's string output into a list of row dictionaries."""
    if not isinstance(raw_result, str) or not raw_result.strip():
        return []
    try:
        parsed = ast.literal_eval(raw_result)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [dict(row) for row in parsed if isinstance(row, dict)]


def _humanize_column(name: Any) -> str:
    text = str(name)
    text = re.sub(r"\(([^)]*)\)", r" \1", text)
    text = text.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.title() if text else "Result"


def _format_value(column: Any, value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        key = str(column).lower()
        if any(word in key for word in ("capital", "amount", "revenue", "turnover")):
            return f"₹{value:,.2f}"
        if isinstance(value, float) and value.is_integer():
            return f"{int(value):,}"
        return f"{value:,}"
    return str(value)


def summarize_rows(rows: list[dict[str, Any]]) -> str:
    """Create a faithful, concise summary without another model generation.

    A 0.5B model is prone to inventing names and numbers when asked to summarize.
    Rendering the known columns/values directly keeps the answer trustworthy.
    """
    if not rows:
        return "No matching records were found."

    if len(rows) == 1:
        row = rows[0]
        if len(row) == 1:
            column, value = next(iter(row.items()))
            column_lower = str(column).lower()
            if "count" in column_lower and isinstance(value, (int, float)):
                count = int(value)
                noun = "record" if count == 1 else "records"
                return f"The query returned {count:,} matching {noun}."
            return f"{_humanize_column(column)}: {_format_value(column, value)}."

        parts = [
            f"{_humanize_column(column)}: {_format_value(column, value)}"
            for column, value in row.items()
        ]
        return "Result: " + "; ".join(parts) + "."

    lines = [f"The query returned {len(rows)} rows. Top results:"]
    for row in rows[:5]:
        parts = [
            f"{_humanize_column(column)}: {_format_value(column, value)}"
            for column, value in row.items()
        ]
        lines.append("- " + "; ".join(parts))
    if len(rows) > 5:
        lines.append(f"- ... and {len(rows) - 5} more rows in the results table.")
    return "\n".join(lines)


class LocalSQLAssistant:
    """Generate SQL, execute through the guarded DB, and repair on error."""

    def __init__(
        self,
        db: GuardedSQLDatabase,
        llm: ChatOllama,
        schema: str,
    ) -> None:
        self.db = db
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SQL_SYSTEM_PROMPT),
                ("human", SQL_HUMAN_PROMPT),
            ]
        )
        self.sql_chain = prompt | llm
        self.full_schema = schema

        # Cache the real column names. Repair attempts remind the tiny model what
        # exists, and final failures can explain the missing field deterministically.
        self.table_columns: dict[str, list[str]] = {}
        self.table_schemas: dict[str, str] = {}
        # Undirected join graph built from the database's foreign keys. It lets
        # schema selection add the bridge table (usually employees) instead of
        # giving a 0.5B model two unrelated tables and letting it guess a join.
        self.table_neighbors: dict[str, set[str]] = {}
        try:
            inspector = sqlalchemy_inspect(db._engine)
            for table in db.get_usable_table_names():
                self.table_columns[table] = [
                    str(column["name"]) for column in inspector.get_columns(table)
                ]
                self.table_schemas[table] = db.get_table_info([table])
            self.table_neighbors = {table: set() for table in self.table_columns}
            for table in self.table_columns:
                for foreign_key in inspector.get_foreign_keys(table):
                    referred = foreign_key.get("referred_table")
                    if referred and referred in self.table_neighbors:
                        self.table_neighbors[table].add(referred)
                        self.table_neighbors[referred].add(table)
        except Exception:
            self.table_columns = {}
            self.table_schemas = {}
            self.table_neighbors = {}

        self.value_hints = self._build_value_hints()
        self.date_coverage = self._build_date_coverage()
        self.domain_hints, self.example_blocks = self._build_guidance()

    _ENUM_COLUMNS = (
        ("departments", "department_name"),
        ("employees", "employment_status"),
        ("employees", "employment_type"),
        ("employees", "work_mode"),
        ("employees", "gender"),
        ("attendance", "attendance_status"),
        ("projects", "project_status"),
        ("payroll", "payment_status"),
        ("performance_reviews", "promotion_recommended"),
        ("employee_skills", "skill"),
        ("employee_skills", "proficiency"),
    )

    def _read_distinct(self, table: str, column: str) -> list[str]:
        """Read a small list of stored filter values without exposing rows.

        Knowing that the department is literally "IT & Security" prevents the
        model from splitting it into ('IT', 'Security') and returning zero rows.
        """
        if column not in self.table_columns.get(table, []):
            return []
        query = (
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT 26'
        )
        try:
            with self.db._engine.connect() as connection:
                rows = connection.exec_driver_sql(query).fetchall()
        except Exception:
            return []
        values = [str(row[0]).strip() for row in rows if str(row[0]).strip()]
        # More than 25 values is a free-text column, not a filter list.
        return values if len(values) <= 25 else []

    def _build_value_hints(self) -> dict[tuple[str, str], list[str]]:
        hints: dict[tuple[str, str], list[str]] = {}
        for table, column in self._ENUM_COLUMNS:
            values = self._read_distinct(table, column)
            if values:
                hints[(table, column)] = values
        return hints

    def _build_date_coverage(self) -> dict[str, tuple[str, str]]:
        """Cache the date range of the time-series tables.

        Used to explain an empty "today" result instead of leaving the user to
        wonder whether the query or the data is broken.
        """
        coverage: dict[str, tuple[str, str]] = {}
        for table, column in (
            ("attendance", "attendance_date"),
            ("payroll", "pay_month"),
        ):
            if column not in self.table_columns.get(table, []):
                continue
            try:
                with self.db._engine.connect() as connection:
                    row = connection.exec_driver_sql(
                        f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}"'
                    ).fetchone()
            except Exception:
                continue
            if row and row[0] and row[1]:
                coverage[table] = (str(row[0]), str(row[1]))
        return coverage

    def _coverage_note(self, sql: str) -> str:
        """Explain empty results for date-filtered attendance or payroll."""
        lowered = sql.lower()
        notes: list[str] = []
        for table, column in (
            ("attendance", "attendance_date"),
            ("payroll", "pay_month"),
        ):
            if table not in lowered or column not in lowered:
                continue
            coverage = self.date_coverage.get(table)
            if coverage:
                notes.append(
                    f"The {table} table only contains {column} values from "
                    f"{coverage[0]} to {coverage[1]}."
                )
        return " ".join(notes)

    def _build_guidance(self) -> tuple[str, list[tuple[str, str]]]:
        """Build compact, schema-aware hints and examples from real columns."""
        hints: list[str] = []
        examples: list[tuple[str, str]] = []
        columns = {
            table: {column.lower() for column in table_columns}
            for table, table_columns in self.table_columns.items()
        }

        def has(table: str, column: str) -> bool:
            return column.lower() in columns.get(table, set())

        def add_example(question: str, sql: str) -> None:
            examples.append((question, sql))

        if has("employees", "employment_status"):
            hints.append(
                "Employee status uses employees.employment_status (values include "
                "Active, On Leave, Resigned). There is no employees.status column."
            )
            add_example(
                "How many active employees are there?",
                "SELECT COUNT(*) AS active_employees FROM employees "
                "WHERE LOWER(employment_status) = LOWER('Active');",
            )

        if has("employees", "work_mode"):
            hints.append(
                "Remote/office/hybrid questions use employees.work_mode, not "
                "employment_type."
            )
            add_example(
                "How many employees work remotely?",
                "SELECT COUNT(*) AS remote_employees FROM employees "
                "WHERE LOWER(work_mode) = LOWER('Remote');",
            )

        if has("employees", "bonus_percentage") and has("departments", "department_name"):
            add_example(
                "What is the average bonus by department?",
                "SELECT d.department_name, AVG(e.bonus_percentage) AS average_bonus "
                "FROM employees AS e JOIN departments AS d "
                "ON e.department_id = d.department_id "
                "GROUP BY d.department_name ORDER BY average_bonus DESC LIMIT 15;",
            )

        if has("performance_reviews", "review_date") and has("employees", "full_name"):
            hints.append(
                "\"latest review\" is per employee: use review_date = (SELECT "
                "MAX(review_date) FROM performance_reviews p2 WHERE "
                "p2.employee_id = e.employee_id). Do not compare review_id to the "
                "single global MAX(review_id)."
            )
            add_example(
                "List employees recommended for promotion in their latest review.",
                "SELECT e.full_name, pr.rating FROM employees AS e "
                "JOIN performance_reviews AS pr ON e.employee_id = pr.employee_id "
                "WHERE pr.promotion_recommended = 1 AND pr.review_date = "
                "(SELECT MAX(review_date) FROM performance_reviews AS p2 "
                "WHERE p2.employee_id = e.employee_id) LIMIT 15;",
            )

        if has("employees", "salary") and (has("employees", "full_name") or has("employees", "first_name")):
            name_col = "full_name" if has("employees", "full_name") else "first_name"
            add_example(
                "List the top 5 highest-paid employees.",
                f"SELECT {name_col}, salary FROM employees "
                "ORDER BY salary DESC LIMIT 5;",
            )

        if has("employees", "hire_date"):
            hints.append(
                "For last year/this year/today, use SQLite relative dates such as "
                "date('now','-1 year') and date('now'). Never hardcode years for relative questions."
            )
            add_example(
                "How many employees were hired last year?",
                "SELECT COUNT(*) AS hired_last_year FROM employees "
                "WHERE hire_date >= date('now','-1 year') AND hire_date < date('now');",
            )

        if has("attendance", "attendance_date") and has("attendance", "attendance_status"):
            hints.append(
                "Attendance and absence questions use attendance.attendance_date "
                "and attendance.attendance_status (Present, Absent, Leave, Work From Home)."
            )
            add_example(
                "How many employees were absent today?",
                "SELECT COUNT(*) AS absent_today FROM attendance "
                "WHERE attendance_date = date('now') "
                "AND LOWER(attendance_status) = LOWER('Absent');",
            )

        if has("projects", "project_status"):
            hints.append(
                "Project state is projects.project_status. There is no Delayed status; "
                "treat On Hold as delayed. For overdue, use end_date < date('now')."
            )
            add_example(
                "Which projects are delayed?",
                "SELECT project_name, end_date, project_status FROM projects "
                "WHERE LOWER(project_status) = LOWER('On Hold') LIMIT 15;",
            )

        if has("payroll", "pay_month") and has("payroll", "net_pay"):
            hints.append(
                "Payroll totals by month use payroll.pay_month and SUM(net_pay); "
                "include pay_month in SELECT and GROUP BY. Salary and department are in employees."
            )
            add_example(
                "What is the total net pay by pay month?",
                "SELECT pay_month, SUM(net_pay) AS total_net_pay FROM payroll "
                "GROUP BY pay_month ORDER BY pay_month LIMIT 15;",
            )

        if has("employees", "gender"):
            hints.append(
                "A gender breakdown needs conditional COUNT/SUM expressions so all "
                "categories appear in one row."
            )
            add_example(
                "How many male and female employees are there?",
                "SELECT "
                "SUM(CASE WHEN LOWER(gender) = 'male' THEN 1 ELSE 0 END) AS male_count, "
                "SUM(CASE WHEN LOWER(gender) = 'female' THEN 1 ELSE 0 END) AS female_count, "
                "SUM(CASE WHEN LOWER(gender) NOT IN ('male','female') THEN 1 ELSE 0 END) AS other_count "
                "FROM employees;",
            )

        if (
            has("employees", "salary")
            and has("employees", "department_id")
            and has("departments", "department_id")
            and has("departments", "department_name")
        ):
            hints.append(
                "Salary is employees.salary. For any by-department question, join "
                "employees.department_id to departments.department_id and display "
                "departments.department_name, not just department_id."
            )
            add_example(
                "What is the average salary by department?",
                "SELECT d.department_name, AVG(e.salary) AS average_salary "
                "FROM employees AS e JOIN departments AS d "
                "ON e.department_id = d.department_id "
                "GROUP BY d.department_name ORDER BY average_salary DESC LIMIT 15;",
            )
            add_example(
                "What is the average years of experience by department?",
                "SELECT d.department_name, AVG(e.years_experience) AS average_years "
                "FROM employees AS e JOIN departments AS d "
                "ON e.department_id = d.department_id "
                "GROUP BY d.department_name ORDER BY average_years DESC LIMIT 15;",
            )

        if has("payroll", "employee_id") and has("employees", "employee_id"):
            hints.append(
                "Join payroll.employee_id to employees.employee_id when a question "
                "needs both employee details and payroll amounts."
            )
        if has("employee_projects", "project_id") and has("projects", "project_id"):
            hints.append(
                "Join employee_projects.project_id to projects.project_id to show "
                "project names alongside allocations. Highest allocation uses "
                "employee_projects.allocation_percent, not role_on_project."
            )
            add_example(
                "Which project has the highest allocation percent?",
                "SELECT p.project_name, ep.allocation_percent "
                "FROM employee_projects AS ep JOIN projects AS p "
                "ON ep.project_id = p.project_id "
                "ORDER BY ep.allocation_percent DESC LIMIT 1;",
            )

        if has("employees", "date_of_birth"):
            hints.append(
                "Age is not stored. Use julianday: "
                "(julianday('now') - julianday(date_of_birth)) / 365.25, and never "
                "AVG(strftime('%Y', date_of_birth))."
            )
            add_example(
                "What is the average age of employees?",
                "SELECT AVG((julianday('now') - julianday(date_of_birth)) / 365.25) "
                "AS average_age FROM employees;",
            )

        if has("employees", "years_experience"):
            add_example(
                "Which employees have more than 10 years of experience?",
                "SELECT full_name, job_title, years_experience FROM employees "
                "WHERE years_experience > 10 ORDER BY years_experience DESC LIMIT 15;",
            )

        if (
            has("employees", "job_title")
            and has("employees", "department_id")
            and has("departments", "department_name")
        ):
            hints.append(
                "A department name in a question is a filter on "
                "departments.department_name (JOIN on department_id). It is never "
                "a job_title value."
            )
            add_example(
                "Who is the highest paid employee in Sales?",
                "SELECT e.full_name, e.salary FROM employees AS e "
                "JOIN departments AS d ON e.department_id = d.department_id "
                "WHERE d.department_name = 'Sales' ORDER BY e.salary DESC LIMIT 1;",
            )

        if has("employee_skills", "skill"):
            hints.append(
                "A skill such as Python, SQL, or Excel is filtered with "
                "employee_skills.skill = 'Python'. proficiency is a level "
                "(Beginner/Intermediate/Advanced/Expert), not a skill name."
            )
            add_example(
                "How many employees have Python skills?",
                "SELECT COUNT(DISTINCT employee_id) AS python_employees "
                "FROM employee_skills WHERE LOWER(skill) = LOWER('Python');",
            )

        if has("employees", "employment_status"):
            hints.append(
                "A percentage of employees uses "
                "ROUND(100.0 * SUM(CASE WHEN condition THEN 1 ELSE 0 END) / "
                "NULLIF(COUNT(*), 0), 2)."
            )
            add_example(
                "What percentage of employees are on leave?",
                "SELECT ROUND(100.0 * SUM(CASE WHEN LOWER(employment_status) = "
                "'on leave' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) "
                "AS leave_percentage FROM employees;",
            )

        if (
            has("employees", "employee_id")
            and has("employee_projects", "employee_id")
            and has("employee_projects", "project_id")
        ):
            hints.append(
                "\"no X\", \"without X\", and \"never X\" mean no related row: use "
                "LEFT JOIN ... WHERE related_key IS NULL, not a numeric column = 0."
            )
            add_example(
                "How many employees have no project allocation?",
                "SELECT COUNT(*) AS employees_without_project FROM employees AS e "
                "LEFT JOIN employee_projects AS ep "
                "ON e.employee_id = ep.employee_id "
                "WHERE ep.employee_id IS NULL;",
            )

        if has("employees", "manager_id") and has("employees", "employee_id"):
            hints.append(
                "A manager is employees.manager_id pointing back to "
                "employees.employee_id. To show manager names or direct-report "
                "counts, self-join: employees AS e JOIN employees AS m "
                "ON e.manager_id = m.employee_id."
            )
            add_example(
                "Which manager has the most direct reports?",
                "SELECT m.full_name AS manager_name, COUNT(e.employee_id) AS reports "
                "FROM employees AS e JOIN employees AS m "
                "ON e.manager_id = m.employee_id "
                "GROUP BY m.employee_id, m.full_name "
                "ORDER BY reports DESC LIMIT 1;",
            )

        if has("performance_reviews", "promotion_recommended"):
            hints.append(
                "Promotion questions use performance_reviews.promotion_recommended, "
                "where 1 means recommended and 0 means not recommended."
            )
            add_example(
                "How many employees were recommended for promotion?",
                "SELECT COUNT(DISTINCT employee_id) AS promoted_employees "
                "FROM performance_reviews WHERE promotion_recommended = 1;",
            )

        if has("payroll", "pay_month"):
            hints.append(
                "\"last month\" or \"latest month\" for payroll means "
                "pay_month = (SELECT MAX(pay_month) FROM payroll). payment_date is "
                "a day in that month, so never compare it to date('now')."
            )
            add_example(
                "How much tax did we pay last month?",
                "SELECT SUM(tax) AS total_tax FROM payroll "
                "WHERE pay_month = (SELECT MAX(pay_month) FROM payroll);",
            )

        if (
            has("performance_reviews", "rating")
            and has("employees", "department_id")
            and has("departments", "department_name")
        ):
            hints.append(
                "performance_reviews has no department_id. Join "
                "performance_reviews.employee_id to employees.employee_id, then "
                "employees.department_id to departments.department_id."
            )
            add_example(
                "Which department has the best average performance rating?",
                "SELECT d.department_name, AVG(pr.rating) AS average_rating "
                "FROM performance_reviews AS pr "
                "JOIN employees AS e ON pr.employee_id = e.employee_id "
                "JOIN departments AS d ON e.department_id = d.department_id "
                "GROUP BY d.department_name "
                "ORDER BY average_rating DESC LIMIT 1;",
            )

        if (
            has("payroll", "net_pay")
            and has("employees", "department_id")
            and has("departments", "department_name")
        ):
            hints.append(
                "payroll has no department_id. For payroll by department, join "
                "payroll.employee_id to employees.employee_id, then "
                "employees.department_id to departments.department_id."
            )
            add_example(
                "What is the total payroll cost by department?",
                "SELECT d.department_name, SUM(p.net_pay) AS total_payroll "
                "FROM payroll AS p "
                "JOIN employees AS e ON p.employee_id = e.employee_id "
                "JOIN departments AS d ON e.department_id = d.department_id "
                "GROUP BY d.department_name "
                "ORDER BY total_payroll DESC LIMIT 15;",
            )

        if has("attendance", "attendance_date"):
            hints.append(
                "Attendance trends group by strftime('%Y-%m', attendance_date); "
                "attendance data only exists for the dates present in the table."
            )

        if self.value_hints:
            value_lines = []
            for (table, column), values in self.value_hints.items():
                shown = ", ".join(values[:20])
                value_lines.append(f"- {table}.{column}: {shown}")
            hints.append(
                "Stored filter values (use these exact strings):\n"
                + "\n".join(value_lines)
            )

        if "employees" in self.table_columns and "companies" not in self.table_columns:
            hints.append(
                "This is a workforce database. Company/headcount/staff questions "
                "refer to employees, not payroll rows."
            )

        # Schema-agnostic fallback examples keep other databases usable too.
        generic_table = next(iter(self.table_columns), None)
        if generic_table:
            generic_columns = self.table_columns[generic_table]
            if not examples:
                add_example(
                    f"How many {generic_table} records are there?",
                    f"SELECT COUNT(*) AS total_count FROM {generic_table};",
                )
            if len(examples) < 4:
                display = ", ".join(f'"{column}"' for column in generic_columns[:4])
                add_example(
                    f"Show 5 {generic_table} records.",
                    f"SELECT {display} FROM {generic_table} LIMIT 5;",
                )

        if not hints:
            hints.append(
                "Use only columns listed in the schema and include category columns "
                "when the question says by/per that category."
            )

        return (
            "\n".join(f"- {hint}" for hint in hints),
            examples,
        )

    _STOPWORDS = {
        "the", "and", "for", "are", "was", "were", "how", "what", "which",
        "who", "when", "where", "why", "does", "with", "from", "that",
        "this", "these", "those", "their", "there", "into", "about", "please",
        "show", "give", "tell", "find", "get", "list", "many", "much", "each",
        "every", "all", "any", "some", "more", "than", "then", "also",
    }

    @staticmethod
    def _schema_tokens(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        normalized: set[str] = set()
        for token in tokens:
            normalized.add(token)
            if len(token) > 4 and token.endswith("ies"):
                normalized.add(token[:-3] + "y")
            elif len(token) > 3 and token.endswith("s"):
                normalized.add(token[:-1])
            if len(token) > 4 and token.endswith("ed"):
                # hired -> hire (and hir) so it can match hire_date.
                normalized.add(token[:-1])
                normalized.add(token[:-2])
            if len(token) > 5 and token.endswith("ing"):
                normalized.add(token[:-3])
            normalized.discard("")
        return normalized

    def _join_path(self, start: str, target: str) -> list[str]:
        """Shortest chain of foreign-key-linked tables between two tables."""
        previous: dict[str, str | None] = {start: None}
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current == target:
                break
            for neighbor in self.table_neighbors.get(current, set()):
                if neighbor not in previous:
                    previous[neighbor] = current
                    queue.append(neighbor)
        if target not in previous:
            return []
        path: list[str] = []
        current: str | None = target
        while current is not None:
            path.append(current)
            current = previous[current]
        return list(reversed(path))

    def _connect_tables(self, selected: list[str]) -> list[str]:
        """Add bridge tables so every selected table can be joined."""
        selected = [table for table in dict.fromkeys(selected) if table in self.table_neighbors]
        if len(selected) < 2 or not self.table_neighbors:
            return selected
        connected = [selected[0]]
        for table in selected[1:]:
            for anchor in connected:
                path = self._join_path(anchor, table)
                if path:
                    for node in path[1:]:
                        if node not in connected:
                            connected.append(node)
                    break
            else:
                connected.append(table)
        return connected

    def _canonicalize_literals(self, sql: str) -> str:
        """Replace filter literals with the database's exact stored casing.

        The model writes 'contract' while the table stores 'Contract'; SQLite's
        = is case-sensitive, so that otherwise correct query returns zero rows.
        """
        if not self.value_hints:
            return sql
        exact = {
            value.lower(): value
            for values in self.value_hints.values()
            for value in values
        }

        # Never touch literals that are already inside LOWER(...). Changing
        # LOWER(status) = 'on leave' to 'On Leave' would make it false.
        protected_spans: list[tuple[int, int]] = []
        for pattern in (
            r"(?is)lower\s*\([^()]*\)\s*(?:=|<>|!=|like)\s*'[^']*'",
            r"(?is)lower\s*\([^()]*\)\s+in\s*\([^)]*\)",
            r"(?is)lower\s*\(\s*'[^']*'\s*\)",
        ):
            protected_spans.extend(
                (match.start(), match.end()) for match in re.finditer(pattern, sql)
            )

        def replace_literal(match: re.Match[str]) -> str:
            if any(start <= match.start() < end for start, end in protected_spans):
                return match.group(0)
            literal = match.group(1)
            stored = exact.get(literal.lower())
            if stored is None:
                return match.group(0)
            return "'" + stored.replace("'", "''") + "'"

        return re.sub(r"'((?:[^']|'')*)'", replace_literal, sql)

    def select_schema(self, question: str) -> tuple[list[str], str]:
        """Return only the tables that are probably relevant to a question.

        Giving an 8-table schema to a 0.5B model makes it pick unrelated fields.
        A small lexical score keeps the prompt short and focused.
        """
        if not self.table_schemas:
            return [], self.full_schema

        question_tokens = {
            token
            for token in self._schema_tokens(question)
            if len(token) >= 3 and token not in self._STOPWORDS
        }
        if not question_tokens:
            return list(self.table_schemas), self.full_schema

        scores: dict[str, int] = {}
        for table, columns in self.table_columns.items():
            table_label = table.lower()
            table_singular = table_label[:-1] if table_label.endswith("s") else table_label
            table_tokens = self._schema_tokens(table)
            column_tokens = {
                token
                for column in columns
                for token in self._schema_tokens(column)
            }
            score = 0
            for question_token in question_tokens:
                # An exact table-name hit is much stronger than a shared word
                # like "employee" inside employee_skills.
                if question_token in (table_label, table_singular):
                    score += 8
                elif question_token in table_tokens:
                    score += 3

                if question_token in column_tokens:
                    score += 6
                elif len(question_token) >= 4 and any(
                    question_token in column_token or column_token in question_token
                    for column_token in column_tokens
                    if len(column_token) >= 4
                ):
                    score += 3
                else:
                    best = max(
                        (
                            difflib.SequenceMatcher(None, question_token, candidate).ratio()
                            for candidate in column_tokens
                        ),
                        default=0.0,
                    )
                    if best >= 0.74:
                        score += 2
            scores[table] = score

        # Explicit intent terms bridge the vocabulary gap between a question and
        # a schema (for example "absent" lives in attendance_status, not a column).
        lowered = question.lower()
        forced: list[str] = []
        if "employees" in self.table_columns:
            if re.search(r"\b(company|companies|staff|workforce|headcount|personnel|people)\b", lowered):
                forced.append("employees")
        if "attendance" in self.table_columns:
            if re.search(r"\b(absent|absence|attendance|present|leave|check[- ]?in|check[- ]?out)\b", lowered):
                forced.append("attendance")
        if "projects" in self.table_columns:
            if re.search(r"\b(delay|delayed|overdue|on hold|project status|projects?)\b", lowered):
                forced.append("projects")
        if "employee_projects" in self.table_columns:
            if re.search(r"\b(allocation|allocated|assigned|role on project)\b", lowered):
                forced.extend(["employee_projects", "projects"])

        best_score = max(scores.values(), default=0)
        if best_score <= 0:
            selected = list(dict.fromkeys(forced or list(self.table_schemas)))
        else:
            threshold = max(2, int(best_score * 0.6))
            selected = [
                table
                for table, score in sorted(scores.items(), key=lambda item: -item[1])
                if score >= threshold
            ][:3]
            if not selected:
                selected = [max(scores, key=scores.get)]
            selected = list(dict.fromkeys([*forced, *selected]))

        # A project-status question only needs projects. Joining employee_projects
        # here encourages a 0.5B model to invent aliases and fail.
        if (
            "projects" in self.table_columns
            and re.search(r"\b(delay|delayed|overdue|on hold|project status)\b", lowered)
            and not re.search(r"\b(allocation|assigned|employee|team)\b", lowered)
        ):
            selected = ["projects"]

        selected = [table for table in selected if table in self.table_schemas][:4]
        selected = self._connect_tables(selected)[:5]
        schema_text = "\n\n".join(
            self.table_schemas[table] for table in selected
        )
        return selected, schema_text or self.full_schema

    def schema_note(self, tables: list[str] | None = None) -> str:
        names = tables if tables else list(self.table_columns)
        lines = [
            f"- {table}: {', '.join(self.table_columns[table])}"
            for table in names
            if table in self.table_columns
        ]
        if not lines:
            return ""
        return (
            "Available tables and columns:\n"
            + "\n".join(lines)
            + "\nUse the table that actually contains each field, or JOIN tables."
        )

    @staticmethod
    def _alias_map(sql: str) -> dict[str, str]:
        """Map table aliases to real table names in a SELECT."""
        aliases: dict[str, str] = {}
        reserved = {"where", "join", "left", "inner", "outer", "group", "order", "limit", "on"}
        for table, alias in re.findall(
            r"(?i)\b(?:from|join)\s+[\"`]?(\w+)[\"`]?\s+(?:as\s+)?([a-z_]\w*)",
            sql,
        ):
            if alias.lower() not in reserved:
                aliases[alias] = table
        return aliases

    def _repair_hint(self, sql: str, error: str) -> str:
        """Add explicit SQLite fixes for dialect mistakes made by tiny models."""
        combined = f"{sql}\n{error}"
        hints: list[str] = []
        if re.search(r"\bTOP\s+\d+", combined, flags=re.IGNORECASE):
            hints.append(
                "SQLite has no TOP clause. Keep the column list and add "
                "LIMIT n after ORDER BY."
            )
        if re.search(
            r"\b(?:DATE_SUB|DATE_ADD|CURDATE|DATEADD|DATE_TRUNC|NOW)\s*\(",
            combined,
            flags=re.IGNORECASE,
        ):
            hints.append(
                "Use SQLite date functions instead, for example "
                "date('now') or date('now','-1 year')."
            )
        if re.search(r"\bILIKE\b|\s::\s", combined, flags=re.IGNORECASE):
            hints.append("Use plain LIKE and SQLite casts only.")

        missing = self._missing_from_schema(error)
        if missing:
            kind, identifier = missing
            if kind == "column":
                locations = [
                    f"`{table}`.`{column}`"
                    for table, columns in self.table_columns.items()
                    for column in columns
                    if column.lower() == identifier.lower()
                ]
                if locations:
                    owning_table = locations[0].split(".")[0]
                    aliases = self._alias_map(sql)
                    owning_alias = next(
                        (alias for alias, table in aliases.items() if table == owning_table),
                        owning_table,
                    )
                    hints.append(
                        f"`{identifier}` exists in {', '.join(locations)}. Write "
                        f"`{owning_alias}.{identifier}` (or bare `{identifier}`), not "
                        "another table's alias."
                    )
                else:
                    hints.append(
                        f"`{identifier}` does not exist. Use only the fields listed "
                        "above, or answer NO_SQL: the schema lacks that field."
                    )
            elif identifier in self.table_schemas:
                hints.append(
                    f"The table `{identifier}` exists; the schema has been expanded "
                    "to include it. Use it in the corrected query."
                )

        if not hints:
            return ""
        return "SQLite hints:\n" + "\n".join(f"- {hint}" for hint in hints) + "\n"

    @staticmethod
    def _missing_from_schema(error: str) -> tuple[str, str] | None:
        """Return (kind, identifier) for no-such-column/table SQLite errors."""
        match = re.search(r"no such column:\s*([^\s\]]+)", error, flags=re.IGNORECASE)
        if match:
            identifier = match.group(1).strip("'\"`")
            # SQLite may report an alias-qualified name such as p.salary; the
            # schema lookup only needs the final field name.
            identifier = identifier.rsplit(".", 1)[-1]
            return "column", identifier
        match = re.search(r"no such table(?: or view)?:\s*([^\s\]]+)", error, flags=re.IGNORECASE)
        if match:
            return "table", match.group(1).strip("'\"")
        return None

    def _schema_fallback(self, error: str) -> str | None:
        """Turn repeated missing-field errors into a useful schema answer."""
        missing = self._missing_from_schema(error)
        if missing is None:
            return None
        kind, identifier = missing

        by_table = "\n".join(
            f"- {table}: {', '.join(columns)}"
            for table, columns in self.table_columns.items()
        )
        if kind == "table":
            tables = ", ".join(f"`{table}`" for table in self.table_columns)
            return (
                f"The database has no `{identifier}` table, so this question cannot "
                f"be answered. Available tables: {tables or 'see the schema'}"
            )

        locations = [
            (table, column)
            for table, columns in self.table_columns.items()
            for column in columns
            if column.lower() == identifier.lower()
        ]
        if locations:
            location_text = ", ".join(
                f"`{table}`.`{column}`" for table, column in locations
            )
            return (
                f"The query referenced `{identifier}` in a table that does not "
                f"contain it. That field is available as {location_text}; use "
                "that table or JOIN it to the queried table.\n\n"
                f"Available fields by table:\n{by_table}"
            )

        close_pairs = []
        for table, columns in self.table_columns.items():
            for match in difflib.get_close_matches(identifier.lower(), [c.lower() for c in columns], n=2, cutoff=0.45):
                original = next(column for column in columns if column.lower() == match)
                close_pairs.append(f"`{table}`.`{original}`")
        close_text = ""
        if close_pairs:
            close_text = " Closest available fields: " + ", ".join(close_pairs) + "."
        return (
            f"The database has no `{identifier}` field, so this question cannot be "
            f"answered from the current schema.{close_text}\n\n"
            f"Available fields by table:\n{by_table or 'see the schema'}"
        )

    def select_examples(self, question: str, limit: int = 6) -> str:
        """Pick the few-shot examples closest to the question.

        Sending all examples bloats a 4096-token context and distracts a 0.5B
        model. Lexical overlap keeps only the relevant patterns in the prompt.
        """
        if not self.example_blocks:
            return ""
        question_tokens = {
            token
            for token in self._schema_tokens(question)
            if len(token) >= 3 and token not in self._STOPWORDS
        }
        scored = [
            (len(question_tokens & self._schema_tokens(example_question)), index)
            for index, (example_question, _sql) in enumerate(self.example_blocks)
        ]
        chosen = [
            index
            for score, index in sorted(scored, key=lambda item: (-item[0], item[1]))
            if score > 0
        ][:limit]
        # Always show at least three examples so the tiny model imitates the
        # bare "SELECT ...;" output format instead of writing prose.
        if len(chosen) < 3:
            for index in range(len(self.example_blocks)):
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) >= 3:
                    break
        return "\n".join(
            f"Q: {example_question}\nSQL: {example_sql}"
            for example_question, example_sql in (
                self.example_blocks[index] for index in sorted(chosen)
            )
        )

    def _semantic_repair_hint(self, question: str, sql: str) -> str:
        """Detect SQL that executes but answers a different question.

        A syntax check cannot catch AVG(year) used as an age or a department
        name compared to job_title. These deterministic checks feed a targeted
        correction back to the model instead of showing a wrong result.
        """
        q = question.lower()
        s = sql.lower()
        compact = re.sub(r"\s+", "", s)
        hints: list[str] = []

        if re.search(r"\b(?:age|ages)\b", q) and "date_of_birth" in s and "julianday" not in s:
            hints.append(
                "Age must be computed from date_of_birth with "
                "(julianday('now') - julianday(date_of_birth)) / 365.25, not by "
                "averaging the birth year."
            )

        if (
            "years_experience" in s
            and re.search(r"\b(?:more|less|at least|over|under|between)\b|\b\d+\s*years?\b", q)
            and not re.search(
                r"years_experience\s*(?:>=|<=|>|<|=|between|like|in)", s
            )
        ):
            hints.append(
                "Filter years of experience with a comparison, for example "
                "WHERE years_experience > 10. Do not add an unrelated self-join."
            )

        if (
            re.search(r"\bsalar(?:y|ies)\b|\bcompensation\b", q)
            and "date_of_birth" in s
        ):
            hints.append(
                "Salary is employees.salary. Do not use date_of_birth or an age "
                "calculation for a salary question."
            )

        if (
            re.search(r"\breview\b", q)
            and re.search(r"\blatest\b|\bmost recent\b", q)
            and re.search(r"max\s*\(\s*review_id", s)
        ):
            hints.append(
                "\"Latest review\" is per employee. Use MAX(review_date) in a "
                "correlated subquery on employee_id, not the global MAX(review_id)."
            )

        if re.search(r"\bpercent(?:age)?\b", q) and "/" not in s:
            hints.append(
                "A percentage needs division: "
                "100.0 * numerator / NULLIF(denominator, 0)."
            )

        if (
            re.search(r"\b(?:no|without|never)\b", q)
            and re.search(r"(?:=|<>|!=)\s*0\b", s)
            and not re.search(r"\b(?:left join|not exists|not in|is null)\b", s)
        ):
            hints.append(
                "\"no/without/never\" means a missing related row: use "
                "LEFT JOIN ... WHERE related_key IS NULL or NOT EXISTS."
            )

        if (
            re.search(r"\bskills?\b", q)
            and "proficiency" in s
            and not re.search(r"\bskill\s*(?:=|in|like)\b", s)
        ):
            hints.append(
                "Filter a named skill with employee_skills.skill = 'Python'. "
                "proficiency only stores Beginner/Intermediate/Advanced/Expert."
            )

        if (
            re.search(r"\bpromot\w*\b", q)
            and ("performance_reviews", "promotion_recommended") in self.value_hints
            and "promotion_recommended" not in s
        ):
            hints.append(
                "Promotion questions use "
                "performance_reviews.promotion_recommended = 1."
            )

        if re.search(r"\bmanagers?\b", q) and "manager_id" in s:
            if not re.search(r"\bjoin\s+employees\b", s):
                hints.append(
                    "Manager questions need a self-join: employees AS e "
                    "JOIN employees AS m ON e.manager_id = m.employee_id."
                )
            else:
                # Detect GROUP BY on the report alias instead of the manager
                # alias, which collapses every manager count to 1.
                counted = re.search(r"count\s*\(\s*(\w+)\.employee_id", s)
                grouped = re.search(r"group\s+by\s+(\w+)\.", s)
                if counted and grouped and counted.group(1) == grouped.group(1):
                    hints.append(
                        "When counting direct reports, COUNT the report rows but "
                        "GROUP BY the manager alias, for example "
                        "COUNT(e.employee_id) ... GROUP BY m.employee_id."
                    )

        if "job_title" in s:
            for (_table, column), values in self.value_hints.items():
                if column != "department_name":
                    continue
                if any(
                    re.search(rf"\b{re.escape(value.lower())}\b", q)
                    for value in values
                ):
                    hints.append(
                        "That name is a department, not a job title. JOIN "
                        "departments and filter departments.department_name."
                    )
                    break

        if (
            re.search(r"\blast month\b", q)
            and "payroll" in s
            and "max(pay_month)" not in compact
            and "date('now')" in s
        ):
            hints.append(
                "For last month's payroll use pay_month = "
                "(SELECT MAX(pay_month) FROM payroll). Do not compare "
                "payment_date to date('now')."
            )

        return "\n".join(hints)

    def _generate_sql(self, question: str, repair_context: str, schema: str) -> str:
        response = self.sql_chain.invoke(
            {
                "schema": schema,
                "domain_hints": self.domain_hints,
                "examples": self.select_examples(question),
                "question": question,
                "repair_context": repair_context,
            }
        )
        sql = clean_generated_sql(getattr(response, "content", response))
        return self._canonicalize_literals(sql) if sql else sql

    def answer(self, question: str) -> AnswerResult:
        """Run the generation/execute/repair cycle for one user question."""
        queries: list[str] = []
        repair_context = ""
        last_error = "The model did not return SQL."
        fallback_result: AnswerResult | None = None
        selected_tables, schema_text = self.select_schema(question)

        for _attempt in range(MAX_SQL_ATTEMPTS):
            try:
                sql = self._generate_sql(question, repair_context, schema_text)
            except Exception as exc:
                last_error = _short_error(exc)
                # A model/server failure is unlikely to be fixed by another
                # generation, so fail immediately with a useful error.
                raise SQLPipelineError(last_error, queries) from exc

            if sql.upper().startswith("NO_SQL"):
                reason = sql.split(":", 1)[-1].strip().rstrip(" .")
                if not reason:
                    reason = "the current schema does not contain the requested data"
                return AnswerResult(
                    answer=(
                        "That question cannot be answered from the current database: "
                        f"{reason}."
                    ),
                    queries=queries,
                    rows=[],
                )

            if not sql:
                last_error = "The model returned an empty SQL response."
            elif sql in queries:
                # The tiny model repeated a failing query; repairing again would
                # only waste a generation. The original SQLite error remains in
                # last_error for the schema fallback below.
                break
            else:
                queries.append(sql)
                try:
                    raw_result = self.db.run(sql, include_columns=True)
                    rows = parse_result_rows(raw_result)
                    semantic = self._semantic_repair_hint(question, sql)
                    if semantic:
                        # Keep the executed result as a last-resort answer so a
                        # stubborn model cannot turn a usable result into an error.
                        fallback_result = AnswerResult(
                            answer=summarize_rows(rows),
                            queries=list(queries),
                            rows=rows,
                        )
                        last_error = "The SQL ran, but it likely answers a different question."
                        repair_context = (
                            "The previous SQL ran but does not correctly answer the "
                            "question. Rewrite it.\n"
                            f"{self.schema_note(selected_tables)}\n\n"
                            f"Previous SQL:\n{sql}\n"
                            f"Problem:\n{semantic}\n"
                        )
                        continue

                    answer = summarize_rows(rows)
                    zero_count = (
                        len(rows) == 1
                        and bool(rows[0])
                        and all(
                            isinstance(value, (int, float)) and value == 0
                            for value in rows[0].values()
                        )
                    )
                    if not rows or zero_count:
                        note = self._coverage_note(sql)
                        if note:
                            answer = f"{answer}\n\n{note}"
                    return AnswerResult(
                        answer=answer,
                        queries=queries,
                        rows=rows,
                    )
                except Exception as exc:
                    last_error = _short_error(exc)

            # If the model referenced a real table/field that was filtered out of
            # the focused schema, give the next attempt the complete schema.
            missing = self._missing_from_schema(last_error)
            if missing:
                kind, identifier = missing
                if kind == "table" and identifier in self.table_schemas:
                    selected_tables = list(self.table_schemas)
                    schema_text = self.full_schema
                elif kind == "column":
                    locations = [
                        table
                        for table, columns in self.table_columns.items()
                        if any(column.lower() == identifier.lower() for column in columns)
                    ]
                    if any(table not in selected_tables for table in locations):
                        selected_tables = list(self.table_schemas)
                        schema_text = self.full_schema

            bad_sql = sql or "(empty response)"
            repair_context = (
                "The previous attempt failed. Correct it using only real schema "
                "fields.\n"
                f"{self.schema_note(selected_tables)}\n\n"
                f"Previous SQL:\n{bad_sql}\n"
                f"Error:\n{last_error}\n"
                f"{self._repair_hint(bad_sql, last_error)}\n"
            )

        if fallback_result is not None:
            return fallback_result

        fallback = self._schema_fallback(last_error)
        if fallback is not None:
            return AnswerResult(answer=fallback, queries=queries, rows=[])

        raise SQLPipelineError(
            f"Could not generate a valid query after {MAX_SQL_ATTEMPTS} attempts. "
            f"Last error: {last_error}",
            queries,
        )


# ---------------------------------------------------------------------------
# Cached application resources
# ---------------------------------------------------------------------------

@dataclass
class AssistantBundle:
    assistant: LocalSQLAssistant
    engine: Engine
    db: GuardedSQLDatabase
    db_path: Path
    model: str
    base_url: str


def build_assistant(db_path: Path, model: str, base_url: str) -> AssistantBundle:
    """Build the read-only database, local model, and SQL pipeline once."""
    engine = create_readonly_engine(db_path)

    try:
        db = GuardedSQLDatabase(
            engine=engine,
            sample_rows_in_table_info=0,   # Schema only; do not print data rows.
            max_string_length=MAX_STRING_LENGTH,
            max_rows=MAX_ROWS,
        )
        if not db.get_usable_table_names():
            raise ValueError(f"No readable tables found in {db_path}.")

        llm = ChatOllama(
            model=model,
            base_url=base_url.rstrip("/"),
            temperature=0.0,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
            keep_alive="10m",
        )
        assistant = LocalSQLAssistant(db=db, llm=llm, schema=db.get_table_info())
    except Exception:
        engine.dispose()
        raise

    return AssistantBundle(
        assistant=assistant,
        engine=engine,
        db=db,
        db_path=db_path,
        model=model,
        base_url=base_url.rstrip("/"),
    )


def get_cached_assistant(db_path: Path, model: str, base_url: str) -> AssistantBundle:
    """Return the existing pipeline unless configuration changed.

    Streamlit reruns the script on every interaction. Keeping the bundle in
    st.session_state prevents rebuilding the engine and Ollama client on each
    chat message.
    """
    signature = (APP_VERSION, str(db_path), model, base_url.rstrip("/"))
    current: AssistantBundle | None = st.session_state.get("assistant_bundle")

    if current is not None and st.session_state.get("assistant_signature") == signature:
        return current

    if current is not None:
        try:
            current.engine.dispose()
        except Exception:
            pass
        st.session_state["assistant_bundle"] = None
        st.session_state["assistant_signature"] = None

    bundle = build_assistant(db_path, model, base_url)
    st.session_state["assistant_bundle"] = bundle
    st.session_state["assistant_signature"] = signature
    return bundle


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize chat and pipeline state exactly once per browser session."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant_bundle" not in st.session_state:
        st.session_state["assistant_bundle"] = None
    if "assistant_signature" not in st.session_state:
        st.session_state["assistant_signature"] = None


def render_sidebar() -> tuple[str, str, str]:
    """Render configuration and return database path, model, and base URL."""
    with st.sidebar:
        st.header("Local setup")
        db_path = st.text_input(
            "SQLite database",
            value=str(DEFAULT_DB_PATH),
            help="A local SQLite database. The app opens it read-only.",
        )
        model = st.text_input(
            "Ollama model",
            value=DEFAULT_MODEL,
            help="Default: Falcon-H1 0.5B Q6_K (~430 MB in Ollama).",
        )
        base_url = st.text_input(
            "Ollama server",
            value=DEFAULT_BASE_URL,
            help="The local Ollama HTTP endpoint.",
        )

        st.divider()
        st.caption(
            f"Read-only SQLite  |  SELECT only  |  hard limit {MAX_ROWS} rows\n\n"
            "All inference and data stay on this machine."
        )

        if st.button("Clear chat", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.caption(
            "First run? Import the Falcon-H1 model with "
            "`ollama create falcon-h1-0.5b:q6k -f Modelfile.falcon-h1` "
            "(see README). The sidebar path defaults to company_database.sqlite "
            "when present."
        )

    return db_path, model, base_url


def render_assistant_details(
    queries: list[str] | None,
    rows: list[dict[str, Any]] | None,
) -> None:
    """Show generated SQL and tabular results without exceeding the row cap."""
    if queries:
        with st.expander("View generated SQL", expanded=False):
            st.caption(
                f"An outer LIMIT {MAX_ROWS} is enforced by the database wrapper "
                "during execution."
            )
            for index, query in enumerate(queries, start=1):
                st.caption(f"Attempt {index}")
                st.code(query, language="sql")

    if rows:
        with st.expander(f"View query results ({len(rows)} rows)", expanded=False):
            st.dataframe(rows, width="stretch", hide_index=True)


def render_history() -> None:
    """Render persisted chat messages, SQL, and result tables."""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_assistant_details(
                    message.get("queries"),
                    message.get("rows"),
                )


def main() -> None:
    st.set_page_config(
        page_title="Corporate Data Assistant",
        page_icon="🏢",
        layout="wide",
    )
    st.title("Corporate Data Assistant")
    st.caption(
        "Ask natural-language questions about employees, departments, projects, "
        "payroll, and attendance. Falcon-H1 writes read-only SQL; the app runs it "
        "safely and reports the results."
    )

    init_session_state()
    db_input, model_input, base_url_input = render_sidebar()

    model = model_input.strip() or DEFAULT_MODEL
    base_url = base_url_input.strip() or DEFAULT_BASE_URL

    try:
        db_path = resolve_db_path(db_input)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not db_path.is_file():
        st.error(f"SQLite database not found: `{db_path}`")
        st.info("Run `python mock_mca_db.py` or enter a different path in the sidebar.")
        st.stop()

    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(
                "Ask about employees, departments, projects, payroll, or attendance. "
                "For example: *Which 5 employees have the highest salary?* or "
                "*What is the average salary by department?*"
            )

    render_history()

    question = st.chat_input("Ask a question about your company data...")
    if not question or not question.strip():
        return

    question = question.strip()
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    error_answer: str | None = None
    queries: list[str] = []
    rows: list[dict[str, Any]] = []

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL and querying the read-only database..."):
            try:
                bundle = get_cached_assistant(db_path, model, base_url)
                result = bundle.assistant.answer(question)
                answer = result.answer
                queries = result.queries
                rows = result.rows
            except SQLPipelineError as exc:
                error_answer = (
                    "I could not generate a valid SQL query for that question. "
                    "Try rephrasing it, or open **Technical details** below for "
                    "the last SQLite error."
                )
                queries = exc.queries
                st.error(error_answer)
                with st.expander("Technical details", expanded=False):
                    st.code(str(exc))
            except Exception as exc:
                # Ollama connectivity/model errors and other unexpected failures.
                error_answer = (
                    "I could not complete that request. Check that Ollama is "
                    f"running at `{base_url}` and that the `{model}` model exists."
                )
                st.error(error_answer)
                with st.expander("Technical details", expanded=False):
                    st.code(f"{type(exc).__name__}: {exc}")
                queries = []

        if error_answer is None:
            st.markdown(answer)
            render_assistant_details(queries, rows)
        elif queries:
            render_assistant_details(queries, [])

    # Persist both successful and failed turns so the chat survives reruns.
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": error_answer if error_answer is not None else answer,
            "queries": queries,
            "rows": rows,
        }
    )


if __name__ == "__main__":
    main()
