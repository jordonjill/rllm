import json
import re
import sqlite3
import threading
from pathlib import Path

import pandas as pd
from asteval import Interpreter
from pandas.api.types import is_numeric_dtype

from rllm.tools.tool_base import Tool

from .constants import TABLES_ROOT

_DB_CONN: sqlite3.Connection | None = None
_DB_LOCK = threading.Lock()
_TABLES: dict[str, pd.DataFrame] = {}
_SQL_NAMES: dict[str, str] = {}
_TABLE_INFO_CACHE: dict[str, str] = {}
_TABLE_NAME_MAP: dict[str, str] = {}
_PRELOADED = False
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+((?:[`\"\[]?[A-Za-z_][A-Za-z0-9_]*[`\"\]]?\s*\.\s*)?[`\"\[]?[A-Za-z_][A-Za-z0-9_]*[`\"\]]?)",
    re.IGNORECASE,
)


def _sanitize_sql_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    if not cleaned:
        cleaned = "table"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _sanitize_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    final = []
    for idx, raw in enumerate(columns):
        base = str(raw).strip() or f"col_{idx}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        final.append(base if count == 1 else f"{base}_{count}")
    return final


def _normalize_table_name(table_name: str | object) -> str | None:
    if not isinstance(table_name, str):
        return None
    name = table_name.strip()
    if name.endswith(".csv"):
        name = name[:-4]
    return _TABLE_NAME_MAP.get(name.lower())


def _extract_query_table_refs(query: str) -> list[str]:
    refs: list[str] = []
    for match in _TABLE_REF_RE.finditer(query):
        token = match.group(1).strip()
        if token:
            refs.append(token)
    return refs


def _normalize_ref_token(token: str) -> str:
    parts = [part.strip().strip('`"[]') for part in re.split(r"\s*\.\s*", token) if part.strip()]
    return ".".join(parts).lower()


def _known_table_refs() -> set[str]:
    return {name.lower() for name in _TABLES} | {sql_name.lower() for sql_name in _SQL_NAMES.values()}


def _safe_json_value(val):
    if pd.isna(val):
        return None
    if hasattr(val, "isoformat"):
        return str(val)
    return val


def _compute_table_info(table_name: str, df: pd.DataFrame) -> str:
    payload: dict[str, object] = {
        "table_name": table_name,
        "row_count": int(len(df)),
        "column_names": df.columns.tolist(),
        "column_dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    numeric_columns = [col for col in df.columns if is_numeric_dtype(df[col])]
    payload["numeric_columns"] = numeric_columns

    sample_values: dict[str, list] = {}
    for col in df.columns:
        uniq = df[col].dropna().head(100).astype(str).unique().tolist()[:8]
        sample_values[col] = [_safe_json_value(v) for v in uniq]
    payload["sample_values"] = sample_values

    return json.dumps(payload, ensure_ascii=False)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = _sanitize_columns(df.columns.tolist())
    return df


def _preload_all() -> None:
    global _DB_CONN, _PRELOADED

    if _PRELOADED:
        return

    _DB_CONN = sqlite3.connect(":memory:", check_same_thread=False)

    if not TABLES_ROOT.exists():
        _PRELOADED = True
        return

    for csv_path in sorted(TABLES_ROOT.glob("*.csv")):
        table_name = csv_path.stem
        try:
            df = _read_csv(csv_path)
        except Exception:
            continue

        sql_name = _sanitize_sql_name(table_name)
        df.to_sql(sql_name, _DB_CONN, index=False, if_exists="replace")

        _TABLES[table_name] = df
        _SQL_NAMES[table_name] = sql_name
        _TABLE_INFO_CACHE[table_name] = _compute_table_info(table_name, df)
        _TABLE_NAME_MAP[table_name.lower()] = table_name

    _PRELOADED = True


_preload_all()


class GetTableNames(Tool):
    NAME = "get_table_names"
    DESCRIPTION = """
List available table names.

Arguments:
    keyword: Optional substring filter for table names.
Returns:
    List of table names.
"""

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.get_table_names)

    def get_table_names(self, keyword: str = "") -> list[str]:
        names = sorted(_TABLES.keys())
        key = str(keyword).strip().lower()
        if not key:
            return names
        return [name for name in names if key in name.lower()]


class GetTableInfo(Tool):
    NAME = "get_table_info"
    DESCRIPTION = """
Return schema and sample data for one table.

Arguments:
    table_name: Name returned by get_table_names.
Returns:
    JSON string with row count, column names, dtypes, and sample values.
"""

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.get_table_info)

    def get_table_info(self, table_name: str) -> str:
        table = _normalize_table_name(table_name)
        if table is None:
            return f"Error: table '{table_name}' not found."
        return _TABLE_INFO_CACHE[table]


class SQLQuery(Tool):
    NAME = "sql_query"
    DESCRIPTION = """
Execute SQLite SQL for a single table.

Arguments:
    table_name: Target table name.
    query: SQL query to execute.
Returns:
    JSON records string.
"""

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.sql_query)

    def sql_query(self, table_name: str, query: str) -> str:
        if not query or not query.strip():
            return "Error: query must not be empty."

        table = _normalize_table_name(table_name)
        if table is None:
            return f"Error: table '{table_name}' not found."

        sql_name = _SQL_NAMES[table]
        query_upper = re.sub(r"\s+", " ", query).upper().strip()

        if re.search(r"\bSELECT\s+\*\s+FROM\b", query_upper):
            return f'Error: "SELECT *" is not allowed. Please list columns explicitly, e.g., SELECT column1, column2 FROM {table} LIMIT 5.'

        forbidden = ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ", "ATTACH ", "PRAGMA ")
        if any(token in query_upper for token in forbidden):
            return "Error: only read-only SELECT queries are allowed."

        if "SELECT" not in query_upper or "FROM" not in query_upper:
            return "Error: query must contain SELECT and FROM clauses."
        if re.search(r"\b(SQLITE_MASTER|SQLITE_SCHEMA)\b", query_upper):
            return "Error: querying SQLite internal tables is not allowed."

        sql_filters = (
            " WHERE ",
            " HAVING ",
            " GROUP BY ",
            " ORDER BY ",
            " LIMIT ",
            " OFFSET ",
            " COUNT(",
            " SUM(",
            " AVG(",
            " MIN(",
            " MAX(",
        )
        if not any(clause in query_upper for clause in sql_filters):
            return "Error: include at least one filter/limit clause or aggregate function."

        referenced_tables = _extract_query_table_refs(query)
        if not referenced_tables:
            return "Error: query must reference at least one table."

        allowed_tables = {table.lower(), _SQL_NAMES[table].lower()}
        known_table_refs = _known_table_refs()
        for ref in referenced_tables:
            ref_key = _normalize_ref_token(ref)
            if not ref_key:
                continue
            if "." in ref_key:
                return "Error: schema-qualified table references are not allowed."
            if ref_key in {"sqlite_master", "sqlite_schema"}:
                return "Error: querying SQLite internal tables is not allowed."
            # Permit CTE/subquery aliases (unknown refs), but block access to
            # other real preloaded tables.
            if ref_key in known_table_refs and ref_key not in allowed_tables:
                return f"Error: query may only reference table '{table}'."

        quoted_sql_name = f'"{sql_name}"'
        table_name_str = str(table_name).strip()
        pattern = rf'(FROM|JOIN)\s+[`"\']?{re.escape(table_name_str)}[`"\']?(?=\s|$|;|,|\))'
        query_for_execution = re.sub(pattern, rf"\g<1> {quoted_sql_name}", query, flags=re.IGNORECASE)

        # Also normalize canonical table name if caller uses the normalized name.
        pattern2 = rf'(FROM|JOIN)\s+[`"\']?{re.escape(table)}[`"\']?(?=\s|$|;|,|\))'
        query_for_execution = re.sub(pattern2, rf"\g<1> {quoted_sql_name}", query_for_execution, flags=re.IGNORECASE)

        try:
            with _DB_LOCK:
                resp = pd.read_sql_query(query_for_execution, _DB_CONN)
        except Exception as exc:
            return f"Error: {exc}"

        if len(resp) > 200:
            resp = resp.head(200)

        return resp.to_json(orient="records", force_ascii=False)


class Calculator(Tool):
    NAME = "calculator"
    DESCRIPTION = """
Safely evaluate a math expression.

Arguments:
    expression: A math expression string.
Returns:
    float result.
"""

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.calculator)

    def calculator(self, expression: str) -> float | str:
        if not isinstance(expression, str):
            return "Error: expression must be a string."

        expr = expression.strip()
        expr = expr.replace("\n", " ").replace("\r", " ")
        expr = expr.replace("^", "**")
        expr = expr.replace("$", "").replace("€", "").replace("£", "")
        expr = expr.replace("\u00a0", " ")
        expr = expr.replace("–", "-").replace("—", "-").replace("−", "-")
        expr = re.sub(r"\d{1,3}(?:,\d{3})+", lambda m: m.group(0).replace(",", ""), expr)

        try:
            aeval = Interpreter()
            result = aeval.eval(expr)
            return float(result)
        except Exception as exc:
            return f"Error evaluating expression: {exc}"
