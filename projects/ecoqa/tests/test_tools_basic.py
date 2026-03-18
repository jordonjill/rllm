import json

from projects.ecoqa.eco_qa_tools import GetTableInfo, GetTableNames, SQLQuery


def test_get_table_names_contains_interest_rates():
    tool = GetTableNames()
    names = tool.get_table_names()
    assert "interest_rates" in names


def test_get_table_info_returns_expected_keys():
    tool = GetTableInfo()
    info = tool.get_table_info("interest_rates")
    payload = json.loads(info)
    expected = {"table_name", "row_count", "column_names", "column_dtypes", "numeric_columns", "sample_values"}
    assert expected.issubset(payload.keys())


def test_sql_query_blocks_select_star_and_allows_valid_query():
    tool = SQLQuery()
    blocked = tool.sql_query("interest_rates", "SELECT * FROM interest_rates LIMIT 1")
    assert "not allowed" in blocked.lower()

    ok = tool.sql_query("interest_rates", "SELECT year FROM interest_rates LIMIT 1")
    rows = json.loads(ok)
    assert isinstance(rows, list) and len(rows) == 1 and "year" in rows[0]


def test_sql_query_rejects_cross_table_query():
    tool = SQLQuery()
    blocked = tool.sql_query("interest_rates", "SELECT year FROM exchange_rates LIMIT 1")
    assert "only reference table 'interest_rates'" in blocked


def test_sql_query_rejects_schema_qualified_reference():
    tool = SQLQuery()
    blocked = tool.sql_query("interest_rates", "SELECT year FROM main.exchange_rates LIMIT 1")
    assert "schema-qualified table references are not allowed" in blocked


def test_sql_query_rejects_sqlite_internal_tables():
    tool = SQLQuery()
    blocked = tool.sql_query("interest_rates", 'SELECT name FROM sqlite_master WHERE type = "table" LIMIT 1')
    assert "sqlite internal tables" in blocked.lower()
