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


def test_sql_query_allows_cte_alias_when_base_table_is_expected():
    tool = SQLQuery()
    sql = """
WITH levels AS (
  SELECT 'national' AS geo_level
  UNION ALL
  SELECT 'region' AS geo_level
)
SELECT levels.geo_level, COALESCE(COUNT(r.geo_level), 0) AS count
FROM levels
LEFT JOIN retail_sales AS r
  ON r.geo_level = levels.geo_level
 AND r.year = 2024
 AND r.total_sales_100m_cny > 3000
GROUP BY levels.geo_level
ORDER BY levels.geo_level
"""
    ok = tool.sql_query("retail_sales", sql)
    rows = json.loads(ok)
    assert isinstance(rows, list) and len(rows) == 2
    assert {"geo_level", "count"}.issubset(rows[0].keys())


def test_sql_query_cte_cannot_read_other_real_table():
    tool = SQLQuery()
    sql = """
WITH tmp AS (
  SELECT year FROM exchange_rates WHERE year = 2024 LIMIT 1
)
SELECT year FROM tmp LIMIT 1
"""
    blocked = tool.sql_query("interest_rates", sql)
    assert "only reference table 'interest_rates'" in blocked


def test_sql_query_rejects_schema_qualified_reference():
    tool = SQLQuery()
    blocked = tool.sql_query("interest_rates", "SELECT year FROM main.exchange_rates LIMIT 1")
    assert "schema-qualified table references are not allowed" in blocked


def test_sql_query_rejects_sqlite_internal_tables():
    tool = SQLQuery()
    blocked = tool.sql_query("interest_rates", 'SELECT name FROM sqlite_master WHERE type = "table" LIMIT 1')
    assert "sqlite internal tables" in blocked.lower()
