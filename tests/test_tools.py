import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from config import ToolsConfig
from tools.base import ToolRegistry, ToolResult
from tools.personal_data import PersonalDataTool, _period_bounds
from tools.weather import WeatherTool


@pytest.fixture
def expenses():
    tmp = Path(tempfile.mkdtemp()) / "test.db"
    tool = PersonalDataTool(ToolsConfig(), tmp)
    today = date.today()
    tool.add_expense(today.isoformat(), "food", "Lunch", 250.0)
    tool.add_expense(today.isoformat(), "transport", "Cab", 300.0)
    tool.add_expense(today.replace(day=1).isoformat(), "food", "Groceries", 1200.0)
    return tool


def test_total_sums_the_period(expenses):
    r = expenses.execute(query_kind="total", period="this_month")
    assert r.ok and r.data["total"] == 1750.0
    assert r.data["transaction_count"] == 3


def test_category_filter_narrows_the_total(expenses):
    r = expenses.execute(query_kind="total", period="this_month", category="food")
    assert r.data["total"] == 1450.0


def test_today_is_narrower_than_this_month(expenses):
    assert expenses.execute(query_kind="total", period="today").data["total"] == 550.0


def test_breakdown_is_sorted_descending(expenses):
    rows = expenses.execute(query_kind="by_category").data["breakdown"]
    totals = [r["total"] for r in rows]
    assert totals == sorted(totals, reverse=True)


def test_empty_period_is_flagged_not_faked(expenses):
    r = expenses.execute(query_kind="total", period="last_month")
    assert r.ok and r.data["total"] == 0 and "note" in r.data


def test_sql_injection_in_category_is_inert(expenses):
    """Category is a bound parameter, so this is just a string that matches
    nothing -- it must not drop the table."""
    r = expenses.execute(query_kind="total", category="food'; DROP TABLE expenses; --")
    assert r.ok and r.data["total"] == 0
    assert expenses.count() == 3


def test_invalid_period_falls_back_safely(expenses):
    assert expenses.execute(query_kind="total", period="last_tuesday").ok


def test_period_bounds_use_local_dates():
    start, end, label = _period_bounds("this_month")
    assert start == date.today().replace(day=1).isoformat()
    assert end == date.today().isoformat()
    assert label == "this month"


# --- registry -------------------------------------------------------------
def test_registry_rejects_unknown_tool():
    reg = ToolRegistry([WeatherTool(ToolsConfig())])
    r = reg.execute("rm_rf_slash", {})
    assert not r.ok and "No such tool" in r.error


def test_registry_rejects_duplicate_names():
    reg = ToolRegistry([WeatherTool(ToolsConfig())])
    with pytest.raises(ValueError):
        reg.register(WeatherTool(ToolsConfig()))


def test_bad_arguments_do_not_raise(expenses):
    reg = ToolRegistry([expenses])
    r = reg.execute("query_expenses", {"nonexistent_kwarg": 1})
    assert not r.ok and "Invalid arguments" in r.error


def test_error_payload_is_explicit_for_the_model():
    payload = ToolResult(False, error="network down").to_model_payload()
    assert payload["status"] == "error"
    assert "result" not in payload


def test_every_tool_declares_a_valid_schema():
    for tool in ToolRegistry([WeatherTool(ToolsConfig())]).specs():
        assert tool["name"] and tool["description"]
        assert tool["input_schema"]["type"] == "object"


@pytest.mark.network
def test_weather_live():
    r = WeatherTool(ToolsConfig()).execute(location="Hyderabad", day_offset=1)
    assert r.ok and "high_c" in r.data
