#!/usr/bin/env python3
"""Generate 15 China economic CSV tables for SQL retrieval training."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

OUT_DIR = Path("csv")
OUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2016, 2026))
BASE_YEAR = YEARS[0]
LATEST_YEAR = YEARS[-1]


def dnoise(key: str, amp: float) -> float:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    x = h / 0xFFFFFFFF
    return (x * 2.0 - 1.0) * amp


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pick(key: str, items: list[str]) -> str:
    idx = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % len(items)
    return items[idx]


def status(freq: str, year: int, period: int | None = None) -> str:
    if year <= LATEST_YEAR - 1:
        return "final"
    if freq == "M":
        if period is None:
            return "revised"
        if period >= 10:
            return "preliminary"
        if period >= 7:
            return "revised"
        return "final"
    if freq == "Q":
        if period is None:
            return "revised"
        if period == 4:
            return "preliminary"
        if period == 3:
            return "revised"
        return "final"
    if freq == "Y":
        return "preliminary" if year == 2025 else "final"
    return "revised"


def release_date(stat_date: date, freq: str) -> str:
    days = {"M": 40, "Q": 55, "Y": 110}[freq]
    return (stat_date + timedelta(days=days)).isoformat()


def month_iter():
    for y in YEARS:
        for m in range(1, 13):
            yield y, m, date(y, m, 1)


def quarter_iter():
    quarter_month = {1: 1, 2: 4, 3: 7, 4: 10}
    for y in YEARS:
        for q in range(1, 5):
            yield y, q, date(y, quarter_month[q], 1)


def year_iter():
    for y in YEARS:
        yield y, date(y, 1, 1)


regions = [
    {"geo_code": "CN-R-NC", "geo_name": "华北"},
    {"geo_code": "CN-R-NE", "geo_name": "东北"},
    {"geo_code": "CN-R-EC", "geo_name": "华东"},
    {"geo_code": "CN-R-CC", "geo_name": "华中"},
    {"geo_code": "CN-R-SC", "geo_name": "华南"},
    {"geo_code": "CN-R-SW", "geo_name": "西南"},
    {"geo_code": "CN-R-NW", "geo_name": "西北"},
]
region_name_by_code = {r["geo_code"]: r["geo_name"] for r in regions}

# 31 省级行政区（不含港澳台）
provinces_raw = [
    ("CN-BJ", "北京", "CN-R-NC", 4.6),
    ("CN-TJ", "天津", "CN-R-NC", 1.8),
    ("CN-HE", "河北", "CN-R-NC", 4.6),
    ("CN-SX", "山西", "CN-R-NC", 2.6),
    ("CN-NM", "内蒙古", "CN-R-NC", 1.8),
    ("CN-LN", "辽宁", "CN-R-NE", 3.1),
    ("CN-JL", "吉林", "CN-R-NE", 1.2),
    ("CN-HL", "黑龙江", "CN-R-NE", 1.7),
    ("CN-SH", "上海", "CN-R-EC", 4.7),
    ("CN-JS", "江苏", "CN-R-EC", 12.8),
    ("CN-ZJ", "浙江", "CN-R-EC", 8.7),
    ("CN-AH", "安徽", "CN-R-EC", 4.8),
    ("CN-FJ", "福建", "CN-R-EC", 5.1),
    ("CN-JX", "江西", "CN-R-EC", 3.2),
    ("CN-SD", "山东", "CN-R-EC", 11.5),
    ("CN-HA", "河南", "CN-R-CC", 6.2),
    ("CN-HB", "湖北", "CN-R-CC", 5.8),
    ("CN-HN", "湖南", "CN-R-CC", 5.0),
    ("CN-GD", "广东", "CN-R-SC", 13.5),
    ("CN-GX", "广西", "CN-R-SC", 2.7),
    ("CN-HI", "海南", "CN-R-SC", 0.8),
    ("CN-CQ", "重庆", "CN-R-SW", 3.0),
    ("CN-SC", "四川", "CN-R-SW", 6.5),
    ("CN-GZ", "贵州", "CN-R-SW", 2.5),
    ("CN-YN", "云南", "CN-R-SW", 2.7),
    ("CN-XZ", "西藏", "CN-R-SW", 0.3),
    ("CN-SN", "陕西", "CN-R-NW", 3.6),
    ("CN-GS", "甘肃", "CN-R-NW", 1.1),
    ("CN-QH", "青海", "CN-R-NW", 0.5),
    ("CN-NX", "宁夏", "CN-R-NW", 0.7),
    ("CN-XJ", "新疆", "CN-R-NW", 1.7),
]

raw_sum = sum(x[3] for x in provinces_raw)
provinces = []
for code, name, r_code, raw in provinces_raw:
    provinces.append(
        {
            "geo_level": "province",
            "geo_code": code,
            "geo_name": name,
            "parent_geo_code": r_code,
            "parent_geo_name": region_name_by_code[r_code],
            "scale": raw / raw_sum,
        }
    )

region_scale = defaultdict(float)
for p in provinces:
    region_scale[p["parent_geo_code"]] += p["scale"]

region_geos = []
for r in regions:
    region_geos.append(
        {
            "geo_level": "region",
            "geo_code": r["geo_code"],
            "geo_name": r["geo_name"],
            "parent_geo_code": "CN",
            "parent_geo_name": "中国",
            "scale": region_scale[r["geo_code"]],
        }
    )

province_by_code = {p["geo_code"]: p for p in provinces}

cities_raw = [
    ("CN-GZ-GZ", "广州", "CN-GD", 0.18),
    ("CN-GD-SZ", "深圳", "CN-GD", 0.22),
    ("CN-ZJ-HZ", "杭州", "CN-ZJ", 0.21),
    ("CN-JS-NJ", "南京", "CN-JS", 0.15),
    ("CN-HB-WH", "武汉", "CN-HB", 0.24),
    ("CN-SC-CD", "成都", "CN-SC", 0.24),
    ("CN-SN-XA", "西安", "CN-SN", 0.23),
    ("CN-JS-SZ", "苏州", "CN-JS", 0.22),
    ("CN-SD-QD", "青岛", "CN-SD", 0.16),
    ("CN-ZJ-NB", "宁波", "CN-ZJ", 0.14),
]

city_geos = []
for code, name, p_code, city_share in cities_raw:
    p = province_by_code[p_code]
    city_geos.append(
        {
            "geo_level": "city",
            "geo_code": code,
            "geo_name": name,
            "parent_geo_code": p_code,
            "parent_geo_name": p["geo_name"],
            "scale": p["scale"] * city_share,
        }
    )

national_geo = {
    "geo_level": "national",
    "geo_code": "CN",
    "geo_name": "中国",
    "parent_geo_code": "",
    "parent_geo_name": "",
    "scale": 1.0,
}

all_geos = [national_geo] + region_geos + provinces + city_geos
no_city_geos = [national_geo] + region_geos + provinces
energy_geos = [national_geo] + region_geos
national_only = [national_geo]


def geo_ratio(geo: dict, key: str, amp: float = 0.05) -> float:
    if geo["geo_level"] == "national":
        return 1.0
    return 1.0 + dnoise(f"{geo['geo_code']}:{key}", amp)


def round2(v: float) -> float:
    return round(v, 2)


def round3(v: float) -> float:
    return round(v, 3)


def add_common(row: dict, geo: dict, src: str, stat_date: date, freq: str, period: int | None = None) -> None:
    row["geo_level"] = geo["geo_level"]
    row["geo_code"] = geo["geo_code"]
    row["geo_name"] = geo["geo_name"]
    row["parent_geo_code"] = geo["parent_geo_code"]
    row["parent_geo_name"] = geo["parent_geo_name"]
    row["source_org"] = src
    row["release_date"] = release_date(stat_date, freq)
    row["data_status"] = status(freq, stat_date.year, period)


def write_csv(filename: str, headers: list[str], rows: list[dict]) -> None:
    path = OUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# 预计算全国 CPI 同比，用于实际利率和实际消费增速
nat_cpi_yoy = {}
for y, m, _ in month_iter():
    t = (y - BASE_YEAR) * 12 + (m - 1)
    season = math.sin(2 * math.pi * m / 12)
    nat_cpi_yoy[(y, m)] = clamp(0.8 + 0.12 * (y - BASE_YEAR) + 0.5 * season + dnoise(f"cpi_nat:{y}-{m}", 0.25), -0.8, 3.8)


# 1) gdp_expenditure
rows = []
# 全国名义GDP基准序列（亿元，2016-2025，训练用）
annual_gdp = {
    2016: 744127.0,
    2017: 832036.0,
    2018: 919281.0,
    2019: 990865.0,
    2020: 1015986.0,
    2021: 1149237.0,
    2022: 1210207.0,
    2023: 1260582.0,
    2024: 1323000.0,
    2025: 1388000.0,
}
q_share = {1: 0.22, 2: 0.24, 3: 0.26, 4: 0.28}
for y, q, d in quarter_iter():
    nat_nominal = annual_gdp[y] * q_share[q] * (1.0 + dnoise(f"nat:gdp:{y}Q{q}", 0.008))
    nat_yoy = clamp(4.8 + 0.35 * (y - BASE_YEAR) + 0.3 * (q - 2) + dnoise(f"nat:gdp_yoy:{y}Q{q}", 0.35), 2.0, 8.5)
    for geo in all_geos:
        scale = geo["scale"] * geo_ratio(geo, f"gdp:{y}Q{q}", 0.03)
        nominal = nat_nominal if geo["geo_level"] == "national" else nat_nominal * scale
        real = nominal / (1.0 + 0.018 + 0.003 * (y - BASE_YEAR))
        c_ratio = clamp(0.53 + dnoise(f"c_ratio:{geo['geo_code']}:{y}Q{q}", 0.02), 0.46, 0.62)
        i_ratio = clamp(0.42 + dnoise(f"i_ratio:{geo['geo_code']}:{y}Q{q}", 0.02), 0.34, 0.50)
        g_ratio = clamp(0.15 + dnoise(f"g_ratio:{geo['geo_code']}:{y}Q{q}", 0.012), 0.10, 0.20)
        consumption = nominal * c_ratio
        investment = nominal * i_ratio
        government = nominal * g_ratio
        net_export = nominal - consumption - investment - government
        exports = nominal * clamp(0.20 + dnoise(f"exp:{geo['geo_code']}:{y}Q{q}", 0.025), 0.12, 0.30)
        imports = max(1.0, exports - net_export)
        growth = nat_yoy + dnoise(f"gdp_growth:{geo['geo_code']}:{y}Q{q}", 0.8)
        contrib_c = clamp(58 + dnoise(f"contrib_c:{geo['geo_code']}:{y}Q{q}", 9), 35, 85)
        contrib_i = clamp(31 + dnoise(f"contrib_i:{geo['geo_code']}:{y}Q{q}", 8), 5, 55)
        contrib_n = clamp(100 - contrib_c - contrib_i, -25, 35)

        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "quarter": q,
            "nominal_gdp_100m_cny": round2(nominal),
            "real_gdp_100m_cny": round2(real),
            "consumption_100m_cny": round2(consumption),
            "investment_100m_cny": round2(investment),
            "government_spending_100m_cny": round2(government),
            "exports_100m_cny": round2(exports),
            "imports_100m_cny": round2(imports),
            "net_export_100m_cny": round2(net_export),
            "gdp_yoy_pct": round3(growth),
            "consumption_contrib_pct": round3(contrib_c),
            "investment_contrib_pct": round3(contrib_i),
            "net_export_contrib_pct": round3(contrib_n),
            "seasonal_adj": pick(f"sa:{geo['geo_code']}", ["SA", "NSA", "NSA"]),
        }
        add_common(row, geo, "国家统计局", d, "Q", q)
        rows.append(row)

write_csv(
    "gdp_expenditure.csv",
    [
        "ref_date",
        "year",
        "quarter",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "nominal_gdp_100m_cny",
        "real_gdp_100m_cny",
        "consumption_100m_cny",
        "investment_100m_cny",
        "government_spending_100m_cny",
        "exports_100m_cny",
        "imports_100m_cny",
        "net_export_100m_cny",
        "gdp_yoy_pct",
        "consumption_contrib_pct",
        "investment_contrib_pct",
        "net_export_contrib_pct",
        "seasonal_adj",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 2) cpi_inflation
rows = []
for y, m, d in month_iter():
    t = (y - BASE_YEAR) * 12 + (m - 1)
    season = math.sin(2 * math.pi * m / 12)
    nat_overall = 100 + 0.17 * t + 0.35 * season
    nat_yoy = nat_cpi_yoy[(y, m)]
    nat_mom = clamp(0.05 + 0.22 * season + dnoise(f"mom_nat:{y}-{m}", 0.15), -1.2, 1.8)
    for geo in all_geos:
        shift = dnoise(f"cpi_shift:{geo['geo_code']}", 1.2)
        overall = nat_overall + shift + dnoise(f"cpi_idx:{geo['geo_code']}:{y}-{m}", 0.4)
        food = overall + dnoise(f"food:{geo['geo_code']}:{y}-{m}", 2.2)
        energy = overall + dnoise(f"energy:{geo['geo_code']}:{y}-{m}", 3.2)
        core = overall - 0.5 + dnoise(f"core:{geo['geo_code']}:{y}-{m}", 0.6)
        services = overall + 0.7 + dnoise(f"services:{geo['geo_code']}:{y}-{m}", 0.8)
        if geo["geo_code"] == "CN":
            yoy = nat_yoy
            mom = nat_mom
        else:
            yoy = clamp(nat_yoy + dnoise(f"cpi_yoy:{geo['geo_code']}:{y}-{m}", 0.9), -3.0, 8.0)
            mom = clamp(nat_mom + dnoise(f"cpi_mom:{geo['geo_code']}:{y}-{m}", 0.4), -2.0, 3.0)
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "overall_cpi_idx": round3(overall),
            "food_cpi_idx": round3(food),
            "energy_cpi_idx": round3(energy),
            "core_cpi_idx": round3(core),
            "services_cpi_idx": round3(services),
            "yoy_change_pct": round3(yoy),
            "mom_change_pct": round3(mom),
            "base_period": "2016-01=100",
        }
        add_common(row, geo, "国家统计局", d, "M", m)
        rows.append(row)

write_csv(
    "cpi_inflation.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "overall_cpi_idx",
        "food_cpi_idx",
        "energy_cpi_idx",
        "core_cpi_idx",
        "services_cpi_idx",
        "yoy_change_pct",
        "mom_change_pct",
        "base_period",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 3) employment_stats
rows = []
for y, m, d in month_iter():
    season = math.sin(2 * math.pi * m / 12)
    nat_labor = 87800 + 85 * (y - BASE_YEAR) + dnoise(f"labor:{y}-{m}", 120)
    nat_unemp = clamp(5.25 - 0.05 * (y - BASE_YEAR) + 0.10 * season + dnoise(f"unemp_nat:{y}-{m}", 0.18), 4.2, 6.8)
    nat_part = clamp(66.2 + 0.08 * (y - BASE_YEAR) + dnoise(f"part_nat:{y}-{m}", 0.15), 64.0, 69.5)
    nat_youth = clamp(15.8 - 0.3 * (y - BASE_YEAR) + dnoise(f"youth_nat:{y}-{m}", 0.9), 10.0, 22.0)
    nat_wage = 9150 + 175 * (y - BASE_YEAR) + 90 * season
    nat_new_jobs = clamp(102 + 6 * (y - BASE_YEAR) + 4 * season + dnoise(f"jobs_nat:{y}-{m}", 6), 70, 150)

    for geo in all_geos:
        scale = geo["scale"] * geo_ratio(geo, f"emp:{y}-{m}", 0.03)
        if geo["geo_level"] == "national":
            labor = nat_labor
        else:
            labor = nat_labor * scale
        unemp = clamp(nat_unemp + dnoise(f"u:{geo['geo_code']}:{y}-{m}", 0.7), 3.2, 9.5)
        employed = labor * (1 - unemp / 100)
        unemployed = labor - employed
        youth = clamp(nat_youth + dnoise(f"yu:{geo['geo_code']}:{y}-{m}", 2.0), 8.0, 28.0)
        part = clamp(nat_part + dnoise(f"pa:{geo['geo_code']}:{y}-{m}", 1.2), 58.0, 75.0)
        wage = nat_wage * (1 + dnoise(f"wg:{geo['geo_code']}", 0.25))
        if geo["geo_level"] != "national":
            wage = wage * (0.88 + 0.22 * geo_ratio(geo, "wg2", 0.05))
        new_jobs = nat_new_jobs if geo["geo_level"] == "national" else nat_new_jobs * scale * (1 + dnoise(f"nj:{geo['geo_code']}:{y}-{m}", 0.15))

        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "labor_force_10k": round2(labor),
            "employed_10k": round2(employed),
            "unemployed_10k": round2(unemployed),
            "unemployment_rate_pct": round3(unemp),
            "youth_unemp_rate_pct": round3(youth),
            "participation_rate_pct": round3(part),
            "avg_monthly_wage_cny": round2(wage),
            "new_urban_jobs_10k": round2(new_jobs),
        }
        add_common(row, geo, "国家统计局", d, "M", m)
        rows.append(row)

write_csv(
    "employment_stats.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "labor_force_10k",
        "employed_10k",
        "unemployed_10k",
        "unemployment_rate_pct",
        "youth_unemp_rate_pct",
        "participation_rate_pct",
        "avg_monthly_wage_cny",
        "new_urban_jobs_10k",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 4) money_supply
rows = []
for y, m, d in month_iter():
    t = (y - BASE_YEAR) * 12 + (m - 1)
    nat_m2 = 2870000 + 9800 * t + dnoise(f"m2_nat:{y}-{m}", 16000)
    nat_m1 = 690000 + 2800 * t + dnoise(f"m1_nat:{y}-{m}", 7000)
    nat_m0 = 118000 + 650 * t + dnoise(f"m0_nat:{y}-{m}", 1500)
    nat_m2_yoy = clamp(8.8 - 0.08 * (y - BASE_YEAR) + dnoise(f"m2y_nat:{y}-{m}", 0.6), 6.0, 11.8)
    nat_m1_yoy = clamp(2.2 + 0.35 * (y - BASE_YEAR) + dnoise(f"m1y_nat:{y}-{m}", 1.2), -3.0, 8.0)
    nat_m0_yoy = clamp(7.0 + dnoise(f"m0y_nat:{y}-{m}", 1.5), 2.0, 14.0)
    nat_tsf = clamp(33500 + 280 * (m % 6) + dnoise(f"tsf_nat:{y}-{m}", 3800), 18000, 60000)
    nat_loan = 2480000 + 8500 * t + dnoise(f"loan_nat:{y}-{m}", 12000)
    nat_rrr = clamp(7.45 - 0.02 * (y - BASE_YEAR) + dnoise(f"rrr_nat:{y}-{m}", 0.06), 6.5, 8.5)

    for geo in no_city_geos:
        scale = geo["scale"] * geo_ratio(geo, f"money:{y}-{m}", 0.025)
        m2 = nat_m2 if geo["geo_level"] == "national" else nat_m2 * scale
        m1 = nat_m1 if geo["geo_level"] == "national" else nat_m1 * scale
        m0 = nat_m0 if geo["geo_level"] == "national" else nat_m0 * scale
        tsf = nat_tsf if geo["geo_level"] == "national" else nat_tsf * scale * (1 + dnoise(f"tsf:{geo['geo_code']}:{y}-{m}", 0.15))
        loan = nat_loan if geo["geo_level"] == "national" else nat_loan * scale
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "m0_100m_cny": round2(m0),
            "m1_100m_cny": round2(m1),
            "m2_100m_cny": round2(m2),
            "m0_yoy_pct": round3(clamp(nat_m0_yoy + dnoise(f"m0y:{geo['geo_code']}:{y}-{m}", 1.0), -5.0, 20.0)),
            "m1_yoy_pct": round3(clamp(nat_m1_yoy + dnoise(f"m1y:{geo['geo_code']}:{y}-{m}", 1.0), -8.0, 12.0)),
            "m2_yoy_pct": round3(clamp(nat_m2_yoy + dnoise(f"m2y:{geo['geo_code']}:{y}-{m}", 0.8), 4.0, 14.0)),
            "social_financing_flow_100m_cny": round2(tsf),
            "loan_balance_100m_cny": round2(loan),
            "reserve_ratio_pct": round3(clamp(nat_rrr + dnoise(f"rrr:{geo['geo_code']}:{y}-{m}", 0.15), 5.8, 9.0)),
        }
        add_common(row, geo, "中国人民银行", d, "M", m)
        rows.append(row)

write_csv(
    "money_supply.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "m0_100m_cny",
        "m1_100m_cny",
        "m2_100m_cny",
        "m0_yoy_pct",
        "m1_yoy_pct",
        "m2_yoy_pct",
        "social_financing_flow_100m_cny",
        "loan_balance_100m_cny",
        "reserve_ratio_pct",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 5) interest_rates
rows = []
for y, m, d in month_iter():
    season = math.sin(2 * math.pi * m / 12)
    policy = clamp(2.45 - 0.02 * (y - BASE_YEAR) + dnoise(f"policy:{y}-{m}", 0.03), 2.25, 2.60)
    lpr1y = clamp(3.45 - 0.03 * (y - BASE_YEAR) + dnoise(f"lpr1:{y}-{m}", 0.05), 3.20, 3.65)
    lpr5y = clamp(3.95 - 0.05 * (y - BASE_YEAR) + dnoise(f"lpr5:{y}-{m}", 0.06), 3.40, 4.30)
    repo7d = clamp(1.85 + 0.15 * season + dnoise(f"repo:{y}-{m}", 0.10), 1.4, 2.6)
    bond1y = clamp(1.90 + 0.10 * season + dnoise(f"b1:{y}-{m}", 0.11), 1.4, 2.8)
    bond10y = clamp(2.45 + 0.09 * season + dnoise(f"b10:{y}-{m}", 0.12), 1.9, 3.4)
    spread = int(round((bond10y - bond1y) * 100))
    real_rate = policy - nat_cpi_yoy[(y, m)]

    for geo in national_only:
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "policy_rate_pct": round3(policy),
            "lpr_1y_pct": round3(lpr1y),
            "lpr_5y_pct": round3(lpr5y),
            "repo_7d_pct": round3(repo7d),
            "bond_1y_pct": round3(bond1y),
            "bond_10y_pct": round3(bond10y),
            "yield_spread_10y_1y_bps": spread,
            "real_policy_rate_pct": round3(real_rate),
        }
        add_common(row, geo, "中国人民银行", d, "M", m)
        rows.append(row)

write_csv(
    "interest_rates.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "policy_rate_pct",
        "lpr_1y_pct",
        "lpr_5y_pct",
        "repo_7d_pct",
        "bond_1y_pct",
        "bond_10y_pct",
        "yield_spread_10y_1y_bps",
        "real_policy_rate_pct",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 6) industrial_output
rows = []
for y, m, d in month_iter():
    t = (y - BASE_YEAR) * 12 + (m - 1)
    season = math.sin(2 * math.pi * m / 12)
    nat_total_idx = 100 + 0.62 * t + 0.65 * season
    nat_yoy = clamp(4.8 + 0.25 * (y - BASE_YEAR) + 0.7 * season + dnoise(f"io_yoy:{y}-{m}", 0.6), 1.0, 11.0)
    nat_mom = clamp(0.45 + 0.15 * season + dnoise(f"io_mom:{y}-{m}", 0.25), -1.5, 2.8)
    nat_cap = clamp(75.0 + 0.2 * (y - BASE_YEAR) + dnoise(f"io_cap:{y}-{m}", 0.6), 70.0, 81.0)

    for geo in all_geos:
        s = geo_ratio(geo, f"io:{y}-{m}", 0.05)
        total_idx = nat_total_idx * s
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "total_index": round3(total_idx),
            "manufacturing_index": round3(total_idx * (1.01 + dnoise(f"mfg:{geo['geo_code']}:{y}-{m}", 0.03))),
            "mining_index": round3(total_idx * (0.94 + dnoise(f"min:{geo['geo_code']}:{y}-{m}", 0.05))),
            "utilities_index": round3(total_idx * (1.05 + dnoise(f"uti:{geo['geo_code']}:{y}-{m}", 0.04))),
            "hightech_index": round3(total_idx * (1.12 + dnoise(f"ht:{geo['geo_code']}:{y}-{m}", 0.05))),
            "equipment_mfg_index": round3(total_idx * (1.07 + dnoise(f"eq:{geo['geo_code']}:{y}-{m}", 0.05))),
            "capacity_utilization_pct": round3(clamp(nat_cap + dnoise(f"cap:{geo['geo_code']}:{y}-{m}", 2.2), 62, 90)),
            "yoy_growth_pct": round3(clamp(nat_yoy + dnoise(f"ioy:{geo['geo_code']}:{y}-{m}", 1.6), -4, 18)),
            "mom_growth_sa_pct": round3(clamp(nat_mom + dnoise(f"iom:{geo['geo_code']}:{y}-{m}", 0.6), -3.5, 4.0)),
        }
        add_common(row, geo, "国家统计局", d, "M", m)
        rows.append(row)

write_csv(
    "industrial_output.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "total_index",
        "manufacturing_index",
        "mining_index",
        "utilities_index",
        "hightech_index",
        "equipment_mfg_index",
        "capacity_utilization_pct",
        "yoy_growth_pct",
        "mom_growth_sa_pct",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 7) retail_sales
rows = []
for y, m, d in month_iter():
    t = (y - BASE_YEAR) * 12 + (m - 1)
    season = math.sin(2 * math.pi * m / 12)
    nat_total = 36800 + 360 * t + 2200 * season + dnoise(f"ret_nat:{y}-{m}", 1200)
    nat_yoy = clamp(4.4 + 0.45 * (y - BASE_YEAR) + 0.9 * season + dnoise(f"ret_yoy_nat:{y}-{m}", 0.9), -2.0, 13.0)
    nat_mom = clamp(0.55 + 0.35 * season + dnoise(f"ret_mom_nat:{y}-{m}", 0.35), -3.5, 5.5)
    for geo in all_geos:
        scale = geo["scale"] * geo_ratio(geo, f"ret:{y}-{m}", 0.04)
        total = nat_total if geo["geo_level"] == "national" else nat_total * scale
        auto = total * clamp(0.11 + dnoise(f"auto:{geo['geo_code']}:{y}-{m}", 0.03), 0.05, 0.22)
        food = total * clamp(0.15 + dnoise(f"food:{geo['geo_code']}:{y}-{m}", 0.03), 0.08, 0.28)
        ecommerce = total * clamp(0.27 + dnoise(f"ecom:{geo['geo_code']}:{y}-{m}", 0.05), 0.12, 0.45)
        service = total * clamp(0.31 + dnoise(f"servret:{geo['geo_code']}:{y}-{m}", 0.05), 0.15, 0.52)
        yoy = clamp(nat_yoy + dnoise(f"ry:{geo['geo_code']}:{y}-{m}", 2.0), -10.0, 25.0)
        mom = clamp(nat_mom + dnoise(f"rm:{geo['geo_code']}:{y}-{m}", 0.9), -8.0, 10.0)
        real_yoy = yoy - (nat_cpi_yoy[(y, m)] + dnoise(f"real_adj:{geo['geo_code']}:{y}-{m}", 0.4))

        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "total_sales_100m_cny": round2(total),
            "auto_sales_100m_cny": round2(auto),
            "food_sales_100m_cny": round2(food),
            "ecommerce_sales_100m_cny": round2(ecommerce),
            "service_retail_100m_cny": round2(service),
            "yoy_change_pct": round3(yoy),
            "mom_change_sa_pct": round3(mom),
            "real_yoy_change_pct": round3(real_yoy),
        }
        add_common(row, geo, "国家统计局", d, "M", m)
        rows.append(row)

write_csv(
    "retail_sales.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "total_sales_100m_cny",
        "auto_sales_100m_cny",
        "food_sales_100m_cny",
        "ecommerce_sales_100m_cny",
        "service_retail_100m_cny",
        "yoy_change_pct",
        "mom_change_sa_pct",
        "real_yoy_change_pct",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 8) trade_balance
rows = []
export_cats = ["机电产品", "高新技术产品", "汽车", "纺织服装", "农产品", "化工产品", "钢材", "服务贸易"]
import_cats = ["原油", "集成电路", "铁矿砂", "天然气", "农产品", "化工品", "机械设备", "服务贸易"]
partners = ["美国", "欧盟", "东盟", "日本", "韩国", "中国香港", "俄罗斯", "巴西", "印度", "澳大利亚"]
for y, m, d in month_iter():
    season = math.sin(2 * math.pi * m / 12)
    nat_exports = clamp(315000 + 2400 * (y - BASE_YEAR) + 9200 * season + dnoise(f"ex_nat:{y}-{m}", 16000), 240000, 420000)
    nat_imports = clamp(246000 + 1800 * (y - BASE_YEAR) + 7600 * season + dnoise(f"im_nat:{y}-{m}", 15000), 180000, 360000)
    nat_ex_yoy = clamp(3.8 + 0.4 * (y - BASE_YEAR) + dnoise(f"exy_nat:{y}-{m}", 3.5), -15, 20)
    nat_im_yoy = clamp(2.5 + 0.5 * (y - BASE_YEAR) + dnoise(f"imy_nat:{y}-{m}", 3.5), -18, 22)

    for geo in all_geos:
        scale = geo["scale"] * geo_ratio(geo, f"trade:{y}-{m}", 0.07)
        exports = nat_exports if geo["geo_level"] == "national" else nat_exports * scale
        imports = nat_imports if geo["geo_level"] == "national" else nat_imports * scale * (1 + dnoise(f"impadj:{geo['geo_code']}:{y}-{m}", 0.08))
        trade_bal = exports - imports
        ex_yoy = clamp(nat_ex_yoy + dnoise(f"exy:{geo['geo_code']}:{y}-{m}", 5.5), -35, 35)
        im_yoy = clamp(nat_im_yoy + dnoise(f"imy:{geo['geo_code']}:{y}-{m}", 5.5), -35, 35)

        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "exports_musd": round2(exports),
            "imports_musd": round2(imports),
            "trade_balance_musd": round2(trade_bal),
            "export_yoy_pct": round3(ex_yoy),
            "import_yoy_pct": round3(im_yoy),
            "top_export_category": pick(f"excat:{geo['geo_code']}:{y}-{m}", export_cats),
            "top_import_category": pick(f"imcat:{geo['geo_code']}:{y}-{m}", import_cats),
            "top_export_partner": pick(f"expar:{geo['geo_code']}:{y}-{m}", partners),
            "top_import_partner": pick(f"impar:{geo['geo_code']}:{y}-{m}", partners),
        }
        add_common(row, geo, "海关总署", d, "M", m)
        rows.append(row)

write_csv(
    "trade_balance.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "exports_musd",
        "imports_musd",
        "trade_balance_musd",
        "export_yoy_pct",
        "import_yoy_pct",
        "top_export_category",
        "top_import_category",
        "top_export_partner",
        "top_import_partner",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 9) real_estate
rows = []
for y, m, d in month_iter():
    season = math.sin(2 * math.pi * m / 12)
    nat_price_idx = 100.0 - 0.28 * ((y - BASE_YEAR) * 12 + (m - 1)) / 12 + dnoise(f"hp_nat:{y}-{m}", 0.8)
    nat_new_sales = clamp(9650 + 140 * season + dnoise(f"new_nat:{y}-{m}", 700) - 130 * (y - BASE_YEAR), 6500, 12000)
    nat_old_sales = clamp(12800 + 180 * season + dnoise(f"old_nat:{y}-{m}", 900) - 100 * (y - BASE_YEAR), 8500, 16000)
    nat_starts = clamp(7600 + dnoise(f"starts_nat:{y}-{m}", 900) - 120 * (y - BASE_YEAR), 4200, 11000)
    nat_invest = clamp(9000 + 150 * season + dnoise(f"inv_nat:{y}-{m}", 900) - 80 * (y - BASE_YEAR), 6500, 13000)
    nat_mortgage = clamp(4.05 - 0.18 * (y - BASE_YEAR) + dnoise(f"mort_nat:{y}-{m}", 0.08), 3.2, 4.8)
    nat_vacancy = clamp(13.5 + 0.25 * (y - BASE_YEAR) + dnoise(f"vac_nat:{y}-{m}", 0.9), 8.0, 22.0)
    nat_inventory = clamp(14.2 + 0.2 * (y - BASE_YEAR) + dnoise(f"invmo_nat:{y}-{m}", 1.2), 8.0, 24.0)
    nat_price_yoy = clamp(-0.8 - 0.35 * (y - BASE_YEAR) + dnoise(f"hpy_nat:{y}-{m}", 1.0), -7.0, 4.0)

    for geo in all_geos:
        scale = geo["scale"] * geo_ratio(geo, f"re:{y}-{m}", 0.06)
        new_sales = nat_new_sales if geo["geo_level"] == "national" else nat_new_sales * scale
        old_sales = nat_old_sales if geo["geo_level"] == "national" else nat_old_sales * scale
        starts = nat_starts if geo["geo_level"] == "national" else nat_starts * scale
        invest = nat_invest if geo["geo_level"] == "national" else nat_invest * scale
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "home_price_idx": round3(nat_price_idx + dnoise(f"hpi:{geo['geo_code']}:{y}-{m}", 3.0)),
            "new_home_sales_10k_sqm": round2(new_sales),
            "existing_home_sales_10k_sqm": round2(old_sales),
            "housing_starts_10k_sqm": round2(starts),
            "real_estate_investment_100m_cny": round2(invest),
            "mortgage_rate_pct": round3(clamp(nat_mortgage + dnoise(f"mr:{geo['geo_code']}:{y}-{m}", 0.35), 2.8, 6.0)),
            "vacancy_rate_pct": round3(clamp(nat_vacancy + dnoise(f"vac:{geo['geo_code']}:{y}-{m}", 3.2), 4.0, 30.0)),
            "inventory_months": round3(clamp(nat_inventory + dnoise(f"imo:{geo['geo_code']}:{y}-{m}", 3.0), 4.0, 36.0)),
            "home_price_yoy_pct": round3(clamp(nat_price_yoy + dnoise(f"hpy:{geo['geo_code']}:{y}-{m}", 2.0), -15.0, 10.0)),
        }
        add_common(row, geo, "国家统计局", d, "M", m)
        rows.append(row)

write_csv(
    "real_estate.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "home_price_idx",
        "new_home_sales_10k_sqm",
        "existing_home_sales_10k_sqm",
        "housing_starts_10k_sqm",
        "real_estate_investment_100m_cny",
        "mortgage_rate_pct",
        "vacancy_rate_pct",
        "inventory_months",
        "home_price_yoy_pct",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 10) energy_prices
rows = []
for y, m, d in month_iter():
    season = math.sin(2 * math.pi * m / 12)
    wti = clamp(78 + 2.0 * season + 1.5 * (y - 2024) + dnoise(f"wti:{y}-{m}", 4.5), 55, 105)
    brent = clamp(wti + 3.5 + dnoise(f"brent:{y}-{m}", 1.6), 58, 110)
    lng = clamp(12 + 2.1 * season + dnoise(f"lng:{y}-{m}", 2.8), 5, 26)
    coal = clamp(890 - 18 * (y - BASE_YEAR) + dnoise(f"coalp:{y}-{m}", 90), 520, 1450)
    elec = clamp(0.575 + dnoise(f"elec:{y}-{m}", 0.03), 0.40, 0.85)
    carbon = clamp(72 + 3.5 * (y - BASE_YEAR) + dnoise(f"carbon:{y}-{m}", 8), 35, 145)
    renew = clamp(31.5 + 1.2 * (y - BASE_YEAR) + 1.3 * season + dnoise(f"renew:{y}-{m}", 1.0), 20, 55)
    nat_power = clamp(7600 + 180 * season + 65 * (y - BASE_YEAR) + dnoise(f"pow_nat:{y}-{m}", 280), 6200, 9800)

    for geo in energy_geos:
        scale = geo["scale"] * geo_ratio(geo, f"eng:{y}-{m}", 0.03)
        power = nat_power if geo["geo_level"] == "national" else nat_power * scale
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "coal_price_cny_per_ton": round3(clamp(coal + dnoise(f"coal:{geo['geo_code']}:{y}-{m}", 75), 450, 1600)),
            "lng_price_usd_per_mmbtu": round3(clamp(lng + dnoise(f"lng_r:{geo['geo_code']}:{y}-{m}", 2.1), 4, 30)),
            "wti_usd_per_bbl": round3(wti),
            "brent_usd_per_bbl": round3(brent),
            "electricity_price_cny_per_kwh": round3(clamp(elec + dnoise(f"elec_r:{geo['geo_code']}:{y}-{m}", 0.06), 0.35, 1.0)),
            "carbon_price_cny_per_tco2": round3(clamp(carbon + dnoise(f"car_r:{geo['geo_code']}:{y}-{m}", 10), 25, 180)),
            "renewable_generation_share_pct": round3(clamp(renew + dnoise(f"ren_r:{geo['geo_code']}:{y}-{m}", 2.2), 10, 70)),
            "power_generation_100m_kwh": round2(power),
        }
        add_common(row, geo, "国家能源局", d, "M", m)
        rows.append(row)

write_csv(
    "energy_prices.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "coal_price_cny_per_ton",
        "lng_price_usd_per_mmbtu",
        "wti_usd_per_bbl",
        "brent_usd_per_bbl",
        "electricity_price_cny_per_kwh",
        "carbon_price_cny_per_tco2",
        "renewable_generation_share_pct",
        "power_generation_100m_kwh",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 11) stock_indices
rows = []
for y, m, d in month_iter():
    t = (y - BASE_YEAR) * 12 + (m - 1)
    sh = 3150 + 9 * t + 80 * math.sin(2 * math.pi * m / 12) + dnoise(f"sh:{y}-{m}", 120)
    sz = 10300 + 25 * t + 220 * math.sin(2 * math.pi * m / 12) + dnoise(f"sz:{y}-{m}", 300)
    csi300 = 3850 + 12 * t + 90 * math.sin(2 * math.pi * m / 12) + dnoise(f"csi:{y}-{m}", 140)
    chinext = 2180 + 11 * t + 130 * math.sin(2 * math.pi * m / 12) + dnoise(f"cyb:{y}-{m}", 210)
    vol = clamp(20 - 0.2 * (y - BASE_YEAR) + dnoise(f"vol:{y}-{m}", 4.5), 10, 45)
    pe = clamp(12.4 + 0.25 * (y - BASE_YEAR) + dnoise(f"pe:{y}-{m}", 1.3), 8, 24)
    turnover = clamp(182000 + 1200 * t + dnoise(f"turn:{y}-{m}", 30000), 100000, 320000)
    northbound = clamp(180 + dnoise(f"nb:{y}-{m}", 1250), -3800, 4200)

    for geo in national_only:
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "shanghai_comp_close": round3(sh),
            "shenzhen_comp_close": round3(sz),
            "csi300_close": round3(csi300),
            "chinext_close": round3(chinext),
            "volatility_idx": round3(vol),
            "pe_ttm_csi300": round3(pe),
            "market_turnover_100m_cny": round2(turnover),
            "northbound_netflow_100m_cny": round2(northbound),
        }
        add_common(row, geo, "沪深交易所", d, "M", m)
        rows.append(row)

write_csv(
    "stock_indices.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "shanghai_comp_close",
        "shenzhen_comp_close",
        "csi300_close",
        "chinext_close",
        "volatility_idx",
        "pe_ttm_csi300",
        "market_turnover_100m_cny",
        "northbound_netflow_100m_cny",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 12) exchange_rates
rows = []
for y, m, d in month_iter():
    season = math.sin(2 * math.pi * m / 12)
    usd_cny = clamp(7.05 + 0.04 * season + dnoise(f"usdcny:{y}-{m}", 0.12), 6.5, 7.7)
    eur_cny = clamp(7.75 + 0.10 * season + dnoise(f"eurcny:{y}-{m}", 0.18), 6.9, 8.8)
    jpy_cny = clamp(0.047 + dnoise(f"jpycny:{y}-{m}", 0.004), 0.038, 0.065)
    gbp_cny = clamp(8.95 + 0.12 * season + dnoise(f"gbpcny:{y}-{m}", 0.20), 7.8, 10.2)
    cfets = clamp(100.5 + dnoise(f"cfets:{y}-{m}", 2.3), 92, 109)
    dxy = clamp(103.0 + 0.3 * math.cos(2 * math.pi * m / 12) + dnoise(f"dxy:{y}-{m}", 3.0), 92, 115)
    fx_res = clamp(31900 + 110 * (y - BASE_YEAR) + dnoise(f"fxres:{y}-{m}", 420), 30000, 34000)
    fx_vol = clamp(5.3 + dnoise(f"fxvol:{y}-{m}", 1.2), 2.0, 12.0)

    for geo in national_only:
        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "month": m,
            "usd_cny": round(usd_cny, 4),
            "eur_cny": round(eur_cny, 4),
            "jpy_cny": round(jpy_cny, 6),
            "gbp_cny": round(gbp_cny, 4),
            "cfets_index": round3(cfets),
            "dxy_index": round3(dxy),
            "fx_reserve_100m_usd": round2(fx_res),
            "fx_vol_1m_pct": round3(fx_vol),
        }
        add_common(row, geo, "国家外汇管理局", d, "M", m)
        rows.append(row)

write_csv(
    "exchange_rates.csv",
    [
        "ref_date",
        "year",
        "month",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "usd_cny",
        "eur_cny",
        "jpy_cny",
        "gbp_cny",
        "cfets_index",
        "dxy_index",
        "fx_reserve_100m_usd",
        "fx_vol_1m_pct",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 13) corporate_credit
rows = []
sectors = [
    "科技",
    "金融",
    "能源",
    "消费",
    "工业",
    "医疗健康",
    "公用事业",
    "地产",
    "材料",
    "通信服务",
    "可选消费",
]
for y, q, d in quarter_iter():
    for geo in no_city_geos:
        scale = geo["scale"] * geo_ratio(geo, f"cc:{y}Q{q}", 0.04)
        for sec in sectors:
            sec_risk = 0.0
            if sec in {"地产", "可选消费"}:
                sec_risk = 1.2
            elif sec in {"公用事业", "金融"}:
                sec_risk = -0.5
            ig = clamp(3.35 + 0.08 * (y - BASE_YEAR) + 0.10 * (q - 2) + sec_risk * 0.08 + dnoise(f"ig:{geo['geo_code']}:{sec}:{y}Q{q}", 0.35), 2.0, 7.0)
            hy = clamp(7.6 + 0.15 * (y - BASE_YEAR) + sec_risk * 0.45 + dnoise(f"hy:{geo['geo_code']}:{sec}:{y}Q{q}", 0.9), 4.0, 16.0)
            ig_sp = int(round(clamp(120 + sec_risk * 18 + dnoise(f"igsp:{geo['geo_code']}:{sec}:{y}Q{q}", 26), 40, 380)))
            hy_sp = int(round(clamp(460 + sec_risk * 45 + dnoise(f"hysp:{geo['geo_code']}:{sec}:{y}Q{q}", 70), 160, 1200)))
            default = clamp(1.05 + sec_risk * 0.45 + dnoise(f"def:{geo['geo_code']}:{sec}:{y}Q{q}", 0.6), 0.1, 9.0)
            issuance_nat = 42800 + 1800 * (y - BASE_YEAR) + 650 * (q - 1)
            issuance = issuance_nat if geo["geo_level"] == "national" else issuance_nat * scale * (1 + dnoise(f"iss:{geo['geo_code']}:{sec}:{y}Q{q}", 0.2))
            down_ratio = clamp(3.2 + sec_risk * 1.1 + dnoise(f"down:{geo['geo_code']}:{sec}:{y}Q{q}", 1.1), 0.3, 18)
            npl = clamp(1.35 + sec_risk * 0.28 + dnoise(f"npl:{geo['geo_code']}:{sec}:{y}Q{q}", 0.4), 0.3, 6.0)
            row = {
                "ref_date": d.isoformat(),
                "year": y,
                "quarter": q,
                "sector": sec,
                "inv_grade_yield_pct": round3(ig),
                "high_yield_yield_pct": round3(hy),
                "ig_spread_bps": ig_sp,
                "hy_spread_bps": hy_sp,
                "default_rate_pct": round3(default),
                "new_issuance_100m_cny": round2(issuance),
                "downgrade_ratio_pct": round3(down_ratio),
                "npl_ratio_pct": round3(npl),
            }
            add_common(row, geo, "银行间市场交易商协会", d, "Q", q)
            rows.append(row)

write_csv(
    "corporate_credit.csv",
    [
        "ref_date",
        "year",
        "quarter",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "sector",
        "inv_grade_yield_pct",
        "high_yield_yield_pct",
        "ig_spread_bps",
        "hy_spread_bps",
        "default_rate_pct",
        "new_issuance_100m_cny",
        "downgrade_ratio_pct",
        "npl_ratio_pct",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 14) regional_gdp
rows = []
nat_population = {
    2016: 141200,
    2017: 141260,
    2018: 141270,
    2019: 141220,
    2020: 141180,
    2021: 141130,
    2022: 141050,
    2023: 141000,
    2024: 140900,
    2025: 140800,
}  # 万人
nat_gdp = annual_gdp.copy()  # 亿元
for y, d in year_iter():
    for geo in all_geos:
        if geo["geo_level"] == "national":
            gdp = nat_gdp[y]
            pop = nat_population[y]
            share = 100.0
        else:
            gdp = nat_gdp[y] * geo["scale"] * geo_ratio(geo, f"rgdp:{y}", 0.04)
            pop = nat_population[y] * geo["scale"] * geo_ratio(geo, f"pop:{y}", 0.03)
            share = (gdp / nat_gdp[y]) * 100
        growth = clamp(4.9 + 0.3 * (y - BASE_YEAR) + dnoise(f"grow:{geo['geo_code']}:{y}", 1.0), 1.0, 10.5)
        per_capita = gdp * 1e8 / (max(pop, 1.0) * 1e4)
        pri = clamp(7.8 + dnoise(f"pri:{geo['geo_code']}:{y}", 5.0), 1.0, 24.0)
        sec = clamp(37.5 + dnoise(f"sec:{geo['geo_code']}:{y}", 8.0), 18.0, 58.0)
        ter = clamp(100 - pri - sec, 30.0, 78.0)
        sec = clamp(100 - pri - ter, 10.0, 65.0)
        ind_va = gdp * sec / 100
        srv_va = gdp * ter / 100

        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "gdp_100m_cny": round2(gdp),
            "gdp_growth_pct": round3(growth),
            "gdp_per_capita_cny": round2(per_capita),
            "population_10k": round2(pop),
            "gdp_share_pct": round3(share),
            "primary_share_pct": round3(pri),
            "secondary_share_pct": round3(sec),
            "tertiary_share_pct": round3(ter),
            "industry_value_added_100m_cny": round2(ind_va),
            "services_value_added_100m_cny": round2(srv_va),
        }
        add_common(row, geo, "国家统计局", d, "Y", None)
        rows.append(row)

write_csv(
    "regional_gdp.csv",
    [
        "ref_date",
        "year",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "gdp_100m_cny",
        "gdp_growth_pct",
        "gdp_per_capita_cny",
        "population_10k",
        "gdp_share_pct",
        "primary_share_pct",
        "secondary_share_pct",
        "tertiary_share_pct",
        "industry_value_added_100m_cny",
        "services_value_added_100m_cny",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


# 15) government_finance
rows = []
gov_geos = [national_geo] + region_geos + provinces
for y, d in year_iter():
    nat_rev = 216000 + 5200 * (y - BASE_YEAR)
    nat_exp = 278000 + 6800 * (y - BASE_YEAR)
    nat_debt = 680000 + 42000 * (y - BASE_YEAR)
    for geo in gov_geos:
        scale = geo["scale"] * geo_ratio(geo, f"gov:{y}", 0.04)
        if geo["geo_level"] == "national":
            rev = nat_rev
            exp = nat_exp
            debt = nat_debt
            gdp = nat_gdp[y]
        else:
            rev = nat_rev * scale
            exp = nat_exp * scale * (1 + dnoise(f"expadj:{geo['geo_code']}:{y}", 0.08))
            debt = nat_debt * scale * (1 + dnoise(f"debtadj:{geo['geo_code']}:{y}", 0.10))
            gdp = nat_gdp[y] * geo["scale"]

        tax = rev * clamp(0.79 + dnoise(f"tax:{geo['geo_code']}:{y}", 0.05), 0.60, 0.92)
        non_tax = rev - tax
        deficit = exp - rev
        debt_ratio = debt / max(gdp, 1) * 100
        fiscal_bal = (rev - exp) / max(gdp, 1) * 100
        transfer = exp * clamp(0.18 + dnoise(f"tr:{geo['geo_code']}:{y}", 0.05), 0.05, 0.42)
        social = exp * clamp(0.24 + dnoise(f"ss:{geo['geo_code']}:{y}", 0.06), 0.10, 0.45)

        row = {
            "ref_date": d.isoformat(),
            "year": y,
            "tax_revenue_100m_cny": round2(tax),
            "non_tax_revenue_100m_cny": round2(non_tax),
            "total_revenue_100m_cny": round2(rev),
            "expenditure_100m_cny": round2(exp),
            "deficit_100m_cny": round2(deficit),
            "debt_balance_100m_cny": round2(debt),
            "debt_to_gdp_pct": round3(debt_ratio),
            "fiscal_balance_pct_gdp": round3(fiscal_bal),
            "transfer_payment_100m_cny": round2(transfer),
            "social_security_100m_cny": round2(social),
        }
        add_common(row, geo, "财政部", d, "Y", None)
        rows.append(row)

write_csv(
    "government_finance.csv",
    [
        "ref_date",
        "year",
        "geo_level",
        "geo_code",
        "geo_name",
        "parent_geo_code",
        "parent_geo_name",
        "tax_revenue_100m_cny",
        "non_tax_revenue_100m_cny",
        "total_revenue_100m_cny",
        "expenditure_100m_cny",
        "deficit_100m_cny",
        "debt_balance_100m_cny",
        "debt_to_gdp_pct",
        "fiscal_balance_pct_gdp",
        "transfer_payment_100m_cny",
        "social_security_100m_cny",
        "source_org",
        "release_date",
        "data_status",
    ],
    rows,
)


def load_csv(filename: str) -> tuple[list[str], list[dict]]:
    path = OUT_DIR / filename
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return headers, rows


def save_csv(filename: str, headers: list[str], rows: list[dict]) -> None:
    path = OUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def fval(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v in ("", None):
        return 0.0
    return float(v)


def round_by_field(field: str, value: float):
    if field.endswith("_bps"):
        return int(round(value))
    if field.endswith("_pct") or field.endswith("_idx") or field.endswith("_cny_per_kwh") or field.endswith("_mmbtu"):
        return round(value, 3)
    if field in {"usd_cny", "eur_cny", "gbp_cny"}:
        return round(value, 4)
    if field == "jpy_cny":
        return round(value, 6)
    return round(value, 2)


def rebalance_additive_levels(rows: list[dict], key_fields: list[str], value_fields: list[str]) -> None:
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in key_fields)].append(r)

    def scale_group(group_rows: list[dict], field: str, target: float) -> None:
        if not group_rows:
            return
        cur = sum(fval(r, field) for r in group_rows)
        if abs(cur) < 1e-12:
            each = target / len(group_rows)
            for r in group_rows:
                r[field] = round_by_field(field, each)
            return
        ratio = target / cur
        for r in group_rows:
            r[field] = round_by_field(field, fval(r, field) * ratio)

    for g_rows in grouped.values():
        nat = next((r for r in g_rows if r.get("geo_level") == "national"), None)
        regions = [r for r in g_rows if r.get("geo_level") == "region"]
        provinces = [r for r in g_rows if r.get("geo_level") == "province"]

        if nat and regions:
            for field in value_fields:
                scale_group(regions, field, fval(nat, field))

        if regions and provinces:
            for reg in regions:
                reg_code = reg["geo_code"]
                provs = [p for p in provinces if p.get("parent_geo_code") == reg_code]
                if not provs:
                    continue
                for field in value_fields:
                    scale_group(provs, field, fval(reg, field))


def postprocess_consistency() -> None:
    # CPI map for cross-table real terms
    _, cpi_rows = load_csv("cpi_inflation.csv")
    cpi_yoy = {(r["ref_date"], r["geo_code"]): fval(r, "yoy_change_pct") for r in cpi_rows}

    # regional_gdp: enforce hierarchy and derived identities
    rg_headers, rg_rows = load_csv("regional_gdp.csv")
    rebalance_additive_levels(
        rg_rows,
        ["ref_date", "year"],
        ["gdp_100m_cny", "population_10k", "industry_value_added_100m_cny", "services_value_added_100m_cny"],
    )
    rg_by_year_geo = {}
    grouped_rg = defaultdict(list)
    for r in rg_rows:
        grouped_rg[r["year"]].append(r)
    for year, rows_y in grouped_rg.items():
        nat = next(r for r in rows_y if r["geo_level"] == "national")
        nat_gdp = max(fval(nat, "gdp_100m_cny"), 1.0)
        for r in rows_y:
            gdp = max(fval(r, "gdp_100m_cny"), 0.01)
            pop = max(fval(r, "population_10k"), 0.01)
            pri = clamp(fval(r, "primary_share_pct"), 1.0, 30.0)
            sec = clamp(fval(r, "secondary_share_pct"), 10.0, 70.0)
            ter = 100.0 - pri - sec
            if ter < 15.0:
                sec = 85.0 - pri
                ter = 15.0
            r["primary_share_pct"] = round3(pri)
            r["secondary_share_pct"] = round3(sec)
            r["tertiary_share_pct"] = round3(ter)
            r["industry_value_added_100m_cny"] = round2(gdp * sec / 100.0)
            r["services_value_added_100m_cny"] = round2(gdp * ter / 100.0)
            r["gdp_per_capita_cny"] = round2(gdp * 1e8 / (pop * 1e4))
            r["gdp_share_pct"] = round3(100.0 if r["geo_level"] == "national" else gdp / nat_gdp * 100.0)
            rg_by_year_geo[(year, r["geo_code"])] = gdp
    save_csv("regional_gdp.csv", rg_headers, rg_rows)

    # gdp_expenditure: enforce additive hierarchy and accounting identities
    gdp_headers, gdp_rows = load_csv("gdp_expenditure.csv")
    rebalance_additive_levels(
        gdp_rows,
        ["ref_date", "year", "quarter"],
        [
            "nominal_gdp_100m_cny",
            "real_gdp_100m_cny",
            "consumption_100m_cny",
            "investment_100m_cny",
            "government_spending_100m_cny",
            "exports_100m_cny",
        ],
    )
    for r in gdp_rows:
        nominal = fval(r, "nominal_gdp_100m_cny")
        cons = fval(r, "consumption_100m_cny")
        inv = fval(r, "investment_100m_cny")
        gov = fval(r, "government_spending_100m_cny")
        exports = fval(r, "exports_100m_cny")
        net = nominal - cons - inv - gov
        imports = exports - net
        if imports < 0:
            imports = 0.0
            net = exports
        r["net_export_100m_cny"] = round2(net)
        r["imports_100m_cny"] = round2(imports)
        cc = fval(r, "consumption_contrib_pct")
        ic = fval(r, "investment_contrib_pct")
        r["net_export_contrib_pct"] = round3(100.0 - cc - ic)
    save_csv("gdp_expenditure.csv", gdp_headers, gdp_rows)

    # trade_balance: enforce additive hierarchy + strict identity
    tr_headers, tr_rows = load_csv("trade_balance.csv")
    rebalance_additive_levels(tr_rows, ["ref_date", "year", "month"], ["exports_musd", "imports_musd"])
    for r in tr_rows:
        r["trade_balance_musd"] = round2(fval(r, "exports_musd") - fval(r, "imports_musd"))
    save_csv("trade_balance.csv", tr_headers, tr_rows)

    # retail_sales: enforce additive hierarchy and real growth linkage with CPI
    rt_headers, rt_rows = load_csv("retail_sales.csv")
    rebalance_additive_levels(
        rt_rows,
        ["ref_date", "year", "month"],
        [
            "total_sales_100m_cny",
            "auto_sales_100m_cny",
            "food_sales_100m_cny",
            "ecommerce_sales_100m_cny",
            "service_retail_100m_cny",
        ],
    )
    for r in rt_rows:
        key = (r["ref_date"], r["geo_code"])
        if key in cpi_yoy:
            r["real_yoy_change_pct"] = round3(fval(r, "yoy_change_pct") - cpi_yoy[key])
    save_csv("retail_sales.csv", rt_headers, rt_rows)

    # money_supply: enforce hierarchy and M2>=M1>=M0
    ms_headers, ms_rows = load_csv("money_supply.csv")
    rebalance_additive_levels(
        ms_rows,
        ["ref_date", "year", "month"],
        ["m0_100m_cny", "m1_100m_cny", "m2_100m_cny", "social_financing_flow_100m_cny", "loan_balance_100m_cny"],
    )
    for r in ms_rows:
        m0 = fval(r, "m0_100m_cny")
        m1 = max(fval(r, "m1_100m_cny"), m0)
        m2 = max(fval(r, "m2_100m_cny"), m1)
        r["m0_100m_cny"] = round2(m0)
        r["m1_100m_cny"] = round2(m1)
        r["m2_100m_cny"] = round2(m2)
    save_csv("money_supply.csv", ms_headers, ms_rows)

    # real_estate: enforce additive hierarchy for volume/investment indicators
    re_headers, re_rows = load_csv("real_estate.csv")
    rebalance_additive_levels(
        re_rows,
        ["ref_date", "year", "month"],
        ["new_home_sales_10k_sqm", "existing_home_sales_10k_sqm", "housing_starts_10k_sqm", "real_estate_investment_100m_cny"],
    )
    save_csv("real_estate.csv", re_headers, re_rows)

    # corporate_credit: enforce hierarchy for issuance and spread order
    cc_headers, cc_rows = load_csv("corporate_credit.csv")
    rebalance_additive_levels(cc_rows, ["ref_date", "year", "quarter", "sector"], ["new_issuance_100m_cny"])
    for r in cc_rows:
        ig = fval(r, "inv_grade_yield_pct")
        hy = max(fval(r, "high_yield_yield_pct"), ig + 0.6)
        ig_sp = int(round(fval(r, "ig_spread_bps")))
        hy_sp = int(round(max(fval(r, "hy_spread_bps"), ig_sp + 80)))
        r["high_yield_yield_pct"] = round3(hy)
        r["ig_spread_bps"] = ig_sp
        r["hy_spread_bps"] = hy_sp
    save_csv("corporate_credit.csv", cc_headers, cc_rows)

    # government_finance: enforce hierarchy and derive ratios from regional_gdp
    gf_headers, gf_rows = load_csv("government_finance.csv")
    rebalance_additive_levels(
        gf_rows,
        ["ref_date", "year"],
        [
            "tax_revenue_100m_cny",
            "non_tax_revenue_100m_cny",
            "total_revenue_100m_cny",
            "expenditure_100m_cny",
            "debt_balance_100m_cny",
            "transfer_payment_100m_cny",
            "social_security_100m_cny",
        ],
    )
    for r in gf_rows:
        rev = fval(r, "total_revenue_100m_cny")
        tax = min(fval(r, "tax_revenue_100m_cny"), rev)
        exp = fval(r, "expenditure_100m_cny")
        debt = fval(r, "debt_balance_100m_cny")
        year = r["year"]
        geo = r["geo_code"]
        gdp = max(rg_by_year_geo.get((year, geo), 1.0), 1.0)
        r["tax_revenue_100m_cny"] = round2(tax)
        r["non_tax_revenue_100m_cny"] = round2(max(rev - tax, 0.0))
        r["deficit_100m_cny"] = round2(exp - rev)
        r["debt_to_gdp_pct"] = round3(debt / gdp * 100.0)
        r["fiscal_balance_pct_gdp"] = round3((rev - exp) / gdp * 100.0)
    save_csv("government_finance.csv", gf_headers, gf_rows)

    # interest_rates: strict real-rate identity with national CPI
    ir_headers, ir_rows = load_csv("interest_rates.csv")
    for r in ir_rows:
        key = (r["ref_date"], r["geo_code"])
        if key in cpi_yoy:
            r["real_policy_rate_pct"] = round3(fval(r, "policy_rate_pct") - cpi_yoy[key])
    save_csv("interest_rates.csv", ir_headers, ir_rows)


postprocess_consistency()
print("Generated 15 CSV files in", OUT_DIR)
