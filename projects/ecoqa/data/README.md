# 中国经济多表CSV训练集 README

本 README 提供 15 张 CSV 的中文数据字典与规模统计，适用于 SQL 检索/多表 Join 训练。

## 一、数据集概览

- 时间跨度：2016-01 到 2025-12
- 空间层级：全国 + 7大区域 + 31省级 + 10重点城市（按表覆盖）
- 数据类型：规则化仿真数据（趋势贴近现实，内部联动与恒等式校准）

## 二、各表规模统计

| 表名 | 中文名称 | 频率 | 行数 | 字段数 | 年份范围 |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `corporate_credit.csv` | 企业信贷表（季度，行业细分） | 季度 | 17160 | 20 | 2016-2025 |
| `cpi_inflation.csv` | CPI通胀表（月度） | 月度 | 5880 | 19 | 2016-2025 |
| `employment_stats.csv` | 就业市场表（月度） | 月度 | 5880 | 19 | 2016-2025 |
| `energy_prices.csv` | 能源价格表（月度） | 月度 | 960 | 19 | 2016-2025 |
| `exchange_rates.csv` | 汇率与外储表（月度） | 月度 | 120 | 19 | 2016-2025 |
| `gdp_expenditure.csv` | GDP构成表（支出法，季度） | 季度 | 1960 | 24 | 2016-2025 |
| `government_finance.csv` | 政府财政表（年度） | 年度 | 390 | 20 | 2016-2025 |
| `industrial_output.csv` | 工业产出表（月度） | 月度 | 5880 | 20 | 2016-2025 |
| `interest_rates.csv` | 利率表（月度） | 月度 | 120 | 19 | 2016-2025 |
| `money_supply.csv` | 货币供应量表（月度） | 月度 | 4680 | 20 | 2016-2025 |
| `real_estate.csv` | 房地产市场表（月度） | 月度 | 5880 | 20 | 2016-2025 |
| `regional_gdp.csv` | 区域GDP表（年度） | 年度 | 490 | 20 | 2016-2025 |
| `retail_sales.csv` | 零售销售表（月度） | 月度 | 5880 | 19 | 2016-2025 |
| `stock_indices.csv` | 股票指数表（月度） | 月度 | 120 | 19 | 2016-2025 |
| `trade_balance.csv` | 进出口贸易表（月度） | 月度 | 5880 | 20 | 2016-2025 |

- 总数据量：61280 行

## 三、字段注释（按表）

### `corporate_credit.csv`

- 中文名：企业信贷表（季度，行业细分）
- 说明：用于分析不同行业的信用利差、违约、发行和不良风险。
- 行数/字段数：17160 / 20

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `quarter` | 季度 | 1~4（仅季度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `sector` | 行业名称 | 示例：科技 / 金融 / 能源 / 消费 / 工业 / 医疗健康 / 公用事业 / 地产 / ... |
| `inv_grade_yield_pct` | 投资级融资收益率 | 单位：% |
| `high_yield_yield_pct` | 高收益融资收益率 | 单位：% |
| `ig_spread_bps` | 投资级信用利差 | 单位：bp |
| `hy_spread_bps` | 高收益信用利差 | 单位：bp |
| `default_rate_pct` | 违约率 | 单位：% |
| `new_issuance_100m_cny` | 新增发行规模 | 单位：亿元 |
| `downgrade_ratio_pct` | 评级下调比例 | 单位：% |
| `npl_ratio_pct` | 不良率 | 单位：% |
| `source_org` | 数据来源机构 | 示例：银行间市场交易商协会 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `cpi_inflation.csv`

- 中文名：CPI通胀表（月度）
- 说明：用于分析居民消费价格指数及分项价格变化。
- 行数/字段数：5880 / 19

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `overall_cpi_idx` | 综合CPI指数 | 基期指数 |
| `food_cpi_idx` | 食品CPI指数 | 基期指数 |
| `energy_cpi_idx` | 能源CPI指数 | 基期指数 |
| `core_cpi_idx` | 核心CPI指数 | 剔除波动项后的价格指数 |
| `services_cpi_idx` | 服务CPI指数 | 服务类价格指数 |
| `yoy_change_pct` | CPI同比 | 单位：% |
| `mom_change_pct` | CPI环比 | 单位：% |
| `base_period` | 指数基期 | 示例：2016-01=100 |
| `source_org` | 数据来源机构 | 示例：国家统计局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `employment_stats.csv`

- 中文名：就业市场表（月度）
- 说明：用于分析劳动力规模、失业率、工资与新增就业。
- 行数/字段数：5880 / 19

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `labor_force_10k` | 劳动力总人数 | 单位：万人 |
| `employed_10k` | 就业人数 | 单位：万人 |
| `unemployed_10k` | 失业人数 | 单位：万人 |
| `unemployment_rate_pct` | 失业率 | 单位：% |
| `youth_unemp_rate_pct` | 青年失业率 | 单位：% |
| `participation_rate_pct` | 劳动参与率 | 单位：% |
| `avg_monthly_wage_cny` | 平均月工资 | 单位：元 |
| `new_urban_jobs_10k` | 城镇新增就业 | 单位：万人 |
| `source_org` | 数据来源机构 | 示例：国家统计局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `energy_prices.csv`

- 中文名：能源价格表（月度）
- 说明：用于分析煤价、油气价格、电价、碳价与可再生占比。
- 行数/字段数：960 / 19

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `coal_price_cny_per_ton` | 动力煤价格 | 单位：元/吨 |
| `lng_price_usd_per_mmbtu` | LNG价格 | 单位：美元/MMBtu |
| `wti_usd_per_bbl` | WTI原油价格 | 单位：美元/桶 |
| `brent_usd_per_bbl` | Brent原油价格 | 单位：美元/桶 |
| `electricity_price_cny_per_kwh` | 电价 | 单位：元/千瓦时 |
| `carbon_price_cny_per_tco2` | 碳价 | 单位：元/吨CO2 |
| `renewable_generation_share_pct` | 可再生发电占比 | 单位：% |
| `power_generation_100m_kwh` | 发电量 | 单位：亿千瓦时 |
| `source_org` | 数据来源机构 | 示例：国家能源局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `exchange_rates.csv`

- 中文名：汇率与外储表（月度）
- 说明：用于分析人民币对主要货币汇率、美元指数与外储。
- 行数/字段数：120 / 19

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `usd_cny` | 美元兑人民币 | 汇率（1美元=多少人民币） |
| `eur_cny` | 欧元兑人民币 | 汇率 |
| `jpy_cny` | 日元兑人民币 | 汇率 |
| `gbp_cny` | 英镑兑人民币 | 汇率 |
| `cfets_index` | CFETS人民币汇率指数 | 指数 |
| `dxy_index` | 美元指数DXY | 指数 |
| `fx_reserve_100m_usd` | 外汇储备 | 单位：亿美元 |
| `fx_vol_1m_pct` | 1个月汇率波动率 | 单位：% |
| `source_org` | 数据来源机构 | 示例：国家外汇管理局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `gdp_expenditure.csv`

- 中文名：GDP构成表（支出法，季度）
- 说明：用于分析经济总量与三大需求（消费、投资、净出口）及贡献率。
- 行数/字段数：1960 / 24

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `quarter` | 季度 | 1~4（仅季度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `nominal_gdp_100m_cny` | 名义GDP | 单位：亿元 |
| `real_gdp_100m_cny` | 实际GDP | 单位：亿元（不变价） |
| `consumption_100m_cny` | 最终消费支出 | 单位：亿元 |
| `investment_100m_cny` | 资本形成总额 | 单位：亿元 |
| `government_spending_100m_cny` | 政府消费支出 | 单位：亿元 |
| `exports_100m_cny` | 出口总额 | 单位：亿元 |
| `imports_100m_cny` | 进口总额 | 单位：亿元 |
| `net_export_100m_cny` | 净出口 | 单位：亿元；=出口-进口 |
| `gdp_yoy_pct` | GDP同比增速 | 单位：% |
| `consumption_contrib_pct` | 消费对GDP增长贡献率 | 单位：% |
| `investment_contrib_pct` | 投资对GDP增长贡献率 | 单位：% |
| `net_export_contrib_pct` | 净出口对GDP增长贡献率 | 单位：%；与前两项近似合计100 |
| `seasonal_adj` | 季节调整标记 | 示例：NSA / SA |
| `source_org` | 数据来源机构 | 示例：国家统计局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `government_finance.csv`

- 中文名：政府财政表（年度）
- 说明：用于分析财政收支、赤字、债务和财政可持续性。
- 行数/字段数：390 / 20

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `tax_revenue_100m_cny` | 税收收入 | 单位：亿元 |
| `non_tax_revenue_100m_cny` | 非税收入 | 单位：亿元 |
| `total_revenue_100m_cny` | 财政总收入 | 单位：亿元 |
| `expenditure_100m_cny` | 财政总支出 | 单位：亿元 |
| `deficit_100m_cny` | 财政赤字 | 单位：亿元；=支出-收入 |
| `debt_balance_100m_cny` | 政府债务余额 | 单位：亿元 |
| `debt_to_gdp_pct` | 债务率 | 单位：%；=债务/GDP |
| `fiscal_balance_pct_gdp` | 财政平衡占GDP比 | 单位：%；=(收入-支出)/GDP |
| `transfer_payment_100m_cny` | 转移支付支出 | 单位：亿元 |
| `social_security_100m_cny` | 社会保障与就业支出 | 单位：亿元 |
| `source_org` | 数据来源机构 | 示例：财政部 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary |

### `industrial_output.csv`

- 中文名：工业产出表（月度）
- 说明：用于分析工业生产指数、结构分项与产能利用率。
- 行数/字段数：5880 / 20

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `total_index` | 工业总产出指数 | 指数 |
| `manufacturing_index` | 制造业指数 | 指数 |
| `mining_index` | 采矿业指数 | 指数 |
| `utilities_index` | 公用事业指数 | 指数 |
| `hightech_index` | 高技术制造业指数 | 指数 |
| `equipment_mfg_index` | 装备制造业指数 | 指数 |
| `capacity_utilization_pct` | 产能利用率 | 单位：% |
| `yoy_growth_pct` | 工业增加值同比 | 单位：% |
| `mom_growth_sa_pct` | 季调环比 | 单位：% |
| `source_org` | 数据来源机构 | 示例：国家统计局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `interest_rates.csv`

- 中文名：利率表（月度）
- 说明：用于分析政策利率、LPR、国债收益率与实际利率。
- 行数/字段数：120 / 19

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `policy_rate_pct` | 政策利率 | 单位：% |
| `lpr_1y_pct` | 1年期LPR | 单位：% |
| `lpr_5y_pct` | 5年期LPR | 单位：% |
| `repo_7d_pct` | 7天回购利率 | 单位：% |
| `bond_1y_pct` | 1年期国债收益率 | 单位：% |
| `bond_10y_pct` | 10年期国债收益率 | 单位：% |
| `yield_spread_10y_1y_bps` | 期限利差（10Y-1Y） | 单位：bp |
| `real_policy_rate_pct` | 实际政策利率 | 单位：%；=政策利率-CPI同比 |
| `source_org` | 数据来源机构 | 示例：中国人民银行 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `money_supply.csv`

- 中文名：货币供应量表（月度）
- 说明：用于分析M0/M1/M2、社融和贷款余额。
- 行数/字段数：4680 / 20

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `m0_100m_cny` | M0余额 | 单位：亿元 |
| `m1_100m_cny` | M1余额 | 单位：亿元 |
| `m2_100m_cny` | M2余额 | 单位：亿元 |
| `m0_yoy_pct` | M0同比 | 单位：% |
| `m1_yoy_pct` | M1同比 | 单位：% |
| `m2_yoy_pct` | M2同比 | 单位：% |
| `social_financing_flow_100m_cny` | 社会融资规模增量 | 单位：亿元 |
| `loan_balance_100m_cny` | 贷款余额 | 单位：亿元 |
| `reserve_ratio_pct` | 存款准备金率 | 单位：% |
| `source_org` | 数据来源机构 | 示例：中国人民银行 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `real_estate.csv`

- 中文名：房地产市场表（月度）
- 说明：用于分析房价、销售、开工、投资和库存压力。
- 行数/字段数：5880 / 20

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `home_price_idx` | 房价指数 | 指数 |
| `new_home_sales_10k_sqm` | 新房销售面积 | 单位：万平方米 |
| `existing_home_sales_10k_sqm` | 二手房销售面积 | 单位：万平方米 |
| `housing_starts_10k_sqm` | 房屋新开工面积 | 单位：万平方米 |
| `real_estate_investment_100m_cny` | 房地产开发投资 | 单位：亿元 |
| `mortgage_rate_pct` | 按揭贷款利率 | 单位：% |
| `vacancy_rate_pct` | 空置率 | 单位：% |
| `inventory_months` | 库存去化周期 | 单位：月 |
| `home_price_yoy_pct` | 房价同比 | 单位：% |
| `source_org` | 数据来源机构 | 示例：国家统计局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `regional_gdp.csv`

- 中文名：区域GDP表（年度）
- 说明：用于分析地区经济总量、人均、结构占比和产业增加值。
- 行数/字段数：490 / 20

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `gdp_100m_cny` | 地区生产总值 | 单位：亿元 |
| `gdp_growth_pct` | GDP增速 | 单位：% |
| `gdp_per_capita_cny` | 人均GDP | 单位：元 |
| `population_10k` | 人口规模 | 单位：万人 |
| `gdp_share_pct` | GDP占全国比重 | 单位：% |
| `primary_share_pct` | 第一产业占比 | 单位：% |
| `secondary_share_pct` | 第二产业占比 | 单位：% |
| `tertiary_share_pct` | 第三产业占比 | 单位：% |
| `industry_value_added_100m_cny` | 第二产业增加值 | 单位：亿元 |
| `services_value_added_100m_cny` | 第三产业增加值 | 单位：亿元 |
| `source_org` | 数据来源机构 | 示例：国家统计局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary |

### `retail_sales.csv`

- 中文名：零售销售表（月度）
- 说明：用于分析社零总量、分项消费和名义/实际增速。
- 行数/字段数：5880 / 19

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `total_sales_100m_cny` | 社会消费品零售总额 | 单位：亿元 |
| `auto_sales_100m_cny` | 汽车类零售额 | 单位：亿元 |
| `food_sales_100m_cny` | 食品类零售额 | 单位：亿元 |
| `ecommerce_sales_100m_cny` | 网上零售额 | 单位：亿元 |
| `service_retail_100m_cny` | 服务消费零售额 | 单位：亿元 |
| `yoy_change_pct` | 同比增速 | 单位：% |
| `mom_change_sa_pct` | 季调环比增速 | 单位：% |
| `real_yoy_change_pct` | 实际同比增速 | 单位：%；约等于名义同比-CPI同比 |
| `source_org` | 数据来源机构 | 示例：国家统计局 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `stock_indices.csv`

- 中文名：股票指数表（月度）
- 说明：用于分析A股核心指数、估值、波动和成交。
- 行数/字段数：120 / 19

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `shanghai_comp_close` | 上证综指月末收盘 | 指数点 |
| `shenzhen_comp_close` | 深证成指月末收盘 | 指数点 |
| `csi300_close` | 沪深300月末收盘 | 指数点 |
| `chinext_close` | 创业板指月末收盘 | 指数点 |
| `volatility_idx` | 市场波动指数 | 指数 |
| `pe_ttm_csi300` | 沪深300滚动市盈率 | 倍数 |
| `market_turnover_100m_cny` | 市场成交额 | 单位：亿元 |
| `northbound_netflow_100m_cny` | 北向资金净流入 | 单位：亿元，可为负 |
| `source_org` | 数据来源机构 | 示例：沪深交易所 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

### `trade_balance.csv`

- 中文名：进出口贸易表（月度）
- 说明：用于分析进出口规模、贸易差额与结构分布。
- 行数/字段数：5880 / 20

| 字段名 | 中文说明 | 取值/单位/备注 |
| :--- | :--- | :--- |
| `ref_date` | 统计期日期 | 格式：YYYY-MM-DD；月度为当月1日，季度为季初，年度为1月1日 |
| `year` | 年份 | 例如：2016~2025 |
| `month` | 月份 | 1~12（仅月度表） |
| `geo_level` | 地区层级 | 枚举：national/region/province/city |
| `geo_code` | 地区编码 | 如：CN、CN-R-EC、CN-GD |
| `geo_name` | 地区名称 | 如：中国、华东、广东 |
| `parent_geo_code` | 上级地区编码 | 如省份上级为大区编码 |
| `parent_geo_name` | 上级地区名称 | 如中国、华东等 |
| `exports_musd` | 出口总额 | 单位：百万美元 |
| `imports_musd` | 进口总额 | 单位：百万美元 |
| `trade_balance_musd` | 贸易差额 | 单位：百万美元；=出口-进口 |
| `export_yoy_pct` | 出口同比 | 单位：% |
| `import_yoy_pct` | 进口同比 | 单位：% |
| `top_export_category` | 主要出口品类 | 示例：农产品 / 化工产品 / 服务贸易 / 机电产品 / 汽车 / 纺织服装 / 钢材 / 高新技术产品 |
| `top_import_category` | 主要进口品类 | 示例：农产品 / 化工品 / 原油 / 天然气 / 服务贸易 / 机械设备 / 铁矿砂 / 集成电路 |
| `top_export_partner` | 主要出口目的地 | 示例：澳大利亚 / 日本 / 东盟 / 欧盟 / 美国 / 俄罗斯 / 巴西 / 韩国 / ... |
| `top_import_partner` | 主要进口来源地 | 示例：美国 / 韩国 / 中国香港 / 印度 / 俄罗斯 / 欧盟 / 东盟 / 巴西 / ... |
| `source_org` | 数据来源机构 | 示例：海关总署 |
| `release_date` | 发布日期 | 格式：YYYY-MM-DD |
| `data_status` | 数据状态 | 枚举：final / preliminary / revised |

## 四、常用 SQL 关联键建议

- 地区关联：`geo_code`
- 月度关联：`year + month + geo_code`
- 季度关联：`year + quarter + geo_code`
- 年度关联：`year + geo_code`
