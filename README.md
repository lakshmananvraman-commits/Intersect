# Utility-Scale Solar Project Financial Valuation & Risk Engine
### Comparative 150 MWac vs. 300 MWac Infrastructure Model · IRA § 6418 Tax Transferability · Stochastic Risk Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Executive Overview

This project provides an institutional-grade 35-year unlevered project finance model and dynamic risk dashboard evaluating the commercial viability of a utility-scale solar facility located in California (CAISO). 

The analysis compares a **150 MWac / 210 MWdc Base Case** against a **300 MWac / 420 MWdc Alternate Case**, stress-testing whether project scale can overcome an aggressive, buyer-favorable 15-year Power Purchase Agreement (PPA) priced at **$25.00/MWh**. 

The valuation model incorporates post-Inflation Reduction Act (IRA) tax mechanics, accelerated MACRS depreciation schedules, project-level Net Operating Loss (NOL) carryforwards under federal limits, and an interactive Monte Carlo simulation engine modeling 20 years of unhedged merchant power market exposure.

---

## Key Valuation Findings

| Metric | Base Case (150 MWac) | Alternate Case (300 MWac) | Variance / Scale Delta |
| :--- | :--- | :--- | :--- |
| **Total Capital Expenditure (CapEx)** | $209.5M | $400.6M | -$18.4M gross CapEx savings |
| **All-in Installed Unit Cost** | $0.9976 / Wp | $0.9538 / Wp | -$0.0438 / Wp (-4.4%) |
| **Unlevered Pre-Tax IRR** | 6.37% | 6.68% | +32 bps |
| **Unlevered After-Tax IRR (Primary)** | **7.34%** *(Misses Target)* | **7.72%** *(Clears Target)* | **+38 bps** |
| **PPA-Term Only IRR (Years 0–15)** | **-2.49%** | **-1.79%** | +70 bps (PPA alone does not repay build) |
| **After-Tax Net Present Value (@ 7%)** | $8.3M | $33.6M | +$25.3M absolute NPV expansion |
| **Equal-Capacity Scale Value** | — | — | **+$17.0M** (Alt NPV less 2× Base NPV) |
| **Solved 15-Yr PPA for 7.50% Target** | **$26.02 / MWh** | **$23.62 / MWh** | -$2.40 / MWh headroom |
| **Competitive Bidding Floor** | — | **$22.62 / MWh** | -$2.38 / MWh discount matching Base return |

---

## Core Methodological Highlights

### 1. Scale Economics & Capital Efficiency ($17.0M Net NPV Expansion)
Doubling project capacity to 300 MWac captures **$18.4M in direct CapEx savings** via two primary levers:
*   **Fixed LGIA Interconnection Dilution ($10.0M):** The $10.0M interconnection cost under the Large Generator Interconnection Agreement is fixed regardless of capacity, halving unit interconnection costs from $0.0476/Wp to $0.0238/Wp.
*   **Balance of System (BOS) Volume Procurement ($8.4M):** A $0.02/Wp scale discount applied across 420 MWdc of installed hardware.
*   **Tax Drag & ITC Giveback Adjustment (-$3.5M):** Lower eligible CapEx reduces upfront ITC proceeds (-$2.15M PV at 93¢ transfer pricing) and shrinks depreciable basis, increasing lifetime taxable earnings (-$1.31M PV). Net equal-capacity value creation settles at an audited **$17.0M**.

### 2. Post-IRA Tax Structuring & IRC § 172(a)(2) Limitations
The asset generates **$55.6M (Base)** and **$108.7M (Alternate)** in 30% Investment Tax Credits alongside 16-year blended MACRS accelerated depreciation. 
Because standalone clean energy assets generate massive early tax deductions that drive taxable income to zero, four distinct tax monetization structures were modeled:
1.  **Primary Structure (IRA § 6418 Transferability):** Credits are sold to corporate buyers at **93¢ per dollar** in Year 1. Early depreciation deductions accumulate as Net Operating Losses (NOLs). In accordance with **IRC § 172(a)(2)**, post-2017 NOL carryforwards are capped at offsetting **80% of positive taxable income**, resulting in a realistic 4.2% residual corporate tax drag starting in Year 18.
2.  **100¢ Face Value Benchmark:** Assumes full face-value credit monetization without transfer broker discounts.
3.  **Corporate Sponsor Appetite:** Assumes an integrated corporate parent immediately monetizes tax losses against external operating income.
4.  **Stranded Credit (Stand-Alone Carryforward):** Credits sit unmonetized in a project account until taxable income emerges in Year 18, decaying lifetime IRR to **6.10%**.

### 3. Offtake Structuring & The 54% Merchant Tail
*   At $25.00/MWh, the 15-year PPA functions strictly as a downside debt-financing floor rather than an equity driver (yielding negative contract-term returns).
*   **Payback occurs in Year 17** (two years after contract expiration). 
*   **54% of the asset's discounted present value** sits in the uncontracted Years 16–35 merchant tail, making long-term merchant pricing the single most sensitive underwriting variable.

### 4. Stochastic Risk Simulation (Monte Carlo Engine)
A vectorized 1,000-trial Monte Carlo simulation tests merchant tail uncertainty using lognormal macro price curves ($\sigma = 25\%$) paired with compounding annual weather and operational noise ($\sigma = 8\%$). The simulation demonstrates that project scale reduces target underperformance probability from **60% (Base)** down to **43% (Alternate)**.

---

## Repository Structure
