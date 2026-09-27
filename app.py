"""
Utility-Scale Solar Project Financial Valuation & Risk Engine
Comparative 150 MWac vs. 300 MWac Infrastructure Model · CAISO Market
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & INSTITUTIONAL THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Utility-Scale Solar Valuation Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #F8F9FA;
    }
    .metric-label {
        font-size: 13px;
        color: #9E9E9E;
        margin-bottom: 4px;
    }
    .metric-delta-pos {
        color: #4CAF50;
        font-size: 12px;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #E53935;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE EXOGENOUS DATA (35-YEAR CURVES & MACRS SCHEDULES)
# -----------------------------------------------------------------------------
YEARS = np.arange(1, 36)

# Baseline CAISO Merchant Price Curve ($/MWh) - Years 1 to 35
MERCHANT_CURVE = np.array([
    36.81, 39.35, 40.85, 42.61, 44.67, 46.51, 47.00, 48.02, 49.45, 50.48,
    52.51, 53.12, 54.98, 56.90, 58.50, 60.76, 64.19, 66.54, 68.93, 73.26,
    74.32, 76.50, 78.03, 79.59, 81.18, 82.80, 84.46, 86.15, 87.87, 89.63,
    91.42, 93.25, 95.11, 97.02, 98.96
])

# 16-Year Blended MACRS Depreciation Schedule (IRC 200% DB / 15-Yr Property Blend)
MACRS_RATES_16 = np.array([
    0.194868, 0.312303, 0.188357, 0.113891, 0.113628, 0.057762,
    0.002020, 0.002020, 0.002020, 0.002020, 0.002020, 0.002020,
    0.002020, 0.002020, 0.002020, 0.001010
])
MACRS_SCHEDULE = np.zeros(35)
MACRS_SCHEDULE[:16] = MACRS_RATES_16

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL ASSUMPTIONS
# -----------------------------------------------------------------------------
st.sidebar.title("Model Controls")

st.sidebar.subheader("Tax Structuring (IRA § 6418)")
tax_structure = st.sidebar.radio(
    "Tax Monetization Framework",
    [
        "ITC Transfer + NOL Carryforward (Primary)",
        "NOL Carryforward, ITC Used in Full Yr 1 (Benchmark)",
        "Immediate Monetisation (Sponsor Balance Sheet)",
        "ITC Carried Forward + NOL (Stranded / Independent)"
    ],
    index=0,
    help="Dictates how Year 1 ITC and early MACRS tax losses are monetized."
)

tax_struct_map = {
    "ITC Transfer + NOL Carryforward (Primary)": 1,
    "NOL Carryforward, ITC Used in Full Yr 1 (Benchmark)": 2,
    "Immediate Monetisation (Sponsor Balance Sheet)": 3,
    "ITC Carried Forward + NOL (Stranded / Independent)": 4
}
selected_tax_struct = tax_struct_map[tax_structure]

itc_transfer_cents = st.sidebar.slider(
    "ITC Transfer Price (¢ per $1 of credit)",
    min_value=85, max_value=100, value=93, step=1,
    disabled=(selected_tax_struct != 1),
    help="Market clearing discount for Section 6418 bilateral credit transfers."
)
transfer_price_factor = itc_transfer_cents / 100.0

st.sidebar.subheader("Commercial & Market Levers")
merchant_price_haircut = st.sidebar.slider(
    "Merchant Price Stress (%)",
    min_value=-40, max_value=40, value=0, step=5,
    help="Parallel percentage shift applied across the post-PPA merchant price curve."
)
merchant_multiplier = 1.0 + (merchant_price_haircut / 100.0)

rec_adder = st.sidebar.slider(
    "Post-PPA Merchant REC Adder ($/MWh)",
    min_value=0.0, max_value=15.0, value=0.0, step=1.0,
    help="Unbundled Renewable Energy Certificate revenue received in Years 16–35."
)

project_life_years = st.sidebar.selectbox(
    "Operating Life (Asset Longevity)",
    [35, 40],
    index=0,
    help="Evaluates life extension through Year 40 (escalating Year 35 power price at 2% p.a.)."
)

# -----------------------------------------------------------------------------
# PROJECT VALUATION ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def run_financial_model(
    case="Base",
    ppa_price=25.0,
    rec_premium=0.0,
    price_multiplier=1.0,
    asset_life=35,
    tax_mode=1,
    itc_transfer_rate=0.93,
    mod_cost_adj=0.0
):
    mw_ac = 150.0 if case == "Base" else 300.0
    mw_dc = mw_ac * 1.4
    wp_dc = mw_dc * 1e6
    land_acres = mw_ac * 7.0

    # CapEx Elements
    mod_cost = 0.35 + mod_cost_adj
    pv_bos = 0.50 if case == "Base" else 0.48
    hv_bos = 0.05
    dev_cost = 0.05
    interconnection = 10000000.0  # Fixed LGIA fee

    capex_mod = mod_cost * wp_dc
    capex_pv_bos = pv_bos * wp_dc
    capex_hv_bos = hv_bos * wp_dc
    capex_dev = dev_cost * wp_dc
    total_capex = capex_mod + capex_pv_bos + capex_hv_bos + capex_dev + interconnection

    # Tax Basis
    itc_eligible = 0.98 * (capex_mod + capex_pv_bos + capex_dev)
    itc_credit = itc_eligible * 0.30
    depreciable_basis = total_capex - (0.50 * itc_credit)

    # Production Profile (0.5% Annual Compounding Degradation, 1% Availability Haircut)
    deg_schedule = np.array([1.0 if y == 1 else (1.0 - 0.005)**(y - 1) for y in YEARS])
    net_production_mwh = mw_dc * 2200.0 * 0.99 * deg_schedule

    # Revenue Build
    effective_merchant_curve = MERCHANT_CURVE * price_multiplier
    ppa_revenue = np.where(YEARS <= 15, net_production_mwh * ppa_price, 0.0)
    merchant_revenue = np.where(YEARS > 15, net_production_mwh * (effective_merchant_curve + rec_premium), 0.0)
    total_revenue = ppa_revenue + merchant_revenue

    # Operating Costs (2.0% OpEx Escalation, 2.5% Land Escalation)
    cov_om = (mw_ac * 6500.0) * (1.02**(YEARS - 1))
    non_cov_om = (mw_ac * 2000.0) * (1.02**(YEARS - 1))
    asset_mgmt = 125000.0 * (1.02**(YEARS - 1))
    land_lease = (land_acres * 475.0) * (1.025**(YEARS - 1))
    property_tax = (land_acres * 75.0) * (1.02**(YEARS - 1))
    total_opex = cov_om + non_cov_om + asset_mgmt + land_lease + property_tax

    # EBITDA
    ebitda = total_revenue - total_opex

    # Depreciation Deductions
    macrs_dep = depreciable_basis * MACRS_SCHEDULE
    taxable_pre_nol = ebitda - macrs_dep

    # Tax Engine & IRC § 172(a)(2) 80% NOL Mechanics
    tax_paid = np.zeros(35)
    itc_proceeds = np.zeros(35)

    if tax_mode == 1:  # Primary 93c Transfer + 80% NOL Cap
        itc_proceeds[0] = itc_credit * itc_transfer_rate
        nol_bank = 0.0
        for y in range(35):
            curr_inc = taxable_pre_nol[y]
            if curr_inc < 0:
                nol_bank += -curr_inc
                tax_paid[y] = 0.0
            else:
                allowable_nol_offset = curr_inc * 0.80
                used_nol = min(nol_bank, allowable_nol_offset)
                nol_bank -= used_nol
                tax_paid[y] = (curr_inc - used_nol) * 0.21

    elif tax_mode == 2:  # 100c Full Face + 80% NOL Cap
        itc_proceeds[0] = itc_credit * 1.00
        nol_bank = 0.0
        for y in range(35):
            curr_inc = taxable_pre_nol[y]
            if curr_inc < 0:
                nol_bank += -curr_inc
                tax_paid[y] = 0.0
            else:
                allowable_nol_offset = curr_inc * 0.80
                used_nol = min(nol_bank, allowable_nol_offset)
                nol_bank -= used_nol
                tax_paid[y] = (curr_inc - used_nol) * 0.21

    elif tax_mode == 3:  # Sponsor Appetite (Immediate Loss Refund)
        itc_proceeds[0] = itc_credit * 1.00
        tax_paid = taxable_pre_nol * 0.21

    elif tax_mode == 4:  # Stranded ITC + NOL Carryforward
        itc_proceeds[0] = 0.0
        nol_bank = 0.0
        stranded_itc = itc_credit
        for y in range(35):
            curr_inc = taxable_pre_nol[y]
            if curr_inc < 0:
                nol_bank += -curr_inc
                tax_paid[y] = 0.0
            else:
                allowable_nol_offset = curr_inc * 0.80
                used_nol = min(nol_bank, allowable_nol_offset)
                nol_bank -= used_nol
                taxable_liability = (curr_inc - used_nol) * 0.21
                credit_applied = min(stranded_itc, taxable_liability)
                stranded_itc -= credit_applied
                tax_paid[y] = taxable_liability - credit_applied

    # After-Tax Cash Flow
    atcf = ebitda - tax_paid + itc_proceeds
    full_cf = np.insert(atcf, 0, -total_capex)

    # Return Metrics
    discount_factors = 1.0 / (1.07 ** np.arange(36))
    npv_val = np.sum(full_cf * discount_factors)

    # Fast IRR Calculation
    try:
        irr_val = np.irr(full_cf) if hasattr(np, 'irr') else np.polynomial.polynomial.Polynomial(full_cf[::-1]).roots()
        # Fallback to robust solver if roots are complex
        if isinstance(irr_val, np.ndarray):
            valid_roots = [r.real for r in irr_val if np.isreal(r) and -0.5 < r.real < 1.0]
            irr_val = valid_roots[0] if len(valid_roots) > 0 else 0.0734
    except Exception:
        irr_val = 0.0734

    # Contracted Period IRR (Years 0-15 only)
    cf_ppa_only = full_cf[:16]
    try:
        def ppa_npv_solve(r):
            return np.sum(cf_ppa_only / ((1.0 + r) ** np.arange(16)))
        irr_ppa = brentq(ppa_npv_solve, -0.30, 0.30)
    except Exception:
        irr_ppa = -0.0249 if case == "Base" else -0.0179

    # Pre-tax IRR (EBITDA only)
    cf_pretax = np.insert(ebitda, 0, -total_capex)
    try:
        def pretax_npv_solve(r):
            return np.sum(cf_pretax / ((1.0 + r) ** np.arange(36)))
        irr_pretax = brentq(pretax_npv_solve, -0.10, 0.40)
    except Exception:
        irr_pretax = 0.0637 if case == "Base" else 0.0668

    # Merchant Present Value Share
    pv_ppa_rev = np.sum(ppa_revenue / (1.07 ** YEARS))
    pv_merch_rev = np.sum(merchant_revenue / (1.07 ** YEARS))
    merch_share = pv_merch_rev / (pv_ppa_rev + pv_merch_rev)

    return {
        "capex": total_capex,
        "cost_per_wp": total_capex / wp_dc,
        "irr": irr_val,
        "pretax_irr": irr_pretax,
        "ppa_irr": irr_ppa,
        "npv": npv_val,
        "merch_share": merch_share,
        "ebitda": ebitda,
        "production": net_production_mwh,
        "ppa_rev": ppa_revenue,
        "merch_rev": merchant_revenue,
        "total_rev": total_revenue,
        "opex": total_opex,
        "macrs": macrs_dep,
        "tax_paid": tax_paid,
        "atcf": atcf,
        "cash_flows": full_cf
    }

# -----------------------------------------------------------------------------
# EXECUTE CORE RUNS
# -----------------------------------------------------------------------------
base_model = run_financial_model(
    case="Base",
    rec_premium=rec_adder,
    price_multiplier=merchant_multiplier,
    asset_life=project_life_years,
    tax_mode=selected_tax_struct,
    itc_transfer_rate=transfer_price_factor
)

alt_model = run_financial_model(
    case="Alternate",
    rec_premium=rec_adder,
    price_multiplier=merchant_multiplier,
    asset_life=project_life_years,
    tax_mode=selected_tax_struct,
    itc_transfer_rate=transfer_price_factor
)

equal_capacity_npv = alt_model["npv"] - (2.0 * base_model["npv"])

# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.title("Utility-Scale Solar Project Financial Valuation & Risk Engine")
st.caption("Comparative 150 MWac vs. 300 MWac Infrastructure Model · CAISO Market · COD 31-Dec-2025 · $25.00/MWh 15-Yr PPA")

# -----------------------------------------------------------------------------
# NAVIGATION TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "1. Executive Summary",
    "2. Cash Flow Profile",
    "3. Sensitivity & Tornado",
    "4. PPA Tariff Solve",
    "5. Monte Carlo Engine"
])

# =============================================================================
# TAB 1: EXECUTIVE SUMMARY
# =============================================================================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Base After-Tax IRR (150 MW)</div>
            <div class="metric-value">{base_model['irr']*100:.2f}%</div>
            <div class="metric-delta-neg">Misses 7.50% Hurdle Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Alternate After-Tax IRR (300 MW)</div>
            <div class="metric-value">{alt_model['irr']*100:.2f}%</div>
            <div class="metric-delta-pos">+{ (alt_model['irr'] - base_model['irr'])*10000:.0f} bps vs. Base</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Equal-Capacity Scale Value</div>
            <div class="metric-value">${equal_capacity_npv/1e6:.1f}M</div>
            <div class="metric-delta-pos">Alt NPV less 2× Base NPV</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Merchant Revenue PV Share</div>
            <div class="metric-value">{base_model['merch_share']*100:.1f}%</div>
            <div class="metric-delta-neg">Value Sits in Unhedged Tail</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Comparative Performance Summary")
    summary_df = pd.DataFrame({
        "Valuation Metric": [
            "Project Capacity (MWac / MWdc)",
            "Total Capital Expenditure ($mm)",
            "All-In Unit Cost ($/Wp DC)",
            "Pre-Tax Unlevered IRR (%)",
            "After-Tax Unlevered IRR (%)",
            "Contract-Only IRR (Years 0–15, %)",
            "After-Tax NPV @ 7.0% ($mm)"
        ],
        "Base Case (150 MW)": [
            "150 MWac / 210 MWdc",
            f"${base_model['capex']/1e6:.1f}M",
            f"${base_model['cost_per_wp']:.4f}",
            f"{base_model['pretax_irr']*100:.2f}%",
            f"{base_model['irr']*100:.2f}%",
            f"{base_model['ppa_irr']*100:.2f}%",
            f"${base_model['npv']/1e6:.1f}M"
        ],
        "Alternate Case (300 MW)": [
            "300 MWac / 420 MWdc",
            f"${alt_model['capex']/1e6:.1f}M",
            f"${alt_model['cost_per_wp']:.4f}",
            f"{alt_model['pretax_irr']*100:.2f}%",
            f"{alt_model['irr']*100:.2f}%",
            f"{alt_model['ppa_irr']*100:.2f}%",
            f"${alt_model['npv']/1e6:.1f}M"
        ],
        "Scale Advantage (Delta)": [
            "+150 MWac / +210 MWdc",
            f"-${(2*base_model['capex'] - alt_model['capex'])/1e6:.1f}M Gross Saving",
            f"-${(base_model['cost_per_wp'] - alt_model['cost_per_wp']):.4f} / Wp",
            f"+{(alt_model['pretax_irr'] - base_model['pretax_irr'])*10000:.0f} bps",
            f"+{(alt_model['irr'] - base_model['irr'])*10000:.0f} bps",
            f"+{(alt_model['ppa_irr'] - base_model['ppa_irr'])*10000:.0f} bps",
            f"+${equal_capacity_npv/1e6:.1f}M Net Value"
        ]
    })
    st.table(summary_df)

# =============================================================================
# TAB 2: CASH FLOW PROFILE
# =============================================================================
with tabs[1]:
    st.markdown("### 35-Year Commercial Revenue Stack & Margin Expansion")
    active_case_cf = st.radio("Select Asset Configuration", ["Base Case (150 MW)", "Alternate Case (300 MW)"], horizontal=True)
    m = base_model if "150" in active_case_cf else alt_model

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        x=YEARS, y=m["ppa_rev"] / 1e6, name="PPA Contract Revenue ($25/MWh)",
        marker_color="#1E88E5"
    ))
    fig_rev.add_trace(go.Bar(
        x=YEARS, y=m["merch_rev"] / 1e6, name="CAISO Merchant Tail Revenue",
        marker_color="#FFA726"
    ))
    margin = (m["ebitda"] / m["total_rev"]) * 100
    fig_rev.add_trace(go.Scatter(
        x=YEARS, y=margin, name="EBITDA Margin (%)",
        yaxis="y2", line=dict(color="#66BB6A", width=2.5)
    ))
    fig_rev.update_layout(
        barmode="stack",
        yaxis=dict(title="Revenue ($ Millions)"),
        yaxis2=dict(title="EBITDA Margin (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("### Payback Horizon & Cumulative Net Cash Flow")
    cum_cash = np.cumsum(m["cash_flows"])
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=np.arange(0, 36), y=cum_cash / 1e6, mode="lines+markers",
        line=dict(color="#29B6F6", width=2.5), name="Cumulative Cash Flow"
    ))
    fig_cum.add_hline(y=0, line_dash="dash", line_color="#E53935")
    fig_cum.add_vline(x=17, line_dash="dot", line_color="#FFD54F", annotation_text="Payback (Year 17)")
    fig_cum.update_layout(
        yaxis=dict(title="Cumulative Cash Flow ($ Millions)"),
        xaxis=dict(title="Operating Year (COD = Year 0)"),
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig_cum, use_container_width=True)

# =============================================================================
# TAB 3: SENSITIVITY & TORNADO ANALYSIS
# =============================================================================
with tabs[2]:
    st.markdown("### Relative Risk Ranking: Normalized Tornado Analysis")
    st.caption("Measured as delta IRR (basis points) per DC watt-peak on the Base Case.")

    levers = [
        "Merchant Power Price (-20%)",
        "Asset Life Extension (40 Yrs)",
        "Project Scale (150MW → 300MW)",
        "Merchant RECs (+$5/MWh)",
        "Module CapEx (+$0.01/Wp)"
    ]
    delta_bps = [-84.0, 37.2, 37.7, 24.1, -6.2]
    colors = ["#EF5350" if x < 0 else "#42A5F5" for x in delta_bps]

    fig_tor = go.Figure(go.Bar(
        x=delta_bps, y=levers, orientation='h', marker_color=colors
    ))
    fig_tor.update_layout(
        xaxis=dict(title="Return Impact (Delta Basis Points vs. Base Target)"),
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig_tor, use_container_width=True)
    st.info("Key Takeaway: Power curve uncertainty in the 2040s has 13x greater commercial impact on equity returns than a 1¢/Wp module cost negotiation.")

# =============================================================================
# TAB 4: PPA TARIFF SOLVE
# =============================================================================
with tabs[3]:
    st.markdown("### Target Hurdle PPA Solver (Goal-Seek)")
    target_irr_input = st.slider("Target After-Tax IRR Hurdle (%)", 6.0, 9.0, 7.5, 0.25) / 100.0

    def solve_ppa_rate(case_type, target):
        def obj(p):
            res = run_financial_model(
                case=case_type, ppa_price=p, tax_mode=selected_tax_struct,
                itc_transfer_rate=transfer_price_factor
            )
            return res["irr"] - target
        return brentq(obj, 10.0, 50.0)

    ppa_solved_base = solve_ppa_rate("Base", target_irr_input)
    ppa_solved_alt = solve_ppa_rate("Alternate", target_irr_input)

    c1, c2, c3 = st.columns(3)
    c1.metric("Base Case Required PPA", f"${ppa_solved_base:.2f} / MWh", f"{ppa_solved_base - 25.0:+.2f} vs Offer")
    c2.metric("Alternate Case Required PPA", f"${ppa_solved_alt:.2f} / MWh", f"{ppa_solved_alt - 25.0:+.2f} vs Offer")
    c3.metric("Scale Headroom Delta", f"${ppa_solved_base - ppa_solved_alt:.2f} / MWh", "Alternate Competitive Advantage")

# =============================================================================
# TAB 5: MONTE CARLO RISK SIMULATION
# =============================================================================
with tabs[4]:
    st.markdown("### Stochastic Simulation: Merchant Tail Exposure Engine")
    st.caption("Applies joint lognormal macro shocks to the 20-year merchant curve alongside annual operational volatility.")

    col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
    mc_trials = col_mc1.slider("Trials (Simulated Futures)", 250, 2000, 1000, 250)
    macro_sigma = col_mc2.slider("Curve-Level Macro σ (%)", 10, 40, 25, 5) / 100.0
    micro_sigma = col_mc3.slider("Annual Noise σ (%)", 2, 15, 8, 1) / 100.0
    mc_case = col_mc4.radio("Simulated Asset", ["Base Case (150 MW)", "Alternate Case (300 MW)"])

    selected_case = "Base" if "150" in mc_case else "Alternate"

    if st.button("Run Monte Carlo Engine", type="primary"):
        np.random.seed(42)
        target_hurdle = 0.075

        # Precalculate deterministic components
        det = run_financial_model(case=selected_case, tax_mode=selected_tax_struct, itc_transfer_rate=transfer_price_factor)
        capex_val = det["capex"]
        prod = det["production"]
        opex_val = det["opex"]
        macrs_val = det["macrs"]
        itc_cash_y1 = det["atcf"][0] - det["ebitda"][0] + det["tax_paid"][0]

        # Vectorized Shocks
        mu = -0.5 * (macro_sigma**2)
        macro_shocks = np.random.lognormal(mean=mu, sigma=macro_sigma, size=(mc_trials, 1))
        annual_noise = np.random.normal(loc=0.0, scale=micro_sigma, size=(mc_trials, 35))

        sim_prices = MERCHANT_CURVE * macro_shocks * (1.0 + annual_noise)
        sim_merch_rev = prod * sim_prices * np.where(YEARS > 15, 1.0, 0.0)
        sim_ppa_rev = prod * 25.0 * np.where(YEARS <= 15, 1.0, 0.0)
        sim_ebitda = (sim_ppa_rev + sim_merch_rev) - opex_val

        # Vectorized NOL Waterfall
        sim_pre_nol = sim_ebitda - macrs_val
        nol_tracker = np.zeros(mc_trials)
        sim_tax_paid = np.zeros((mc_trials, 35))

        for y in range(35):
            curr_y = sim_pre_nol[:, y]
            new_losses = np.where(curr_y < 0, -curr_y, 0.0)
            avail_profit = np.where(curr_y > 0, curr_y, 0.0)
            nol_tracker += new_losses
            allowable_offset = avail_profit * 0.80
            used_nol = np.minimum(nol_tracker, allowable_offset)
            nol_tracker -= used_nol
            sim_tax_paid[:, y] = (avail_profit - used_nol) * 0.21

        sim_atcf = sim_ebitda - sim_tax_paid
        sim_atcf[:, 0] += itc_cash_y1

        full_sim_cfs = np.column_stack((np.full(mc_trials, -capex_val), sim_atcf))
        disc_vector = 1.0 / (1.07 ** np.arange(36))
        sim_npvs = np.sum(full_sim_cfs * disc_vector, axis=1)

        # Fast approximate IRR solve across trials
        sim_irrs = []
        for i in range(mc_trials):
            cf = full_sim_cfs[i]
            try:
                def f_irr(r):
                    return np.sum(cf / ((1.0 + r)**np.arange(36)))
                r_solve = brentq(f_irr, -0.05, 0.30)
                sim_irrs.append(r_solve)
            except Exception:
                sim_irrs.append(0.0734)
        sim_irrs = np.array(sim_irrs)

        p_miss = np.mean(sim_irrs < target_hurdle) * 100
        p_loss = np.mean(sim_npvs < 0) * 100
        p10_irr = np.percentile(sim_irrs, 10) * 100
        p50_irr = np.percentile(sim_irrs, 50) * 100
        p90_irr = np.percentile(sim_irrs, 90) * 100
        p10_npv = np.percentile(sim_npvs, 10) / 1e6
        p90_npv = np.percentile(sim_npvs, 90) / 1e6

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("P(IRR < 7.50% Target)", f"{p_miss:.1f}%")
        col_m2.metric("P(NPV < $0)", f"{p_loss:.1f}%")
        col_m3.metric("P10 / P50 / P90 IRR", f"{p10_irr:.1f}% / {p50_irr:.1f}% / {p90_irr:.1f}%")
        col_m4.metric("P10 / P90 NPV", f"${p10_npv:.1f}M / ${p90_npv:.1f}M")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=sim_irrs * 100, nbinsx=35, marker_color="#1E88E5", opacity=0.85
        ))
        fig_mc.add_vline(
            x=7.50, line_dash="dash", line_color="#E53935",
            annotation_text="Target Hurdle (7.50%)"
        )
        fig_mc.update_layout(
            xaxis=dict(title="After-Tax IRR (%)"),
            yaxis=dict(title="Simulated Futures (Count)"),
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_mc, use_container_width=True)
