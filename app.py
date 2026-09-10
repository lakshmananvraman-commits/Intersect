"""
Intersect Solar Project — Interactive Stress-Test Dashboard
============================================================
Streamlit app. Every assumption from the assessment brief is a live control.
Deploy free on Streamlit Community Cloud: push this repo to GitHub, then
share.streamlit.io -> New app -> pick the repo -> main file = app.py

Author: Lakshmanan Vaidhyaraman
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Supplied data (from the assessment workbook)
# ----------------------------------------------------------------------------
MERCHANT_DEFAULT = [36.81, 39.35, 40.85, 42.61, 44.67, 46.51, 47.00, 48.02, 49.45, 50.48,
                    52.51, 53.12, 54.98, 56.90, 58.50, 60.76, 64.19, 66.54, 68.93, 73.26,
                    74.32, 76.50, 78.03, 79.59, 81.18, 82.80, 84.46, 86.15, 87.87, 89.63,
                    91.42, 93.25, 95.11, 97.02, 98.96]
DEPR_DEFAULT = [0.19486836652891062, 0.31230254979336591, 0.18835654023552645,
                0.11389143346487204, 0.11362818066780520, 0.05776180040334328] + \
               [0.00202011883222884] * 9 + [0.00101005941611442]

# ----------------------------------------------------------------------------
# Core engine (identical logic to the Excel model, both tax conventions)
# ----------------------------------------------------------------------------
def irr(cf, lo=-0.99, hi=2.0):
    def f(r): return sum(c / (1 + r) ** i for i, c in enumerate(cf))
    if f(lo) * f(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return (lo + hi) / 2


def npv(rate, cf):
    return sum(c / (1 + rate) ** i for i, c in enumerate(cf))


def run_model(p, mwac, pv_bos, ppa=None, life=None, rec=None, mod=None, mh=None,
              price_mode=None, tax_mode=None, merchant=None, depr=None, transfer=None):
    """p = dict of shared assumptions; keyword args override for scenarios."""
    ppa = p['ppa'] if ppa is None else ppa
    life = p['life'] if life is None else life
    rec = p['rec'] if rec is None else rec
    mod = p['module'] if mod is None else mod
    mh = p['merch_haircut'] if mh is None else mh
    price_mode = p['price_mode'] if price_mode is None else price_mode
    tax_mode = p['tax_mode'] if tax_mode is None else tax_mode
    transfer = p['transfer'] if transfer is None else transfer
    merchant = np.array(p['merchant'] if merchant is None else merchant, dtype=float)
    depr = np.array(p['depr'] if depr is None else depr, dtype=float)

    mwdc = mwac * p['dcac']; wp = mwdc * 1e6; acres = mwac * p['acres_per_mw']
    cx = dict(modules=mod * wp, pv_bos=pv_bos * wp, hv_bos=p['hv_bos'] * wp,
              interconnect=p['interconnect'], development=p['dev'] * wp)
    capex = sum(cx.values())
    itc_basis = p['itc_elig'] * (cx['modules'] + cx['pv_bos'] + cx['development'])
    itc = p['itc_rate'] * itc_basis
    dbasis = capex - 0.5 * itc

    n = int(life); yr = np.arange(1, n + 1)
    prod = mwdc * p['yld'] * (1 - p['avail']) * (1 - p['degr']) ** (yr - 1)

    nm = len(merchant)
    price = np.zeros(n)
    for i in range(n):
        if price_mode == 2 and i >= p['ppa_term'] + 1:
            price[i] = merchant[p['ppa_term']] * (1 + p['post_esc']) ** (i - p['ppa_term'])
        elif i < nm:
            price[i] = merchant[i]
        else:
            price[i] = merchant[-1] * (1 + p['post_esc']) ** (i - nm + 1)
    price *= (1 - mh)

    con = yr <= p['ppa_term']
    ppa_rev = np.where(con, prod * ppa * (1 + p['ppa_esc']) ** (yr - 1), 0.0)
    merch_rev = np.where(con, 0.0, prod * price)
    rec_rev = np.where(con, 0.0, prod * rec)
    rev = ppa_rev + merch_rev + rec_rev

    e = lambda r: (1 + r) ** (yr - 1)
    opex = (mwac * p['om_cov'] * e(p['om_esc']) + mwac * p['om_non'] * e(p['om_esc'])
            + p['am'] * e(p['am_esc']) + acres * p['land'] * e(p['land_esc'])
            + acres * p['ptax'] * e(p['ptax_esc']))
    ebitda = rev - opex

    dep = np.zeros(n); k = min(len(depr), n); dep[:k] = depr[:k] * dbasis
    tax = np.zeros(n); nol = 0.0
    for i in range(n):
        ti = ebitda[i] - dep[i]
        if tax_mode == "Immediate monetisation":
            tax[i] = ti * p['tax_rate']
        else:
            used = min(max(ti, 0.0), nol)
            tax[i] = (max(ti, 0.0) - used) * p['tax_rate']
            nol = nol - used + max(-ti, 0.0)
    itc_cash = itc * (transfer if tax_mode.startswith("ITC transfer") else 1.0)
    itc_flow = np.zeros(n); itc_flow[0] = itc_cash
    atcf = ebitda - tax + itc_flow

    pre = np.concatenate(([-capex], ebitda)); aft = np.concatenate(([-capex], atcf))
    conf = np.concatenate(([-capex], atcf[:p['ppa_term']]))
    return dict(capex=capex, cx=cx, itc=itc, perwp=capex / wp, prod=prod, price=price,
                ppa_rev=ppa_rev, merch_rev=merch_rev, rec_rev=rec_rev, rev=rev, opex=opex,
                ebitda=ebitda, margin=np.where(rev > 0, ebitda / rev, np.nan), dep=dep, tax=tax,
                atcf=atcf, aft=aft, irr_pre=irr(pre), irr_at=irr(aft), irr_con=irr(conf),
                npv=npv(p['disc'], aft), cum=np.cumsum(atcf) - capex, yr=yr)


def solve_ppa(p, mwac, pv_bos, target, **kw):
    lo, hi = 0.0, 400.0
    for _ in range(60):
        mid = (lo + hi) / 2
        v = run_model(p, mwac, pv_bos, ppa=mid, **kw)['irr_at']
        if np.isnan(v) or v < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Solar Project Stress Test", page_icon="☀️", layout="wide")
st.title("☀️ Utility-Scale Solar — Stress-Test Dashboard")
st.caption("150 MWac Base vs 300 MWac Alternate · every assumption is live · "
           "reconciles to Intersect_Solar_Model_v3.xlsx to the cent at default inputs, all three tax structures")

with st.sidebar:
    st.header("Assumptions")
    with st.expander("Contract & production", expanded=True):
        ppa = st.number_input("PPA price ($/MWh)", 0.0, 200.0, 25.0, 0.25)
        ppa_term = st.slider("PPA term (years)", 5, 30, 15)
        ppa_esc = st.number_input("PPA escalation (%/yr)", 0.0, 5.0, 0.0, 0.25) / 100
        life = st.slider("Useful life (years)", 20, 45, 35)
        yld = st.number_input("Yield (kWh/kWp, pre-derate)", 1000, 3000, 2200, 10)
        degr = st.number_input("Degradation (%/yr)", 0.0, 2.0, 0.5, 0.05) / 100
        avail = st.number_input("Availability de-rate (%)", 0.0, 10.0, 1.0, 0.25) / 100
        dcac = st.number_input("DC/AC ratio", 1.0, 2.0, 1.4, 0.05)
    with st.expander("CapEx ($/Wp unless noted)"):
        module = st.number_input("Modules", 0.0, 2.0, 0.35, 0.01)
        pv_bos_b = st.number_input("PV BOS — Base", 0.0, 2.0, 0.50, 0.01)
        pv_bos_a = st.number_input("PV BOS — Alternate", 0.0, 2.0, 0.48, 0.01)
        hv_bos = st.number_input("HV BOS", 0.0, 1.0, 0.05, 0.01)
        dev = st.number_input("Development", 0.0, 1.0, 0.05, 0.01)
        interconnect = st.number_input("Interconnection ($, fixed per project)", 0.0, 1e8, 1e7, 5e5, format="%.0f")
    with st.expander("OpEx"):
        om_cov = st.number_input("Covered O&M ($/MWac/yr)", 0.0, 50000.0, 6500.0, 100.0)
        om_non = st.number_input("Non-covered O&M ($/MWac/yr)", 0.0, 50000.0, 2000.0, 100.0)
        om_esc = st.number_input("O&M escalation (%/yr)", 0.0, 6.0, 2.0, 0.25) / 100
        am = st.number_input("Asset management ($/yr, fixed)", 0.0, 2e6, 125000.0, 5000.0)
        am_esc = st.number_input("Asset mgmt escalation (%/yr)", 0.0, 6.0, 2.0, 0.25) / 100
        acres_per_mw = st.number_input("Land (acres/MWac)", 1.0, 15.0, 7.0, 0.5)
        land = st.number_input("Land lease ($/acre/yr)", 0.0, 5000.0, 475.0, 25.0)
        land_esc = st.number_input("Land escalation (%/yr)", 0.0, 6.0, 2.5, 0.25) / 100
        ptax = st.number_input("Property tax ($/acre/yr)", 0.0, 2000.0, 75.0, 5.0)
        ptax_esc = st.number_input("Property tax escalation (%/yr)", 0.0, 6.0, 2.0, 0.25) / 100
    with st.expander("Tax & finance", expanded=True):
        tax_mode = st.radio("Tax structure",
                            ["ITC transfer + NOL carryforward (primary)", "NOL carryforward (conservative)", "Immediate monetisation"],
                            help="Primary: ITC sold to a third party under IRA §6418 at a discount; depreciation losses stay with the project. "
                                 "NOL: project uses the full ITC itself and carries losses forward (floor). Immediate: a sponsor with taxable "
                                 "income absorbs every loss as cash (ceiling).")
        transfer = st.slider("ITC transfer price (¢ per $1 of credit)", 80, 100, 93, 1,
                             disabled=not tax_mode.startswith("ITC transfer")) / 100
        itc_rate = st.number_input("ITC rate (%)", 0.0, 60.0, 30.0, 1.0) / 100
        itc_elig = st.number_input("ITC eligibility on modules+BOS+dev (%)", 0.0, 100.0, 98.0, 1.0) / 100
        tax_rate = st.number_input("Federal tax rate (%)", 0.0, 50.0, 21.0, 0.5) / 100
        disc = st.number_input("Discount rate (%)", 0.0, 20.0, 7.0, 0.25) / 100
        target = st.number_input("Target after-tax IRR for PPA solve (%)", 0.0, 20.0, 7.5, 0.25) / 100
    with st.expander("Merchant tail"):
        rec = st.number_input("Merchant REC adder ($/MWh, post-PPA)", 0.0, 50.0, 0.0, 0.5)
        merch_haircut = st.slider("Merchant price haircut (%)", -50, 50, 0) / 100
        price_mode = st.radio("Post-PPA price path", [1, 2], format_func=lambda x:
                              "1 — supplied curve, then +esc after Y35 (company-confirmed)" if x == 1 else "2 — anchor Y16, +esc from Y17 (exploratory)")
        post_esc = st.number_input("Escalation beyond curve (%/yr)", 0.0, 6.0, 2.0, 0.25) / 100

P = dict(ppa=ppa, ppa_term=ppa_term, ppa_esc=ppa_esc, life=life, yld=yld, degr=degr, avail=avail,
         dcac=dcac, module=module, hv_bos=hv_bos, dev=dev, interconnect=interconnect,
         om_cov=om_cov, om_non=om_non, om_esc=om_esc, am=am, am_esc=am_esc, acres_per_mw=acres_per_mw,
         land=land, land_esc=land_esc, ptax=ptax, ptax_esc=ptax_esc, tax_mode=tax_mode,
         itc_rate=itc_rate, itc_elig=itc_elig, tax_rate=tax_rate, disc=disc, rec=rec, transfer=transfer,
         merch_haircut=merch_haircut, price_mode=price_mode, post_esc=post_esc,
         merchant=MERCHANT_DEFAULT, depr=DEPR_DEFAULT)

tab_sum, tab_cf, tab_sens, tab_ppa, tab_mc, tab_data = st.tabs(
    ["Summary", "Cash flows", "Sensitivities", "PPA solve", "Monte Carlo", "Data & model"])

# ------------------------------- data tab (editable curves) -----------------
with tab_data:
    st.subheader("Editable supplied curves")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Merchant price curve ($/MWh)**")
        mdf = st.data_editor(pd.DataFrame({"Year": range(1, 36), "Price": MERCHANT_DEFAULT}),
                             hide_index=True, num_rows="dynamic", height=420, key="merch")
        P['merchant'] = mdf["Price"].astype(float).tolist()
    with c2:
        st.markdown("**Depreciation schedule (share of basis)**")
        ddf = st.data_editor(pd.DataFrame({"Year": range(1, 17), "Share": DEPR_DEFAULT}),
                             hide_index=True, height=420, key="depr")
        P['depr'] = ddf["Share"].astype(float).tolist()
        st.metric("Schedule total", f"{sum(P['depr'])*100:.2f}%")
    st.markdown("""
**Conventions (all editable above):** CapEx at t=0; Year 1 = first operating year; end-year discounting;
unlevered all-equity; ITC monetised in Year 1; depreciable basis = CapEx − 50% × ITC;
RECs bundled in the PPA (no separate REC revenue during the contract); no terminal value.
""")

# ------------------------------- run cases ----------------------------------
B = run_model(P, 150, pv_bos_b); A = run_model(P, 300, pv_bos_a)
fmt_pct = lambda v: "n/m" if np.isnan(v) else f"{v*100:.2f}%"
fmt_mm = lambda v: f"${v/1e6:,.1f}mm"

with tab_sum:
    c = st.columns(4)
    c[0].metric("Base after-tax IRR", fmt_pct(B['irr_at']), delta=f"{(B['irr_at']-target)*1e4:+.0f} bps vs target")
    c[1].metric("Alternate after-tax IRR", fmt_pct(A['irr_at']), delta=f"{(A['irr_at']-B['irr_at'])*1e4:+.0f} bps vs Base")
    c[2].metric("Base NPV @ disc", fmt_mm(B['npv']))
    c[3].metric("Alternate NPV @ disc", fmt_mm(A['npv']))
    df = pd.DataFrame({
        "Metric": ["Nameplate (MWac / MWdc)", "Total CapEx", "All-in cost ($/Wp)", "ITC", "Year-1 production (MWh)",
                   "Year-1 EBITDA", "Pre-tax IRR", "After-tax IRR", "After-tax IRR (PPA term)", "NPV",
                   "Undiscounted payback (yr)"],
        "Base": [f"150 / {150*dcac:.0f}", fmt_mm(B['capex']), f"${B['perwp']:.4f}", fmt_mm(B['itc']), f"{B['prod'][0]:,.0f}",
                 fmt_mm(B['ebitda'][0]), fmt_pct(B['irr_pre']), fmt_pct(B['irr_at']), fmt_pct(B['irr_con']), fmt_mm(B['npv']),
                 int(np.argmax(B['cum'] > 0) + 1) if (B['cum'] > 0).any() else "never"],
        "Alternate": [f"300 / {300*dcac:.0f}", fmt_mm(A['capex']), f"${A['perwp']:.4f}", fmt_mm(A['itc']), f"{A['prod'][0]:,.0f}",
                      fmt_mm(A['ebitda'][0]), fmt_pct(A['irr_pre']), fmt_pct(A['irr_at']), fmt_pct(A['irr_con']), fmt_mm(A['npv']),
                      int(np.argmax(A['cum'] > 0) + 1) if (A['cum'] > 0).any() else "never"],
    })
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.subheader("Tax structure comparison (everything else as set in the sidebar)")
    trow = []
    for lbl in ["ITC transfer + NOL carryforward (primary)", "NOL carryforward (conservative)", "Immediate monetisation"]:
        b_ = run_model(P, 150, pv_bos_b, tax_mode=lbl); a_ = run_model(P, 300, pv_bos_a, tax_mode=lbl)
        trow.append([lbl, fmt_pct(b_['irr_at']), fmt_mm(b_['npv']), fmt_pct(a_['irr_at']), fmt_mm(a_['npv']),
                     f"${solve_ppa(P, 150, pv_bos_b, target, tax_mode=lbl):.2f}", f"${solve_ppa(P, 300, pv_bos_a, target, tax_mode=lbl):.2f}"])
    st.dataframe(pd.DataFrame(trow, columns=["Structure", "Base IRR", "Base NPV", "Alt IRR", "Alt NPV", "Base PPA @ target", "Alt PPA @ target"]),
                 hide_index=True, use_container_width=True)
    st.subheader("Scale economics")
    s1, s2, s3 = st.columns(3)
    s1.metric("CapEx saving vs 2×Base", fmt_mm(2*B['capex'] - A['capex']))
    s2.metric("Equal-capacity NPV gain (Alt − 2×Base)", fmt_mm(A['npv'] - 2*B['npv']))
    s3.metric("$/Wp reduction", f"${B['perwp']-A['perwp']:.4f}", delta=f"{(A['perwp']/B['perwp']-1)*100:.1f}%")

with tab_cf:
    which = st.radio("Case", ["Base", "Alternate"], horizontal=True)
    M = B if which == "Base" else A
    fig = go.Figure()
    fig.add_bar(x=M['yr'], y=M['ppa_rev']/1e6, name="PPA revenue", marker_color="#14213D")
    fig.add_bar(x=M['yr'], y=M['merch_rev']/1e6, name="Merchant energy", marker_color="#E9A13B")
    if rec > 0: fig.add_bar(x=M['yr'], y=M['rec_rev']/1e6, name="RECs", marker_color="#8FB8A8")
    fig.add_scatter(x=M['yr'], y=M['margin']*100, name="EBITDA margin (%)", yaxis="y2",
                    line=dict(color="#2F8F7A", width=3))
    fig.update_layout(barmode="stack", height=420, yaxis_title="$mm", legend=dict(orientation="h", y=-0.2),
                      yaxis2=dict(title="EBITDA margin %", overlaying="y", side="right", range=[0, 100]),
                      margin=dict(l=40, r=40, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_scatter(x=np.arange(0, len(M['cum'])+1), y=np.concatenate(([-M['capex']], M['cum']))/1e6,
                     fill="tozeroy", name="Cumulative after-tax cash", line=dict(color="#14213D"))
    fig2.add_hline(y=0, line_dash="dot")
    fig2.update_layout(height=280, yaxis_title="$mm", xaxis_title="Operating year", margin=dict(l=40, r=40, t=30, b=40))
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("Annual table"):
        st.dataframe(pd.DataFrame({"Year": M['yr'], "Production MWh": M['prod'].round(0), "Merchant $/MWh": M['price'].round(2),
                                   "Revenue": M['rev'].round(0), "OpEx": M['opex'].round(0), "EBITDA": M['ebitda'].round(0),
                                   "Depreciation": M['dep'].round(0), "Tax": M['tax'].round(0), "After-tax CF": M['atcf'].round(0)}),
                     hide_index=True, use_container_width=True, height=400)
        csv = pd.DataFrame({"Year": np.arange(0, len(M['aft'])), "After-tax CF": M['aft']}).to_csv(index=False)
        st.download_button("Download cash flows (CSV)", csv, f"{which.lower()}_cashflows.csv")

with tab_sens:
    st.subheader("One-at-a-time sensitivities (both cases)")
    shocks = {
        "PPA +$1/MWh": dict(ppa=ppa + 1), "PPA −$1/MWh": dict(ppa=ppa - 1),
        "RECs +$5/MWh (post-PPA)": dict(rec=rec + 5),
        "Modules +$0.01/Wp": dict(mod=module + 0.01), "Modules −$0.01/Wp": dict(mod=module - 0.01),
        "Merchant −20%": dict(mh=merch_haircut + 0.20),
        "40-yr life (Y35 price +2% for Y36+)": dict(life=40, price_mode=1),
    }
    rows = []
    for lbl, kw in shocks.items():
        mb = run_model(P, 150, pv_bos_b, **kw); ma = run_model(P, 300, pv_bos_a, **kw)
        rows.append([lbl, fmt_pct(mb['irr_at']), f"{(mb['irr_at']-B['irr_at'])*1e4:+.0f}", fmt_mm(mb['npv']-B['npv']),
                     fmt_pct(ma['irr_at']), f"{(ma['irr_at']-A['irr_at'])*1e4:+.0f}", fmt_mm(ma['npv']-A['npv'])])
    st.dataframe(pd.DataFrame(rows, columns=["Sensitivity", "Base IRR", "Δbps", "ΔNPV", "Alt IRR", "Δbps", "ΔNPV"]),
                 hide_index=True, use_container_width=True)
    st.subheader("Tornado — NPV impact per DC watt (Base case, so scale is comparable)")
    wpb = 150 * dcac * 1e6; wpa = 300 * dcac * 1e6
    torn = {
        "Merchant −20%": (run_model(P, 150, pv_bos_b, mh=merch_haircut + 0.2)['npv'] - B['npv']) / wpb,
        "Life 35→40": (run_model(P, 150, pv_bos_b, life=40, price_mode=1)['npv'] - B['npv']) / wpb,
        "Scale 150→300": A['npv'] / wpa - B['npv'] / wpb,
        "RECs +$5": (run_model(P, 150, pv_bos_b, rec=rec + 5)['npv'] - B['npv']) / wpb,
        "Modules +$0.01": (run_model(P, 150, pv_bos_b, mod=module + 0.01)['npv'] - B['npv']) / wpb,
    }
    items = sorted(torn.items(), key=lambda x: abs(x[1]))
    fig = go.Figure(go.Bar(x=[v * wpb / 1e6 for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=["#C15A3F" if v < 0 else "#2F8F7A" for _, v in items],
                           text=[f"{v*wpb/1e6:+.1f}" for _, v in items], textposition="outside"))
    fig.update_layout(height=340, xaxis_title="Δ NPV, $mm (150 MWac-equivalent)", margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

with tab_ppa:
    st.subheader(f"15-year PPA price required for a {target*100:.2f}% after-tax IRR")
    pb = solve_ppa(P, 150, pv_bos_b, target); pa = solve_ppa(P, 300, pv_bos_a, target)
    pe = solve_ppa(P, 300, pv_bos_a, B['irr_at']) if not np.isnan(B['irr_at']) else np.nan
    c = st.columns(3)
    c[0].metric("Base", f"${pb:.2f}/MWh", delta=f"{pb-ppa:+.2f} vs current")
    c[1].metric("Alternate", f"${pa:.2f}/MWh", delta=f"{pa-pb:+.2f} vs Base")
    c[2].metric("Alternate PPA matching Base IRR", f"${pe:.2f}/MWh" if not np.isnan(pe) else "n/m")
    rng = np.arange(0.05, 0.1001, 0.005)
    lad_b = [solve_ppa(P, 150, pv_bos_b, t) for t in rng]; lad_a = [solve_ppa(P, 300, pv_bos_a, t) for t in rng]
    fig = go.Figure()
    fig.add_scatter(x=rng*100, y=lad_b, name="Base", line=dict(color="#E9A13B", width=3))
    fig.add_scatter(x=rng*100, y=lad_a, name="Alternate", line=dict(color="#2F8F7A", width=3))
    fig.add_hline(y=ppa, line_dash="dot", annotation_text=f"current ${ppa:.2f}")
    fig.update_layout(height=360, xaxis_title="Target after-tax IRR (%)", yaxis_title="Required PPA ($/MWh)",
                      margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**NPV impact of overlays at the solved PPA**")
    rows = []
    for nm_, mw, pv, p_ in [("Base", 150, pv_bos_b, pb), ("Alternate", 300, pv_bos_a, pa)]:
        r0 = run_model(P, mw, pv, ppa=p_)['npv']
        rows.append([nm_, fmt_mm(run_model(P, mw, pv, ppa=p_, rec=rec+5)['npv'] - r0),
                     fmt_mm(run_model(P, mw, pv, ppa=p_, life=40, price_mode=1)['npv'] - r0)])
    st.dataframe(pd.DataFrame(rows, columns=["Case", "+$5 RECs", "40-yr life"]), hide_index=True)

with tab_mc:
    st.subheader("Monte Carlo on the merchant tail (the dominant risk)")
    st.caption("Each trial multiplies the whole post-PPA price curve by a lognormal shock and adds an "
               "independent annual noise term. Everything else held at sidebar values.")
    c1, c2, c3, c4 = st.columns(4)
    n_sim = c1.slider("Trials", 200, 5000, 1000, 100)
    level_sd = c2.slider("Curve-level σ (%)", 0, 60, 25) / 100
    noise_sd = c3.slider("Annual noise σ (%)", 0, 30, 8) / 100
    case = c4.radio("Case", ["Base", "Alternate"], horizontal=True)
    mw, pv = (150, pv_bos_b) if case == "Base" else (300, pv_bos_a)
    rng_ = np.random.default_rng(42)
    base_curve = np.array(P['merchant'], dtype=float)
    irrs, npvs = [], []
    for _ in range(n_sim):
        lvl = np.exp(rng_.normal(-0.5 * level_sd**2, level_sd))
        noise = np.exp(rng_.normal(-0.5 * noise_sd**2, noise_sd, size=len(base_curve)))
        m = run_model(P, mw, pv, merchant=(base_curve * lvl * noise).tolist())
        irrs.append(m['irr_at']); npvs.append(m['npv'])
    irrs = np.array(irrs); npvs = np.array(npvs)
    k = st.columns(4)
    k[0].metric("P(IRR < target)", f"{np.mean(irrs < target)*100:.0f}%")
    k[1].metric("P(NPV < 0)", f"{np.mean(npvs < 0)*100:.0f}%")
    k[2].metric("P10 / P50 / P90 IRR", f"{np.nanpercentile(irrs,10)*100:.1f} / {np.nanpercentile(irrs,50)*100:.1f} / {np.nanpercentile(irrs,90)*100:.1f}%")
    k[3].metric("P10 / P90 NPV", f"{np.percentile(npvs,10)/1e6:.0f} / {np.percentile(npvs,90)/1e6:.0f} $mm")
    fig = go.Figure(go.Histogram(x=irrs*100, nbinsx=50, marker_color="#14213D"))
    fig.add_vline(x=target*100, line_color="#C15A3F", line_dash="dash", annotation_text="target")
    fig.update_layout(height=320, xaxis_title="After-tax IRR (%)", yaxis_title="Trials", margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Model: unlevered, nominal, end-year discounting. Reconciled to Intersect_Solar_Model_v2.xlsx and to an "
           "independent engine under both tax conventions. Not investment advice.")
