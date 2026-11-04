"""
Account E - The Big Game: Attention, Creative & Culture
Streamlit MVP demo for Team-6's Mandate:
  "Which creative attributes should we advise clients to build into their
   Super Bowl ads to drive both strong live-audience approval and lasting
   online attention?"

Run with:  streamlit run app.py
"""

import os
import re

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

try:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Local runs read GEMINI_API_KEY from .env (python-dotenv, above). On
    # Streamlit Community Cloud there is no .env file (and this repo is
    # public, so one should never be committed there) - the key instead
    # comes from the app's Settings -> Secrets panel, exposed as st.secrets.
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        GEMINI_API_KEY = None
GEMINI_MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Palette (Anthropic dataviz reference palette - the same colors used in the
# team's slide deck and static charts, so the app matches the deck).
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

ATTRS = ["funny", "show_product_quickly", "patriotic", "celebrity", "danger", "animals", "use_sex"]
ATTR_LABELS = {
    "funny": "Funny",
    "show_product_quickly": "Shows product quickly",
    "patriotic": "Patriotic",
    "celebrity": "Celebrity",
    "danger": "Danger / action",
    "animals": "Animals",
    "use_sex": "Sex appeal",
}

# ---------------------------------------------------------------------------
# Creative Brief Assistant - helpers (AI Integration: Defined, now built)
#
# The stats table below is computed in plain Python/scipy, exactly like the
# Compare tab's Mann-Whitney test. The AI never sees raw data and never
# assigns its own Use/Avoid/Unproven label - it is handed the label already
# decided by code and told it may not override it. After each reply, the
# numbers the AI cites are checked against this same table.
# ---------------------------------------------------------------------------
MIN_GROUP_N = 10
SIG_LEVEL = 0.05


def _label_for_group(p_val, n_with, n_without, with_mean, without_mean):
    """Use/Avoid requires a real sample on both sides and a significant p-value;
    everything else is Unproven. This is the one rule the AI is not allowed to
    relabel, no matter how it reads the numbers."""
    if with_mean is None or without_mean is None or p_val is None:
        return "Unproven"
    if n_with < MIN_GROUP_N or n_without < MIN_GROUP_N or p_val >= SIG_LEVEL:
        return "Unproven"
    return "Use" if with_mean > without_mean else "Avoid"


def compute_attribute_stats(fdf):
    """Builds the one stats table the Creative Brief Assistant is allowed to
    cite from, for every attribute, under the current sidebar filters."""
    am = fdf[fdf["has_admeter"] == True]
    yt = fdf[fdf["has_youtube"] == True]
    wiki = fdf[fdf["has_wiki_attention"] == True]

    rows = []
    for a in ATTRS:
        with_grp = am[am[a] == True]["ad_meter_percentile"].dropna()
        without_grp = am[am[a] == False]["ad_meter_percentile"].dropna()
        n_with, n_without = len(with_grp), len(without_grp)

        p_val = None
        if n_with >= 3 and n_without >= 3:
            _, p_val = stats.mannwhitneyu(with_grp, without_grp, alternative="two-sided")

        with_mean = float(with_grp.mean()) if n_with else None
        without_mean = float(without_grp.mean()) if n_without else None
        label = _label_for_group(p_val, n_with, n_without, with_mean, without_mean)

        yt_with = yt[yt[a] == True]["view_count"].dropna()
        wiki_with = wiki[wiki[a] == True]["wiki_attention_spike"].dropna()

        rows.append({
            "attribute": ATTR_LABELS[a],
            "n_with": n_with,
            "n_without": n_without,
            "approval_with_pct": round(with_mean * 100, 1) if with_mean is not None else None,
            "approval_without_pct": round(without_mean * 100, 1) if without_mean is not None else None,
            "p_value": round(p_val, 3) if p_val is not None else None,
            "label": label,
            "avg_youtube_views": int(yt_with.mean()) if len(yt_with) else None,
            "avg_wiki_attention_spike": round(float(wiki_with.mean()), 2) if len(wiki_with) else None,
        })
    return rows


def build_system_instruction(rows, year_range, brand_pick):
    header = "| Attribute | n(with) | n(without) | Approval w/ | Approval w/o | p-value | Label | Avg YouTube views (w/) | Avg Wiki spike (w/) |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['attribute']} | {r['n_with']} | {r['n_without']} | "
            f"{r['approval_with_pct']}% | {r['approval_without_pct']}% | "
            f"{r['p_value']} | {r['label']} | {r['avg_youtube_views']} | {r['avg_wiki_attention_spike']} |"
        )
    table_md = "\n".join(lines)

    filters_desc = f"years {year_range[0]}-{year_range[1]}"
    filters_desc += f", brands: {', '.join(brand_pick)}" if brand_pick else ", all brands"

    return f"""You are the Creative Brief Assistant for Account E's Super Bowl ad strategy team.

Current dashboard filters: {filters_desc}.

STATS TABLE - the only numbers you may cite. Do not use any number, percentage, or
statistic that is not in this table, and do not compute new ones:

{table_md}

The "Label" column is pre-computed by code and is the ONLY label you may use for
that attribute (Use / Avoid / Unproven). Never relabel an attribute yourself, even
if it seems to make a better story - Unproven means n < {MIN_GROUP_N} in either
group or p >= {SIG_LEVEL}, and that rule is not negotiable.

When asked for a creative brief, respond with 3-5 recommendations. For each one:
- Name the attribute and its label (Use, Avoid, or Unproven)
- Cite the exact statistic from the table that supports it (percentages, p-value, or n)
- Keep it to one or two sentences

End with one line of caveats naming any real data gaps (e.g. attributes left
Unproven because of small sample size).

If asked something the stats table can't answer, say so plainly rather than
inventing a number."""


def validate_citations(response_text, rows):
    """Lightweight check that every number-looking token the AI cited actually
    appears (or closely rounds to) a number in the stats table. Mirrors the
    deck's 'a brief citing any value not in the table is regenerated' rule -
    here it flags for a human to check rather than silently retrying."""
    table_numbers = []
    for r in rows:
        for key in ("n_with", "n_without", "approval_with_pct", "approval_without_pct",
                    "p_value", "avg_youtube_views", "avg_wiki_attention_spike"):
            v = r.get(key)
            if v is not None:
                table_numbers.append(float(v))

    tokens = re.findall(r"\d[\d,]*\.?\d*%?", response_text)
    suspicious = []
    for tok in tokens:
        bare = tok.rstrip("%").replace(",", "")
        try:
            val = float(bare)
        except ValueError:
            continue
        if val < MIN_GROUP_N and "%" not in tok:
            continue  # skip small counts like "3-5 recommendations"
        if any(abs(val - n) < 0.6 for n in table_numbers):
            continue
        suspicious.append(tok)
    return suspicious


st.set_page_config(page_title="Account E - Super Bowl Ad Strategy", page_icon="🏈", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        background-image: none !important;
    }
    [data-testid="stSidebar"] {
        background-color: #c0392b !important;
        background-image: none !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div {
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    df = pd.read_csv("superbowl_unified.csv")
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("Account E: Super Bowl Ad Strategy")
    st.markdown(
        "**Mandate:** Which creative attributes should we advise clients to build into "
        "their Super Bowl ads to drive both strong **live-audience approval** and "
        "**lasting online attention**? &mdash; built on FiveThirtyEight's base dataset, "
        "augmented with YouTube, USA Today Ad Meter, and Wikipedia pageview data."
    )
with header_right:
    st.markdown(
        f"""
        <svg viewBox="0 0 160 200" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:150px;">
          <g transform="rotate(-15 80 40)">
            <ellipse cx="80" cy="40" rx="28" ry="16" fill="{ORANGE}" stroke="{INK_PRIMARY}" stroke-width="2"/>
            <line x1="52" y1="40" x2="108" y2="40" stroke="{INK_PRIMARY}" stroke-width="1.5"/>
            <line x1="68" y1="34" x2="68" y2="46" stroke="{INK_PRIMARY}" stroke-width="1.5"/>
            <line x1="76" y1="33" x2="76" y2="47" stroke="{INK_PRIMARY}" stroke-width="1.5"/>
            <line x1="84" y1="33" x2="84" y2="47" stroke="{INK_PRIMARY}" stroke-width="1.5"/>
            <line x1="92" y1="34" x2="92" y2="46" stroke="{INK_PRIMARY}" stroke-width="1.5"/>
          </g>
          <ellipse cx="38" cy="75" rx="9" ry="16" fill="none" stroke="{BLUE}" stroke-width="5"/>
          <ellipse cx="122" cy="75" rx="9" ry="16" fill="none" stroke="{BLUE}" stroke-width="5"/>
          <rect x="45" y="58" width="70" height="6" rx="3" fill="{BLUE}"/>
          <path d="M50,64 L110,64 L98,100 L62,100 Z" fill="{BLUE}"/>
          <rect x="74" y="100" width="12" height="30" fill="{BLUE}"/>
          <rect x="55" y="130" width="50" height="12" rx="3" fill="{BLUE}"/>
          <rect x="5" y="160" width="150" height="28" rx="6" fill="{INK_PRIMARY}"/>
          <text x="80" y="179" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" letter-spacing="1.5" fill="white">SUPER BOWL</text>
        </svg>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

all_brands = sorted(df["brand"].dropna().unique())
brand_pick = st.sidebar.multiselect("Brand (leave empty = all brands)", all_brands)

st.sidebar.markdown("**Creative attributes present**")
attr_filter = {}
for a in ATTRS:
    attr_filter[a] = st.sidebar.checkbox(ATTR_LABELS[a], value=False, key=f"filt_{a}")

st.sidebar.markdown("**Data source required**")
req_youtube = st.sidebar.checkbox("Has YouTube stats", value=False)
req_admeter = st.sidebar.checkbox("Has Ad Meter score", value=False)
req_wiki = st.sidebar.checkbox("Has Wikipedia attention data", value=False)

# apply filters
mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
if brand_pick:
    mask &= df["brand"].isin(brand_pick)
for a, on in attr_filter.items():
    if on:
        mask &= df[a] == True
if req_youtube:
    mask &= df["has_youtube"] == True
if req_admeter:
    mask &= df["has_admeter"] == True
if req_wiki:
    mask &= df["has_wiki_attention"] == True

fdf = df[mask].copy()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.subheader("📌 Key Metrics")
kpi_data = [
    ("Ads in current view", f"{len(fdf):,}", BLUE),
    ("With YouTube stats", f"{int(fdf['has_youtube'].sum()):,}", ORANGE),
    ("With Ad Meter score", f"{int(fdf['has_admeter'].sum()):,}", AQUA),
    ("With Wikipedia attention", f"{int(fdf['has_wiki_attention'].sum()):,}", VIOLET),
]
kpi_cols = st.columns(4)
for col, (label, value, color) in zip(kpi_cols, kpi_data):
    col.markdown(
        f"""
        <div style="
            background-color: #f0efec;
            border-left: 5px solid {color};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 4px;
        ">
            <div style="color: #52514e; font-size: 0.85rem;">{label}</div>
            <div style="color: #0b0b0b; font-size: 1.9rem; font-weight: 700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if len(fdf) == 0:
    st.warning("No ads match the current filters. Loosen a filter in the sidebar.")
    st.stop()

tab_compare, tab_correlate, tab_rank, tab_locate, tab_phase2, tab_ai = st.tabs(
    ["📊 Compare", "📈 Correlate", "🏆 Rank", "🗺️ Locate / Overview",
     "🧩 Phase 2: Attribute Combo Explorer", "🤖 Creative Brief Assistant"]
)

# ---------------------------------------------------------------------------
# TAB 1 - COMPARE: does a given attribute move Ad Meter approval?
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Does a creative attribute change live-audience approval?")
    st.caption(
        "Task: COMPARE. Marks: points + connecting line (a dumbbell). "
        "Channels: position/length (mean approval) x color (with / without)."
    )

    attr_choice = st.selectbox(
        "Pick an attribute", ATTRS, format_func=lambda a: ATTR_LABELS[a], key="compare_attr"
    )

    am = fdf[fdf["has_admeter"] == True]
    with_grp = am[am[attr_choice] == True]["ad_meter_percentile"].dropna()
    without_grp = am[am[attr_choice] == False]["ad_meter_percentile"].dropna()

    if len(with_grp) < 3 or len(without_grp) < 3:
        st.info("Not enough Ad Meter-scored ads with this attribute (and this filter) to compare reliably.")
    else:
        u_stat, p_val = stats.mannwhitneyu(with_grp, without_grp, alternative="two-sided")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[without_grp.mean(), with_grp.mean()], y=[0, 0],
            mode="lines", line=dict(color=GRID, width=3), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[without_grp.mean()], y=[0], mode="markers",
            marker=dict(color=ORANGE, size=18), name=f"Without {ATTR_LABELS[attr_choice]}",
        ))
        fig.add_trace(go.Scatter(
            x=[with_grp.mean()], y=[0], mode="markers",
            marker=dict(color=BLUE, size=18), name=f"With {ATTR_LABELS[attr_choice]}",
        ))
        fig.update_layout(
            yaxis=dict(visible=False, range=[-1, 1]),
            xaxis=dict(title="Mean Ad Meter approval percentile", tickformat=".0%", gridcolor=GRID),
            plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
            height=260, margin=dict(t=20, b=40),
            legend=dict(orientation="h", y=1.2),
        )
        st.plotly_chart(fig, width='stretch')

        sig = "statistically significant (p < 0.05)" if p_val < 0.05 else "not statistically significant"
        st.caption(
            f"With: {with_grp.mean():.1%} avg approval (n={len(with_grp)}) &nbsp;|&nbsp; "
            f"Without: {without_grp.mean():.1%} avg approval (n={len(without_grp)}) &nbsp;|&nbsp; "
            f"Mann-Whitney p = {p_val:.3f} ({sig})"
        )

    with st.expander("Why a dumbbell chart, and what we rejected"):
        st.markdown(
            "- **Rejected: a single bar per group.** Two separate bars make the *gap* between "
            "with/without hard to read at a glance; a dumbbell puts the gap itself on the page as "
            "the connecting line, which is the actual quantity the client cares about.\n"
            "- **Rejected: a box plot.** Box plots show spread well, but the client's question is "
            "about the *mean difference*, not distribution shape, and a box plot is heavier to read "
            "for a non-technical audience.\n"
            "- **Chosen: dumbbell (2 points + a line).** Position encodes the two means directly; "
            "color is the only extra channel needed (with/without), keeping cognitive load low."
        )

# ---------------------------------------------------------------------------
# TAB 2 - CORRELATE: does approval predict reach?
# ---------------------------------------------------------------------------
with tab_correlate:
    st.subheader("Does a well-liked ad travel further online?")
    st.caption(
        "Task: CORRELATE. Marks: points (scatter). "
        "Channels: position (x = approval percentile, y = YouTube views, log scale)."
    )

    both = fdf[(fdf["has_youtube"] == True) & (fdf["has_admeter"] == True)].dropna(
        subset=["ad_meter_percentile", "view_count"]
    )

    if len(both) < 5:
        st.info("Not enough ads with both an Ad Meter score and YouTube views under this filter.")
    else:
        rho, p_val = stats.spearmanr(both["ad_meter_percentile"], both["view_count"])
        fig = px.scatter(
            both, x="ad_meter_percentile", y="view_count",
            hover_data=["brand", "year"],
            labels={"ad_meter_percentile": "Ad Meter approval percentile", "view_count": "YouTube views"},
            color_discrete_sequence=[BLUE],
        )
        fig.update_yaxes(type="log", gridcolor=GRID)
        fig.update_xaxes(tickformat=".0%", gridcolor=GRID)
        fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="white")))
        fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, height=440)
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Spearman rho = {rho:.2f}, p = {p_val:.3f}, n = {len(both)}")

    with st.expander("Why a scatter plot, and what we rejected"):
        st.markdown(
            "- **Rejected: a bar chart of average views per approval bucket.** Bucketing approval "
            "into bins throws away the exact relationship and can hide or exaggerate the trend "
            "depending on bin width.\n"
            "- **Rejected: a line chart.** There's no natural ordering/time axis here - approval "
            "percentile is continuous, not sequential, so connecting points with a line would "
            "imply an order that doesn't exist.\n"
            "- **Chosen: scatter, log y-axis.** Each ad is one point; position shows the real "
            "relationship with no binning, and the log scale is necessary because views span "
            "several orders of magnitude (hundreds to 180M+)."
        )

# ---------------------------------------------------------------------------
# TAB 3 - RANK: biggest Wikipedia attention spikes
# ---------------------------------------------------------------------------
with tab_rank:
    st.subheader("Which ads got the biggest online attention spike?")
    st.caption(
        "Task: RANK / IDENTIFY EXTREMES. Marks: bars. "
        "Channels: length (spike size) x position (sorted)."
    )

    top_n = st.slider("How many ads to show", 5, 20, 10, key="rank_n")

    wiki = fdf[fdf["has_wiki_attention"] == True].copy()
    wiki["label"] = wiki["brand"] + " " + wiki["year"].astype(str)
    wiki = wiki.drop_duplicates(subset=["brand", "year"])
    top = wiki.nlargest(top_n, "wiki_attention_spike").sort_values("wiki_attention_spike")

    if len(top) == 0:
        st.info("No ads with Wikipedia attention data under this filter.")
    else:
        fig = px.bar(
            top, x="wiki_attention_spike", y="label", orientation="h",
            labels={"wiki_attention_spike": "Attention spike (x normal)", "label": ""},
            color_discrete_sequence=[AQUA],
        )
        fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, height=max(300, 28 * len(top)))
        fig.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig, width='stretch')

    with st.expander("Why a horizontal bar chart, and what we rejected"):
        st.markdown(
            "- **Rejected: a vertical bar chart.** Ad/brand names are long text labels - horizontal "
            "bars let labels run left-to-right in full instead of being rotated or truncated.\n"
            "- **Rejected: a lollipop/dot chart.** Looks similar but the filled bar's length is a "
            "slightly stronger magnitude cue than a dot at the end of a thin stem, and these values "
            "span a wide range (0.25x to 30x) where that clarity matters.\n"
            "- **Chosen: sorted horizontal bars.** Sorting turns the chart into an instant ranking - "
            "the biggest spikes are simply at the top."
        )

# ---------------------------------------------------------------------------
# TAB 4 - LOCATE / OVERVIEW: data coverage by year
# ---------------------------------------------------------------------------
with tab_locate:
    st.subheader("Which years does each data source actually cover?")
    st.caption(
        "Task: LOCATE / OVERVIEW. Marks: bars. "
        "Channels: length (count) x position (year) x color (source)."
    )

    years = sorted(fdf["year"].unique())
    sources = ["has_youtube", "has_admeter", "has_wiki_attention"]
    source_labels = ["YouTube metrics", "Ad Meter score", "Wikipedia attention"]

    rows = []
    for s, lbl in zip(sources, source_labels):
        for y in years:
            rows.append({"year": y, "source": lbl, "count": int(fdf[(fdf["year"] == y) & (fdf[s] == True)].shape[0])})
    cov = pd.DataFrame(rows)

    fig = px.bar(
        cov, x="year", y="count", color="source", barmode="group",
        color_discrete_map={"YouTube metrics": BLUE, "Ad Meter score": ORANGE, "Wikipedia attention": AQUA},
        labels={"count": "Number of ads with this data", "year": "Year"},
    )
    fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, height=440, legend_title_text="")
    fig.update_xaxes(type="category", gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Why a grouped bar chart, and what we rejected"):
        st.markdown(
            "- **Rejected: a stacked bar chart.** Stacking would imply the three sources add up to "
            "some meaningful total, but an ad can appear in more than one source at once - stacking "
            "would misrepresent that as a sum.\n"
            "- **Rejected: a line chart per source.** Lines imply a continuous trend between years, "
            "but coverage is a per-year count, not a smooth process, and the gap years (2021-2022, "
            "no Ad Meter before 2023) are better shown as literal gaps/zero bars than interpolated.\n"
            "- **Chosen: grouped bars, one color per source.** Makes the coverage gaps (e.g. zero "
            "Ad Meter bars before 2023) immediately visible, which is the actual point of this chart."
        )

# ---------------------------------------------------------------------------
# TAB 5 - PHASE 2: Attribute Combo Explorer (self-generated opportunity feature)
# ---------------------------------------------------------------------------
with tab_phase2:
    st.subheader("Phase 2 opportunity: Attribute Combo Explorer")
    st.caption(
        "A client pitching a concept can check how similar past ads (by creative attribute "
        "combination) actually performed, instead of relying on gut feel. This goes beyond the "
        "single-attribute Compare tab by testing attribute *combinations* at once."
    )

    combo = st.multiselect(
        "Build a creative concept: which attributes would the ad have?",
        ATTRS, format_func=lambda a: ATTR_LABELS[a], key="combo_pick",
    )

    if not combo:
        st.info("Pick one or more attributes above to see how that combination has historically performed.")
    else:
        combo_mask = np.ones(len(df), dtype=bool)
        for a in combo:
            combo_mask &= (df[a] == True).values
        combo_df = df[combo_mask]

        overall_approval = df.loc[df["has_admeter"] == True, "ad_meter_percentile"].mean()
        overall_views = df.loc[df["has_youtube"] == True, "view_count"].mean()

        combo_approval = combo_df.loc[combo_df["has_admeter"] == True, "ad_meter_percentile"]
        combo_views = combo_df.loc[combo_df["has_youtube"] == True, "view_count"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Ads matching this combo", f"{len(combo_df)}")
        c2.metric(
            "Avg approval (this combo)",
            f"{combo_approval.mean():.0%}" if len(combo_approval) else "n/a",
            delta=f"{(combo_approval.mean() - overall_approval):+.0%} vs. overall" if len(combo_approval) else None,
        )
        c3.metric(
            "Avg YouTube views (this combo)",
            f"{combo_views.mean():,.0f}" if len(combo_views) else "n/a",
            delta=f"{(combo_views.mean() - overall_views):+,.0f} vs. overall" if len(combo_views) else None,
        )

        if len(combo_approval) < 5 or len(combo_views) < 5:
            st.warning(
                f"Only {len(combo_df)} historical ads match this exact combination - treat these "
                "numbers as a directional signal, not a guarantee."
            )

        st.markdown("**Matching ads**")
        show_cols = ["year", "brand", "ad_title"] + combo + ["ad_meter_percentile", "view_count"]
        st.dataframe(combo_df[show_cols].sort_values("year", ascending=False), width='stretch', hide_index=True)

    with st.expander("Why this feature, and what's not built yet"):
        st.markdown(
            "- **What it does today:** filters the full 412-ad history to an exact attribute "
            "combination and compares its average approval/reach to the dataset overall - entirely "
            "computed from data already in the app, no new data needed.\n"
            "- **What's deferred to Sprint-to-Final:** the AI layer (see below) will turn this from "
            "\"look up similar past ads\" into \"describe your concept in a sentence and get the "
            "closest-matching historical combination suggested automatically.\""
        )

# ---------------------------------------------------------------------------
# TAB 6 - CREATIVE BRIEF ASSISTANT (AI Integration: Defined, now built)
# ---------------------------------------------------------------------------
with tab_ai:
    st.subheader("Creative Brief Assistant")
    st.caption(
        "Task: GENERATE. Turns the current filtered view into a short, evidence-backed "
        "creative brief for next year's concept pitch. Every recommendation must cite a "
        "number from the stats table below - the AI cannot invent its own statistics or "
        "override the Use / Avoid / Unproven label."
    )

    if not _GENAI_AVAILABLE:
        st.warning(
            "The Gemini SDK isn't installed in this environment. Run "
            "`pip install google-genai python-dotenv` (or `uv sync` if this project uses uv) "
            "and restart the app."
        )
    elif not GEMINI_API_KEY:
        st.warning(
            "No GEMINI_API_KEY found. Add one to a `.env` file in this folder:\n\n"
            "`GEMINI_API_KEY=your_api_key_here`"
        )
    else:
        stats_rows = compute_attribute_stats(fdf)

        if "brief_messages" not in st.session_state:
            st.session_state.brief_messages = []

        for msg in st.session_state.brief_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("flag"):
                    st.caption(msg["flag"])

        prompt = st.chat_input(
            "Ask for a creative brief, e.g. “What should we recommend for next year's pitch?”"
        )
        if prompt:
            st.session_state.brief_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            client = genai.Client(api_key=GEMINI_API_KEY)
            system_instruction = build_system_instruction(stats_rows, year_range, brand_pick)

            contents = []
            for msg in st.session_state.brief_messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
                )

            with st.chat_message("assistant"):
                placeholder = st.empty()
                assistant_content = ""
                try:
                    for chunk in client.models.generate_content_stream(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(system_instruction=system_instruction),
                    ):
                        assistant_content += chunk.text or ""
                        placeholder.markdown(assistant_content)
                except Exception as e:
                    assistant_content = f"Sorry, the request to Gemini failed: {e}"
                    placeholder.markdown(assistant_content)

                suspicious = validate_citations(assistant_content, stats_rows)
                if suspicious:
                    flag = (
                        "⚠️ Numbers not found in the stats table: "
                        + ", ".join(suspicious) + " - verify before using."
                    )
                else:
                    flag = "✅ All cited numbers matched the stats table."
                st.caption(flag)

            st.session_state.brief_messages.append(
                {"role": "assistant", "content": assistant_content, "flag": flag}
            )

        if st.session_state.brief_messages and st.button("Clear conversation"):
            st.session_state.brief_messages = []
            st.rerun()

    with st.expander("How this is validated"):
        st.markdown(
            "- The **Label** column (Use / Avoid / Unproven) is computed in Python from the "
            "same Mann-Whitney test used in the Compare tab, *before* the AI ever sees it - "
            "the model is told which label is allowed and cannot override it.\n"
            f"- Use or Avoid requires p < {SIG_LEVEL} **and** n ≥ {MIN_GROUP_N} in both "
            "groups; everything else is labeled Unproven, including low-n attributes like "
            "Patriotic.\n"
            "- After each reply, the numbers in the AI's response are checked against the "
            "stats table; any number that doesn't match is flagged for a human to "
            "double-check rather than trusted automatically.\n"
            f"- Model: Gemini (`{GEMINI_MODEL}`) via the Gemini API."
        )

# ---------------------------------------------------------------------------


st.caption(
    f"Dataset: {len(df)} Super Bowl ads, {df['year'].min()}-{df['year'].max()} "
    f"(FiveThirtyEight base + YouTube Data API + USA Today Ad Meter + Wikipedia pageviews)."
)
