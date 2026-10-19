"""
Account E - The Big Game: Attention, Creative & Culture
Streamlit MVP demo for Team-6's Mandate:
  "Which creative attributes should we advise clients to build into their
   Super Bowl ads to drive both strong live-audience approval and lasting
   online attention?"

Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

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

tab_compare, tab_correlate, tab_rank, tab_locate, tab_phase2 = st.tabs(
    ["📊 Compare", "📈 Correlate", "🏆 Rank", "🗺️ Locate / Overview", "🧩 Phase 2: Attribute Combo Explorer"]
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


st.caption(
    f"Dataset: {len(df)} Super Bowl ads, {df['year'].min()}-{df['year'].max()} "
    f"(FiveThirtyEight base + YouTube Data API + USA Today Ad Meter + Wikipedia pageviews)."
)
