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
# "gemini-3-flash-preview" was retired (404 NOT_FOUND). To pin a different model
# without editing code, set GEMINI_MODEL in .env or in Streamlit Secrets. If the
# chosen model returns 404, the app automatically tries the fallbacks below.
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.6-flash"
GEMINI_FALLBACKS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


def stream_gemini(client, contents, system_instruction):
    """Streams a reply, moving to the next model only if one is not found (404)."""
    last_err = None
    for model_name in [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]:
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
            ):
                started = True
                yield chunk
            return
        except Exception as e:
            if started or not ("404" in str(e) or "NOT_FOUND" in str(e)):
                raise
            last_err = e
    raise last_err

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

Talk like a sharp creative strategist having a real conversation, not a form you're
filling out. Match your answer to the question: a quick question gets a quick,
direct answer; "give me a creative brief" gets a fuller write-up. Don't force every
reply into a numbered list or a fixed number of bullets - use plain paragraphs or a
short list, whichever reads more naturally.

You may draw on general advertising/marketing knowledge for color, context, or
framing (e.g. why a trend might exist, how it compares to industry norms) - but
make it clear when you're doing that, and never let it imply something IS true of
Account E's ads unless the stats table backs it up. Every claim you make about what
*this* data shows must trace back to that table (approval percentages and/or sample
size n - do not state the p-value number itself, use it only internally to judge
confidence). If asked something the stats table can't answer, say so plainly rather
than inventing a number.

When it's natural, end a full brief with a short line of caveats about real data
gaps (e.g. attributes left Unproven because of small sample size) - but don't
force this onto a one-line answer where it doesn't fit."""


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
    return _flag_numbers(response_text, table_numbers)


def _flag_numbers(response_text, allowed_numbers):
    """Returns the number-looking tokens in the response that don't match any
    allowed number. Years (e.g. 2000, 2025) and tiny counts are ignored - they
    are filter bounds / list positions, not statistics."""
    tokens = re.findall(r"\d[\d,]*\.?\d*%?", response_text)
    suspicious = []
    for tok in tokens:
        bare = tok.rstrip("%").replace(",", "").rstrip(".")
        try:
            val = float(bare)
        except ValueError:
            continue
        if "%" not in tok and 1900 <= val <= 2100 and val == int(val):
            continue  # a year, not a statistic
        if val < MIN_GROUP_N and "%" not in tok:
            continue  # skip small counts like "3-5 recommendations"
        if any(abs(val - n) < 0.6 or (n >= 1000 and abs(val - n) / n < 0.005)
               for n in allowed_numbers):
            continue
        suspicious.append(tok)
    return suspicious


# ---------------------------------------------------------------------------
# Homepage assistant ("Ask the Dashboard") - helpers
#
# One chat box in the sidebar that can answer questions about every page. It is
# handed (1) a plain-language guide to each page and (2) the live numbers for the
# current sidebar filters. Same rules as the Creative Brief Assistant: the model
# may only cite numbers from that context, and every number in its reply is
# checked against it afterwards.
# ---------------------------------------------------------------------------
PAGE_GUIDE = """PAGE GUIDE (what each part of the dashboard is):

SIDEBAR FILTERS + KEY METRICS: Year range, brand, "creative attributes present" and
"data source required" narrow every page except Phase 2 (which always uses the full
history). The four Key Metrics cards count ads in the current view and how many have
YouTube stats, an Ad Meter score, and Wikipedia attention data.

COMPARE tab: "Does a creative attribute change live-audience approval?" A dumbbell chart:
one row per creative attribute, each with two dots joined by a line. The blue dot is the average
Ad Meter approval of ads WITH the attribute, the orange dot is ads WITHOUT it; the line between
them is the gap, so a longer line means a bigger effect. Rows are sorted with the biggest boost
at the top and the biggest drop at the bottom, and each row is tagged Use / Avoid / Unproven.
Hovering a dot shows its average and number of ads. A dropdown below the chart shows the full
numbers for one attribute, including a Mann-Whitney significance check. A dumbbell was
chosen over single bars and box plots because it puts the gap itself on the page.

CORRELATE tab: "Does a well-liked ad travel further online?" A scatter plot, one dot per ad:
x = approval percentile, y = YouTube views (log scale, because views span hundreds to
hundreds of millions). Spearman rho summarises it: near 0 means approval and views are not
related, toward +1 means better-liked ads also get more views.

RANK tab: "Which ads got the biggest online attention spike?" Horizontal bars of the ads with
the largest Wikipedia pageview spikes (a spike of 10 means 10x normal pageviews), sorted so the
biggest are on top. A slider picks how many ads to show (5 to 20).

LOCATE / OVERVIEW tab: "Which years does each data source cover?" Grouped bars per year: how
many ads have YouTube, Ad Meter, and Wikipedia data. Its point is to make coverage gaps visible.

PHASE 2 - ATTRIBUTE COMBO EXPLORER tab: the user builds a concept by picking several creative
attributes; it finds past ads with exactly that combination and shows how many match, their
average approval and average YouTube views versus the overall averages, and lists the matching
ads. Few matches means the result is only a directional signal.

LABELS: Use = approval is significantly higher with the attribute; Avoid = significantly lower;
Unproven = not enough evidence (too few ads, or the difference could be chance). The label is
computed by code and must never be changed.
"""


def build_page_context(fdf, df, stats_rows, year_range, brand_pick, active_attrs, active_sources):
    """Live numbers for the current sidebar filters, as plain text the assistant may cite."""
    from itertools import combinations

    out = ["LIVE DATA FOR THE CURRENT VIEW (the only numbers you may cite):", ""]
    filt = f"years {year_range[0]}-{year_range[1]}"
    filt += f"; brands: {', '.join(brand_pick)}" if brand_pick else "; all brands"
    if active_attrs:
        filt += f"; only ads with: {', '.join(active_attrs)}"
    if active_sources:
        filt += f"; only ads with: {', '.join(active_sources)}"
    out.append(f"Filters: {filt}.")
    out.append(
        f"Key metrics: {len(fdf)} ads in view; {int(fdf['has_youtube'].sum())} with YouTube stats; "
        f"{int(fdf['has_admeter'].sum())} with Ad Meter score; "
        f"{int(fdf['has_wiki_attention'].sum())} with Wikipedia attention data."
    )

    # Compare (and Creative Brief)
    out += ["", "COMPARE (also the basis for creative briefs) - approval with vs. without each attribute:",
            "| Attribute | n(with) | n(without) | Approval w/ | Approval w/o | p-value | Label | Avg YouTube views (w/) | Avg Wiki spike (w/) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in stats_rows:
        ayv = f"{r['avg_youtube_views']:,}" if r["avg_youtube_views"] is not None else "n/a"
        out.append(
            f"| {r['attribute']} | {r['n_with']} | {r['n_without']} | "
            f"{r['approval_with_pct']}% | {r['approval_without_pct']}% | {r['p_value']} | "
            f"{r['label']} | {ayv} | {r['avg_wiki_attention_spike']} |"
        )

    # Correlate
    both = fdf[(fdf["has_youtube"] == True) & (fdf["has_admeter"] == True)].dropna(
        subset=["ad_meter_percentile", "view_count"])
    out += ["", "CORRELATE - approval percentile vs. YouTube views:"]
    if len(both) >= 5:
        rho, _ = stats.spearmanr(both["ad_meter_percentile"], both["view_count"])
        out.append(f"Spearman rho = {rho:.2f} across {len(both)} ads that have both an Ad Meter score and YouTube views.")
    else:
        out.append("Not enough ads with both an Ad Meter score and YouTube views in this view.")

    # Rank
    wk = fdf[fdf["has_wiki_attention"] == True].copy()
    wk["label"] = wk["brand"] + " " + wk["year"].astype(str)
    wk = wk.drop_duplicates(subset=["brand", "year"])
    top = wk.nlargest(10, "wiki_attention_spike")
    out += ["", "RANK - ads with the biggest Wikipedia attention spike (x normal pageviews), top 10:"]
    if len(top):
        for _, rw in top.iterrows():
            out.append(f"- {rw['label']}: {round(float(rw['wiki_attention_spike']), 2)}x")
    else:
        out.append("No ads with Wikipedia attention data in this view.")

    # Locate
    out += ["", "LOCATE / OVERVIEW - number of ads with each data source, by year:"]
    for s, lbl in (("has_youtube", "YouTube"), ("has_admeter", "Ad Meter"), ("has_wiki_attention", "Wikipedia")):
        yrs = sorted(fdf.loc[fdf[s] == True, "year"].unique())
        if len(yrs):
            out.append(f"{lbl}: {int(fdf[s].sum())} ads in total, covering years {int(yrs[0])} to {int(yrs[-1])}.")
        else:
            out.append(f"{lbl}: no ads in this view.")
    for y in sorted(fdf["year"].unique()):
        sub = fdf[fdf["year"] == y]
        out.append(
            f"- {int(y)}: YouTube {int(sub['has_youtube'].sum())}, Ad Meter {int(sub['has_admeter'].sum())}, "
            f"Wikipedia {int(sub['has_wiki_attention'].sum())}"
        )

    # Phase 2 (always full history, ignores sidebar filters)
    ov_app = df.loc[df["has_admeter"] == True, "ad_meter_percentile"].mean()
    ov_views = df.loc[df["has_youtube"] == True, "view_count"].mean()
    out += ["", "PHASE 2 - ATTRIBUTE COMBO EXPLORER (full history, ignores sidebar filters):",
            f"Overall average approval {ov_app * 100:.1f}%; overall average YouTube views {int(ov_views):,}.",
            "Most common 2-attribute combinations (5+ ads):"]
    pairs = []
    for a, b in combinations(ATTRS, 2):
        sub = df[(df[a] == True) & (df[b] == True)]
        if len(sub) >= 5:
            ap = sub.loc[sub["has_admeter"] == True, "ad_meter_percentile"]
            vw = sub.loc[sub["has_youtube"] == True, "view_count"]
            pairs.append((len(sub), a, b, ap, vw))
    for n, a, b, ap, vw in sorted(pairs, key=lambda t: -t[0])[:12]:
        appr = f"{ap.mean() * 100:.1f}%" if len(ap) else "n/a"
        views = f"{int(vw.mean()):,}" if len(vw) else "n/a"
        out.append(f"- {ATTR_LABELS[a]} + {ATTR_LABELS[b]}: {n} ads; avg approval {appr}; avg YouTube views {views}")
    return "\n".join(out)


def build_home_instruction(context_text):
    return f"""You are the dashboard guide for Account E's Super Bowl ad strategy dashboard. You
answer questions about ANY page of it: Compare, Correlate, Rank, Locate / Overview, and Phase 2 -
what a chart shows, how to read it, why it was chosen, and what the current numbers say. You also
write short, evidence-backed creative briefs for next year's pitch when asked, using the
Use / Avoid / Unproven labels.

{PAGE_GUIDE}
{context_text}

RULES
- Every number you state about this data must appear exactly in LIVE DATA above. Never invent,
  estimate, or compute new numbers (no differences, ratios, or averages of your own - quote the
  two numbers instead). Write numbers exactly as shown, large ones in full with commas (no "K"/"M").
- The Label (Use / Avoid / Unproven) in the table is final. Never relabel an attribute. Do not state
  the p-value number itself; use it only internally to judge confidence.
- You may use general knowledge to explain how to read a chart or why a pattern might exist, but say
  clearly when you are doing that, and never imply it is something Account E's data shows.
- If the data above can't answer the question, say so plainly. If the question has nothing to do with
  this dashboard, say you can only help with it.
- Style: this box is a narrow sidebar. Be conversational and brief - usually 2 to 4 short sentences,
  plain paragraphs, no headings or tables. Go longer only if the user asks for a full brief or a
  detailed walkthrough.
- End EVERY reply with one final line in exactly this form, naming the tab(s) your answer draws on
  (choose from: Compare, Correlate, Rank, Locate / Overview, Phase 2, Overview):
  [[tabs: Compare]]"""


def validate_against_context(response_text, context_text):
    allowed = []
    for tok in re.findall(r"\d[\d,]*\.?\d*", context_text):
        try:
            allowed.append(float(tok.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return _flag_numbers(response_text, allowed)


_TABS_MARKER = re.compile(r"\[\[\s*tabs?\s*:\s*(.*?)\]\]", re.I | re.S)


def split_tabs_marker(text):
    """Pulls the trailing [[tabs: ...]] marker off a reply -> (clean_text, tabs or None)."""
    m = _TABS_MARKER.search(text)
    if not m:
        return re.sub(r"\[\[.*$", "", text, flags=re.S).strip(), None
    return (text[:m.start()] + text[m.end():]).strip(), m.group(1).strip()


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
    /* Homepage assistant in the sidebar: white chat cards with dark text on the red pane */
    [data-testid="stSidebar"] [data-testid="stChatMessage"] {
        background: #ffffff; border-radius: 10px; padding: 8px 10px;
    }
    [data-testid="stSidebar"] [data-testid="stChatMessage"] * { color: #1f1f1f !important; }
    [data-testid="stSidebar"] [data-testid="stChatMessage"] p { font-size: 0.85rem; }
    [data-testid="stSidebar"] [data-testid="stChatInput"] { background: #ffffff; border-radius: 10px; }
    [data-testid="stSidebar"] [data-testid="stChatInput"] textarea { color: #1f1f1f !important; }
    [data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.45);
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
# Homepage assistant: "Ask the Dashboard" (sidebar, answers about every page)
# ---------------------------------------------------------------------------
_HOME_SUGGESTIONS = ["Explain the dumbbell chart", "What should we avoid?", "What is Phase 2 about?"]


def _ask_home(question):
    st.session_state["home_pending"] = question


with st.sidebar:
    st.markdown("---")
    st.markdown("### 💬 Ask the Dashboard")
    st.caption("Ask about any page. Answers use only this dashboard's numbers for your current filters.")

    if not _GENAI_AVAILABLE:
        st.warning("The Gemini SDK isn't installed here. Run `pip install google-genai python-dotenv`.")
    elif not GEMINI_API_KEY:
        st.warning("No GEMINI_API_KEY found, so the chat is switched off.")
    else:
        if "home_messages" not in st.session_state:
            st.session_state.home_messages = []

        for _i, _q in enumerate(_HOME_SUGGESTIONS):
            st.button(_q, key=f"home_sugg_{_i}", on_click=_ask_home, args=(_q,), width="stretch")

        _home_box = st.container()
        _home_prompt = st.chat_input("Ask anything about this dashboard...", key="home_chat_input")
        if not _home_prompt:
            _home_prompt = st.session_state.pop("home_pending", None)

        with _home_box:
            for _m in st.session_state.home_messages:
                with st.chat_message(_m["role"]):
                    st.markdown(_m["content"])
                    if _m.get("flag"):
                        st.caption(_m["flag"])

            if _home_prompt:
                st.session_state.home_messages.append({"role": "user", "content": _home_prompt})
                with st.chat_message("user"):
                    st.markdown(_home_prompt)

                _active_attrs = [ATTR_LABELS[a] for a, on in attr_filter.items() if on]
                _active_sources = [lbl for lbl, on in (("YouTube stats", req_youtube),
                                                       ("an Ad Meter score", req_admeter),
                                                       ("Wikipedia attention data", req_wiki)) if on]
                _ctx = build_page_context(fdf, df, compute_attribute_stats(fdf), year_range,
                                          brand_pick, _active_attrs, _active_sources)

                _contents = [
                    types.Content(
                        role="model" if _m["role"] == "assistant" else "user",
                        parts=[types.Part.from_text(text=_m["content"])],
                    )
                    for _m in st.session_state.home_messages[-12:]
                ]

                with st.chat_message("assistant"):
                    _ph = st.empty()
                    _raw = ""
                    _failed = False
                    try:
                        _client = genai.Client(api_key=GEMINI_API_KEY)
                        for _chunk in stream_gemini(_client, _contents, build_home_instruction(_ctx)):
                            _raw += _chunk.text or ""
                            _ph.markdown(split_tabs_marker(_raw)[0])
                    except Exception as _e:
                        _failed = True
                        _raw = f"Sorry, the request to Gemini failed: {_e}"

                    _clean, _tabs = split_tabs_marker(_raw)
                    _ph.markdown(_clean)
                    _flag = None
                    if not _failed:
                        _bad = validate_against_context(_clean, _ctx)
                        _where = f"📍 {_tabs}  " if _tabs else ""
                        if _bad:
                            _flag = f"{_where}⚠️ Not found in dashboard data: {', '.join(_bad[:6])} - verify."
                        else:
                            _flag = f"{_where}✅ numbers verified"
                        st.caption(_flag)

                st.session_state.home_messages.append({"role": "assistant", "content": _clean, "flag": _flag})

        if st.session_state.home_messages and st.button("Clear chat", key="home_clear"):
            st.session_state.home_messages = []
            st.rerun()

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
    ["📊 Compare", "📈 Correlate", "🏆 Rank", "🗺️ Locate / Overview",
     "🧩 Phase 2: Attribute Combo Explorer"]
)

# ---------------------------------------------------------------------------
# TAB 1 - COMPARE: does a given attribute move Ad Meter approval?
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Does a creative attribute change live-audience approval?")
    st.caption(
        "Task: COMPARE. Marks: points + connecting line (a dumbbell), one row per attribute. "
        "Channels: position/length (mean approval) x color (with / without)."
    )

    # One row per attribute: orange dot = ads WITHOUT it, blue dot = ads WITH it, the line is the gap.
    cmp_rows = [
        r for r in compute_attribute_stats(fdf)
        if r["n_with"] >= 3 and r["n_without"] >= 3
        and r["approval_with_pct"] is not None and r["approval_without_pct"] is not None
    ]
    if not cmp_rows:
        st.info("Not enough Ad Meter-scored ads under this filter to compare reliably.")
    else:
        # biggest boost at the top, biggest drop at the bottom
        cmp_rows.sort(key=lambda r: r["approval_with_pct"] - r["approval_without_pct"], reverse=True)
        ylabels = [f"{r['attribute']}  ·  {r['label']}" for r in cmp_rows]
        fig = go.Figure()
        for r, yl in zip(cmp_rows, ylabels):
            fig.add_trace(go.Scatter(
                x=[r["approval_without_pct"], r["approval_with_pct"]], y=[yl, yl],
                mode="lines", line=dict(color=GRID, width=4), showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[r["approval_without_pct"] for r in cmp_rows], y=ylabels, mode="markers",
            marker=dict(color=ORANGE, size=15), name="Without the attribute",
            customdata=[[r["n_without"]] for r in cmp_rows],
            hovertemplate="<b>%{y}</b><br>Without: %{x:.1f}% avg approval (n=%{customdata[0]})<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[r["approval_with_pct"] for r in cmp_rows], y=ylabels, mode="markers",
            marker=dict(color=BLUE, size=15), name="With the attribute",
            customdata=[[r["n_with"]] for r in cmp_rows],
            hovertemplate="<b>%{y}</b><br>With: %{x:.1f}% avg approval (n=%{customdata[0]})<extra></extra>",
        ))
        lo = min(min(r["approval_with_pct"], r["approval_without_pct"]) for r in cmp_rows)
        hi = max(max(r["approval_with_pct"], r["approval_without_pct"]) for r in cmp_rows)
        fig.update_layout(
            yaxis=dict(categoryorder="array", categoryarray=ylabels[::-1], gridcolor=GRID, automargin=True),
            xaxis=dict(title="Mean Ad Meter approval percentile (%)", ticksuffix="%",
                       range=[max(0, lo - 5), min(100, hi + 5)], gridcolor=GRID),
            plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
            height=120 + 52 * len(cmp_rows), margin=dict(t=20, b=40),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Each row is one creative attribute; the tag after the name is its verdict "
            "(Use / Avoid / Unproven). Hover a dot for the average and the number of ads behind it."
        )

    attr_choice = st.selectbox(
        "See the full numbers for one attribute", ATTRS,
        format_func=lambda a: ATTR_LABELS[a], key="compare_attr",
    )

    am = fdf[fdf["has_admeter"] == True]
    with_grp = am[am[attr_choice] == True]["ad_meter_percentile"].dropna()
    without_grp = am[am[attr_choice] == False]["ad_meter_percentile"].dropna()

    if len(with_grp) < 3 or len(without_grp) < 3:
        st.info("Not enough Ad Meter-scored ads with this attribute (and this filter) to compare reliably.")
    else:
        u_stat, p_val = stats.mannwhitneyu(with_grp, without_grp, alternative="two-sided")
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
