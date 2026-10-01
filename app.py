import io
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

BASE = "https://api.apollo.io/api/v1"
CACHE_FILE = "apollo_cache.json"

FINANCE_TITLES = [
    "Head of Finance Operations", "Director of Finance Operations", "VP Finance Operations",
    "Financial Operations", "Controller", "VP Finance", "Head of Finance",
    "Director of Finance", "Revenue Operations", "Billing", "Chief Financial Officer", "CFO",
]
TITLE_SCORES = [
    ("finance operations", 100), ("financial operations", 100),
    ("controller", 90), ("billing", 85), ("revenue operations", 80),
    ("vp finance", 75), ("vp, finance", 75), ("head of finance", 75),
    ("director of finance", 70), ("cfo", 60), ("chief financial", 60),
    ("accounting", 50), ("finance", 40),
]


def get_api_key() -> str | None:
    try:
        return st.secrets["APOLLO_API_KEY"]
    except Exception:
        return os.environ.get("APOLLO_API_KEY")


def score_title(title):
    t = (title or "").lower()
    for keyword, score in TITLE_SCORES:
        if keyword in t:
            return score
    return 0


def clean_domain(value):
    if pd.isna(value):
        return None
    d = str(value).strip().lower()
    for prefix in ("https://", "http://", "www."):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.split("/")[0] or None


def find_best_contact(domain, headers):
    payload = {
        "q_organization_domains_list": [domain],
        "person_titles": FINANCE_TITLES,
        "include_similar_titles": True,
        "person_seniorities": ["c_suite", "vp", "head", "director", "manager"],
        "per_page": 25,
        "page": 1,
    }
    r = requests.post(f"{BASE}/mixed_people/api_search", json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    people = [p for p in r.json().get("people", []) if score_title(p.get("title")) > 0]
    if not people:
        return None
    best = max(people, key=lambda p: score_title(p.get("title")))

    r = requests.post(
        f"{BASE}/people/match",
        json={"id": best["id"], "reveal_personal_emails": False},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    p = r.json().get("person") or {}
    return {
        "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
        "title": p.get("title"),
        "email": p.get("email"),
        "email_status": p.get("email_status"),
        "linkedin": p.get("linkedin_url"),
    }


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def status_of(cache, d):
    entry = cache.get(d)
    if entry is None:
        return "pending"
    if "error" in entry:
        return "error"
    if entry.get("not_found"):
        return "not_found"
    return "found"


st.set_page_config(page_title="Finance Contact Finder", page_icon="📇", layout="wide")
st.title("📇 Finance Contact Finder")
st.caption("Upload a company list, find one finance-operations contact per company via Apollo.")

api_key = get_api_key()
if not api_key:
    st.error("Set APOLLO_API_KEY as an environment variable or in .streamlit/secrets.toml.")
    st.stop()
headers = {"X-Api-Key": api_key, "Content-Type": "application/json", "Cache-Control": "no-cache"}

with st.sidebar:
    st.header("Settings")
    max_credits = st.number_input("Max credits this run", 1, 5000, 10, help="Start small, then raise.")
    delay = st.slider("Delay between calls (s)", 0.0, 3.0, 1.0, 0.5)
    if st.button("Clear cache"):
        save_cache({})
        st.success("Cache cleared.")

file = st.file_uploader("Excel file with company domains", type=["xlsx"])
if not file:
    st.stop()

df = pd.read_excel(file)
col = st.selectbox("Which column holds the domain?", df.columns)
df["_domain"] = df[col].map(clean_domain)
domains = list(dict.fromkeys(df["_domain"].dropna()))

cache = load_cache()
todo = [d for d in domains if d not in cache]
c1, c2, c3 = st.columns(3)
c1.metric("Unique companies", len(domains))
c2.metric("Already cached (free)", len(domains) - len(todo))
c3.metric("To look up", len(todo))

if st.button("Run lookup", type="primary", disabled=not todo):
    bar = st.progress(0.0)
    status = st.empty()
    credits_used = 0
    for i, d in enumerate(todo):
        if credits_used >= max_credits:
            st.warning(f"Credit cap ({max_credits}) reached. Raise it and run again to continue.")
            break
        status.text(f"Looking up {d} ({i + 1}/{len(todo)})")
        try:
            result = find_best_contact(d, headers)
            cache[d] = result or {"not_found": True}
            if result:
                credits_used += 1
        except requests.HTTPError as e:
            code = e.response.status_code
            if code == 429:
                status.text("Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            cache[d] = {"error": f"HTTP {code}"}
        except Exception as e:
            cache[d] = {"error": str(e)}
        save_cache(cache)
        bar.progress((i + 1) / len(todo))
        time.sleep(delay)
    status.text(f"Done. Credits used this run: {credits_used}")

# Merge cache back onto the original rows
out = df.copy()
for k in ("name", "title", "email", "email_status", "linkedin"):
    out[f"contact_{k}"] = out["_domain"].map(lambda d, k=k: (cache.get(d) or {}).get(k))
out["lookup_status"] = out["_domain"].map(lambda d: status_of(cache, d))
out = out.drop(columns="_domain")

st.subheader("Results")
choice = st.multiselect("Filter by status", ["found", "not_found", "error", "pending"],
                        default=["found", "not_found", "error", "pending"])
st.dataframe(out[out["lookup_status"].isin(choice)], use_container_width=True)

buf = io.BytesIO()
out.to_excel(buf, index=False)
st.download_button("Download Excel", buf.getvalue(), "companies_enriched.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
