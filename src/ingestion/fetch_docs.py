# clinical_agent/src/ingestion/fetch_docs.py
import requests
from typing import Dict, List
from collections import defaultdict

CTG_ENDPOINT = "https://clinicaltrials.gov/api/query/full_studies"

# lightweight, robust fetcher; keeps short text snippets usable in prompts
def fetch_condition_snippets(conditions: List[str], per_condition: int = 5, timeout: int = 12) -> Dict[str, List[str]]:
    out = defaultdict(list)
    for cond in conditions:
        try:
            params = {
                "expr": cond,
                "min_rnk": 1,
                "max_rnk": max(3, per_condition * 2),
                "fmt": "json",
            }
            r = requests.get(CTG_ENDPOINT, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            studies = data.get("FullStudiesResponse", {}).get("FullStudies", []) or []
            for s in studies:
                desc = (
                    s.get("Study", {})
                     .get("ProtocolSection", {})
                     .get("DescriptionModule", {})
                     .get("BriefSummary", "")
                ) or ""
                if not desc:
                    continue
                cleaned = " ".join(desc.split())
                if 120 <= len(cleaned) <= 800:
                    out[cond].append(cleaned)
                if len(out[cond]) >= per_condition:
                    break
        except Exception:
            # ignore networking hiccups, continue
            pass
    return dict(out)
