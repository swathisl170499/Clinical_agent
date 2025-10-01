# clinical_agent/src/data/generate_data.py
import re
import random
import time
from typing import List, Dict, Tuple, Optional
import pandas as pd
from faker import Faker
from tqdm import tqdm
import os
from ingestion.fetch_docs import fetch_condition_snippets
from ingestion.generate_notes import generate_patient_notes
from ingestion.web_search import fetch_web_clinical_docs
from embeddings.embed_store import FAISSEmbedder
from llms.codestral_llm import CodeStralLLM


# -------- Config --------
NUM_RECORDS = 200
BATCH_SIZE = 10
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_CSV = os.path.join(PROJECT_ROOT, "clinical_data.csv")

PROJECT_ID = "clinical-copilot"   # <-- your GCP project (fixed)
REGION = "us-central1"
MODEL_NAME = "mistralai/codestral-2501@001"
TEMP = 0.4
MAX_TOKENS = 260
USE_CODESTRAL = os.getenv("USE_CODESTRAL", "1") == "1"   # toggle for offline demo

fake = Faker()

CONDITIONS = [
    "Hypertension", "Diabetes", "Asthma", "Heart Failure",
    "Chronic Kidney Disease", "Cancer", "COPD", "COVID-19",
    "Migraine", "Arthritis"`
]
US_STATES = [
    "California", "New York", "Texas", "Florida", "Michigan", "Ohio", "Illinois",
    "Georgia", "North Carolina", "Pennsylvania", "Washington", "Massachusetts"
]
GENDERS = ["Male", "Female", "Other"]

# Meds per condition (for extraction/fallback; NOT included verbatim in prompts)
MEDS: Dict[str, List[str]] = {
    "Hypertension": ["Lisinopril", "Losartan", "Amlodipine", "Hydrochlorothiazide", "Metoprolol"],
    "Diabetes": ["Metformin", "Insulin Glargine", "Glipizide", "Dapagliflozin", "Sitagliptin"],
    "Asthma": ["Albuterol", "Fluticasone", "Budesonide", "Montelukast", "Salmeterol"],
    "Heart Failure": ["Furosemide", "Spironolactone", "Carvedilol", "Sacubitril/Valsartan"],
    "Chronic Kidney Disease": ["Erythropoietin", "Calcitriol", "Sevelamer", "Furosemide"],
    "Cancer": ["Paclitaxel", "Cisplatin", "Doxorubicin", "Cyclophosphamide", "Pembrolizumab"],
    "COPD": ["Tiotropium", "Formoterol", "Budesonide", "Theophylline"],
    "COVID-19": ["Remdesivir", "Dexamethasone", "Tocilizumab", "Molnupiravir"],
    "Migraine": ["Sumatriptan", "Topiramate", "Propranolol", "Amitriptyline"],
    "Arthritis": ["Ibuprofen", "Methotrexate", "Etanercept", "Celecoxib"]
}
CLASS_HINTS = {
    "Hypertension": {
        "Lisinopril": "ACE inhibitor",
        "Losartan": "ARB",
        "Amlodipine": "Calcium channel blocker",
        "Hydrochlorothiazide": "Thiazide diuretic",
        "Metoprolol": "Beta blocker",
    },
    "Diabetes": {
        "Metformin": "biguanide",
        "Insulin Glargine": "basal insulin",
        "Glipizide": "sulfonylurea",
        "Dapagliflozin": "SGLT2 inhibitor",
        "Sitagliptin": "DPP-4 inhibitor",
    },
    "Asthma": {
        "Albuterol": "short-acting beta-agonist (SABA)",
        "Fluticasone": "inhaled corticosteroid (ICS)",
        "Budesonide": "inhaled corticosteroid (ICS)",
        "Montelukast": "leukotriene receptor antagonist",
        "Salmeterol": "long-acting beta-agonist (LABA)",
    },
    "Heart Failure": {
        "Furosemide": "loop diuretic",
        "Spironolactone": "mineralocorticoid receptor antagonist (MRA)",
        "Carvedilol": "beta blocker",
        "Sacubitril/Valsartan": "ARNI",
    },
    "Chronic Kidney Disease": {
        "Erythropoietin": "erythropoiesis-stimulating agent",
        "Calcitriol": "vitamin D analog",
        "Sevelamer": "phosphate binder",
        "Furosemide": "loop diuretic",
    },
    "Cancer": {
        "Paclitaxel": "taxane chemotherapy",
        "Cisplatin": "platinum chemotherapy",
        "Doxorubicin": "anthracycline chemotherapy",
        "Cyclophosphamide": "alkylating agent",
        "Pembrolizumab": "PD-1 inhibitor (immunotherapy)",
    },
    "COPD": {
        "Tiotropium": "long-acting muscarinic antagonist (LAMA)",
        "Formoterol": "long-acting beta-agonist (LABA)",
        "Budesonide": "inhaled corticosteroid (ICS)",
        "Theophylline": "methylxanthine",
    },
    "COVID-19": {
        "Remdesivir": "antiviral",
        "Dexamethasone": "corticosteroid",
        "Tocilizumab": "IL-6 inhibitor",
        "Molnupiravir": "antiviral",
    },
    "Migraine": {
        "Sumatriptan": "triptan",
        "Topiramate": "antiepileptic (prophylaxis)",
        "Propranolol": "beta blocker (prophylaxis)",
        "Amitriptyline": "TCA (prophylaxis)",
    },
    "Arthritis": {
        "Ibuprofen": "NSAID",
        "Methotrexate": "csDMARD",
        "Etanercept": "biologic DMARD (TNF inhibitor)",
        "Celecoxib": "COX-2 selective NSAID",
    },
}

TAGS = {
    "Hypertension": ["tachycardia", "ankle_edema", "dry_cough", "prediabetes", "overweight", "fatigue"],
    "Diabetes": ["hypoglycemia_risk", "overweight", "renal_concern", "high_a1c"],
    "Asthma": ["nocturnal_symptoms", "exercise_induced", "frequent_rescue"],
    "Heart Failure": ["reduced_ef", "edema", "orthopnea"],
    "Chronic Kidney Disease": ["hyperkalemia_risk", "phosphate_up", "anemia"],
    "Cancer": ["neuropathy", "nausea", "cytopenias"],
    "COPD": ["frequent_exacerbations", "sputum", "exercise_limitation"],
    "COVID-19": ["hypoxia", "fever", "fatigue"],
    "Migraine": ["aura", "frequent_attacks", "sleep_disruption"],
    "Arthritis": ["gi_risk", "stiffness", "multi_joint"]
}

def _vitals(cond: str) -> str:
    if cond == "Hypertension":
        sbp = random.randint(130, 160); dbp = random.randint(78, 96)
    elif cond in ("Asthma", "COPD"):
        sbp = random.randint(110, 135); dbp = random.randint(68, 86)
    elif cond in ("Heart Failure", "Chronic Kidney Disease"):
        sbp = random.randint(115, 145); dbp = random.randint(70, 90)
    else:
        sbp = random.randint(110, 140); dbp = random.randint(68, 92)
    hr = random.randint(60, 96)
    rr = random.randint(12, 22)
    temp = round(random.uniform(36.4, 37.6), 1)
    spo2 = random.randint(93, 99)
    return f"BP {sbp}/{dbp} mmHg, HR {hr} bpm, RR {rr}/min, Temp {temp}°C, SpO2 {spo2}%"

def _least_used(drugs: List[str], counter: Dict[str, int]) -> str:
    return sorted(drugs, key=lambda d: counter.get(d, 0))[0]

def _choose_med(cond: str, age: int, tags: List[str], cov: Dict[str, int]) -> Tuple[str, str]:
    drugs = MEDS[cond]; classes = CLASS_HINTS[cond]
    # Some sensible rules to diversify
    if cond == "Hypertension":
        if "tachycardia" in tags: return "Metoprolol", classes["Metoprolol"]
        if "ankle_edema" in tags:
            pool = ["Lisinopril", "Losartan", "Hydrochlorothiazide"]
            pick = _least_used(pool, cov); return pick, classes[pick]
        if "dry_cough" in tags:
            pool = ["Losartan", "Amlodipine", "Hydrochlorothiazide"]
            pick = _least_used(pool, cov); return pick, classes[pick]
    if cond == "Diabetes":
        if "high_a1c" in tags:
            pool = ["Insulin Glargine", "Metformin"]; pick = _least_used(pool, cov); return pick, classes[pick]
        if "renal_concern" in tags:
            pool = ["Dapagliflozin", "Insulin Glargine"]; pick = _least_used(pool, cov); return pick, classes[pick]
    if cond == "Asthma":
        if "frequent_rescue" in tags or "nocturnal_symptoms" in tags:
            pool = ["Fluticasone", "Budesonide"]; pick = _least_used(pool, cov); return pick, classes[pick]
    if cond == "Heart Failure":
        if "edema" in tags: return "Furosemide", classes["Furosemide"]
    if cond == "Chronic Kidney Disease":
        if "anemia" in tags: return "Erythropoietin", classes["Erythropoietin"]
    # default least used
    pick = _least_used(drugs, cov); return pick, classes[pick]

def _compile_med_regex(cond: str) -> re.Pattern:
    names = MEDS.get(cond, [])
    if not names:
        return re.compile(r"\b([A-Z][a-zA-Z]+(?:/[A-Z][a-zA-Z]+)?)\s+\d+\s?(mg|mcg|units)\b", re.I)
    choices = "|".join(map(re.escape, names))
    return re.compile(rf"\b(?:{choices})\b", re.I)

def _extract_med(note: str, cond: str) -> Tuple[str, str]:
    med_re = _compile_med_regex(cond)
    dose_re = re.compile(r"\b(\d+\s?(?:mg|mcg|units))\b", re.I)
    freq_re = re.compile(r"\b(once daily|twice daily|bid|tid|qhs|daily|every morning|every evening|weekly|prn)\b", re.I)
    m = med_re.search(note or "")
    if not m: return "", ""
    med = m.group(0)
    # try to find dose/freq in same sentence
    for s in re.split(r"(?<=[.!?])\s+", note):
        if med.lower() in s.lower():
            d = dose_re.search(s); f = freq_re.search(s)
            return med, " ".join(x for x in [(d.group(0) if d else ""), (f.group(0) if f else "")] if x).strip()
    return med, ""

def _fallback_note(cond: str, state: str, labs: float, drug: str) -> str:
    vit = _vitals(cond)
    dose = random.choice(["5 mg", "10 mg", "25 mg", "50 mg"])
    freq = random.choice(["once daily", "twice daily", "qHS", "in the morning"])
    return (
        f"Adult patient in {state} with {cond} for routine follow-up. "
        f"Objective exam: {vit}. Key lab {labs} reviewed. "
        f"Assessment: {cond}, stable with partial response. Plan: {drug} {dose} {freq}, "
        f"reinforce lifestyle changes and home monitoring; follow up in 4–8 weeks."
    )

def _doctor_prompt(cond: str, state: str, labs: float, class_hint: Optional[str], vit: str, ctg_snippet: str) -> str:
    class_txt = (
        f"For pharmacotherapy, select a real medication from the {class_hint} class and include dose/frequency. "
        if class_hint else
        "Choose a real guideline-concordant medication and include dose/frequency. "
    )
    return (
        "Write a concise DOCTOR note in natural prose (no bullet points; no AI/meta language). "
        "Use a SOAP-style flow and vary wording.\n"
        f"Patient: adult in {state} with {cond}. Objective vitals (copy verbatim): {vit}. "
        f"Key lab value: {labs}. "
        f"ClinicalTrials.gov snippet (context, do not quote verbatim): {ctg_snippet} "
        f"{class_txt}"
        "Include symptoms, adherence, exam highlights, treatment response, explicit drug with dose/frequency, "
        "brief lifestyle advice, and follow-up timing. Keep it 4–7 sentences."
    )

def _base_rows(n: int) -> pd.DataFrame:
    rows = []
    coverage = {c: {m: 0 for m in MEDS[c]} for c in CONDITIONS}
    for _ in range(n):
        cond = random.choice(CONDITIONS)
        age = random.randint(18, 85)
        state = random.choice(US_STATES)
        labs = round(random.uniform(3.5, 9.8), 2)
        tags = random.sample(TAGS.get(cond, []), k=random.randint(0, min(2, len(TAGS.get(cond, [])))))
        med, klass = _choose_med(cond, age, tags, coverage[cond])
        coverage[cond][med] += 1
        rows.append({
            "patient_id": fake.uuid4(),
            "name": fake.name(),
            "age": age,
            "gender": random.choice(GENDERS),
            "condition": cond,
            "enrollment_date": fake.date_between(start_date='-2y', end_date='today'),
            "site_location": state,
            "lab_results": labs,
            "tags": ",".join(tags),
            "planned_med": med,             # internal helpers (not final output)
            "planned_class": klass,
            "medications": "",
            "visit_notes": ""
        })
    return pd.DataFrame(rows)

def generate():
    # fetch web snippets to drive variety
    snippets = fetch_condition_snippets(CONDITIONS, per_condition=5)
    llm = CodeStralClient(PROJECT_ID, REGION, MODEL_NAME, TEMP, MAX_TOKENS) if USE_CODESTRAL else None

    df = _base_rows(NUM_RECORDS)
    notes: List[str] = []
    meds: List[str] = []

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Generating notes"):
        batch = df.iloc[start:start+BATCH_SIZE]
        prompts = []
        ctg_contexts = []
        for _, row in batch.iterrows():
            cond = row["condition"]
            vit = _vitals(cond)
            ctg_snippet = random.choice(snippets.get(cond, [""])) if snippets.get(cond) else ""
            ctg_contexts.append(ctg_snippet)
            prompts.append(_doctor_prompt(cond, row["site_location"], row["lab_results"], row["planned_class"], vit, ctg_snippet))

        outputs: List[str] = []
        if llm:
            try:
                outputs = llm.generate(prompts)
            except Exception as e:
                print(f"⚠️ CodeStral error: {e} — using fallbacks for this batch.")
                outputs = []

        # if no outputs, fallback handcrafted notes
        if not outputs:
            outputs = []
            for _, row in batch.iterrows():
                outputs.append(_fallback_note(row["condition"], row["site_location"], row["lab_results"], row["planned_med"]))

        # extract/ensure medication text
        for (idx, out_text), (_, row) in zip(zip(batch.index, outputs), batch.iterrows()):
            cond = row["condition"]
            med, dosefreq = _extract_med(out_text, cond)
            if not med:
                # enforce planned med if not recognized
                add_dose = random.choice(["5 mg", "10 mg", "25 mg", "50 mg"])
                add_freq = random.choice(["once daily", "twice daily", "qHS", "in the morning"])
                if not out_text.strip().endswith("."):
                    out_text += "."
                out_text += f" Prescription issued: {row['planned_med']} {add_dose} {add_freq}."
                med = row["planned_med"]
                dosefreq = f"{add_dose} {add_freq}"
            df.at[idx, "visit_notes"] = out_text.strip()
            df.at[idx, "medications"] = f"{med} {dosefreq}".strip()

    # persist without helper columns
    out = df.drop(columns=["planned_med", "planned_class"], errors="ignore")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Saved {len(out)} records to {OUT_CSV}")

    # auto-build hybrid index so RAG is hot
    try:
        print("🔁 Building hybrid index …")
        from clinical_agent.src.embeddings.hybrid_index import HybridIndex
        h = HybridIndex()
        h.build()   # reads clinical_data.csv + builds FAISS+BM25
        print("✅ Hybrid index built.")
    except Exception as e:
        print(f"⚠️ Could not build hybrid index now: {e}. You can run it later manually.")

if __name__ == "__main__":
    generate()
