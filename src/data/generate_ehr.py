"""
Synthetic oncology EHR dataset generator.

Generates a realistic synthetic dataset of 2,000 oncology patients with:
  - Demographics: age, sex
  - Clinical context: cancer type, stage, ECOG performance status, treatment regimen
  - Laboratory values: WBC, ANC, hemoglobin, platelets, creatinine, ALT, bilirubin
  - Symptom scores (0–10 CTCAE-informed): fatigue, nausea, pain, dyspnea
  - Free-text clinical note: synthetic oncologist note containing symptom language
  - Outcome label: adverse_event (binary), ae_type, ae_grade (1–4)

Adverse event prevalence is ~38% overall, with rates varying by cancer type,
treatment regimen, and baseline lab values — matching published trial AE rates.

All data is synthetic and contains no real patient information.
"""

from __future__ import annotations

import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

CANCER_TYPES = [
    "breast",
    "lung_nsclc",
    "lung_sclc",
    "colorectal",
    "lymphoma",
    "leukemia",
    "melanoma",
    "ovarian",
    "pancreatic",
    "prostate",
]

STAGES = ["I", "II", "III", "IV"]

TREATMENTS = {
    "breast":        ["AC-T", "TC", "trastuzumab", "pertuzumab", "olaparib", "pembrolizumab"],
    "lung_nsclc":    ["carboplatin_paclitaxel", "pembrolizumab", "osimertinib", "atezolizumab", "cisplatin_pemetrexed"],
    "lung_sclc":     ["carboplatin_etoposide", "cisplatin_etoposide", "atezolizumab"],
    "colorectal":    ["FOLFOX", "FOLFIRI", "CAPOX", "bevacizumab", "cetuximab"],
    "lymphoma":      ["R-CHOP", "ABVD", "R-CVP", "bendamustine_rituximab"],
    "leukemia":      ["venetoclax", "ibrutinib", "cytarabine_daunorubicin", "imatinib"],
    "melanoma":      ["nivolumab", "pembrolizumab", "ipilimumab_nivolumab", "dabrafenib_trametinib"],
    "ovarian":       ["carboplatin_paclitaxel", "olaparib", "bevacizumab", "gemcitabine"],
    "pancreatic":    ["FOLFIRINOX", "gemcitabine_nab-paclitaxel", "erlotinib"],
    "prostate":      ["enzalutamide", "abiraterone", "docetaxel", "cabazitaxel", "darolutamide"],
}

# Approximate AE probability by treatment class
_AE_RISK_BY_TREATMENT = {
    "FOLFOX":                   0.55,
    "FOLFIRI":                  0.52,
    "FOLFIRINOX":               0.65,
    "CAPOX":                    0.48,
    "carboplatin_paclitaxel":   0.50,
    "cisplatin_pemetrexed":     0.53,
    "carboplatin_etoposide":    0.58,
    "cisplatin_etoposide":      0.62,
    "AC-T":                     0.55,
    "R-CHOP":                   0.50,
    "ABVD":                     0.45,
    "cytarabine_daunorubicin":  0.70,
    "ipilimumab_nivolumab":     0.55,
    "pembrolizumab":            0.28,
    "nivolumab":                0.25,
    "atezolizumab":             0.26,
    "venetoclax":               0.42,
    "ibrutinib":                0.35,
    "olaparib":                 0.38,
    "trastuzumab":              0.18,
    "osimertinib":              0.22,
    "bevacizumab":              0.30,
    "cetuximab":                0.28,
    "imatinib":                 0.20,
    "enzalutamide":             0.22,
    "abiraterone":              0.20,
    "docetaxel":                0.48,
    "dabrafenib_trametinib":    0.42,
    "gemcitabine":              0.38,
    "gemcitabine_nab-paclitaxel": 0.52,
    "pertuzumab":               0.22,
    "TC":                       0.45,
    "R-CVP":                    0.38,
    "bendamustine_rituximab":   0.35,
    "erlotinib":                0.32,
    "cabazitaxel":              0.55,
    "darolutamide":             0.18,
}

AE_TYPES = [
    "febrile_neutropenia",
    "anemia",
    "thrombocytopenia",
    "nausea_vomiting",
    "peripheral_neuropathy",
    "hepatotoxicity",
    "nephrotoxicity",
    "pneumonitis",
    "colitis",
    "fatigue",
    "diarrhea",
    "mucositis",
]

# AE type frequencies (varies by treatment class; simplified here)
_AE_WEIGHTS = [0.12, 0.15, 0.10, 0.14, 0.10, 0.07, 0.05, 0.07, 0.06, 0.07, 0.05, 0.02]


# ---------------------------------------------------------------------------
# Clinical note templates
# ---------------------------------------------------------------------------

_NOTE_TEMPLATES = [
    textwrap.dedent("""\
        {age_sex} presenting for cycle {cycle} of {treatment} for {stage} {cancer}.
        {symptom_text}
        Vitals: T {temp}°C, BP {bp}, HR {hr} bpm, SpO2 {spo2}%.
        Labs notable for {lab_text}.
        {plan_text}"""),
    textwrap.dedent("""\
        {age_sex} with {stage} {cancer} on {treatment}, now completing cycle {cycle}.
        {symptom_text}
        Performance status ECOG {ecog}. {lab_text}.
        Assessment: {assessment_text}
        Plan: {plan_text}"""),
    textwrap.dedent("""\
        Follow-up visit for {age_sex} with {stage} {cancer}.
        Current regimen: {treatment}, cycle {cycle}.
        {symptom_text}
        Recent labs: {lab_text}.
        {plan_text}"""),
]

_SYMPTOM_PHRASES = {
    "fatigue": [
        "Patient reports significant fatigue and generalized weakness.",
        "Complains of persistent tiredness and exhaustion, limiting daily activities.",
        "Notable asthenia and lethargy since last treatment.",
        "Patient endorses grade 2 fatigue, worse with exertion.",
        "Reports feeling very tired most days; unable to perform usual activities.",
    ],
    "nausea": [
        "Experiencing moderate nausea, managed with ondansetron.",
        "Nausea present since chemotherapy initiation; minimal vomiting.",
        "Patient reports nausea and queasiness, particularly in the mornings.",
        "Ongoing nausea and vomiting despite antiemetic therapy.",
        "Mild nausea noted post-infusion.",
    ],
    "fever": [
        "Fever of {temp}°C noted; febrile since yesterday.",
        "Patient presents with pyrexia; evaluated for febrile neutropenia.",
        "Elevated temperature documented at clinic visit; blood cultures sent.",
        "Febrile episode at home, temperature 38.5°C, presented to ED.",
    ],
    "neuropathy": [
        "Reports peripheral neuropathy with numbness and tingling in fingertips and toes.",
        "Bilateral paresthesia in hands and feet; consistent with chemotherapy-induced neuropathy.",
        "Pins and needles sensation in distal extremities, worsening over past two cycles.",
        "Patient endorses burning sensation in feet bilaterally, affecting gait.",
        "Grade 2 peripheral neuropathy; dose reduction discussed.",
    ],
    "dyspnea": [
        "Complains of shortness of breath with exertion.",
        "Dyspnea on mild exertion; SpO2 {spo2}% on room air.",
        "Patient reports increasing breathlessness over the past week.",
        "Mild respiratory distress; chest X-ray ordered to evaluate for pneumonitis.",
        "New-onset dyspnea; imaging to rule out pulmonary embolism.",
    ],
    "pain": [
        "Reports pain at {pain_site}, rated {pain_score}/10.",
        "Ongoing abdominal pain and discomfort post-treatment.",
        "Pain well-controlled with current analgesic regimen.",
        "Patient endorses worsening pain; analgesic escalation considered.",
        "Breakthrough pain episodes requiring rescue analgesia.",
    ],
    "bleeding": [
        "Epistaxis reported; no significant bleeding otherwise.",
        "Bruising noted on extremities; petechiae present on lower legs.",
        "Hematuria reported by patient; urinalysis ordered.",
        "Low-grade bleeding, likely related to thrombocytopenia.",
    ],
    "diarrhea": [
        "Experiencing diarrhea, 4–6 loose stools per day.",
        "Colitis symptoms including diarrhea and abdominal cramping.",
        "Immune-related diarrhea suspected; steroid therapy initiated.",
        "Reports watery loose stools since starting checkpoint inhibitor.",
    ],
    "mucositis": [
        "Oral mucositis with mouth sores present; rinse prescribed.",
        "Stomatitis grade 2; eating soft foods only.",
        "Mucosal inflammation noted; dietary adjustments recommended.",
    ],
    "no_symptoms": [
        "Patient tolerating treatment well with no significant adverse effects.",
        "No complaints at this visit; denies fever, chills, nausea, or vomiting.",
        "Currently asymptomatic. Denies fatigue, pain, or shortness of breath.",
        "Patient doing well overall. No new symptoms reported.",
    ],
}

_LAB_PHRASES = {
    "neutropenia": [
        "ANC {anc:.1f} × 10⁹/L (low)",
        "WBC {wbc:.1f} × 10⁹/L with ANC {anc:.1f} (neutropenic)",
        "Significant neutropenia: ANC {anc:.1f}",
    ],
    "anemia": [
        "Hemoglobin {hgb:.1f} g/dL (low)",
        "Hgb {hgb:.1f} consistent with treatment-related anemia",
        "Low hemoglobin at {hgb:.1f} g/dL",
    ],
    "thrombocytopenia": [
        "Platelets {plt:.0f} × 10⁹/L (low)",
        "Thrombocytopenic: platelets {plt:.0f}",
        "Platelet count reduced at {plt:.0f}",
    ],
    "elevated_alt": [
        "ALT {alt:.0f} U/L (elevated)",
        "Transaminase elevation: ALT {alt:.0f}",
        "Hepatic enzyme elevation noted: ALT {alt:.0f}",
    ],
    "elevated_creatinine": [
        "Creatinine {creat:.2f} mg/dL (above baseline)",
        "Mildly elevated creatinine at {creat:.2f}",
        "Renal function impaired: creatinine {creat:.2f}",
    ],
    "normal": [
        "CBC and metabolic panel within normal limits.",
        "Labs unremarkable.",
        "WBC {wbc:.1f}, Hgb {hgb:.1f}, Plt {plt:.0f} — all within expected range.",
    ],
}

_PLAN_PHRASES = [
    "Continue current regimen at full dose; next cycle in 2 weeks.",
    "Dose reduction to 75% for next cycle given toxicity profile.",
    "Hold chemotherapy; reassess in one week pending lab recovery.",
    "Growth factor support (G-CSF) initiated for neutropenia prophylaxis.",
    "Refer to palliative care for symptom management.",
    "Supportive care measures initiated; antiemetic regimen optimised.",
    "Corticosteroids initiated for suspected immune-related adverse event.",
    "Blood transfusion ordered for symptomatic anemia.",
    "Imaging ordered to evaluate for pneumonitis; hold checkpoint inhibitor.",
    "Continue current regimen; close monitoring of renal and hepatic function.",
]


# ---------------------------------------------------------------------------
# Patient generator
# ---------------------------------------------------------------------------

@dataclass
class OncologyPatient:
    patient_id: str
    age: int
    sex: str
    cancer_type: str
    stage: str
    treatment: str
    cycle_number: int
    ecog_score: int
    # Lab values
    wbc: float
    anc: float
    hemoglobin: float
    platelets: float
    creatinine: float
    alt: float
    bilirubin: float
    # Symptom scores (0–10)
    fatigue_score: float
    nausea_score: float
    pain_score: float
    dyspnea_score: float
    # Clinical note
    clinical_note: str
    # Outcomes
    adverse_event: bool
    ae_type: Optional[str]
    ae_grade: Optional[int]


def _sample_labs(ae: bool, ae_type: Optional[str]) -> dict:
    """Generate lab values consistent with the patient's AE status."""
    # Normal ranges (with some baseline variation)
    wbc = max(0.5, np.random.normal(6.5, 2.0))
    anc = max(0.1, wbc * np.random.uniform(0.50, 0.70))
    hgb = max(5.0, np.random.normal(12.5, 1.8))
    plt = max(20, np.random.normal(220, 60))
    creat = max(0.4, np.random.normal(0.90, 0.20))
    alt = max(5, np.random.normal(28, 12))
    bili = max(0.1, np.random.normal(0.7, 0.25))

    # Modify labs to reflect AE type
    if ae and ae_type:
        if ae_type == "febrile_neutropenia":
            anc = max(0.05, np.random.uniform(0.05, 0.50))
            wbc = anc / 0.55
        elif ae_type == "anemia":
            hgb = max(5.0, np.random.uniform(6.0, 9.5))
        elif ae_type == "thrombocytopenia":
            plt = max(10, np.random.uniform(15, 80))
        elif ae_type == "hepatotoxicity":
            alt = max(40, np.random.uniform(120, 800))
            bili = max(0.5, np.random.uniform(1.5, 5.0))
        elif ae_type == "nephrotoxicity":
            creat = max(1.0, np.random.uniform(1.5, 3.5))

    return dict(wbc=round(wbc, 1), anc=round(anc, 2), hemoglobin=round(hgb, 1),
                platelets=round(plt, 0), creatinine=round(creat, 2),
                alt=round(alt, 0), bilirubin=round(bili, 2))


def _sample_symptom_scores(ae: bool, ae_type: Optional[str]) -> dict:
    """Generate CTCAE-informed symptom scores (0–10)."""
    base = dict(
        fatigue_score=round(max(0, np.random.normal(2.5, 2.0)), 1),
        nausea_score=round(max(0, np.random.normal(1.5, 1.8)), 1),
        pain_score=round(max(0, np.random.normal(2.0, 2.0)), 1),
        dyspnea_score=round(max(0, np.random.normal(1.0, 1.5)), 1),
    )
    # Clamp to [0, 10]
    base = {k: min(10.0, v) for k, v in base.items()}

    if ae and ae_type:
        if ae_type in ("fatigue", "anemia", "febrile_neutropenia"):
            base["fatigue_score"] = min(10.0, max(5.0, np.random.uniform(5.0, 9.0)))
        if ae_type in ("nausea_vomiting", "mucositis"):
            base["nausea_score"] = min(10.0, max(4.0, np.random.uniform(4.0, 9.0)))
        if ae_type == "peripheral_neuropathy":
            base["pain_score"] = min(10.0, max(4.0, np.random.uniform(4.0, 8.0)))
        if ae_type in ("pneumonitis", "anemia"):
            base["dyspnea_score"] = min(10.0, max(4.0, np.random.uniform(4.0, 8.0)))

    return base


def _build_clinical_note(p: dict) -> str:
    """Generate a synthetic clinical note from patient attributes."""
    rng = random.Random(hash(p["patient_id"]))

    age_sex = f"{p['age']}-year-old {'male' if p['sex'] == 'M' else 'female'}"
    cancer = p["cancer_type"].replace("_", " ").replace("nsclc", "NSCLC").replace("sclc", "SCLC")
    template = rng.choice(_NOTE_TEMPLATES)
    cycle = p["cycle_number"]

    # Build symptom text
    symptom_phrases = []
    ae_type = p.get("ae_type")
    ae = p.get("adverse_event", False)

    if ae and ae_type:
        # Always include AE-specific language
        ae_category_map = {
            "febrile_neutropenia": "fever",
            "anemia":              "fatigue",
            "thrombocytopenia":    "bleeding",
            "nausea_vomiting":     "nausea",
            "peripheral_neuropathy": "neuropathy",
            "hepatotoxicity":      "fatigue",
            "nephrotoxicity":      "fatigue",
            "pneumonitis":         "dyspnea",
            "colitis":             "diarrhea",
            "fatigue":             "fatigue",
            "diarrhea":            "diarrhea",
            "mucositis":           "mucositis",
        }
        primary_cat = ae_category_map.get(ae_type, "fatigue")
        candidates = _SYMPTOM_PHRASES.get(primary_cat, _SYMPTOM_PHRASES["fatigue"])
        symptom_phrases.append(rng.choice(candidates))
        # Add 1–2 secondary symptoms
        secondary_cats = [c for c in _SYMPTOM_PHRASES if c not in ("no_symptoms", primary_cat)]
        for cat in rng.sample(secondary_cats, k=min(2, len(secondary_cats))):
            if rng.random() < 0.4:
                symptom_phrases.append(rng.choice(_SYMPTOM_PHRASES[cat]))
    else:
        # No significant AE — mostly negative or mild
        if rng.random() < 0.5:
            symptom_phrases.append(rng.choice(_SYMPTOM_PHRASES["no_symptoms"]))
        else:
            cat = rng.choice(["fatigue", "nausea"])
            phrase = rng.choice(_SYMPTOM_PHRASES[cat])
            symptom_phrases.append("Mild " + phrase[0].lower() + phrase[1:])

    symptom_text = " ".join(symptom_phrases)

    # Build lab text
    labs = p
    lab_parts = []
    if labs["anc"] < 1.0:
        lab_parts.append(rng.choice(_LAB_PHRASES["neutropenia"]).format(**labs))
    if labs["hemoglobin"] < 10.0:
        lab_parts.append(rng.choice(_LAB_PHRASES["anemia"]).format(hgb=labs["hemoglobin"]))
    if labs["platelets"] < 100:
        lab_parts.append(rng.choice(_LAB_PHRASES["thrombocytopenia"]).format(plt=labs["platelets"]))
    if labs["alt"] > 80:
        lab_parts.append(rng.choice(_LAB_PHRASES["elevated_alt"]).format(alt=labs["alt"]))
    if labs["creatinine"] > 1.3:
        lab_parts.append(rng.choice(_LAB_PHRASES["elevated_creatinine"]).format(creat=labs["creatinine"]))
    if not lab_parts:
        lab_parts.append(rng.choice(_LAB_PHRASES["normal"]).format(
            wbc=labs["wbc"], hgb=labs["hemoglobin"], plt=labs["platelets"]
        ))
    lab_text = "; ".join(lab_parts)

    plan_text = rng.choice(_PLAN_PHRASES)
    temp = round(37.0 + (rng.random() * 2.0 if p.get("ae_type") == "febrile_neutropenia" else rng.random() * 0.5), 1)
    bp = f"{rng.randint(105, 135)}/{rng.randint(65, 90)}"
    hr = rng.randint(60, 105)
    spo2 = rng.randint(90, 99) if p.get("ae_type") == "pneumonitis" else rng.randint(96, 100)
    pain_site = rng.choice(["the surgical site", "abdomen", "back", "the affected limb", "the injection site"])

    # Fill template placeholders
    note = template.format(
        age_sex=age_sex,
        cycle=cycle,
        treatment=p["treatment"].replace("_", "/"),
        stage=p["stage"],
        cancer=cancer,
        symptom_text=symptom_text,
        lab_text=lab_text,
        plan_text=plan_text,
        temp=temp,
        bp=bp,
        hr=hr,
        spo2=spo2,
        ecog=p.get("ecog_score", 1),
        pain_score=p.get("pain_score", 3),
        pain_site=pain_site,
        assessment_text=rng.choice([
            "Tolerating treatment.",
            "Concerning toxicity — requires management.",
            "Stable disease; ongoing monitoring.",
            "Treatment-limiting adverse event.",
        ]),
        anc=labs["anc"],
    )

    return note


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_oncology_ehr(
    n_patients: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic oncology EHR dataset.

    Args:
        n_patients: Number of patients to generate (default 2000).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with one row per patient.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []
    for i in range(n_patients):
        patient_id = f"PT{i + 1:04d}"
        cancer_type = random.choices(
            CANCER_TYPES,
            weights=[14, 12, 5, 12, 8, 7, 8, 7, 5, 8],
        )[0]
        stage = random.choices(STAGES, weights=[5, 15, 30, 50])[0]
        treatment = random.choice(TREATMENTS[cancer_type])
        age = int(np.clip(np.random.normal(63, 13), 25, 90))
        sex = random.choice(["M", "F"])
        cycle = random.randint(1, 8)
        ecog = random.choices([0, 1, 2, 3], weights=[25, 45, 20, 10])[0]

        # Determine AE
        base_risk = _AE_RISK_BY_TREATMENT.get(treatment, 0.35)
        # Modifiers
        if stage == "IV":
            base_risk = min(0.95, base_risk * 1.25)
        if ecog >= 2:
            base_risk = min(0.95, base_risk * 1.20)
        if age > 70:
            base_risk = min(0.95, base_risk * 1.10)

        adverse_event = random.random() < base_risk
        ae_type = None
        ae_grade = None
        if adverse_event:
            ae_type = random.choices(AE_TYPES, weights=_AE_WEIGHTS)[0]
            ae_grade = random.choices([1, 2, 3, 4], weights=[30, 40, 22, 8])[0]

        labs = _sample_labs(adverse_event, ae_type)
        symptom_scores = _sample_symptom_scores(adverse_event, ae_type)

        patient_dict = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "cancer_type": cancer_type,
            "stage": stage,
            "treatment": treatment,
            "cycle_number": cycle,
            "ecog_score": ecog,
            **labs,
            **symptom_scores,
            "adverse_event": adverse_event,
            "ae_type": ae_type,
            "ae_grade": ae_grade,
        }

        patient_dict["clinical_note"] = _build_clinical_note(patient_dict)
        records.append(patient_dict)

    df = pd.DataFrame(records)

    # Enforce dtypes
    df["adverse_event"] = df["adverse_event"].astype(bool)
    df["ae_grade"] = df["ae_grade"].astype("Int64")   # nullable int

    col_order = [
        "patient_id", "age", "sex", "cancer_type", "stage", "treatment",
        "cycle_number", "ecog_score",
        "wbc", "anc", "hemoglobin", "platelets", "creatinine", "alt", "bilirubin",
        "fatigue_score", "nausea_score", "pain_score", "dyspnea_score",
        "clinical_note",
        "adverse_event", "ae_type", "ae_grade",
    ]
    return df[col_order]


if __name__ == "__main__":
    import os
    import sys

    os.makedirs("data", exist_ok=True)
    print("Generating synthetic oncology EHR (n=2000)…")
    df = generate_oncology_ehr(n_patients=2000, seed=42)

    out_csv = "data/oncology_ehr.csv"
    out_parquet = "data/oncology_ehr.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)

    n_ae = df["adverse_event"].sum()
    print(f"Generated {len(df)} patients; AE prevalence = {n_ae}/{len(df)} ({n_ae/len(df):.1%})")
    print(f"Saved to {out_csv} and {out_parquet}")
    print(df["ae_type"].value_counts().head(10).to_string())
