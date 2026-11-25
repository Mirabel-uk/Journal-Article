# TTAF: Trustworthy Tabular AI Evaluation Framework

**Threshold Transportability and Post-Training Adaptation in Clinical Tabular AI — A Gestational Diabetes Mellitus Case Study**

Mirabel Chinenye Clara Opara · University of Hull · MRes Computer Science, 2024–2026

---

## Overview

This repository contains the full experimental pipeline for the TTAF framework, which evaluates the deployment readiness of tabular machine learning models by assessing threshold transportability across population distribution shifts. The framework is applied to gestational diabetes mellitus (GDM) prediction using one internal clinical dataset and two external proxy cohorts.

The central finding is that high internal discrimination (ROC-AUC ≈ 0.995–0.998) does not guarantee clinically usable decision behaviour under distribution shift. The TabTransformer's internal recall of 0.945 collapsed to 0.097 on the PIMA external cohort at the internally selected threshold — a failure mode invisible to conventional evaluation metrics. The Threshold-Instability Score (TS) framework introduced here provides a principled, clinically weighted mechanism for post-training threshold adaptation without retraining.

---

## Repository Contents

```
Journal-Article/
│
├── TTAF_Revised_notebook.ipynb      # Full experimental pipeline (see below)
├── README.md                        # This file
│
└── data/                            # Data sources (see Data section below)
```

### TTAF_Revised_Notebook.ipynb

The notebook implements the complete TTAF pipeline. Run all cells in order from top to bottom to reproduce all manuscript results. All random seeds are fixed (seed = 42).

| Section | Content |
|---|---|
| Setup | Environment setup, reproducibility controls, output folder structure |
| 2 | Harmonise internal dataset — rename variables to five-feature predictor space |
| 3 | Load and harmonise external datasets (PIMA, Pregnancy) |
| 4 | Preprocessing — median imputation (RF) and median imputation + standardisation (TT) |
| 5 | Train final models — Random Forest and TabTransformer; repeated-seed stability check; high-discrimination leakage audit |
| 6 | Final threshold optimisation (deployment-oriented and stress-test thresholds) |
| 7 | Final performance tables; bootstrap confidence intervals; probability distribution shift analysis; post-training threshold adaptation (TS framework, four strategies, weight sensitivity analysis); formal transportability validation |
| Ablation | Glucose-exclusion ablation study (Reviewer 2, Comment 1) — models retrained without plasma glucose |
| 8 | Calibration and recalibration (isotonic regression) |
| 9 | Decision curve analysis (internal, PIMA, Pregnancy cohorts) |
| 10 | Main manuscript figures (ROC curves, calibration plots, error rate) |
| 11 | Subgroup robustness analysis (maternal age, BMI category, parity) |
| 12 | Statistical comparison — McNemar exact test |
| 13 | XAI reliability — SHAP cross-seed stability, SHAP–LIME cross-method agreement, perturbation robustness |
| 14 | Final export checklist — all tables and figures produced |

---

## Datasets

Three datasets are used in this study. All are publicly available and de-identified.

| Dataset | Source | N | Outcome | Role |
|---|---|---|---|---|
| Internal GDM dataset | Institutional (Hull) | ≈3,525 | GDM diagnosis (OGTT-based) | Development + internal test |
| PIMA Indians Diabetes Database | UCI Machine Learning Repository | 768 | Type 2 diabetes (2-hr plasma glucose ≥ 200 mg/dL) | External proxy — methodological stress test (stronger shift) |
| Pregnancy complication dataset | Kaggle | 3,000 | Any pregnancy complication (binarised) | External proxy — methodological stress test (moderate shift) |

**Important:** The PIMA and Pregnancy datasets are not GDM-specific cohorts. They are used exclusively as methodological stress tests to evaluate threshold behaviour under structured distribution shift. All findings from external evaluation should be interpreted as evidence of threshold instability under controlled conditions, not as evidence of deployment failure in prospectively matched GDM populations.

### Feature harmonisation

Five features were harmonised across all three datasets:

| Harmonised name | Internal source | PIMA source | Pregnancy source |
|---|---|---|---|
| `age` | Age | Age | Age |
| `pregnancies` | No of Pregnancy | Pregnancies | No of Pregnancies |
| `bmi` | BMI | BMI | BMI |
| `dia_bp` | Dia BP | BloodPressure | BloodPressure |
| `glucose` | OGTT | Glucose | Glucose |

---

## Models

Two model architectures are evaluated:

**Random Forest (RF)** — scikit-learn `RandomForestClassifier`. Chosen as a well-established ensemble baseline with stable probability outputs under distribution shift.

**TabTransformer (TT)** — attention-based architecture for tabular data (Huang et al., NeurIPS 2020). Chosen as a representative deep tabular model with strong internal discrimination but susceptibility to probability mass compression under shift.

Both models are trained on the internal dataset with a fixed random seed and evaluated on internal held-out test data and both external cohorts without retraining.

---

## Threshold-Instability Score (TS)

The TS is the primary methodological contribution. It provides a clinically weighted composite measure of threshold transportability failure:

```
TS(τ) = w1·ΔSpecificity + w2·ΔRecall + w3·ΔNetBenefit + w4·ΔCalibrationShift
```

Default weights: w1 = 0.20, w2 = 0.40, w3 = 0.30, w4 = 0.10

The w2 = 0.40 weight on recall reflects the clinical priority of minimising missed GDM cases in a screening context. Weights can be adapted to local clinical priorities. A five-scenario sensitivity analysis (baseline, recall ±30%, equal weights, specificity-focused) confirms that the qualitative conclusions — specifically that TS-minimised threshold adaptation substantially outperforms fixed threshold transfer — are robust to weight perturbation.

---

## Key Results

| Model | Dataset | Fixed τ | Adapted τ* | Fixed Recall | Adapted Recall | Fixed TS | Adapted TS |
|---|---|---|---|---|---|---|---|
| RF | PIMA | 0.40 | 0.05 | 0.567 | 0.985 | 0.318 | 0.180 |
| RF | Pregnancy | 0.40 | 0.32 | 0.474 | 0.692 | 0.317 | 0.238 |
| TT | PIMA | 0.55 | 0.01 | 0.097 | 0.937 | 0.455 | 0.155 |
| TT | Pregnancy | 0.55 | 0.14 | 0.068 | 0.634 | 0.489 | 0.242 |

Split-validation across 100 stratified bootstrap splits (70/30 target-calibration/target-test) confirmed negligible optimism (maximum mean recall difference 0.021) in all four model–dataset combinations.

---

## Citation

If you use this code or the TTAF framework in your work, please cite:

> Opara, M.C.C. (2026). *Threshold Transportability and Post-Training Adaptation in Clinical Tabular AI: A Gestational Diabetes Mellitus Case Study*. MRes thesis, University of Hull.

---

## Ethics and Data Availability

This study used retrospective, de-identified data. The internal gestational diabetes dataset was collected under prior institutional ethical approval. No additional ethical approval was required for the secondary use of the PIMA or Pregnancy datasets, both of which are publicly available. All three datasets are de-identified.

---

## Dependencies

The notebook runs on Python 3.8+. Key libraries:

- `scikit-learn` — Random Forest, isotonic regression calibration, PSI computation
- `torch` — TabTransformer implementation
- `shap` — SHAP explainability
- `lime` — LIME explainability
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualisation

---

## Contact

Mirabel Chinenye Clara Opara  
University of Hull  
GitHub: [Mirabel-uk](https://github.com/Mirabel-uk)
