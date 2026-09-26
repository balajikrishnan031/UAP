# Universal Adaptive Prediction (UAP) Engine

> **A Closed-Loop, Data-Driven Machine Learning Framework with Adaptive Parameter Intelligence, Dynamic Ensembling, Stress-Testing, and Explainable Recourse.**

---

## 🌟 Executive Summary

Traditional Machine Learning and standard AutoML frameworks rely on brute-force model searches and treat datasets as static matrices. **UAP (Universal Adaptive Prediction)** introduces an **Adaptive Parameter Intelligence Layer** that automatically profiles data health, computes dynamic feature weights ($W_i$), routes execution to optimal model architectures, stress-tests models against adversarial noise, explains predictions via TreeSHAP, and guards against production data drift using the Population Stability Index ($PSI$).

---

## 🏛️ Master 4-Module Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │                 RAW DATASET                  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: DATASET PROFILER & PROBLEM DETECTOR                                   │
│  • Automated Type Extractor (Continuous, Categorical, Temporal, Text)           │
│  • Task Classifier (Regression, Binary / Multiclass Classification)             │
│  • Health Metrics: Imbalance (IR), Dimensionality (DR), Outlier (OSI), NLI      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼ [Artifact 1: DCV JSON]
┌─────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: ADAPTIVE FEATURE WEIGHTING (AFWE) & DYNAMIC ROUTING (DMRE)            │
│  • Multi-Factor Weighting: W_i = Rel(X_i, Y) * (1 - Red) * (1 - Noise)          │
│  • Continuous Matrix Scaling: X_weighted = X * diag(W)                          │
│  • Meta-Learner Routing Matrix: Selects / Excludes Candidate Architectures      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼ [Artifact 2: MSC JSON + X_weighted]
┌─────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: BAYESIAN HPO, ADAPTIVE CV & PERTURBATION STRESS-TESTER               │
│  • Task-Aware Partitioning (Stratified, Purged Walk-Forward, Group CV)          │
│  • Tree-structured Parzen Estimator (TPE) + Multi-Fidelity Halving              │
│  • Synthetic Corruption Battery (Gaussian Noise + Feature Dropout)             │
│  • Robustness Decay Score (RDS) & Uncertainty Blended Stacking                  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼ [Artifact 3: EMA JSON]
┌─────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: EXPLAINABLE AI, ACTIONABLE WHAT-IF & SERVING                         │
│  • TreeSHAP Attribution Matrix (Global ranking & Local Waterfall decomposition) │
│  • Actionable Counterfactual Recourse Engine (Respects feature immutability)     │
│  • Real-Time Drift Monitor: Population Stability Index (PSI >= 0.25 Alert)      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. Module 1: Dataset Characteristic Vector (DCV)
- **Regression Index ($RI$)**:
  $$RI = \frac{U(Y)}{N}$$
  If $RI > 0.05 \implies$ Regression; $U(Y) = 2 \implies$ Binary Classification; $2 < U(Y) \le 50 \implies$ Multiclass.
- **Imbalance Ratio ($IR$)**:
  $$IR = \frac{N_{\text{majority}}}{N_{\text{minority}}}$$
- **Dimensionality Ratio ($DR$)**:
  $$DR = \frac{D}{N}$$
- **Outlier Severity Index ($OSI$)**:
  $$OSI = \frac{\text{Outlier Cells}}{N \times D_{\text{cont}}}, \quad \text{where outlier if } x < Q_1 - 1.5 \cdot IQR \text{ or } x > Q_3 + 1.5 \cdot IQR$$
- **Non-Linearity Index ($NLI$)**:
  $$NLI = \max\left(0, 1 - \frac{\text{MSE}_{\text{Tree Stump}}}{\text{MSE}_{\text{Linear Model}}}\right)$$

### 2. Module 2: Adaptive Feature Weighting Engine (AFWE)
For each feature $X_i$, its dynamic quality weight $W_i \in [0, 1]$ is:
$$W_i = \underbrace{\text{Rel}(X_i, Y)}_{\text{Mutual Info + Rank Correlation}} \times \underbrace{\left(1 - \text{Red}(X_i)\right)}_{\text{Redundancy Penalty}} \times \underbrace{\left(1 - \text{Noise}(X_i)\right)}_{\text{Noise \& Outlier Discount}}$$
Transformed feature matrix:
$$X_{\text{weighted}} = X \cdot \operatorname{diag}(W_1, W_2, \dots, W_d)$$

### 3. Module 3: Robustness Decay Score ($RDS$)
Evaluates model degradation when holdout data is perturbed by 15% noise:
$$RDS = \frac{\text{Metric}_{\text{Clean}} - \text{Metric}_{\text{Corrupted}}}{\text{Metric}_{\text{Clean}}}$$
If $RDS \le 0.20 \implies$ Deployment status **PASSED**.

### 4. Module 4: Population Stability Index ($PSI$)
Monitors live production inference streams against baseline training distribution:
$$PSI = \sum_{b=1}^{B} (Q_b - P_b) \times \ln\left(\frac{Q_b}{P_b}\right)$$
- $PSI < 0.10$: **Stable**
- $0.10 \le PSI \le 0.25$: **Moderate Drift Warning**
- $PSI > 0.25$: **Critical Drift Alert (Triggers Automated Retraining Loop back to Module 1)**

---

## 🚀 Quickstart & Usage

### 1. Run End-to-End Pipeline Demonstration
Executes all 4 modules on a challenge imbalanced non-linear dataset and prints the generated `DCV`, `MSC`, and `EMA` artifacts:
```bash
python run_demo.py
```

### 2. Launch Interactive Streamlit Dashboard
Experience the visual UI for data profiling, adaptive feature weights, model comparisons, SHAP waterfall plots, real-time What-If scenario probing, and live drift detection:
```bash
streamlit run app.py
```

### 3. Run Benchmark Suite
Compares UAP against standard XGBoost baselines across classification and regression challenge tasks:
```bash
python benchmarks/benchmark_suite.py
```

### 4. Run Automated Unit Tests
```bash
python -m unittest discover -s tests
```

---

## 📦 Project Structure

```
Prediction Model/
├── uap/
│   ├── core/
│   │   └── contracts.py           # Schemas for DCV, MSC, EMA artifacts
│   ├── module1_profiler/
│   │   └── profiler.py            # Task detection, IR, DR, OSI, NLI
│   ├── module2_weighting_routing/
│   │   ├── afwe.py                # Adaptive Feature Weighting Engine
│   │   └── dmre.py                # Dynamic Model Routing Engine
│   ├── module3_training_hpo/
│   │   ├── partitioning.py        # Adaptive CV schemes
│   │   ├── optimizer.py           # Optuna/TPE Bayesian HPO & Ensembling
│   │   └── stress_tester.py       # Noise perturbation & RDS analysis
│   ├── module4_xai_serving/
│   │   ├── xai.py                 # TreeSHAP Global & Local attributions
│   │   ├── what_if.py             # Actionable Counterfactual Recourse
│   │   └── drift.py               # Population Stability Index (PSI)
│   └── engine.py                  # Master UAPEngine pipeline orchestrator
├── synthetic_data/
│   └── generator.py               # UAPSyntheticDataGenerator
├── benchmarks/
│   └── benchmark_suite.py         # Comparative benchmark harness
├── tests/
│   └── test_uap.py                # Unit test suite
├── run_demo.py                    # Complete CLI demonstration
├── app.py                         # Interactive Streamlit Web UI
└── README.md                      # Documentation & Architecture Guide
```
