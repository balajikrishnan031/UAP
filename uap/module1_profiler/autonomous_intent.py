"""
UAP 5.0 Autonomous Intent & Objective Discovery Engine.
Allows zero-configuration dataset ingestion:
When a user uploads a dataset without specifying what they want or what the target is,
this engine automatically:
1. Semantically scans columns, data types, distributions, and cardinalities.
2. Identifies the domain (Healthcare, Banking, Telecom, IoT, Retail, Energy, etc.).
3. Discovers and ranks candidate target columns with confidence scores.
4. Infers the business objective (Risk Minimization, Value Prediction, Anomaly Detection).
5. If the user had an alternative expectation in mind, provides immediate intent alignment options.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from uap.core.contracts import ProblemType

@dataclass
class TargetCandidate:
    column_name: str
    problem_type: ProblemType
    confidence_score: float
    domain_semantics: str
    inferred_business_objective: str
    rationale: str
    sample_values: List[Any]

@dataclass
class IntentDiscoveryCard:
    detected_domain: str
    recommended_target: str
    problem_type: ProblemType
    primary_business_goal: str
    risk_posture: str
    confidence_score: float
    candidate_ranking: List[TargetCandidate]
    alternative_intent_options: List[str]
    autonomous_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_domain": self.detected_domain,
            "recommended_target": self.recommended_target,
            "problem_type": self.problem_type.value,
            "primary_business_goal": self.primary_business_goal,
            "risk_posture": self.risk_posture,
            "confidence_score": round(self.confidence_score, 3),
            "candidate_ranking": [
                {
                    "column_name": c.column_name,
                    "problem_type": c.problem_type.value,
                    "confidence_score": round(c.confidence_score, 3),
                    "domain_semantics": c.domain_semantics,
                    "inferred_business_objective": c.inferred_business_objective,
                    "sample_values": c.sample_values[:5],
                }
                for c in self.candidate_ranking
            ],
            "alternative_intent_options": self.alternative_intent_options,
            "autonomous_rationale": self.autonomous_rationale,
        }

    def format_text_report(self) -> str:
        border = "═" * 70
        divider = "─" * 70
        lines = [
            border,
            "        🧠 UAP 5.0 AUTONOMOUS INTENT & GOAL DISCOVERY CARD",
            border,
            f"  Detected Domain          : {self.detected_domain.upper()}",
            f"  Recommended Target Column: {self.recommended_target}",
            f"  Problem Formulation      : {self.problem_type.value}",
            f"  Primary Business Goal    : {self.primary_business_goal}",
            f"  Auto-Calibrated Risk     : {self.risk_posture}",
            f"  Discovery Confidence     : {self.confidence_score * 100:.1f}%",
            divider,
            "  INTENT RANKING (Top Candidates Found in Dataset):",
        ]
        for idx, c in enumerate(self.candidate_ranking[:3], 1):
            mark = "★ [PRIMARY CHOICE]" if idx == 1 else f"  [ALTERNATIVE #{idx}]"
            lines.append(f"  {mark} '{c.column_name}' ({c.problem_type.value})")
            lines.append(f"      Semantic Goal: {c.inferred_business_objective}")
            lines.append(f"      Confidence   : {c.confidence_score * 100:.1f}% | Rationale: {c.rationale}")

        if self.alternative_intent_options:
            lines.append(divider)
            lines.append("  DID YOU INTEND A DIFFERENT OBJECTIVE? (Alternative Intent Options):")
            for alt in self.alternative_intent_options:
                lines.append(f"    • {alt}")
        lines.append(border)
        return "\n".join(lines)


class AutonomousIntentEngine:
    """
    Cognitive Intent Discovery System:
    Automatically figures out what the user needs from a dataset before running any ML pipeline.
    """

    # Semantic Keyword Dictionaries
    DOMAIN_KEYWORDS = {
        "Healthcare & Medicine": [
            "patient", "age", "sex", "chol", "blood", "heart", "cancer", "tumor",
            "diagnosis", "cell", "radius", "texture", "death", "survival", "treatment",
            "hospital", "symptom", "disease"
        ],
        "Banking & Financial Risk": [
            "credit", "loan", "default", "amount", "interest", "installment", "balance",
            "income", "debt", "risk", "fraud", "transaction", "applicant", "account"
        ],
        "Telecom & Customer Churn": [
            "churn", "tenure", "contract", "customer", "charges", "service", "calls",
            "internet", "stream", "phone", "plan", "usage"
        ],
        "Industrial IoT & Reliability": [
            "machine", "sensor", "rpm", "torque", "temperature", "wear", "vibration",
            "failure", "maintenance", "pressure", "equipment", "tool"
        ],
        "Macroeconomics & Real Estate": [
            "house", "home", "price", "val", "median", "income", "rooms", "bedrooms",
            "population", "latitude", "longitude", "zip", "rent", "sqft"
        ],
        "Energy & Smart Utilities": [
            "grid", "load", "demand", "mw", "megawatt", "voltage", "power", "electricity",
            "generator", "consumption"
        ],
        "Smart Agriculture": [
            "crop", "soil", "nitrogen", "phosphorus", "potassium", "rainfall", "humidity",
            "ph", "yield", "fertilizer", "seed"
        ],
        "Cybersecurity & Defense": [
            "intrusion", "attack", "bytes", "packet", "login", "failed", "serror",
            "protocol", "flag", "malicious", "firewall"
        ],
    }

    TARGET_SEMANTIC_PATTERNS = [
        # Explicit target labels
        (r"^(target|label|y|class|outcome|response|ground_truth)$", 1.0, "Canonical Ground Truth Target"),
        (r"(churn|is_churn|churned)$", 0.95, "Customer Retention & Churn Prevention"),
        (r"(fraud|is_fraud|is_fraudulent)$", 0.95, "Financial Fraud & Anomaly Prevention"),
        (r"(default|is_default|bad_loan|risk)$", 0.92, "Credit Default Risk Minimization"),
        (r"(diagnosis|malignant|cancer|disease|death_event|mortality)$", 0.95, "Clinical Diagnosis & Mortality Prediction"),
        (r"(failure|machine_failure|tool_wear_failure)$", 0.92, "Preventive Equipment Downtime Mitigation"),
        (r"(intrusion|is_intrusion|attack)$", 0.95, "Cyber Intrusion Threat Mitigation"),
        (r"(yield|optimal_yield|crop_recommendation)$", 0.88, "Agricultural Yield Maximization"),
        (r"(price|medhouseval|median_house_value|val|cost|revenue|sales)$", 0.90, "Economic Valuation & Revenue Forecasting"),
        (r"(load|demand|grid_load|pm25|aqi)$", 0.88, "Continuous Resource & Environmental Forecasting"),
    ]

    def discover(self, df: pd.DataFrame, user_goal: Optional[str] = None) -> IntentDiscoveryCard:
        """
        Discovers domain, candidate targets, and business objectives from a dataset.
        If user_goal prompt is provided, aligns candidates with the prompt.
        """
        n_rows, n_cols = df.shape
        columns = list(df.columns)

        # 1. Domain Detection
        detected_domain, domain_score = self._infer_domain(columns)

        # 2. Candidate Target Scoring
        candidates: List[TargetCandidate] = []
        for col_idx, col in enumerate(columns):
            s = df[col]
            cand = self._score_target_candidate(
                col_name=col,
                series=s,
                col_idx=col_idx,
                total_cols=n_cols,
                n_rows=n_rows,
                user_goal=user_goal,
                detected_domain=detected_domain,
            )
            if cand is not None:
                candidates.append(cand)

        # Sort by confidence score descending
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)

        if not candidates:
            # Fallback to the last column
            last_col = columns[-1]
            prob_type = ProblemType.REGRESSION if pd.api.types.is_numeric_dtype(df[last_col]) and df[last_col].nunique() > 20 else ProblemType.BINARY_CLASSIFICATION
            primary_candidate = TargetCandidate(
                column_name=last_col,
                problem_type=prob_type,
                confidence_score=0.50,
                domain_semantics="Dataset Outcome Feature",
                inferred_business_objective=f"Predict outcome '{last_col}'",
                rationale="Defaulted to final column in dataset layout.",
                sample_values=list(df[last_col].dropna().unique()[:5])
            )
            candidates.append(primary_candidate)

        primary = candidates[0]

        # 3. Formulate Alternative Intent Options
        alternative_options = []
        for alt in candidates[1:4]:
            alternative_options.append(
                f"Switch target to '{alt.column_name}' ({alt.problem_type.value}) -> Goal: {alt.inferred_business_objective}"
            )
        if len(candidates) == 1:
            alternative_options.append("No ambiguous target found. Dataset appears cleanly structured for a single task.")

        # 4. Formulate Risk Posture
        if "Healthcare" in detected_domain or "Clinical" in detected_domain or "Medical" in detected_domain:
            risk_posture = "High-Safety Clinical Tier (Zero-Tolerance False Negatives, 95% Conformal)"
        elif "Banking" in detected_domain or "Financial" in detected_domain or "Fraud" in detected_domain:
            risk_posture = "Asymmetric Financial Tier (Penalize Default/Fraud Misses, Selective Abstention)"
        else:
            risk_posture = "Balanced Industrial Standard (Macro-F1 / R2, 95% Conformal Bounds)"

        rationale = (
            f"Autonomous schema scan identified dataset as '{detected_domain}' with {n_rows} instances and {n_cols} attributes. "
            f"Target column '{primary.column_name}' was selected with {primary.confidence_score*100:.1f}% confidence "
            f"as a {primary.problem_type.value} problem based on semantic naming and distinct class entropy. "
            f"Configured objective: {primary.inferred_business_objective}."
        )

        return IntentDiscoveryCard(
            detected_domain=detected_domain,
            recommended_target=primary.column_name,
            problem_type=primary.problem_type,
            primary_business_goal=primary.inferred_business_objective,
            risk_posture=risk_posture,
            confidence_score=primary.confidence_score,
            candidate_ranking=candidates,
            alternative_intent_options=alternative_options,
            autonomous_rationale=rationale,
        )

    def _infer_domain(self, columns: List[str]) -> Tuple[str, float]:
        col_text = " ".join([c.lower() for c in columns])
        domain_matches = {}

        for domain, kws in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in kws if kw in col_text)
            domain_matches[domain] = matches

        best_domain = max(domain_matches.items(), key=lambda x: x[1])
        if best_domain[1] > 0:
            return best_domain[0], float(best_domain[1])
        return "General Industrial Domain", 0.0

    def _score_target_candidate(
        self,
        col_name: str,
        series: pd.Series,
        col_idx: int,
        total_cols: int,
        n_rows: int,
        user_goal: Optional[str],
        detected_domain: str,
    ) -> Optional[TargetCandidate]:
        clean_s = series.dropna()
        if len(clean_s) == 0:
            return None

        u_count = clean_s.nunique()
        is_num = pd.api.types.is_numeric_dtype(clean_s)
        col_lower = col_name.lower().strip()

        # Disqualify IDs, hashes, timestamps
        if col_lower in ["id", "index", "uid", "user_id", "timestamp", "date", "time_stamp", "guid"]:
            return None
        if not is_num and u_count > n_rows * 0.90 and u_count > 50:
            return None  # Unique IDs / Text

        base_score = 0.0
        rationale_bits = []
        domain_semantics = "Generic Feature"
        inferred_goal = f"Predict {col_name}"

        # 1. Pattern Matching against Target Semantic Patterns
        matched_pattern = False
        for pattern, weight, desc in self.TARGET_SEMANTIC_PATTERNS:
            if re.search(pattern, col_lower):
                base_score += weight * 0.60
                domain_semantics = desc
                inferred_goal = desc
                rationale_bits.append(f"Name matched semantic pattern '{pattern}' ({desc})")
                matched_pattern = True
                break

        # 2. Position bias (targets are commonly the last or first column)
        if col_idx == total_cols - 1:
            base_score += 0.20
            rationale_bits.append("Located at final column in schema")
        elif col_idx == 0 and ("target" in col_lower or "label" in col_lower):
            base_score += 0.15
            rationale_bits.append("Located at first column with target naming")

        # 3. User Goal Keyword Matching (High Priority Override)
        if user_goal:
            user_words = [w.strip() for w in user_goal.lower().split() if len(w.strip()) > 2]
            clean_col = col_lower.replace("is_", "").replace("_", "")
            matched_goal = any(
                w in col_lower or col_lower in w or w[:4] in clean_col or clean_col in w
                for w in user_words
            )
            if matched_goal:
                base_score += 0.85
                rationale_bits.append(f"Explicitly aligns with user prompt '{user_goal}'")

        # 4. Problem Type Deduction
        if u_count == 2:
            problem_type = ProblemType.BINARY_CLASSIFICATION
            base_score += 0.15
            rationale_bits.append("Binary cardinal state (2 classes)")
        elif 3 <= u_count <= 15:
            problem_type = ProblemType.MULTICLASS_CLASSIFICATION
            base_score += 0.10
            rationale_bits.append(f"Categorical cardinal state ({u_count} classes)")
        elif is_num and u_count > 20:
            problem_type = ProblemType.REGRESSION
            if matched_pattern or col_idx == total_cols - 1:
                base_score += 0.10
                rationale_bits.append("Continuous numeric variation (Regression)")
            else:
                base_score -= 0.15  # Continuous feature, less likely to be target unless named
        else:
            return None

        # Filter out low-score feature columns
        final_score = float(np.clip(base_score, 0.05, 1.0))
        if final_score < 0.30 and col_idx != total_cols - 1:
            return None

        sample_vals = list(clean_s.unique()[:5])

        return TargetCandidate(
            column_name=col_name,
            problem_type=problem_type,
            confidence_score=final_score,
            domain_semantics=domain_semantics,
            inferred_business_objective=inferred_goal,
            rationale="; ".join(rationale_bits) or "Schema candidate",
            sample_values=sample_vals,
        )
