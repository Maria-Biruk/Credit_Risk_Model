Credit Scoring Business Understanding

1. Basel II and the Need for Interpretability

The Basel II Accord requires financial institutions to build credit risk models that are transparent, interpretable, and well-documented. This ensures regulatory compliance and financial stability.

Credit risk models must be explainable so that:

Regulators can understand how decisions are made
Banks can justify lending decisions
Feature impact is traceable and auditable
Model behavior remains stable over time

Because of this, interpretability is a core requirement in credit scoring systems.

2. Why a Proxy Variable is Necessary and Its Risks

This dataset does not contain a direct default label, which is required for supervised learning.

Therefore, we construct a proxy target variable using behavioral signals such as:

Recency of transactions
Frequency of activity
Monetary value patterns (RFM)

This proxy approximates customer risk behavior.

However, proxy modeling introduces risks:

Label noise (incorrect classification of good/bad customers)
Bias in defining risk rules
Misalignment with real-world default behavior
Reduced model generalization accuracy

Thus, proxy design must be carefully validated.

3. Trade-offs Between Interpretable and High-Performance Models
   Interpretable Models (Logistic Regression + WoE)
   Easy to explain to regulators
   Stable and predictable behavior
   Widely used in traditional credit scoring
   Supports scorecard development

Limitations:

Cannot capture complex nonlinear patterns
Lower predictive accuracy on large datasets
High-Performance Models (Gradient Boosting, XGBoost, LightGBM)
High predictive accuracy
Captures nonlinear relationships and feature interactions
Works well on behavioral data

Limitations:

Hard to interpret (“black-box” models)
Requires explainability tools (SHAP, LIME)
More difficult to justify in regulated environments
Final Insight

Modern credit risk systems combine both approaches:

Interpretable models for compliance
High-performance models for prediction
Explainability tools to bridge both
