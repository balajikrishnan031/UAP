import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from tmrm import TMRM

print("=" * 80)
print("TMRM v4.0 vs INDUSTRY REGRESSORS (HEAD-TO-HEAD BENCHMARK AUDIT)")
print("=" * 80)

results = []

def run_benchmark(name, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Models
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    ridge = Ridge().fit(X_tr, y_tr)
    tmrm = TMRM(random_state=42).fit(X_tr, y_tr)
    
    rf_r2 = r2_score(y_te, rf.predict(X_te))
    gb_r2 = r2_score(y_te, gb.predict(X_te))
    ridge_r2 = r2_score(y_te, ridge.predict(X_te))
    tmrm_r2 = r2_score(y_te, tmrm.predict(X_te))
    
    rf_rmse = np.sqrt(mean_squared_error(y_te, rf.predict(X_te)))
    gb_rmse = np.sqrt(mean_squared_error(y_te, gb.predict(X_te)))
    ridge_rmse = np.sqrt(mean_squared_error(y_te, ridge.predict(X_te)))
    tmrm_rmse = np.sqrt(mean_squared_error(y_te, tmrm.predict(X_te)))
    
    results.append({
        "Dataset": name,
        "Samples": len(X),
        "Features": X.shape[1],
        "TMRM_R2": round(tmrm_r2, 4),
        "RF_R2": round(rf_r2, 4),
        "GB_R2": round(gb_r2, 4),
        "Ridge_R2": round(ridge_r2, 4),
        "TMRM_RMSE": round(tmrm_rmse, 3),
        "RF_RMSE": round(rf_rmse, 3),
        "GB_RMSE": round(gb_rmse, 3),
        "Ricci_Curvature": round(tmrm.ricci_curvature_, 3),
        "Metric_Mix": tmrm.metric_weights_
    })

# 1. Clinical Diabetes Progression
d = load_diabetes()
run_benchmark("Diabetes Clinical Progression", d.data, d.target)

# 2. California Housing (Continuous Price Surface)
cal = fetch_california_housing()
run_benchmark("California Housing Price", cal.data[:2500], cal.target[:2500])

# 3. Friedman-1 Complex Non-Linear Continuous Surface
X_f, y_f = make_friedman1(n_samples=1200, n_features=10, noise=1.0, random_state=42)
run_benchmark("Friedman-1 Non-Linear Energy", X_f, y_f)

df_res = pd.DataFrame(results)
print("\n" + df_res.to_string(index=False))
print("\n" + "=" * 80)
