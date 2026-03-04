import streamlit as st
import numpy as np
from scipy.stats import t
 
# ----------------------------
# T-Test Function
# ----------------------------
def t_test_independent_pooled(a, b, alpha=0.05, alternative="two-sided"):
    a = np.array(a)
    b = np.array(b)
 
    n1, n2 = len(a), len(b)
    xbar1, xbar2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
 
    sp2 = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)
    se = np.sqrt(sp2 * (1/n1 + 1/n2))
 
    t_cal = (xbar1 - xbar2) / se
    df = n1 + n2 - 2
 
    if alternative == "two-sided":
        t_crit = t.ppf(1 - alpha/2, df)
        p_value = 2 * (1 - t.cdf(abs(t_cal), df))
        reject = abs(t_cal) > t_crit
 
    elif alternative == "greater":
        t_crit = t.ppf(1 - alpha, df)
        p_value = 1 - t.cdf(t_cal, df)
        reject = t_cal > t_crit
 
    elif alternative == "less":
        t_crit = t.ppf(alpha, df)
        p_value = t.cdf(t_cal, df)
        reject = t_cal < t_crit
 
    return {
        "t_cal": t_cal,
        "df": df,
        "p_value": p_value,
        "decision": "Reject H0" if reject else "Fail to Reject H0"
    }
 
# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Independent T-Test (Pooled Variance)")
st.write("Enter your two sample datasets separated by commas.")
 
# Input fields
sample1 = st.text_area("Sample 1 (comma separated)", "12, 14, 15, 10, 13")
sample2 = st.text_area("Sample 2 (comma separated)", "18, 17, 16, 19, 20")
 
alpha = st.number_input("Significance Level (α)", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
 
alternative = st.selectbox(
    "Alternative Hypothesis",
    ["two-sided", "greater", "less"]
)
 
# Button
if st.button("Run T-Test"):
    try:
        # Convert input strings to lists
        a = [float(x.strip()) for x in sample1.split(",")]
        b = [float(x.strip()) for x in sample2.split(",")]
 
        if len(a) < 2 or len(b) < 2:
            st.error("Each sample must contain at least 2 values.")
        else:
            result = t_test_independent_pooled(a, b, alpha, alternative)
 
            st.subheader("Results")
            st.write(f"t statistic: {result['t_cal']:.4f}")
            st.write(f"Degrees of Freedom: {result['df']}")
            st.write(f"p-value: {result['p_value']:.6f}")
            st.write(f"Decision: **{result['decision']}**")
 
    except:
        st.error("Please enter valid numeric values separated by commas.")