import numpy as np
from scipy.stats import norm
from flask import Flask, request, jsonify

app = Flask(__name__)

# -----------------------------
# Two Proportion Test Function
# -----------------------------
def twoP_test(x, y, alpha=0.05, alternative='two-sided'):

    n1 = len(x)
    n2 = len(y)

    p1 = sum(x) / n1
    p2 = sum(y) / n2

    pbar = ((n1 * p1 + n2 * p2) / (n1 + n2))

    se = np.sqrt((pbar * (1 - pbar)) * ((1 / n1) + (1 / n2)))

    z = (p1 - p2) / se

    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == "right":
        p = 1 - norm.cdf(z)
    else:
        p = norm.cdf(z)

    result = "hypothesis rejected" if p < alpha else "hypothesis accepted"

    return z, p, result


# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return "Flask server working"


# -----------------------------
# API Route
# -----------------------------
@app.route("/test", methods=["POST"])
def run_test():

    data = request.json

    x = data["x"]
    y = data["y"]

    z, p, result = twoP_test(x, y)

    return jsonify({
        "z_statistic": z,
        "p_value": p,
        "decision": result
    })


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)