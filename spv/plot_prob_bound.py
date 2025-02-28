import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def bernoulli_lower_bound(p_hat, n, alpha):
    """
    Returns a one-sided Hoeffding-based lower confidence bound for a Bernoulli proportion:
        p >= p_hat - sqrt( ln(2/alpha) / (2n) ).
    Clamps at 0. Probability >= 1 - alpha.
    """
    eps = math.sqrt(np.log(2.0 / alpha) / (2.0 * n))
    return max(0.0, p_hat - eps)


sns.set_theme(style="whitegrid", font_scale=1.2)
file_name = ''
if file_name == '':
    raise ValueError("Please specify the verification results file")
model_steps = pd.read_csv(file_name)

alpha = 0.01  # confidence level
k_values = np.arange(5, 21, 1)
model_ids = list(model_steps.keys())
num_models = len(model_ids)

colors = sns.color_palette("colorblind", 6)
colors[1] = '#FFC107'

plt.figure(figsize=(8, 6), dpi=300)

for i, model_id_str in enumerate(model_ids):
    steps = model_steps[model_id_str]
    n = len(steps)
    if n == 0:
        continue
    p_lower_arr = []
    for k in k_values:
        # Hoeffding-based lower bound
        p_hat = np.mean(steps >= k)
        p_lower = bernoulli_lower_bound(p_hat, n, alpha)
        p_lower_arr.append(p_lower)
    p_lower_arr = np.array(p_lower_arr)
    plt.plot(
        k_values, 
        p_lower_arr,
        label=f"{model_id_str} (Ours)" if 'SPV' in model_id_str else model_id_str,
        color=colors[i],
        linewidth=3
    )

plt.grid(False)
plt.xlabel(r"$k$ (Target Verification Step Threshold)", fontsize=14)
plt.ylabel(r"Lower-Bound Probability [ $VerifiedSteps \geq k$ ]", fontsize=14)
plt.title("CARLA", fontsize=16)
plt.xticks([5, 10, 15, 20], fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()

