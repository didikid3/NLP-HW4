import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data.csv")

# Make prompt_id categorical for nicer coloring
df["prompt_id"] = df["prompt_id"].astype(str)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="model",
    y="rmse",
    hue="prompt_id"
)

plt.title("RMSE by Model and Prompt")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.legend(title="Prompt ID")
plt.tight_layout()
plt.show()
