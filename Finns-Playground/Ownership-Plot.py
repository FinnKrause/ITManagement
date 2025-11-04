import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    "steam-200k.csv", header=None, names=["UserID", "Game", "Event", "Hours", "Zeros"]
)

# Work only with purchases
purchases = df[df["Event"] == "purchase"].copy()

# Universe of users
all_users = df["UserID"].unique()
total_users = len(all_users)

# Count unique owners per game
owners_per_game = purchases.groupby("Game")["UserID"].nunique()

# Compute non-owners per game (relative to the same global user set)
nonowners_per_game = total_users - owners_per_game

# Ownership "rate" per your definition: owners / non-owners * 100
# Handle games with nonowners == 0 (everyone owns it) to avoid division by zero
rate_per_game = (owners_per_game / nonowners_per_game.replace(0, np.nan)) * 100

# Assemble a tidy DataFrame
ownership_df = pd.DataFrame(
    {
        "owners": owners_per_game,
        "non_owners": nonowners_per_game,
        "ownership_rate": rate_per_game,
    }
).sort_values("ownership_rate", ascending=False)

# --- Print a quick summary ---
print("Top 10 games by your ownership rate (owners / non-owners * 100):")
print(ownership_df.head(10))

# --- Plot the top 20 for readability ---
top_n = 20
plt.figure(figsize=(12, 6))
ownership_df["ownership_rate"].head(top_n).plot(kind="bar", edgecolor="black")
plt.title(f"Top {top_n} Games by Ownership Rate (owners / non-owners × 100)")
plt.xlabel("Game")
plt.ylabel("Ownership Rate")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()
