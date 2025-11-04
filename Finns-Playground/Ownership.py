import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "steam-200k.csv", header=None, names=["UserID", "Game", "Event", "Hours", "Zeros"]
)

gameToCheck = "Dota 2"

haveGame = df[(df["Event"] == "purchase") & (df["Game"] == gameToCheck)]
all_users = df["UserID"].unique()
owners = haveGame["UserID"].unique()
not_owners = np.setdiff1d(all_users, owners)

print(f"{len(owners)} players have {gameToCheck}.")
print(f"{len(not_owners)} players do not have {gameToCheck}.")

print(
    f"The ratio for the game {gameToCheck} is: {int((len(owners)*100)/len(not_owners))}%"
)
