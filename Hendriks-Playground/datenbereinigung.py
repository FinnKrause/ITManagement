import pandas as pd
import numpy as np

# CSV laden
df = pd.read_csv("steam-200k.csv", header=None, names=["user_id", "game", "behavior", "hours", "zero"])

#zero spalte Eliminieren, durch ganzen Datensatz durch nur 0 Werte --> Redundant
df = df.drop(columns=["zero"])

#löschen der zero spalte anzeigen lassen
#print(df.columns)
#check hat funktioniert


def count_purchase_play(df):
    anzahl_purchase = (df["behavior"] == "purchase").sum()
    anzahl_play = (df["behavior"] == "play").sum()
    return anzahl_purchase, anzahl_play

# Anwendung:
anz_purchase, anz_play = count_purchase_play(df)
print("Anzahl 'purchase'-Zeilen:", anz_purchase)
print("Anzahl 'play'-Zeilen:", anz_play)
print (anz_play + anz_purchase)

def filter_by_game(df, game_name):
    return df[df["game"] == game_name]

spiel_df = filter_by_game(df, "Hyperdimension Neptunia Re;Birth1")

unique_users = df["user_id"].nunique()
print("\nAnzahl verschiedener User:", unique_users)

unique_games = df["game"].nunique()
print("Anzahl verschiedener Spiele:", unique_games)