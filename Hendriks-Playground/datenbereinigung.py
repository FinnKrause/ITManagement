import pandas as pd
import numpy as np

# CSV laden
df = pd.read_csv("steam-200k.csv", header=None, names=["user_id", "game", "behavior", "hours", "zero"])

#zero spalte Eliminieren, durch ganzen Datensatz durch nur 0 Werte --> Redundant
df = df.drop(columns=["zero"])

#löschen der zero spalte anzeigen lassen
#print(df.columns)
#check hat funktioniert


#Jedes Tupel, welches im dritten Attribut „purchase“ stehen hat, lässt sich restlos aus anderen Tupeln ableiten und bietet daher keinen Mehrwert für die Analyse 
df = df[df["behavior"] == "play"]

#anzeigen lassen
print(df["behavior"].unique())