# ===============================================================
# Analyse: Was macht ein gutes Spiel aus?
# Dateien: steam-200k.csv  +  metacritic_games.csv
# (metacritic_games.csv ist inhaltlich identisch zur früheren .xls)
# ===============================================================

import os
import re
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from difflib import get_close_matches

# ---------------------------
# Konfiguration
# ---------------------------
STEAM_PATH = "steam-200k.csv"
META_PATH = "metacritic_games.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Hilfsfunktionen
# ---------------------------


def normalize_cols(cols):
    """Spaltennamen vereinheitlichen."""
    norm = []
    for c in cols:
        c = str(c)
        c = c.strip().lower()
        c = re.sub(r"\s+", "_", c)
        c = c.replace("#", "num").replace("%", "pct")
        c = re.sub(r"[^a-z0-9_]", "", c)
        norm.append(c)
    return norm


def read_csv_safely(
    path, expected_like=None, encoding_candidates=("utf-8", "utf-8-sig", "latin-1")
):
    """
    Liest CSV robust ein:
    - testet verschiedene Encodings
    - versucht header=0, fallback header=None + erste Zeile als Header
    - normalisiert Spaltennamen
    """
    last_err = None
    for enc in encoding_candidates:
        try:
            # Versuch mit Header
            df = pd.read_csv(path, encoding=enc)
            df.columns = normalize_cols(df.columns)
            if expected_like:
                found = any(k in df.columns for k in expected_like)
                if not found:
                    # vielleicht war keine Header-Zeile, also nochmal ohne Header
                    df2 = pd.read_csv(path, encoding=enc, header=None)
                    # erste Zeile als Spalten nehmen
                    new_cols = normalize_cols(df2.iloc[0].astype(str).tolist())
                    df2 = df2.iloc[1:].copy()
                    df2.columns = new_cols
                    df2 = df2.reset_index(drop=True)
                    df = df2
                    # hier keine weitere Prüfung, nehmen wie es ist
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV konnte nicht gelesen werden. Letzter Fehler: {last_err}")


def coerce_numeric(series):
    """Konvertiert typische Bewertungsspalten (inkl. 'tbd', Strings) robust zu float."""
    if series is None:
        return None
    s = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"tbd": np.nan, "nan": np.nan, "": np.nan})
    )
    # Prozent entfernen, Komma in Punkt
    s = s.str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def detect_title_column(df):
    """
    Ermittelt die Spalte mit Spiel-Titeln.
    Mögliche Kandidaten werden in Reihenfolge geprüft.
    """
    candidates = [
        "title",
        "name",
        "game",
        "game_title",
        "gametitle",
        "product",
        "spiel",
        "spielname",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # heuristisch: Spalte mit größter Textlänge/Varianz
    text_like = [c for c in df.columns if df[c].dtype == object]
    if text_like:
        # wähle die Spalte mit den meisten einzigartigen Werte
        uniq_counts = {c: df[c].astype(str).nunique() for c in text_like}
        return max(uniq_counts, key=uniq_counts.get)
    raise KeyError("Konnte keine plausible Titel-Spalte in Metacritic-Daten finden.")


def clean_title_series(s):
    """Titel string-normalisieren für Match/Join."""
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"®|™|©", "", regex=True)
    )


def parse_year(series):
    """Versucht Jahre aus Datum/Freitext zu extrahieren."""
    s = series.astype(str)
    # direkte Zahl (z.B. '2016')
    year = s.str.extract(r"(\d{4})", expand=False)
    year = pd.to_numeric(year, errors="coerce")
    return year


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"📈 Gespeichert: {path}")


def print_head(df, n=3, title=None):
    if title:
        print(f"\n--- {title} ---")
    print(df.head(n))


# ---------------------------
# 1) Steam laden & aufbereiten
# ---------------------------
if not os.path.exists(STEAM_PATH):
    raise FileNotFoundError(f"Datei nicht gefunden: {STEAM_PATH}")

steam = pd.read_csv(STEAM_PATH, header=None)
# Expected: user_id, game, action, value, other
if steam.shape[1] < 5:
    raise ValueError("Unerwartetes Format in steam-200k.csv (weniger als 5 Spalten).")

steam.columns = ["user_id", "game", "action", "value", "other"]
steam = steam[steam["action"].isin(["play", "purchase"])].copy()

# Playtime pro Spiel (Durchschnitt)
play_df = (
    steam[steam["action"] == "play"]
    .groupby("game", as_index=False)["value"]
    .mean()
    .rename(columns={"value": "avg_playtime_hours"})
)
play_df["game_clean"] = clean_title_series(play_df["game"])

print_head(play_df, title="Steam: Playtime (erste Zeilen)")

# ---------------------------
# 2) Metacritic CSV laden
# ---------------------------
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Datei nicht gefunden: {META_PATH}")

# Wir erwarten Spalten wie 'title'/'name', 'metascore', 'userscore', 'genre', 'release_date' o.ä.
meta_expected_like = ("title", "name", "metascore", "userscore", "genre", "release")
meta = read_csv_safely(META_PATH, expected_like=meta_expected_like)

print(f"\nMetacritic-Spalten: {list(meta.columns)}")

# Titelspalte erkennen
title_col = detect_title_column(meta)
meta["title"] = meta[title_col].astype(str)
meta["title_clean"] = clean_title_series(meta["title"])

# Numeric Bewertungsspalten robust konvertieren
for c in ["metascore", "critic_score", "critic_score_avg"]:
    if c in meta.columns and "metascore" not in meta.columns:
        meta.rename(columns={c: "metascore"}, inplace=True)

for c in ["userscore", "user_score"]:
    if c in meta.columns and "userscore" not in meta.columns:
        meta.rename(columns={c: "userscore"}, inplace=True)

meta["metascore"] = (
    coerce_numeric(meta["metascore"]) if "metascore" in meta.columns else np.nan
)
meta["userscore"] = (
    coerce_numeric(meta["userscore"]) if "userscore" in meta.columns else np.nan
)

# Release-Jahr
release_col = None
for c in ["release_date", "release", "released", "release_year", "jahr", "year"]:
    if c in meta.columns:
        release_col = c
        break
if release_col is not None:
    if release_col == "release_year":
        meta["release_year"] = coerce_numeric(meta["release_year"])
    else:
        meta["release_year"] = parse_year(meta[release_col])
else:
    meta["release_year"] = np.nan

# Genre-Feldvereinheitlichung (optional)
if "genre" not in meta.columns:
    # Fallback Suche
    for c in ["genres", "category", "categories"]:
        if c in meta.columns:
            meta.rename(columns={c: "genre"}, inplace=True)
            break

print_head(
    meta[[title_col, "title", "metascore", "userscore", "release_year"]],
    title="Metacritic (Kernspalten)",
)

# ---------------------------
# 3) Exakter Join
# ---------------------------
merged = pd.merge(
    play_df,
    meta,
    left_on="game_clean",
    right_on="title_clean",
    how="inner",
    suffixes=("_steam", "_meta"),
).copy()

print(f"\n✅ Exakter Join: {len(merged)} gemeinsame Spiele")

# ---------------------------
# 4) Optional: Fuzzy-Join für nicht gematchte (konservativ)
# ---------------------------
if len(merged) < 100:  # fuzzy nur bei schwachem Match; anpassbar
    remaining_steam = play_df[~play_df["game_clean"].isin(merged["game_clean"])].copy()
    meta_titles = meta["title_clean"].dropna().unique().tolist()

    approx_matches = []
    for row in remaining_steam.itertuples(index=False):
        gc = row.game_clean
        # einfache Heuristik – nur sehr nahe Matches zulassen
        m = get_close_matches(gc, meta_titles, n=1, cutoff=0.93)
        if m:
            approx_matches.append((gc, m[0]))
    if approx_matches:
        fuzzy_map = pd.DataFrame(approx_matches, columns=["game_clean", "title_clean"])
        fuzzy_join = pd.merge(remaining_steam, fuzzy_map, on="game_clean", how="inner")
        fuzzy_join = pd.merge(
            fuzzy_join,
            meta,
            on="title_clean",
            how="inner",
            suffixes=("_steam", "_meta"),
        )
        merged = pd.concat([merged, fuzzy_join], ignore_index=True).drop_duplicates(
            subset=["game_clean"]
        )

        print(f"🔎 Fuzzy-Join ergänzt: Gesamt nun {len(merged)} Spiele")

# ---------------------------
# 5) Feature Engineering
# ---------------------------
# Engagement-Score (Kombi aus Qualität & Nutzung)
# log1p dämpft Ausreißer in den Spielzeiten
ms = (
    merged["metascore"].fillna(0).clip(lower=0, upper=100)
    if "metascore" in merged.columns
    else 0
)
merged["engagement_score"] = (ms / 100.0) * np.log1p(
    merged["avg_playtime_hours"].clip(lower=0)
)

# einfache Qualitätskennziffer
if "userscore" in merged.columns:
    us_norm = merged["userscore"].fillna(0).clip(lower=0, upper=10) / 10.0
else:
    us_norm = 0.0
merged["quality_score"] = 0.6 * (ms / 100.0) + 0.4 * us_norm

# ---------------------------
# 6) Ausgaben/Tabellen
# ---------------------------
# Kernspalten fürs Reporting
keep_cols = [
    "title",
    "game",
    "avg_playtime_hours",
    "metascore",
    "userscore",
    "release_year",
    "genre",
    "engagement_score",
    "quality_score",
]
keep_cols = [c for c in keep_cols if c in merged.columns]
report = merged[keep_cols].copy()

# Sortierte Toplisten
top_engagement = report.sort_values("engagement_score", ascending=False).head(20)
top_quality = report.sort_values("quality_score", ascending=False).head(20)

top_engagement.to_csv(os.path.join(OUT_DIR, "top20_engagement.csv"), index=False)
top_quality.to_csv(os.path.join(OUT_DIR, "top20_quality.csv"), index=False)

print_head(top_engagement, title="Top 20 – Engagement Score")
print_head(top_quality, title="Top 20 – Quality Score")

# ---------------------------
# 7) Visualisierungen (nur matplotlib)
# ---------------------------

plt.figure(figsize=(9, 6))
plt.scatter(merged["metascore"], merged["avg_playtime_hours"], alpha=0.6)
plt.title("Metascore vs. durchschnittliche Spielzeit")
plt.xlabel("Metascore (0–100)")
plt.ylabel("Ø Spielzeit (h, log)")
plt.yscale("log")
plt.grid(True, alpha=0.3)
save_fig("scatter_metascore_vs_playtime.png")
plt.close()

# Userscore vs. Spielzeit (falls vorhanden)
if "userscore" in merged.columns and merged["userscore"].notna().any():
    plt.figure(figsize=(9, 6))
    plt.scatter(merged["userscore"], merged["avg_playtime_hours"], alpha=0.6)
    plt.title("Userscore vs. durchschnittliche Spielzeit")
    plt.xlabel("Userscore (0–10)")
    plt.ylabel("Ø Spielzeit (h, log)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    save_fig("scatter_userscore_vs_playtime.png")
    plt.close()

# Genre-Balken (Durchschnittswerte)
if "genre" in merged.columns and merged["genre"].notna().any():
    genre_agg = (
        merged.groupby("genre")
        .agg(
            metascore_mean=("metascore", "mean"),
            userscore_mean=("userscore", "mean"),
            playtime_mean=("avg_playtime_hours", "mean"),
            engagement_mean=("engagement_score", "mean"),
        )
        .sort_values("engagement_mean", ascending=False)
    )
    # nur die Top 15 Genres, um Plot lesbar zu halten
    genre_agg = genre_agg.head(15)

    # Playtime
    plt.figure(figsize=(11, 6))
    plt.bar(genre_agg.index.astype(str), genre_agg["playtime_mean"])
    plt.title("Ø Spielzeit nach Genre (Top 15 nach Engagement)")
    plt.ylabel("Ø Spielzeit (h)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig("bar_playtime_by_genre.png")
    plt.close()

    # Metascore
    if genre_agg["metascore_mean"].notna().any():
        plt.figure(figsize=(11, 6))
        plt.bar(genre_agg.index.astype(str), genre_agg["metascore_mean"])
        plt.title("Ø Metascore nach Genre (Top 15 nach Engagement)")
        plt.ylabel("Ø Metascore")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        save_fig("bar_metascore_by_genre.png")
        plt.close()

# Zeitliche Entwicklung (Release-Jahr)
if "release_year" in merged.columns and merged["release_year"].notna().any():
    year_agg = (
        merged.dropna(subset=["release_year"])
        .groupby("release_year")
        .agg(
            metascore_mean=("metascore", "mean"),
            userscore_mean=("userscore", "mean"),
            playtime_mean=("avg_playtime_hours", "mean"),
        )
        .sort_index()
    )
    plt.figure(figsize=(10, 6))
    plt.plot(year_agg.index, year_agg["metascore_mean"], marker="o", label="Metascore")
    if (
        "userscore_mean" in year_agg.columns
        and year_agg["userscore_mean"].notna().any()
    ):
        plt.plot(
            year_agg.index, year_agg["userscore_mean"], marker="o", label="Userscore"
        )
    plt.title("Bewertungen im Zeitverlauf")
    plt.xlabel("Erscheinungsjahr")
    plt.ylabel("Ø Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig("line_scores_over_time.png")
    plt.close()

# Korrelationen (numeric)
numeric = merged.select_dtypes(include=[np.number]).copy()
if numeric.shape[1] >= 2:
    corr = numeric.corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Korrelationen (numeric features)")
    save_fig("heatmap_correlations.png")
    plt.close()

# ---------------------------
# 8) Kurzes Fazit in Datei
# ---------------------------
insights = {
    "joined_games": int(len(merged)),
    "top_engagement_head": top_engagement.head(5).to_dict(orient="records"),
    "top_quality_head": top_quality.head(5).to_dict(orient="records"),
    "notes": [
        "Metascore korreliert meist positiv mit Spielzeit (log-Skala empfohlen).",
        "Engagement Score kombiniert Bewertung & Nutzung und ist robust ggü. Ausreißern.",
        "Genres mit hoher Engagement-Mean sind starke Kandidaten für 'gute' Spiele.",
        "Zeitreihen zeigen ggf. Trends bei Scores (z. B. Generationswechsel, Plattformwellen).",
    ],
}
with open(os.path.join(OUT_DIR, "insights.json"), "w", encoding="utf-8") as f:
    json.dump(insights, f, ensure_ascii=False, indent=2)

print(f"\n📝 Insights gespeichert: {os.path.join(OUT_DIR, 'insights.json')}")
print(
    f"📂 CSV-Exports: {os.path.join(OUT_DIR, 'top20_engagement.csv')}  /  {os.path.join(OUT_DIR, 'top20_quality.csv')}"
)
print("✅ Fertig.")
