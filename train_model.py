import io
import pickle
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

LEAGUES = ["F1", "E0", "SP1", "I1", "D1", "N1", "B1"]
SEASONS = ["2324", "2425", "2526"]

TEAM_ALIASES = {
    "Paris SG": "PSG",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Bay Munich": "Bayern Munich",
    "Leverkusen": "Bayer Leverkusen",
    "Inter": "Inter Milan",
    "Milan": "AC Milan",
}


def download_data():
    all_dfs = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for season in SEASONS:
        for league in LEAGUES:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code == 200:
                    df = pd.read_csv(io.StringIO(res.text))
                    cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
                    if all(c in df.columns for c in cols):
                        df = df[cols].dropna()
                        all_dfs.append(df)
            except Exception:
                continue

    return (
        pd.concat(all_dfs, ignore_index=True)
        if all_dfs
        else pd.DataFrame()
    )


def train():
    print("⏳ Téléchargement des données...")
    df = download_data()

    if df.empty:
        print("❌ Impossible de charger les données.")
        return

    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_ALIASES)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_ALIASES)

    stats = {}
    teams = set(df["HomeTeam"].unique()).union(set(df["AwayTeam"].unique()))
    for t in teams:
        stats[t] = {"goals_scored": [], "goals_conceded": []}

    X, y_1n2, y_btts = [], [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = float(row["FTHG"]), float(row["FTAG"])

        h_s, a_s = stats.get(home, {}), stats.get(away, {})

        h_gf = (
            np.mean(h_s["goals_scored"][-5:])
            if h_s and len(h_s["goals_scored"]) > 0
            else 1.3
        )
        h_ga = (
            np.mean(h_s["goals_conceded"][-5:])
            if h_s and len(h_s["goals_conceded"]) > 0
            else 1.3
        )
        a_gf = (
            np.mean(a_s["goals_scored"][-5:])
            if a_s and len(a_s["goals_scored"]) > 0
            else 1.1
        )
        a_ga = (
            np.mean(a_s["goals_conceded"][-5:])
            if a_s and len(a_s["goals_conceded"]) > 0
            else 1.4
        )

        X.append([h_gf, h_ga, a_gf, a_ga])

        ftr = row["FTR"]
        y_1n2.append(1 if ftr == "H" else (2 if ftr == "A" else 0))
        y_btts.append(1 if (hg > 0 and ag > 0) else 0)

        stats[home]["goals_scored"].append(hg)
        stats[home]["goals_conceded"].append(ag)
        stats[away]["goals_scored"].append(ag)
        stats[away]["goals_conceded"].append(hg)

    features = ["h_gf", "h_ga", "a_gf", "a_ga"]
    X_df = pd.DataFrame(X, columns=features)

    model_1n2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_1n2.fit(X_df, y_1n2)

    model_btts = RandomForestClassifier(n_estimators=100, random_state=42)
    model_btts.fit(X_df, y_btts)

    data_to_save = {
        "model_1n2": model_1n2,
        "model_btts": model_btts,
        "stats": stats,
        "features": features,
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(data_to_save, f)

    print("🎉 Entraînement réussi !")


if __name__ == "__main__":
    train()