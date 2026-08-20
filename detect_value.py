import pickle
import numpy as np
import pandas as pd
from main import get_football_matches


def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(
            "❌ Le fichier 'model.pkl' est introuvable. Exécute 'python"
            " train_model.py' d'abord."
        )
        return None


def find_value_bets():
    data = load_model()
    if not data:
        return

    model = data["model"]
    stats = data["stats"]
    feature_names = data["features"]

    print("🔍 Récupération des matchs Winamax...")
    df = get_football_matches()

    if df is None or df.empty:
        print("Aucun match récupéré.")
        return

    value_bets = []

    for _, row in df.iterrows():
        match = row["match"]
        if " - " not in match:
            continue

        home_team, away_team = match.split(" - ", 1)

        # Statistique récente des équipes
        h_stats = stats.get(home_team)
        a_stats = stats.get(away_team)

        h_gf = (
            np.mean(h_stats["goals_scored"][-5:])
            if h_stats and len(h_stats["goals_scored"]) > 0
            else 1.3
        )
        h_ga = (
            np.mean(h_stats["goals_conceded"][-5:])
            if h_stats and len(h_stats["goals_conceded"]) > 0
            else 1.3
        )
        a_gf = (
            np.mean(a_stats["goals_scored"][-5:])
            if a_stats and len(a_stats["goals_scored"]) > 0
            else 1.1
        )
        a_ga = (
            np.mean(a_stats["goals_conceded"][-5:])
            if a_stats and len(a_stats["goals_conceded"]) > 0
            else 1.4
        )

        # Formatage sous forme de DataFrame pour respecter le modèle
        features_df = pd.DataFrame(
            [[h_gf, h_ga, a_gf, a_ga]], columns=feature_names
        )

        # Prédiction
        probs = model.predict_proba(features_df)[0]
        prob_N, prob_1, prob_2 = probs[0], probs[1], probs[2]

        # Calcul de l'Expected Value (EV)
        ev_1 = (prob_1 * row["cote_1"]) - 1
        ev_N = (prob_N * row["cote_N"]) - 1
        ev_2 = (prob_2 * row["cote_2"]) - 1

        # Seuil de déclenchement : EV > +3%
        threshold = 0.03

        if ev_1 > threshold:
            value_bets.append({
                "match": match,
                "pari": f"Victoire {home_team}",
                "cote": row["cote_1"],
                "prob_ml_%": round(prob_1 * 100, 1),
                "ev_%": round(ev_1 * 100, 2),
            })
        if ev_N > threshold:
            value_bets.append({
                "match": match,
                "pari": "Match Nul",
                "cote": row["cote_N"],
                "prob_ml_%": round(prob_N * 100, 1),
                "ev_%": round(ev_N * 100, 2),
            })
        if ev_2 > threshold:
            value_bets.append({
                "match": match,
                "pari": f"Victoire {away_team}",
                "cote": row["cote_2"],
                "prob_ml_%": round(prob_2 * 100, 1),
                "ev_%": round(ev_2 * 100, 2),
            })

    results_df = pd.DataFrame(value_bets)

    if not results_df.empty:
        print("\n🚀 VALUE BETS DÉTECTÉS (Espérance de gain positive) :")
        print(
            results_df.sort_values(by="ev_%", ascending=False).to_string(
                index=False
            )
        )
        results_df.to_csv("value_bets.csv", index=False, encoding="utf-8-sig")
        print("\n📁 Résultat exporté dans 'value_bets.csv'")
    else:
        print("\nAucun Value Bet détecté sur les cotes actuelles.")


if __name__ == "__main__":
    find_value_bets()