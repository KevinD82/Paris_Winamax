import pickle
import numpy as np
import pandas as pd
import streamlit as st
from main import get_football_matches, get_match_extra_bets

st.set_page_config(page_title="Winamax ML Dashboard", layout="wide")


@st.cache_resource
def load_ml_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def load_matches():
    return get_football_matches()


st.title("⚽ Winamax Betting & ML Analyzer")

model_data = load_ml_model()
df_matches = load_matches()

if df_matches is None or df_matches.empty:
    st.error("❌ Impossible de récupérer les matchs depuis Winamax.")
else:
    if "league" not in df_matches.columns:
        st.cache_data.clear()
        df_matches = get_football_matches()
        if "league" not in df_matches.columns:
            df_matches["league"] = "Autre / International"

    st.sidebar.header("🔍 Filtres & Sélection")

    available_dates = list(dict.fromkeys(df_matches["date"].tolist()))
    selected_date = st.sidebar.selectbox(
        "📅 1. Choisissez une date :", available_dates, index=0
    )

    df_filtered_date = df_matches[df_matches["date"] == selected_date]

    available_leagues = ["Tous les championnats"] + sorted(
        list(set(df_filtered_date["league"].tolist()))
    )
    selected_league = st.sidebar.selectbox(
        "🏆 2. Choisissez un championnat :", available_leagues, index=0
    )

    if selected_league != "Tous les championnats":
        df_filtered_league = df_filtered_date[
            df_filtered_date["league"] == selected_league
        ].copy()
    else:
        df_filtered_league = df_filtered_date.copy()

    df_filtered_league["display_name"] = (
        df_filtered_league["heure"] + "  |  " + df_filtered_league["match"]
    )

    selected_display = st.sidebar.selectbox(
        f"⚽ 3. Matchs ({len(df_filtered_league)} disponibles) :",
        df_filtered_league["display_name"].tolist(),
    )

    match_data = df_filtered_league[
        df_filtered_league["display_name"] == selected_display
    ].iloc[0]

    st.caption(f"🏆 **Compétition :** {match_data['league']}")
    st.header(f"⚽ {match_data['match']}")

    col_d, col_h = st.columns(2)
    col_d.metric("📅 Date", match_data["date"])
    col_h.metric("⏰ Coup d'envoi", match_data["heure"])

    st.divider()

    extra_bets = get_match_extra_bets(match_data["id"])

    prob_1, prob_N, prob_2 = None, None, None
    prob_btts_oui, prob_btts_non = None, None

    if model_data and " - " in match_data["match"]:
        home_team, away_team = match_data["match"].split(" - ", 1)
        stats = model_data.get("stats", {})

        h_stats = next(
            (
                v
                for k, v in stats.items()
                if k.lower() in home_team.lower()
                or home_team.lower() in k.lower()
            ),
            None,
        )
        a_stats = next(
            (
                v
                for k, v in stats.items()
                if k.lower() in away_team.lower()
                or away_team.lower() in k.lower()
            ),
            None,
        )

        if h_stats and a_stats:
            features = model_data["features"]
            h_gf = np.mean(h_stats["goals_scored"][-5:])
            h_ga = np.mean(h_stats["goals_conceded"][-5:])
            a_gf = np.mean(a_stats["goals_scored"][-5:])
            a_ga = np.mean(a_stats["goals_conceded"][-5:])

            X_match = pd.DataFrame(
                [[h_gf, h_ga, a_gf, a_ga]], columns=features
            )

            probs_1n2 = model_data["model_1n2"].predict_proba(X_match)[0]
            prob_N, prob_1, prob_2 = probs_1n2[0], probs_1n2[1], probs_1n2[2]

            probs_btts = model_data["model_btts"].predict_proba(X_match)[0]
            prob_btts_non, prob_btts_oui = probs_btts[0], probs_btts[1]

    tab1, tab2 = st.tabs(["📊 Cotes 1N2 & ML", "🎯 BTTS, Buts & Buteurs"])

    with tab1:
        st.subheader("📈 Pari Principal 1N2")
        c1, cN, c2 = st.columns(3)

        ev_1 = (prob_1 * match_data["cote_1"]) - 1 if prob_1 else 0
        ev_N = (prob_N * match_data["cote_N"]) - 1 if prob_N else 0
        ev_2 = (prob_2 * match_data["cote_2"]) - 1 if prob_2 else 0

        c1.metric(
            f"1 - {match_data['match'].split(' - ')[0]}",
            match_data["cote_1"],
            delta=f"Prob: {prob_1*100:.1f}% | EV: {ev_1*100:+.1f}%"
            if prob_1
            else None,
        )
        cN.metric(
            "N - Nul",
            match_data["cote_N"],
            delta=f"Prob: {prob_N*100:.1f}% | EV: {ev_N*100:+.1f}%"
            if prob_N
            else None,
        )
        c2.metric(
            f"2 - {match_data['match'].split(' - ')[1] if ' - ' in match_data['match'] else 'Ext'}",
            match_data["cote_2"],
            delta=f"Prob: {prob_2*100:.1f}% | EV: {ev_2*100:+.1f}%"
            if prob_2
            else None,
        )

        st.divider()

        # --- SECTION RÉCAPITULATIF DES MEILLEURS PARIS ---
        st.subheader("🌟 Récapitulatif des meilleurs paris recommandés")

        best_recommendations = []

        # 1. Analyse 1N2
        if prob_1:
            best_1n2_ev = max(ev_1, ev_N, ev_2)
            best_1n2_prob = (
                prob_1
                if best_1n2_ev == ev_1
                else (prob_N if best_1n2_ev == ev_N else prob_2)
            )
            if best_1n2_ev > 0.05 and best_1n2_prob >= 0.40:
                best_team = (
                    home_team
                    if best_1n2_ev == ev_1
                    else ("Match Nul" if best_1n2_ev == ev_N else away_team)
                )
                best_cote = (
                    match_data["cote_1"]
                    if best_1n2_ev == ev_1
                    else (
                        match_data["cote_N"]
                        if best_1n2_ev == ev_N
                        else match_data["cote_2"]
                    )
                )
                best_recommendations.append({
                    "Type": "Résultat 1N2",
                    "Pari": f"Victoire {best_team}"
                    if best_team != "Match Nul"
                    else "Match Nul",
                    "Cote": best_cote,
                    "Probabilité": f"{best_1n2_prob*100:.1f}%",
                    "EV (Value)": f"+{best_1n2_ev*100:.1f}%",
                })

        # 2. Analyse BTTS
        btts_odds = extra_bets.get("btts")
        if btts_odds and prob_btts_oui:
            cote_b_oui = btts_odds.get("Oui")
            cote_b_non = btts_odds.get("Non")
            if cote_b_oui:
                ev_b_oui = (prob_btts_oui * cote_b_oui) - 1
                if ev_b_oui > 0.05:
                    best_recommendations.append({
                        "Type": "Les 2 équipes marquent",
                        "Pari": "Oui",
                        "Cote": cote_b_oui,
                        "Probabilité": f"{prob_btts_oui*100:.1f}%",
                        "EV (Value)": f"+{ev_b_oui*100:.1f}%",
                    })
            if cote_b_non:
                ev_b_non = (prob_btts_non * cote_b_non) - 1
                if ev_b_non > 0.05:
                    best_recommendations.append({
                        "Type": "Les 2 équipes marquent",
                        "Pari": "Non",
                        "Cote": cote_b_non,
                        "Probabilité": f"{prob_btts_non*100:.1f}%",
                        "EV (Value)": f"+{ev_b_non*100:.1f}%",
                    })

        # 3. Analyse Buteurs principaux
        scorers_dict = extra_bets.get("scorers", {})
        if scorers_dict:
            best_scorer = None
            max_p_prob = 0
            for pl, b_data in scorers_dict.items():
                c_b = b_data.get("buteur")
                if c_b and isinstance(c_b, float):
                    p_impl = 1.0 / c_b
                    if p_impl > max_p_prob:
                        max_p_prob = p_impl
                        best_scorer = (pl, c_b, p_impl)

            if best_scorer and best_scorer[2] >= 0.35:
                best_recommendations.append({
                    "Type": "Buteur le plus probable",
                    "Pari": best_scorer[0],
                    "Cote": best_scorer[1],
                    "Probabilité Implicite": f"{best_scorer[2]*100:.1f}%",
                    "EV (Value)": "N/A (Proba Bookmaker)",
                })

        if best_recommendations:
            df_recap = pd.DataFrame(best_recommendations)
            st.dataframe(df_recap, use_container_width=True, hide_index=True)
        else:
            st.info(
                "ℹ️ Aucun Value Bet majeur à forte espérance de gain détecté pour ce match."
            )

    with tab2:
        st.subheader("⚽ Les 2 équipes marquent (BTTS)")
        btts_odds = extra_bets.get("btts")

        col_b_oui, col_b_non = st.columns(2)

        cote_b_oui = btts_odds["Oui"] if btts_odds else "N/A"
        cote_b_non = btts_odds["Non"] if btts_odds else "N/A"

        ev_b_oui = (
            (prob_btts_oui * cote_b_oui - 1)
            if (prob_btts_oui and isinstance(cote_b_oui, float))
            else None
        )
        ev_b_non = (
            (prob_btts_non * cote_b_non - 1)
            if (prob_btts_non and isinstance(cote_b_non, float))
            else None
        )

        col_b_oui.metric(
            "Oui",
            f"Cote: {cote_b_oui}",
            delta=f"IA: {prob_btts_oui*100:.1f}% | EV: {ev_b_oui*100:+.1f}%"
            if ev_b_oui is not None
            else (f"IA: {prob_btts_oui*100:.1f}%" if prob_btts_oui else None),
        )

        col_b_non.metric(
            "Non",
            f"Cote: {cote_b_non}",
            delta=f"IA: {prob_btts_non*100:.1f}% | EV: {ev_b_non*100:+.1f}%"
            if ev_b_non is not None
            else (f"IA: {prob_btts_non*100:.1f}%" if prob_btts_non else None),
        )

        st.divider()

        # --- TABLEAU NOMBRE DE BUTS AVEC COLORATION CORRIGÉE ---
        st.subheader("📊 Nombre total de buts (+ / -)")
        ou_list = extra_bets.get("over_under", [])

        if ou_list:
            parsed_ou = {}
            for item in ou_list:
                lbl = item["label"]
                cote = item["cote"]
                if "Plus de " in lbl:
                    val = lbl.replace("Plus de ", "").strip()
                    parsed_ou.setdefault(val, {})["Plus"] = cote
                elif "Moins de " in lbl:
                    val = lbl.replace("Moins de ", "").strip()
                    parsed_ou.setdefault(val, {})["Moins"] = cote

            if parsed_ou:
                rows = []
                for val, cotes in sorted(
                    parsed_ou.items(),
                    key=lambda x: float(x[0].replace(",", "."))
                    if x[0].replace(",", ".").replace(".", "", 1).isdigit()
                    else 99,
                ):
                    c_plus = cotes.get("Plus")
                    c_moins = cotes.get("Moins")

                    txt_plus = (
                        f"{c_plus:.2f} ({100/c_plus:.1f}%)"
                        if c_plus
                        else "-"
                    )
                    txt_moins = (
                        f"{c_moins:.2f} ({100/c_moins:.1f}%)"
                        if c_moins
                        else "-"
                    )

                    rows.append({
                        "Seuil": f"{val} buts",
                        "Plus de (Cote / Prob)": txt_plus,
                        "Moins de (Cote / Prob)": txt_moins,
                        "_p_plus": (100 / c_plus) if c_plus else 0,
                        "_p_moins": (100 / c_moins) if c_moins else 0,
                    })

                df_ou = pd.DataFrame(rows)

                def highlight_max_prob_ou(row):
                    styles = pd.Series("", index=row.index)
                    p_p = row["_p_plus"]
                    p_m = row["_p_moins"]
                    if p_p > p_m and p_p > 0:
                        styles["Plus de (Cote / Prob)"] = (
                            "background-color: #d4edda; color: #155724;"
                            " text-align: center;"
                        )
                        styles["Moins de (Cote / Prob)"] = "text-align: center;"
                    elif p_m > p_p and p_m > 0:
                        styles["Plus de (Cote / Prob)"] = "text-align: center;"
                        styles["Moins de (Cote / Prob)"] = (
                            "background-color: #d4edda; color: #155724;"
                            " text-align: center;"
                        )
                    else:
                        styles["Plus de (Cote / Prob)"] = "text-align: center;"
                        styles["Moins de (Cote / Prob)"] = "text-align: center;"
                    return styles

                styled_ou = df_ou.style.apply(highlight_max_prob_ou, axis=1)

                col_ou_box, _ = st.columns([3, 2])
                with col_ou_box:
                    st.dataframe(
                        styled_ou,
                        column_order=[
                            "Seuil",
                            "Plus de (Cote / Prob)",
                            "Moins de (Cote / Prob)",
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("Format des cotes non reconnu.")
        else:
            st.info("Cotes Plus/Moins indisponibles.")

        st.divider()

        # --- TABLEAU BUTEURS AVEC COLORATION CORRIGÉE ---
        st.subheader("👟 Buteurs & Joueurs décisifs")
        scorers_dict = extra_bets.get("scorers", {})

        if scorers_dict:
            rows_scorers = []
            for player, bets_p in scorers_dict.items():
                c_b = bets_p.get("buteur")
                c_p = bets_p.get("passeur")
                c_d = bets_p.get("decisif")

                txt_b = (
                    f"{c_b:.2f} ({100/c_b:.1f}%)"
                    if (c_b and isinstance(c_b, float))
                    else "-"
                )
                txt_p = (
                    f"{c_p:.2f} ({100/c_p:.1f}%)"
                    if (c_p and isinstance(c_p, float))
                    else "-"
                )
                txt_d = (
                    f"{c_d:.2f} ({100/c_d:.1f}%)"
                    if (c_d and isinstance(c_d, float))
                    else "-"
                )

                rows_scorers.append({
                    "Joueur": player,
                    "Buteur": txt_b,
                    "Passeur": txt_p,
                    "Décisif": txt_d,
                    "_p_b": (100 / c_b) if isinstance(c_b, float) else 0,
                    "_p_p": (100 / c_p) if isinstance(c_p, float) else 0,
                    "_p_d": (100 / c_d) if isinstance(c_d, float) else 0,
                })

            df_scorers = pd.DataFrame(rows_scorers)

            def highlight_max_scorer(row):
                styles = pd.Series("", index=row.index)
                probs = {
                    "Buteur": row["_p_b"],
                    "Passeur": row["_p_p"],
                    "Décisif": row["_p_d"],
                }
                max_col = max(probs, key=probs.get)
                max_val = probs[max_col]

                for col in ["Buteur", "Passeur", "Décisif"]:
                    if probs[col] == max_val and max_val > 0:
                        styles[col] = (
                            "background-color: #d4edda; color: #155724;"
                            " text-align: center;"
                        )
                    else:
                        styles[col] = "text-align: center;"
                return styles

            styled_scorers = df_scorers.style.apply(
                highlight_max_scorer, axis=1
            )

            col_sc_box, _ = st.columns([4, 2])
            with col_sc_box:
                st.dataframe(
                    styled_scorers,
                    column_order=["Joueur", "Buteur", "Passeur", "Décisif"],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Cotes des buteurs indisponibles.")