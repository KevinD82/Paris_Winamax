import json
import re
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def get_football_matches():
    url = "https://www.winamax.fr/paris-sportifs/sports/1"

    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        response = session.get(url, timeout=10)
    except Exception:
        return None

    if response.status_code != 200 or "PRELOADED_STATE" not in response.text:
        return None

    try:
        start_str = "PRELOADED_STATE = "
        start_idx = response.text.find(start_str) + len(start_str)
        sub_html = response.text[start_idx:]
        end_match = re.search(r";(?:var|<\/script>|\n)", sub_html)
        json_str = (
            sub_html[: end_match.start()] if end_match else sub_html.split(";")[0]
        )
        data = json.loads(json_str)
    except Exception:
        return None

    matches = data.get("matches", {})
    bets = data.get("bets", {})
    odds = data.get("odds", {})
    tournaments = data.get("tournaments", {})

    parsed_data = []
    tz_france = timezone(timedelta(hours=2))

    for m_id, m in matches.items():
        if m.get("status") not in ["PREMATCH", "LIVE"]:
            continue

        match_name = m.get("title") or m.get("matchName")
        if not match_name:
            c1, c2 = m.get("competitor1Name", ""), m.get("competitor2Name", "")
            if c1 and c2:
                match_name = f"{c1} - {c2}"
            else:
                continue

        if "MultiFoot" in match_name or "Total" in match_name:
            continue

        tournament_id = str(m.get("tournamentId"))
        tournament_info = tournaments.get(tournament_id, {})
        league_name = (
            tournament_info.get("tournamentName")
            or m.get("tournamentName")
            or "Autre / International"
        )

        match_timestamp = m.get("matchStart") or m.get("startDate")
        if not match_timestamp:
            continue

        ts = int(match_timestamp)
        if ts > 10000000000:
            ts = ts // 1000

        date_dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(
            tz_france
        )

        main_bet_id = str(m.get("mainBetId"))
        bet = bets.get(main_bet_id)
        if not bet or len(bet.get("outcomes", [])) != 3:
            continue

        outcomes = bet.get("outcomes", [])
        c1_val, cN_val, c2_val = (
            odds.get(str(outcomes[0])),
            odds.get(str(outcomes[1])),
            odds.get(str(outcomes[2])),
        )

        if c1_val and cN_val and c2_val:
            parsed_data.append({
                "id": str(m_id),
                "timestamp": ts,
                "league": league_name,
                "match": match_name,
                "date": date_dt.strftime("%d/%m/%Y"),
                "heure": date_dt.strftime("%H:%M"),
                "cote_1": float(c1_val),
                "cote_N": float(cN_val),
                "cote_2": float(c2_val),
            })

    df = pd.DataFrame(parsed_data)
    if not df.empty:
        df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df


def get_match_extra_bets(match_id):
    url = f"https://www.winamax.fr/paris-sportifs/match/{match_id}"
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        res = session.get(url, timeout=10)
    except Exception:
        return {}

    if res.status_code != 200 or "PRELOADED_STATE" not in res.text:
        return {}

    try:
        start_str = "PRELOADED_STATE = "
        start_idx = res.text.find(start_str) + len(start_str)
        sub_html = res.text[start_idx:]
        end_match = re.search(r";(?:var|<\/script>|\n)", sub_html)
        json_str = (
            sub_html[: end_match.start()] if end_match else sub_html.split(";")[0]
        )
        data = json.loads(json_str)
    except Exception:
        return {}

    bets = data.get("bets", {})
    odds = data.get("odds", {})

    extra = {"btts": None, "over_under": [], "scorers": {}}

    EXACT_BTTS_TITLES = {
        "les 2 équipes marquent",
        "les deux équipes marquent",
        "les 2 equipes marquent",
        "les deux equipes marquent",
        "les deux équipes marqueront-elles ?",
        "les 2 équipes marqueront-elles ?",
    }

    for b_id, b_obj in bets.items():
        title = b_obj.get("betTitle", "").strip().lower()

        # 1. BTTS
        if title in EXACT_BTTS_TITLES and not extra["btts"]:
            outs = b_obj.get("outcomes", [])
            if len(outs) == 2:
                c_oui, c_non = odds.get(str(outs[0])), odds.get(str(outs[1]))
                if c_oui and c_non:
                    extra["btts"] = {
                        "Oui": float(c_oui),
                        "Non": float(c_non),
                    }

        # 2. Plus / Moins de buts du match
        elif any(
            keyword in title
            for keyword in ["nombre total de buts", "plus / moins", "nombre de buts"]
        ) and not any(k in title for k in ["1ère", "2ème", "mi-temps", "équipe"]):
            outs = b_obj.get("outcomes", [])
            for o_id in outs:
                o_obj = data.get("outcomes", {}).get(str(o_id), {})
                label = o_obj.get("label", "").strip()
                c_val = odds.get(str(o_id))
                if label and c_val:
                    if not any(item["label"] == label for item in extra["over_under"]):
                        extra["over_under"].append(
                            {"label": label, "cote": float(c_val)}
                        )

        # 3. Buteur / Passeur / Joueur décisif
        else:
            is_buteur = title in [
                "buteur au cours du match",
                "buteur",
                "buteur (temps réglementaire)",
            ]
            is_passeur = "passeur" in title and "décisif" not in title
            is_decisif = "joueur décisif" in title or "décisif (buteur" in title

            if is_buteur or is_passeur or is_decisif:
                bet_type = (
                    "buteur"
                    if is_buteur
                    else ("passeur" if is_passeur else "decisif")
                )
                outs = b_obj.get("outcomes", [])
                for o_id in outs:
                    o_obj = data.get("outcomes", {}).get(str(o_id), {})
                    label = o_obj.get("label", "").strip()
                    c_val = odds.get(str(o_id))
                    if label and c_val and label.lower() not in ["oui", "non"]:
                        if label not in extra["scorers"]:
                            extra["scorers"][label] = {
                                "buteur": None,
                                "passeur": None,
                                "decisif": None,
                            }
                        extra["scorers"][label][bet_type] = float(c_val)

    return extra