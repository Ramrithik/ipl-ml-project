import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

MODEL_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models')
FORM_WINDOW   = 5
STATS_WINDOW  = 10

@st.cache_resource
def load_artifacts():
    def _l(name):
        with open(os.path.join(MODEL_DIR, name), 'rb') as f: return pickle.load(f)
    return (
        _l('prematch_model.pkl'), _l('prematch_features.pkl'),
        _l('inplay_model.pkl'), _l('inplay_features.pkl'),
        _l('teams_list.pkl'), _l('venues_list.pkl'),
        _l('match_history.pkl'), _l('batting_stats.pkl'),
        _l('home_venues.pkl'), _l('elo_ratings.pkl'),
        _l('latest_bat_stats.pkl'), _l('latest_bowl_stats.pkl'),
        _l('team_squad_bat.pkl'), _l('team_squad_bowl.pkl')
    )

(prematch_model, PREMATCH_FEATURES, inplay_model, INPLAY_FEATURES, TEAMS, VENUES,
 HISTORY, BATTING_STATS, HOME_VENUES, ELO,
 LATEST_BAT, LATEST_BOWL, SQUAD_BAT, SQUAD_BOWL) = load_artifacts()

def compute_features(team1, team2, venue, t1_xi, t2_xi, toss_winner=None, toss_decision=None, target_score=None, target_wickets=None):
    hist = HISTORY
    t1e  = ELO.get(team1, 1500.0);  t2e = ELO.get(team2, 1500.0)
    elo_diff = t1e - t2e

    t1a = hist[(hist['team1'] == team1) | (hist['team2'] == team1)]
    t2a = hist[(hist['team1'] == team2) | (hist['team2'] == team2)]
    t1_wr = (t1a['winner'] == team1).sum() / len(t1a) if len(t1a) else 0.5
    t2_wr = (t2a['winner'] == team2).sum() / len(t2a) if len(t2a) else 0.5

    t1v = hist[((hist['team1'] == team1) | (hist['team2'] == team1)) & (hist['venue'] == venue)]
    t1_venue_wr = (t1v['winner'] == team1).sum() / len(t1v) if len(t1v) else t1_wr

    n1 = max(len(t1a.tail(FORM_WINDOW)), 1); n2 = max(len(t2a.tail(FORM_WINDOW)), 1)
    t1_form = (t1a.tail(FORM_WINDOW)['winner'] == team1).sum() / n1
    t2_form = (t2a.tail(FORM_WINDOW)['winner'] == team2).sum() / n2
    is_home_t1 = 1 if HOME_VENUES.get(team1) == venue else 0

    vp = hist[hist['venue'] == venue]
    v_bat_wr = 0.5
    if len(vp):
        v_bat_wins = vp[vp["toss_decision"]=="bat"]
        v_bat_wr = (v_bat_wins["winner"] == v_bat_wins["toss_winner"]).mean() if len(v_bat_wins) else 0.5
        
    v_ids = set(vp['id']) if len(vp) else set()
    vb = BATTING_STATS[BATTING_STATS['id'].isin(v_ids)]
    venue_avg_score = vb['runs_scored'].mean() if len(vb) else 165.0

    def get_xi_agg(xi_list):
        b_r = LATEST_BAT[LATEST_BAT["batter"].isin(xi_list)]
        ba = b_r["rolling_avg_runs"].mean() if len(b_r) else 20.0
        bs = b_r["rolling_avg_sr"].mean() if len(b_r) else 120.0
        bw_r = LATEST_BOWL[LATEST_BOWL["bowler"].isin(xi_list)]
        bw = bw_r["rolling_avg_wkt"].mean() if len(bw_r) else 1.0
        be = bw_r["rolling_avg_econ"].mean() if len(bw_r) else 8.0
        return ba, bs, bw, be

    t1_ba, t1_bs, t1_bw, t1_be = get_xi_agg(t1_xi)
    t2_ba, t2_bs, t2_bw, t2_be = get_xi_agg(t2_xi)

    vals = {
        'elo_diff': elo_diff, 'team1_overall_wr': t1_wr, 'team2_overall_wr': t2_wr,
        'team1_venue_wr': t1_venue_wr, 'team1_form': t1_form, 'team2_form': t2_form,
        'is_home_team1': is_home_t1, 'venue_bat_wr': v_bat_wr, 'venue_avg_score': venue_avg_score,
        't1_xi_bat_avg': t1_ba, 't1_xi_bat_sr': t1_bs, 't1_xi_bowl_wkt': t1_bw, 't1_xi_bowl_econ': t1_be,
        't2_xi_bat_avg': t2_ba, 't2_xi_bat_sr': t2_bs, 't2_xi_bowl_wkt': t2_bw, 't2_xi_bowl_econ': t2_be,
    }

    if target_score is not None:
        vals['target_score'] = target_score
        vals['target_wickets'] = target_wickets
        vals['target_vs_venue_avg'] = target_score - venue_avg_score
        sample = pd.DataFrame([[vals.get(f, 0.0) for f in INPLAY_FEATURES]], columns=INPLAY_FEATURES)
    else:
        toss_decision_bat = 1 if toss_decision == 'bat' else 0
        vals['toss_winner_is_team1'] = 1 if toss_winner == team1 else 0
        vals['toss_decision_bat'] = toss_decision_bat
        vals['is_optimal_toss'] = 1 if (v_bat_wr > 0.55 and toss_decision_bat) or (v_bat_wr < 0.45 and not toss_decision_bat) else 0
        sample = pd.DataFrame([[vals.get(f, 0.0) for f in PREMATCH_FEATURES]], columns=PREMATCH_FEATURES)

    stats = {'t1_elo': t1e, 't2_elo': t2e, 'venue_avg_score': venue_avg_score, 'v_bat_wr': v_bat_wr}
    return sample, stats

st.set_page_config(page_title='Dual IPL Predictor', page_icon='🏏', layout='wide')
import ui_styles
st.title('🏏 Ultimate Dual IPL Predictor')
st.caption('Contains both True XI Pre-Match Model and In-Play Model.')

st.sidebar.header("📋 True Playing XI Selector")
t1_squad = list(set(SQUAD_BAT.get("Chennai Super Kings", [])))
t1_squad.extend(SQUAD_BOWL.get("Chennai Super Kings", []))
t1_squad = sorted(list(set(t1_squad)))

t1 = st.sidebar.selectbox('Team 1', TEAMS, index=TEAMS.index("Chennai Super Kings") if "Chennai Super Kings" in TEAMS else 0)
t2 = st.sidebar.selectbox('Team 2', [t for t in TEAMS if t != t1])

def get_squad(team):
    sq = list(set(SQUAD_BAT.get(team, [])))
    sq.extend(SQUAD_BOWL.get(team, []))
    return sorted(list(set(sq)))

t1_xi = st.sidebar.multiselect(f"{t1} Playing XI", get_squad(t1))
t2_xi = st.sidebar.multiselect(f"{t2} Playing XI", get_squad(t2))

venue = st.sidebar.selectbox('Venue', VENUES)

tab1, tab2 = st.tabs(["🔮 Pre-Match Predictor", "🏃‍♂️ In-Play Predictor"])

with tab1:
    st.subheader("Predict before the match starts")
    c3, c4 = st.columns(2)
    with c3: tw_pre = st.selectbox('Toss Winner', [t1, t2], key='tw_pre')
    with c4: td_pre = st.selectbox('Decision', ['bat', 'field'], key='td_pre')

    if st.button('🔮 Predict Pre-Match', use_container_width=True, type='primary'):
        sample, stats = compute_features(t1, t2, venue, t1_xi, t2_xi, toss_winner=tw_pre, toss_decision=td_pre)
        proba = prematch_model.predict_proba(sample)[0]
        winner = t1 if proba[1] > proba[0] else t2
        st.success(f'🏆 Pre-Match Predicted Winner: **{winner}**')
        ca, cb = st.columns(2)
        with ca:
            st.metric(f'🏏 {t1}', f'{proba[1]*100:.1f}%', delta=f'Elo {stats["t1_elo"]:.0f}')
            st.progress(int(proba[1]*100))
        with cb:
            st.metric(f'🏏 {t2}', f'{proba[0]*100:.1f}%', delta=f'Elo {stats["t2_elo"]:.0f}')
            st.progress(int(proba[0]*100))

with tab2:
    st.subheader("Predict run chase after 1st innings")
    vp = HISTORY[HISTORY['venue'] == venue]
    v_bat_wr = 0.5
    if len(vp):
        v_bat_wins = vp[vp["toss_decision"]=="bat"]
        v_bat_wr = (v_bat_wins["winner"] == v_bat_wins["toss_winner"]).mean() if len(v_bat_wins) else 0.5
    st.info(f"Batting first win rate at {venue} is {v_bat_wr*100:.1f}%")
    c3, c4 = st.columns(2)
    with c3: score_inp = st.number_input('Target Score', min_value=0, max_value=300, value=170, step=1)
    with c4: wkts_inp = st.number_input('Wickets Lost', min_value=0, max_value=10, value=6, step=1)

    if st.button('🏃‍♂️ Predict Run Chase', use_container_width=True, type='primary'):
        sample, stats = compute_features(t1, t2, venue, t1_xi, t2_xi, target_score=score_inp, target_wickets=wkts_inp)
        proba = inplay_model.predict_proba(sample)[0]
        winner = t1 if proba[1] > proba[0] else t2
        st.success(f'🏆 In-Play Predicted Winner: **{winner}**')
        ca, cb = st.columns(2)
        with ca:
            st.metric(f'🏏 {t1} (Defending)', f'{proba[1]*100:.1f}%', delta=f'Elo {stats["t1_elo"]:.0f}')
            st.progress(int(proba[1]*100))
        with cb:
            st.metric(f'🏏 {t2} (Chasing)', f'{proba[0]*100:.1f}%', delta=f'Elo {stats["t2_elo"]:.0f}')
            st.progress(int(proba[0]*100))
