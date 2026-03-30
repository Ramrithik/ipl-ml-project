import streamlit as st
import pandas as pd
import pickle
import os

MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
FORM_WINDOW  = 5
STATS_WINDOW = 10

@st.cache_resource
def load_artifacts():
    def _load(name):
        with open(os.path.join(MODEL_DIR, name), 'rb') as f:
            return pickle.load(f)

    return (
        _load('win_model.pkl'),
        _load('le_team.pkl'),
        _load('le_venue.pkl'),
        _load('teams_list.pkl'),
        _load('venues_list.pkl'),
        _load('features.pkl'),
        _load('match_history.pkl'),
        _load('batting_stats.pkl'),
        _load('bowling_stats.pkl'),
        _load('home_venues.pkl'),
    )


(
    model, le_team, le_venue,
    TEAMS, VENUES, FEATURES,
    HISTORY, BATTING_STATS, BOWLING_STATS, HOME_VENUES,
) = load_artifacts()

def compute_features(team1, team2, venue, toss_winner, toss_decision):
    """
    Build the feature vector for a new match using REAL historical stats
    instead of hard-coded 0.5 placeholders.
    """

    t1_enc = le_team.transform([team1])[0]
    t2_enc = le_team.transform([team2])[0]
    v_enc  = le_venue.transform([venue])[0]

    toss_is_t1 = 1 if toss_winner   == team1 else 0
    toss_bat   = 1 if toss_decision == 'bat' else 0

    hist = HISTORY  

    
    t1_all = hist[(hist['team1'] == team1) | (hist['team2'] == team1)]
    t1_wins = (t1_all['winner'] == team1).sum()
    t1_overall_wr = t1_wins / len(t1_all) if len(t1_all) else 0.5

    t2_all = hist[(hist['team1'] == team2) | (hist['team2'] == team2)]
    t2_wins = (t2_all['winner'] == team2).sum()
    t2_overall_wr = t2_wins / len(t2_all) if len(t2_all) else 0.5

    
    h2h = hist[
        ((hist['team1'] == team1) & (hist['team2'] == team2)) |
        ((hist['team1'] == team2) & (hist['team2'] == team1))
    ]
    h2h_t1_wins = (h2h['winner'] == team1).sum()
    h2h_wr = h2h_t1_wins / len(h2h) if len(h2h) else 0.5
    t1_venue = hist[
        ((hist['team1'] == team1) | (hist['team2'] == team1)) &
        (hist['venue'] == venue)
    ]
    t1_venue_wins = (t1_venue['winner'] == team1).sum()
    t1_venue_wr = t1_venue_wins / len(t1_venue) if len(t1_venue) else t1_overall_wr

    t1_recent = t1_all.tail(FORM_WINDOW)
    t1_form = (t1_recent['winner'] == team1).sum() / len(t1_recent) if len(t1_recent) else 0.5

    t2_recent = t2_all.tail(FORM_WINDOW)
    t2_form = (t2_recent['winner'] == team2).sum() / len(t2_recent) if len(t2_recent) else 0.5

    is_home_t1 = 1 if HOME_VENUES.get(team1) == venue else 0
    is_home_t2 = 1 if HOME_VENUES.get(team2) == venue else 0

    t1_past_ids = set(t1_all['id'])
    t2_past_ids = set(t2_all['id'])

    def _batting_avg(team, ids):
        rows = BATTING_STATS[
            BATTING_STATS['match_id'].isin(ids) & (BATTING_STATS['batting_team'] == team)
        ].tail(STATS_WINDOW)
        if len(rows) == 0:
            return 0.0, 0.0
        return rows['runs_scored'].mean(), rows['run_rate'].mean()

    def _bowling_avg(team, ids):
        rows = BOWLING_STATS[
            BOWLING_STATS['match_id'].isin(ids) & (BOWLING_STATS['bowling_team'] == team)
        ].tail(STATS_WINDOW)
        if len(rows) == 0:
            return 0.0
        return rows['wickets_taken'].mean()

    t1_avg_runs, t1_avg_rr = _batting_avg(team1, t1_past_ids)
    t1_avg_wkt             = _bowling_avg(team1, t1_past_ids)
    t2_avg_runs, t2_avg_rr = _batting_avg(team2, t2_past_ids)
    t2_avg_wkt             = _bowling_avg(team2, t2_past_ids)
    sample = pd.DataFrame([[
        t1_enc, t2_enc, v_enc,
        toss_is_t1, toss_bat,
        t1_overall_wr, t2_overall_wr,
        h2h_wr, t1_venue_wr,
        t1_form, t2_form,
        is_home_t1, is_home_t2,
        t1_avg_runs, t1_avg_rr, t1_avg_wkt,
        t2_avg_runs, t2_avg_rr, t2_avg_wkt,
    ]], columns=FEATURES)

    stats_display = {
        'team1_overall_wr' : t1_overall_wr,
        'team2_overall_wr' : t2_overall_wr,
        'h2h_win_rate'     : h2h_wr,
        'team1_venue_wr'   : t1_venue_wr,
        'team1_form'       : t1_form,
        'team2_form'       : t2_form,
        'is_home_team1'    : bool(is_home_t1),
        'is_home_team2'    : bool(is_home_t2),
        'h2h_total'        : len(h2h),
        't1_avg_runs'      : t1_avg_runs,
        't2_avg_runs'      : t2_avg_runs,
    }

    return sample, stats_display

st.set_page_config(page_title='IPL Win Predictor', page_icon='🏏', layout='centered')
st.title('🏏 IPL Win Probability Predictor')
st.caption('Powered by XGBoost · Rolling stats · Home advantage · Recent form · IPL 2008–2024')
st.divider()

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox('Team 1', TEAMS)
with col2:
    team2 = st.selectbox('Team 2', [t for t in TEAMS if t != team1])

venue = st.selectbox('Venue', VENUES)

col3, col4 = st.columns(2)
with col3:
    toss_winner   = st.selectbox('Toss winner', [team1, team2])
with col4:
    toss_decision = st.selectbox('Toss decision', ['bat', 'field'])

st.divider()

if st.button(' Predict Winner', use_container_width=True, type='primary'):
    try:
        sample, stats = compute_features(team1, team2, venue, toss_winner, toss_decision)

        prob   = model.predict_proba(sample)[0]
        t1_pct = round(prob[1] * 100, 1)
        t2_pct = round(prob[0] * 100, 1)
        winner = team1 if prob[1] > 0.5 else team2

        st.success(f'🏆 Predicted winner: **{winner}**')

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label=team1, value=f'{t1_pct}%')
            st.progress(int(t1_pct))
        with col_b:
            st.metric(label=team2, value=f'{t2_pct}%')
            st.progress(int(t2_pct))
        st.divider()
        st.subheader('Key Factors')

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**{team1}**")
            st.write(f"Overall Win Rate: {stats['team1_overall_wr']*100:.1f}%")
            st.write(f"Recent Form (last {FORM_WINDOW}): {stats['team1_form']*100:.1f}%")
            st.write(f"Venue Win Rate: {stats['team1_venue_wr']*100:.1f}%")
            st.write(f"Avg Runs (last {STATS_WINDOW}): {stats['t1_avg_runs']:.1f}")
            st.write(f"Home Advantage: {' Yes' if stats['is_home_team1'] else '❌ No'}")

        with sc2:
            st.markdown(f"**{team2}**")
            st.write(f"Overall Win Rate: {stats['team2_overall_wr']*100:.1f}%")
            st.write(f"Recent Form (last {FORM_WINDOW}): {stats['team2_form']*100:.1f}%")
            st.write(f"Avg Runs (last {STATS_WINDOW}): {stats['t2_avg_runs']:.1f}")
            st.write(f"Home Advantage: {' Yes' if stats['is_home_team2'] else '❌ No'}")

        st.divider()
        st.caption(
            f"H2H record ({stats['h2h_total']} matches): "
            f"{team1} wins {stats['h2h_win_rate']*100:.0f}% of encounters"
        )

    except Exception as e:
        st.error(f'Prediction failed: {e}')
