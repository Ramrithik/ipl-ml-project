import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

FORM_WINDOW  = 5   
STATS_WINDOW = 10   
RANDOM_STATE = 42
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)
matches    = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))
deliveries = pd.read_csv(os.path.join(DATA_DIR, 'deliveries.csv'))
print(f"  Matches shape   : {matches.shape}")
print(f"  Deliveries shape: {deliveries.shape}")
print("\n" + "=" * 60)
print("STEP 2: Cleaning team names")
print("=" * 60)

name_map = {
    'Delhi Daredevils'             : 'Delhi Capitals',
    'Kings XI Punjab'              : 'Punjab Kings',
    'Deccan Chargers'              : 'Sunrisers Hyderabad',
    'Rising Pune Supergiants'      : 'Rising Pune Supergiant',
    'Pune Warriors'                : 'Pune Warriors India',
    'Royal Challengers Bangalore'  : 'Royal Challengers Bengaluru',
}

for col in ['team1', 'team2', 'winner', 'toss_winner']:
    if col in matches.columns:
        matches[col] = matches[col].replace(name_map)

for col in ['batting_team', 'bowling_team']:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].replace(name_map)
df = matches.dropna(subset=['winner']).copy()
if 'result' in df.columns:
    df = df[df['result'] != 'tie'].copy()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['team1_won']            = (df['winner']      == df['team1']).astype(int)
df['toss_winner_is_team1'] = (df['toss_winner'] == df['team1']).astype(int)
df['toss_decision_bat']    = (df['toss_decision'] == 'bat').astype(int)

print(f"  Usable matches: {len(df)}")
print(f"  Date range    : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Unique teams  : {sorted(df['team1'].unique())}")
print("\n" + "=" * 60)
print("STEP 3: Aggregating deliveries to match-level stats")
print("=" * 60)

batting_stats = deliveries.groupby(['match_id', 'batting_team']).agg(
    runs_scored = ('total_runs', 'sum'),
    balls_faced = ('ball', 'count'),
    fours       = ('batsman_runs', lambda x: (x == 4).sum()),
    sixes       = ('batsman_runs', lambda x: (x == 6).sum()),
).reset_index()

batting_stats['overs_faced'] = batting_stats['balls_faced'] / 6.0
batting_stats['run_rate']    = batting_stats['runs_scored'] / batting_stats['overs_faced'].clip(lower=0.1)
bowling_stats = deliveries.groupby(['match_id', 'bowling_team']).agg(
    wickets_taken = ('is_wicket', 'sum'),
).reset_index()

match_dates = df[['id', 'date']].rename(columns={'id': 'match_id'})
batting_stats = batting_stats.merge(match_dates, on='match_id', how='left').sort_values('date').reset_index(drop=True)
bowling_stats = bowling_stats.merge(match_dates, on='match_id', how='left').sort_values('date').reset_index(drop=True)

print(f"  Batting stats rows : {len(batting_stats)}")
print(f"  Bowling stats rows : {len(bowling_stats)}")

print("\n" + "=" * 60)
print("STEP 4: Detecting home venues from data")
print("=" * 60)

all_team_venue = pd.concat([
    df[['team1', 'venue']].rename(columns={'team1': 'team'}),
    df[['team2', 'venue']].rename(columns={'team2': 'team'}),
])
venue_counts     = all_team_venue.groupby(['team', 'venue']).size().reset_index(name='count')
home_venues_df   = venue_counts.loc[venue_counts.groupby('team')['count'].idxmax()][['team', 'venue']]
home_venue_map   = dict(zip(home_venues_df['team'], home_venues_df['venue']))

for team, venue in sorted(home_venue_map.items()):
    print(f"  {team:36s} : {venue}")

print("\n" + "=" * 60)
print("STEP 5: Computing rolling features (leak-free)")
print("=" * 60)
start_time = time.time()


def team_wins_in(filtered_df, team):
    """Count wins for `team` in the given filtered match dataframe."""
    return (filtered_df['winner'] == team).sum()


def get_rolling_batting(bat_df, match_ids, team, window):
    """Average runs scored and run rate over last `window` batting innings."""
    team_bat = bat_df[
        bat_df['match_id'].isin(match_ids) & (bat_df['batting_team'] == team)
    ].tail(window)
    if len(team_bat) == 0:
        return 0.0, 0.0
    return team_bat['runs_scored'].mean(), team_bat['run_rate'].mean()


def get_rolling_bowling(bowl_df, match_ids, team, window):
    """Average wickets taken over last `window` bowling innings."""
    team_bowl = bowl_df[
        bowl_df['match_id'].isin(match_ids) & (bowl_df['bowling_team'] == team)
    ].tail(window)
    if len(team_bowl) == 0:
        return 0.0
    return team_bowl['wickets_taken'].mean()


features_rows = []

for idx in range(len(df)):
    row   = df.iloc[idx]
    team1 = row['team1']
    team2 = row['team2']
    venue = row['venue']

    past = df.iloc[:idx]

    t1_all = past[(past['team1'] == team1) | (past['team2'] == team1)]
    t1_overall_wr = team_wins_in(t1_all, team1) / len(t1_all) if len(t1_all) else 0.5

    t2_all = past[(past['team1'] == team2) | (past['team2'] == team2)]
    t2_overall_wr = team_wins_in(t2_all, team2) / len(t2_all) if len(t2_all) else 0.5

    h2h = past[
        ((past['team1'] == team1) & (past['team2'] == team2)) |
        ((past['team1'] == team2) & (past['team2'] == team1))
    ]
    h2h_wr = team_wins_in(h2h, team1) / len(h2h) if len(h2h) else 0.5

    t1_venue = past[
        ((past['team1'] == team1) | (past['team2'] == team1)) &
        (past['venue'] == venue)
    ]
    t1_venue_wr = team_wins_in(t1_venue, team1) / len(t1_venue) if len(t1_venue) else t1_overall_wr

    t1_recent = t1_all.tail(FORM_WINDOW)
    t1_form = team_wins_in(t1_recent, team1) / len(t1_recent) if len(t1_recent) else 0.5

    t2_recent = t2_all.tail(FORM_WINDOW)
    t2_form = team_wins_in(t2_recent, team2) / len(t2_recent) if len(t2_recent) else 0.5

    is_home_t1 = 1 if home_venue_map.get(team1) == venue else 0
    is_home_t2 = 1 if home_venue_map.get(team2) == venue else 0

    t1_past_ids = set(t1_all['id'])
    t2_past_ids = set(t2_all['id'])

    t1_avg_runs, t1_avg_rr = get_rolling_batting(batting_stats, t1_past_ids, team1, STATS_WINDOW)
    t1_avg_wkt             = get_rolling_bowling(bowling_stats, t1_past_ids, team1, STATS_WINDOW)

    t2_avg_runs, t2_avg_rr = get_rolling_batting(batting_stats, t2_past_ids, team2, STATS_WINDOW)
    t2_avg_wkt             = get_rolling_bowling(bowling_stats, t2_past_ids, team2, STATS_WINDOW)

    features_rows.append({
        'team1_overall_wr' : t1_overall_wr,
        'team2_overall_wr' : t2_overall_wr,
        'h2h_win_rate'     : h2h_wr,
        'team1_venue_wr'   : t1_venue_wr,
        'team1_form'       : t1_form,
        'team2_form'       : t2_form,
        'is_home_team1'    : is_home_t1,
        'is_home_team2'    : is_home_t2,
        'team1_avg_runs'   : t1_avg_runs,
        'team1_avg_rr'     : t1_avg_rr,
        'team1_avg_wkt'    : t1_avg_wkt,
        'team2_avg_runs'   : t2_avg_runs,
        'team2_avg_rr'     : t2_avg_rr,
        'team2_avg_wkt'    : t2_avg_wkt,
    })

    if (idx + 1) % 250 == 0:
        print(f"  Processed {idx + 1}/{len(df)} matches...")

features_df = pd.DataFrame(features_rows)
for col in features_df.columns:
    df[col] = features_df[col].values

elapsed = time.time() - start_time
print(f"  Done - {len(features_df.columns)} rolling features in {elapsed:.1f}s")


print("\n" + "=" * 60)
print("STEP 6: Label-encoding teams & venues")
print("=" * 60)

le_team  = LabelEncoder()
le_venue = LabelEncoder()

all_teams = pd.concat([df['team1'], df['team2']]).unique()
le_team.fit(all_teams)
le_venue.fit(df['venue'])

def safe_transform(le, series):
    known = set(le.classes_)
    return series.apply(lambda x: le.transform([x])[0] if x in known else -1)

df['team1_enc'] = safe_transform(le_team,  df['team1'])
df['team2_enc'] = safe_transform(le_team,  df['team2'])
df['venue_enc'] = safe_transform(le_venue, df['venue'])

print(f"  Teams : {len(le_team.classes_)}")
print(f"  Venues: {len(le_venue.classes_)}")


print("\n" + "=" * 60)
print("STEP 7: Feature matrix & time-based train/test split")
print("=" * 60)

FEATURES = [
    'team1_enc', 'team2_enc', 'venue_enc',
    'toss_winner_is_team1', 'toss_decision_bat',
    'team1_overall_wr', 'team2_overall_wr',
    'h2h_win_rate', 'team1_venue_wr',
    'team1_form', 'team2_form',
    'is_home_team1', 'is_home_team2',
    'team1_avg_runs', 'team1_avg_rr', 'team1_avg_wkt',
    'team2_avg_runs', 'team2_avg_rr', 'team2_avg_wkt',
]

X = df[FEATURES].copy().fillna(0.5)
y = df['team1_won']


split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"  Total features: {len(FEATURES)}")
print(f"  Training      : {len(X_train)} matches (up to {df['date'].iloc[split_idx - 1].date()})")
print(f"  Testing       : {len(X_test)} matches (from {df['date'].iloc[split_idx].date()})")
print(f"  Train win rate: {y_train.mean()*100:.1f}%")
print(f"  Test  win rate: {y_test.mean()*100:.1f}%")

print("\n" + "=" * 60)
print("STEP 8: Training multiple models + ensemble")
print("=" * 60)
tune_start = time.time()

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n  [A] Tuning XGBoost...")
xgb_param_grid = {
    'n_estimators'    : [100, 200, 400],
    'max_depth'       : [2, 3, 4],
    'learning_rate'   : [0.01, 0.05, 0.1],
    'subsample'       : [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'min_child_weight': [1, 3, 5],
}

tscv = TimeSeriesSplit(n_splits=5)
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0),
    xgb_param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"      Best CV: {xgb_grid.best_score_*100:.2f}%  |  Test: {xgb_acc*100:.2f}%")
print(f"      Params: {xgb_grid.best_params_}")

print("\n  [B] Tuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    rf_param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"      Best CV: {rf_grid.best_score_*100:.2f}%  |  Test: {rf_acc*100:.2f}%")


print("\n  [C] Training Logistic Regression...")
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
lr_param_grid = {'lr__C': [0.01, 0.1, 1, 10]}
lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=0)
lr_grid.fit(X_train, y_train)
lr_model = lr_grid.best_estimator_
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
print(f"      Best CV: {lr_grid.best_score_*100:.2f}%  |  Test: {lr_acc*100:.2f}%")


print("\n  [D] Tuning Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8],
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    gb_param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)
gb_grid.fit(X_train, y_train)
gb_model = gb_grid.best_estimator_
gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
print(f"      Best CV: {gb_grid.best_score_*100:.2f}%  |  Test: {gb_acc*100:.2f}%")


print("\n  [E] Building Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', lr_model),
        ('gb', gb_model),
    ],
    voting='soft',
)
ensemble.fit(X_train, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"      Ensemble Test Accuracy: {ens_acc*100:.2f}%")

tune_elapsed = time.time() - tune_start
print(f"\n  Total tuning time: {tune_elapsed:.1f}s")

results = {
    'XGBoost': (xgb_model, xgb_acc),
    'RandomForest': (rf_model, rf_acc),
    'LogisticRegression': (lr_model, lr_acc),
    'GradientBoosting': (gb_model, gb_acc),
    'Ensemble': (ensemble, ens_acc),
}

print("\n  Model comparison:")
for name, (mdl, acc) in sorted(results.items(), key=lambda x: -x[1][1]):
    marker = " <-- BEST" if acc == max(v[1] for v in results.values()) else ""
    print(f"    {name:24s}  {acc*100:.2f}%{marker}")

best_name = max(results, key=lambda k: results[k][1])
model = results[best_name][0]
best_acc = results[best_name][1]
print(f"\n  >> Selected: {best_name} ({best_acc*100:.2f}%)")
print("\n" + "=" * 60)
print("STEP 9: Final evaluation")
print("=" * 60)

preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)

print(f"\n  >> Test Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, preds, target_names=['Team2 wins', 'Team1 wins']))


if hasattr(model, 'feature_importances_'):
    feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("  Feature importance:")
    for fname, fimp in feat_imp.items():
        bar = '#' * int(fimp * 40)
        print(f"    {fname:24s} {fimp:.4f}  {bar}")

print("\n" + "=" * 60)
print("STEP 10: Saving artifacts to models/")
print("=" * 60)

os.makedirs(MODEL_DIR, exist_ok=True)

artifacts = {
    'win_model.pkl'    : model,
    'features.pkl'     : FEATURES,
    'le_team.pkl'      : le_team,
    'le_venue.pkl'     : le_venue,
    'teams_list.pkl'   : sorted(le_team.classes_.tolist()),
    'venues_list.pkl'  : sorted(le_venue.classes_.tolist()),
    'home_venues.pkl'  : home_venue_map,
    'batting_stats.pkl': batting_stats,
    'bowling_stats.pkl': bowling_stats,
    'match_history.pkl': df[['id', 'date', 'team1', 'team2', 'venue', 'winner', 'team1_won']].copy(),
}

for filename, obj in artifacts.items():
    path = os.path.join(MODEL_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  [OK] {filename}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
