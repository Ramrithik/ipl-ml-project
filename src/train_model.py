import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
ARCHIVE_CSV = r"C:\mydesk\archive\IPL.csv"

FORM_WINDOW   = 5
PLAYER_WINDOW = 10
RANDOM_STATE  = 42
ELO_K         = 32
ELO_INIT      = 1500

NAME_MAP = {
    "Delhi Daredevils"            : "Delhi Capitals",
    "Kings XI Punjab"             : "Punjab Kings",
    "Deccan Chargers"             : "Sunrisers Hyderabad",
    "Rising Pune Supergiants"     : "Rising Pune Supergiant",
    "Pune Warriors"               : "Pune Warriors India",
    "Royal Challengers Bangalore" : "Royal Challengers Bengaluru",
}

print("STEP 1: Loading Data")
raw = pd.read_csv(ARCHIVE_CSV, low_memory=False)
for col in ["batting_team","bowling_team","match_won_by","toss_winner"]:
    if col in raw.columns: raw[col] = raw[col].replace(NAME_MAP)

first_balls = raw[raw["innings"] == 1].groupby("match_id").first().reset_index()
match_df = first_balls[["match_id","date","batting_team","bowling_team",
                         "match_won_by","toss_winner","toss_decision",
                         "venue","season","result_type","win_outcome"]].copy()
match_df = match_df.rename(columns={"batting_team":"team1","bowling_team":"team2","match_won_by":"winner","match_id":"id"})
match_df = match_df[match_df["result_type"].isna()].dropna(subset=["winner"]).copy()
match_df["date"] = pd.to_datetime(match_df["date"], format="mixed", dayfirst=True)
match_df = match_df.sort_values("date").reset_index(drop=True)
match_df["team1_won"] = (match_df["winner"] == match_df["team1"]).astype(int)
match_df["toss_winner_is_team1"] = (match_df["toss_winner"] == match_df["team1"]).astype(int)
match_df["toss_decision_bat"] = (match_df["toss_decision"] == "bat").astype(int)

innings1 = raw[raw["innings"] == 1]
i1_stats = innings1.groupby("match_id").agg(target_score=("runs_total", "sum"), target_wickets=("striker_out", "sum")).reset_index().rename(columns={"match_id": "id"})
match_df = match_df.merge(i1_stats, on="id", how="left").fillna(0)
mid_date = match_df[["id","date"]].rename(columns={"id":"match_id"})

print("STEP 2: Player stats")
bat_agg = raw.groupby(["match_id","batter"]).agg(
    batting_team=("batting_team","first"), batter_runs=("batter_runs","sum"), batter_balls=("batter_balls","sum")
).reset_index()
bat_agg["batter_sr"] = bat_agg["batter_runs"] / bat_agg["batter_balls"].clip(lower=1) * 100
bat_agg = bat_agg.merge(mid_date, on="match_id", how="left").dropna(subset=["date"]).sort_values(["batter","date"]).reset_index(drop=True)

bowl_agg = raw.groupby(["match_id","bowler"]).agg(
    bowling_team=("bowling_team","first"), bowler_wickets=("bowler_wicket","sum"),
    bowler_runs_conc=("runs_bowler","sum"), bowler_balls=("valid_ball","sum")
).reset_index()
bowl_agg["bowler_economy"] = bowl_agg["bowler_runs_conc"] / (bowl_agg["bowler_balls"].clip(lower=1)/6)
bowl_agg = bowl_agg.merge(mid_date, on="match_id", how="left").dropna(subset=["date"]).sort_values(["bowler","date"]).reset_index(drop=True)

bat_agg["rolling_avg_runs"] = bat_agg.groupby("batter")["batter_runs"].transform(lambda x: x.shift(1).rolling(PLAYER_WINDOW, min_periods=1).mean())
bat_agg["rolling_avg_sr"]   = bat_agg.groupby("batter")["batter_sr"].transform(lambda x: x.shift(1).rolling(PLAYER_WINDOW, min_periods=1).mean())
bowl_agg["rolling_avg_wkt"]  = bowl_agg.groupby("bowler")["bowler_wickets"].transform(lambda x: x.shift(1).rolling(PLAYER_WINDOW, min_periods=1).mean())
bowl_agg["rolling_avg_econ"] = bowl_agg.groupby("bowler")["bowler_economy"].transform(lambda x: x.shift(1).rolling(PLAYER_WINDOW, min_periods=1).mean())

bat_dict = bat_agg.groupby("match_id").apply(lambda g: g[["batter","batting_team","rolling_avg_runs","rolling_avg_sr"]].to_dict("records")).to_dict()
bowl_dict = bowl_agg.groupby("match_id").apply(lambda g: g[["bowler","bowling_team","rolling_avg_wkt","rolling_avg_econ"]].to_dict("records")).to_dict()

pl_bat = raw.groupby(["match_id", "batting_team"])["batter"].unique().to_dict()
pl_bowl = raw.groupby(["match_id", "bowling_team"])["bowler"].unique().to_dict()

print("STEP 3: Team & Venue stats")
batting_stats = raw.groupby(["match_id","batting_team"]).agg(runs_scored=("runs_total","sum"), balls_faced=("valid_ball","sum")).reset_index().rename(columns={"match_id":"id"})
all_tv = pd.concat([match_df[["team1","venue"]].rename(columns={"team1":"team"}), match_df[["team2","venue"]].rename(columns={"team2":"team"})])
vc = all_tv.groupby(["team","venue"]).size().reset_index(name="count")
home_venue_map = dict(zip(vc.loc[vc.groupby("team")["count"].idxmax()]["team"], vc.loc[vc.groupby("team")["count"].idxmax()]["venue"]))

print("STEP 4: Computing advanced features")
df = match_df.copy()
elo = {t: float(ELO_INIT) for t in pd.concat([df["team1"],df["team2"]]).unique()}

features_rows = []
for idx in range(len(df)):
    row = df.iloc[idx]; team1 = row["team1"]; team2 = row["team2"]; venue = row["venue"]; mid = row["id"]; past = df.iloc[:idx]
    
    t1_elo = elo[team1]; t2_elo = elo[team2]; elo_diff = t1_elo - t2_elo
    t1_all = past[(past["team1"]==team1)|(past["team2"]==team1)]; t2_all = past[(past["team1"]==team2)|(past["team2"]==team2)]
    t1_wr = (t1_all["winner"]==team1).sum()/len(t1_all) if len(t1_all) else 0.5
    t2_wr = (t2_all["winner"]==team2).sum()/len(t2_all) if len(t2_all) else 0.5
    
    v_past = past[past["venue"]==venue]
    t1_v = v_past[(v_past["team1"]==team1)|(v_past["team2"]==team1)]
    t1_venue_wr = (t1_v["winner"]==team1).sum()/len(t1_v) if len(t1_v) else t1_wr
    
    n1 = max(len(t1_all.tail(FORM_WINDOW)),1); n2 = max(len(t2_all.tail(FORM_WINDOW)),1)
    t1_form = (t1_all.tail(FORM_WINDOW)["winner"]==team1).sum()/n1 if len(t1_all) else 0.5
    t2_form = (t2_all.tail(FORM_WINDOW)["winner"]==team2).sum()/n2 if len(t2_all) else 0.5
    is_home_t1 = 1 if home_venue_map.get(team1)==venue else 0
    
    v_bat_wr = 0.5
    if len(v_past):
        v_bat_wins = v_past[v_past["toss_decision"]=="bat"]
        v_bat_wr = (v_bat_wins["winner"] == v_bat_wins["toss_winner"]).mean() if len(v_bat_wins) else 0.5
    
    toss_decision_bat = row["toss_decision_bat"]
    is_optimal_toss = 1 if (v_bat_wr > 0.55 and toss_decision_bat) or (v_bat_wr < 0.45 and not toss_decision_bat) else 0
    
    venue_avg_score = batting_stats[batting_stats["id"].isin(set(v_past["id"]))]["runs_scored"].mean() if len(v_past) else 165.0
    
    def get_xi_features(team, exact_batters, exact_bowlers):
        b_runs = []; b_sr = []; bw_wkt = []; bw_econ = []
        for b in exact_batters:
            p_past = bat_agg[(bat_agg["batter"]==b) & (bat_agg["date"] < row["date"])]
            if len(p_past): b_runs.append(p_past["batter_runs"].mean()); b_sr.append(p_past["batter_sr"].mean())
        for b in exact_bowlers:
            p_past = bowl_agg[(bowl_agg["bowler"]==b) & (bowl_agg["date"] < row["date"])]
            if len(p_past): bw_wkt.append(p_past["bowler_wickets"].mean()); bw_econ.append(p_past["bowler_economy"].mean())
        
        return (np.mean(b_runs) if b_runs else 20.0, np.mean(b_sr) if b_sr else 120.0,
                np.mean(bw_wkt) if bw_wkt else 1.0, np.mean(bw_econ) if bw_econ else 8.0)

    t1_xi_bat = pl_bat.get((mid, team1), [])
    t1_xi_bowl = pl_bowl.get((mid, team1), [])
    t2_xi_bat = pl_bat.get((mid, team2), [])
    t2_xi_bowl = pl_bowl.get((mid, team2), [])

    t1_xi_ba, t1_xi_bs, t1_xi_bw, t1_xi_be = get_xi_features(team1, t1_xi_bat, t1_xi_bowl)
    t2_xi_ba, t2_xi_bs, t2_xi_bw, t2_xi_be = get_xi_features(team2, t2_xi_bat, t2_xi_bowl)
    
    features_rows.append({
        "target_score": row["target_score"], "target_wickets": row["target_wickets"], 
        "target_vs_venue_avg": row["target_score"] - venue_avg_score,
        "toss_winner_is_team1": row["toss_winner_is_team1"], "toss_decision_bat": toss_decision_bat,
        "is_optimal_toss": is_optimal_toss, "venue_bat_wr": v_bat_wr,
        "elo_diff": elo_diff, "team1_overall_wr": t1_wr, "team2_overall_wr": t2_wr,
        "team1_venue_wr": t1_venue_wr, "team1_form": t1_form, "team2_form": t2_form,
        "is_home_team1": is_home_t1, "venue_avg_score": venue_avg_score,
        "t1_xi_bat_avg": t1_xi_ba, "t1_xi_bat_sr": t1_xi_bs, "t1_xi_bowl_wkt": t1_xi_bw, "t1_xi_bowl_econ": t1_xi_be,
        "t2_xi_bat_avg": t2_xi_ba, "t2_xi_bat_sr": t2_xi_bs, "t2_xi_bowl_wkt": t2_xi_bw, "t2_xi_bowl_econ": t2_xi_be,
    })
    
    outcome = row["team1_won"]
    win_out = str(row.get("win_outcome", ""))
    M = 1.0
    if "runs" in win_out:
        try:
            r = int(win_out.split(" ")[0])
            M = 2.0 if r > 50 else (1.5 if r > 20 else 1.0)
        except: pass
    elif "wickets" in win_out:
        try:
            w = int(win_out.split(" ")[0])
            M = 2.0 if w >= 8 else (1.5 if w >= 5 else 1.0)
        except: pass

    e1 = 1.0 / (1.0 + 10.0 ** ((t2_elo - t1_elo)/400.0))
    elo[team1] = t1_elo + ELO_K * M * (outcome - e1)
    elo[team2] = t2_elo + ELO_K * M * ((1-outcome)-(1-e1))

features_df = pd.DataFrame(features_rows).fillna(0)
for col in features_df.columns: df[col] = features_df[col].values

INPLAY_FEATURES = ['target_score', 'target_wickets', 'target_vs_venue_avg', 'elo_diff', 'team1_overall_wr', 'team2_overall_wr', 'team1_venue_wr', 'team1_form', 'team2_form', 'is_home_team1', 't1_xi_bat_avg', 't1_xi_bat_sr', 't1_xi_bowl_wkt', 't1_xi_bowl_econ', 't2_xi_bat_avg', 't2_xi_bat_sr', 't2_xi_bowl_wkt', 't2_xi_bowl_econ', 'venue_avg_score', 'is_optimal_toss', 'venue_bat_wr']

PREMATCH_FEATURES = ['toss_winner_is_team1', 'toss_decision_bat', 'is_optimal_toss', 'venue_bat_wr', 'elo_diff', 'team1_overall_wr', 'team2_overall_wr', 'team1_venue_wr', 'team1_form', 'team2_form', 'is_home_team1', 't1_xi_bat_avg', 't1_xi_bat_sr', 't1_xi_bowl_wkt', 't1_xi_bowl_econ', 't2_xi_bat_avg', 't2_xi_bat_sr', 't2_xi_bowl_wkt', 't2_xi_bowl_econ', 'venue_avg_score']

y = df["team1_won"]
train_mask = df["date"].dt.year < 2023; test_mask = df["date"].dt.year >= 2023
y_train, y_test = y[train_mask], y[test_mask]
pos_w = float((y_train==0).sum()) / float((y_train==1).sum())

def build_stacking_model(X_tr, X_te):
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, scale_pos_weight=pos_w, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    mlp = Pipeline([("sc", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42, early_stopping=True))])
    stack = StackingClassifier(estimators=[("xgb",xgb), ("rf",rf), ("mlp",mlp)], final_estimator=LogisticRegression(class_weight="balanced"), cv=5, n_jobs=-1)
    stack.fit(X_tr, y_train)
    return stack, accuracy_score(y_test, stack.predict(X_te))

print("STEP 5: Training PRE-MATCH Model (with True XI, Toss Synergy & Margin ELO)")
X_pre = df[PREMATCH_FEATURES].copy()
prematch_model, pre_acc = build_stacking_model(X_pre[train_mask], X_pre[test_mask])
print(f"  Pre-Match Model Acc: {pre_acc*100:.2f}%")

print("STEP 6: Training IN-PLAY Model")
X_inp = df[INPLAY_FEATURES].copy()
inplay_model, inp_acc = build_stacking_model(X_inp[train_mask], X_inp[test_mask])
print(f"  In-Play Model Acc: {inp_acc*100:.2f}%")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "prematch_model.pkl"), "wb") as f: pickle.dump(prematch_model, f)
with open(os.path.join(MODEL_DIR, "prematch_features.pkl"), "wb") as f: pickle.dump(PREMATCH_FEATURES, f)
with open(os.path.join(MODEL_DIR, "inplay_model.pkl"), "wb") as f: pickle.dump(inplay_model, f)
with open(os.path.join(MODEL_DIR, "inplay_features.pkl"), "wb") as f: pickle.dump(INPLAY_FEATURES, f)

le_team = LabelEncoder(); le_venue = LabelEncoder()
le_team.fit(pd.concat([df["team1"],df["team2"]]).unique()); le_venue.fit(df["venue"].unique())
with open(os.path.join(MODEL_DIR, "teams_list.pkl"), "wb") as f: pickle.dump(sorted(le_team.classes_.tolist()), f)
with open(os.path.join(MODEL_DIR, "venues_list.pkl"), "wb") as f: pickle.dump(sorted(le_venue.classes_.tolist()), f)
with open(os.path.join(MODEL_DIR, "match_history.pkl"), "wb") as f: pickle.dump(df, f)
with open(os.path.join(MODEL_DIR, "batting_stats.pkl"), "wb") as f: pickle.dump(batting_stats, f)
with open(os.path.join(MODEL_DIR, "home_venues.pkl"), "wb") as f: pickle.dump(home_venue_map, f)
with open(os.path.join(MODEL_DIR, "elo_ratings.pkl"), "wb") as f: pickle.dump(elo, f)

latest_bat  = bat_agg.sort_values("date").groupby("batter").last()[["rolling_avg_runs","rolling_avg_sr","batting_team"]].reset_index()
latest_bowl = bowl_agg.sort_values("date").groupby("bowler").last()[["rolling_avg_wkt","rolling_avg_econ","bowling_team"]].reset_index()

squads_bat = raw.groupby("batting_team")["batter"].unique().to_dict()
squads_bowl = raw.groupby("bowling_team")["bowler"].unique().to_dict()

with open(os.path.join(MODEL_DIR, "latest_bat_stats.pkl"), "wb") as f: pickle.dump(latest_bat, f)
with open(os.path.join(MODEL_DIR, "latest_bowl_stats.pkl"), "wb") as f: pickle.dump(latest_bowl, f)
with open(os.path.join(MODEL_DIR, "team_squad_bat.pkl"), "wb") as f: pickle.dump(squads_bat, f)
with open(os.path.join(MODEL_DIR, "team_squad_bowl.pkl"), "wb") as f: pickle.dump(squads_bowl, f)

print("ALL DONE!")
