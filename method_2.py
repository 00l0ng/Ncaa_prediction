
# Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score

# Feature Engineering & Data Processing
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Tournament Simulation & Bracket Modeling
import itertools
import networkx as nx
# Data Path
data_path = r'kaggle-ncaa\march-machine-learning-mania-2025/'
# Game Results (Regular Season & Tournament)
df_regular_season_compact = pd.read_csv(data_path + "WRegularSeasonCompactResults.csv")
df_regular_season_detailed = pd.read_csv(data_path + "WRegularSeasonDetailedResults.csv")
df_tourney_compact = pd.read_csv(data_path + "WNCAATourneyCompactResults.csv")#tourney 锦标赛
df_tourney_detailed = pd.read_csv(data_path + "WNCAATourneyDetailedResults.csv")

# Tournament Seeding
df_seeds = pd.read_csv(data_path + "WNCAATourneySeeds.csv")


# Team Metadata
df_teams = pd.read_csv(data_path + "WTeams.csv")
df_team_spellings = pd.read_csv(data_path + "WTeamSpellings.csv", encoding="ISO-8859-1")

# Tournament Bracket Structure
df_tourney_slots = pd.read_csv(data_path + "WNCAATourneySlots.csv")


# Display basic info about datasets
print("Regular Season Compact Results:\n", df_regular_season_compact.head(), "\n")
print("NCAA Tournament Compact Results:\n", df_tourney_compact.head(), "\n")
print("Tournament Seeds:\n", df_seeds.head(), "\n")
print("Teams:\n", df_teams.head(), "\n")
print("Tournament Slots:\n", df_tourney_slots.head(), "\n")

# Check missing values in all datasets
datasets = {
    "Regular Season Compact": df_regular_season_compact,
    "Regular Season Detailed": df_regular_season_detailed,
    "Tournament Compact": df_tourney_compact,
    "Tournament Detailed": df_tourney_detailed,
    "Tournament Seeds": df_seeds,
    "Teams": df_teams,
    "Team Spellings": df_team_spellings,
    "Tournament Slots": df_tourney_slots,
}

# Display missing Values in each dataset
for name, df in datasets.items():
    print(f"Missing Values in {name}:\n", df.isnull().sum(), "\n")

# Load the dataset
regular_season_detailed = df_regular_season_detailed
current_year = 2025
recent_years = range(current_year - 5, current_year)  # 2020-2024
recent_data = regular_season_detailed[regular_season_detailed["Season"].isin(recent_years)]
# Compute team-level statistics
winning_stats = recent_data.groupby("WTeamID").agg({
    "WScore": ["mean", "sum"],
    "WFGM": "mean", "WFGA": "mean", "WFGM3": "mean", "WFGA3": "mean",
    "WFTM": "mean", "WFTA": "mean", "WOR": "mean", "WDR": "mean",
    "WAst": "mean", "WTO": "mean", "WStl": "mean", "WBlk": "mean", "WPF": "mean"
}).reset_index()


# Rename columns for clarity
winning_stats.columns = ["TeamID", "AvgPointsScored", "TotalPointsScored",
                         "AvgFGM", "AvgFGA", "AvgFGM3", "AvgFGA3",
                         "AvgFTM", "AvgFTA", "AvgOReb", "AvgDReb",
                         "AvgAssists", "AvgTurnovers", "AvgSteals",
                         "AvgBlocks", "AvgFouls"]
# 计算每个球队的获胜次数
win_count = recent_data.groupby("WTeamID").size().reset_index(name="WinCount")
# 将获胜次数合并到统计结果中
winning_stats = winning_stats.merge(win_count, left_on="TeamID", right_on="WTeamID", how="left")
winning_stats.drop(columns=["WTeamID"], inplace=True)

# Compute losing team stats similarly
losing_stats = recent_data.groupby("LTeamID").agg({
    "LScore": ["mean", "sum"],
    "LFGM": "mean", "LFGA": "mean", "LFGM3": "mean", "LFGA3": "mean",
    "LFTM": "mean", "LFTA": "mean", "LOR": "mean", "LDR": "mean",
    "LAst": "mean", "LTO": "mean", "LStl": "mean", "LBlk": "mean", "LPF": "mean"
}).reset_index()

# Rename columns for clarity
losing_stats.columns = ["TeamID", "AvgPointsAllowed", "TotalPointsAllowed",
                        "AvgFGM_Allowed", "AvgFGA_Allowed", "AvgFGM3_Allowed", "AvgFGA3_Allowed",
                        "AvgFTM_Allowed", "AvgFTA_Allowed", "AvgOReb_Allowed", "AvgDReb_Allowed",
                        "AvgAssists_Allowed", "AvgTurnovers_Allowed", "AvgSteals_Allowed",
                        "AvgBlocks_Allowed", "AvgFouls_Allowed"]
# 计算每个球队的获胜次数
Los_count = recent_data.groupby("LTeamID").size().reset_index(name="LosCount")
# 将获胜次数合并到统计结果中
losing_stats = losing_stats.merge(Los_count, left_on="TeamID", right_on="LTeamID", how="left")
losing_stats.drop(columns=["LTeamID"], inplace=True)

# Merge winning and losing stats to get full team stats
team_stats = pd.merge(winning_stats, losing_stats, on="TeamID", how="outer").fillna(0)

# Compute Win Percentage
team_stats["WinRate_score"] = team_stats["TotalPointsScored"] / (team_stats["TotalPointsScored"] + team_stats["TotalPointsAllowed"])
team_stats["WinRate_count"] =team_stats["WinCount"] / (team_stats["WinCount"] + team_stats["LosCount"])
# Save the processed features
team_stats.to_csv(r"\kaggle-ncaa\MTeam_Stats.csv", index=False)



# Load tournament seed data
tourney_seeds = df_seeds
current_year = 2025
recent_years = range(current_year - 5, current_year)  # 2020-2024
tourney_seeds = tourney_seeds[tourney_seeds["Season"].isin(recent_years)]
# Extract numerical seed
tourney_seeds["SeedValue"] = tourney_seeds["Seed"].apply(lambda x: int(x[1:3]))

# Save processed seeds
#种子编号越低，表示队伍或选手的实力越强。
tourney_seeds.to_csv(r"\kaggle-ncaa\Processed_Seeds.csv", index=False)





# Merge team stats with seeds
final_data = pd.merge(team_stats, tourney_seeds, on="TeamID", how="left")


# Fill missing values
final_data.fillna(0, inplace=True)

# Save the final feature dataset
final_data.to_csv(r"\kaggle-ncaa\WFinal_Feature_Dataset.csv", index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score

# 加载数据
data_path = r'\kaggle-ncaa\march-machine-learning-mania-2025/'
df_regular_season_detailed = pd.read_csv(data_path + "WRegularSeasonDetailedResults.csv")
df_seeds = pd.read_csv(data_path + "WNCAATourneySeeds.csv")


# 提取2019年-2023年的数据特征
def extract_features(years):
    recent_data = df_regular_season_detailed[df_regular_season_detailed["Season"].isin(years)]

    # 计算球队的统计数据
    winning_stats = recent_data.groupby("WTeamID").agg({
        "WScore": ["mean", "sum"],
        "WFGM": "mean", "WFGA": "mean", "WFGM3": "mean", "WFGA3": "mean",
        "WFTM": "mean", "WFTA": "mean", "WOR": "mean", "WDR": "mean",
        "WAst": "mean", "WTO": "mean", "WStl": "mean", "WBlk": "mean", "WPF": "mean"
    }).reset_index()
    winning_stats.columns = ["TeamID", "AvgPointsScored", "TotalPointsScored",
                             "AvgFGM", "AvgFGA", "AvgFGM3", "AvgFGA3",
                             "AvgFTM", "AvgFTA", "AvgOReb", "AvgDReb",
                             "AvgAssists", "AvgTurnovers", "AvgSteals",
                             "AvgBlocks", "AvgFouls"]
    win_count = recent_data.groupby("WTeamID").size().reset_index(name="WinCount")
    winning_stats = winning_stats.merge(win_count, left_on="TeamID", right_on="WTeamID", how="left")
    winning_stats.drop(columns=["WTeamID"], inplace=True)

    losing_stats = recent_data.groupby("LTeamID").agg({
        "LScore": ["mean", "sum"],
        "LFGM": "mean", "LFGA": "mean", "LFGM3": "mean", "LFGA3": "mean",
        "LFTM": "mean", "LFTA": "mean", "LOR": "mean", "LDR": "mean",
        "LAst": "mean", "LTO": "mean", "LStl": "mean", "LBlk": "mean", "LPF": "mean"
    }).reset_index()
    losing_stats.columns = ["TeamID", "AvgPointsAllowed", "TotalPointsAllowed",
                            "AvgFGM_Allowed", "AvgFGA_Allowed", "AvgFGM3_Allowed", "AvgFGA3_Allowed",
                            "AvgFTM_Allowed", "AvgFTA_Allowed", "AvgOReb_Allowed", "AvgDReb_Allowed",
                            "AvgAssists_Allowed", "AvgTurnovers_Allowed", "AvgSteals_Allowed",
                            "AvgBlocks_Allowed", "AvgFouls_Allowed"]
    Los_count = recent_data.groupby("LTeamID").size().reset_index(name="LosCount")
    losing_stats = losing_stats.merge(Los_count, left_on="TeamID", right_on="LTeamID", how="left")
    losing_stats.drop(columns=["LTeamID"], inplace=True)

    team_stats = pd.merge(winning_stats, losing_stats, on="TeamID", how="outer").fillna(0)
    team_stats["WinRate_score"] = team_stats["TotalPointsScored"] / (
                team_stats["TotalPointsScored"] + team_stats["TotalPointsAllowed"])
    team_stats["WinRate_count"] = team_stats["WinCount"] / (team_stats["WinCount"] + team_stats["LosCount"])

    # 处理种子数据
    tourney_seeds = df_seeds[df_seeds["Season"].isin(years)]
    tourney_seeds["SeedValue"] = tourney_seeds["Seed"].apply(lambda x: int(x[1:3]))

    # 合并数据
    final_data = pd.merge(team_stats, tourney_seeds, on="TeamID", how="left")
    final_data.fillna(0, inplace=True)

    return final_data


# 提取2019年-2023年的数据特征
features_2019_2023 = extract_features(range(2019, 2024))

# 提取2024年的胜率
features_2024 = extract_features([2024])
features_2024["WinRate"] = features_2024["WinRate_count"]

# 训练XGBoost模型
train = features_2019_2023.drop(columns=["Season", "WinRate_count", "WinRate_score","Seed","Season"])
test=features_2024[['TeamID','WinRate_count']]
merged_data = pd.merge(test, train, on="TeamID", how="inner")
X=merged_data.drop(columns=['TeamID','WinRate_count'])
y=merged_data["WinRate_count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor()
model.fit(X_train, y_train)

# 使用2020年-2024年的数据预测2025年的胜率
features_2020_2024 = extract_features(range(2020, 2025))
X_2025 = features_2020_2024.drop(columns=["TeamID", "Season", "WinRate_count", "WinRate_score","Seed","Season"])
predictions_2025 = model.predict(X_2025)
features_2020_2024["PredictedWinRate"] = predictions_2025

# 输出结果
print(features_2020_2024[["TeamID", "PredictedWinRate"]])



teams_2025 = features_2020_2024[["TeamID", "PredictedWinRate"]]

# 生成所有可能的对决组合
from itertools import combinations

matchups = list(combinations(teams_2025["TeamID"], 2))

# 计算每场对决的胜率概率
results = []
k = 1  # 缩放因子，可以根据实际情况调整

for team1, team2 in matchups:
    # 获取两支球队的预测胜率
    win_rate_1 = teams_2025.loc[teams_2025["TeamID"] == team1, "PredictedWinRate"].values[0]
    win_rate_2 = teams_2025.loc[teams_2025["TeamID"] == team2, "PredictedWinRate"].values[0]

    # 计算胜率差异
    diff = win_rate_1 - win_rate_2

    # 使用逻辑函数计算概率
    prob = 1 / (1 + np.exp(-k * diff))

    # 生成ID和概率
    matchup_id = f"2025_{team1}_{team2}"
    results.append([matchup_id, prob])

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=["ID", "Pred"])

# 输出结果
print(results_df)
results_df.to_csv(r"\kaggle-ncaa\Wresult_df.csv", index=False)
results=pd.read_csv(r"\kaggle-ncaa\Wresult_df.csv")
results = results.drop_duplicates(subset='ID', keep='first')

import pandas as pd
from itertools import combinations
# 从 CSV 文件加载数据
df = pd.read_csv(r"\kaggle-ncaa\march-machine-learning-mania-2025\WTeams.csv")

# 生成所有可能的组合
team_combinations = list(combinations(df["TeamID"], 2))

# 创建一个空的 DataFrame 用于存储结果
result_df = pd.DataFrame(columns=["ID", "Pred"])

# 遍历每一对组合
for team1_id, team2_id in team_combinations:
    team1 = df[df["TeamID"] == team1_id].iloc[0]
    team2 = df[df["TeamID"] == team2_id].iloc[0]

    # 生成 ID
    id_str = f"2025_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
    pred=0.5


    # 将结果添加到 DataFrame 中
    new_row = pd.DataFrame({"ID": [id_str], "Pred": [pred]})
    result_df = pd.concat([result_df, new_row], ignore_index=True)

merged_results = result_df.merge(results, on='ID', how='left')
merged_results.fillna(0.5,inplace=True)
merged_results.to_csv(r"C:\Users\张世越\Desktop\kaggle-ncaa\Wresult_df.csv", index=False)



import numpy as np
import pandas as pd
from sklearn import *
import glob
import os

path = r'\kaggle-ncaa\march-machine-learning-mania-2025\**'
# 过滤掉非文件路径
file_paths = [p for p in glob.glob(path, recursive=True) if os.path.isfile(p)]

data = {os.path.basename(p).split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in file_paths}

# 打印 data 字典的键，此时应该是简单的文件名
print(data.keys())

# 继续后续代码
teams = pd.concat([data['MTeams'], data['WTeams']])
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
teams_spelling.columns = ['TeamID', 'TeamNameCount']
teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
del teams_spelling

season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
gcities = pd.concat([data['MGameCities'], data['WGameCities']])
seasons = pd.concat([data['MSeasons'], data['WSeasons']])

seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}
cities = data['Cities']
sub = data['SampleSubmissionStage2']
del data

season_cresults['ST'] = 'S'
season_dresults['ST'] = 'S'
tourney_cresults['ST'] = 'T'
tourney_dresults['ST'] = 'T'
#games = pd.concat((season_cresults, tourney_cresults), axis=0, ignore_index=True)
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)

c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

games = games[games['ST']=='T']

sub['WLoc'] = 3
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']
sub = sub.fillna(-1)

games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col]


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
import optuna

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X = games[col].fillna(-1)
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

sub_X = sub[col].fillna(-1)
sub_X_imputed = imputer.transform(sub_X)
sub_X_scaled = scaler.transform(sub_X_imputed)


def objective(trial):
    et_params = {
        'et__n_estimators': trial.suggest_int('et__n_estimators', 200, 270),  # Reduced upper bound
        'et__max_depth': trial.suggest_int('et__max_depth', 10, 20),  # Reduced max depth
        'et__min_samples_split': trial.suggest_int('et__min_samples_split', 2, 5),  # Increased min samples
        'et__max_features': trial.suggest_categorical('et__max_features', ['sqrt', 'log2']),  # Removed None option
        'et__criterion': trial.suggest_categorical('et__criterion', ['squared_error', 'absolute_error']),
        'et__n_jobs': -1,
        'et__random_state': 42
    }
    rf_params = {
        'rf__n_estimators': trial.suggest_int('rf__n_estimators',200, 270),  # Moderate number of trees
        'rf__max_depth': trial.suggest_int('rf__max_depth', 10, 20),  # Limited depth
        'rf__min_samples_split': trial.suggest_int('rf__min_samples_split', 2, 5),  # Higher min samples
        'rf__max_features': trial.suggest_categorical('rf__max_features', ['sqrt', 'log2']),  # Restrict features
        'rf__bootstrap': True,  # Enable bootstrapping
        'rf__n_jobs': -1,
        'rf__random_state': 42
    }
    rf_params = {k.replace('rf__', ''): v for k, v in rf_params.items() if k.startswith('rf__')}
    et_params = {k.replace('et__', ''): v for k, v in et_params.items() if k.startswith('et__')}
    et = ExtraTreesRegressor(**et_params)
    rf = RandomForestRegressor(**rf_params)
    voting_regressor = VotingRegressor(estimators=[('et', et), ('rf', rf)])
    model = Pipeline(steps=[
        ('voting', voting_regressor)
    ])
    model.fit(X_scaled, games['Pred'])
    cv_scores = cross_val_score(model, X_scaled, games['Pred'] , cv=5, scoring="neg_mean_squared_error")
    return -cv_scores.mean()

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)
best_params = {'et__n_estimators': 253,
 'et__max_depth': 12,
 'et__min_samples_split': 3,
 'et__max_features': 'sqrt',
 'et__criterion': 'squared_error',
 'rf__n_estimators': 256,
 'rf__max_depth': 20,
 'rf__min_samples_split': 2,
 'rf__max_features': 'log2'}

sub_X.shape
rf_best_params = {k.replace('rf__', ''): v for k, v in best_params.items() if k.startswith('rf__')}
et_best_params = {k.replace('et__', ''): v for k, v in best_params.items() if k.startswith('et__')}
et = ExtraTreesRegressor(**et_best_params)
rf = RandomForestRegressor(**rf_best_params)
voting_regressor = VotingRegressor(estimators=[('et', et), ('rf', rf)])
pipe = Pipeline(steps=[
        ('voting', voting_regressor)
    ])
pipe.fit(X_scaled, games['Pred'])
pred = pipe.predict(sub_X_scaled).clip(0.001, 0.999)
train_pred = pipe.predict(X_scaled).clip(0.001, 0.999)
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(train_pred, games['Pred'])
sub['Pred'] = ir.transform(pred)

sub[['ID', 'Pred']].to_csv('submission.csv', index=False)
print(sub[['ID', 'Pred']].head())

results=pd.read_csv(r"\kaggle-ncaa\result.csv")
merged_results = sub[['ID', 'Pred']].merge(results, on='ID', how='left')
merged_results.to_csv(r"\kaggle-ncaa\result.csv")