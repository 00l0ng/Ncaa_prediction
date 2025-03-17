import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score

# 加载数据
data_path = r'C:\Users\张世越\Desktop\kaggle-ncaa\march-machine-learning-mania-2025/'
df_regular_season_detailed = pd.read_csv(data_path + "MRegularSeasonDetailedResults.csv")
df_seeds = pd.read_csv(data_path + "MNCAATourneySeeds.csv")
df_rankings = pd.read_csv(data_path + "MMasseyOrdinals.csv")


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

    # 处理排名数据
    rankings = df_rankings[df_rankings["Season"].isin(years)]
    avg_rankings = rankings.groupby(["Season", "TeamID"]).agg({"OrdinalRank": "mean"}).reset_index()
    avg_rankings.rename(columns={"OrdinalRank": "AvgRank"}, inplace=True)

    # 合并数据
    final_data = pd.merge(team_stats, tourney_seeds, on="TeamID", how="left")
    final_data = pd.merge(final_data, avg_rankings, on=["Season", "TeamID"], how="left")
    final_data.fillna(0, inplace=True)

    return final_data


# 提取2019年-2023年的数据特征
features_2019_2023 = extract_features(range(2019, 2024))
features_2019_2023 = features_2019_2023.drop_duplicates()

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
X_2025=X_2025.drop_duplicates()
predictions_2025 = model.predict(X_2025)
features_2020_2024["PredictedWinRate"] = predictions_2025

# 输出结果
print(features_2020_2024[["TeamID", "PredictedWinRate"]])



# 示例数据
teams_2025 = features_2020_2024[["TeamID", "PredictedWinRate"]]

teams_2025 = teams_2025.drop_duplicates(subset='TeamID', keep='first')

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
