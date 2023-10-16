import pandas as pd

# Load your data
awards = pd.read_csv("venv/Data/awards_data.csv")
player_data = pd.read_csv("venv/Data/player_stats.csv")
team_data = pd.read_csv("venv/Data/team_stats.csv")
#rebounding_data = pd.read_csv("venv/Data/team_rebounding_data_22.csv")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

# TODO: Data Cleaning & Preprocessing

# Merged dataset to use for the model
data = pd.merge(player_data, awards, on=['nbapersonid', 'season'], how='left')
data = pd.merge(data, team_data, on=['team', 'season', 'nbateamid'])

# Adjust for certain seasons
szn_adjustments = {2011: 66, 2019: 72, 2020: 72}
data['games_start'] = data.apply(lambda x: round(x['games_start'] * (82 / szn_adjustments.get(x['season'], 82))),
                                 axis=1)
data['mins'] = data.apply(lambda x: round(x['mins'] * (82 / szn_adjustments.get(x['season'], 82))), axis=1)
data['games_x'] = data.apply(lambda x: round(x['games_x'] * (82 / szn_adjustments.get(x['season'], 82))), axis=1)

data['ppg'] = data['points'] / data['games_x']

data['YIL'] = data['season'] - data['draftyear']

peak_year = 6
data['years_from_peak'] = data['YIL'].apply(lambda x: abs(x - peak_year))

# TODO: Years in the league and how it affects players


# Sample data for testing code, replace this with your actual data loading and preprocessing

features = data.groupby('nbapersonid').agg({
    'mins': 'mean',
    'games_x': 'mean',
    'games_start': 'mean',
    'fgp': 'mean',
    'usg': 'mean',
    'VORP': 'mean',
    'ppg': 'mean',
    'years_from_peak' : 'mean',
    'points' : 'mean'
}).reset_index()

# Fill NaN with 0s
features = features.fillna(0)

# Feature Selection
target = features['ppg']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features.drop(['ppg'], axis=1), target, test_size=0.2)

# Feature Scaling
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
rf = RandomForestRegressor(random_state=42)
lr = LinearRegression()
svr = SVR()

# Hyperparameter Tuning using GridSearchCV
# Adjust param_grid based on your dataset characteristics and computational capability

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

# Ensemble Learning: Voting Classifier
voting_reg = VotingRegressor(
    estimators=[
        ('rf', best_rf),
        ('lr', lr),
        ('svr', svr)
    ],
)

voting_reg.fit(X_train_scaled, y_train)
predictions = voting_reg.predict(X_test_scaled)


# Regression Report to be printed
def regression_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    report_data = {
        'Mean Absolute Error': [mae],
        'Mean Squared Error': [mse],
        'Root Mean Squared Error': [rmse],
        'R-squared': [r2]
    }

    report_df = pd.DataFrame.from_dict(report_data)

    return report_df


reg_report = regression_report(y_test, predictions)
print(reg_report)

'''new_players_scaled = scaler.transform(features,axis = 1))

#Calcualte and predict
new_player_probs = voting_reg.predict_proba(new_players_scaled)
probs = pd.DataFrame(new_player_probs, columns = voting_reg.classes_)

#Put results of probability of each career outcome classifer into table
results = pd.concat([features['nbapersonid'], probs], axis = 1)
print(results)'''


importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Assuming `features` still contains your feature names
feature_names = features.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlim([-1, X_train.shape[1]])
plt.show()
