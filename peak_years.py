import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

awards = pd.read_csv("venv/Data/awards_data.csv")
player_data = pd.read_csv("venv/Data/player_stats.csv")
team_data = pd.read_csv("venv/Data/team_stats.csv")

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

avg_ppg_by_YIL = data.groupby('YIL')['ppg'].mean().reset_index()

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(avg_ppg_by_YIL['YIL'], avg_ppg_by_YIL['ppg'], marker='o')
plt.title('Average PPG by Years in League')
plt.xlabel('Years in League')
plt.ylabel('Average PPG')
plt.grid(True)
plt.show()

#The graph shows that 6 years in the league seems to be the peak
#year for our data