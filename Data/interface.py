import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#import plotly.express as px
data_folder = Path(__file__).parent / "CSV files"
file_to_open = data_folder / "player_stats.csv"
player_data = pd.read_csv(file_to_open)
file_to_open = data_folder / "awards_data.csv"
awards_data = pd.read_csv(file_to_open)


player_data = pd.read_csv(file_to_open)

data = pd.DataFrame(player_data)

awards_merged_data = pd.merge(awards_data, player_data[['playerid', 'season', 'player_name']], on=['playerid', 'season'], how='left')
result = awards_merged_data[['playerid', 'player_name']]

def still_playing(row):
    if row['season'] == 2021:
        return True
    else:
        return False

recent_szn = data.loc[data.groupby('player')['season'].idxmax()]

recent_szn['curr_playing'] = recent_szn.apply(still_playing, axis = 1)



#Percentiles
superstar_stat = np.percentile(recent_szn['points'],99.5)
elite_stat = np.percentile(recent_szn['points'],97)
great_stat = np.percentile(recent_szn['points'],95)
highend_starter_stat = np.percentile(recent_szn['points'],90)
lowend_starter_stat = np.percentile(recent_szn['points'],85)
rotation_stat = np.percentile(recent_szn['points'],70)
bench_stat = np.percentile(recent_szn['points'],50)
lowend_stat = np.percentile(recent_szn['points'],40)

def tier_check(row):
    if row['points'] >= superstar_stat:
        return 'Superstar'
    elif row['points'] >= elite_stat:
        return 'Elite'
    elif row['points'] >= highend_starter_stat:
        return 'High End'
    elif row['points'] >= lowend_starter_stat:
        return 'Low End'
    elif row['points'] >= rotation_stat:
        return 'Rotation'
    elif row['points'] >= bench_stat:
        return 'Bench'
    elif row['points'] >= lowend_stat:
        return 'Bad'

recent_szn['tier'] = recent_szn.apply(tier_check, axis = 1)


st.title("NBA Scoring Efficiency Predictor")

st.header("Pick a Player:", divider = 'green')

player = st.selectbox("Select One",(recent_szn['player']),placeholder = " ", index = None)

#Session States
if 'compare_clicked' not in st.session_state:
    st.session_state.compare_clicked = False

if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

if player != st.session_state.selected_player:
    st.session_state.compare_clicked = False
    st.session_state.selected_player = player



compare_container = st.empty()

if player:
    st.header(f'Projections for {player}:',divider = 'green')

    certain_player_data = data[data['player'] == player]

    curr_player_data = recent_szn[recent_szn['player'] == player]

    if curr_player_data['curr_playing'].iloc[0]:
        ppg = curr_player_data['points'].iloc[0]
        #ppg

        year = 2022

        tier = curr_player_data['tier'].iloc[0]

        percentile = round(stats.percentileofscore(recent_szn['points'], curr_player_data['points'], kind='rank')[0])

        st.subheader('Analysis:')

        st.text_area(' ',f"The selected player is projected {ppg} ppg in {year}. {player} is projected to be a {tier} level scorer and falls into the {percentile}th percentile scoring level compared to the rest of the league.")

    else:
        percentile = round(stats.percentileofscore(recent_szn['points'], curr_player_data['points'], kind='rank')[0])

        st.subheader('Analysis:')

        st.text_area(' ',
                     f"This player will not play in the 2022 season but in is most recent season, he was in the {percentile}th percentile of scorers.")


    if not st.session_state.compare_clicked:
        if compare_container.button('Comparison'):
            st.session_state.compare_clicked = True

    if st.session_state.compare_clicked:
        player2 = st.selectbox(f"Select player to compare to {player}", (recent_szn['player']), index=None)

        certain_player2_data = data[data['player'] == player2]

        curr_player2_data = recent_szn[recent_szn['player'] == player2]


        st.text_area('Comparison',f"{curr_player_data['player']} is projected more ppg that {curr_player2_data['player']}")



        st.header("Visualization: ", divider = 'green')

        #Data Cleaning for plot
        certain_player_data = certain_player_data[['player','season','points']]
        certain_player2_data = certain_player2_data[['player','season','points']]

        merged_data = pd.merge(certain_player_data,certain_player2_data, on = 'season',how = 'outer', suffixes = ('_1','_2'))
        player1_name = certain_player_data['player'].iloc[0]
        player2_name = certain_player2_data['player'].iloc[0]

        merged_data['player_1'].fillna(player1_name, inplace=True)
        merged_data['player_2'].fillna(player2_name, inplace=True)
        merged_data['points_1'].fillna(0, inplace=True)
        merged_data['points_2'].fillna(0, inplace=True)

        merged_data.sort_values('season', ascending = True, inplace=True)

        st.dataframe(merged_data)

        #Plot Set-Up
        bar_width = 0.35
        index = np.arange(len(merged_data))
        plt.figure(figsize=(10, 6))
        bar1 = plt.bar(index,merged_data['points_1'], bar_width, label = f'{player1_name}', color = 'b')
        bar2 = plt.bar(index + bar_width,merged_data['points_2'], bar_width, label = f'{player2_name}', color = 'r')

        #Plot
        plt.xlabel('Season')
        plt.ylabel('Points')
        plt.title(f'PPG Comparison: {player1_name} vs {player2}')
        plt.xticks(index+ bar_width/2, merged_data['season'].values)
        plt.legend()
        plt.tight_layout()

        #Display
        st.pyplot(plt)

        if compare_container.button("Reset"):
            st.session_state.compare_clicked = False

    else:
        st.header("Visualization: ", divider='green')

        st.bar_chart(certain_player_data, x='season', y='points')



