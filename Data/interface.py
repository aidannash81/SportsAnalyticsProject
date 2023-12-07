import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(layout="wide")

#import plotly.express as px
data_folder = Path(__file__).parent / "Data"
file_to_open = data_folder / "predicted_player_data.csv"
player_data = pd.read_csv(file_to_open)
data_folder2 = Path(__file__).parent / "Data"
file_to_open2 = data_folder2 / "predicted_awards_data.csv"
awards_data = pd.read_csv(file_to_open2)

player_data = pd.read_csv(file_to_open)

data = pd.DataFrame(player_data)

awards = pd.DataFrame(awards_data)

data = data.drop_duplicates(subset=['season', 'player'], keep='first')

awards = awards.drop_duplicates(subset=['season', 'player'], keep='first')


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

dir_path = os.path.dirname(os.path.realpath(__file__))
logo_path = os.path.join(dir_path, "Data", "RimReaderLogo.png")
logo = Image.open(logo_path)
st.image(logo, use_column_width=True,caption="Made by Trace Batesky")

tab1, tab2, tab3 = st.tabs(['Predictions','Comparisons','Awards'])

with tab1:
    st.header("NBA Scoring Efficiency Predictor",divider = 'red')

    st.header("Pick a Player:")

    player = st.selectbox("Select One",(recent_szn['player']),placeholder = " ", index = None)

    if player:
        col1,col2 = st.columns(2)

        with col1:

            st.header(f'Projections for {player}:', divider='red')

            col1_1,col1_2 = st.columns(2)

            with col1_1:

                certain_player_data = data[data['player'] == player]

                curr_player_data = recent_szn[recent_szn['player'] == player]

                if curr_player_data['curr_playing'].iloc[0]:
                    ppg = curr_player_data['points'].iloc[0]
                    #ppg

                    year = 2022

                    tier = curr_player_data['tier'].iloc[0]

                    percentile = round(stats.percentileofscore(recent_szn['points'], curr_player_data['points'], kind='rank')[0])

                    st.subheader('Analysis:')

                    certain_player_data['ppg'] = round(certain_player_data['ppg'],1)

                    certain_player_data['season'] = certain_player_data['season'].astype(str)

                    st.dataframe(certain_player_data[['season','ppg']])

                    st.text_area(' ',f"The selected player is projected {ppg} ppg in {year}. {player} is projected to be a {tier} level scorer and falls into the {percentile}th percentile scoring level compared to the rest of the league.")

                else:
                    percentile = round(stats.percentileofscore(recent_szn['points'], curr_player_data['points'], kind='rank')[0])

                    st.subheader('Analysis:')

                    certain_player_data['ppg'] = round(certain_player_data['ppg'], 1)

                    certain_player_data['season'] = certain_player_data['season'].astype(str)

                    st.dataframe(certain_player_data[['season', 'ppg']])

                    st.text_area(' ',
                                 f"This player will not play in the 2022 season but in is most recent season, he was in the {percentile}th percentile of scorers.")
            with col1_2:
                tier = curr_player_data['tier'].iloc[0]
                pred_ppg = round(curr_player_data['predictions'].iloc[0],1)
                st.subheader(f"Predicted PPG: {pred_ppg}")
                st.subheader(f"Projected Tier: {tier}")

        with col2:
            st.header("Visualization: ", divider='red')

            st.bar_chart(certain_player_data, x='season', y='points')

with tab2:
    st.header('Comparison Feature',divider = 'red')

    player1 = st.selectbox(f"Select first player to compare", (recent_szn['player']), index = None)

    certain_player_data = data[data['player'] == player1]

    curr_player_data = recent_szn[recent_szn['player'] == player1]

    if player1:
        player2 = st.selectbox(f"Select player to compare to {player1}", (recent_szn['player']), index=None)

        certain_player2_data = data[data['player'] == player2]

        curr_player2_data = recent_szn[recent_szn['player'] == player2]

        st.subheader('Comparison: ')
        st.text_area(' ',f"{curr_player_data['player'].iloc[0]} is projected more ppg that {curr_player2_data['player'].iloc[0]}")

        st.header("Visualization: ", divider = 'red')

        #Data Cleaning for plot
        certain_player_data = certain_player_data[['player','season','ppg']]
        certain_player2_data = certain_player2_data[['player','season','ppg']]

        merged_data = pd.merge(certain_player_data,certain_player2_data, on = 'season',how = 'outer', suffixes = ('_1','_2'))
        player1_name = certain_player_data['player'].iloc[0]
        player2_name = certain_player2_data['player'].iloc[0]

        merged_data['player_1'].fillna(player1_name, inplace=True)
        merged_data['player_2'].fillna(player2_name, inplace=True)
        merged_data['ppg_1'].fillna(0, inplace=True)
        merged_data['ppg_2'].fillna(0, inplace=True)


        player1_data = data[data['player'] == player1_name]
        player2_data = data[data['player'] == player2_name]

        merged_data['season'] = merged_data['season'].astype(str)

        new_row_data = {
            'player_1': player1_name,
            'player_2': player2_name,
            'season': "2022 (Pred)",
            'ppg_1': round(player1_data['predictions'].iloc[0], 1),
            'ppg_2': round(player2_data['predictions'].iloc[0], 1)
        }

        # Convert the dictionary to a DataFrame
        new_row_df = pd.DataFrame([new_row_data])

        # Append the new DataFrame to the existing one
        merged_data = merged_data.append(new_row_df, ignore_index=True)



        merged_data.sort_values('season', ascending = True, inplace=True)

        st.dataframe(merged_data)

        #Plot Set-Up
        bar_width = 0.35
        index = np.arange(len(merged_data))
        plt.figure(figsize=(10, 6))
        bar1 = plt.bar(index,merged_data['ppg_1'], bar_width, label = f'{player1_name}', color = 'b')
        bar2 = plt.bar(index + bar_width,merged_data['ppg_2'], bar_width, label = f'{player2_name}', color = 'r')

        #Plot
        plt.xlabel('Season')
        plt.ylabel('PPG')
        plt.title(f'PPG Comparison: {player1_name} vs {player2}')
        plt.xticks(index+ bar_width/2, merged_data['season'].values)
        plt.legend()
        plt.tight_layout()

        #Display
        st.pyplot(plt)

with tab3:

    st.header("Award Prediction", divider = 'red')

    award_player = st.selectbox(f"Select first player to compare", (recent_szn['player']), index=None, key=2)

    if award_player:
        st.header(f'Award Projections for {award_player}:', divider='red')

        certain_player_data = data[data['player'] == award_player]

        curr_player_data = recent_szn[recent_szn['player'] == award_player]

        last_szn = recent_szn[recent_szn['player'] == award_player]['season'].iloc[0]

        player_awards_data = awards[awards['player'] == award_player]

        if curr_player_data['curr_playing'].iloc[0]:
            st.write(f"Prediction: {player_awards_data['predictions'].iloc[0]}")
        else:
            st.text_area(" ",f"{award_player} is not playing in the 2022 season. In his last season, {last_szn}")

        st.header("Previous Awards")

        awards = awards.sort_values(by = "season", ascending= False)

        awards['season'] = awards['season'].astype(str)

        full_awards = st.toggle("Full Awards List")
        if full_awards:
            st.dataframe(awards[awards['player'] == award_player])
        else:
            st.dataframe(awards[awards['player'] == award_player][['season','player','All NBA First Team','All NBA Second Team', 'All NBA Third Team','all_star_game']])




