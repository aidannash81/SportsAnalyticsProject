import streamlit as st
import pandas as pd
#import plotly.express as px
import os

player_data = pd.read_csv("CSV files/player_stats.csv")

print(os.getcwd())

#from full_model_dir.full_model import data

st.title("NBA Scoring Efficiency Predictor")

st.header("Pick a Player:", divider = 'green')

#from full_model import main  # adjust the import statement to match your project structure

player = st.selectbox("Select", player_data['player'])

if player:
    st.header(f'Projections for {player}:')

# this ensures the code inside this block only runs when the script is executed directly
#if __name__ == "__main__":
#    new_data = main()
