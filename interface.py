import streamlit as st
import pandas as pd
#import plotly.express as px

st.title("NBA Scoring Efficiency Predictor")

st.header("Pick a Player:", divider = 'green')

player = st.selectbox("Select One",('Hi'))

if player:
    st.header(f'Projections for {player}:')