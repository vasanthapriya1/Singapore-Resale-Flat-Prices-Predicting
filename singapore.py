import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import datetime as dt

with open('dt.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open(r'Category_Columns_Encoded_Data.json', 'r') as f:
    data = json.load(f)

# Creating option menu in the side bar

with st.sidebar:

    selected = option_menu("Menu", ["Home","Resale Price"], 
                           icons=["house","cash"],
                           menu_icon= "menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "green"},
                                   "nav-link-selected": {"background-color": "green"}}
                          )
    
if selected == 'Home':

    st.title(":blue[*SINGAPORE RESALE FLAT PRICES PREDICTING*]")
    
    col1, col2 = st.columns(2)

    with col1:
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :violet[*Overview*] : Build regression model to predict resale price")
        col1.markdown("# ")
        col1.markdown("## :violet[*Domain*] : Resale Flat Prices")
        col1.markdown("# ")
        col1.markdown("## :violet[*Technologies used*] : Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Streamlit.")

    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        col2.image("flat.jpg")
        col2.markdown("# ")
        col2.markdown("# ")
        col2.markdown("# ")
        col2.image("flat1.jpg")

if selected == 'Resale Price':
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    col1, col2 = st.columns(2, gap='large')

    with col1:
        min_month = 1
        max_month = 12

        # Slider for selecting the month
        selected_month = st.slider("Select the Item Month", min_value=min_month, max_value=max_month, value=min_month,
                                step=1)

        town = st.selectbox('Select the **Town**', data['town_before'])

        flat_type = st.selectbox('Select the **Flat Type**', data['flat_type_before'])

        block = st.selectbox('Select the **Block**', data['block_before'])

        street_name = st.selectbox('Select the **Street Name**', data['street_name_before'])
        
    with col2:
        
        # floor_area_sqm = st.number_input('Select the **floor_area_sqm**', value=60.0, min_value=28.0, max_value=173.0,
        #                                 step=1.0)

        flat_model = st.selectbox('Select the **flat_model**', data['flat_model_before'])

        lease_commence_date = st.number_input('Enter the **Lease Commence Year**', min_value=1966, max_value=2022,
                                            value=2017)

        # end_of_lease = st.number_input('Select the **end_of_lease**', value=0, min_value=0, max_value=3000)

        year = st.number_input("Select the transaction Year which you want**", min_value=1990, max_value=2024, value=dt.datetime.now().year)
        
        min_storey = st.number_input('Select the **min_storey**', value=0, min_value=0, max_value=100)

        max_storey = st.number_input('Select the **max_storey**', value=10, min_value=0, max_value=100)

    st.markdown('Click below button to predict the **Flat Resale Price**')
    prediction = st.button('**Predict**')

    # Convert categorical values to corresponding encoded values

    town_encoded = data['town_before'].index(town)
    flat_type_encoded = data['flat_type_before'].index(flat_type)
    block_encoded = data['block_before'].index(block)
    street_name_encoded = data['street_name_before'].index(street_name)
    flat_model_encoded = data['flat_model_before'].index(flat_model)

    # Prediction logic
    test_data = [
        selected_month,
        town_encoded,  
        flat_type_encoded, 
        block_encoded,  
        street_name_encoded, 
        flat_model_encoded, 
        lease_commence_date,
        year,
        min_storey,
        max_storey,
    ]

    if prediction:
        # Perform prediction
        test_data_array = np.array([test_data], dtype=np.float32)  # Convert to 2D array with float data type

        predicted_price = dt_model.predict(test_data_array)  # Assuming your model's predict method takes a 2D array

        # Display predicted price
        st.markdown(f"### :blue[Flat Resale Price is] :green[$ {round(predicted_price[0], 3)}]")




        