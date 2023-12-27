# Import important modules
import streamlit as st
import pandas as pd

# Config layout of page and add important info
st.set_page_config(layout='wide')
st.header('MW-PECVD Optimisation Tool')
st.subheader('By Jahanzeb Khan')
st.markdown('This app uses a multi-layer perceptron (MLP) deep learning model to recommend the optimum parameters for you to obtain your desired coatings while aiming to be economic and sustainable. \
            These predictions are made based on data obtained\
             using a Hauzer Flexicoat 850 Coating system with no bias\
             applied across the substrates used and no measurements taken\
             of the self-bias applied')

#get lookup table from csv file
model_df = pd.read_csv('Code/Data/model.csv')

#find boundaries stated in lookup table
hardness_min = model_df['Hardness (GPa)'].min()
hardness_max = model_df['Hardness (GPa)'].max()
ym_min = model_df['Young\'s Modulus (GPa)'].min()
ym_max = model_df['Young\'s Modulus (GPa)'].max()


hardness_target = st.number_input('Enter your desired hardness (GPa)', min_value = hardness_min, max_value=hardness_max)
ym_target = st.number_input('Enter your desired hardness (GPa)', min_value = ym_min, max_value=ym_max)

#iterate through lookup table from top to find optimum parameters 
for i, row in model_df.iterrows():
    hardness = row['Hardness (GPa)']
    ym = row['Young\'s Modulus (GPa)']
    if hardness >= hardness_target and ym >= ym_target:
        power = row['MW Power (W)']
        working_pres = round(row['Working Pressure (mbar)'], 3)
        flow_ratio = row['Gas Flowrate Ratio (%C2H2)']
        break


# Display predictions to user
display_data = {'MW-PECVD Parameter': ['MW Power (W)','Working Pressure (mbar)','Gas Flowrate Ratio (%C2H2)'], 'Optimum Value': [power, working_pres, flow_ratio]}
display_df = pd.DataFrame(data=display_data)
st.dataframe(display_df,hide_index=True)