import pandas as pd
from sklearn import linear_model
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import time


#import training data
training_df = pd.read_csv('Data/final_training.csv')
power = training_df.power
working_pres = training_df.working_pres
flow_ratio = training_df.flow_ratio
hardness = training_df.hardness
ym = training_df.ym

#training dataframe
datadict = {'MW Power (W)':power,\
            'Working Pressure (mbar)':working_pres,\
            'Gas Flowrate Ratio (%C2H2)':flow_ratio,\
            'Hardness (GPa)':hardness,\
            'Young\'s Modulus (GPa)':ym}
df = pd.DataFrame(datadict)
features = ['MW Power (W)', 'Working Pressure (mbar)', 'Gas Flowrate Ratio (%C2H2)']
targets = ['Hardness (GPa)', 'Young\'s Modulus (GPa)']
x = df[features]
y = df[targets]


#train MLP Regressor model
nn=make_pipeline(MinMaxScaler(),MLPRegressor(random_state=42, max_iter=34000)) #42
nn.fit(x,y)


#generate model lookup table

#create lists of parameter values with appropriate interval for each
power_values = np.arange(min(power),
                         (max(power)+1),1).tolist()
pres_values = np.arange(min(working_pres),
                        (max(working_pres)+0.001), 0.001).tolist()
ratio_values = np.arange(min(flow_ratio),
                         (max(flow_ratio)+1), 1).tolist()

parameters_dict = {'MW Power (W)':[],
                   'Working Pressure (mbar)':[],
                   'Gas Flowrate Ratio (%C2H2)':[]}

#create dataframe of all possible combinations from lists
for power in power_values:
    for pres in pres_values:
        for ratio in ratio_values:
            parameters_dict['MW Power (W)'].append(power)
            parameters_dict['Working Pressure (mbar)'].append(pres)
            parameters_dict['Gas Flowrate Ratio (%C2H2)'].append(ratio)
                                
parameters_df = pd.DataFrame(parameters_dict)

#add predicted hardness and ym values to dataframe
predict_array = nn.predict(parameters_df)
predict_df = pd.DataFrame(predict_array, columns = targets)

#round up property values to 1 d.p.
predict_df = predict_df.round(1)

#output csv file
model_df = parameters_df.join(predict_df)
model_df.to_csv('Data/model.csv', index=False, header=True)





