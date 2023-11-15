import pandas as pd
from sklearn import linear_model
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
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
from sklearn.pipeline import Pipeline
import time




#training data
training_df = pd.read_csv('Data/training.csv')
power = training_df.power
working_pres = training_df.working_pres
flow_ratio = training_df.flow_ratio
hardness = training_df.hardness
ym = training_df.ym



#test data
testing_df = pd.read_csv('Data/testing.csv')
test_power = testing_df.power
test_working_pres = testing_df.working_pres
test_ratio = testing_df.flow_ratio
test_hardness = testing_df.hardness
test_ym = testing_df.ym


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


#test dataframe
test_datadict = {'MW Power (W)':test_power,\
            'Working Pressure (mbar)':test_working_pres,\
            'Gas Flowrate Ratio (%C2H2)':test_ratio,\
            'Hardness (GPa)':test_hardness,\
            'Young\'s Modulus (GPa)':test_ym}
test_df = pd.DataFrame(test_datadict)
test_x = test_df[features]
test_y = test_df[targets]


scaler = MinMaxScaler()

data=scaler.fit_transform(x)
data=pd.DataFrame(data, columns = features)
print(data)
data.to_csv('Data/scaled.csv', index=False, header=True)

