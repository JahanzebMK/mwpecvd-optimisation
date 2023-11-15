import pandas as pd
from sklearn import linear_model
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_percentage_error
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


#import training data
training_df = pd.read_csv('Data/training.csv')
power = training_df.power
working_pres = training_df.working_pres
flow_ratio = training_df.flow_ratio
hardness = training_df.hardness
ym = training_df.ym

features = ['MW Power (W)', 'Working Pressure (mbar)', 'Gas Flowrate Ratio (%C2H2)']
targets = ['Hardness (GPa)', 'Young\'s Modulus (GPa)']

#import testing data
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

x=df[features]
y=df[targets]


#training dataframe
test_datadict = {'MW Power (W)':test_power,\
            'Working Pressure (mbar)':test_working_pres,\
            'Gas Flowrate Ratio (%C2H2)':test_ratio,\
            'Hardness (GPa)':test_hardness,\
            'Young\'s Modulus (GPa)':test_ym}
test_df = pd.DataFrame(test_datadict)

test_x=test_df[features]
test_y=test_df[targets]



#Linear regression model training

#initialise timer
t1 = time.perf_counter() 

#bring in linear regression model from scikit-learn library
model = Pipeline(steps=[('sca', MinMaxScaler()),
                        ('reg', linear_model.LinearRegression())])

#train model with training data
model.fit(x,y)

#output time taken for model to train
t = (time.perf_counter() - t1) 
print(t)



#Linear regression model testing

#output values predicted by model and actual value
pred = model.predict(test_x)
print(pred)
print(test_y)

#output coefficients from model
coefficients = pd.concat([pd.DataFrame(features),
                          pd.DataFrame(np.transpose(model.named_steps['reg'].coef_))],
                         axis = 1)
print(coefficients)


#partial dependence plots
display1 = PartialDependenceDisplay.from_estimator(
    model,
    x,
    features,
    kind="both",
    subsample=50,
    target = 0,
    n_jobs=3,
    grid_resolution=20,
    random_state=0,
    ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
)
display1.figure_.subplots_adjust(wspace=0.4, hspace=0.3)
display1.figure_.suptitle("Partial dependence of Hardness (GPa) on Deposition Parameters")


display2 = PartialDependenceDisplay.from_estimator(
    model,
    x,
    features,
    kind="both",
    subsample=50,
    target = 1,
    n_jobs=3,
    grid_resolution=20,
    random_state=0,
    ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
)
display2.figure_.subplots_adjust(wspace=0.4, hspace=0.3)
display2.figure_.suptitle("Partial dependence of Young's Modulus (GPa) on Deposition Parameters")


plt.show()
