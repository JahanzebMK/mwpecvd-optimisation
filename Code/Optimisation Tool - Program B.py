import pandas as pd

hardness_valid = False
ym_valid = False

#get lookup table from csv file
model_df = pd.read_csv('Data/model.csv')

#find boundaries stated in lookup table
hardness_min = model_df['Hardness (GPa)'].min()
hardness_max = model_df['Hardness (GPa)'].max()
ym_min = model_df['Young\'s Modulus (GPa)'].min()
ym_max = model_df['Young\'s Modulus (GPa)'].max()

hardness = 0
ym = 0

#ensure users desired properties are withing bounds from lookup table
while hardness_valid == False:
    hardness_target = float(input('Enter desired Hardness of DLC coating in GPa between\
                                '+str(hardness_min)+'GPa and '+str(hardness_max)+'GPa:'))
    if hardness_target<hardness_min:
        print('Hardness value entered is below range of this tool')
        continue
    if hardness_target>hardness_max:
        print('Hardness value entered is above range of this tool')
        continue
    else:
        hardness_valid = True


while ym_valid == False:
    ym_target = float(input('Enter desired Young\'s Modulus of DLC coating in GPa between\
                            '+str(ym_min)+'GPa and '+str(ym_max)+'GPa:'))
    if ym_target<ym_min:
        print('Young\'s Modulus value entered is below range of this tool')
        continue
    if ym_target>ym_max:
        print('Young\'s Modulus value entered is above range of this tool')
        continue
    else:
        ym_valid = True
        

#iterate through lookup table from top to find optimum parameters 
for i, row in model_df.iterrows():
    hardness = row['Hardness (GPa)']
    ym = row['Young\'s Modulus (GPa)']
    if hardness >= hardness_target and ym >= ym_target:
        power = row['MW Power (W)']
        working_pres = round(row['Working Pressure (mbar)'], 3)
        flow_ratio = row['Gas Flowrate Ratio (%C2H2)']
        break

#output optimum parameters and limitations of model 
print(f'\nMost economic settings to achieve a Hardness of {hardness}GPa and a Young\'s Modulus of {ym}GPa \n\
Power (W): {power} \n\
Working Pressure (mbar): {working_pres} \n\
Gas Flowrate Ratio (%C2H2): {flow_ratio} \n\
\nThese predictions are made based on data obtained\
 using a Hauzer Flexicoat 850 Coating system with no bias\
 applied across the substrates used and no measurements taken\
 of the self-bias applied')

