# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:32:10 2021

@author: Devineni
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:26:27 2021

@author: Devineni
"""
# Necessary modules
import pandas as pd
import numpy as np
# from statistics import mean
import time
import datetime as dt
import matplotlib.pyplot as plt
pd.options.plotting.backend = "matplotlib"
from tabulate import tabulate
from sqlalchemy import create_engine

# functions to print in colour
def prRed(skk): print("\033[31;1;m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[33;1;m {}\033[00m" .format(skk)) 
# The following is a general syntax to dedine a MySQL connection
# =============================================================================
# engine = create_engine("mysql+pymysql://admin:the_secure_password_4_ever@localhost/",\
#                          pool_pre_ping=True) # Krishna's local address
# =============================================================================
engine = create_engine("mysql+pymysql://wojtek:Password#102@wojtek.mysql.database.azure.com/",\
                      pool_pre_ping=True) # Cloud server address

#%% Function import
"""Syntax to import a function from any folder. Useful if the function.py file 
   is in another folder other than the working folder"""
import sys  
# sys.path.append("C:/Users/Devineni/OneDrive - bwedu/4_Recirculation/python_files/")  
from Outdoor_CO2 import outdoor # This function calculates the outdoor CO2 data

#%% Control plot properties"
"""This syntax controls the plot properties(default plot font, shape, etc), 
    more attributes can be added and removed depending on the requirement """

from pylab import rcParams
rcParams['figure.figsize'] = 7,4.5
plt.rcParams["font.family"] = "calibri"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.size"] = 10
	
plt.close("all")

#%% Load relevant data
i = 16 # to select the experiment (see Timeframes.xlsx)
j = 2 # to select the sensor in the ventilation device
offset = 0 
# time = pd.read_excel("C:/Users/Devineni/OneDrive - bwedu/4_Recirculation/Times_thesis.xlsx", sheet_name="Timeframes")
# The dataframe time comes from the excel sheet in the path above, to make -
# - changes go to this excel sheet, edit and upload it to mysql.

time = pd.read_sql_query("SELECT * FROM testdb.timeframes;", con = engine)      #standard syntax to fetch a table from Mysql

start, end = str(time["Start"][i] - dt.timedelta(minutes=20)), str(time["End"][i]) # selects start and end times to slice dataframe, the sliced data starts with a point in time 20 minutes before the beginning of the experiment
t0 = time["Start"][i]                                                           #actual start of the experiment

table = time["tables"][i].split(",")[j]                                         #Name of the ventilation device

dum = [["Experiment",time["short_name"][i] ], ["Sensor", table]]                # Creates a list of 2 rows filled with string tuples specifying the experiment and the sensor.
print(tabulate(dum))                                                            # Prints the inut details in a table

database = time["database"][i]                                                  # Selects the name of the database as a string 


background, dummy = outdoor(str(t0), str(end), plot = False)                    # Syntax to call the background concentration function, "dummy" is only necessary since the function "outdoor" returns a tuple of a dataframe and a string.
background = background["CO2_ppm"].mean()                                       # Future: implement cyclewise background concentration; Till now it takes the mean outdoor concentration of the whole experiment.

df = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND\
                       '{}'".format(database, table, start, end), con = engine) # Slices the data measured by the selected ventilation device sensor during the decay curve experiment of interest.
df = df.loc[:,["datetime", "CO2_ppm"]]                                          # Slices the datetime and CO2 data out of the MySQL-export data.
df["original"] = df["CO2_ppm"]                                                  # Copies the original absolute CO2-concentrations data form CO2_ppm in a "backup"-column originals 

df["CO2_ppm"] = df["CO2_ppm"] - background                                      # substracts the background concentrations -> CO2_ppm contains CO2-concentration of some instance of time above background concentration.

if df["CO2_ppm"].min() < 0:                                                     # Sometimes the accumulated amount of CO2 concentraion becomes negative. This is not possible and would lead to a mistake for the integral calculation. An artificial offset lifts the whole decay curve at >=0.
    offset = df["CO2_ppm"].min()
    df["CO2_ppm"] = df["CO2_ppm"] - offset
    
df = df.loc[~df.duplicated(subset=["datetime"])]                                # Checks for duplicated in datetime and removed them; @Krishna: How can such a duplicate occur?
diff = (df["datetime"][1]-df["datetime"][0]).seconds                            # integer diff in s; Calculates the length of the time interval between two timestamps 
df = df.set_index("datetime")                                                   # Resets the index of the dataframe df from the standard integer {0, 1, 2, ...} to be exchanged by the datetime column containing the timestamps.
while not(t0 in df.index.to_list()):                                            # The t0 from the excel sheet may not be precice that the sensor starts 
    t0 = t0 + dt.timedelta(seconds=1)                                           # - starts at the same time so i used this while loop to calculate the 
    print(t0)                                                                   # - the closest t0 after the original t0

df["roll"] = df["CO2_ppm"].rolling(int(120/diff)).mean()                             # moving average for 2 minutes, used to calculate Cend 



c0 = df["CO2_ppm"].loc[t0]                                                      # C0
Cend37 = round((c0)*0.37, 2)   

cend = df.loc[df["roll"].le(Cend37)]                                            # Cend

if len(cend) == 0:                                                              # Syntax to find the tn of the experiment
    tn = str(df.index[-1])
    print("The device has not reached 37% of its initial concentration")
else:
    tn = str(cend.index[0])

#%%% Plot Original
fig,ax = plt.subplots()
df.plot(title = "original", color = [ 'green', 'silver'], ax = ax)

pdf = df.copy().reset_index()
#%%% Only in VS Code
import plotly.express as px
fig = px.scatter(pdf, x="datetime", y="CO2_ppm")
fig.show()
#%%% Find max min points
from scipy.signal import argrelextrema                                          # Calculates the relative extrema of data.
n = 10                                                                          # How many points on each side to use for the comparison to consider comparator(n, n+x) to be True.

df['max'] = df.iloc[argrelextrema(df['CO2_ppm'].values, np.greater_equal,\
                                  order=n)[0]]['CO2_ppm']                       # Gives all the peaks 
df['min'] = df.iloc[argrelextrema(df['CO2_ppm'].values, np.less_equal,\
                                  order=n)[0]]['CO2_ppm']                       # Gives all the valleys
    
df['max'].plot(marker='o', ax = ax)                                             # This needs to be verified with the graph if python recognizes all peaks
df['min'].plot(marker="v", ax = ax)                                             # - and valleys. If not adjust the n value.

#%%% Filter supply and exhaust phases 
df.loc[df['min'] > -400, 'mask'] = False                                        # Marks all min as False                         
df.loc[df['max'] > 0, 'mask'] = True                                            # Marks all min as True
df["mask"] = df["mask"].fillna(method='ffill').astype("bool")                   # Use forward to fill True and False 
df = df.dropna(subset= ["mask"])
df["sup"] = df["mask"]                                                          # Create seperate columns for sup and exhaust
df["exh"] = df["mask"]



df.loc[df['min'] > 0, 'sup'] = True                                             # The valleys have to be belong to supply as well 
df.loc[df['max'] > 0, 'exh'] = False                                            # The peaks have to belong to max, before it was all filled be backfill



df_sup = df.loc[df["sup"].to_list()]                                            

a = df_sup.resample("5S").mean()                                                # Resampled beacuase, the data will be irregular
plt.figure() 
#%%% Plot supply                                                                   # This can be verified from this graph        
a["CO2_ppm"].plot(title = "supply") 
df_sup2 = a.loc[:,["CO2_ppm"]]

df_exh = df.loc[~df["exh"].values]
b = df_exh.resample("5S").mean()
plt.figure()

#%%% Plot exhaust
b["CO2_ppm"].plot(title = "exhaust")                                            # Similar procedure is repeated from exhaust
df_exh2 = b.loc[:,["CO2_ppm"]]

#%%% Plot for extra prespective
fig,ax = plt.subplots()

df_sup.plot(y="CO2_ppm", style="yv-", ax = ax, label = "supply")
df_exh.plot(y="CO2_ppm", style="r^-", ax = ax, label = "exhaust")



#%% Marking dataframes supply
"""Marks every supply dataframe with a number for later anaysis """
n = 1
df_sup3 = df_sup2.copy().reset_index()

start_date = str(t0); end_date = tn # CHANGE HERE 

mask = (df_sup3['datetime'] > start_date) & (df_sup3['datetime'] <= end_date)

df_sup3 = df_sup3.loc[mask]


for i,j in df_sup3.iterrows():
    try:
        # print(not pd.isnull(j["CO2_ppm"]), (np.isnan(df_sup3["CO2_ppm"][i+1])))
        if (not pd.isnull(j["CO2_ppm"])) and (np.isnan(df_sup3["CO2_ppm"][i+1])):
            df_sup3.loc[i,"num"] = n
            n = n+1
        elif (not pd.isnull(j["CO2_ppm"])):
            df_sup3.loc[i,"num"] = n
    except KeyError:
        print("ignore the key error")
    

df_sup_list = []
for i in range(1, int(df_sup3.num.max()+1)):
    df_sup_list.append(df_sup3.loc[df_sup3["num"]==i])

#%%% Supply tau 
# This method can be replicated in excel for crossreference
"""Calculates tau based in ISO 16000-8"""
df_tau_sup = []
for idf in df_sup_list:
    if len(idf) > 3:
        a = idf.reset_index(drop = True)
        a["log"] = np.log(a["CO2_ppm"])
        
        diff = (a["datetime"][1] - a["datetime"][0]).seconds
        
        
        a["runtime"] = np.arange(0,len(a) * diff, diff)
        
        a["t-te"] = a["runtime"] - a["runtime"][len(a)-1]
        
        a["lnte/t"] = a["log"] - a["log"][len(a)-1]
        
        a["slope"] = a["lnte/t"] / a["t-te"]
        slope_sup = a["slope"].mean()
        #############
        x1 = a["runtime"].values
        y1 = a["log"].values
        from scipy.stats import linregress
        slope = linregress(x1,y1)[0]
        ############
        a.loc[[len(a)-1], "slope"] = abs(slope_sup)
        
        sumconz = a["CO2_ppm"].iloc[1:-1].sum()
        
        tail = a["CO2_ppm"][len(a)-1]/abs(slope_sup)
        area_sup_1= (diff * (a["CO2_ppm"][0]/2 + sumconz +a["CO2_ppm"][len(a)-1]/2))
        from numpy import trapz
        area_sup_2 = trapz(a["CO2_ppm"].values, dx=diff) # proof that both methods have same answer
        

        tau = (area_sup_2 + tail)/a["CO2_ppm"][0]
        a["tau_sec"] = tau
        df_tau_sup.append(a)
    else:
        pass
#%%% Supply final tau
tau_list_sup = []
for h in df_tau_sup:
    tau_list_sup.append(h["tau_sec"][0])

tau_s = np.mean(tau_list_sup)

    
#%% Marking dataframes exhaust
"""Marks every exhaust dataframe with a number for later anaysis """

n = 1
df_exh3 = df_exh2.copy().reset_index()


mask = (df_exh3['datetime'] > start_date) & (df_exh3['datetime'] <= end_date)

df_exh3 = df_exh3.loc[mask]


for i,j in df_exh3.iterrows():
    try:
        # print(not pd.isnull(j["CO2_ppm"]), (np.isnan(df_exh3["CO2_ppm"][i+1])))
        if (not pd.isnull(j["CO2_ppm"])) and (np.isnan(df_exh3["CO2_ppm"][i+1])):
            df_exh3.loc[i,"num"] = n
            n = n+1
        elif (not pd.isnull(j["CO2_ppm"])):
            df_exh3.loc[i,"num"] = n
    except KeyError:
        print("ignore the key error")

df_exh_list = []
for i in range(1, int(df_exh3.num.max()+1)):
    df_exh_list.append(df_exh3.loc[df_exh3["num"]==i])
#%%% Exhaust tau
# this method can be replicated in Excel for crossverification
"""Calculates tau based in area under the curve"""

df_tau_exh = []
for e in df_exh_list:
    if len(e) > 3:
        b = e.reset_index(drop = True)
        b["log"] = np.log(b["CO2_ppm"])
        
        diff = (b["datetime"][1] - b["datetime"][0]).seconds
        
        
        b["runtime"] = np.arange(0,len(b) * diff, diff)
        
        b["t-te"] = b["runtime"] - b["runtime"][len(b)-1]
        
        b["lnte/t"] = b["log"] - b["log"][len(b)-1]
        
        b["slope"] = b["lnte/t"] / b["t-te"]
        slope = b["slope"].mean()
        #############
        x = b["runtime"].values
        y = b["log"].values
        from scipy.stats import linregress
        slope = linregress(x,y)[0]
        ############
        b.loc[[len(b)-1], "slope"] = abs(slope)
        
        sumconz = b["CO2_ppm"].iloc[1:-1].sum()
        
        tail = b["CO2_ppm"][len(b)-1]/abs(slope)
        area1= (diff * (b["CO2_ppm"][0]/2 + sumconz +b["CO2_ppm"][len(b)-1]/2))
        from numpy import trapz
        area2 = trapz(b["CO2_ppm"].values, dx=diff)                             # proof that both methods have same answer
        
        tau2 = (area2 + tail)/b["CO2_ppm"][len(b)-1]
        b["tau_sec"] = tau2
        df_tau_exh.append(b)
    else:
        pass
#%%% Exhaust final tau 
tau_list_exh = []
for jdf in df_tau_exh:
    tau_list_exh.append(jdf["tau_sec"][0])

tau_e = np.mean(tau_list_exh)



#%% Final result print
prYellow("Recirculation:  {} %".format(round((tau_s/tau_e)*100) )  )









