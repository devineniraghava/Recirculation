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

import pandas as pd
import numpy as np
from statistics import mean
import time
import datetime as dt
import matplotlib.pyplot as plt
from tabulate import tabulate

# import mysql.connector
import os
from sqlalchemy import create_engine

from easygui import *
# functions to print in colour
def prRed(skk): print("\033[31;1;m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[33;1;m {}\033[00m" .format(skk)) 

from uncertainties import ufloat
# engine = create_engine("mysql+pymysql://root:Password123@localhost/",pool_pre_ping=True)
engine = create_engine("mysql+pymysql://wojtek:Password#102@wojtek.mysql.database.azure.com/",pool_pre_ping=True)

#%% Function import
# syntax to import a function from any folder $ very important
import sys  
sys.path.append("C:/Users/Devineni/OneDrive - bwedu/4_Recirculation/python_files/")  
from Outdoor_CO2 import outdoor

#%% Control plot properties

# this is for the precise date formatter
import matplotlib.dates as mdates

import matplotlib.units as munits
# this syntax controls the plot properties, more attributes can be addes and -
# - removed depending on the requirement
from pylab import rcParams
rcParams['figure.figsize'] = 7,4.5
plt.rcParams["font.family"] = "calibri"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.size"] = 10
	
plt.close("all")

#%%
 # in ppm
i = 17 # to select the experiment
j = 0 # to select the sensor in the ventilation device

# time = pd.read_excel("C:/Users/Devineni/OneDrive - bwedu/4_Recirculation/Times_thesis.xlsx", sheet_name="Timeframes")
# The dataframe time comes from the excel sheet in the path above, to edit go -
# - this excel sheet edit and upload it to mysql
time = pd.read_sql_query("SELECT * FROM testdb.timeframes;", con = engine)
start, end = str(time["Start"][i] - dt.timedelta(minutes=20)), str(time["End"][i])
t0 = time["Start"][i]

table = time["tables"][i].split(",")[j] # CHANGE HERE

dum = [["Experiment",time["short_name"][i] ], ["Sensor", table]]
print(tabulate(dum))

database = time["database"][i]


background, dummy = outdoor(str(t0), str(end), plot = False)
background = background["CO2_ppm"].mean()

df = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND\
                       '{}'".format(database, table, start, end), con = engine)
df = df.loc[:,["datetime", "CO2_ppm"]]
df["original"] = df["CO2_ppm"] 

df["CO2_ppm"] = df["CO2_ppm"] - background + 30
df = df.loc[~df.duplicated(subset=["datetime"])]

df = df.set_index("datetime")
while not(t0 in df.index.to_list()):
    t0 = t0 + dt.timedelta(seconds=1)
    print(t0)

df["roll"] = df["CO2_ppm"].rolling(24).mean()



c0 = df["CO2_ppm"].loc[t0]
Cend37 = round((c0 - background)*0.37, 2)

cend = df.loc[df["roll"].le(Cend37)]

if len(cend) == 0:
    tn = str(df.index[-1])
else:
    tn = str(cend.index[0])


fig,ax = plt.subplots()
df.plot(title = "original", color = [ 'green', 'silver'], ax = ax)
from scipy.signal import argrelextrema
n = 10


df['max'] = df.iloc[argrelextrema(df['CO2_ppm'].values, np.greater_equal, order=n)[0]]['CO2_ppm']
df['min'] = df.iloc[argrelextrema(df['CO2_ppm'].values, np.less_equal, order=n)[0]]['CO2_ppm']
df['max'].plot(marker='o', ax = ax)
df['min'].plot(marker="v", ax = ax)


df.loc[df['min'] > -400, 'mask'] = False

df.loc[df['max'] > 0, 'mask'] = True

df["mask"] = df["mask"].fillna(method='ffill').astype("bool")

df = df.dropna(subset= ["mask"])

df["sup"] = df["mask"];df["exh"] = df["mask"]



df.loc[df['min'] > 0, 'sup'] = True
df.loc[df['max'] > 0, 'exh'] = False



df_sup = df.loc[df["sup"].to_list()]

a = df_sup.resample("5S").mean()
plt.figure()
a["CO2_ppm"].plot(title = "supply")
df_sup2 = a.loc[:,["CO2_ppm"]]

df_exh = df.loc[~df["exh"].values]
b = df_exh.resample("5S").mean()
plt.figure()



b["CO2_ppm"].plot(title = "exhaust")
df_exh2 = b.loc[:,["CO2_ppm"]]

# Plot for extra prespective
# fig,ax = plt.subplots()


# df_sup.plot(y="CO2_ppm", style="yv-", ax = ax, label = "supply")
# df_exh.plot(y="CO2_ppm", style="r^-", ax = ax, label = "exhaust")



#%%

from pandas.api.types import is_numeric_dtype
n = 1
df_sup3 = df_sup2.copy().reset_index()

start_date = str(t0); end_date = tn # CHANGE HERE 

mask = (df_sup3['datetime'] > start_date) & (df_sup3['datetime'] <= end_date)

df_sup3 = df_sup3.loc[mask]


for i,j in df_sup3.iterrows():
    try:
        print(not pd.isnull(j["CO2_ppm"]), (np.isnan(df_sup3["CO2_ppm"][i+1])))
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

#%% supply

df_tau_sup = []
for df in df_sup_list:
    if len(df) > 3:
        a = df.reset_index(drop = True)
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
        

        tau = ((diff * (a["CO2_ppm"][0]/2 + sumconz +a["CO2_ppm"][len(a)-1]/2)) + tail)/a["CO2_ppm"][0]
        a["tau_sec"] = tau
        df_tau_sup.append(a)
    else:
        pass
#%%
tau_list_sup = []
for h in df_tau_sup:
    tau_list_sup.append(h["tau_sec"][0])

tau_s = np.mean(tau_list_sup)

    
#%% exhaust
n = 1
df_exh3 = df_exh2.copy().reset_index()


mask = (df_exh3['datetime'] > start_date) & (df_exh3['datetime'] <= end_date)

df_exh3 = df_exh3.loc[mask]


for i,j in df_exh3.iterrows():
    try:
        print(not pd.isnull(j["CO2_ppm"]), (np.isnan(df_exh3["CO2_ppm"][i+1])))
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
        area2 = trapz(b["CO2_ppm"].values, dx=diff) # proof that both methods have same answer
        

        tau2 = ((diff * (b["CO2_ppm"][0]/2 + sumconz +b["CO2_ppm"][len(b)-1]/2)) + tail)/b["CO2_ppm"][len(b)-1]
        b["tau_sec"] = tau2
        df_tau_exh.append(b)
    else:
        pass
#%%
tau_list_exh = []
for df in df_tau_exh:
    tau_list_exh.append(df["tau_sec"][0])

tau_e = np.mean(tau_list_exh)








#%%
# from sqlalchemy import create_engine

# engine = create_engine("mysql+pymysql://remoteroot:Password123@Raghavakrishna-PC/",pool_pre_ping=True)

# df = pd.read_sql_query("select * from mysql.user;", con = engine)








#%%
prYellow("Recirculation:  {} %".format(round((tau_s/tau_e)*100) )  )









