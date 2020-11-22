# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:50:19 2020

@author: sovik
"""

import pandas as pd
import numpy as np

#from statsmodels.tsa.ar_model import AR
#from tqdm import tqdm#, tqdm_notebook
import matplotlib.pyplot as plt 

from sklearn.metrics import r2_score
import statsmodels.api as sm
import itertools


from sklearn.metrics import mean_squared_error
from math import sqrt


#%%

file1 = pd.read_table('HDeviceCGM.txt', sep ='|')#, parse_dates=['DexInternalTm'])

#%%

file1colnames = pd.DataFrame(list(file1.columns))

#%%

lenfile1colnames = len(file1colnames)
list_of_uniques = {}

#%%

for i in range(lenfile1colnames):
    abc = file1[file1colnames.iloc[i,0]].unique()
    xyz = file1colnames.iloc[i,0]+"_unique"
    list_of_uniques[file1colnames.iloc[i,0]] = abc
    #file1colnames.iloc[i,0]+"_unique" = pd.DataFrame(abc)
    #print(i)
    #print(abc)
    
#%%
    
file11 = file1[['SiteID','RecordType', 'PtID', 'ParentHDeviceUploadsID']].drop_duplicates()#.reset_index
file11 = file11.reset_index()
    
#%%
    
test = file1[(file1['SiteID'] == file11['SiteID'].iloc[1]) & (file1['RecordType'] == file11['RecordType'].iloc[1]) & (file1['PtID'] == file11['PtID'].iloc[1]) & (file1['ParentHDeviceUploadsID'] == file11['ParentHDeviceUploadsID'].iloc[1])]
test['DexInternalDtTmDaysFromEnroll'] = -test['DexInternalDtTmDaysFromEnroll']
abcunique = pd.DataFrame(test['DexInternalDtTmDaysFromEnroll'].unique()).sort_values(by=0,ascending=True)
abcuniquelen = len(abcunique)
abcunique = abcunique.reset_index(drop=True)
abcunique['Day'] = pd.Series(list(range(1,abcuniquelen+1, 1)))
test = test.sort_values(by=['DexInternalDtTmDaysFromEnroll','DexInternalTm'], ascending=True)

for i in range(abcuniquelen):
    test.loc[(test['DexInternalDtTmDaysFromEnroll'] == abcunique.iloc[i,0]),'DexInternalDtTmDaysFromEnroll2']=abcunique.iloc[i,1]

test['DateTime'] = test['DexInternalDtTmDaysFromEnroll2'].astype(int).astype(str) + ' days ' + test['DexInternalTm'].astype(str)

index = test['DateTime']
test.index=pd.TimedeltaIndex(test['DateTime'])
test = test.drop('DateTime', 1)

test.reset_index(level=0, inplace=True)

test['DateTime']=test['DateTime'].dt.round("5min")
index = test['DateTime']
test.index=pd.TimedeltaIndex(test['DateTime'])

#%%
test['DateTimeHours'] = test['DateTime'].dt.components.hours
test['DateTimeMinutes'] = test['DateTime'].dt.components.minutes

#%%

dfdisttrain_df= test.groupby(['DateTimeHours', 'DateTimeMinutes'], as_index=False)['GlucoseValue'].mean()
dfdistcounttrain_df= test.groupby(['DateTimeHours', 'DateTimeMinutes'], as_index=False)['GlucoseValue'].count()

dfdistcounttrain_df.rename(columns={'GlucoseValue':'GlucoseValueCount'}, inplace=True)

#dfdistalltrain_df = pd.merge(dfdisttrain_df, dfdistcounttrain_df, on=['DateTimeMinutes'])

#%%

test2 = test[(test['DexInternalDtTmDaysFromEnroll2'] == 3)]

#%%

y1=pd.DataFrame(test2['GlucoseValue'])
y2=pd.DataFrame(dfdisttrain_df['GlucoseValue'])

#=y1

#%%
#
#test_size = 30
#train_size=int(len(y)) - test_size
##print(test_size)
#
##%%
#
#traindf=y[0:train_size] 
#testdf=y[train_size:train_size+test_size]

#%%

ts = y1#test['GlucoseValue']
#ts = np.array(ts, dtype=float)

test_size = 12
train_size=int(len(ts)) - test_size

traindf=ts[0:train_size] 
testdf=ts[train_size:train_size+test_size]

#%%
#y_hat_avg = testdf.copy()
#y_hat_avg['avg_forecast'] = traindf['GlucoseValue'].mean()
#plt.figure(figsize=(12,8))
plt.plot(traindf['GlucoseValue'], label='Train')
plt.plot(testdf['GlucoseValue'], label='Test')
#plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()

#%%

#pd.plotting.autocorrelation_plot(traindf)


#%%

#from statsmodels.tsa.seasonal import seasonal_decompose
#components = seasonal_decompose(ts, freq=60, model='multiplicative')
#components.plot()

#sm.graphics.tsa.plot_acf(test['GlucoseValue'], lags=40)#, ax=axes[0])

#%%
#import statsmodels.api as sm


# Grid Search
p = d = q = range(0,3) # p, d, and q can be either 0, 1, or 2
pdq = list(itertools.product(p,d,q)) # gets all possible combinations of p, d, and q 
p2 = d2 = q2 = range(0, 2) # second set of p's, d's, and q's
s = range(10, 12) # Range between 8 and 13, last best was 11, befor that 9
#pdq2 = list(itertools.product(p2,d2,q2)) # simular too code above but for seasonal parameters
pdq2 = list(itertools.product(p2,d2,q2,s)) # simular too code above but for seasonal parameters
#s = range(0, 10) # here I use twelve but the number here is representative of the periodicty of the seasonal cycle
#pdqs2 = [(c[0], c[1], c[2], s) for c in pdq2]
pdqs2 = [(c[0], c[1], c[2], c[3]) for c in pdq2]
 
combs = {}
aics = []
# Grid Search Continued
for combination in pdq:
    for seasonal_combination in pdqs2:
        try:
            model = sm.tsa.statespace.SARIMAX(traindf, order=combination, seasonal_order=seasonal_combination,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
            model = model.fit()
            combs.update({model.aic : [combination, seasonal_combination]})
            aics.append(model.aic)
            
        except:
            continue
            
best_aic = min(aics)

#%%
# Modeling and forcasting
model = sm.tsa.statespace.SARIMAX(traindf, order=combs[best_aic][0], seasonal_order=combs[best_aic][1],
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
model = model.fit()

#%%
pred = pd.DataFrame(model.forecast(12).round(1))
pred.index = testdf.index

predtest = testdf.merge(pred, left_index=True, right_index=True)

#%%

plt.plot(traindf['GlucoseValue'], label='Train')
plt.plot(testdf['GlucoseValue'], label='Test')
plt.plot(pred[0], label='Forecast')
plt.legend(loc='best')
plt.show()

#%%


r2test = r2_score(testdf['GlucoseValue'], pred)

#%%

#import numpy as np

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

mapetest=mape(testdf['GlucoseValue'], pred)

#%%

rmstest = sqrt(mean_squared_error(testdf['GlucoseValue'], pred))


#%%
#%%
#%%
#import matplotlib.pyplot as plt 
#
##Plotting data
#traindf.GlucoseValue.plot(figsize=(15,8), fontsize=14)
#testdf.GlucoseValue.plot(figsize=(15,8), fontsize=14)
#plt.show()
#
##%%
#
#y_hat_avg = testdf.copy()
#y_hat_avg['avg_forecast'] = traindf['GlucoseValue'].mean()
#plt.figure(figsize=(12,8))
#plt.plot(traindf['GlucoseValue'], label='Train')
#plt.plot(testdf['GlucoseValue'], label='Test')
#plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
#plt.legend(loc='best')
#plt.show()
#
##%%
#
#y_hat_avg = testdf.copy()
#y_hat_avg['moving_avg_forecast'] = traindf['GlucoseValue'].rolling(60).mean().iloc[-1]
#plt.figure(figsize=(16,8))
#plt.plot(traindf['GlucoseValue'], label='Train')
#plt.plot(testdf['GlucoseValue'], label='Test')
#plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
#plt.legend(loc='best')
#plt.show()
#
##%%
#
#import statsmodels.api as sm
#sm.tsa.seasonal_decompose(traindf.GlucoseValue, freq=1).plot()
##result = sm.tsa.stattools.adfuller(traindf.GlucoseValue)
#plt.show()
##%%
#from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
#import numpy as np
#
#y_hat_avg = testdf.copy()
#fit1 = ExponentialSmoothing(np.asarray(traindf['GlucoseValue']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
#y_hat_avg['Holt_Winter'] = fit1.forecast(len(testdf))
#plt.figure(figsize=(16,8))
#plt.plot(traindf['GlucoseValue'], label='Train')
#plt.plot(testdf['GlucoseValue'], label='Test')
#plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
#plt.legend(loc='best')
#plt.show()
#
##%%
#
#y_hat_avg = testdf.copy()
#fit1 = sm.tsa.statespace.ARIMAX(traindf['GlucoseValue'], order=(2, 1, 4)).fit()
#y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
#plt.figure(figsize=(16,8))
#plt.plot( train['Count'], label='Train')
#plt.plot(test['Count'], label='Test')
#plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
#plt.legend(loc='best')
#plt.show()
#
##%%
##%%
#
#import warnings
#import itertools
#import numpy as np
#import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")
#plt.style.use('fivethirtyeight')
#import pandas as pd
#import statsmodels.api as sm
#import matplotlib
#
##%%
#
#p = d = q = range(0, 2)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
##print('Examples of parameter combinations for Seasonal ARIMA...')
##print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
##print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
##print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
##print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#
##%%
#
#from statsmodels.tsa.stattools import adfuller
#import matplotlib.pylab as plt
#
#def test_stationarity(timeseries):
#    
#    #Determing rolling statistics
#    #rolmean = pd.rolling_mean(timeseries, window=12)
#    rolmean = timeseries.rolling(30).mean()
#    #rolstd = pd.rolling_std(timeseries, window=12)
#    rolstd = timeseries.rolling(30).std()
##Plot rolling statistics:
#    plt.plot(timeseries, color='blue',label='Original')
#    plt.plot(rolmean, color='red', label='Rolling Mean')
#    plt.plot(rolstd, color='black', label = 'Rolling Std')
#    plt.legend(loc='best')
#    plt.title('Rolling Mean & Standard Deviation')
#    plt.show()
#    #Perform Dickey-Fuller test:
#    print('Results of Dickey-Fuller Test:')
#    dftest = adfuller(timeseries, autolag='AIC')
#    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#    for key,value in dftest[4].items():
#        dfoutput['Critical Value (%s)'%key] = value
#    print(dfoutput)
#
##%%
#y=test['GlucoseValue']
#
#
##%%
#
#import warnings
#import itertools
#import numpy as np
#import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")
#plt.style.use('fivethirtyeight')
#import pandas as pd
#import statsmodels.api as sm
#import matplotlib
#
#matplotlib.rcParams['axes.labelsize'] = 14
#matplotlib.rcParams['xtick.labelsize'] = 12
#matplotlib.rcParams['ytick.labelsize'] = 12
#matplotlib.rcParams['text.color'] = 'k'
#
##%%
#from pylab import rcParams
#rcParams['figure.figsize'] = 18, 8
#
#decomposition = sm.tsa.seasonal_decompose(y, model='additive')
#fig = decomposition.plot()
#plt.show()
#
##%%
#
#p = d = q = range(0, 2)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#
#print('Examples of parameter combinations for Seasonal ARIMA...')
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#
##%%
#for param in pdq:
#    for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(y,
#                                            order=param,
#                                            seasonal_order=param_seasonal,
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#
#            results = mod.fit()
#
#            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#        except:
#            continue
#
##%%
#            
#mod = sm.tsa.statespace.SARIMAX(y,
#                                order=(1, 1, 1),
#                                seasonal_order=(1, 1, 1, 12),
#                                enforce_stationarity=False,
#                                enforce_invertibility=False)
#
#results = mod.fit()
#
#print(results.summary().tables[1])
#
##%%
#
#results.plot_diagnostics(figsize=(16, 8))
#plt.show()
#
##%%
#
#train_size=int(len(y) *0.80)
#test_size = int(len(y)) - train_size
#print(test_size)
#
#
##%%
#
##pred = results.get_forecast(start='17 days 22:15:00', dynamic=False)
#
##pred= results.predict(start =train_size, end=train_size+test_size+(steps)-1)
#pred = results.forecast(steps=6)
#
#pred_ci = pred.conf_int()
#
#ax = y.plot(label='observed')
#pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
#
#ax.fill_between(pred_ci.index,
#                pred_ci.iloc[:, 0],
#                pred_ci.iloc[:, 1], color='k', alpha=.2)
#
#ax.set_xlabel('Date')
##ax.set_ylabel('Furniture Sales')
#plt.legend()
#
#plt.show()
#    
##%%
#    
#test_stationarity(ts)
#
##%%
#
#from pmdarima.arima import auto_arima
#
#
##%%
##from random import randrange
##from pandas import Series
#from matplotlib import pyplot
#from statsmodels.tsa.seasonal import seasonal_decompose
##series = [i+randrange(10) for i in range(1,100)]
#result = seasonal_decompose(dfdisttrain_df['GlucoseValue'], model='multiplicative')
#result.plot()
#pyplot.show()
#
##%%
#
#testtimeseries = test
#
##%%
#
#split_date ='15 days 00:00:00'
#df_training = testtimeseries.loc[testtimeseries.index <= split_date]
#df_test = testtimeseries.loc[testtimeseries.index > split_date]
#
##test = test.DatetimeIndex('DexInternalTm')
#
##%%
##from tqdm import tqdm, tqdm_notebook
#
#resultsDict={}
#predictionsDict={}
#
##%%
#
#from utils.metrics import evaluate
#from utils.plots import bar_metrics
#
#import matplotlib.pyplot as plt
#
#
##%%
#
## Walk throught the test data, training and predicting 1 day ahead for all the test data
#index = len(df_training)
#yhat = list()
#for t in tqdm(range(len(df_test.GlucoseValue))):
#    temp_train = testtimeseries[:len(df_training)+t]
#    model = AR(temp_train.GlucoseValue)
#    model_fit = model.fit()
#    predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
#    yhat = yhat + [predictions]
#    
#yhat = pd.concat(yhat)
#resultsDict['AR'] = evaluate(df_test.GlucoseValue, yhat.values)
#predictionsDict['AR'] = yhat.values
#
##%%
#
#
#plt.plot(df_test.GlucoseValue.values, label='Original')
#plt.plot(yhat.values,color='red',label='AR predicted')
#plt.legend()
#
##%%
#test1 = test.GlucoseValue.resample('30Min').mean()
#
##%%
##%%
##%%
#
#testtimeseries = pd.DataFrame(test['GlucoseValue'])
#
##%%
#
#
##Set up helper function for data visualization 
#def plt_(testtimeseries, title):    
#    plt.figure(figsize=(12,6))
#    plt.plot(testtimeseries, color = 'b')
#    #plt.ylabel('Tonnes')
#    plt.title(title)
#    plt.show()
#
#plt_(testtimeseries, "GlucoseValue")
#
##%%
##%%
#
#
#
#ax = df_test.GlucoseValue#.plot(figsize = (16,5), title = "Monthly Beer Production")
##ax.set(xlabel='Dates', ylabel='Total Production')
#
##%%
##test['DexInternalTm'] = pd.to_datetime(test['DexInternalTm'])#, unit='s')
##test['DexInternalTm2']=test['DexInternalTm'].dt.round("5min")
#
##%%
#
##test['DexInternalDtTmDaysFromEnroll2'] = 0
#
##%%
##test['DexInternalDtTmDaysFromEnroll2'] = test[(test['DexInternalDtTmDaysFromEnroll'] == abcunique[i])]
#
##%%
#
##for i in range(abcuniquelen):
##    test.loc[(test['DexInternalDtTmDaysFromEnroll'] == abcunique.iloc[i,0]),'DexInternalDtTmDaysFromEnroll2']=abcunique.iloc[i,1]
#
#
##%%
##index = test['DexInternalTm']
##test.index=pd.TimedeltaIndex(test['DexInternalTm'])
#
##test = test.DatetimeIndex('DexInternalTm')
#
##index=pd.DatetimeIndex(index)
##%%
#
#list_of_testlen = {}
#
#for i in range(abcuniquelen):
#    test10 = test[(test['DexInternalDtTmDaysFromEnroll'] == abcunique[i])]
#    test1 = test10.GlucoseValue.resample('5Min').mean()
#    test1 = test1.reset_index(drop=True)
#    list_of_testlen[abcunique[i]] = test1
#
##%%
#
#res = {k: v for k, v in list_of_testlen.items() if len(v) >= 200}
#
##%%
#
#abc123123 = pd.DataFrame.from_dict(res, orient='index')
#
##%%
#
#abc123123123 = abc123123.interpolate(method='linear', limit_direction='forward', axis=1)
#
##%%
##lengths = [len(v) for v in list_of_testlen.values()]
#    
#
##%%
#
##test1 = test['DexInternalTm'].resample('1Min')#, how='mean')
#test1 = pd.DataFrame(test.resample('1Min'))#, how='mean')
#
#
##%%
#
#test.to_csv('testfile.csv', index = False)
#
#
##%%
#
#a = list_of_uniques['PtID']
