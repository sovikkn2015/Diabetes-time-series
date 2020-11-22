# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 03:23:53 2020

@author: sovik
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:50:19 2020

@author: sovik
"""

import pandas as pd

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



#%%

file1 = pd.read_table('HDeviceWizard.txt', sep ='|')#, parse_dates=['DexInternalTm'])

#%%

file1colnames = pd.DataFrame(list(file1.columns))

#%%

lenfile1colnames = len(file1colnames)

list_of_uniques = {}

#%%

file11 = file1[['SiteID', 'PtId', 'ParentHDeviceUploadsID']].drop_duplicates()#.reset_index
file11 = file11.reset_index()

#%%

test = file1[(file1['SiteID'] == file11['SiteID'].iloc[1]) & (file1['PtId'] == file11['PtId'].iloc[1]) & (file1['ParentHDeviceUploadsID'] == file11['ParentHDeviceUploadsID'].iloc[1])]

#%%

#a = test['DeviceTm']

#%%
#test['DeviceTm'] = map(abs, test['DeviceTm'])

test['DeviceDtTmDaysFromEnroll'] = test['DeviceDtTmDaysFromEnroll'].astype(int).apply(lambda x: abs(x))
abcunique = pd.DataFrame(test['DeviceDtTmDaysFromEnroll'].unique()).sort_values(by=0,ascending=True)
abcuniquelen = len(abcunique)
abcunique = abcunique.reset_index(drop=True)
abcunique['Day'] = pd.Series(list(range(1,abcuniquelen+1, 1)))
test = test.sort_values(by=['DeviceDtTmDaysFromEnroll','DeviceTm'], ascending=True)

for i in range(abcuniquelen):
    test.loc[(test['DeviceDtTmDaysFromEnroll'] == abcunique.iloc[i,0]),'DeviceDtTmDaysFromEnroll2']=abcunique.iloc[i,1]

test['DateTime'] = test['DeviceDtTmDaysFromEnroll2'].astype(int).astype(str) + ' days ' + test['DeviceTm'].astype(str)

index = test['DateTime']
test.index=pd.TimedeltaIndex(test['DateTime'])
test = test.drop('DateTime', 1)

test.reset_index(level=0, inplace=True)

test['DateTime']=test['DateTime'].dt.round("15min")
index = test['DateTime']
test.index=pd.TimedeltaIndex(test['DateTime'])

test['DeviceTmHours'] = test['DateTime'].dt.components.hours

#%%

## DataFrame for both the sample time series as well as average overall time for the selected combination

## Generating average of GlucoseValue for daily level for sample combination using all days to enroll
test['DeviceTmHours'] = test['DateTime'].dt.components.hours

dfdist_df= test.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].mean()
dfdistcount_df= test.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].count()
dfdistcount_df.rename(columns={'CarbInput':'CarbInputCount', 'BgInput':'BgInputCount'}, inplace=True)
dfdistall_df = pd.merge(dfdist_df, dfdistcount_df, on=['DeviceTmHours'])

plt.scatter(dfdistall_df['DeviceTmHours'], dfdistall_df['CarbInput'])



#%%

train_df, test_df = train_test_split(test, test_size=0.2, random_state=100)

#%%

#
#import numpy
#import matplotlib.pyplot as plt
#from sklearn.metrics import r2_score
#
##x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
##y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
#
#mymodel = numpy.poly1d(numpy.polyfit(train_df['DeviceTmHours'], train_df['CarbInput'], 7))
#
#myline = numpy.linspace(0, 23, 100)
#
#plt.scatter(train_df['DeviceTmHours'], train_df['CarbInput'])
#plt.plot(myline, mymodel(myline))
#plt.show()
#
#r2 = r2_score(train_df['CarbInput'], mymodel(train_df['DeviceTmHours']))


#%%

# Hours vs carbInput - Train
dfdisttrain_df= train_df.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].mean()
dfdistcounttrain_df= train_df.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].count()

dfdistcounttrain_df.rename(columns={'CarbInput':'CarbInputCount', 'BgInput':'BgInputCount'}, inplace=True)

dfdistalltrain_df = pd.merge(dfdisttrain_df, dfdistcounttrain_df, on=['DeviceTmHours'])

#%%


# Hours vs carbInput -Test
dfdisttest_df= test_df.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].mean()
dfdistcounttest_df= test_df.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].count()

dfdistcounttest_df.rename(columns={'CarbInput':'CarbInputCount', 'BgInput':'BgInputCount'}, inplace=True)

dfdistalltest_df = pd.merge(dfdisttest_df, dfdistcounttest_df, on=['DeviceTmHours'])

#%%

#%%



#x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
#y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(dfdistalltrain_df['DeviceTmHours'], dfdistalltrain_df['CarbInput'], 3))

myline = numpy.linspace(0, 23, 100)


#%%

plt.scatter(dfdistalltrain_df['DeviceTmHours'], dfdistalltrain_df['CarbInput'], label='Train CarbInput')
#plt.scatter(test_df['DeviceTmHours'], test_df['CarbInput'])
plt.scatter(dfdistalltest_df['DeviceTmHours'], dfdistalltest_df['CarbInput'], label='Test CarbInput')
plt.plot(myline, mymodel(myline))
plt.legend(loc='best')

plt.show()


#%%
r2train = r2_score(dfdistalltrain_df['CarbInput'], mymodel(dfdistalltrain_df['DeviceTmHours']))
#r2test = r2_score(test_df['CarbInput'], mymodel(test_df['DeviceTmHours']))
r2test = r2_score(dfdistalltest_df['CarbInput'], mymodel(dfdistalltest_df['DeviceTmHours']))


#%%
#%%
#%%

plt.scatter(dfdistalltest_df['DeviceTmHours'], dfdistalltest_df['CarbInput'])
plt.plot(myline, mymodel(myline))
plt.show()


#%%

# Hours vs carbInput
dfdist= test.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].mean()
dfdistcount= test.groupby('DeviceTmHours', as_index=False)['CarbInput', 'BgInput'].count()

dfdistcount.rename(columns={'CarbInput':'CarbInputCount', 'BgInput':'BgInputCount'}, inplace=True)

dfdistall = pd.merge(dfdist, dfdistcount, on=['DeviceTmHours'])

#%%
import seaborn as sns

#df = sns.load_dataset("iris")
sns.pairplot(dfdistall)#, hue="species")
   
#%%

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
#y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(dfdistall['DeviceTmHours'], dfdistall['CarbInput'], 7))

myline = numpy.linspace(0, 23, 100)

plt.scatter(dfdistall['DeviceTmHours'], dfdistall['CarbInput'])
plt.plot(myline, mymodel(myline))
plt.show()

r2 = r2_score(dfdistall['CarbInput'], mymodel(dfdistall['DeviceTmHours']))

#%%

a =test['DateTime'].dt.components.hours

#%%
test['DeviceTm2']=test['DeviceTm'].dt.round("60min")


#test['DeviceTm'].abs()

#%%
#file1.to_csv('HDeviceWizard.csv', index = False)

#%%


