# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:14:08 2019

@author: Sadil Khan
"""
#  importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Defining Cross-Validation
def cross_validation(estimator,x,y,scoring,cv):
    score=cross_val_score(estimator=estimator,X=x,y=y,scoring=scoring,cv=cv,n_jobs=-1)
    return score.mean(),score.var()


# Defining Grid_Search Cv
def gridsearch(estimator,parameter):
    grid=GridSearchCV(estimator,param_grid=parameter,scoring='accuracy',cv=5,n_jobs=-1)
    grid.fit(x,y)
    return (grid.best_score_,grid.best_params_)
    
# importing Datasets
match=pd.read_csv("matches.csv")
bowl=pd.read_csv("deliveries.csv")

batsmen=['LMP Simmons' ,'MN Samuels', 'Swapnil Singh', 'R Tewatia' ,'MM Patel',
 'SS Tiwary' ,'TA Boult' ,'CJ Jordan' ,'IR Jaggi' ,'PP Chawla' ,'AS Rajpoot',
 'SC Ganguly', 'RT Ponting' ,'DJ Hussey' ,'Mohammad Hafeez' ,'R Dravid',
 'W Jaffer', 'JH Kallis','CL White', 'MV Boucher' ,'B Akhil' ,'AA Noffke',
 'SB Joshi', 'ML Hayden' ,'MEK Hussey' ,'JDP Oram', 'S Badrinath' ,'K Goel',
 'JR Hopes' ,'KC Sangakkara' ,'SM Katich' ,'T Kohli', 'M Kaif']


# Replacing Data with numerics and Ranging Data
replace={'Sunrisers Hyderabad':1, 'Royal Challengers Bangalore':2,
       'Mumbai Indians':3, 'Rising Pune Supergiant':4, 'Gujarat Lions':5,
       'Kolkata Knight Riders':6, 'Kings XI Punjab':7, 'Delhi Daredevils':8,
       'Chennai Super Kings':9, 'Rajasthan Royals':10, 'Deccan Chargers':11,
       'Kochi Tuskers Kerala':12, 'Pune Warriors':13, 'Rising Pune Supergiants':14}
replace_venue={'Rajiv Gandhi International Stadium, Uppal':1,
       'Maharashtra Cricket Association Stadium':2,
       'Saurashtra Cricket Association Stadium':3, 'Holkar Cricket Stadium':4,
       'M Chinnaswamy Stadium':5, 'Wankhede Stadium':6, 'Eden Gardens':7,
       'Feroz Shah Kotla':8,
       'Punjab Cricket Association IS Bindra Stadium, Mohali':9,
       'Green Park':10, 'Punjab Cricket Association Stadium, Mohali':11,
       'Sawai Mansingh Stadium':13, 'MA Chidambaram Stadium, Chepauk':12,
       'Dr DY Patil Sports Academy':14, 'Newlands':15, "St George's Park":16,
       'Kingsmead':17, 'SuperSport Park':18, 'Buffalo Park':19,
       'New Wanderers Stadium':20, 'De Beers Diamond Oval':21,
       'OUTsurance Oval':22, 'Brabourne Stadium':23,
       'Sardar Patel Stadium, Motera':24, 'Barabati Stadium':25,
       'Vidarbha Cricket Association Stadium, Jamtha':26,
       'Himachal Pradesh Cricket Association Stadium':27, 'Nehru Stadium':28,
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':29,
       'Subrata Roy Sahara Stadium':30,
       'Shaheed Veer Narayan Singh International Stadium':31,
       'JSCA International Stadium Complex':32, 'Sheikh Zayed Stadium':33,
       'Sharjah Cricket Stadium':34, 'Dubai International Cricket Stadium':35}
bins=[-1,20,40,50,200]
ranges=['0','1','2','3']

                                 # For First Batsman Simmons
                                 
first=bowl[(bowl.batsman==batsmen[0]) | (bowl.non_striker==batsmen[0])].copy()
first.bowling_team.value_counts()
match_id_unique=np.array(first["match_id"].unique())


s=first["batsman_runs"][(first.match_id==match_id_unique[0]) & (first.batsman==batsmen[0])].sum()

simmons=pd.DataFrame({"bowling_team":first["bowling_team"][first.match_id==match_id_unique[0]].unique(),
                 "venue":match["venue"][match.id==match_id_unique[0]],
                 "year":match["season"][match.id==match_id_unique[0]].unique(),
                 "runs":s})

for i in range(len(match_id_unique)-1):
        s=first["batsman_runs"][(first.match_id==match_id_unique[i+1]) & (first.batsman==batsmen[0])].sum()
        df1=pd.DataFrame({"bowling_team":first["bowling_team"][first.match_id==match_id_unique[i]].unique(),
                 "venue":match["venue"][match.id==match_id_unique[i+1]],
                 "year":match["season"][match.id==match_id_unique[i+1]].unique(),
                 "runs":s})
        simmons=pd.concat([simmons,df1])




# Replacing Bowling Team ,Venue with Numerics and runs with Ranges
simmons['bowling_team']=simmons['bowling_team'].replace(replace)
simmons["year"]=simmons["year"]-min(simmons["year"])+1
simmons["venue"]=simmons["venue"].replace(replace_venue)
simmons["runs"]=pd.cut(simmons.runs,bins=bins,labels=ranges)

simmons1=simmons.copy()
# OverSampling
for j in ['1','2'] :
    x=simmons1[simmons1.runs==j]
    if j=='1':
        for i in range(1):
            simmons=pd.concat([simmons,x],axis=0)
    else:
        for i in range(5):
            simmons=pd.concat([simmons,x],axis=0)
        

x=simmons.iloc[:,:-1]
y=simmons.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

algorithm=[] # Name Of the ML Algorithm
accuracy=[] # Cross Validation Score
variance=[] # Variance of Different Score

# Fitting Random Forest Classifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',min_samples_split=9,
                                max_depth=4,min_samples_leaf=5,min_impurity_decrease=0.27,random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

accuracy_score(y_test,y_pred)
accuracy_score(y_train,classifier.predict(x_train))

cross_validation(classifier,x,y,'accuracy',5)

parameters={"min_samples_split":[10,9],'max_depth':[4,5,6],'min_samples_leaf':[5,4,3],
            'min_impurity_decrease':[0.27,0.26,0.29]}
gridsearch(estimator=classifier,parameter=parameters)

Prediction=pd.DataFrame({"True_Value":y,"Predicted_Value_RFC":classifier.predict(x)})

algorithm.append("Random Forest")
accuracy.append(cross_validation(classifier,x,y,'accuracy',5)[0])
variance.append(cross_validation(classifier,x,y,'accuracy',5)[1])

# Fitting Naive Bayes
naive=GaussianNB()
naive.fit(x_train,y_train)

accuracy_score(y_test,naive.predict(x_test))
accuracy_score(y_train,naive.predict(x_train))

cross_validation(naive,x,y,'accuracy',5)

algorithm.append("Naive Bayes")
accuracy.append(cross_validation(naive,x,y,'accuracy',5)[0])
variance.append(cross_validation(naive,x,y,'accuracy',5)[1])

Prediction_naive=pd.DataFrame({"True_Value":y,"Predicted_Value_Naive":naive.predict(x)})
Prediction=pd.concat([Prediction,Prediction_naive],join='inner',axis=1)

# Fitting Xgboost


























