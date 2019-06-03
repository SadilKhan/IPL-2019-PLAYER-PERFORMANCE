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
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

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
player_2019 = pd.read_csv('example.txt',header = None)
player_2019.set_index(0, inplace = True)
match.head()

#Output
""" id  season       city   ...           umpire1        umpire2 umpire3
0   1    2017  Hyderabad   ...       AY Dandekar       NJ Llong     NaN
1   2    2017       Pune   ...    A Nand Kishore         S Ravi     NaN
2   3    2017     Rajkot   ...       Nitin Menon      CK Nandan     NaN
3   4    2017     Indore   ...      AK Chaudhary  C Shamshuddin     NaN
4   5    2017  Bangalore   ...               NaN            NaN     NaN

[5 rows x 18 columns]"""

                              # Batsman

# Name of all the batsman
batsmen=bowl.batsman.unique()

# Some rows have total score more than 7  on a single ball whereas
bowl=bowl.drop(index=[150968,151395,151833,152292,152856,153391,
                      153511,153518,153986,154788,155509,155660,155690,
                      155974,156797,157768,157812,158159,
                      158215,158336,158515,159131,159222,
                      159250,159921,161072,161560,163046,163370,164022])


# Fixing Innings(All the innings will be 1 or 2.Some are 3 and 4.Need to change!)
def fix_inning(id):
    a=bowl["batting_team"][(bowl.match_id==id)&(bowl.inning==1)].unique()
    b=bowl["batting_team"][(bowl.match_id==id)&(bowl.inning==2)].unique()
    index=bowl[bowl.match_id==id].index
    for i in index:
        if bowl["batting_team"][i]==a:
            bowl["inning"][i]=1
        else:
            bowl["inning"][i]=2
            
            
for id in bowl["match_id"][bowl.inning>2].unique():
    fix_inning(id)



# Replacing Data with numerics and Ranging Data
replace={'Sunrisers Hyderabad':1, 'Royal Challengers Bangalore':2,
       'Mumbai Indians':3, 'Rising Pune Supergiant':4, 'Gujarat Lions':5,
       'Kolkata Knight Riders':6, 'Kings XI Punjab':7, 'Delhi Daredevils':8,
       'Chennai Super Kings':9, 'Rajasthan Royals':10, 'Deccan Chargers':11,
       'Kochi Tuskers Kerala':12, 'Pune Warriors':13, 'Rising Pune Supergiants':14}
replace_venue={'Unknown':0,'Rajiv Gandhi International Stadium, Uppal':1,
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


# Creating Necessary Dataset
    # We are creating a dataset taking the batsman name player_name
first=bowl[(bowl.batsman==batsmen[0]) | (bowl.non_striker==batsmen[0])].copy()
    
    # From the previous dataset we are extracting unique match_id
match_id_unique=np.array(first["match_id"].unique())
    
    # On the basis of unique match_id ,we are going to calculate total score of a batsman
s=first["batsman_runs"][(first.match_id==match_id_unique[0]) & (first.batsman==batsmen[0])].sum()
player=pd.DataFrame({"bowling_team":first["bowling_team"][first.match_id==match_id_unique[0]].unique(),
                     "venue":match["venue"][match.id==match_id_unique[0]],
                     "runs":s})
    # We are going to use the folloing method because venues are provided for only 636 matches whereas we have 696 matches
    
for i in range(len(match_id_unique)-1):
    if match_id_unique[i+1]>636:
        s=first["batsman_runs"][(first.match_id==match_id_unique[i+1]) & (first.batsman==batsmen[0])].sum()
        df1=pd.DataFrame({"bowling_team":first["bowling_team"][first.match_id==match_id_unique[i]].unique(),
                          "venue":"Unknown","runs":s})
        player=pd.concat([player,df1])
    else:
        s=first["batsman_runs"][(first.match_id==match_id_unique[i+1]) & (first.batsman==batsmen[0])].sum()
        df1=pd.DataFrame({"bowling_team":first["bowling_team"][first.match_id==match_id_unique[i]].unique(),
                          "venue":match["venue"][match.id==match_id_unique[i+1]],
                          "runs":s})
        player=pd.concat([player,df1])

        
# We will use Classification Method instead of regression since the dataset is small 
# and classification will work better in this case
bins=[-1,player.runs.mean()*2,200]
ranges=['0','1']          
        
# Replacing Bowling Team ,Venue with Numerics and runs with Ranges
player['bowling_team']=player['bowling_team'].replace(replace)
player["venue"]=player["venue"].replace(replace_venue)
player["runs"]=pd.cut(player.runs,bins=bins,labels=ranges)

# Separating Dependent and Independent Variables      
y=player.pop("runs")
x=player.values

# Spliting Dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=0)



# XGBoost Model
boost=xgb.XGBClassifier(max_depth=1,n_estimators=10,learning_rate=0.09,gamma=0.07)
boost.fit(x_train,y_train)

accuracy_score(y_train,boost.predict(x_train))
    # Output: 0.8725490196078431
accuracy_score(y_test,boost.predict(x_test))
  # Output: 0.8333333333333334


    
    ## Cross Validation
cross_validation(boost,x,y,'accuracy',5)
       #Output: (0.8683794466403162, 5.624209095595867e-06)

# Overview: Overall our model is good for this batsman. accuracy is high and variance is low. 
# So ,we choose this model to predict all the other batsman
    
    
   # The Hyperparameters are already tuned for best performances. For tuning we have used GridSearch

                     # GridSearch
    
parameters=[{"gamma":[0.07,0.09,0.05,0.08,0.06],
                      "learning_rate":[0.09,0.08,0.1],"max_depth":[1,2,3,4]}]
gridsearch(estimator=boost,parameter=parameters)
 # Output : (0.868421052631579, {'gamma': 0.07, 'learning_rate': 0.09, 'max_depth': 1})


                                 # Bowler
       
        ## Get SL Malinga's all Ball by Ball data

delivery_data=bowl.copy()
match_data=match.copy()

player_team="Mumbai Indians"
name='SL Malinga'

#extracting bowler_data
bowler_data=delivery_data[delivery_data.bowler==name]

#creating a dataframe containing names of all team except player's own team
teams=pd.DataFrame(index=match_data.team1.unique())
teams=teams.drop('Mumbai Indians')
teams
bowler_data
bowler_data.player_dismissed = bowler_data.player_dismissed.notnull().astype('int')
bowler_data.player_dismissed.sum()


data=match_data[(match_data.team1==player_team)|(match_data.team2==player_team)]
data.head()                         

# Output
"""id  season       city   ...                  umpire1       umpire2 umpire3
1    2    2017       Pune   ...           A Nand Kishore        S Ravi     NaN
6    7    2017     Mumbai   ...              Nitin Menon     CK Nandan     NaN
9   10    2017     Mumbai   ...              Nitin Menon     CK Nandan     NaN
11  12    2017  Bangalore   ...    KN Ananthapadmanabhan  AK Chaudhary     NaN
15  16    2017     Mumbai   ...           A Nand Kishore        S Ravi     NaN"""                    

#bowling first
data=match_data[(match_data.team1==player_team)]
bowling_first=list()
for team in teams.index:
    for venue in data.venue.unique():
        matches=data[(data.venue==venue)&(data.team2==team)].id
        
        wk=0
        for match in matches:
                wk=0
#                print(bowler_data[(bowler_data.match_id==match)&(bowler_data.dismissal_kind!='run out')])
#                if(bowler_data[bowler_data.match_id==match]):
#                print(match)
                t=bowler_data[(bowler_data.match_id==match)&(bowler_data.dismissal_kind!='run out')].player_dismissed.sum()
                wk=wk+t
                bowling_first=bowling_first+[[team,venue,wk]]
         
#bowling second        
data=match_data[match_data.team2==player_team]
bowling_second=list()
for team in teams.index:
    for venue in data.venue.unique():
        matches=data[(data.venue==venue)&(data.team1==team)].id
        wk=0
        
        for match in matches:
            wk=0
#            print(match)
            t=bowler_data[(bowler_data.match_id==match)&(bowler_data.dismissal_kind!='run out')].player_dismissed.sum()
            wk=wk+t
           
            bowling_second=bowling_second+[[team,venue,wk]]
                
        
bowling_first=bowling_first+bowling_second

df=pd.DataFrame(data=bowling_first,columns=['team','venue','wicket'])    
#df=df[df.wicket!=0]
df.wicket.sum()

df.head()

# Output
"""                  team  ...   wicket
0  Sunrisers Hyderabad  ...        4
1  Sunrisers Hyderabad  ...        1
2  Sunrisers Hyderabad  ...        0
3  Sunrisers Hyderabad  ...        0
4        Gujarat Lions  ...        0"""


#df['wicket']
bin = [-1,1,3,10]
category = pd.cut(df.wicket,bin)
category = category.to_frame()
category.columns = ['range']

df_new = pd.concat([df,category],axis = 1)
dffin = df_new['range'].apply(str)

range1 = {'(-1, 1]':0,'(1, 3]':1,'(3, 10]':2}
dffin = [range1[item] for item in dffin] 

dffin = pd.DataFrame(dffin,columns=['range1'])
df_final = pd.concat([df_new,dffin],axis = 1)
df_final = df_final.drop(columns=['wicket','range'],axis = 1)

df_final.head()

# Output

"""                  team  ...   range1
0  Sunrisers Hyderabad  ...        2
1  Sunrisers Hyderabad  ...        0
2  Sunrisers Hyderabad  ...        0
3  Sunrisers Hyderabad  ...        0
4        Gujarat Lions  ...        0

[5 rows x 3 columns]"""

df_final["team"]=df_final["team"].replace(replace)
df_final["venue"]=df_final["venue"].replace(replace_venue)

processed_df = df_final.copy()


x = processed_df.drop(['range1'], axis=1).values
y = processed_df['range1'].values

x

# Output
"""array([[ 1,  6],
       [ 1,  1],
       [ 1,  1],
       [ 1,  1],
       [ 5,  6],
       [ 5, 10],
       [ 4,  2],
       [ 4,  1],
       [ 2,  6],
       [ 2,  6],
       [ 2,  6],
       [ 2,  6],
       [ 2, 14],
       [ 2, 16],
       [ 2, 20],
       [ 2, 23],
       [ 2,  5],
       [ 2,  5],
       [ 2, 35],
       [ 6,  6],
       [ 6,  6],
       [ 6,  7],
       [ 6,  7],
       [ 6,  7],
       [ 6,  7],
       [ 6, 16],
       [ 6, 19],
       [ 6, 25],
       [ 8,  6],
       [ 8,  6],
       [ 8,  6],
       [ 8,  6],
       [ 8,  6],
       [ 8,  8],
       [ 8,  8],
       [ 8,  8],
       [ 8,  8],
       [ 8, 14],
       [ 8, 19],
       [ 8, 18],
       [ 8, 23],
       [ 8, 34],
       [ 8, 29],
       [ 7,  6],
       [ 7,  6],
       [ 7,  6],
       [ 7, 11],
       [ 7, 11],
       [ 7,  9],
       [ 7, 29],
       [ 9,  6],
       [ 9,  6],
       [ 9,  6],
       [ 9,  6],
       [ 9,  6],
       [ 9,  7],
       [ 9,  7],
       [ 9, 15],
       [ 9, 16],
       [ 9, 23],
       [ 9, 12],
       [ 9, 35],
       [10,  6],
       [10,  6],
       [10,  6],
       [10,  6],
       [10, 13],
       [10, 13],
       [10, 13],
       [10, 23],
       [10, 24],
       [10, 24],
       [11,  1],
       [11,  1],
       [11, 14],
       [11, 14],
       [11, 23],
       [12,  6],
       [13,  6],
       [13, 14],
       [13, 30],
       [14,  6],
       [ 1,  6],
       [ 1,  6],
       [ 1, 29],
       [ 1, 35],
       [ 1,  1],
       [ 1,  1],
       [ 5,  6],
       [ 5,  3],
       [ 4,  6],
       [ 4,  6],
       [ 2,  6],
       [ 2,  6],
       [ 2,  6],
       [ 2,  5],
       [ 2,  5],
       [ 2,  5],
       [ 2,  5],
       [ 2,  5],
       [ 2,  5],
       [ 2, 12],
       [ 6,  6],
       [ 6,  6],
       [ 6,  6],
       [ 6,  6],
       [ 6,  6],
       [ 6,  5],
       [ 6,  7],
       [ 6,  7],
       [ 6,  7],
       [ 6,  7],
       [ 6, 23],
       [ 6, 33],
       [ 8,  6],
       [ 8,  8],
       [ 8,  8],
       [ 8,  8],
       [ 8,  8],
       [ 7,  6],
       [ 7,  6],
       [ 7,  6],
       [ 7,  6],
       [ 7,  4],
       [ 7, 11],
       [ 7, 11],
       [ 7, 11],
       [ 7, 11],
       [ 7, 17],
       [ 7, 18],
       [ 7, 23],
       [ 7, 27],
       [ 9,  6],
       [ 9,  6],
       [ 9,  5],
       [ 9, 12],
       [ 9, 12],
       [ 9, 12],
       [ 9, 12],
       [ 9, 14],
       [ 9, 23],
       [ 9,  8],
       [10,  6],
       [10,  7],
       [10, 14],
       [10, 17],
       [10, 13],
       [10, 13],
       [11,  6],
       [11,  6],
       [11, 17],
       [11, 18],
       [11, 29],
       [13,  6],
       [13,  6],
       [13, 30],
       [14,  2]], dtype=int64)"""

# Spliting the Dataset for Validation
    
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# Applying XGBClassifier

# XGBoost Model
boost=xgb.XGBClassifier(max_depth=1,n_estimators=10,learning_rate=0.08,gamma=0.07)
boost.fit(x_train,y_train)

accuracy_score(y_train,boost.predict(x_train))
    # Output: 0.712
accuracy_score(y_test,boost.predict(x_test))
  # Output: 0.71875
cross_validation(boost,x,y,'accuracy',5)
# Output: (0.7006854838709679, 0.00015107310093652462))


# Gridsearching
parameters=[{"gamma":[0.07,0.09,0.05,0.08,0.06],
                      "learning_rate":[0.09,0.08,0.1],"max_depth":[1,2,3,4]}]
gridsearch(estimator=boost,parameter=parameters)

# Output :(0.7006369426751592, {'gamma': 0.07, 'learning_rate': 0.08, 'max_depth': 1})



