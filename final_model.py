__author__ = 'jalFaizy'

# import modules
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV as cccv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

start_time = time.time()

print 'Making data ready'
probs = pd.read_csv(os.path.join(os.path.abspath('.') + '/train/' + 'problems.csv'))
users = pd.read_csv(os.path.join(os.path.abspath('.') + '/train/' + 'users.csv'))
subs = pd.read_csv(os.path.join(os.path.abspath('.') + '/train/' + 'submissions.csv'))

probs_subs = pd.merge(probs, subs, on='problem_id')
train_all = pd.merge(users, probs_subs, on='user_id')

skillset = train_all.skills.str.get_dummies(sep='|')
total_skills_ = skillset.sum(axis = 1)
total_skills = pd.DataFrame({'total_skills':total_skills_})

train_all = pd.concat([train_all, skillset[['C', 'C++', 'Java', 'Python']], total_skills], axis = 1)

column_dropper = ['result', 'language_used', 'execution_time', 'tag3', 'tag4', 'tag5']
train_all.drop(column_dropper, axis = 1, inplace = True)

probs_test = pd.read_csv(os.path.join(os.path.abspath('.') + '/test/' + 'problems.csv'))
users_test = pd.read_csv(os.path.join(os.path.abspath('.') + '/test/' + 'users.csv'))
tests_test = pd.read_csv(os.path.join(os.path.abspath('.') + '/test/' + 'test.csv'))

probs_subs_test = pd.merge(probs_test, tests_test, on='problem_id')
test_all = pd.merge(users_test, probs_subs_test, on='user_id')

skillset = test_all.skills.str.get_dummies(sep='|')
total_skills_ = skillset.sum(axis = 1)
total_skills = pd.DataFrame({'total_skills':total_skills_})

test_all = pd.concat([test_all, skillset[['C', 'C++', 'Java', 'Python']], total_skills], axis = 1)

test_all.drop(['tag3', 'tag4', 'tag5'], axis = 1, inplace = True)

categorical_columns = ['user_type', 'level', 'tag1', 'tag2', 'skills']
for var in categorical_columns:
    lb = LabelEncoder()
    full_var_data = pd.concat((train_all[var],test_all[var]),axis=0).astype('str')
    lb.fit(full_var_data )
    train_all[var] = lb.transform(train_all[var].astype('str'))
    test_all[var] = lb.transform(test_all[var].astype('str'))
    
train_x = train_all.ix[:, train_all.columns != 'solved_status'].values
train_y = train_all['solved_status'].values

test_x = test_all.ix[:, test_all.columns != 'Id'].values

X = train_x.astype('float32')
X = MinMaxScaler().fit_transform(X).astype('float32')

X_test = test_x.astype('float32')
X_test = MinMaxScaler().fit_transform(X_test).astype('float32')

y = []
for i in train_y:
    if i == 'SO':
        y.append(1)
    else:
        y.append(0)
        
y = np.array(y)

clf1 = RandomForestClassifier(n_estimators = 50, n_jobs = 3)
cccv1 = cccv(clf1, method = 'isotonic', cv = 2)

print 'Training classifiers'
cccv1.fit(X, y)
pred = cccv1.predict(X_test)

pd.DataFrame({'Id': test_all['Id'].values, 'solved_status' : pred }).to_csv('./submissions/sub_finalist.csv', index = False)

print 'Time Required: ', time.time() - start_time
