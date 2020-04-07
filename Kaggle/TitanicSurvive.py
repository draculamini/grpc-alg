# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/7'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_path = "data/titanic/train.csv"
test_path = "data/titanic/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

drop_features = ["PassengerId", "Survived"]
train_drop = train.drop(drop_features, axis=1)
print(train_drop.head())

print(train_drop.dtypes.sort_values())

print(train_drop.select_dtypes(include='int64').head())
print(train_drop.select_dtypes(include='float64').head())
print(train_drop.select_dtypes(include='object').head())

print(train.isnull().sum()[lambda x: x>0])
print(test.isnull().sum()[lambda x: x>0])

print(train.info())
print(train.describe())

titanic=pd.concat([train, test], sort=False)
len_train=train.shape[0]
print(titanic.info())

titanic['Title'] = titanic.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
print(titanic.head())

print(titanic.Title.value_counts())

List=titanic.Title.value_counts().index[4:].tolist()
print(List)
mapping={}

for s in List:
    mapping[s] = 'Rare'
titanic['Title']=titanic['Title'].map(lambda x: mapping[x] if x in mapping else x)

print(titanic.Title.value_counts())

grouped=titanic.groupby(['Title'])
median=grouped.Age.median()
print("== median ==")
print(median)

def newage(cols):
    age = cols[0]
    title = cols[1]

    if pd.isnull(age):
        return median[title]
    return age

titanic.Age = titanic[['Age', 'Title']].apply(newage, axis=1)

print(titanic.info())
# titanic['has_Cabin'].loc[titanic.Cabin.isnull()]=1
# titanic['has_Cabin'].loc[titanic.Cabin.isnull()]=0

print("titanic.Cabin.show()\n", titanic.Cabin.head(20))
titanic.Cabin = titanic.Cabin.fillna('U')
titanic.Fare = titanic.Fare.fillna(0.0)
print(titanic[:10])

most_embarked = titanic.Embarked.value_counts().index[0]
titanic.Embarked=titanic.Embarked.fillna(most_embarked)

print(titanic.info())

titanic['Cabin'] = titanic.Cabin.apply(lambda cabin:cabin[0])
print(titanic.Cabin.value_counts().head(20))
titanic['Cabin'].loc[titanic.Cabin=='T']='G'
print(titanic.Cabin.value_counts())
# pd.crosstab(titanic.Cabin[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()

# pd.crosstab(titanic.Parch[:len_train],train.Survived).plot.bar(stacked=True)
# pd.crosstab(titanic.SibSp[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()

titanic['FamilySize'] = titanic.Parch + titanic.SibSp + 1

# pd.crosstab(titanic.FamilySize[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()

titanic=titanic.drop(['SibSp','Parch'],axis=1)

# pd.crosstab(titanic.Sex[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()

# pd.crosstab(titanic.Pclass[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()

# pd.crosstab(titanic.Pclass[:len_train], train.Survived).plot.bar(stacked=True)
# plt.show()

# pd.crosstab(titanic.Title[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()

# pd.crosstab([titanic.Title[:len_train],titanic.Sex[:len_train]],train.Survived).plot.bar(stacked=True)
# plt.show()

# pd.crosstab(pd.cut(titanic.Age,8)[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()
titanic.Age=pd.cut(titanic.Age,8,labels=False)
# pd.crosstab(titanic.Embarked[:len_train],train.Survived).plot.bar(stacked=True)
# plt.show()

titanic=titanic.drop('Name',axis=1)
titanic=titanic.drop('Ticket',axis=1)
# pd.crosstab(pd.qcut(titanic.Fare,4)[:len_train],train.Survived).plot.bar(stacked=True)

print("== titanic.Fare ==\n", titanic.Fare.head())

titanic.Fare=pd.cut(titanic.Fare,4,labels=False)
print("titanic.Fare.value_counts() \n ",titanic.Fare.value_counts())


# print("np.isnan(titanic).any()\n ", np.isnan(titanic.).any())


print("== titanic.Fare ==\n", titanic.Fare.head())

titanic.Sex=titanic.Sex.map({'male':1,'female':0})
titanic.Cabin=titanic.Cabin.map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'U':7})
titanic.Embarked=titanic.Embarked.map({'C':0,'Q':1,'S':2})
titanic.Title=titanic.Title.map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4})

train = titanic[:len_train]
test = titanic[len_train:]

X_train=train.loc[:, 'Pclass':]
y_train=train['Survived']
X_test=test.loc[:, 'Pclass':]


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
tree_clf = DecisionTreeClassifier()
model = tree_clf.fit(X_train, y_train)

print(log_reg.score(X_train,y_train))
print(svm_clf.score(X_train,y_train))
print(tree_clf.score(X_train,y_train))


# print('包含空值的DF为：\n',X_test[np.isnan(X_test['Fare'])])

print("-- X_train.info() --")
print(X_train.info())

print("X_train.Fare.value_counts() \n ", X_train.Fare.value_counts())

print("----")


pred = model.predict(X_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred})
output.to_csv('data/titanic//submission.csv', index=False)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

RF=RandomForestClassifier(random_state=1)
PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]
GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)
scores_rf=cross_val_score(GSRF,X_train,y_train,scoring='accuracy',cv=5)


model=GSRF.fit(X_train, y_train)

# print(GSRF.score(X_train,y_train))

# pred=model.predict(X_test)
# output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred})
# output.to_csv('data/titanic//submission.csv', index=False)



