# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/2/19'

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import plot_importance
from matplotlib import pyplot

dataset = loadtxt("src_data.csv", delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]
# print(X)
# print(Y)

seed = 7
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier(learning_rate=0.02, max_depth=5,
                      n_estimators=1000, min_child_weight=1,
                      colsample_bytree=0.8,
                      gamma=0.01,
                      subsample=0.8,
                      nthread=4,
                      objective="binary:logistic",
                      scale_pos_weight=1,
                      seed=27)
eval_set = [(x_test, y_test)]
model.fit(x_train, y_train, early_stopping_rounds=50, eval_metric="logloss", eval_set=eval_set, verbose=True)
y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %2f%% " % (accuracy * 100.0))

# model.get_booster().save_model('xgb.model')

plot_importance(model)
pyplot.show()