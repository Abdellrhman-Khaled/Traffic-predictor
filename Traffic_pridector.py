# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import (datasets, preprocessing)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# %matplotlib inline

traffic_data = pd.read_csv("/content/Traffic dataset.csv")

"""# New section"""

traffic_data.head()

traffic_data.max()

traffic_data.sort_values(by='Total', ascending=False)

traffic_data.mean()

traffic_data.duplicated().sum()==0

traffic_data.isnull().sum()==0

sns.distplot(traffic_data['Total'],kde=False)

sns.distplot(traffic_data['Total'])

sns.barplot(x='Day of the week',y='Total',data=traffic_data)

sns.barplot(x='Date',y='Total',data=traffic_data)

sns.barplot(x='Date',y='Total',data=traffic_data)

traffic_data.head()

traffic_data.describe()

traffic_data['Traffic Situation']=traffic_data['Traffic Situation'].replace("low",0)
traffic_data['Traffic Situation']=traffic_data['Traffic Situation'].replace("normal",1)
traffic_data['Traffic Situation']=traffic_data['Traffic Situation'].replace("high",2)
traffic_data['Traffic Situation']=traffic_data['Traffic Situation'].replace("heavy",3)

traffic_data.head()

sns.barplot(x='Time',y='Traffic Situation',data=traffic_data)

sns.barplot(x='CarCount',y='Traffic Situation',data=traffic_data)

sns.barplot(x='BikeCount',y='Traffic Situation',data=traffic_data)

sns.barplot(x='BusCount',y='Traffic Situation',data=traffic_data)

sns.barplot(x='TruckCount',y='Traffic Situation',data=traffic_data)

sns.barplot(x='Date',y='Traffic Situation',data=traffic_data)

sns.barplot(x='Day of the week',y='Traffic Situation',data=traffic_data)

sns.barplot(x='Total',y='Traffic Situation',data=traffic_data)

traffic_data.drop(['Total'], axis=1, inplace=True)
# ,'Total'  'Date'

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
traffic_data['Time'] = le.fit_transform(traffic_data['Time'])
traffic_data.head()

# traffic_data['Time'] = pd.to_numeric(traffic_data['Time'].str.replace(':', '').str.replace(' AM', '').str.replace(' PM', ''))
# traffic_data['Time']=traffic_data['Time'].astype(float)
# traffic_data.head()

sns.barplot(x='Time',y='Traffic Situation',data=traffic_data)

columns=traffic_data.columns

for col in columns:
    if traffic_data[col].dtype!='object':
        sns.boxplot(x=traffic_data[col])
        plt.show()

for col in columns:
    if traffic_data[col].dtype != 'object':
        # write your code here
        sns.displot(traffic_data,x=col,kde=True)

        plt.show()

"""Most of the columns in the dataset are approximately normally distributed, except for BikeCount, which is right-skewed."""

def calculate_outliers(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

lower_limit, upper_limit = calculate_outliers((traffic_data['BikeCount']))
lower_outliers=len(traffic_data[(traffic_data['BikeCount'])<lower_limit])
upper_outliers=len(traffic_data[(traffic_data['BikeCount'])>upper_limit])
print(f"total outliers in col BikeCount  = ",upper_outliers+lower_outliers," with precentage of ",(upper_outliers+lower_outliers)/traffic_data.shape[0]*100)

lower_limit, upper_limit = calculate_outliers(np.log1p(traffic_data['BikeCount']))
lower_outliers=len(traffic_data[np.log1p(traffic_data['BikeCount'])<lower_limit])
upper_outliers=len(traffic_data[np.log1p(traffic_data['BikeCount'])>upper_limit])
print(f"total outliers in col BikeCount with log = ",upper_outliers+lower_outliers," with precentage of ",(upper_outliers+lower_outliers)/traffic_data.shape[0]*100)
lower_limit, upper_limit = calculate_outliers(np.cbrt(traffic_data['BikeCount']))
lower_outliers=len(traffic_data[np.cbrt(traffic_data['BikeCount'])<lower_limit])
upper_outliers=len(traffic_data[np.cbrt(traffic_data['BikeCount'])>upper_limit])
print(f"total outliers in col BikeCount with cubic = ",upper_outliers+lower_outliers," with precentage of ",(upper_outliers+lower_outliers)/traffic_data.shape[0]*100)

traffic_data['BikeCount']=np.log1p(traffic_data['BikeCount'])

lower_limit, upper_limit = calculate_outliers(traffic_data['BikeCount']) #lower and upper range
traffic_data['BikeCount'] = np.where(traffic_data['BikeCount'] < lower_limit, lower_limit, traffic_data['BikeCount'])
traffic_data['BikeCount'] = np.where(traffic_data['BikeCount'] > upper_limit, upper_limit, traffic_data['BikeCount'])

sns.boxplot(x=traffic_data['BikeCount'])

for z in traffic_data.index:
  if traffic_data.loc[z,"Day of the week"]=="Saturday":
    traffic_data.loc[z,"Day of the week"]=0
  elif traffic_data.loc[z,"Day of the week"]=="Sunday":
    traffic_data.loc[z,"Day of the week"]=1
  elif traffic_data.loc[z,"Day of the week"]=="Monday":
    traffic_data.loc[z,"Day of the week"]=2
  elif traffic_data.loc[z,"Day of the week"]=="Tuesday":
    traffic_data.loc[z,"Day of the week"]=3
  elif traffic_data.loc[z,"Day of the week"]=="Wednesday":
    traffic_data.loc[z,"Day of the week"]=4
  elif traffic_data.loc[z,"Day of the week"]=="Thursday":
    traffic_data.loc[z,"Day of the week"]=5
  else:
    traffic_data.loc[z,"Day of the week"]=6

features = traffic_data.drop(['Traffic Situation'], axis=1)
target = traffic_data['Traffic Situation']
numeric_columns = ['Time', 'Date', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount'] #,'Total'
scaler = MinMaxScaler()
features[numeric_columns] = scaler.fit_transform(features[numeric_columns])
le = LabelEncoder()
features['Day of the week'] = le.fit_transform(features['Day of the week'])
le_target = LabelEncoder()
target = le_target.fit_transform(target)
normalized_encoded_data = pd.concat([features, pd.Series(target, name='Traffic Situation')], axis=1)
traffic_data = normalized_encoded_data
traffic_data.head()

X = traffic_data.drop('Traffic Situation', axis=1)
y = traffic_data['Traffic Situation']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn import svm
clf = svm.SVC(kernel= 'linear')
clf.fit(x_train, y_train)
pred1 = clf.predict(x_test)

from sklearn import metrics

accuracy1 = metrics.accuracy_score(y_test, pred1)
print("Accuracy: ", accuracy1 * 100, "%")

print("Precision: ", metrics.precision_score(y_test, pred1, average=None) * 100, "%")

print("F1 Score: ", metrics.f1_score(y_test, pred1, average=None) * 100, "%")

from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test,pred1, pos_label=1)
auc1 = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-', color="orange",label='SVM (auc = %0.3f)' % auc1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
pred2 = classifier.predict(x_test)

accuracy2 = metrics.accuracy_score(y_test, pred2)
print("Accuracy: ", accuracy2 * 100, "%")

print("Precision: ", metrics.precision_score(y_test, pred2, average=None) * 100, "%")

print("F1 Score: ", metrics.f1_score(y_test, pred2, average=None) * 100, "%")

fpr, tpr, threshold = roc_curve(y_test,pred2, pos_label=1)
auc2 = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-', label='Logistic (auc = %0.3f)' % auc2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
pred3 = tree.predict(x_test)

accuracy3 = metrics.accuracy_score(y_test, pred3)
print("Accuracy: ", accuracy3 * 100, "%")

print("Precision: ", metrics.precision_score(y_test, pred3, average=None) * 100, "%")

print("F1 Score: ", metrics.f1_score(y_test, pred3, average=None) * 100, "%")

fpr, tpr, threshold = roc_curve(y_test, pred3, pos_label=1)
auc3 = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-', color="green",label='Decision Tree (auc = %0.3f)' % auc3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

from sklearn.neighbors import KNeighborsClassifier
kneighbor = KNeighborsClassifier()
kneighbor.fit(x_train, y_train)
pred4 = kneighbor.predict(x_test)

accuracy4 = metrics.accuracy_score(y_test, pred4)
print("Accuracy: ", accuracy4 * 100, "%")

print("Precision: ", metrics.precision_score(y_test, pred4, average=None) * 100, "%")

print("F1 Score: ", metrics.f1_score(y_test, pred4, average=None) * 100, "%")

fpr, tpr, threshold = roc_curve(y_test, pred4, pos_label=1)
auc4 = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-', color="red" , label='KNN (auc = %0.3f)' % auc4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred5 = rf.predict(x_test)

accuracy5 = metrics.accuracy_score(y_test, pred5)
print("Accuracy: ", accuracy5 * 100, "%")

print("Precision: ", metrics.precision_score(y_test, pred5, average=None) * 100, "%")

print("F1 Score: ", metrics.f1_score(y_test, pred5, average=None) * 100, "%")

fpr, tpr, threshold = roc_curve(y_test, pred5, pos_label=1)
auc4 = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-', color="purple" , label='Random Forest (auc = %0.3f)' % auc4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

dictionary = {'Accuracy' : [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5], 'Algorithms' : ['LogisticRegression', 'SVM', 'DecisionTree', 'KNN', 'Random Forest']}
data = pd.DataFrame(dictionary)
sns.barplot(x='Algorithms', y='Accuracy' , data=data)