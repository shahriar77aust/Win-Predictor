"""
Created on February 02 00:15:18 2018

@author: Ahmed Shahriar
"""

#imporing packages
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import linear_model



#reading data
#data=pd.read_csv('data.csv',encoding='latin-1')
data = pd.read_csv('data.csv')
#Target in numpy array
y = data['Result'].values
#removing 'Result' attribute or target attribute
data.drop('Result',axis=1, inplace=True)
#features in numpy array
X = data.values


#creating objects of the classifier
sgd = linear_model.SGDClassifier()
svc = svm.SVC(kernel='linear',C=0.4)
knn = KNeighborsClassifier(n_neighbors=5)
abc = AdaBoostClassifier()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier(random_state=0)


#create lists
#For SGD
accuracy_sgd = []
precision_sgd = []
recall_sgd = []
F1_sgd = []
#For random forest
accuracy_rfc = []
precision_rfc = []
recall_rfc = []
F1_rfc = []
##for Decision Tree
accuracy_dtc = []
precision_dtc = []
recall_dtc = []
F1_dtc = []
#for support vector
accuracy_svc = []
precision_svc = []
recall_svc = []
F1_svc = []
#for k neighbour
accuracy_knn = []
precision_knn = []
recall_knn = []
F1_knn = []
#for AdaBoost
accuracy_abc = []
precision_abc = []
recall_abc = []
F1_abc = []


#K Fold Cross validation
kf = KFold(n_splits=5,random_state = 0,shuffle = True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Scaling features
    scaler = preprocessing.StandardScaler().fit(X_train.astype(float))
       
    X_train = scaler.transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))
    
    
    #Model Traning
    sgd.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    abc.fit(X_train, y_train)
    
    #predection
    y_pred_sgd = sgd.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_abc = abc.predict(X_test)
    ###################################################################################
    accuracy_sgd.append( metrics.accuracy_score(y_test, y_pred_sgd))
    accuracy_rfc.append( metrics.accuracy_score(y_test, y_pred_rfc))
    accuracy_dtc.append( metrics.accuracy_score(y_test, y_pred_dtc))
    accuracy_svc.append( metrics.accuracy_score(y_test, y_pred_svc))
    accuracy_knn.append( metrics.accuracy_score(y_test, y_pred_knn))
    accuracy_abc.append( metrics.accuracy_score(y_test, y_pred_abc))
    #########################################################################################
    precision_sgd.append(metrics.precision_score(y_test, y_pred_sgd,average='macro'))
    precision_rfc.append(metrics.precision_score(y_test, y_pred_rfc,average='macro'))
    precision_dtc.append(metrics.precision_score(y_test, y_pred_dtc,average='macro'))
    precision_svc.append(metrics.precision_score(y_test, y_pred_svc,average='macro'))
    precision_knn.append(metrics.precision_score(y_test, y_pred_knn,average='macro'))
    precision_abc.append(metrics.precision_score(y_test, y_pred_abc,average='macro'))
    #########################################################################################
    recall_sgd.append(metrics.recall_score(y_test, y_pred_sgd,average='macro'))
    recall_rfc.append(metrics.recall_score(y_test, y_pred_rfc,average='macro'))
    recall_dtc.append(metrics.recall_score(y_test, y_pred_dtc,average='macro'))
    recall_svc.append(metrics.recall_score(y_test, y_pred_svc,average='macro'))
    recall_knn.append(metrics.recall_score(y_test, y_pred_knn,average='macro'))
    recall_abc.append(metrics.recall_score(y_test, y_pred_abc,average='macro'))
    ############################################################################################# 
    F1_sgd.append(metrics.f1_score(y_test, y_pred_sgd,average='macro'))
    F1_rfc.append(metrics.f1_score(y_test, y_pred_rfc,average='macro'))
    F1_dtc.append(metrics.f1_score(y_test, y_pred_dtc,average='macro'))
    F1_svc.append(metrics.f1_score(y_test, y_pred_svc,average='macro'))
    F1_knn.append(metrics.f1_score(y_test, y_pred_knn,average='macro'))
    F1_abc.append(metrics.f1_score(y_test, y_pred_abc,average='macro'))
    ############################################################################################



#Performance metric
print ("For SGD:")
print ("Average Accuracy : ",np.mean(accuracy_sgd))
print ("Average Precision : ",np.mean(precision_sgd))
print ("Average Recall : ",np.mean(recall_sgd))
print ("Average F1 : ",np.mean(F1_sgd))
print ("\n")    
    
print ("For Random Forest Classifier:")
print ("Average Accuracy : ",np.mean(accuracy_rfc))
print ("Average Precision : ",np.mean(precision_rfc))
print ("Average Recall : ",np.mean(recall_rfc))
print ("Average F1 : ",np.mean(F1_rfc))
print ("\n")

print ("For Decision Tree Classifier:")
print ("Average Accuracy : ",np.mean(accuracy_dtc))
print ("Average Precision : ",np.mean(precision_dtc))
print ("Average Recall : ",np.mean(recall_dtc))
print ("Average F1 : ",np.mean(F1_dtc))
print ("\n")

print ("For Support Vector Classification:")
print ("Average Accuracy : ",np.mean(accuracy_svc))
print ("Average Precision : ",np.mean(precision_svc))
print ("Average Recall : ",np.mean(recall_svc))
print ("Average F1 : ",np.mean(F1_svc))
print ("\n")

print ("For KNeighbors Classifier:")
print ("Average Accuracy : ",np.mean(accuracy_knn))
print ("Average Precision : ",np.mean(precision_knn))
print ("Average Recall : ",np.mean(recall_knn))
print ("Average F1 : ",np.mean(F1_knn))
print ("\n")

print ("FOR AdaBoost Classifier:")
print ("Average Accuracy : ",np.mean(accuracy_abc))
print ("Average Precision : ",np.mean(precision_abc))
print ("AverageRecall : ",np.mean(recall_abc))
print ("Average F1 : ",np.mean(F1_abc))
print ("\n")