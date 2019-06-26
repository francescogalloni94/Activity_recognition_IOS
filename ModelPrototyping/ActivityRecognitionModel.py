import pandas as pd
import numpy as np
from SensorFeaturesExtractor import SensorFeaturesExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

pwd_x_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/Inertial Signals'
pwd_y_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/y_train.txt'
pwd_x_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/Inertial Signals'
pwd_y_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/y_test.txt'
pwd_labels = '../ActivityRecognitionDataset/UCI HAR Dataset/activity_labels.txt'

print("Reading data...")
body_acc_x_train = pd.read_csv(pwd_x_train+'/body_acc_x_train.txt',delim_whitespace=True,header=None).values
body_acc_y_train = pd.read_csv(pwd_x_train+'/body_acc_y_train.txt',delim_whitespace=True,header=None).values
body_acc_z_train = pd.read_csv(pwd_x_train+'/body_acc_z_train.txt',delim_whitespace=True,header=None).values

body_gyro_x_train = pd.read_csv(pwd_x_train+'/body_gyro_x_train.txt',delim_whitespace=True,header=None).values
body_gyro_y_train = pd.read_csv(pwd_x_train+'/body_gyro_y_train.txt',delim_whitespace=True,header=None).values
body_gyro_z_train = pd.read_csv(pwd_x_train+'/body_gyro_z_train.txt',delim_whitespace=True,header=None).values

total_acc_x_train = pd.read_csv(pwd_x_train+'/total_acc_x_train.txt',delim_whitespace=True,header=None).values
total_acc_y_train = pd.read_csv(pwd_x_train+'/total_acc_y_train.txt',delim_whitespace=True,header=None).values
total_acc_z_train = pd.read_csv(pwd_x_train+'/total_acc_z_train.txt',delim_whitespace=True,header=None).values

y_train = pd.read_csv(pwd_y_train,delim_whitespace=True,header=None).values
y_train = np.ravel(y_train)

body_acc_x_test = pd.read_csv(pwd_x_test+'/body_acc_x_test.txt',delim_whitespace=True,header=None).values
body_acc_y_test = pd.read_csv(pwd_x_test+'/body_acc_y_test.txt',delim_whitespace=True,header=None).values
body_acc_z_test = pd.read_csv(pwd_x_test+'/body_acc_z_test.txt',delim_whitespace=True,header=None).values

body_gyro_x_test = pd.read_csv(pwd_x_test+'/body_gyro_x_test.txt',delim_whitespace=True,header=None).values
body_gyro_y_test = pd.read_csv(pwd_x_test+'/body_gyro_y_test.txt',delim_whitespace=True,header=None).values
body_gyro_z_test = pd.read_csv(pwd_x_test+'/body_gyro_z_test.txt',delim_whitespace=True,header=None).values

total_acc_x_test = pd.read_csv(pwd_x_test+'/total_acc_x_test.txt',delim_whitespace=True,header=None).values
total_acc_y_test = pd.read_csv(pwd_x_test+'/total_acc_y_test.txt',delim_whitespace=True,header=None).values
total_acc_z_test = pd.read_csv(pwd_x_test+'/total_acc_z_test.txt',delim_whitespace=True,header=None).values

y_test = pd.read_csv(pwd_y_test,delim_whitespace=True,header=None).values
y_test = np.ravel(y_test)

labels_mapping = pd.read_csv(pwd_labels,delim_whitespace=True,header=None).values

print("Feature extraction...")
body_acc_features_train = SensorFeaturesExtractor(body_acc_x_train,body_acc_y_train,body_acc_z_train)
body_acc_features_train = body_acc_features_train.extractFeature()

body_gyro_features_train = SensorFeaturesExtractor(body_gyro_x_train,body_gyro_y_train,body_gyro_z_train)
body_gyro_features_train = body_gyro_features_train.extractFeature()

total_acc_features_train = SensorFeaturesExtractor(total_acc_x_train,total_acc_y_train,total_acc_z_train)
total_acc_features_train = total_acc_features_train.extractFeature()

X_train = np.concatenate((body_acc_features_train,body_gyro_features_train,total_acc_features_train),axis=1)
print(X_train.shape)


body_acc_features_test = SensorFeaturesExtractor(body_acc_x_test,body_acc_y_test,body_acc_z_test)
body_acc_features_test = body_acc_features_test.extractFeature()

body_gyro_features_test = SensorFeaturesExtractor(body_gyro_x_test,body_gyro_y_test,body_gyro_z_test)
body_gyro_features_test = body_gyro_features_test.extractFeature()

total_acc_features_test = SensorFeaturesExtractor(total_acc_x_test,total_acc_y_test,total_acc_z_test)
total_acc_features_test = total_acc_features_test.extractFeature()

X_test = np.concatenate((body_acc_features_test,body_gyro_features_test,total_acc_features_test),axis=1)
print(X_test.shape)

print("shuffling data and new train test division...")
X = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

print('Feature scaling...')
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

print("Random Forest grid search cross validation training...")
random_forest_parameters = [{'n_estimators':[10,50,100,150,200,250,300],'criterion':['entropy','gini']}]
rf_classifier = GridSearchCV(estimator=RandomForestClassifier(),param_grid=random_forest_parameters,scoring='f1_weighted',cv=5,n_jobs=-1)
rf_classifier = rf_classifier.fit(X_train,y_train)
print('Best parameters value: '+str(rf_classifier.best_params_)+'\n')
print('Best scores on 5-Fold cross validation: '+str(rf_classifier.best_score_)+'\n')
y_pred = rf_classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix:\n')
print(cm)
print('Accuracy score: '+str(accuracy_score(y_test,y_pred)))
print('F1 score: '+str(f1_score(y_test,y_pred,average='weighted'))+'\n')


