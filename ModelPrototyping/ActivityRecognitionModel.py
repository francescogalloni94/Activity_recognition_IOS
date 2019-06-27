import pandas as pd
import numpy as np
from SensorFeaturesExtractor import SensorFeaturesExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Input,Dense
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from confusionMatrix import plot_confusion_matrix
import coremltools

pwd_x_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/Inertial Signals'
pwd_y_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/y_train.txt'
pwd_x_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/Inertial Signals'
pwd_y_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/y_test.txt'
pwd_labels = '../ActivityRecognitionDataset/UCI HAR Dataset/activity_labels.txt'

print("Reading data...\n")
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

def get_labels(labels_mapping,y_true,y_pred):
    y_true_labels = []
    y_pred_labels = []
    for element in y_true:
        label = labels_mapping[element-1][1]
        y_true_labels.append(label)
    for element in y_pred:
        label = labels_mapping[element-1][1]
        y_pred_labels.append(label)
    return np.array(y_true_labels),np.array(y_pred_labels)

print("Feature extraction...\n")
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

print("shuffling data and new train test division...\n")
X = np.concatenate((X_train,X_test))
y = np.concatenate((y_train,y_test))
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

print('Feature scaling...\n')
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

print("Random Forest grid search cross validation training...")
rf_parameters = [{'n_estimators':[10,50,100,150,200,250,300],'criterion':['entropy','gini']}]
rf_classifier = GridSearchCV(estimator=RandomForestClassifier(),param_grid=rf_parameters,scoring='f1_weighted',cv=5,n_jobs=-1)
rf_classifier = rf_classifier.fit(X_train,y_train)
print('Best parameters value: '+str(rf_classifier.best_params_)+'\n')
print('Best scores on 5-Fold cross validation: '+str(rf_classifier.best_score_)+'\n')
y_pred_rf = rf_classifier.predict(X_test)
y_test_rf,y_pred_rf = get_labels(labels_mapping,y_test,y_pred_rf)
cm_rf = confusion_matrix(y_test_rf,y_pred_rf)
print('Confusion Matrix:\n')
print(cm_rf)
plot_confusion_matrix(cm_rf,filename='RF_cm.png',title='Random Forest Activity Recognition')
print('Accuracy score: '+str(accuracy_score(y_test_rf,y_pred_rf)))
print('F1 score: '+str(f1_score(y_test_rf,y_pred_rf,average='weighted'))+'\n')

rf_coreml_model = coremltools.converters.sklearn.convert(rf_classifier.best_estimator_)
rf_coreml_model.save('rf_ar.mlmodel')


print("Support vector machine grid search cross validation training...")
svm_parameters = [{'kernel':['linear','poly','rbf'],'gamma':['scale']}]
svm_classifier = GridSearchCV(estimator=SVC(),param_grid=svm_parameters,scoring='f1_weighted',cv=5,n_jobs=-1)
svm_classifier = svm_classifier.fit(X_train,y_train)
print('Best parameters value: '+str(svm_classifier.best_params_)+'\n')
print('Best scores on 5-Fold cross validation: '+str(svm_classifier.best_score_)+'\n')
y_pred_svm = svm_classifier.predict(X_test)
y_test_svm,y_pred_svm = get_labels(labels_mapping,y_test,y_pred_svm)
cm_svm = confusion_matrix(y_test_svm,y_pred_svm)
print('Confusion Matrix:\n')
print(cm_svm)
plot_confusion_matrix(cm_svm,filename='SVM_cm.png',title='Support Vector Machine Activity Recognition')
print('Accuracy score: '+str(accuracy_score(y_test_svm,y_pred_svm)))
print('F1 score: '+str(f1_score(y_test_svm,y_pred_svm,average='weighted'))+'\n')

svm_coreml_model = coremltools.converters.sklearn.convert(svm_classifier.best_estimator_)
svm_coreml_model.save('svm_ar.mlmodel')


def create_model(input_shape,hidden_layers,hidden_units,classes):
    input = Input((input_shape,))
    for i in range(hidden_layers):
        if i==0:
             structure = Dense(units=hidden_units,kernel_initializer='glorot_uniform',activation='relu')(input)
        else:
            structure = Dense(units=hidden_units, kernel_initializer='glorot_uniform', activation='relu')(structure)


    structure = Dense(units=classes,kernel_initializer='glorot_uniform',activation='softmax')(structure)
    model = Model(inputs=input,outputs=structure)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("Neural Network grid search cross validation training...")
keras_classifier = KerasClassifier(build_fn=create_model,input_shape=0,hidden_layers=0,hidden_units=0,classes=0)
nn_parameters = [{'input_shape':[X_train.shape[1]],'hidden_layers':[1,2,3],'hidden_units':[10,20,50],
                  'batch_size':[100],'epochs':[50],'classes':[6]}]
nn_classifier = GridSearchCV(estimator=keras_classifier,param_grid=nn_parameters,scoring='f1_weighted',cv=5,n_jobs=-1,error_score='raise')
nn_classifier = nn_classifier.fit(X_train,y_train)
print('Best parameters value: '+str(nn_classifier.best_params_)+'\n')
print('Best scores on 5-Fold cross validation: '+str(nn_classifier.best_score_)+'\n')
y_pred_nn = nn_classifier.predict(X_test)
y_test_nn,y_pred_nn = get_labels(labels_mapping,y_test,y_pred_nn)
cm_nn = confusion_matrix(y_test_nn,y_pred_nn)
print('Confusion Matrix:\n')
print(cm_nn)
plot_confusion_matrix(cm_nn,filename='NN_cm.png',title='Neural Network Activity Recognition')
print('Accuracy score: '+str(accuracy_score(y_test_nn,y_pred_nn)))
print('F1 score: '+str(f1_score(y_test_nn,y_pred_nn,average='weighted'))+'\n')


nn_parameters = nn_classifier.best_params_
nn_model = create_model(X_train.shape[1],nn_parameters['hidden_layers'],nn_parameters['hidden_units'],6)
encoder = OneHotEncoder(categories='auto')
y_train_nn = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
nn_model.fit(x=X_train,y=y_train_nn,epochs=50,batch_size=100)

nn_coreml_model = coremltools.converters.keras.convert(nn_model)
nn_coreml_model.save('nn_ar.mlmodel')


