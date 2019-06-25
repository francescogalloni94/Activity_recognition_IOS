import pandas as pd
import numpy as np
from SensorFeaturesExtractor import SensorFeaturesExtractor

pwd_x_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/Inertial Signals'
pwd_y_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/y_train.txt'
pwd_x_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/Inertial Signals'
pwd_y_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/y_test.txt'
pwd_labels = '../ActivityRecognitionDataset/UCI HAR Dataset/activity_labels.txt'

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

labels_mapping = pd.read_csv(pwd_labels,delim_whitespace=True,header=None).values

body_acc_features_train = SensorFeaturesExtractor(body_acc_x_train,body_acc_y_train,body_acc_z_train)
body_acc_features_train = body_acc_features_train.extractFeature()

body_gyro_features_train = SensorFeaturesExtractor(body_gyro_x_train,body_gyro_y_train,body_gyro_z_train)
body_gyro_features_train = body_gyro_features_train.extractFeature()

total_acc_features_train = SensorFeaturesExtractor(total_acc_x_train,total_acc_y_train,total_acc_z_train)
total_acc_features_train = total_acc_features_train.extractFeature()

X_train = np.concatenate((body_acc_features_train,body_gyro_features_train,total_acc_features_train),axis=1)
print(X_train.shape)







