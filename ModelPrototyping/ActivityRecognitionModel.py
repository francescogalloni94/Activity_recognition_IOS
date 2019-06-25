import pandas as pd

pwd_x_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/Inertial Signals'
pwd_y_train = '../ActivityRecognitionDataset/UCI HAR Dataset/train/y_train.txt'
pwd_x_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/Inertial Signals'
pwd_y_test = '../ActivityRecognitionDataset/UCI HAR Dataset/test/y_test.txt'

body_acc_x_train = pd.read_csv(pwd_x_train+'/body_acc_x_train.txt',delim_whitespace=True)
body_acc_y_train = pd.read_csv(pwd_x_train+'/body_acc_y_train.txt',delim_whitespace=True)
body_acc_z_train = pd.read_csv(pwd_x_train+'/body_acc_z_train.txt',delim_whitespace=True)

body_gyro_x_train = pd.read_csv(pwd_x_train+'/body_gyro_x_train.txt',delim_whitespace=True)
body_gyro_y_train = pd.read_csv(pwd_x_train+'/body_gyro_y_train.txt',delim_whitespace=True)
body_gyro_z_train = pd.read_csv(pwd_x_train+'/body_gyro_z_train.txt',delim_whitespace=True)

total_acc_x_train = pd.read_csv(pwd_x_train+'/total_acc_x_train.txt',delim_whitespace=True)
total_acc_y_train = pd.read_csv(pwd_x_train+'/total_acc_y_train.txt',delim_whitespace=True)
total_acc_z_train = pd.read_csv(pwd_x_train+'/total_acc_z_train.txt',delim_whitespace=True)

y_train = pd.read_csv(pwd_y_train,delim_whitespace=True)


body_acc_x_test = pd.read_csv(pwd_x_test+'/body_acc_x_test.txt',delim_whitespace=True)
body_acc_y_test = pd.read_csv(pwd_x_test+'/body_acc_y_test.txt',delim_whitespace=True)
body_acc_z_test = pd.read_csv(pwd_x_test+'/body_acc_z_test.txt',delim_whitespace=True)

body_gyro_x_test = pd.read_csv(pwd_x_test+'/body_gyro_x_test.txt',delim_whitespace=True)
body_gyro_y_test = pd.read_csv(pwd_x_test+'/body_gyro_y_test.txt',delim_whitespace=True)
body_gyro_z_test = pd.read_csv(pwd_x_test+'/body_gyro_z_test.txt',delim_whitespace=True)

total_acc_x_test = pd.read_csv(pwd_x_test+'/total_acc_x_test.txt',delim_whitespace=True)
total_acc_y_test = pd.read_csv(pwd_x_test+'/total_acc_y_test.txt',delim_whitespace=True)
total_acc_z_test = pd.read_csv(pwd_x_test+'/total_acc_z_test.txt',delim_whitespace=True)

y_test = pd.read_csv(pwd_y_test,delim_whitespace=True)





