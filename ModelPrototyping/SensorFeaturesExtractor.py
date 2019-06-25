import numpy as np
from statsmodels import robust
from scipy.fftpack import fft

class SensorFeaturesExtractor:

    def __init__(self,sensor_x,sensor_y,sensor_z):
        self.sensor_x = sensor_x
        self.sensor_y = sensor_y
        self.sensor_z = sensor_z


    def extractFeature(self):

        means_x = self.compute_features(self.sensor_x)
        means_y = self.compute_features(self.sensor_y)
        means_z = self.compute_features(self.sensor_z)
        print(means_x.shape)
        #print(means_y.shape)
        #print(means_z.shape)


    def compute_features(self,sensor):
        features_matrix = []
        for i in range(sensor.shape[0]):
            feature_vector = []
            row = sensor[i,:]
            mean = np.mean(row)
            std = np.std(row)
            mad = robust.mad(row)
            min = np.min(row)
            max = np.max(row)
            frequency_row = fft(row)

            feature_vector.append(mean)
            feature_vector.append(std)
            feature_vector.append(mad)
            feature_vector.append(min)
            feature_vector.append(max)
            features_matrix.append(feature_vector)
        return np.array(features_matrix)






