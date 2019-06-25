import numpy as np
from statsmodels import robust
from scipy.fftpack import fft
from scipy.stats import iqr
from scipy.stats import skew
from scipy.stats import kurtosis


class SensorFeaturesExtractor:

    def __init__(self,sensor_x,sensor_y,sensor_z):
        self.sensor_x = sensor_x
        self.sensor_y = sensor_y
        self.sensor_z = sensor_z


    def extractFeature(self):

        means_x = self.compute_features(self.sensor_x)
        means_y = self.compute_features(self.sensor_y)
        means_z = self.compute_features(self.sensor_z)
        concatenate = np.concatenate((means_x,means_y,means_z),axis=1)
        return concatenate


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
            interquartile_range = iqr(row)
            frequency_row = fft(row)
            mean_freq = np.mean(frequency_row)
            max_freq = np.max(frequency_row)
            skew_freq = skew(frequency_row)
            kurtosis_freq = kurtosis(frequency_row)
            feature_vector.append(mean)
            feature_vector.append(std)
            feature_vector.append(mad)
            feature_vector.append(min)
            feature_vector.append(max)
            feature_vector.append(interquartile_range)
            feature_vector.append(mean_freq)
            feature_vector.append(max_freq)
            feature_vector.append(skew_freq)
            feature_vector.append(kurtosis_freq)
            features_matrix.append(feature_vector)
        return np.array(features_matrix)






