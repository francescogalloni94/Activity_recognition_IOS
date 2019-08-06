//
//  FeatureExtraction.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 02/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import Foundation
import Accelerate



class FeatureExtraction{
    
    private var totalAccX : [[Double]]
    private var totalAccY : [[Double]]
    private var totalAccZ : [[Double]]
    private var bodyAccX : [[Double]]
    private var bodyAccY : [[Double]]
    private var bodyAccZ : [[Double]]
    private var gyroX : [[Double]]
    private var gyroY : [[Double]]
    private var gyroZ : [[Double]]
    
    private var totalAccXFeatures = [[Double]]()
    private var totalAccYFeatures = [[Double]]()
    private var totalAccZFeatures = [[Double]]()
    private var bodyAccXFeatures = [[Double]]()
    private var bodyAccYFeatures = [[Double]]()
    private var bodyAccZFeatures = [[Double]]()
    private var gyroXFeatures = [[Double]]()
    private var gyroYFeatures = [[Double]]()
    private var gyroZFeatures = [[Double]]()
    
    
    
    init(totalX:[[Double]],totalY:[[Double]],totalZ:[[Double]],bodyX:[[Double]],bodyY:[[Double]],bodyZ:[[Double]]
        ,gyroX:[[Double]],gyroY:[[Double]],gyroZ:[[Double]]){
        self.totalAccX = totalX
        self.totalAccY = totalY
        self.totalAccZ = totalZ
        self.bodyAccX = bodyX
        self.bodyAccY = bodyY
        self.bodyAccZ = bodyZ
        self.gyroX = gyroX
        self.gyroY = gyroY
        self.gyroZ = gyroZ
        
        self.totalAccXFeatures = computeFeatures(input: self.totalAccX)
        self.totalAccYFeatures = computeFeatures(input: self.totalAccY)
        self.totalAccZFeatures = computeFeatures(input: self.totalAccZ)
        self.bodyAccXFeatures = computeFeatures(input: self.bodyAccX)
        self.bodyAccYFeatures = computeFeatures(input: self.bodyAccY)
        self.bodyAccZFeatures = computeFeatures(input: self.bodyAccZ)
        self.gyroXFeatures = computeFeatures(input: self.gyroX)
        self.gyroYFeatures = computeFeatures(input: self.gyroY)
        self.gyroZFeatures = computeFeatures(input: self.gyroZ)
        
    }
    
    private func computeFeatures(input:[[Double]])->[[Double]]{
        var featuresMatrix = [[Double]]()
        for element in input {
            var featureVector = [Double]()
            var mean = Sigma.average(element)
            var std = Sigma.standardDeviationPopulation(element)
            var mad = mean_absolute_dev(input:element)
            var min = Sigma.min(element)
            var max = Sigma.max(element)
            var iqr = interquartile_range(input: element)
            var fftResult = fft(element)
            var meanFreq = Sigma.average(fftResult)
            var maxFreq = Sigma.max(fftResult)
            var skewFreq = Sigma.skewnessA(fftResult)
            var kurtosisFreq = Sigma.kurtosisA(fftResult)
            
            //da togliere dopo simulazione
            if skewFreq == nil{
                skewFreq = 0.0
            }
            if kurtosisFreq == nil{
                kurtosisFreq = 0.0
            }
            //
            
            featureVector.append(mean!)
            featureVector.append(std!)
            featureVector.append(mad)
            featureVector.append(min!)
            featureVector.append(max!)
            featureVector.append(iqr)
            featureVector.append(meanFreq!)
            featureVector.append(maxFreq!)
            featureVector.append(skewFreq!)
            featureVector.append(kurtosisFreq!)
            featuresMatrix.append(featureVector)
            
        }
        return featuresMatrix
    }
    
    private func mean_absolute_dev(input:[Double])->Double{
        let median = Sigma.median(input)
        var mad = 0.0
        for element in input{
            mad += abs(element-median!)
        }
        return mad/Double(input.count)
    }
    
    private func interquartile_range(input:[Double])->Double{
        let ThirdPercentile = Sigma.percentile(input,percentile:0.75)
        let FirstPercentile = Sigma.percentile(input, percentile: 0.25)
        let iqr = ThirdPercentile!-FirstPercentile!
        return iqr
    }
    
    
    //presa da libreria Surge che usa il framework accelerate
    private func fft(_ input: [Double]) -> [Double] {
        var real = [Double](input)
        var imaginary = [Double](repeating: 0.0, count: input.count)
        var splitComplex = DSPDoubleSplitComplex(realp: &real, imagp: &imaginary)
        
        let length = vDSP_Length(floor(log2(Float(input.count))))
        let radix = FFTRadix(kFFTRadix2)
        let weights = vDSP_create_fftsetupD(length, radix)
        withUnsafeMutablePointer(to: &splitComplex) { splitComplex in
            vDSP_fft_zipD(weights!, splitComplex, 1, length, FFTDirection(FFT_FORWARD))
        }
        
        var magnitudes = [Double](repeating: 0.0, count: input.count)
        withUnsafePointer(to: &splitComplex) { splitComplex in
            magnitudes.withUnsafeMutableBufferPointer { magnitudes in
                vDSP_zvmagsD(splitComplex, 1, magnitudes.baseAddress!, 1, vDSP_Length(input.count))
            }
        }
        
        var normalizedMagnitudes = [Double](repeating: 0.0, count: input.count)
        normalizedMagnitudes.withUnsafeMutableBufferPointer { normalizedMagnitudes in
            for (index,element) in magnitudes.enumerated(){
                magnitudes[index] = sqrt(element)
            }
            vDSP_vsmulD(magnitudes, 1, [2.0 / Double(input.count)], normalizedMagnitudes.baseAddress!, 1, vDSP_Length(input.count))
        }
        
        vDSP_destroy_fftsetupD(weights)
        
        return normalizedMagnitudes
    }
    
    
    public func getFeaturesMatrix()->[[Double]]{
        var featureMatrix = [[Double]]()
        for (index,element) in self.totalAccXFeatures.enumerated(){
            var bodyVector = bodyAccXFeatures[index]+bodyAccYFeatures[index]+bodyAccZFeatures[index]
            var gyroVector = gyroXFeatures[index]+gyroYFeatures[index]+gyroZFeatures[index]
            var totalVector = totalAccXFeatures[index]+totalAccYFeatures[index]+totalAccZFeatures[index]
            var featureVector = bodyVector+gyroVector+totalVector
            featureMatrix.append(featureVector)
        }
        return featureMatrix
    }
    
    
    
    
}
