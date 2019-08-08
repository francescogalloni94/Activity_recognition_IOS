//
//  Predictors.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 02/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import Foundation
import CoreML

class Predictors {
    
    private var neural_network: nn_ar
    private var random_forets : rf_ar
    private var svm : svm_ar
    private var scaler : feature_scaler
    
    private let labelMapping = [
        1:"WALKING",
        2:"WALKING_U",
        3:"WALKING_D",
        4:"SITTING",
        5:"STANDING",
        6:"LAYING"]
    
    
    init() {
        self.neural_network = nn_ar()
        self.random_forets = rf_ar()
        self.svm = svm_ar()
        self.scaler = feature_scaler()
    }
    
    private func convertToMultiArray(input:[[Double]])->[MLMultiArray]{
        var multis = [MLMultiArray]()
        for (arrayIndex,array) in input.enumerated(){
            guard let multi = try? MLMultiArray(shape:[90], dataType:.double) else {
                fatalError("Unexpected runtime error. MLMultiArray")
            }
           
            for (index,element) in input[arrayIndex].enumerated(){
                multi[index] = NSNumber(value: element)
            }
            multis.append(multi)
        }
        return multis
    }
    
    private func convertNNInput(input:[MLMultiArray])->[nn_arInput]{
        var converted = [nn_arInput]()
        for (index,element) in input.enumerated(){
            converted.append(nn_arInput(input1: element))
        }
        return converted
    }
    
    private func convertSVMInput(input:[MLMultiArray])->[svm_arInput]{
        var converted = [svm_arInput]()
        for (index,element) in input.enumerated(){
            converted.append(svm_arInput(input: element))
        }
        return converted
    }
    
    private func convertRFInput(input:[MLMultiArray])->[rf_arInput]{
        var converted = [rf_arInput]()
        for (index,element) in input.enumerated(){
            converted.append(rf_arInput(input: element))
        }
        return converted
    }
    
    private func convertScalerInput(input:[[Double]])->[feature_scalerInput]{
        var inputC = convertToMultiArray(input: input)
        var converted = [feature_scalerInput]()
        for (index,element) in inputC.enumerated(){
            converted.append(feature_scalerInput(input: element))
        }
        return converted
    }
    
    private func getFeaturesScaled(input:[[Double]])->[MLMultiArray]{
        var converted = convertScalerInput(input: input)
        guard let output = try? self.scaler.predictions(inputs:converted) else {
            fatalError("Unexpected runtime error.")
        }
        var scaledFeatures = [MLMultiArray]()
        for(index,element) in output.enumerated(){
            scaledFeatures.append(output[index].transformed_features)
        }
        return scaledFeatures
    }
    
    func getNeuralNetworkPrediction(inputs:[[Double]])->String{
        var convertedScaledInput = convertNNInput(input: getFeaturesScaled(input: inputs))
        guard let output = try? self.neural_network.predictions(inputs:convertedScaledInput) else {
            fatalError("Unexpected runtime error.")
        }
        var majority = getMajorityPrediction(labels: convertNNOutput(output: output))
        return labelMapping[majority]!
    }
    
    func getRandomForestPrediction(inputs:[[Double]])->String{
        var convertedScaledInput = convertRFInput(input: getFeaturesScaled(input: inputs))
        guard let output = try? self.random_forets.predictions(inputs:convertedScaledInput) else {
            fatalError("Unexpected runtime error.")
        }
        var majority = getMajorityPrediction(labels: convertRFOutput(output: output))
        return labelMapping[majority]!
    }
    
    func getSVMPrediction(inputs:[[Double]])->String{
        var convertedScaledInput = convertSVMInput(input: getFeaturesScaled(input: inputs))
        guard let output = try? self.svm.predictions(inputs:convertedScaledInput) else {
            fatalError("Unexpected runtime error.")
        }
        var majority = getMajorityPrediction(labels: convertSVMOutput(output: output))
        return labelMapping[majority]!
    }
    
    private func convertRFOutput(output:[rf_arOutput])->[Int]{
        var labels = [Int]()
        for element in output{
            labels.append(Int(element.classLabel))
        }
        return labels
    }
    
    
    private func convertSVMOutput(output:[svm_arOutput])->[Int]{
        var labels = [Int]()
        for element in output{
            labels.append(Int(element.classLabel))
        }
        return labels
    }
    
    private func convertNNOutput(output:[nn_arOutput])->[Int]{
        var labels = [Int]()
        for element in output{
            var maxProb = 0.0
            var activity = 0
            var probVector = element.output1
            for i in 0...5 {
                if probVector[i].doubleValue>maxProb{
                    maxProb = probVector[i].doubleValue
                    activity = i+1
                }
            }
           labels.append(activity)
        }
       return labels
    }
    
    
    private func getMajorityPrediction(labels:[Int])->Int{
        let mappedItems = labels.map{ ($0, 1) }
        let counts = Dictionary(mappedItems, uniquingKeysWith: +)
        var maxFreq = 0
        var maxKey = 0
        for (key,value) in counts{
            if value>maxFreq{
                maxFreq = value
                maxKey = key
            }
        }
        return maxKey
    }
    
    
}
