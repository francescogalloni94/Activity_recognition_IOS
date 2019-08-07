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
    
    var neural_network: nn_ar
    var random_forets : rf_ar
    var svm : svm_ar
    var scaler : feature_scaler
    
    let labelMapping = [
        1:"WALKING",
        2:"WALKING_UPSTAIRS",
        3:"WALKING_DOWNSTAIRS",
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
    
    private func getFeaturesScaled(input:[feature_scalerInput])->[feature_scalerOutput]{
        guard let output = try? self.scaler.predictions(inputs:input) else {
            fatalError("Unexpected runtime error.")
        }
        return output
    }
    
    func getNeuralNetworkPrediction(inputs:[nn_arInput])->[nn_arOutput]{
        guard let output = try? self.neural_network.predictions(inputs:inputs) else {
            fatalError("Unexpected runtime error.")
        }
        return output
    }
    
    func getRandomForestPrediction(inputs:[[Double]])->[rf_arOutput]{
        var convertedInput = convertRFInput(input: convertToMultiArray(input: inputs))
        guard let output = try? self.random_forets.predictions(inputs:convertedInput) else {
            fatalError("Unexpected runtime error.")
        }
        return output
    }
    
    func getSVMPrediction(inputs:[svm_arInput])->[svm_arOutput]{
        guard let output = try? self.svm.predictions(inputs:inputs) else {
            fatalError("Unexpected runtime error.")
        }
        return output
    }
    
}
