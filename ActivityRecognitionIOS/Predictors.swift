//
//  Predictors.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 02/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import Foundation

class Predictors {
    
    var neural_network: nn_ar
    var random_forets : rf_ar
    var svm : svm_ar
    
    init() {
        neural_network = nn_ar()
        self.random_forets = rf_ar()
        self.svm = svm_ar()
    }
    
    func getNeuralNetworkPredictions(inputs:[nn_arInput])->[nn_arOutput]{
        guard let output = try? self.neural_network.predictions(inputs:inputs) else {
            fatalError("Unexpected runtime error.")
        }
        return output
    }
    
    func getRandomForestPrediction(inputs:[rf_arInput])->[rf_arOutput]{
        guard let output = try? self.random_forets.predictions(inputs:inputs) else {
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
