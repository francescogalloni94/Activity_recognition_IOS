//
//  PredictionViewController.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 07/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import UIKit

class PredictionViewController: UIViewController {
    
    var featureMatrix : [[Double]]?
    var predictors : Predictors?

    override func viewDidLoad() {
        super.viewDidLoad()
        self.predictors = Predictors()
        let outputRF = predictors!.getRandomForestPrediction(inputs: self.featureMatrix!)
        print("Random forest: "+outputRF)
        let outputSVM = predictors!.getSVMPrediction(inputs: self.featureMatrix!)
        print("SVM: "+outputSVM)
        let outputNN = predictors!.getNeuralNetworkPrediction(inputs: self.featureMatrix!)
        print("NN: "+outputNN)

    }
    

    

}
