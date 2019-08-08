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
    @IBOutlet weak var svmPrediction: UILabel!
    @IBOutlet weak var rfPrediction: UILabel!
    @IBOutlet weak var nnPrediction: UILabel!
    @IBOutlet weak var stackView: UIStackView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        predictors = Predictors()
        let outputRF = predictors!.getRandomForestPrediction(inputs: featureMatrix!)
        rfPrediction.text = outputRF
        print("Random forest: "+outputRF)
        let outputSVM = predictors!.getSVMPrediction(inputs: featureMatrix!)
        svmPrediction.text = outputSVM
        print("SVM: "+outputSVM)
        let outputNN = predictors!.getNeuralNetworkPrediction(inputs: featureMatrix!)
        nnPrediction.text = outputNN
        print("NN: "+outputNN)

    }
    

    

}
