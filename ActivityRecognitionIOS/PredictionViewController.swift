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
        //print("view controlleeeers")
        //print(self.featureMatrix!)
        self.predictors = Predictors()
        var output = predictors?.getRandomForestPrediction(inputs: self.featureMatrix!)
        print(output![0].classLabel)
        /*print(self.featureMatrix?.count)
        for element in self.featureMatrix!{
            print(element.count)
        }*/
        // Do any additional setup after loading the view.
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
