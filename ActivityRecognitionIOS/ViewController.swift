//
//  ViewController.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 25/06/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var recordingLabel: UILabel!
    @IBOutlet weak var recordingButton: UIButton!
    @IBOutlet weak var stackView: UIStackView!
    var sensorSampler: SensorSamplingOperation!
    let operationQueue = OperationQueue()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        stackView.frame.origin.x = view.frame.width/2 - stackView.frame.width/2
        stackView.frame.origin.y = view.frame.height/2 - stackView.frame.width/2
        let transfrom = CGAffineTransform.init(scaleX: 3.5, y: 3.5)
        activityIndicator.transform = transfrom
        activityIndicator.isHidden = true
    }

    @IBAction func onButtonClick(_ sender: UIButton) {
        let buttonString = recordingButton.titleLabel!.text
        if buttonString == "START" {
            activityIndicator.isHidden = false
            activityIndicator.startAnimating()
            recordingLabel.text = "RECORDING ACTIVITY..."
            recordingButton.setTitle("STOP",for:.normal)
            self.sensorSampler = SensorSamplingOperation()
            sensorSampler.completionBlock = {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else{
                        return
                    }
                    let preprocessing = Preprocessing(xAcc:self.sensorSampler.accXList, yAcc:self.sensorSampler.accYList, zAcc: self.sensorSampler.accZList, xGyro: self.sensorSampler.gyroXList, yGyro: self.sensorSampler.gyroYList, zGyro: self.sensorSampler.gyroZList)
                    let featureExtraction = FeatureExtraction(totalX: preprocessing.preprocessedXTotalAcc, totalY: preprocessing.preprocessedYTotalAcc, totalZ: preprocessing.preprocessedZTotalAcc, bodyX: preprocessing.preprocessedXBodyAcc, bodyY: preprocessing.preprocessedYBodyAcc, bodyZ: preprocessing.preprocessedZBodyAcc, gyroX: preprocessing.preprocessedXGyro, gyroY: preprocessing.preprocessedYGyro, gyroZ: preprocessing.preprocessedZGyro)
                    let featureMatrix = featureExtraction.getFeaturesMatrix()

                }
            }
            operationQueue.addOperation(sensorSampler)
        }else if buttonString == "STOP"{
            sensorSampler.cancel()
            
        }
        
    }
    
}

