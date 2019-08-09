//
//  ViewController.swift
//  DataCollecting
//
//  Created by Francesco Galloni on 09/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import UIKit

class ViewController: UIViewController, UIPickerViewDelegate, UIPickerViewDataSource {
   
    

    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var recordingLabel: UILabel!
    @IBOutlet weak var recordingButton: UIButton!
    @IBOutlet weak var stackView: UIStackView!
    var sensorSampler: SensorSamplingOperation!
    let operationQueue = OperationQueue()
    var featureMatrix = [[Double]]()
    @IBOutlet weak var picker: UIPickerView!
    var pickerData: [String] = [String]()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        viewSetup()
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
                    var selectedLabel = self.pickerData[self.picker.selectedRow(inComponent: 0)]
                    print(selectedLabel)
                    var csvFile = CsvFile(preprocessing: preprocessing, label: selectedLabel)
                    print("finish")
                    self.viewSetup()
                    
                }
            }
            operationQueue.addOperation(sensorSampler)
        }else if buttonString == "STOP"{
            sensorSampler.cancel()
            
            
        }
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        viewSetup()
        
    }
    
    func segue(identifier: String){
        performSegue(withIdentifier: identifier, sender: self)
    }
    
    
    
    func viewSetup(){
        let transfrom = CGAffineTransform.init(scaleX: 3.5, y: 3.5)
        activityIndicator.transform = transfrom
        activityIndicator.isHidden = true
        recordingLabel.text = "START RECORDING AN ACTIVITY"
        recordingButton.setTitle("START",for:.normal)
        pickerData = ["WALKING", "WALKING_U", "WALKING_D", "SITTING", "STANDING", "LAYING"]
        self.picker.delegate = self
        self.picker.dataSource = self
    }
    
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return pickerData.count
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int)->String {
        return pickerData[row]
    }
    
   


}

