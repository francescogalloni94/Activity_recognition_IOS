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
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
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
        }else if buttonString == "STOP"{
            
        }
        
    }
    
}

