//
//  SensorSamplingOperation.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 30/07/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import Foundation
import CoreMotion


class SensorSamplingOperation: Operation{
    
    let motion = CMMotionManager()
    var timer: Timer!
    
    
    func startSensors(){
        if self.motion.isAccelerometerAvailable && self.motion.isGyroAvailable {
            self.motion.accelerometerUpdateInterval = 1.0 / 50.0
            self.motion.gyroUpdateInterval = 1.0 / 50.0
            self.motion.startAccelerometerUpdates()
            self.motion.startGyroUpdates()
        }else{
            print("sensors not available")
        }
        DispatchQueue.main.async {
            self.timer = Timer.scheduledTimer(withTimeInterval: 1.0/50.0, repeats: true){ timer in
                    print("inside timer")
                    if let dataGyro = self.motion.gyroData, let dataAcc = self.motion.accelerometerData {
                        let xGyro = dataGyro.rotationRate.x
                        let yGyro = dataGyro.rotationRate.y
                        let zGyro = dataGyro.rotationRate.z
                        let tGyro = dataGyro.timestamp
                        let xAcc = dataAcc.acceleration.x
                        let yAcc = dataAcc.acceleration.y
                        let zAcc = dataAcc.acceleration.z
                        let tAcc = dataAcc.timestamp
                 
                
                }
            }
        }
        
    }
    
    func stopSensors(){
        self.timer.invalidate()
        self.motion.stopGyroUpdates()
        self.motion.stopAccelerometerUpdates()
        
    }
    
    override public func main() {
        startSensors()
        while true {
          
            if(isCancelled){
                print("operation canceled")
                stopSensors()
                return
            }
        }
    }
}

