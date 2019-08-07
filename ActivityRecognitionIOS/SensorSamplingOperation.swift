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
    var sensorAvaiable = false
    var simulatedAccX = 1.0128170e+000
    var simulatedAccY = -1.2321670e-001
    var simulatedAccZ = 1.0293410e-001
    var simulatedGyroX = 3.0191220e-002
    var simulatedGyroY = 6.6013620e-002
    var simulatedGyroZ = 2.2858640e-002
    var accXList = [Double]()
    var accYList = [Double]()
    var accZList = [Double]()
    var accTimestamps = [Double]()
    var gyroXList = [Double]()
    var gyroYList = [Double]()
    var gyroZList = [Double]()
    var gyroTimestamps = [Double]()
    
    
    func startSensors(){
        if self.motion.isAccelerometerAvailable && self.motion.isGyroAvailable {
            sensorAvaiable = true
            self.motion.accelerometerUpdateInterval = 1.0 / 50.0
            self.motion.gyroUpdateInterval = 1.0 / 50.0
            self.motion.startAccelerometerUpdates()
            self.motion.startGyroUpdates()
        }else{
            print("sensors not available")
        }
        DispatchQueue.main.async {
            self.timer = Timer.scheduledTimer(withTimeInterval: 1.0/50.0, repeats: true){ timer in
                if self.sensorAvaiable{
                    if let dataGyro = self.motion.gyroData, let dataAcc = self.motion.accelerometerData {
                        let xGyro = dataGyro.rotationRate.x
                        self.gyroXList.append(xGyro)
                        let yGyro = dataGyro.rotationRate.y
                        self.gyroYList.append(yGyro)
                        let zGyro = dataGyro.rotationRate.z
                        self.gyroZList.append(zGyro)
                        let tGyro = dataGyro.timestamp
                        self.gyroTimestamps.append(tGyro)
                        let xAcc = dataAcc.acceleration.x
                        self.accXList.append(xAcc)
                        let yAcc = dataAcc.acceleration.y
                        self.accYList.append(yAcc)
                        let zAcc = dataAcc.acceleration.z
                        self.accZList.append(zAcc)
                        let tAcc = dataAcc.timestamp
                        self.accTimestamps.append(tAcc)
                    }
                
                }else{
                    self.accXList.append(self.simulatedAccX)
                    self.accYList.append(self.simulatedAccY)
                    self.accZList.append(self.simulatedAccZ)
                    self.gyroXList.append(self.simulatedGyroX)
                    self.gyroYList.append(self.simulatedGyroY)
                    self.gyroZList.append(self.simulatedGyroZ)
                    
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
                //print("operation canceled")
                stopSensors()
                return
            }
        }
    }
}

