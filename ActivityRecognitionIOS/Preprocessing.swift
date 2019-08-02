//
//  Preprocessing.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 02/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import Foundation

class Preprocessing {
    
    var xAcc: [Double]
    var yAcc: [Double]
    var zAcc: [Double]
    var xGyro: [Double]
    var yGyro: [Double]
    var zGyro: [Double]
    
    init(xAcc:[Double],yAcc:[Double],zAcc:[Double],xGyro:[Double],yGyro:[Double],zGyro:[Double]){
        self.xAcc = xAcc
        self.yAcc = yAcc
        self.zAcc = zAcc
        self.xGyro = xGyro
        self.yGyro = yGyro
        self.zGyro = zGyro
        self.xAcc = medianFilter(input: self.xAcc)
        self.yAcc = medianFilter(input: self.yAcc)
        self.zAcc = medianFilter(input: self.zAcc)
        self.xGyro = medianFilter(input: self.xGyro)
        self.yGyro = medianFilter(input: self.yGyro)
        self.zGyro = medianFilter(input: self.zGyro)
    }
    
    func medianFilter(input:[Double])->[Double]{
        var filteredArray = [Double]()
        for (index,element) in input.enumerated() {
            var medianList = [Double]()
            medianList.append(element)
            var lastElementMinus = input[index]
            var lastElementPlus = input[index]
            for n in 1...3{
                if input.indices.contains(index-n){
                    medianList.append(input[index-n])
                    lastElementMinus = input[index-n]
                }else{
                    medianList.append(lastElementMinus)
                }
                if input.indices.contains(index+n){
                    medianList.append(input[index+n])
                    lastElementPlus = input[index+n]
                }else{
                    medianList.append(lastElementPlus)
                }
            }
            let medianElement = median(array:medianList)
            filteredArray.append(medianElement)
            
        }
       return filteredArray
    }
    
    func median(array: [Double]) -> Double {
        let sorted = array.sorted()
        if sorted.count % 2 == 0 {
            return Double((sorted[(sorted.count / 2)] + sorted[(sorted.count / 2) - 1])) / 2
        } else {
            return Double(sorted[(sorted.count - 1) / 2])
        }
    }
    
    func butterWorthFilter(){
        
    }
    
    func butterWorthGravityFilter(){
        
    }
    
    func segmentation(){
        
    }
}
