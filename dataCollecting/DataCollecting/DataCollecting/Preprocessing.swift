//
//  Preprocessing.swift
//  ActivityRecognitionIOS
//
//  Created by Francesco Galloni on 02/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import Foundation

class Preprocessing {
    
    private var xTotalAcc: [Double]
    private var yTotalAcc: [Double]
    private var zTotalAcc: [Double]
    private var xBodyAcc: [Double]
    private var yBodyAcc: [Double]
    private var zBodyAcc: [Double]
    private var xGyro: [Double]
    private var yGyro: [Double]
    private var zGyro: [Double]
    var preprocessedXTotalAcc = [[Double]]()
    var preprocessedYTotalAcc = [[Double]]()
    var preprocessedZTotalAcc = [[Double]]()
    var preprocessedXBodyAcc = [[Double]]()
    var preprocessedYBodyAcc = [[Double]]()
    var preprocessedZBodyAcc = [[Double]]()
    var preprocessedXGyro = [[Double]]()
    var preprocessedYGyro = [[Double]]()
    var preprocessedZGyro = [[Double]]()
    
    
    init(xAcc:[Double],yAcc:[Double],zAcc:[Double],xGyro:[Double],yGyro:[Double],zGyro:[Double]){
        self.xTotalAcc = xAcc
        self.yTotalAcc = yAcc
        self.zTotalAcc = zAcc
        self.xGyro = xGyro
        self.yGyro = yGyro
        self.zGyro = zGyro
        self.xBodyAcc = [Double]()
        self.yBodyAcc = [Double]()
        self.zBodyAcc = [Double]()
        
        self.xTotalAcc = medianFilter(input: self.xTotalAcc)
        self.yTotalAcc = medianFilter(input: self.yTotalAcc)
        self.zTotalAcc = medianFilter(input: self.zTotalAcc)
        self.xGyro = medianFilter(input: self.xGyro)
        self.yGyro = medianFilter(input: self.yGyro)
        self.zGyro = medianFilter(input: self.zGyro)
        
        self.xTotalAcc = lowPassFilter(input: self.xTotalAcc, cutOffFrequency: 20)
        self.yTotalAcc = lowPassFilter(input: self.yTotalAcc, cutOffFrequency: 20)
        self.zTotalAcc = lowPassFilter(input: self.zTotalAcc, cutOffFrequency: 20)
        self.xGyro = lowPassFilter(input: self.xGyro, cutOffFrequency: 20)
        self.yGyro = lowPassFilter(input: self.yGyro, cutOffFrequency: 20)
        self.zGyro = lowPassFilter(input: self.zGyro, cutOffFrequency: 20)
        
        self.xBodyAcc = gravityFilter(input: self.xTotalAcc)
        self.yBodyAcc = gravityFilter(input: self.yTotalAcc)
        self.zBodyAcc = gravityFilter(input: self.zTotalAcc)
        
        self.preprocessedXTotalAcc = segmentation(input: self.xTotalAcc)
        self.preprocessedYTotalAcc = segmentation(input: self.yTotalAcc)
        self.preprocessedZTotalAcc = segmentation(input: self.zTotalAcc)
        self.preprocessedXBodyAcc = segmentation(input: self.xBodyAcc)
        self.preprocessedYBodyAcc = segmentation(input: self.yBodyAcc)
        self.preprocessedZBodyAcc = segmentation(input: self.zBodyAcc)
        self.preprocessedXGyro = segmentation(input: self.xGyro)
        self.preprocessedYGyro = segmentation(input: self.yGyro)
        self.preprocessedZGyro = segmentation(input: self.zGyro)
    }
    
    private func medianFilter(input:[Double])->[Double]{
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
    
    private func median(array: [Double]) -> Double {
        let sorted = array.sorted()
        if sorted.count % 2 == 0 {
            return Double((sorted[(sorted.count / 2)] + sorted[(sorted.count / 2) - 1])) / 2
        } else {
            return Double(sorted[(sorted.count - 1) / 2])
        }
    }
    
   private func lowPassFilter(input:[Double],cutOffFrequency:Double)->[Double]{
        var filteredArray = [Double]()
        var previousValue = 0.0
        let dt = 1.0/50.0
        let rc = 1.0/(2*Double.pi*cutOffFrequency)
        let alpha = dt/(rc+dt)
        for element in input {
            var newValue = (alpha*element)+(1.0 - alpha)*previousValue;
            filteredArray.append(newValue)
            previousValue = newValue
        }
        
        return filteredArray
    }
    
    private func gravityFilter(input:[Double])->[Double]{
        var gravityComponent = lowPassFilter(input: input, cutOffFrequency: 0.3)
        var bodyAcc = [Double]()
        for (index,element) in input.enumerated() {
            var bodyValue = element-gravityComponent[index]
            bodyAcc.append(bodyValue)
        }
        return bodyAcc
    }
    
    private func segmentation(input:[Double])->[[Double]]{
        // segment of 128 readings with 50% of overlap
        var array = input
        var segmentedArray = [[Double]]()
        while array.count>=128 {
            var segment = [Double]()
            for n in 0...127{
                segment.append(array[n])
            }
            segmentedArray.append(segment)
            for n in 0...63{
                array.removeFirst()
            }
        }
        
        return segmentedArray
    }
}
