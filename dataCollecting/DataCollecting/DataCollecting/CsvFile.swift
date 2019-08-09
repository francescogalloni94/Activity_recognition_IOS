//
//  CsvFile.swift
//  DataCollecting
//
//  Created by Francesco Galloni on 09/08/2019.
//  Copyright © 2019 Francesco Galloni. All rights reserved.
//

import Foundation

class CsvFile{
    
    var preprocessing : Preprocessing
    var label : String
    private let labelMapping = [
        "WALKING":1,
        "WALKING_U":2,
        "WALKING_D":3,
        "SITTING":4,
        "STANDING":5,
        "LAYING":6]
    
    init(preprocessing:Preprocessing,label:String){
        self.preprocessing = preprocessing
        self.label = label
        writeFile(fileString: createLabelString(preprocessed: self.preprocessing.preprocessedXBodyAcc, label: self.label), fileName: self.label+"Labels")
        print(readingFile(fileName: self.label+"Labels"))
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedXBodyAcc), fileName: self.label+"XBodyAcc")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedYBodyAcc), fileName: self.label+"YBodyAcc")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedZBodyAcc), fileName: self.label+"ZBodyAcc")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedXTotalAcc), fileName: self.label+"XTotalAcc")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedYTotalAcc), fileName: self.label+"YTotalAcc")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedZTotalAcc), fileName: self.label+"ZTotalAcc")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedXGyro), fileName: self.label+"XGyro")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedYGyro), fileName: self.label+"YGyro")
        writeFile(fileString: createDataString(data: self.preprocessing.preprocessedZGyro), fileName: self.label+"ZGyro")
        
    }
    
    func createDataString(data:[[Double]])->String{
        var csvString = ""
        for array in data{
            for (index,value) in array.enumerated(){
                if index == array.count-1{
                    csvString += String(value)+"\n"
                }else{
                    csvString += String(value)+","
                }
            }
        }
        return csvString
    }
    
    func createLabelString(preprocessed:[[Double]],label:String)->String{
        var csvString = ""
        var rowCount = preprocessed.count
        for i in 0...rowCount-1 {
            csvString += String(labelMapping[label]!)+"\n"
        }
        return csvString
    }
    
    func writeFile(fileString:String,fileName:String){
        let fileManager = FileManager.default
        do {
            let path = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            let fileURL = path.appendingPathComponent(fileName+".csv")
            try fileString.write(to: fileURL, atomically: true, encoding: .utf8)
        } catch {
            print("error creating file")
        }
    }
    
    func readingFile(fileName:String)->String{
        var file = ""
        let fileManager = FileManager.default
        do {
            let path = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            let fileURL = path.appendingPathComponent(fileName+".csv")
            file = try String(contentsOf: fileURL, encoding: .utf8)
        }
        catch {print("error reading file")}
        return file
    }
}
