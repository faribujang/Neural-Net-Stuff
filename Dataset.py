import os
import time
import tensorflow as tf
from random import randint
from tensorflow.examples.tutorials.mnist import input_data

#Change Log , V1.0 
#Created Imagefile Generator
#Created Dataset Class
#Created Data Assembler Class

#Create New Image File with IDs
CreateImageFile = False
ImageFileName = "Imagefile.txt"

#Open Mnist Dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Converts an 1D array of form inputs [1,2,3,4,5...n] & "," to "[1,2,3,4,5...n]"
def arrayToString(inputarray, delimiter):
    outString = ""
    for i in range(len(inputarray)):
        outString = outString + str(delimiter) + str(inputarray[i])
    outString = outString[1:]
    outString = "["+outString+"]"
    return outString

#Data Set Class, holds sorted data indexs and label position pairs
class Dataset:
    #Initialize Class and sets Hold Data varibles
    def __init__(self):
        #sortedindexs[n] is the array of indexs of class n, so sortedindexs[1], has all indexs of 1s
        self.sortedIndexs = [[], [], [], [], [], [], [], [], [], []]
        self.sortedLabelPositionPairs = []
        self.sortIndexs()

    #Converts a Label Position pair into a callable index
    def getIndexByLabelPositionPair(self, label, labelpositon):  
        return self.sortedIndexs[label][labelpositon]
    
    #Converts a Index into a Label Position Pair
    def getLabelPositionPairByIndex(self, index):
        return tuple(self.sortedLabelPositionPairs[index])

    #Returns Sorted Indexs
    def getSortedIndexs(self):
        return self.sortedIndexs

    #Return Sorted LabelPositionPairs
    def getSortedLabelPositionPairs():
        return self.sortedLabelPositionPairs
    
    #Puts Ids of Mnist dataset into sorted indexs
    def sortIndexs(self):
        #Iterate through all data then apend the ID in to the Sorted Array
        for i in range(self.getNumExamples()):
            TempLabel = self.getLabelByIndex(i).argmax()
            self.sortedLabelPositionPairs.append([TempLabel,len(self.sortedIndexs[TempLabel])])
            self.sortedIndexs[TempLabel].append(i)

    # 3 Individual Datasets exist: Train,Test, and Validation
    # Train - 55,000 Examples      | IDs:     0 - 54999
    # Test - 10,000 Examples       | IDs: 55000 - 64999
    # Validation - 5,000 Examples  | IDs: 65000 - 69999
    def getLabelByIndex(self, LabelIndex):
        # Checks if in above or below test set to save on compute time
        if(LabelIndex<mnist.train.num_examples):
            return mnist.train.labels[LabelIndex]
        if(LabelIndex>=(mnist.train.num_examples+mnist.test.num_examples)):
            return mnist.validation.labels[LabelIndex-(mnist.train.num_examples+mnist.test.num_examples)]
        return mnist.test.labels[LabelIndex-mnist.train.num_examples]
    
    #Same IDs as getLabelByIndex
    def getImageByIndex(self, ImageIndex):
        # Checks if in above or below test set to save on compute time
        if(ImageIndex<mnist.train.num_examples):
            return mnist.train.images[ImageIndex]
        if(ImageIndex>=(mnist.train.num_examples+mnist.test.num_examples)):
            return mnist.validation.images[ImageIndex-(mnist.train.num_examples+mnist.test.num_examples)]
        return mnist.test.images[ImageIndex-mnist.train.num_examples]
    
    #Finds Total Number Of Examples
    def getNumExamples(self):
        return mnist.train.num_examples+mnist.test.num_examples+mnist.validation.num_examples

#Assembles the training data into a useble format for Siamese Neural Nets
class dataAssembler:
    #Intitializes dataAssemler Class , Requires Argument setting label start and #
    def __init__(self, startindex, numberExamples):
        #initializes dataset Class to have acess to data, and set up array to hold working data
        self.data = Dataset()
        self.processabledata = self.reshape2Dto1D(self.trim2darray(self.data.getSortedIndexs(),startindex,numberExamples))
        
        #Set up data holders for epoch-based data point generation
        #Current epoch data size is reduced as data is generated
        #Master datapoint is reselected at start then removed from dataset at end 
        self.currentEpochData = self.processabledata[:]
        self.currentEpochMasterDataPointIndex = randint(0,len(self.currentEpochData)-1)
        self.currentEpochDataPosition = 0
        
        #Internal counters for Epoch, processed samples
        self.completedEpochs = 0
        self.num_examples = self.calculateEpochSize()
    
    def num_examples(self):
        return self.num_examples

    #Generates a batch of data points for a siames neural net, returns tuple( img1, img2, label )
    def generateSiameseBatch(self, batchSize):
        imageset1 = []
        imageset2 = []
        labelset = []
        for i in range(batchSize):
            tempDataPoint = self.generateSiameseDataPoint()
            imageset1.append(tempDataPoint[0])
            imageset2.append(tempDataPoint[1])
            labelset.append(tempDataPoint[2])
        return tuple([imageset1,imageset2,labelset])

    #Generates a single datapoint for a siamese neural net, returns array with (img1,img2,label)
    def generateSiameseDataPoint(self):
        #Create temp varibles to reduce calls to arrays
        index1 = self.currentEpochData[self.currentEpochMasterDataPointIndex]
        index2 = self.currentEpochData[self.currentEpochDataPosition]

        #Load images for data point before icrementing
        imageDataOne = self.data.getImageByIndex(index1)
        imageDataTwo = self.data.getImageByIndex(index2)
        
        imagelabels = None
        #Check if images have equivlent labels
        if(self.data.getLabelByIndex(index1)==self.data.getLabelByIndex(index2)).all():
            imagelabels = [0.0,1.0]
        else:
            imagelabels = [1.0,0.0]

        #After getting Datapoints, increment the current epoch data position
        self.currentEpochDataPosition += 1
        if (self.currentEpochDataPosition == len(self.currentEpochData)):
            self.reassignMasterDataPoint()

        #Return the data point
        return [imageDataOne,imageDataTwo,imagelabels]

    #Prepares Array for another sub epoch step, increments epoch varible, resets position, chooses master point
    def reassignMasterDataPoint(self):
        del self.currentEpochData[self.currentEpochMasterDataPointIndex]
        self.currentEpochDataPosition = 0
        if(len(self.currentEpochData)==0):
            self.completedEpochs += 1
            self.currentEpochData = self.processabledata[:]
        self.currentEpochMasterDataPointIndex = randint(0,len(self.currentEpochData)-1)

    #Trims 2d Array's 2nd dimensiion, keeping 1st dimension size where Array[dimension 1][dimension 2]
    #Trims off values with index greater than limiter in second dimension
    #start is index, count is how many
    def trim2darray(self, input2d, start, count):
        for d1 in range(len(input2d)):
            input2d[d1] = input2d[d1][start:(start+count)]
        return input2d

    #Takes in input 2d array and reformats it as a 1d array.
    def reshape2Dto1D(self,input2d):
        output1d = []
        for d1 in range(len(input2d)):
            for d2 in range(len(input2d[d1])):
                output1d.append(input2d[d1][d2])
        return output1d
    
    #Finds Epoch Example ammount
    def calculateEpochSize(self):
        tempint = len(self.processabledata)
        tempint = int((tempint*(tempint+1))/2)
        return tempint

if __name__ == "__main__":
    #Print Settings
    if CreateImageFile:
        print("Making Image File")
    time.sleep(.5)

    #Open or Create Data File
    if CreateImageFile:
        if not os.path.isfile(ImageFileName):
            datafile = open(ImageFileName, "w+")
        else:
            datafile = open(ImageFileName, "a+")
    
        IDCount = 0
        for i in range(mnist.test.num_examples):
            TempString = ""
            TempString = arrayToString(mnist.test.labels[i],",")+","+arrayToString(mnist.test.images[i],",")
            TempString = "["+str(IDCount)+ "]," + TempString
            IDCount += 1
            datafile.write(TempString+'\n')
        datafile.close()

    #End Main Tasks
    print("Done.")
    #Debug
    tempdata = Dataset()
    for d1 in range(10):
        for d2 in range(10):
            print(tempdata.getSortedIndexs()[d1][d2], end='  ')
        print(" ")

    time.sleep(999)