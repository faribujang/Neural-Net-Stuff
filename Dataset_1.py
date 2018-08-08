from preprocess import features
import numpy as np
import glob
import os

def one_hot_encode(label):
    labels = np.zeros(5)
    labels[label] = 1
    return labels

def get_data():
    """
    Recursively find and return path names of all audio files and their labels
    Return a dictionary in this format
    {file path : label
    ...
    file path : label }
    """
    arr = np.loadtxt("C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\ESC-50-master\\meta\\esc50.csv",dtype=str,delimiter=',',skiprows=1)
    keydict = {}
    for i in range(len(arr)):
        key = 'data/ESC-50-master/audio/'+arr[i][0]
        #print(assign_num_to_label(arr[i][3]))
        val = one_hot_encode(assign_num_to_label(arr[i][3]))
        #print(val)
        keydict.update({key:val})
    i = 0
    return keydict,len(keydict)

# class dataAssembler:
#     #Intitializes dataAssemler Class , Requires Argument setting label start and #
#     def __init__(self, startindex, numberExamples):
#         #initializes dataset Class to have acess to data, and set up array to hold working data
#         self.data = Dataset_1()
#         self.processabledata = self.reshape2Dto1D(self.trim2darray(self.data.getSortedIndexs(),startindex,numberExamples))
        
#         #Set up data holders for epoch-based data point generation
#         #Current epoch data size is reduced as data is generated
#         #Master datapoint is reselected at start then removed from dataset at end 
#         self.currentEpochData = self.processabledata[:]
#         self.currentEpochMasterDataPointIndex = randint(0,len(self.currentEpochData)-1)
#         self.currentEpochDataPosition = 0
        
#         #Internal counters for Epoch, processed samples
#         self.completedEpochs = 0
#         self.num_examples = self.calculateEpochSize()
    
#     def num_examples(self):
#         return self.num_examples

#     #Generates a batch of data points for a siames neural net, returns tuple( img1, img2, label )
#     def generateSiameseBatch(self, batchSize):
#         imageset1 = []
#         imageset2 = []
#         labelset = []
#         for i in range(batchSize):
#             tempDataPoint = self.generateSiameseDataPoint()
#             imageset1.append(tempDataPoint[0])
#             imageset2.append(tempDataPoint[1])
#             labelset.append(tempDataPoint[2])
#         return tuple([imageset1,imageset2,labelset])

#     #Generates a single datapoint for a siamese neural net, returns array with (img1,img2,label)
#     def generateSiameseDataPoint(self):
#         #Create temp varibles to reduce calls to arrays
#         index1 = self.currentEpochData[self.currentEpochMasterDataPointIndex]
#         index2 = self.currentEpochData[self.currentEpochDataPosition]

#         #Load images for data point before icrementing
#         imageDataOne = self.data.getImageByIndex(index1)
#         imageDataTwo = self.data.getImageByIndex(index2)
        
#         imagelabels = None
#         #Check if images have equivlent labels
#         if(self.data.getLabelByIndex(index1)==self.data.getLabelByIndex(index2)).all():
#             imagelabels = [0.0,1.0]
#         else:
#             imagelabels = [1.0,0.0]

#         #After getting Datapoints, increment the current epoch data position
#         self.currentEpochDataPosition += 1
#         if (self.currentEpochDataPosition == len(self.currentEpochData)):
#             self.reassignMasterDataPoint()

#         #Return the data point
#         return [imageDataOne,imageDataTwo,imagelabels]

#     #Prepares Array for another sub epoch step, increments epoch varible, resets position, chooses master point
#     def reassignMasterDataPoint(self):
#         del self.currentEpochData[self.currentEpochMasterDataPointIndex]
#         self.currentEpochDataPosition = 0
#         if(len(self.currentEpochData)==0):
#             self.completedEpochs += 1
#             self.currentEpochData = self.processabledata[:]
#         self.currentEpochMasterDataPointIndex = randint(0,len(self.currentEpochData)-1)

#     #Trims 2d Array's 2nd dimensiion, keeping 1st dimension size where Array[dimension 1][dimension 2]
#     #Trims off values with index greater than limiter in second dimension
#     #start is index, count is how many
#     def trim2darray(self, input2d, start, count):
#         for d1 in range(len(input2d)):
#             input2d[d1] = input2d[d1][start:(start+count)]
#         return input2d

#     #Takes in input 2d array and reformats it as a 1d array.
#     def reshape2Dto1D(self,input2d):
#         output1d = []
#         for d1 in range(len(input2d)):
#             for d2 in range(len(input2d[d1])):
#                 output1d.append(input2d[d1][d2])
#         return output1d
    
#     #Finds Epoch Example ammount
#     def calculateEpochSize(self):
#         tempint = len(self.processabledata)
#         tempint = int((tempint*(tempint+1))/2)
#         return tempint

def assign_num_to_label(label):
    """
    Takes a string label and converts it to a number
    ESC-50 Key
    [0] Door Knock | ESC - 31
    [1] Clock Alarm | ESC - 38
    [2] Siren | ESC - 43, 8k - 7
    [3] Car Horn | ESC - 44, 8k - 1
    [4] Misc
    """
    if label == "door_wood_knock":
        return 0
    if label == "clock_alarm":
        return 1
    if label == "glass_breaking":
        return 2
    if label == "door_wood_knock":
        return 3
    return 4

def next_minibatch(indices,db):
    """
    Return dictionary of next minibatch in this format
    (arr of mfcc) : label
    """
    feat = []
    lab = []
    for i in indices:
        #feat = features(db[i],13,parsePath=True)
        z = db.keys()[i][25:-4]
        lab.append(db[db.keys()[i]])
        if os.path.exists('pickle/'+z+'.npy'):
            print('Preloaded',z)
            feat.append(np.load('pickle/'+z+'.npy'))
        else:
            ftrtmp = np.empty((0,193))
            mfccs, chroma, mel, contrast,tonnetz=features(db.keys()[i])
            ext_mfccs = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ftrtmp = np.vstack([ftrtmp,ext_mfccs])
            np.save('pickle/'+z,ftrtmp[0])
            feat.append(ftrtmp[0])
            #print(np.asarray(feat).shape)
            print('Pickle saved to','pickle/'+z)
    # FEAT OUTPUT 501,26
    return np.asarray(feat),np.asarray(lab)