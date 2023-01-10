import numpy as np
import sys
class MultiLayerPerceptron:


    def __init__(self, batchSize, epoch, nnArchitecture):
        self.batchSize = batchSize
        self.epoch = epoch
        self.network = nnArchitecture
        self.paramValues = {}
    def initLayers(self):
        np.random.seed(99)
        paramValues = {}
        for idx, layer in enumerate(self.network):
            layerIdx = idx + 1
            layerInput = layer["input"]
            layerOutput = layer["output"]
            paramValues["w" + str(layerIdx)] = np.random.randn(layerOutput, layerInput) * 0.1
            paramValues["b" + str(layerIdx)] = np.random.randn(layerOutput, 1) * 0.1
        return paramValues

    def getCostValueByCrossEntroy(self, yOut, y):
        #print("yOut:{} y:{} t:{}".format(yOut.shape, y.shape, np.log(yOut).T.shape))
        m = yOut.shape[1]
        # 𝐿(𝑎,𝑦)=−𝑦(log(𝑎)+(1−𝑦)log(1−𝑎))
        cost = -1 / m * (np.dot(y, np.log(yOut).T) + np.dot(1 - y, np.log(1 - yOut).T))        
        return np.squeeze(cost)

    def getAccuracy(self, yOut, y):
        yOut_ = self.convertProbToClass(yOut)
        return (yOut_ == y).all(axis=0).mean()

    def convertProbToClass(self, probs):
        probs_ = np.copy(probs)
        probs_[probs_> 0.5] = 1
        probs_[probs_<= 0.5] = 0
        return probs_

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidBackward(self, dA, z):
        sig = self.sigmoid(z)
        return dA * sig * (1 - sig)

    def relu(self, z):
        return np.maximum(0, z)

    def reluBackward(self, dA, z):
        dZ = np.array(dA, copy = True)
        #derivative of relu is if z < 0, output is 0. if z > 0, output is 1.
        dZ[z <= 0] = 0
        return dZ

    def getMiniBatches(self, x, y, batchSize):
        #print("Minit batrch")
        #print(x.shape)
        lenOfData = x.shape[0]
        miniBatches = []
        numBatch = lenOfData // batchSize
        #print("Num batch {} s:{} len:{}".format(numBatch, batchSize, lenOfData))
        for i in range(numBatch):
            start = i * batchSize
            end = (i+1) * batchSize
            batchX = x[start: end]
            batchY = y[start: end]
            #print("HHH {}  {}".format(start, end))
            miniBatches.append((batchX, batchY))
        
        if lenOfData % batchSize != 0:
            batchX = x[batchSize * numBatch:]
            batchY = y[batchSize * numBatch:]
            #print("HBBBHH {}  {}".format(batchX.shape, batchY.shape))
            miniBatches.append((batchX, batchY))

        return miniBatches


    def forwardPass(self, x, paramValues, network):
        cache = {}
        aCur = x
        for idx, layer in enumerate(network):
            layerIdx = idx + 1
            aPrev = aCur
            wCur = paramValues["w" + str(layerIdx)]
            bCur = paramValues["b" + str(layerIdx)]
            #print("w:{}  b:{}, a:{}".format(wCur.shape,bCur.shape, aPrev.shape))
            #z = w*aP + b
            #aN = sigmoid(z)
            zCur = np.dot(wCur, aPrev) + bCur            
            if layer["activate"] == "relu":
                aCur = self.relu(zCur)
            else:
                aCur = self.sigmoid(zCur)
        
            #print("a:{} z:{}".format(aCur.shape, zCur.shape))


            cache["a" + str(idx)] = aPrev
            cache["z" + str(layerIdx)] = zCur
        return cache, aCur

    #z = wa+ b -> a = sigmoid(z) -> Loss(a, aOut)
    def singleLayerBackPass(self, dACur, wCur, zCur, aPrev, activateFunction):
        m = aPrev.shape[1]
        #Follow the formulation
        if activateFunction == "relu":
            dZCur = self.reluBackward(dACur, zCur)
        else:
            dZCur = self.sigmoidBackward(dACur, zCur)
        dWCur = np.dot(dZCur, aPrev.T) / m
        dBCur = np.sum(dZCur, axis= 1, keepdims=True) / m
        daPrev = np.dot(wCur.T, dZCur)
        return daPrev, dWCur, dBCur

    def backPropagation(self, yOut, y, cache, paramValues, network):
        y = y.reshape(yOut.shape)
        gradienValues = {}
        #Initial by error function. dLoss/dyOut
        dAPrev = - (np.divide(y, yOut) - np.divide(1-y, 1-yOut))
        for layerIdxPrev, layer in reversed(list(enumerate(network))):
            #4>3>2>1>0
            layerIdxCur = layerIdxPrev + 1
            
            dACur = dAPrev

            aPrev = cache["a" + str(layerIdxPrev)]
            zCur = cache["z" + str(layerIdxCur)]

            wCur = paramValues["w" + str(layerIdxCur)]

            dAPrev, dWCur, dBCur = self.singleLayerBackPass(dACur, wCur, zCur, aPrev, layer["activate"])
            gradienValues["dW" + str(layerIdxCur)] = dWCur
            gradienValues["db" + str(layerIdxCur)] = dBCur
        return gradienValues

    def updateParameter(self, paramValue, gradienValue, network, learingRate):
        for idx, _ in enumerate(network):
            layerIdx = idx + 1
            paramValue["w" + str(layerIdx)] -= learingRate * gradienValue["dW" + str(layerIdx)]
            paramValue["b" + str(layerIdx)] -= learingRate * gradienValue["db" + str(layerIdx)]
        return paramValue

            
    def train(self, data, label, learingRate):
        paramValues = self.initLayers()
        costHistory = []
        accuracyHistory = []
        for i in range(self.epoch):
            miniBatches = self.getMiniBatches(data, label, self.batchSize)

            for miniBatch in miniBatches:
                x, y = np.transpose(miniBatch[0]), np.transpose(miniBatch[1])
                #print("Batch comp x:{} y:{}".format(x.shape, y.shape))
                cache, yOut = self.forwardPass(x, paramValues, self.network)
                cost = self.getCostValueByCrossEntroy(yOut, y)
                accuracy = self.getAccuracy(yOut, y)
                costHistory.append(cost)
                accuracyHistory.append(accuracy)
                gradienValues = self.backPropagation(yOut, y, cache, paramValues, self.network)
                paramValues = self.updateParameter(paramValues, gradienValues, self.network, learingRate)
            
            if i % 50 == 0:
               print("Iteration: {} - cost: {} - accuracy: {}".format(i, cost, accuracy))

        #only keep last
        self.paramValues = paramValues
        #print(math.avg(accuracyHistory))

    def predict(self, x):
        x = np.transpose(x)
        cache, yOut = self.forwardPass(x, self.paramValues, self.network)
        output = self.convertProbToClass(yOut)
        return output
    def predictTest(self, x, y):
        x, y = np.transpose(x), np.transpose(y)
        cache, yOut = self.forwardPass(x, self.paramValues, self.network)
        print(yOut)
        output = self.convertProbToClass(yOut)
        accuracy = self.getAccuracy(yOut, y)
        print("Final acc: {}".format(accuracy))
        print(output)
        return output





class FileUtil:
    def __init__(self):
        return
    def loadData(self, file):        
        return np.loadtxt(file, delimiter=',')
    def loadLabel(self, file):        
        return np.loadtxt(file)
    def writeData(self, out):
        np.savetxt("test_predictions.csv", out, fmt="%d", newline='\n')

        
fileUtil = FileUtil()
trainData = fileUtil.loadData(sys.argv[1])
trainLabel = fileUtil.loadLabel(sys.argv[2])
testData = fileUtil.loadData(sys.argv[3])
#testLabel = fileUtil.loadLabel(sys.argv[4])

networkArchitecture = [
    {"input":2, "output":32, "activate":"relu"},
    {"input":32, "output":16, "activate":"relu"},
    {"input":16, "output":8, "activate":"relu"},
    {"input":8, "output":1, "activate":"sigmoid"}]
network = MultiLayerPerceptron(batchSize=32, epoch=1000, nnArchitecture=networkArchitecture)
network.train(trainData,trainLabel.reshape(trainLabel.shape[0], 1), 0.001)
#out = network.predict(testData, testLabel.reshape(testLabel.shape[0], 1))
out = network.predict(testData)

#Make it from 1* n to n*1 so output will correct
fileUtil.writeData(np.transpose(out))