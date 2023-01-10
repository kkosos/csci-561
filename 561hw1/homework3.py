from random import shuffle
from random import random
import math
class City:
    def __init__(self, x, y, z, index):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.index = index


def writeFile(res):
    f = open("output.txt", "w")
    n = len(res)
    for i in range(n):
        city = res[i]
        f.write("{} {} {}\n".format(city.x, city.y, city.z))
    #return to start city
    city = res[0]
    f.write("{} {} {}".format(city.x, city.y, city.z))
    f.close()


def readFile():
    f = open("input.txt")
    numberCity = int(f.readline())
    citys = []
    index = 0
    print(numberCity)
    while index < numberCity:
        coordinate = f.readline().strip().split(" ")
        citys.append(City(coordinate[0], coordinate[1], coordinate[2], index))
        index += 1
    f.close()
    return citys

def distanceOfEucli(a: City, b: City):
    return math.sqrt(((a.x - b.x)**2 + (a.y-b.y)**2 + (a.z - b.z)**2))

# init population depend on size, return a list of different permutaion of city (path)
def initializePopulation(populationSize: int, cities):
    populationList = []
    cityIndex = [i for i in range(len(cities))]
    for i in range(populationSize):
        shuffle(cityIndex)
        path = cityIndex.copy()
        populationList.append(path)    
    return populationList

# return a integer fitness in current path
def evaluateFitness(cityList, cities):
    n = len(cityList)
    totalFitness = 0
    for i in range(n):
        city1 = cityList[i]
        city2 = cityList[i+1] if i + 1 < n else cityList[0]        
        totalFitness += distanceOfEucli(cities[city1], cities[city2])    
    return 1 / totalFitness

# return a list contains fitness of different permutaion
# return (score, routeIndex)
def evaluateAllFitness(populationList, cities):
    n = len(populationList)

    fitnessList = []
    for i in range(n):
        fitnessList.append((evaluateFitness(populationList[i], cities), i))
    return sorted(fitnessList, reverse=True)

#select parent, return path list
def parentSelect(populationList, fitnessList, bestPathSize):
    n = len(fitnessList)
    result = []
    
    totalFitness = sum(score  for score, index in fitnessList)
    #Lower score is better andneed higher probability
    probabilityList = [ score / totalFitness for score, index in fitnessList]
    cumProbability = [0] * n  #cumulative probility list
    cumProbability[0] = probabilityList[0]

    #cumulate probability
    for i in range(1, n):
        cumProbability[i] = probabilityList[i] + cumProbability[i-1]

    #keep best path first    
    lastPathIndex = -1
    for i in range(bestPathSize):
        _, pathIndex = fitnessList[i]
        if pathIndex != lastPathIndex:
            result.append(populationList[pathIndex])
            lastPathIndex = pathIndex
    #fill up vaccum with wheel select
    needSize = n - len(result)
    for i in range(needSize):
        pick = random()
        for i in range(n):
            if pick <= cumProbability[i]:  
                _, pathIndex = fitnessList[i]
                result.append(populationList[pathIndex])
                break
    return result

#rankselect parent, return path list
def parentRankSelect(populationList, fitnessList, bestPathSize):
    n = len(fitnessList)
    result = []
    
    totalFitness = sum(i  for i in range(n))
    #Lower score is better andneed higher probability
    probabilityList = [ (n-i) / totalFitness for i in range(n)]
    cumProbability = [0] * n  #cumulative probility list
    cumProbability[0] = probabilityList[0]

    #cumulate probability
    for i in range(1, n):
        cumProbability[i] = probabilityList[i] + cumProbability[i-1]

    #keep best path first    
    lastPathIndex = -1
    for i in range(bestPathSize):
        _, pathIndex = fitnessList[i]
        if pathIndex != lastPathIndex:
            result.append(populationList[pathIndex])
            lastPathIndex = pathIndex
    #fill up vaccum with wheel select
    needSize = n - len(result)
    for i in range(needSize):
        pick = random()
        for i in range(bestPathSize // 2, n):
            if pick <= cumProbability[i]:  
                _, pathIndex = fitnessList[i]
                result.append(populationList[pathIndex])
                break
    return result


def OXOneCrossOver(parentA, parentB):
    geneA = int(random() * len(parentA))
    geneB = int(random() * len(parentA))

    startIndex = min(geneA, geneB)
    endIndex = max(geneA, geneB)

    partA = parentA[startIndex: endIndex]
    partB = [i for i in parentB if i not in partA]
    return partA + partB


def crossover(parentPool, bestChildPathSize):
    n = len(parentPool)
    children = []

    for i in range(bestChildPathSize):
        children.append(parentPool[i].copy())
    needSize = len(parentPool) - bestChildPathSize
    #shuffle all parent so we can random pickup
    shuffle(parentPool)
    for i in range(needSize):
        child = OXOneCrossOver(parentPool[i], parentPool[n-i-1])
        children.append(child)
    return children

def geneSwapMutate(genes, mutateRate):
    for i in range(len(genes)):
        if random() < mutateRate:
            swapped = int(random() * len(genes))
            genes[swapped], genes[i] = genes[i], genes[swapped]
    return genes

def geneScrambleMutate(genes, mutateRate):
    if random() < mutateRate:         
        first = int(random() * len(genes))
        second = int(random() * len(genes))
        start, end = min(first, second), max(first, second)
        mutateGene = genes[start:end]
        shuffle(mutateGene)
        genes = genes[:start] + mutateGene + genes[end:]
    return genes

def checkMutation(population, mutateRate, bestPathSize, mutateScambleStrategy):

    #Avoid top child mutate
    start = bestPathSize // 2
    end = len(population)
    if mutateScambleStrategy:
        for i in range(start, end):
            geneScrambleMutate(population[i], mutateRate)
    else:
        for i in range(start, end):
            geneSwapMutate(population[i], mutateRate)

def findbestSolution(populationList, cities, curBestScore):
    finalRes = evaluateAllFitness(populationList, cities)

    
    score, bestPathIndex = finalRes[0]
    res = []
    if score > curBestScore:        
        for cityIndex in populationList[bestPathIndex]:
            res.append(cities[cityIndex])
    return (res, max(score, curBestScore))
    
        

def main():
    numIteration = 750
    strategyChange = numIteration // 2
    initPopSize = 200
    bestPathSize = 30 #weighted value and keep best path
    cities = readFile()  
    populationList = initializePopulation(initPopSize, cities)
    mutateRate = 0.05
    mutateScambleStrategy = True
    useSelectRank = True
    curBestScore = float('-inf')
    res = []
    print("start iteration")
    for i in range(numIteration):
        #print("evaluateAllFitness")
        fitnessList = evaluateAllFitness(populationList, cities)
        #print("parentSelect")
        if useSelectRank:
            parentPool = parentRankSelect(populationList, fitnessList, bestPathSize)
        else:
            parentPool = parentSelect(populationList, fitnessList, bestPathSize)
        #print("crossover")
        children = crossover(parentPool, bestPathSize)
        #print("mutation")
        checkMutation(children, mutateRate, bestPathSize, mutateScambleStrategy)
        populationList = children
        solution, score = findbestSolution(populationList, cities, curBestScore)
        if solution:
            res = solution
            curBestScore = score
            print("find best solution " + str(1/score))
        else:
            #try to escape if didn't find better solution
            strategyChange -= 1
            if strategyChange < 0:
                useSelectRank = True
                mutateRate += 0.1
                strategyChange = numIteration // 3
                #bestPathSize = int(bestPathSize * 0.9)
                print("change strategy")
            #mutateRate %= 0.2
            #mutateScambleStrategy = not mutateScambleStrategy
        
        #print("iteration done" +  str(i))

    
    print("finish iteration")
    #for city in res:
    #    print(str(city.x) + " " + str(city.y) + " " + str(city.z) + " " + str(city.index))
    writeFile(res)

if __name__ == '__main__':
    main()
