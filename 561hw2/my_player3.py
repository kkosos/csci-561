import random
from Go import *


class FileHandler():
    def __init__(self) -> None:
        return
    def readInput(self):
        inputFile = "input.txt"
        inputInfo = list()
        with open(inputFile, 'r') as file:
            for line in file.readlines():
                inputInfo.append(line.strip())
        color = int(inputInfo[0])
        prevBoard = [[int(val) for val in row] for row in inputInfo[1:6]]
        curBoard = [[int(val) for val in row] for row in inputInfo[6:]]
        return color, prevBoard, curBoard

    def writeOutput(self, move):
        with open("output.txt", "w") as file:
            if move == "PASS":
                file.write(move)
            else:
                file.write("{},{}".format(move[0], move[1]))

class MinmaxStrategy:
    def __init__(self, size, player) -> None:
        self.boardSize = size
        self.player = player
        self.opponent = 3 - player
        self.boardHelper = GoBoardHelper(size)
        self.maxDepth = 2
        self.reward = [
            [-50, 0, 7, 0, -50],
            [0, 10, 15, 10, 0],
            [7, 15, 40, 15, 7],
            [0, 10, 15, 10, 0],
            [-50, 0, 7, 0, -50]]
        return 

    def getScore(self, curBoard, player):
        komi = 2.5
        newPlayer, opponenet = 0, 0
        hNewPlayer, hOpponenet = 0, 0
        if self.player == 2: #For white compensation
            newPlayer += komi
        else:
            opponenet += komi
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if curBoard[i][j] == self.player:
                    newPlayer += 1
                    hNewPlayer += newPlayer + self.boardHelper.groupLibertyCount(curBoard, i, j) + self.reward[i][j] * 0.1
                elif curBoard[i][j] == self.opponent:
                    opponenet += 1
                    hOpponenet += opponenet + self.boardHelper.groupLibertyCount(curBoard, i , j) + self.reward[i][j] * 0.1
        return hNewPlayer - hOpponenet if player == self.player else hOpponenet - hNewPlayer


    def findBestMoves(self, curBoard, preBoard,  alpha, beta, depth):
        moves = []
        best = 0
        allMoves = self.boardHelper.findAllMoves(curBoard, preBoard, self.player)
        for move in allMoves:
            nextBoard = self.makeNextBoard(curBoard, self.player, move)
            result = self.minMaxValue(False, nextBoard, curBoard, alpha, beta, depth)

            score = -1 * result
            if score > best or not moves:
                alpha = best = score
                moves = [move]
            elif score == best:
                moves.append(move)
        return moves
    
    def minMaxValue(self, isMax, curBoard, preBoard, alpha, beta, depth):
        curPlayer = self.player if isMax else self.opponent

        best = self.getScore(curBoard, curPlayer)
        if depth == self.maxDepth:
            return best

        allMoves = self.boardHelper.findAllMoves(curBoard, preBoard, curPlayer)
        for move in allMoves:
            nextBoard = self.makeNextBoard(curBoard, curPlayer, move)

            result = self.minMaxValue(not isMax, nextBoard, curBoard, alpha, beta, depth + 1)

            best = max(best, -1 * result)
            newScore = -1 * best
            #alpha-beta pruning
            if isMax:
                if newScore < beta:
                    return best
                if best > alpha:
                    alpha = best                
            else:
                if newScore < alpha:
                    return best 
                if best > beta:
                    beta = best
        return best
        
    def makeNextBoard(self, board, player, move):
        nextBoard = copy.deepcopy(board)
        nextBoard[move[0]][move[1]] = player
        return self.boardHelper.removeDeadStones(nextBoard, 3 - player)


class GoPlayer:
    def __init__(self, strategy) -> None:
        self.strategy = strategy
        return

    def execute(self, board, preBoard, boardSize):

        moves =  self.checkInitStone(board, boardSize)
        if moves:
            return moves
        return self.strategy.findBestMoves(board, preBoard,  -1050, -1050, 0)
        
    def checkInitStone(self, curBoard, boardSize):
        counter = 0
        opponentFirst = 0

        for i in range(boardSize):
            for j in range(boardSize):
                if curBoard[i][j] != 0:
                    counter += 1
                    if counter == 1:
                        opponentFirst = (i, j)
        moves = []
        if (counter == 0 and self.strategy.player == 1):
            moves = [(2,2)]
        elif counter == 1 and self.strategy.player == 2:
            if opponentFirst[0] <= 2:
                if opponentFirst[1] <= 2:
                    moves = [(opponentFirst[0]+1, opponentFirst[1]+1)]
                else:
                    moves = [(opponentFirst[0]+1, opponentFirst[1]-1)]
            if opponentFirst[0] > 2:
                if opponentFirst[1] > 2:
                    moves = [(opponentFirst[0]-1, opponentFirst[1]-1)]
                else:
                    moves = [(opponentFirst[0]-1, opponentFirst[1]+1)]
        return moves

        


def main():
    fileHandler = FileHandler()
    color, prevBoard, curBoard = fileHandler.readInput()
    boardSize = 5
    strategy = MinmaxStrategy(boardSize, color)
    player = GoPlayer(strategy)
    possibleMoves = player.execute(curBoard, prevBoard, boardSize)
    fileHandler.writeOutput(random.choice(possibleMoves) if possibleMoves else "PASS")
    return

if __name__ == '__main__':
    main()
