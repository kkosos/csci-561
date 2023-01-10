import random
import copy


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


class GoBoardHelper:
    def __init__(self, size) -> None:
        self.boardSize = size
        return

    def removeStones(self, curBoard, targetStones):
        for stone in targetStones:
            curBoard[stone[0]][stone[1]] = 0
        return curBoard

    def searchAllDeadStones(self, curBoard, player):
        deads = []
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if curBoard[i][j] == player:
                    if not self.groupLibertyCount(curBoard, i, j) and (i, j) not in deads:
                        deads.append((i, j))
        return deads

    def removeDeadStones(self, curBoard, player):
        deads = self.searchAllDeadStones(curBoard, player)
        return self.removeStones(curBoard, deads) if deads else curBoard
         

    def findAllAdjacentStones(self, row, col):
        neibors = [(row + 1, col), (row - 1, col), (row , col+1), (row, col-1)]
        stones = []
        for p in neibors:
            if 0 <= p[0] < self.boardSize and 0 <= p[1] < self.boardSize:
                stones.append(p)
        return stones

    def findAllyNeighbor(self, curBoard, point):
        row, col = point[0], point[1]
        neighbor = []
        adjacentStones = self.findAllAdjacentStones(row, col)
        for r, c in adjacentStones:
            if curBoard[r][c] == curBoard[row][col]:
                neighbor.append((r,c))
        return neighbor          

    def findAllGroup(self, curBoard, row, col):
        group = []
        queue = [(row, col)]
        #use bfs to search ally stone
        while queue:
            point = queue.pop(0)
            group.append(point)
            for neighbor in self.findAllyNeighbor(curBoard, point):
                if not (neighbor in queue or neighbor in group):
                    queue.append(neighbor)
        return group

    def groupLibertyCount(self, curBoard, row, col):
        count = 0
        for p in self.findAllGroup(curBoard, row, col):
            for neighbor in self.findAllAdjacentStones(p[0], p[1]):
                if curBoard[neighbor[0]][neighbor[1]] == 0:
                    count += 1
        return count

    def isKO(self, curBoard, prevBoard):
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if prevBoard[i][j] != curBoard[i][j]:
                    return False
        return True

    def isSuccessMove(self, curBoard, prevBoard, player, row, col):
        if curBoard[row][col] != 0: return False
        copyBoard = copy.deepcopy(curBoard)
        copyBoard[row][col] = player
        deadStone = self.searchAllDeadStones(copyBoard, 3 - player)
        copyBoard = self.removeDeadStones(copyBoard, 3 - player)
        if self.groupLibertyCount(copyBoard, row, col) >= 1 and not (deadStone and self.isKO(copyBoard, prevBoard )):
            return True


    def findAllMoves(self, curBoard, prevBoard, player):
        moves = []
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.isSuccessMove(curBoard, prevBoard, player, i, j):
                    moves.append((i, j))
        return moves


class MinmaxStrategy:
    def __init__(self, size, player) -> None:
        self.boardSize = size
        self.player = player
        self.opponent = 3 - player
        self.boardHelper = GoBoardHelper(size)
        return 

    def heuristicFunction(self, curBoard, player):
        komi = 2.5
        newPlayer, opponenet = 0, 0
        hNewPlayer, hOpponenet = 0, 0
        if self.player == 2:
            newPlayer += komi
        else:
            opponenet += komi
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if curBoard[i][j] == self.player:
                    newPlayer += 1
                    hNewPlayer += newPlayer + self.boardHelper.groupLibertyCount(curBoard, i, j)
                elif curBoard[i][j] == self.opponent:
                    opponenet += 1
                    hOpponenet += opponenet + self.boardHelper.groupLibertyCount(curBoard, i , j)
        return hNewPlayer - hOpponenet if player == self.player else hOpponenet - hNewPlayer


    def findBestMoves(self, curBoard, preBoard,  alpha, beta, depth):
        moves = []
        best = 0
        allMoves = self.boardHelper.findAllMoves(curBoard, preBoard, self.player)
        for move in allMoves:
            nextBoard = copy.deepcopy(curBoard)
            nextBoard[move[0]][move[1]] = self.player
            nextBoard = self.boardHelper.removeDeadStones(nextBoard, self.opponent)
            
            heuristic = self.heuristicFunction(nextBoard, self.opponent)
            evaluation = self.minMaxValue(False, nextBoard, curBoard, alpha, beta, heuristic, depth)

            curScore = -1 * evaluation
            if curScore > best or not moves:
                alpha = best = curScore
                moves = [move]
            elif curScore == best:
                moves.append(move)
        return moves
    
    def minMaxValue(self, isMax, curBoard, preBoard, alpha, beta, heuristic, depth):
        if depth == 0:
            return heuristic
        best = heuristic
        curPlayer = self.player if isMax else self.opponent
        nextPlayer = self.opponent if isMax else self.player

        allMoves = self.boardHelper.findAllMoves(curBoard, preBoard, curPlayer)
        for move in allMoves:
            nextBoard = copy.deepcopy(curBoard)
            nextBoard[move[0]][move[1]] = curPlayer
            nextBoard = self.boardHelper.removeDeadStones(nextBoard, nextPlayer)

            heuristic = self.heuristicFunction(nextBoard, nextPlayer)
            evaluation = self.minMaxValue(not isMax, nextBoard, curBoard, alpha, beta, heuristic, depth - 1)

            curScore = -1 * evaluation
            if curScore > best:
                best = curScore
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


class GoPlayer:
    def __init__(self, strategy) -> None:
        self.strategy = strategy
        return

    def execute(self, board, preBoard, boardSize):

        moves =  self.checkInitStone(board, boardSize)
        if moves:
            return moves
        return self.strategy.findBestMoves(board, preBoard,  -1500, -1500, 2)
        
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
