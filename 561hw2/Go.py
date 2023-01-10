import copy
class GoBoardHelper:
    def __init__(self, size) -> None:
        self.boardSize = size
        return

    def removeStones(self, curBoard, targetStones):
        for stone in targetStones:
            curBoard[stone[0]][stone[1]] = 0
        return curBoard


    def isKO(self, curBoard, prevBoard):
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if prevBoard[i][j] != curBoard[i][j]:
                    return False
        return True

    def searchAllDeadStones(self, curBoard, player):
        deads = set()
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if curBoard[i][j] == player:
                    if not self.groupLibertyCount(curBoard, i, j):
                        deads.add((i,j))
        return list(deads)

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
        group = set()
        queue = [(row, col)]
        #use bfs to search ally stone
        while queue:
            point = queue.pop(0)
            group.add(point)
            for neighbor in self.findAllyNeighbor(curBoard, point):
                if neighbor in group:
                    continue
                if neighbor in queue:
                    continue
                queue.append(neighbor)
        return list(group)

    def groupLibertyCount(self, curBoard, row, col):
        count = 0
        for p in self.findAllGroup(curBoard, row, col):
            for neighbor in self.findAllAdjacentStones(p[0], p[1]):
                if curBoard[neighbor[0]][neighbor[1]] == 0:
                    count += 1
        return count


    def isSuccessMove(self, curBoard, prevBoard, player, row, col):
        if curBoard[row][col] != 0: return False
        copyBoard = copy.deepcopy(curBoard)
        copyBoard[row][col] = player
        nextPlayer = 3 - player
        deadStone = self.searchAllDeadStones(copyBoard, nextPlayer)
        copyBoard = self.removeDeadStones(copyBoard, nextPlayer)
        
        if self.groupLibertyCount(copyBoard, row, col) >= 1:
            if not (deadStone and self.isKO(copyBoard, prevBoard)):
                return True


    def findAllMoves(self, curBoard, prevBoard, player):
        moves = []
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.isSuccessMove(curBoard, prevBoard, player, i, j):
                    moves.append((i, j))
        return moves