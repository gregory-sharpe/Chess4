import random
import copy



def findRandomMove(validMoves):
    return validMoves[random.randint(0,len(validMoves)-1)]
"""
def simulateMove(gamestate:ChessEngine.GameState,move:ChessEngine.Move):
    copiedgame = copy.deepcopy(gamestate)
    copiedgame.makeMove(move)
    return copiedgame
def greedyFindBestMove(validMoves,gamestate:ChessEngine.GameState):
    player = gamestate.getMovingPlayer()
    currentscore = player.Score
    currentPlayerIndex = gamestate.turnIndex
    largestScoreMove = validMoves[0]
    scoreoflargestScoreMove = 0

    for move in validMoves:
        simulatedGame = simulateMove(gamestate,move)
        simulatedScore = simulatedGame.allPlayers[(currentPlayerIndex)%4].Score
        if simulatedScore > largestScoreMove:
            largestScoreMove = move
            scoreoflargestScoreMove = simulatedScore
    return largestScoreMove
def AlphaBetaPruning(state:ChessEngine.GameState,depth:int)->[ChessEngine.Move]:
    pass



"""