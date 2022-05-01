import functools
import random
import copy



def findRandomMove(validMoves):
    return validMoves[random.randint(0,len(validMoves)-1)]

def simulateMove(gamestate,move):
    copiedgame = copy.deepcopy(gamestate)
    copiedgame.makeMove(move)
    return copiedgame
def greedyFindBestMove(validMoves,gamestate):
    player = gamestate.getMovingPlayer()
    currentscore = player.Score
    currentPlayerIndex = gamestate.turnIndex
    largestScoreMove = validMoves[0]
    scoreoflargestScoreMove = 0

    for move in validMoves:
        simulatedGame = simulateMove(gamestate,move)
        simulatedScore = simulatedGame.allPlayers[(currentPlayerIndex)%4].Score
        if simulatedScore > scoreoflargestScoreMove:
            largestScoreMove = move
            scoreoflargestScoreMove = simulatedScore
    print(largestScoreMove)
    print(scoreoflargestScoreMove)
    print()
    return largestScoreMove
def getScoreOfPlayer(player,state):
    return state.allPlayers[state.allPlayers.index(player)].Score
"""
def sortStates(states,player):
    sortfunction = functools.partial(getScoreOfPlayer(player))
    sorted(states,lambda )
    
def maxn(state,depth):
    if state.gameOver or depth ==0:
        return [state]

    player = state.currentPlayer()
    validmoves = state.validMoves
    stateList = []
    for move in validmoves:
        simulatedstate = simulateMove(state,move) # do a move

        stateList.append(simulatedstate)

        # choose the best move from the list of states.





"""