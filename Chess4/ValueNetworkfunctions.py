#after every move the player must
import statistics
def valueFunctionAllOrNothing(PlayerScores,LargestScore):
    PlayerValues = []
    for playerScore in PlayerScores:
        if playerScore == LargestScore:
            PlayerValues.append(1)
        else:
            PlayerValues.append(-1)
    return PlayerValues
def valueFunctionPercentageOfTopScoreRange1(playerScores,largestScore):
    playerValues = []
    largestScore = 1 if largestScore == 0 else largestScore
    for score in playerScores:
        playerValues.append(score/largestScore)
    return playerValues
#peercce
def valueFunctionPercentageOfTopScoreRange2(playerScores,largestScore):
    # from -1 to 1
    scores = valueFunctionPercentageOfTopScoreRange1(playerScores, largestScore)
    codedPlayerScores =[]
    for score in scores:
        codedPlayerScores.append(2*score-1)
    return codedPlayerScores
def valueFunctionCentered(playerScores):
    mean = statistics.mean(playerScores)
    std = statistics.pstdev(playerScores)
    if std == 0:
        std = 1
    centeredData = []
    for score in playerScores:
        centeredData.append((score-mean)/std)
    return centeredData
def getValue(players,valueFunction= "AllOrNothing"):
    playerScores = []
    for player in players:
        playerScores.append(player.Score)
    largestScore = max(playerScores)
    if valueFunction == "Centered":
        PlayerValues = valueFunctionCentered(playerScores)
    elif valueFunction == "PercentageRange1":
        PlayerValues = valueFunctionPercentageOfTopScoreRange1(playerScores,largestScore)
    elif valueFunction == "PercentageRange2":
        PlayerValues = valueFunctionPercentageOfTopScoreRange2(playerScores,largestScore)
    elif valueFunction == "Raw":
        PlayerValues = playerScores
    else:
        PlayerValues = valueFunctionAllOrNothing(playerScores,largestScore)
    return PlayerValues
def giveBoardAValue():
    pass