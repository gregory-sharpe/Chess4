import numpy as np
# teams
import ChessMain

# bitmap for each colour 4
# bitmap each piece + premotedQueen 7
# bitmap for NullSquare
#12 all together
import ValueNetworkfunctions
#4 channels for team,6 channels for piece , 2 channels for empty square and null square
RED = 10
#RED=0000 0000 0001
BLUE = 20
#BLUE=0000 0000 0010
YELLOW = 30
#YELLOW=0000 0000 0100
GREEN = 40
#GREEN=0000 0000 1000
INACTIVEPIECE = 50
# pieces - an inactive piece will have the value - PIECE ie -1 for an inactive pawn
PAWN = 1
#PAWN=0000 0001 0000
ROOK = 2
#ROOK=0000 0010 0000
KNIGHT = 3

BISHOP = 4
QUEEN = 5
KING = 6
# by making the values as small as possible it reduces memory useage
# other square classifications
NULLSQUARE = 7  # the black squares no piece can move to
EMPTYSQUARE = 0
POSSIBLEMOVES = 14*14*(8*13+8) # starting square *(queen directions * magnitude + knightDirections)

REDTURN = 100
REDKINGCASTLE = 101
REDQUEENCASTLE = 102
REDSCORE = 103
REDACTIVE = 104

BLUETURN = 200
BLUEKINGCASTLE = 201
BLUEQUEENCASTLE = 202
BLUESCORE = 203
BLUEACTIVE = 204

YELLOWTURN = 300
YELLOWKINGCASTLE = 301
YELLOWQUEENCASTLE = 302
YELLOWSCORE = 303
YELLOWACTIVE = 304

GREENTURN = 400
GREENKINGCASTLE = 401
GREENQUEENCASTLE = 402
GREENSCORE = 403
GREENACTIVE = 404

FIFTYMOVEREPEAT = 50

channels = {
    # TODO instead of using the teams as channels directly could use the current playeer , player in 1 move , player in 2 moves etc
    RED: 0,
    BLUE: 1,
    YELLOW: 2,
    GREEN: 3,

    PAWN: 4,
    KNIGHT: 5,
    ROOK: 6,
    BISHOP: 7,
    QUEEN: 8,
    KING: 9,

    NULLSQUARE:10,

    REDTURN:11,
    BLUETURN:12,
    YELLOWTURN:13,
    GREENTURN:14,

    REDACTIVE:15,
    REDSCORE:16,
    REDKINGCASTLE:17,
    REDQUEENCASTLE:18,

    BLUEACTIVE:19,
    BLUESCORE:20,
    BLUEKINGCASTLE:21,
    BLUEQUEENCASTLE:22,

    YELLOWACTIVE:23,
    YELLOWSCORE:24,
    YELLOWKINGCASTLE:25,
    YELLOWQUEENCASTLE:26,
    GREENACTIVE: 27,
    GREENSCORE: 28,
    GREENKINGCASTLE: 29,
    GREENQUEENCASTLE: 30,

    FIFTYMOVEREPEAT:31
    # need a channel for premoted pawns
    # need a channel for whos turn it is to move 
}

"""
channels = {
    RED: 0,
    BLUE: 1,
    YELLOW: 2,
    GREEN: 3
}
"""

TEAMS = [RED, BLUE, YELLOW, GREEN]
TEAMSs = ["r", "b", "y", "g"]
PIECES = [PAWN, ROOK, BISHOP, QUEEN, KING, KNIGHT]
PIECESs = ["p", "R", "B", "Q", "K", "N"]

RedKingPosition = (13,7)
BlueKingPosition= (6,0)
YellowKingPosition = (0,6)
GreenKingPosition = (7,13)


class Player():

    def __init__(self, Team, Location):
        self.team = Team
        self.KingLocation = Location
        self.pins = []
        self.checks = []
        self.canCastleQueenSide = True
        self.canCastleKingSide =True
        self.Score = 0
        self.isHumanPlaying = False
        self.isMcts = False
        self.isBestMcts = False
        self.playing = True
    def __str__(self):
        if self.team == RED:
            return "REDKING"
        elif self.team == BLUE:
            return "BLUEKING"
        elif self.team == GREEN:
            return "GREENKING"
        elif self.team == YELLOW:
            return "YELLOWKING"
    inCheck = False
    Pins = []
    checks = []
class GameState():
    def __init__(self):
        self.fakeGame = False
        self.board = np.zeros((14, 14))
        self.height = 14
        self.width = 14
        self.boards= []
        self.gameOver = False
        self.setBoard()
        self.turnIndex = 0
        self.turn = RED
        self.validMoves = []
        self.moveLog = []
        self.currentPlayer = ()
        self.enPassentSquares = []
        self.RedPlayer = Player(RED, RedKingPosition)
        self.RedPlayer.isHumanPlaying = False
        self.BluePlayer = Player(BLUE, BlueKingPosition)
        self.YellowPlayer = Player(YELLOW, YellowKingPosition)
        self.GreenPlayer = Player(GREEN, GreenKingPosition)
        self.allPlayers = [self.RedPlayer,self.BluePlayer,self.YellowPlayer,self.GreenPlayer]
        self.getValidMoves()
        self.fiftyRuleRepition = 0
        self.movesMade = 7
        self.MaxMoveLimit = 10
        self.gameOutcome = ()
        self.currentState = np.zeros((len(channels), 14, 14))
        self.update_current_state()
        self.availables= self.validMoves
    def finishGame(self):
        self.gameOver = True
        players = self.allPlayers
        print("Raw Scores:" + str(ValueNetworkfunctions.getValue(players,valueFunction="Raw")))
        print("Centered:" + str(ValueNetworkfunctions.getValue(players,valueFunction="Centered")))
        print("Percentage Range 1:"+ str(ValueNetworkfunctions.getValue(players,valueFunction="PercentageRange1")))
        print("Percentage Range 2" + str(ValueNetworkfunctions.getValue(players, valueFunction="PercentageRange2")))
        self.gameOutcome = ValueNetworkfunctions.getValue(players)
        boardStatesWithPlayerWinScore = []
        for boardState in self.boards:
            boardStatesWithPlayerWinScore.append((boardState,self.gameOutcome))
        print(self.gameOutcome)
    def getGameOutcome(self):
        return ValueNetworkfunctions.getValue(self.allPlayers)
    def update_current_state(self):

        #0,1,2,3 = RED,Blue,Yellow,Green
        #4,5,6,7,8,9 = Pawn ,Knight ,Rook ,Bishop,Queen,King
        #10 = NullSquare
        # inactive players will have a -1 in their team channel
        # update the turn channels
        ##BLUETURN = 200
        ##BLUEKINGCASTLE = 201
        ##BLUEQUEENCASTLE = 202
        ##BLUESCORE = 203
        ##BLUEACTIVE = 204
        self.currentState[channels[FIFTYMOVEREPEAT]] = self.fiftyRuleRepition  # TURN

        for team in TEAMS:
            player = self.getPlayerFromTeam(team)
            if team == self.turn:
                self.currentState[channels[team*10]] = 1 # TURN
            else:
                self.currentState[channels[team * 10]] = 0

            self.currentState[channels[team * 10 + 1]] = player.canCastleKingSide
            self.currentState[channels[team * 10 + 2]] = player.canCastleQueenSide
            self.currentState[channels[team * 10 + 3]] = player.Score
            self.currentState[channels[team * 10 + 4]] = player.playing
        for r in range(14):
            for c in range(14):
                self.update_currentStateRC(r,c)
        return self.board
    def update_currentStateRC(self,r,c):
        piece = self.board[r][c]
        #for now all the values will be set to 0 before changing individually. when there are constant values like can caslte only the first k must be changed
        for channel in range(len(channels)):
            self.currentState[channel][r][c]=0
        if piece != EMPTYSQUARE:
            if piece == NULLSQUARE:
                self.currentState[channels[NULLSQUARE]][r][c] = 1
            else:
                team = self.pieceTeamFromNumber(piece)
                type = self.pieceTypeFromNumber(piece)
                self.currentState[channels[team]][r][c] = 1
                self.currentState[channels[type]][r][c] = 1
    def update_currentStateRC4FeatureBoards(self,r,c):
        piece = self.board[r][c]
        if piece != EMPTYSQUARE and piece!= NULLSQUARE:
            team = self.pieceTeamFromNumber(piece)
            if self.getPlayerFromTeam(team).playing:
                self.currentState[channels[team]][r][c] = 1
    def getBoardStateInBits(self):
        # bit boards for type. 8
        # bit boards for inactive pieces.2
        # bit boards for teams.4
        # bit boards for
        pass
    def getMovingPlayer(self):
        if self.turn == RED:
            return self.RedPlayer
        elif self.turn == BLUE:
            return self.BluePlayer
        elif self.turn == GREEN:
            return self.GreenPlayer
        elif self.turn == YELLOW:
            return self.YellowPlayer
    def getPlayerFromTeam(self,team):
        if team == RED:
            return self.RedPlayer
        elif team == BLUE:
            return self.BluePlayer
        elif team == GREEN:
            return self.GreenPlayer
        elif team == YELLOW:
            return self.YellowPlayer
    def setBoard(self):
        startingFormation = np.array([[0, 0, 0, PAWN, PAWN, PAWN, PAWN, PAWN, PAWN, PAWN, PAWN, 0, 0, 0],
                                      [0, 0, 0, ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK, 0, 0, 0]])
        rStartingFormation = startingFormation + RED * np.ones((2, 14))
        rStartingFormation = np.pad(rStartingFormation, ((12, 0), (0, 0)), mode='constant', constant_values=0)

        bStartingFormation = np.transpose(startingFormation) + BLUE * np.ones((14, 2))
        bStartingFormation = np.fliplr(bStartingFormation)
        bStartingFormation = np.flipud(bStartingFormation)
        bStartingFormation = np.pad(bStartingFormation, ((0, 0), (0, 12)), mode='constant', constant_values=0)

        yStartingFormation = np.flipud(startingFormation) + YELLOW * np.ones((2, 14))
        yStartingFormation = np.fliplr(yStartingFormation)
        yStartingFormation = np.pad(yStartingFormation, ((0, 12), (0, 0)), mode='constant', constant_values=0)

        gStartingFormation = np.transpose(startingFormation) + GREEN * np.ones((14, 2))
        gStartingFormation = np.pad(gStartingFormation, ((0, 0), (12, 0)), mode='constant', constant_values=0)
        startingFormation = rStartingFormation + bStartingFormation + yStartingFormation + gStartingFormation
        ## adding void squares
        for xOffSet in [0, 11]:
            for yOffSet in [0, 11]:
                for sqx in range(3):
                    for sqy in range(3):
                        startingFormation[xOffSet + sqx, yOffSet + sqy] = NULLSQUARE
        self.validMoves = []
        #self.getValidMoves()
        self.board = startingFormation

    def availables(self):
        return self.validMoves
    def squareInCheck(self,r,c):
        team = self.turn
        directions = ((-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1))
        knightMoves = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2))
        inCheck = False
        for d in directions:
            possiblePin = ()
            for i in range(1, 13):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if self.inBounds(endRow, endCol):
                    endpiece = self.board[endRow][endCol]
                    if endpiece == NULLSQUARE:
                        break
                    elif endpiece != EMPTYSQUARE and self.pieceTeamFromNumber(endpiece) != team:
                        type = self.pieceTypeFromNumber(endpiece)
                        rookDirections = ((1, 0), (-1, 0), (0, 1), (0, -1))
                        bishopDirections = ((1, 1), (1, -1), (-1, 1), (-1, -1))
                        enemyTeam = self.pieceTeamFromNumber(endpiece)
                        if self.isTeamInactive(enemyTeam):
                            break
                        if ((rookDirections.__contains__(d) and type == ROOK) or
                                (bishopDirections.__contains__(d) and type == BISHOP) or
                                (type == QUEEN) or (type == PAWN and i == 1 and (
                                        enemyTeam == YELLOW and ((-1, -1), (-1, 1)).__contains__(d) or
                                        enemyTeam == RED and ((1, -1), (1, 1)).__contains__(d) or
                                        enemyTeam == GREEN and ((-1, 1), (1, 1)).__contains__(d) or
                                        enemyTeam == BLUE and ((-1, -1), (1, -1)).__contains__(d)
                                        #
                                )
                                ) or (i == 1 and type == KING)):
                            inCheck = True

                        else:

                            # print("being attacked by an enemy piece that cant put the king in check")
                            break
                else:
                    break
        for m in knightMoves:
            endRow = r + m[0]
            endCol = c + m[1]
            if self.inBounds(endRow,endCol):
                endpiece = self.board[endRow][endCol]
                if (self.pieceTeamFromNumber(endpiece)!= team and self.pieceTypeFromNumber(endpiece)== KNIGHT):
                    if self.isTeamInactive(self.pieceTeamFromNumber(endpiece)):
                        continue
                    inCheck = True

        return inCheck

    def checkForPinsAndChecks(self,movingPlayer : Player):
        directions = ((-1,-1) , (0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1))
        movingPlayer.pins=[]
        movingPlayer.checks =[]
        movingPlayer.inCheck = False
        startRow = movingPlayer.KingLocation[0]
        startCol = movingPlayer.KingLocation[1]
        team = movingPlayer.team
        for d in directions:
            possiblePin = ()
            for i in range(1,13):
                endRow = startRow + d[0] * i
                endCol = startCol + d[1] * i
                if self.inBounds(endRow,endCol):
                    endpiece = self.board[endRow][endCol]
                    if endpiece == NULLSQUARE:
                        break;
                    if endpiece!= EMPTYSQUARE and self.pieceTeamFromNumber(endpiece) == team and self.pieceTypeFromNumber(endpiece)!=KING:
                        if possiblePin ==():
                            possiblePin = (endRow,endCol,d[0],d[1])
                        else:
                            break
                    elif endpiece !=EMPTYSQUARE and self.pieceTeamFromNumber(endpiece) != team:
                        type = self.pieceTypeFromNumber(endpiece)
                        rookDirections = ((1, 0), (-1, 0), (0, 1), (0, -1))
                        bishopDirections = ((1, 1), (1, -1), (-1, 1), (-1, -1))
                        enemyTeam = self.pieceTeamFromNumber(endpiece)
                        if self.isTeamInactive(enemyTeam):
                            break
                        if ((rookDirections.__contains__(d) and type == ROOK) or
                                (bishopDirections.__contains__(d) and type == BISHOP) or
                                (type== QUEEN) or (type==PAWN and i== 1 and (
                                        enemyTeam ==YELLOW and ((-1,-1),(-1,1)).__contains__(d) or
                                        enemyTeam ==RED and ((1,-1),(1,1)).__contains__(d) or
                                        enemyTeam == GREEN and ((-1, 1),( 1, 1)).__contains__(d) or
                                        enemyTeam == BLUE and ((-1, -1), (1, -1)).__contains__(d)
#
                                  )
                                ) or (i == 1 and type==KING)):
                            if possiblePin == ():
                                movingPlayer.inCheck = True
                                movingPlayer.checks.append((endRow,endCol,d))


                                break
                            else:
                                movingPlayer.pins.append(possiblePin)
                                break
                        else:

                            #print("being attacked by an enemy piece that cant put the king in check")
                            break
                else:
                    break
        knightMoves = ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(-1,2),(1,-2),(-1,-2))
        for m in knightMoves:
            endRow = startRow + m[0]
            endCol = startCol + m[1]
            if self.inBounds(endRow,endCol):
                endpiece = self.board[endRow][endCol]
                if (self.pieceTeamFromNumber(endpiece)!= team and self.pieceTypeFromNumber(endpiece)== KNIGHT):
                    if self.isTeamInactive(self.pieceTeamFromNumber(endpiece)):
                        continue
                    movingPlayer.inCheck = True
                    movingPlayer.checks.append((endRow,endCol,m))
        return movingPlayer.inCheck,movingPlayer.pins,movingPlayer.checks
    @staticmethod
    def pieceNameFromNumber(PieceAsNumber):
        PieceAsNumberabs = abs(PieceAsNumber)
        Team = TEAMSs[TEAMS.index((PieceAsNumberabs // 10) * 10)]
        Piece = PIECESs[PIECES.index(PieceAsNumberabs % 10)]
        return Team + Piece
    @staticmethod
    def pieceTeamFromNumber(PieceAsNumber):
        PieceAsNumberabs = abs(PieceAsNumber)
        return (PieceAsNumberabs // 10) * 10
    @staticmethod
    def DeactivatePieceFromNumber(PieceAsNumber):
        PieceAsNumberabs = abs(PieceAsNumber)
        type = PieceAsNumberabs%10
        return type + INACTIVEPIECE
    @staticmethod
    def pieceTypeFromNumber(PieceAsNumber):
        PieceAsNumberabs = abs(PieceAsNumber)
        return (PieceAsNumberabs % 10)
    def do_move(self,move):
        self.makeMove(move)
    def makeMove(self, move):

        startsq = self.board[move.startRow][move.starCol]
        endSq = (move.endRow , move.endCol)
        player = self.getMovingPlayer()
        piece = self.board[move.endRow][move.endCol]
        # conditions for 50 repition rule
        self.movesMade +=1
        if self.pieceTypeFromNumber(startsq) == PAWN:
            self.fiftyRuleRepition = 0
        elif piece != NULLSQUARE and piece!= EMPTYSQUARE:
            self.fiftyRuleRepition = 0
        else:
            self.fiftyRuleRepition +=1
        player.Score += self.getScore(piece)
        self.board[move.startRow][move.starCol] = EMPTYSQUARE
        pieceAttacked = self.board[move.endRow][move.endCol]
        self.update_currentStateRC(move.startRow, move.starCol)

        if self.pieceTypeFromNumber(pieceAttacked) == KING:
            enemyteam = self.pieceTeamFromNumber(pieceAttacked)
            currentPlayer = self.getMovingPlayer()
            currentPlayer.Score+=20
            self.removePlayer(self.getPlayerFromTeam(enemyteam),wasKingCapturedOnAnotherTurn=True)

        self.board[move.endRow][move.endCol] = move.pieceMoved
        self.update_currentStateRC(move.endRow, move.endCol)
        self.moveLog.append(move)  # for effiecency use less than the whole board
        self.UpdateKingLocation(endSq,move.pieceMoved)
        if (move.didKingMove):
            player.canCastleKingSide = False
            player.canCastleQueenSide= False
        elif(move.didQueenSideRookMove):
            player.canCastleQueenSide = False
        elif(move.didQueenSideRookMove):
            player.canCastleKingSide = False
        if (move.wasPawnAdvance2):
            self.enPassentSquares.append((move.PawnAdvance1,move.pieceTeam,endSq))

        if (move.wasEnpassent):
            piece = self.board[move.enPassentPawn[0]][move.enPassentPawn[1]]
            player.Score += self.getScore(piece)
            self.board[move.enPassentPawn[0]][move.enPassentPawn[1]] = EMPTYSQUARE
            self.update_currentStateRC(move.enPassentPawn[0], move.enPassentPawn[1])
        if (move.isCastle):
            movingRook = self.board[move.startingRookPosition[0]][move.startingRookPosition[1]]
            self.board[move.startingRookPosition[0]][move.startingRookPosition[1]] = EMPTYSQUARE
            self.board[move.finalRookPosition[0]][move.finalRookPosition[1]] = movingRook
            self.update_currentStateRC(move.finalRookPosition[0], move.finalRookPosition[1])
            self.update_currentStateRC(move.startingRookPosition[0], move.startingRookPosition[1])
        if (move.pawnPremoted):
            print("Pawn Premoted")
            self.premotePawn(move.endRow, move.endCol)
            self.update_currentStateRC(move.endRow, move.endCol)
        self.finishTurn()
    def isTeamInactive(self,team):
        return not self.getPlayerFromTeam(team).playing
    def UpdateKingLocation(self,EndSquare, EndPiece):
        Type = self.pieceTypeFromNumber(EndPiece)
        team = self.pieceTeamFromNumber(EndPiece)
        if Type == KING :
            if team == RED:
                self.RedPlayer.KingLocation = EndSquare
            elif team == BLUE:
                self.BluePlayer.KingLocation = EndSquare
            elif team == YELLOW:
                self.YellowPlayer.KingLocation = EndSquare
            elif team == GREEN:
                self.GreenPlayer.KingLocation = EndSquare
    def undoMove(self):
        if len(self.moveLog) != 0:
            move = self.moveLog.pop()
            startSq = (move.startRow, move.starCol)
            self.board[move.startRow][move.starCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.turnIndex = (self.turnIndex - 1) % 4
            self.turn = TEAMS[self.turnIndex]
            self.validMoves = []
            self.getValidMoves()
            self.getMovingPlayer()
            self.UpdateKingLocation(startSq,move.pieceMoved)
    # for now will go through the entire board and find the location of each piece.
    # check for out of bounds error
    def rPawnMoves(self, r, c):
        if r - 1 < 0 or c < 0 or r > 13 or c > 13:
            return False, (0, 0)
        return True, (r - 1, c)
    def bPawnMoves(self, r, c):
        if r < 0 or c < 0 or r > 13 or c + 1 > 13:
            return False, (0, 0)
        return True, (r, c + 1)
    def yPawnMoves(self, r, c):
        if r < 0 or c < 0 or r + 1 > 13 or c > 13:
            return False, (0, 0)
        return True, (r + 1, c)
    def gPawMoves(self, r, c):
        if r < 0 or c - 1 < 0 or r > 13 or c > 13:
            return False, (0, 0)
        return True, (r, c - 1)
    def colourDependentPawnAdvance(self, r, c, turn):
        advance1 = ()
        Success1 = False
        if turn == RED:
            Success1, advance1 = self.rPawnMoves(r, c)
        elif turn == BLUE:
            Success1, advance1 = self.bPawnMoves(r, c)
        elif turn == YELLOW:
            Success1, advance1 = self.yPawnMoves(r, c)
        elif turn == GREEN:
            Success1, advance1 = self.gPawMoves(r, c)
        return Success1, advance1
    def colourDependentPawnCaputre(self, r, c, ):
        team = self.pieceTeamFromNumber(self.board[r][c])
        capture1 = ()
        capture2 = ()
        Success1 = False
        Success2 = False
        if team == RED:
            Success1, capture1 = self.rPawnMoves(r, c + 1) # right
            Success2, capture2 = self.rPawnMoves(r, c - 1) # left
        elif team == BLUE:
            Success1, capture1 = self.bPawnMoves(r - 1, c) #left
            Success2, capture2 = self.bPawnMoves(r + 1, c)  # right
        elif team == YELLOW:
            Success1, capture1 = self.yPawnMoves(r, c + 1) #left
            Success2, capture2 = self.yPawnMoves(r, c - 1) # right
        elif team == GREEN:
            Success1, capture1 = self.gPawMoves(r - 1, c)#right
            Success2, capture2 = self.gPawMoves(r + 1, c)#left
        return Success1, capture1, Success2, capture2

    def canCapture(self, capture,IsPawn = False):

        rCapture = capture[0]
        cCapture = capture[1]
        piece = self.board[rCapture][cCapture]

        if piece != EMPTYSQUARE and piece != NULLSQUARE:
            if self.turn != self.pieceTeamFromNumber(piece):
                return True
        if IsPawn == True:
            # check if it is a enpassent square
            for enPassentSquare in self.enPassentSquares:
                #print("Checking if can take enPassent")
                if enPassentSquare[0] == capture and self.turn!= self.pieceTeamFromNumber(piece):
                    print("Can take EnPassent")
                    # found a square that can be taken en passent
        return False
    def pawnMoves(self, r, c):
        piecePinned = False
        pinDirection = ()
        movingPlayer = self.getMovingPlayer()
        for i in range(len(movingPlayer.pins)-1,-1,-1):
            if movingPlayer.pins[i][0] == r and movingPlayer.pins[i][1] == c:
                piecePinned = True
                pinDirection = (movingPlayer.pins[i][2],movingPlayer.pins[i][3])
                movingPlayer.pins.remove(movingPlayer.pins[i])
                break
        canAdvanceInBounds, advance1 = self.colourDependentPawnAdvance(r, c, self.turn)

        if canAdvanceInBounds:
            if self.board[advance1[0]][advance1[1]] == EMPTYSQUARE:
                if (self.turn == RED and (not piecePinned or pinDirection == (-1,0)) or
                    self.turn == YELLOW and (not piecePinned or pinDirection ==(1,0)) or
                    self.turn == BLUE and (not piecePinned or pinDirection == (0, 1)) or
                    self.turn == GREEN and (not piecePinned or pinDirection == (0, -1))
                ):
                    _move = Move((r, c), advance1, self.board)
                    if (self.didPawnPremote(advance1)):
                        _move.pawnPremoted = True
                    self.validMoves.append(_move)

                if (self.turn == RED and r == 12) or (self.turn == BLUE and c == 1) or (
                        self.turn == YELLOW and r == 1) or (self.turn == GREEN and c == 12):
                    canAdvanceInBounds, advance2 = self.colourDependentPawnAdvance(advance1[0], advance1[1], self.turn)

                    if (canAdvanceInBounds and self.board[advance2[0]][advance2[1]] == EMPTYSQUARE):
                        if (self.turn == RED and (not piecePinned or pinDirection == (-1, 0)) or
                                self.turn == YELLOW and (not piecePinned or pinDirection == (1, 0)) or
                                self.turn == BLUE and (not piecePinned or pinDirection == (0, 1)) or
                                self.turn == GREEN and (not piecePinned or pinDirection == (0, -1))
                        ):
                            #wasPawnAdvance2 = False,PawnAdvance1 = (0,0),pieceTeam = RED
                            self.validMoves.append(Move((r, c), advance2, self.board,wasPawnAdvance2=True,pieceTeam=self.turn,PawnAdvance1=advance1))
        # capture
        canCaptureLeft, captureLeftsq, cancaptureRight, captureRightSq = self.colourDependentPawnCaputre(r, c)
        leftcaptureDirection = (captureLeftsq[0]-r,captureLeftsq[1]-c)
        rightcaptureDirection = (captureRightSq[0]-r,captureRightSq[1]-c)
        if canCaptureLeft and self.canCapture(captureLeftsq) and (not piecePinned or leftcaptureDirection == pinDirection):
            _move = Move((r, c), captureLeftsq, self.board)
            if (self.didPawnPremote(captureLeftsq)):
                _move.pawnPremoted = True
            self.validMoves.append(_move)
        if cancaptureRight and self.canCapture(captureRightSq) and (not piecePinned or rightcaptureDirection == pinDirection):
            _move = Move((r, c), captureRightSq, self.board)
            if (self.didPawnPremote(captureRightSq)):
                _move.pawnPremoted = True
            self.validMoves.append(_move)
            # enPassent
        for enPassentSquare in self.enPassentSquares:
            #print(enPassentSquare[0])

            if enPassentSquare[0] == captureLeftsq and (not piecePinned or leftcaptureDirection == pinDirection) \
                or enPassentSquare[0] == captureRightSq and (not piecePinned or rightcaptureDirection == pinDirection):
                piece = self.board[enPassentSquare[2][0]][enPassentSquare[2][1]]
                if self.canCapture(enPassentSquare[2]) and self.pieceTeamFromNumber(piece)== enPassentSquare[1]: # if pawn can capture the piece above
                    _move = Move((r,c),enPassentSquare[0],wasEnpassent=True,enPassentPawn=enPassentSquare[2],board=self.board)
                    self.validMoves.append(_move)
                    #wasEnpassent = False,enPassentPawn = (0,0)
                    #wasPawnAdvance2 = False, PawnAdvance1 = (0, 0), pieceTeam = RED
    def inBounds(self, r, c):
        if 0<= r <14 and 0<=c <14:
            return True
        return False
    def recursiveMoves(self,r,c,directions,isRookMove = False):
        piecePinned = False
        pinDirection = ()
        movingPlayer = self.getMovingPlayer()
        kingsideRook = False
        queenSideRook = False
        if(isRookMove):
            queenSideRook,kingsideRook = self.ColourDependentrookStartingPosition(r,c)
        for i in range(len(movingPlayer.pins)-1,-1,-1):
            if movingPlayer.pins[i][0] == r and movingPlayer.pins[i][1] == c:
                pinDirection = (movingPlayer.pins[i][2],movingPlayer.pins[i][3])
                piecePinned = True
                pinDirection = (movingPlayer.pins[i][2],movingPlayer.pins[i][3])
                # if self.board[r][c][1] != Queen :
                #   self.pins.remove(self.pins[i])
                break
        for d in directions:
            for offset in range(1,14):
                endR = r+d[0]* offset
                endC = c + d[1] * offset
                if self.inBounds(endR,endC):
                    if not piecePinned or pinDirection == d or pinDirection == (-d[0],-d[1]):
                        endsq = self.board[endR][endC]
                        _move = Move((r, c), (endR, endC) , self.board)
                        if (isRookMove):
                            _move.didKingSideRookMove = kingsideRook
                            _move.didQueenSideRookMove = queenSideRook
                        if endsq == EMPTYSQUARE:
                            self.validMoves.append(_move)
                        elif self.canCapture((endR,endC)):
                            self.validMoves.append(_move)
                            break
                        else:
                            break
                else:
                    break
    def non_recursiveMoves(self,r,c,directions):
        piecePinned = False
        movingPlayer = self.getMovingPlayer()
        # no need to do the check the only pieces that move non recursively are knights and kings
        # a king cant be pinned and a knight cant move once it is pinned
        for i in range(len(movingPlayer.pins)-1,-1,-1):
            if movingPlayer.pins[i][0] == r and movingPlayer.pins[i][1] == c:
                #pinDirection = (movingPlayer.pins[i][2], movingPlayer.pins[i][3])
                piecePinned = True
                movingPlayer.pins.remove(movingPlayer.pins[i])
                break
        for d in directions:
            endR = r+d[0]
            endC = c + d[1]
            if self.inBounds(endR, endC):
                if not piecePinned :
                    endsq = self.board[endR][endC]
                    if endsq == EMPTYSQUARE:
                        self.validMoves.append(Move((r, c), (endR, endC) , self.board))
                    elif self.canCapture((endR,endC)):
                        self.validMoves.append(Move((r, c), (endR, endC) , self.board))
    def ColourDependentrookStartingPosition(self,r,c):
        team = self.turn
        queenSideRook = False
        kingSideRook = False
        if team == RED:
            kingSideRook = True if (r,c) == (13,10) else False
            queensideRook = True if (r, c) == (13, 3) else False
        elif team == BLUE:
            kingSideRook = True if (r,c) == (3,0) else False
            queensideRook = True if (r, c) == (13, 0) else False
        elif team == YELLOW:
            kingSideRook = True if (r,c) == (0,3) else False
            queensideRook = True if (r, c) == (0,10) else False
        elif team == GREEN:
            kingSideRook = True if (r,c) == (10,13) else False
            queensideRook = True if (r, c) == (3, 13) else False

        return queensideRook,kingSideRook
    def rookMoves(self, r, c):
        rookDirections = ((1,0),(-1,0),(0,1),(0,-1))
        self.recursiveMoves(r,c,rookDirections,isRookMove=True)
    def bishopMoves(self, r, c):
        bishopDirections = ((1,1),(1,-1),(-1,1),(-1,-1))
        self.recursiveMoves(r,c,bishopDirections)
    def knightMoves(self, r, c):
        knightDirections =((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(-1,2),(1,-2),(-1,-2))
        self.non_recursiveMoves(r,c,knightDirections)
    def queenMoves(self, r, c):
        queendirections = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
        self.recursiveMoves(r, c, queendirections)
    def getCastleMoves(self):
        currentPlayer = self.getMovingPlayer()
        redKingSidemovement = (0,1)
        blueKingSidemovement = (-1,0)
        yellowKingSidemovement = (0,-1)
        greenKingSidemovement = (1,0)
        team = self.turn
        kingLocation = currentPlayer.KingLocation
        kingsideDirection = ()
        cancastleKingsideInState = False
        canCastleQueenSideInState = False
        if team == RED:
            kingsideDirection = redKingSidemovement
        elif team == BLUE:
            kingsideDirection = blueKingSidemovement
        elif team == YELLOW:
            kingsideDirection = yellowKingSidemovement
        elif team == GREEN:
            kingsideDirection = greenKingSidemovement
        # checking if the rook still exist
        if currentPlayer.canCastleKingSide:
            kingRookR = kingLocation[0] + 3 * kingsideDirection[0]
            kingRookC = kingLocation[1] + 3 * kingsideDirection[1]
            kingRook = self.board[kingRookR][kingRookC]
            if  self.pieceTypeFromNumber(kingRook)!= ROOK or self.pieceTeamFromNumber(kingRook)!= team:
                currentPlayer.canCastleKingSide = False
                cancastleKingsideInState = False
        if currentPlayer.canCastleQueenSide:
            queenRookR = kingLocation[0] - 4 * kingsideDirection[0]
            queenRookC = kingLocation[1] - 4 * kingsideDirection[1]
            queenRook = self.board[queenRookR][queenRookC]
            if self.pieceTypeFromNumber(queenRook)!= ROOK or self.pieceTeamFromNumber(queenRook)!= team:
                currentPlayer.canCastleQueenSide = False
                canCastleQueenSideInState = False

        if currentPlayer.canCastleKingSide:
            cancastleKingsideInState = True
            for i in range(0,3):
                endr = kingLocation[0]+ i* kingsideDirection[0]
                endc = kingLocation[1]+ i * kingsideDirection[1]
                endSquare = self.board[endr][endc]
                if endSquare != EMPTYSQUARE or self.squareInCheck(endr,endc):
                    cancastleKingsideInState = False


        if currentPlayer.canCastleQueenSide:
            canCastleQueenSideInState = True
            for i in range(0,4):
                endr = kingLocation[0] - i* kingsideDirection[0]
                endc = kingLocation[1] - i * kingsideDirection[1]
                endSquare = self.board[endr][endc]
                if endSquare != EMPTYSQUARE or (self.squareInCheck(endr,endc) and i != 3):
                    canCastleQueenSideInState = False
        if cancastleKingsideInState:
            finalKingLocation = (kingLocation[0]+2*kingsideDirection[0],kingLocation[1] + 2*kingsideDirection[1])
            startingRookPosition = (kingRookR,kingRookC)
            finalRookPosition = (kingRookR-2*kingsideDirection[0],kingRookC-2*kingsideDirection[1])
            _move = Move((kingLocation[0], kingLocation[1]), finalKingLocation, self.board,isCastle=True,didKingMove=True,startingRookPosition = startingRookPosition,finalRookPosition = finalRookPosition)
            self.validMoves.append(_move)
        if canCastleQueenSideInState:
            finalKingLocation = (kingLocation[0] - 2 * kingsideDirection[0], kingLocation[1] - 2 * kingsideDirection[1])
            startingRookPosition = (queenRookR,queenRookC)
            finalRookPosition = (queenRookR+3*kingsideDirection[0],queenRookC+3*kingsideDirection[1])
            _move = Move((kingLocation[0], kingLocation[1]), finalKingLocation, self.board, isCastle=True,didKingMove=True,startingRookPosition=startingRookPosition,finalRookPosition= finalRookPosition)
            self.validMoves.append(_move)
    def kingMoves(self, r, c):
        kingDirections = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
        team = self.turn
        movingPlayer = self.getMovingPlayer()
        inCheck, pins, checks = self.checkForPinsAndChecks(movingPlayer)# potentially not needed
        for d in kingDirections:
            endRow = r + d[0]
            endCol = c + d[1]
            if self.inBounds(endRow,endCol):
                endPiece = self.board[endRow][endCol]
                if endPiece == NULLSQUARE:
                    continue
                if self.pieceTeamFromNumber(endPiece)!= team:
                    if team == RED:
                        self.RedPlayer.KingLocation = (endRow,endCol)
                    elif team == BLUE:
                        self.BluePlayer.KingLocation = (endRow,endCol)
                    elif team == YELLOW:
                        self.YellowPlayer.KingLocation = (endRow,endCol)
                    elif team == GREEN:
                        self.GreenPlayer.KingLocation = (endRow,endCol)

                    inCheck,pins,checks = self.checkForPinsAndChecks(movingPlayer)

                    # valid moves will be called form here.
                    if not inCheck:
                        self.validMoves.append(Move((r, c), (endRow, endCol) , self.board,didKingMove=True))
                    if team == RED:
                        self.RedPlayer.KingLocation = (r, c)
                    elif team == BLUE:
                        self.BluePlayer.KingLocation = (r, c)
                    elif team == YELLOW:
                        self.YellowPlayer.KingLocation = (r, c)
                    elif team == GREEN:
                        self.GreenPlayer.KingLocation = (r, c)

                self.checkForPinsAndChecks(movingPlayer)# resets the pins,checks,incheck to what they should be
    def didPawnPremote(self,rc):
        # pawns are forced to become 1 point Queens
        endPiece = self.board[rc[0]][rc[1]]
        r = rc[0]
        c = rc[1]
        team = self.turn

        if team == RED and r == 6:
            return True
        elif team == BLUE and c == 7:
            return True
        elif team == YELLOW and r == 7:
            return True
        elif team == GREEN and c == 6:
            return True
        else:
                return False
    def PawnPremotion(self, r,c):
        # pawns are forced to become 1 point Queens
        endPiece = self.board[r][c]
        team = self.pieceTeamFromNumber(endPiece)
        if self.pieceTypeFromNumber(endPiece) == PAWN:
            if team == RED and r == 6:
                self.premotePawn(r,c)
            elif team == BLUE and c == 7:
                self.premotePawn(r,c)
            elif team == YELLOW and r == 7:
                self.premotePawn(r,c)
            elif team == GREEN and c == 6:
                self.premotePawn(r,c)
    def premotePawn(self,r,c):
        endPiece = self.board[r][c]
        endPiece = (endPiece -PAWN) # removes pawn label
        endPiece = endPiece + QUEEN
        endPiece = endPiece *-1
        self.board[r][c] = endPiece

    # all moves considering check
    def getLegalMoves(self,movingPlayer : Player):
        # The code for this is incorrect
        # legal moves needs to be called
        moves = []
        inCheck,pins,checks = self.checkForPinsAndChecks(movingPlayer)
        kingRow = movingPlayer.KingLocation[0]
        kingCol = movingPlayer.KingLocation[1]
        if inCheck:
            if len(checks) == 1: # only checked by 1 piece
                moves = self.validMoves
                check = checks[0]
                checkRow = check[0]
                checkCol = check[1]
                pieceChecking = self.board[check[0],check[1]]
                validSquares = []
                if self.pieceTypeFromNumber(pieceChecking) == KNIGHT:
                    validSquares = [(check[0],check[1])]
                else:
                    for i in range(1,13):
                        if not self.inBounds(kingRow + check[2][0]*i,kingCol +check[2][1] * i):
                            break
                        validSquare = (kingRow + check[2][0]*i,kingCol +check[2][1] * i ) # check values may not be right. would probably need direction
                        validSquares.append(validSquare)
                        if validSquare[0] == checkRow and validSquare[1] ==  checkCol:
                            break
                for i in range(len(moves)-1,-1,-1):
                    if self.pieceTypeFromNumber(moves[i].pieceMoved) !=KING:
                        if not (moves[i].endRow,moves[i].endCol) in validSquares:
                            moves.remove(moves[i])
                if moves == []:
                    #print("checkmated1")
                    if not self.fakeGame:
                        print("checkmate")
                        print(self.getMovingPlayer())
                    self.checkMatePlayer(movingPlayer)
                #checkmate if validmoves = []
            else: #double check
                self.kingMoves(kingRow,kingCol)
                if moves == []:
                    if not self.fakeGame:
                        print("checkmate")
                        print(self.getMovingPlayer())
                    #print("checkmated2")
                    self.checkMatePlayer(movingPlayer)

                #checkmate if validmoves = []
        else:
            moves = self.validMoves
            if moves == []:# stalemate
                if not self.fakeGame:
                    print("Stalemate")
                    print(self.getMovingPlayer())
                currentPlayer = self.getMovingPlayer()

                for player in self.allPlayers:
                    if player == currentPlayer:
                        player.Score+=20
                    elif player!=currentPlayer and player.playing == True:
                        player.Score+=10
                self.removePlayer(currentPlayer)
        return moves
    def removePlayer(self,player:Player,wasKingCapturedOnAnotherTurn = False):
        player.playing = False
        self.update_current_state()
        #self.deactivatePlayer(player.team)
        numberOfremainingPlayers = 0
        for possibleRemainingPlayers in self.allPlayers:
            if possibleRemainingPlayers.playing:
                numberOfremainingPlayers += 1
        print("number of players remaining: ")
        print(numberOfremainingPlayers)
        #if numberOfremainingPlayers <= 1:
        #    self.gameOver = True
        #if not wasKingCapturedOnAnotherTurn:
        #    self.finishTurn()
    def deactivatePlayer(self,team):
        for r in range(14):
            for c in range(14):
                piece = self.board[r][c]
                if piece != EMPTYSQUARE and piece !=NULLSQUARE and self.pieceTeamFromNumber(piece) == team:
                    self.board[r][c] = self.DeactivatePieceFromNumber(piece)
    def checkMatePlayer(self,player:Player):
        print("player checkmated")
        self.scorePlayersThatCheckmatedPlayer(player)
        player.playing = False
        self.removePlayer(player)
    def scorePlayersThatCheckmatedPlayer(self,player:Player):
        team = self.turn
        r = player.KingLocation[0]
        c = player.KingLocation[1]
        playersthatCheckmated = []
        directions = ((-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1))
        knightMoves = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2))
        inCheck = False
        for d in directions:
            for i in range(1, 13):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if self.inBounds(endRow, endCol):
                    endpiece = self.board[endRow][endCol]
                    if endpiece == NULLSQUARE:
                        break
                    elif endpiece != EMPTYSQUARE and self.pieceTeamFromNumber(endpiece) != team:
                        type = self.pieceTypeFromNumber(endpiece)
                        rookDirections = ((1, 0), (-1, 0), (0, 1), (0, -1))
                        bishopDirections = ((1, 1), (1, -1), (-1, 1), (-1, -1))
                        enemyTeam = self.pieceTeamFromNumber(endpiece)
                        if self.isTeamInactive(enemyTeam):
                            break
                        if ((rookDirections.__contains__(d) and type == ROOK) or
                                (bishopDirections.__contains__(d) and type == BISHOP) or
                                (type == QUEEN) or (type == PAWN and i == 1 and (
                                        enemyTeam == YELLOW and ((-1, -1), (-1, 1)).__contains__(d) or
                                        enemyTeam == RED and ((1, -1), (1, 1)).__contains__(d) or
                                        enemyTeam == GREEN and ((-1, 1), (1, 1)).__contains__(d) or
                                        enemyTeam == BLUE and ((-1, -1), (1, -1)).__contains__(d)
                                        #
                                )
                                ) or (i == 1 and type == KING)):

                            #inCheck = True
                            if not playersthatCheckmated.__contains__(enemyTeam):
                                playersthatCheckmated.append(enemyTeam)
                        else:

                            break
                else:
                    break
        for m in knightMoves:
            endRow = r + m[0]
            endCol = c + m[1]
            if self.inBounds(endRow, endCol):
                endpiece = self.board[endRow][endCol]
                if (self.pieceTeamFromNumber(endpiece) != team and self.pieceTypeFromNumber(endpiece) == KNIGHT):
                    if self.isTeamInactive(self.pieceTeamFromNumber(endpiece)):
                        continue
                    #inCheck = True
                    if not playersthatCheckmated.__contains__(enemyTeam):
                        playersthatCheckmated.append(enemyTeam)

        for team in playersthatCheckmated:
            if team == RED:
                self.RedPlayer.Score+=20
            elif team == BLUE:
                self.BluePlayer.Score+=20
            elif team == YELLOW:
                self.YellowPlayer.Score+=20
            elif team == GREEN:
                self.GreenPlayer.Score+=20
    def getValidMoves(self):  # context free valid moves. not considering moves
        # would be better to track the position of each piece with a 2d dictionary(TEAMS:PIECES)
        for r in range(ChessMain.DIMENSIONS):
            for c in range(ChessMain.DIMENSIONS):
                if self.board[r][c] != EMPTYSQUARE and self.board[r][c] != NULLSQUARE:

                    if self.pieceTeamFromNumber(self.board[r][c]) == self.turn:
                        piece = self.pieceTypeFromNumber(self.board[r][c])
                        if piece == PAWN:
                            self.pawnMoves(r, c)
                        elif piece == ROOK:
                            self.rookMoves(r,c)
                        elif piece == BISHOP:
                            self.bishopMoves(r, c)
                        elif piece == KNIGHT:
                            self.knightMoves(r, c)
                        elif piece == QUEEN:
                            self.queenMoves(r, c)
                        elif piece == KING:
                            self.kingMoves(r, c)

        return self.validMoves
    def movePriority(self,move):
        if move.wasEnpassent:
            return 0
        return 1

    def finishTurn(self):
        #currentState = self.board.copy()
        #self.boards.append(currentState)
        # the issue could come from when a player is checkmated/stalemated. finish turn is called from a funciton within this finish turn scope and then the code contines from here.
        # if player is active continue with the rest of the code below
        # if not repeat the code above
        # look into the logic of finish turn and how it works when it is called from other parts of the code. checkmate/stalemate
        # try to make the code work without using finish turn from multiple areas to avoid scope issues when backtracking
        activePLayers = 0
        for player in self.allPlayers:
            if player.playing:
                activePLayers+=1
        if activePLayers<=1 or self.fiftyRuleRepition == 50 or self.movesMade == self.MaxMoveLimit:
            self.finishGame()
        while not self.gameOver:
            self.turn = TEAMS[(self.turnIndex + 1) % 4]
            self.turnIndex += 1
            self.removeEnPassentSquares()
            self.validMoves = []
            if not self.getMovingPlayer().playing:
                continue
            self.checkForPinsAndChecks(self.getMovingPlayer())
            self.getValidMoves()
            self.validMoves = self.getLegalMoves(self.getMovingPlayer())
            if self.getMovingPlayer().playing:
                break
            else:
                activePLayers-=1
                if not self.fakeGame:
                    print("here 3")
                if activePLayers<=1:
                    self.finishGame()
                    break

        if (not self.getMovingPlayer().inCheck):
            self.getCastleMoves()
        self.validMoves.sort(key=self.movePriority)
        self.validMoves = list(dict.fromkeys(self.validMoves))
        for team in TEAMS:
            if self.turn ==team:
                self.currentState[channels[team*10]][:,:] = 1
            else:
                self.currentState[channels[team*10]][:,:] = 0



        if self.fakeGame == False and self.validMoves ==[]:
            print("here no moves but didnt stalemate/checkmate")
            print("Hello")
            print("1")
            print("2")
            print(self.getMovingPlayer())
            self.getValidMoves()
            self.validMoves = self.getLegalMoves(self.getMovingPlayer())
    def removeEnPassentSquares(self):
        for i in range(len(self.enPassentSquares) - 1, -1, -1):
            if self.turn == self.enPassentSquares[i][1]:
                self.enPassentSquares.remove(self.enPassentSquares[i])
    def getScore(self,PieceAsNumber):
        if PieceAsNumber== EMPTYSQUARE or PieceAsNumber == NULLSQUARE:
            return 0
        team  = self.pieceTeamFromNumber(PieceAsNumber)
        player = self.getPlayerFromTeam(team)
        if PieceAsNumber!= EMPTYSQUARE and PieceAsNumber!= NULLSQUARE and not player.playing :
            return 0
        if PieceAsNumber<0: # premoted Queen
            return 1

        PieceAsNumbermod = PieceAsNumber%10
        if PieceAsNumbermod == QUEEN:
            return 9
        elif PieceAsNumbermod == ROOK:
            return 5
        elif PieceAsNumbermod == KNIGHT:
            return 3
        elif PieceAsNumbermod == BISHOP:
            return 3
        elif PieceAsNumbermod == PAWN:
            return 1
        return 0
    def current_state(self):
        return self.currentState
    """def current_state(self):
        # May need to return board state from the perspective of the current player
        return self.board
        """
class Move():

    def __eq__(self, other):
        if isinstance(other, Move):
            if self.startSq == other.startSq and self.endSq == other.endSq:
                return True
        return False

    def rankFileNotation(self):
        pass

    def __repr__(self):

        return (','.join([str(value) for value in self.startSq])) + " to " + (
            ','.join([str(value) for value in self.endSq]))
    def __str__(self):

        return (','.join([str(value) for value in self.startSq])) + " to " + (
            ','.join([str(value) for value in self.endSq]))

    def __hash__(self):
        return hash((self.startSq, self.endSq))
# wouldve been better to have pawn moves and rook moves extend the move class to avoid unnescary extra overhead
    def __init__(self, startSq, endSq, board,wasPawnAdvance2 = False,PawnAdvance1 = (0,0),pieceTeam = RED,wasEnpassent = False,enPassentPawn = (0,0),
                 didKingMove = False,didKingSideRookMove = False,didQueenSideRookMove = False,isCastle=False,
                 finalRookPosition = (0,0),startingRookPosition = (0,0)):
        self.startSq = startSq
        self.endSq = endSq
        self.startRow = startSq[0]
        self.starCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]
        self.pieceMoved = board[self.startRow][self.starCol]
        self.pieceCaptured = board[self.endRow][self.endCol]

        self.pawnPremoted = False
        self.wasPawnAdvance2 = wasPawnAdvance2
        self.PawnAdvance1 = PawnAdvance1
        self.pieceTeam = pieceTeam
        self.wasEnpassent = wasEnpassent
        self.enPassentPawn = enPassentPawn

        self.didKingMove = didKingMove
        self.didKingSideRookMove = didKingSideRookMove
        self.didQueenSideRookMove = didQueenSideRookMove
        self.isCastle = isCastle
        self.startingRookPosition = startingRookPosition
        self.finalRookPosition = finalRookPosition
    def encode(self):
        # encoding moves from the startsquare to the end square gives 14*14*14*14 possible outputs
        # using this encoding of choosing a square and using 1 of 29 moves gives 14*14*(8*13+8)
        # starting square gives 14*14 values
        # moves gives 8 for each knight move and for queen moves there are 8 directions with 13 posibble magnitudes
        # giving (13*8+8)
        # encodes a 14*14*29 3d image into a 1d image
        possibleStartingSquares = 14*14
        possibleMoves = (8*13+8)
        startingSquareEncoding = self.starCol*14+self.startRow
        moveEncoding = 0
        knightDirections = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2))
        queendirections = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        directionR = self.endRow-self.startRow
        directionC = self.endCol-self.starCol
        moveVector = (directionR,directionC)
        if moveVector in knightDirections:
            moveEncoding = knightDirections.index(moveVector)
        else:
            # move vector will be a multiple of queenDirections
            # a way to get the direction will be to divide a non-zero vector element
            directionMagnitude = 1
            if (directionR)!=0:
                directionMagnitude = abs(directionR)
            elif (directionC)!=0:
                directionMagnitude = abs(directionC)

            moveVector = (directionR/abs(directionMagnitude),directionC/abs(directionMagnitude))
            queenMoveIndex = queendirections.index(moveVector)
            moveEncoding = 8+(queenMoveIndex*8+directionMagnitude-1)
            #first 8 moves belong to knight directions
            #-1 to ensure index at 0
        # each starting direction has 29 moves. starting square *29 + move
        encoding = possibleMoves*startingSquareEncoding+moveEncoding
        return encoding

def createNPArray(x,y):
    square_state = np.zeros((4, 14, 14))
    square_state[1] = True
    print(square_state[1])
    square_state[0][0][3:7] = True# yellowking
    square_state[0][0][7:11] = 2# yellowqueen
    square_state[0][13][7:11] = 1
    square_state[0][13][3:7] = 2
    print(square_state[0])
if __name__ == "__main__" :
    createNPArray(4,2)
