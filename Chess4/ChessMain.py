import pygame as p
import pygame.freetype
import numpy as np
import AI
import ChessEngine
#from Chess import ChessEngine
RED = 10
BLUE = 20
YELLOW = 30
GREEN = 40

WIDTH= HEIGHT = 700
DIMENSIONS = 14
SQ_SIZE = 50
MAX_FPS = 15
IMAGES = {}
p.init()

def loadImages():
    colours = ChessEngine.TEAMSs # make r b g w
    pieceType = ChessEngine.PIECESs
    for piece in pieceType:
        for colour in colours:

            IMAGES[colour + piece] = p.transform.scale(p.image.load("Images/" + colour + piece + ".png") , (SQ_SIZE , SQ_SIZE))

def drawGameState(screen,gs):
    drawBoard(screen)
    drawPieces(screen,gs)


def drawBoard(screen):

    colours = [p.Color("white"),p.Color("grey")]
    for r in range(DIMENSIONS):
        for c in range(DIMENSIONS):
            colour = colours[(r+c)%2]
            p.draw.rect(screen,colour,p.Rect(c*SQ_SIZE,r*SQ_SIZE,SQ_SIZE,SQ_SIZE))
    # drawing the void squares
    for x in ( [0,11]):
        for y in ([0,11]):
            p.draw.rect(screen,p.Color("black"),p.Rect(x*SQ_SIZE,y*SQ_SIZE,3*SQ_SIZE,3*SQ_SIZE))

def drawPieces(screen,gs):
    for r in range(DIMENSIONS):
        for c in range(DIMENSIONS):
            piece = gs.board[r][c]
            if piece != ChessEngine.EMPTYSQUARE and piece != ChessEngine.NULLSQUARE:
                screen.blit(IMAGES[ChessEngine.GameState.pieceNameFromNumber(piece)],
                            p.Rect(c*SQ_SIZE,r*SQ_SIZE,SQ_SIZE,SQ_SIZE))

def drawScore(screen,gs):
    font = pygame.font.SysFont("monospace",50)
    font2 = pygame.font.SysFont("monospace",20)
    RedScore = font.render(str(gs.RedPlayer.Score), 1, (255, 0, 0))
    screen.blit(RedScore,(600,600))
    if not gs.RedPlayer.playing:
        InactiveText = font2.render("INACTIVE", 1, (255, 0, 0))
        screen.blit(InactiveText, (600, 650))

    BlueScore = font.render(str(gs.BluePlayer.Score), 1, (0, 0, 255))
    screen.blit(BlueScore, (50, 600))
    if not gs.BluePlayer.playing:
        InactiveText = font2.render("INACTIVE", 1, (0, 0, 255))
        screen.blit(InactiveText, (50, 650))

    YellowScore = font.render(str(gs.YellowPlayer.Score), 1, (255, 255, 0))
    screen.blit(YellowScore,(50,50))
    if not gs.YellowPlayer.playing:
        InactiveText = font2.render("INACTIVE", 1, (255, 255, 0))
        screen.blit(InactiveText, (50, 100))

    GreenScore = font.render(str(gs.GreenPlayer.Score), 1, (0, 255, 0))
    screen.blit(GreenScore, (600, 50))
    if not gs.GreenPlayer.playing:
        InactiveText = font2.render("INACTIVE", 1, (0, 255, 0))
        screen.blit(InactiveText, (600, 100))
def showPlayerTurn(screen,gs):
    turn = gs.turn
    font = pygame.font.SysFont("monospace", 20)
    score = ()
    if turn == RED:
        score = font.render("Your Turn", 1, (255, 0, 0))
        screen.blit(score, (575, 550))
    elif turn == BLUE:
        score = font.render("Your Turn", 1, (0, 0, 255))
        screen.blit(score, (25, 550))
    elif turn == GREEN:
        score = font.render("Your Turn", 1, (0, 255, 0))
        screen.blit(score, (575, 0))
    elif turn == YELLOW:
        score = font.render("Your Turn", 1, (255, 255, 0))
        screen.blit(score, (25, 0))
class Game():

    def __init__(self,isDisplayed = True):
        #print("pygame initisialised")
        self.isDisplayed = isDisplayed
        self.gs = ChessEngine.GameState()

    def startGame(self,mcts = None,DontCreateGS= False):
        print("pygame initisialised")
        screen = p.display.set_mode((WIDTH, HEIGHT))
        screen.fill(p.Color("white"))
        clock = p.time.Clock()
        if not DontCreateGS:
            gs = ChessEngine.GameState()
        else:
            gs = self.gs
        gscopyTest = ()  # type: GameState
        loadImages()
        moveMade = False
        running = True
        sqSelected = ()
        CurrentPlayerClickFromTo = []
        temp = 1e-3
        while running:
            player = gs.getMovingPlayer()
            for e in p.event.get():
                if e.type == p.QUIT:
                    running = False
                player = gs.getMovingPlayer()
                if not gs.gameOver and not player.isHumanPlaying and (not player.isMcts or mcts == None ):

                    if len(gs.validMoves) == 0:
                        gs.removePlayer(gs.getMovingPlayer()) #screen This potentially should be removed
                    else:
                        AIMove = AI.findRandomMove(gs.validMoves) # TODO change back to random move
                        #AIMove = AI.greedyFindBestMove(gs.validMoves,gs)
                        gs.makeMove(AIMove)
                    moveMade = True
                elif not gs.gameOver and not player.isHumanPlaying and player.isMcts:
                    #move = mcts.get_action(gs,temp=temp,
                    #                                 return_prob=0)
                    #move = mcts.get_action(gs)
                    move = mcts.get_action(gs,temp=temp,
                                                     return_prob=0)
                    gs.do_move(move)

                elif e.type == p.MOUSEBUTTONDOWN:
                    if not gs.gameOver and player.isHumanPlaying:
                        location = p.mouse.get_pos()
                        col = location[0] // SQ_SIZE
                        row = location[1] // SQ_SIZE
                        if sqSelected == (row, col):
                            sqSelected = ()
                            CurrentPlayerClickFromTo = []
                        else:
                            sqSelected = (row, col)
                            CurrentPlayerClickFromTo.append(sqSelected)
                        if len(CurrentPlayerClickFromTo) == 2:

                            move = ChessEngine.Move(CurrentPlayerClickFromTo[0], CurrentPlayerClickFromTo[1], gs.board)
                            if move in gs.validMoves:  ## move is different to the move in the list validMoves
                                moveIndex = gs.validMoves.index(move)
                                move = gs.validMoves[moveIndex]
                                gs.makeMove(move)
                                moveMade = True
                                sqSelected = ()  # resets user clicks
                                CurrentPlayerClickFromTo = []
                            else:
                                CurrentPlayerClickFromTo = [sqSelected]
                elif e.type == p.KEYDOWN:
                    if e.key == p.K_z:
                        gs.undoMove()
                        sqSelected = ()
                        CurrentPlayerClickFromTo = []
                    elif e.key == p.K_c:
                        gscopyTest = gs.copy()
                    elif e.key == p.K_v:

                        print(gscopyTest.board)
                if moveMade:
                    moveMade = False
                    drawGameState(screen, gs)
                drawGameState(screen, gs)
                showPlayerTurn(screen, gs)
                drawScore(screen, gs)
                clock.tick(MAX_FPS)
                p.display.flip()

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.gs = ChessEngine.GameState()
        for player_ in self.gs.allPlayers:
            player_.isHumanPlaying = False
            player_.isMcts = True
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.gs,
                                                 temp=temp,
                                                 return_prob=1)
            #print("Move made")
            #print(move)
            mcts_probs.append(move_probs)
            current_players.append(self.gs.turn)
            self.gs.do_move(move)
            z = self.gs.turnCentricState
            states.append(z)
            #self.gs.finishGame() # remove after testing
            #self.gs.getLegalMoves()
            #end, winner = self.board.game_end()
            if self.gs.gameOver:

                print("GameOver")
                # winner from the perspective of the current player of each state

                winners_z = np.zeros(len(current_players)) - 1
                npScore = np.array(self.gs.finalScore)
                winners = np.where(npScore == max(npScore))
                relativeScore = self.gs.relativeScore()
                for winner in winners[0]:
                    if winner != -1:
                        winners_z[np.array(current_players) == ChessEngine.TEAMS[winner]] = 1.0

                player.reset_player()
                return self.gs.gameOver, zip(states, mcts_probs, winners_z)
    def start_play(self, current_mcts_player, best_mcts_player,parralelWinDictionary = None,temp=1e-3):


        """start a game between two players"""

        self.gs = ChessEngine.GameState()

        # selecting which player will be controlled by the mcts
        BestmctsPlayers = np.random.choice(4, 2, replace=False)
        for i in range(len(self.gs.allPlayers)):
            if i in BestmctsPlayers:
                self.gs.allPlayers[i].isBestMcts = True

        while True:
            player_in_turn = current_mcts_player
            if  self.gs.getMovingPlayer().isBestMcts:
                player_in_turn = best_mcts_player
            move = player_in_turn.get_action(self.gs,
                                             temp=temp,
                                            return_prob=0)
            self.gs.do_move(move)

            if self.gs.gameOver:
                bestMctsScore = 0
                currentMctsScore =0
                winner = 0
                for i in range(len(self.gs.allPlayers)):
                    if i in BestmctsPlayers:
                        bestMctsScore +=self.gs.allPlayers[i].Score
                    else:
                        currentMctsScore+= self.gs.allPlayers[i].Score
                if currentMctsScore>bestMctsScore:
                    winner = 1
                elif currentMctsScore<bestMctsScore:
                    winner =2
                else:
                    winner = -1
                if parralelWinDictionary !=None:
                    parralelWinDictionary[winner] = parralelWinDictionary[winner]+1
                    print(parralelWinDictionary)
        return winner
def main():
    print("pygame initisialised")
    screen = p.display.set_mode( (WIDTH,HEIGHT))
    screen.fill(p.Color("white"))
    clock = p.time.Clock()
    gs = ChessEngine.GameState()
    gscopyTest = () # type: GameState
    loadImages()
    moveMade = False
    running = True
    sqSelected = ()
    CurrentPlayerClickFromTo = []

    while running:
        player = gs.getMovingPlayer()
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            if not gs.gameOver and not player.isHumanPlaying and not player.isMcts:

                if len(gs.validMoves) ==0:
                    gs.removePlayer(gs.getMovingPlayer())
                else:
                    AIMove= AI.findRandomMove(gs.validMoves)
                    gs.makeMove(AIMove)
                moveMade = True
            elif not gs.gameOver and not player.isHumanPlaying and  player.isMcts:
                True
            elif e.type == p.MOUSEBUTTONDOWN:
                if  not gs.gameOver and player.isHumanPlaying:
                    location = p.mouse.get_pos()
                    col = location[0]//SQ_SIZE
                    row = location[1]//SQ_SIZE
                    if sqSelected == (row,col):
                       sqSelected = ()
                       CurrentPlayerClickFromTo = []
                    else:
                        sqSelected = (row,col)
                        CurrentPlayerClickFromTo.append(sqSelected)
                    if len(CurrentPlayerClickFromTo) == 2:

                        move = ChessEngine.Move(CurrentPlayerClickFromTo[0],CurrentPlayerClickFromTo[1],gs.board)
                        if move in gs.validMoves: ## move is different to the move in the list validMoves
                            moveIndex = gs.validMoves.index(move)
                            move = gs.validMoves[moveIndex]
                            gs.makeMove(move)
                            moveMade = True
                            sqSelected = () # resets user clicks
                            CurrentPlayerClickFromTo = []
                        else:
                            CurrentPlayerClickFromTo = [sqSelected]
            elif e.type == p.KEYDOWN:
                if e.key ==p.K_z:
                    gs.undoMove()
                    sqSelected = ()
                    CurrentPlayerClickFromTo =[]
                elif e.key == p.K_c:
                    gscopyTest = gs.copy()
                elif e.key == p.K_v:

                    print(gscopyTest.board)
            if moveMade:
                moveMade = False
                drawGameState(screen,gs)
            drawGameState(screen, gs)
            showPlayerTurn(screen,gs)
            drawScore(screen,gs)
            clock.tick(MAX_FPS)
            p.display.flip()


if __name__ == "__main__" :
    g = Game()
    g.startGame()
    #main()

