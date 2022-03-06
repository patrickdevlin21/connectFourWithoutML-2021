import numpy as np
import pandas as pd
import random
import copy
import math
import matplotlib.pyplot as plt
import numba
from matplotlib import colors

from IPython.display import clear_output

import colorama

MAX = 1000

def get_color_coded_str(i):
    if (i == -1):
      return "\033[3{}m{}\033[0m".format(4, "B") # 4 = blue
    elif (i == 1):
      return "\033[3{}m{}\033[0m".format(1, "R") # 1 = red
    else:
      return "\033[3{}m{}\033[0m".format(7, "-")

def get_color_coded_background(i):
    return "\033[4{}m {} \033[0m".format(i+1, i)

def print_a_ndarray(map, row_sep=" "):
    n, m = map.shape
    fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
    print(fmt_str.format(*map.ravel()))


def silly_print(rm):
    print(" ")
    print_a_ndarray(np.vectorize(get_color_coded_str)(rm))
    print(" ")
#    print("< < < v > > >")
    print("1 2 3 4 5 6 7")
    print(" ")


class ConnectFour:
    win_length = 4
    num_rows = 6
    num_cols = 7

    #Each ConnectFour object is a board_state, which is an array
    #of 0's (empty squares), 1's (red chips) and -1's (black chips).
    #The red chips are played on even turns (initial turn is turn 0), black chips on odd turns.
    def __init__(self):
        #On initialization, the board_state is empty, the turn is 0, the game is not won, all moves are legal, and the last move is nonsense.
        self.board_state = np.zeros((self.num_rows,self.num_cols), dtype=int)
        self.turn = 0
        self.is_won = False
        self.is_over = False
        self.legal_moves = np.arange(7)
        self.last_move = (-1,-1)

    def reset(self):
        #self.reset resets all the game parameters to their initial state.
        self.board_state = np.zeros((self.num_rows,self.num_cols), dtype=int)
        self.turn = 0
        self.is_won = False
        self.is_over = False
        self.legal_moves = np.arange(7)
        self.last_move = (-1,-1)

    def play_chip(self, column, verbose_mode=True):
        #First check if the game is over:
        if self.is_won == True:
            #print('The game is over; %s has won.' % self.winner)
            return
        elif (self.is_won == False and self.legal_moves.size == 0):
            #print('The game has ended in a draw.')
            return
        else:
            #Player whose turn it is places a chip in column
            #Determine whose turn it is:
            if self.turn % 2 == 0:
                chip = 1
            else:
                chip = -1

            #If suggested move is legal, play the chip in column and go to next turn.
            if column in self.legal_moves:
                self.last_move = np.amax(np.where(self.board_state[:,column] == 0)),column
                self.board_state[self.last_move] = chip
                self.turn += 1
            else:
                if (verbose_mode):
                    print('Column is full; please suggest another move.')
                return False
                


            #Check to see if the game has been won:
            self.check_win()

            #Redetermine the legal moves in the board_state:
            self.legal_moves = np.where(self.board_state[0]==0)[0]

            if self.is_won == True:
                #print('The game is over; %s wins!' % self.winner)
                self.is_over = True
            elif (self.is_won == False and self.legal_moves.size == 0):
                #print('The game has ended in a draw.')
                self.is_over = True

    def check_win(self):
        #Checks for a winning position for both players relative to self.last_move:
        y,x = self.last_move

        #Check for column wins containing self.last_move (there can be only one):
        if (y <= self.num_rows - self.win_length):
            sum_temp = 0
            for k in range(0,self.win_length):
                sum_temp += self.board_state[y+k][x]
            if sum_temp == self.win_length:
                self.winner = 'Player 1'
                self.is_won = True
                return
            elif sum_temp == -self.win_length:
                self.winner = 'Player 2'
                self.is_won = True
                return

        #Check for row wins containing self.last_move:
        for i in range(0,self.win_length):
            if (x-i <= self.num_cols - self.win_length and x-i >= 0):
                sum_temp = 0
                for k in range(0,self.win_length):
                    sum_temp += self.board_state[y][x-i+k]
                if sum_temp == self.win_length:
                    self.winner = 'Player 1'
                    self.is_won = True
                    return
                elif sum_temp == -self.win_length:
                    self.winner = 'Player 2'
                    self.is_won = True
                    return

        #Check for diagonal \\ wins containing self.last_move:
        for i in range(0,self.win_length):
            if (x-i <= self.num_cols-self.win_length and x-i >=0 and y-i <= self.num_rows - self.win_length and y-i >= 0):
                sum_temp = 0
                for k in range(0,self.win_length):
                    sum_temp += self.board_state[y-i+k][x-i+k]
                if sum_temp == self.win_length:
                    self.winner = 'Player 1'
                    self.is_won = True
                    return
                elif sum_temp == -self.win_length:
                    self.winner = 'Player 2'
                    self.is_won = True
                    return

        #Check for diagonal // wins containing self.last_move:
        for i in range(0,self.win_length):
            if (x-i <= self.num_cols-self.win_length and x-i >= 0 and y+i > self.num_rows - self.win_length and y+i <= 5):
                sum_temp = 0
                for k in range(0,self.win_length):
                    sum_temp += self.board_state[y+i-k][x-i+k]
                if sum_temp == self.win_length:
                    self.winner = 'Player 1'
                    self.is_won = True
                    return
                elif sum_temp == -self.win_length:
                    self.winner = 'Player 2'
                    self.is_won = True
                    return


    def threats(self, forCurrentPlayer=True):
        if(forCurrentPlayer):
            pm = 1
        else:
            pm = -1
        if(self.turn % 2 == 1):
            pm = -pm

        goodSquares = []
        #Checks for a winning position for both players relative to self.last_move:
        for x in range(self.num_cols):
            for y in range(self.num_rows):
              if(self.board_state[y][x] == 0):
########  I don't care about threats that are in a column.  Those are silly
#        #Check for column wins containing self.last_move (there can be only one):
#                if (y <= self.num_rows - self.win_length):
#                    sum_temp = 0
#                    for k in range(0,self.win_length):
#                        sum_temp += self.board_state[y+k][x]
#                    if sum_temp == self.win_length:
#                        self.winner = 'Player 1'
#                        self.is_won = True
#                        return
#                    elif sum_temp == -self.win_length:
#                        self.winner = 'Player 2'
#                        self.is_won = True
#                        return

        #Check for row wins containing self.last_move:
                for i in range(0,self.win_length):
                    if (x-i <= self.num_cols - self.win_length and x-i >= 0):
                        sum_temp = 0
                        for k in range(0,self.win_length):
                            sum_temp += self.board_state[y][x-i+k]
                        if sum_temp == (pm)*(self.win_length-1):
                            goodSquares.append([x,y])

        #Check for diagonal \\ wins containing self.last_move:
                for i in range(0,self.win_length):
                    if (x-i <= self.num_cols-self.win_length and x-i >=0 and y-i <= self.num_rows - self.win_length and y-i >= 0):
                        sum_temp = 0
                        for k in range(0,self.win_length):
                            sum_temp += self.board_state[y-i+k][x-i+k]
                        if sum_temp == (pm)*(self.win_length-1):
                            goodSquares.append([x,y])
        
        #Check for diagonal // wins containing self.last_move:
                for i in range(0,self.win_length):
                    if (x-i <= self.num_cols-self.win_length and x-i >= 0 and y+i > self.num_rows - self.win_length and y+i <= 5):
                        sum_temp = 0
                        for k in range(0,self.win_length):
                            sum_temp += self.board_state[y+i-k][x-i+k]
                        if sum_temp == (pm)*(self.win_length-1):
                            goodSquares.append([x,y])
        return goodSquares

# create discrete colormap
cmap = colors.ListedColormap(['b', 'w', 'r'])
bounds = [-1,0,1,2]
norm = colors.BoundaryNorm(bounds, cmap.N)

def plot_matrix(rm, title='Connect Four'):
    fig, ax = plt.subplots()
    ax.imshow(rm, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, 7, 1));
    ax.set_yticks(np.arange(-.5, 6, 1));
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    plt.show()




def copyCatMakesCurrentPlayerLose(theGame):
    candidateMoves = list(set(theGame.legal_moves) - set(movesThatSetUpToWin(theGame)))
    if(len(candidateMoves) == len(theGame.legal_moves)):
        return False

    hypotheticalGame = copy.deepcopy(theGame)
#    print("Candidate moves are: ", candidateMoves)
    for j in candidateMoves:
        numberOfPlays = np.amax(np.where(hypotheticalGame.board_state[:,j] == 0))+1
#        print(j+1, " has ", numberOfPlays, " above it.")
        if((numberOfPlays % 2) == 1):
            return False
        for k in range(6):
            hypotheticalGame.play_chip(j, False)

    if(not hypotheticalGame.is_over):
#        print("Looks like copycat comes into play!")
#        silly_print(hypotheticalGame.board_state)
        return not isThisWonForCurrentPlayer(hypotheticalGame)

    return False


# This returns moves m such that  if current player does m and oppenent responds with m, then current player will lose immediately
def movesThatSetUpToWin(theGame):
    possibleMoves = theGame.legal_moves
    movesThatSuck = []
    for m in possibleMoves:
        hypotheticalGame = copy.deepcopy(theGame)
        hypotheticalGame.play_chip(m)
        if(hypotheticalGame.is_won == False):
            if(m in hypotheticalGame.legal_moves):
                hypotheticalGame.play_chip(m)
                if(hypotheticalGame.is_won):
                    movesThatSuck.append(m)
    return movesThatSuck




def forcedWinningPlays(theGame, depth=1, beLazy=False, withZug=False):
    hypotheticalGame = copy.deepcopy(theGame)
    possibleMoves = theGame.legal_moves
    winningMoves = []
    if(depth<1):
        return []

    for m in possibleMoves:
        hypotheticalGame = copy.deepcopy(theGame)
        hypotheticalGame.play_chip(m)
        if(hypotheticalGame.is_won == True):
            winningMoves.append(m)
        elif(depth > 1):
            if(isThisLostForCurrentPlayer(hypotheticalGame, depth-1, beLazy, withZug)):
                winningMoves.append(m)
#    print("Depth :", depth, ". And I think winning plays are: ", winningMoves)
    return winningMoves


def isThisWonForCurrentPlayer(theGame, depth=1, beLazy=False, withZug=False):
    if(forcedWinningPlays(theGame, depth, beLazy, withZug)):
        return True
    else:
        return False


def movesThatDontLose(theGame, depth=1, beLazy=False, withZug=False):
    if((withZug) and (not beLazy)):
        if copyCatMakesCurrentPlayerLose(theGame):
#            print("Found an instance of copyCat making current player lose from this position: ")
 #           silly_print(theGame.board_state)
            return []

    hypotheticalGame = copy.deepcopy(theGame)
    possibleMoves = theGame.legal_moves
    okMoves = []
    if(depth < 1):
        return possibleMoves
    uncertainMoves = []

    for m in possibleMoves:
        hypotheticalGame = copy.deepcopy(theGame)
        hypotheticalGame.play_chip(m)
        if(hypotheticalGame.is_won == True):
            okMoves.append(m) #any move that wins in one move is great!
        else:
            if(not isThisWonForCurrentPlayer(hypotheticalGame, 1, withZug)):
                uncertainMoves.append(m)  # This does a quick pass to see what moves we're still uncertain about (but we can throw away any move that instantly loses)

#    if(depth < 2):
 #       okMoves.extend(uncertainMoves)
 #       return okMoves
    oldDepth=depth
    depth+=1

    if( (len(uncertainMoves) > 1) or (okMoves) or beLazy):
        depth -= 1 # either the tree would blow up otherwise, OR there is a move that instantly wins, so we don't care

#    if(oldDepth < depth):
#        print("I sense a forcing move.  I'm still searching at depth = ", depth, " to see what it would imply.  It seems like you have to move to ", uncertainMoves, " and the board is currently...")
#        silly_print(theGame.board_state)

    for m in uncertainMoves:
        hypotheticalGame = copy.deepcopy(theGame)
        hypotheticalGame.play_chip(m)
        if(not isThisWonForCurrentPlayer(hypotheticalGame, depth, beLazy, withZug)):
            okMoves.append(m) #In this case, we can be sure that this is an alright move by which the second player doesn't allow the first to win (within our depth)
    return okMoves


def isThisLostForCurrentPlayer(theGame, depth=1, beLazy=False, withZug=False):
    if(movesThatDontLose(theGame,depth, beLazy, withZug)):
        return False
    else:
        return True


def delay_inevitable(theGame, depth=1, withZug=False): # This gets called when we're in a death spiral.  We know we'll lose, so this just picks a move that delays losing for as long as possible
    if(depth < 1):
        return theGame.legal_moves
    okMoves = movesThatDontLose(theGame, depth, True)
    if(okMoves):
        return okMoves
    else:
        return delay_inevitable(theGame, depth-1)

## THINK AHEAD BOT 
# This bot thinks a little bit ahead and sees if it's about to win (or lose) and acts accordingly.  Else, it plays randomly
def think_ahead_simple(theGame, depth=1, myTurn = True):
    winningPlays = forcedWinningPlays(theGame,depth)
    if(winningPlays):  # i.e., if this list is non-empty
        print("I have a winning move!", winningPlays)
        return winningPlays[0]

    okMoves = movesThatDontLose(theGame,depth)
    if(okMoves): # i.e., if there is a move that doesn't obviously suck
        a = random.choice(okMoves)
        if(len(okMoves)==1):
            print("THIS IS MY ONLY MOVE!!!  I FIGHT FOR HONOR!!!!")
        elif(len(okMoves) < len(theGame.legal_moves)):
            print("You're not giving me too many choices...  I'm torn between: ", [x+1 for x in okMoves], ".  I guess I'll pick ", a+1)
        else:
            print("Well, I don't see anything great to play, so I'll just go with: ", a+1)
        return a
    else: ## If we get here, the bot knows that it lost, and now it's just going to delay the inevitable
        a = random.choice(delay_inevitable(theGame, depth-1))
        print("Oh shit.  I lost.  Oh, well.  I'll just delay the inevitable to the extent possible: ", a+1)
        return a


def columnsIWantOpponentToPlay(theGame, depth=1, beLazy=False, withZug=False):  ## These are the moves that the current player would like their opponent to play
    hypotheticalGame = copy.deepcopy(theGame)
    hypotheticalGame.turn += 1
    return columnsThatWouldLose(hypotheticalGame, depth, beLazy, withZug)


def columnsThatWouldLose(theGame, depth=1, beLazy=False, withZug=False):  ## This is the moves that the current player definitely shouldn't play
    return list(set(theGame.legal_moves) - set(movesThatDontLose(theGame, depth, beLazy, withZug)))


def columnsOwnedIfMoveThenRespond(theGame, move, depth=1, beLazy=False, withZug=False):
    if(depth < 1):
        return 0

    hypotheticalGame = copy.deepcopy(theGame)
    hypotheticalGame.play_chip(move)  # This is what the game state would look like if the current player did the move they're thinking about

    # Among all possible responses (that don't lose), which response leaves us with the fewest number of spots where we'd be happy to allow opponent to move?
    possibleResponses = movesThatDontLose(hypotheticalGame, depth, beLazy, withZug) #throw in the "-1" so total move-depth-consideration stays the same size
    if(len(possibleResponses) < 3):
        innerDep = 2
    else:
        innerDep = 1  ##These are here to make sure things don't blow up computationally

    best = 7
    bestResponses = []

    for r in possibleResponses:
        responseGame = copy.deepcopy(hypotheticalGame)
        responseGame.play_chip(r)
        responseGame.turn+=1
        c = len(columnsThatWouldLose(responseGame, innerDep, beLazy, withZug))
        if(c < best):
            best = c
            bestResponses = [r]
        elif(c == best):
            bestResponses.append(r)

#    print("If I move: ", move +1, " then you might say ", [y+1 for y in possibleResponses] ," but I expect you'll respond with a move like ", [x+1 for x in bestResponses], " which would leave me with ", best, " columns.")
#    silly_print(hypotheticalGame.board_state)
    return best



def movesToMaximizeColumnsOwned(theGame, possibleMoves, depth=1, beLazy=True, withZug=False):  ## Setting this default of beLazy to true makes it so we iterate through this part faster
    best = -1
    sensibleMoves = []
    if(len(possibleMoves) < 3):
        innerDep=max(2,depth)
    else:
        innerDep=max(1, depth)

    for m in possibleMoves:
        c = columnsOwnedIfMoveThenRespond(theGame, m, innerDep, beLazy, withZug)
        if(c > best):
            sensibleMoves = [m]
            best = c
        elif(c == best):
            sensibleMoves.append(m)
    return sensibleMoves



def movesToOptimizeThreats(theGame, possibleMoves, depth, beLazy=True, withZug=False):
#    currentPlayerInitialThreats = theGame.threats(True) #the TRUE = current player
#    opponentInitialThreats = theGame.threats(False)


    if(len(possibleMoves) < 3):
        innerDep=max(2,depth)
    else:
        innerDep=max(1, depth)

    if(beLazy):
        innerDep = depth



    if(theGame.threats(True) > theGame.threats(False)): # If you have more threats, then play as defensively as possible!
        playDefensively = True
    else:
        playDefensively = False

    if(playDefensively):
      best=35
      nowPossibleMoves = []
      for m in possibleMoves:
        c = unblockedThreatsIfMoveThenRespond(theGame, m, innerDep, beLazy, withZug, True) #final TRUE is "forOpponent"
        if(c < best):
            nowPossibleMoves = [m]
            best = c
        elif(c == best):
            nowPossibleMoves.append(m)

    else:
     nowPossibleMoves = possibleMoves


    sensibleMoves = []
    best = -1

    for m in nowPossibleMoves:
        c = unblockedThreatsIfMoveThenRespond(theGame, m, innerDep, beLazy, withZug)
        if(c > best):
            sensibleMoves = [m]
            best = c
        elif(c == best):
            sensibleMoves.append(m)
#    return sensibleMoves

    if(playDefensively):
        return sensibleMoves

    evenBetterMoves = []
    best = 35

    for m in sensibleMoves:
        c = unblockedThreatsIfMoveThenRespond(theGame, m, innerDep, beLazy, withZug, True) #final TRUE is "forOpponent"
        if(c < best):
            evenBetterMoves = [m]
            best = c
        elif(c == best):
            evenBetterMoves.append(m)

    return evenBetterMoves


def unblockedThreatsIfMoveThenRespond(theGame, move, depth, beLazy, withZug, forOpponent = False):
    if(depth < 0):
        return 0


    hypotheticalGame = copy.deepcopy(theGame)
    hypotheticalGame.play_chip(move)  # This is what the game state would look like if the current player did the move they're thinking about

    # Among all possible responses (that don't lose), which response leaves us with the fewest number of spots where we'd be happy to allow opponent to move?
    possibleResponses = movesThatDontLose(hypotheticalGame, depth, beLazy, withZug) #throw in the "-1" so total move-depth-consideration stays the same size


    if(forOpponent):
        best = len(hypotheticalGame.threats())
        for m in possibleResponses:
            c = unblockedThreatsIfMoveThenRespond(hypotheticalGame, m, depth, beLazy, withZug, False) #final TRUE is "forOpponent"
            if(c > best):
                best = c
                print("You would gain a threat if I moved ", move+1, " and you responded ", m+1)
        return best

    best = 35
    bestResponses = []

    for r in possibleResponses:
        responseGame = copy.deepcopy(hypotheticalGame)
        responseGame.play_chip(r)
        c = len(responseGame.threats())
        if(c < best):
            best = c
            bestResponses = [r]
        elif(c == best):
            bestResponses.append(r)

#    print("If I move: ", move +1, " then you might say ", [y+1 for y in possibleResponses] ," but I expect you'll respond with a move like ", [x+1 for x in bestResponses], " which would leave me with ", best, " columns.")
#    silly_print(hypotheticalGame.board_state)
    return best



######
##  Note!
######

#  My code for "does copy cat win" isn't correct because what could happen is we follow copy cat all the way to the end, and THEN the player manages to convert one of those previously unplayable squares into a win.  So I should just check that at the end of the call




##  Here's the part for the alpha-beta pruning

def crudeScore(theGame):
    if(theGame.is_won):
        return -MAX #This means the game already ended a turn ago!
    if(isThisWonForCurrentPlayer(theGame,1)):  # This means the player has a winning move
        return MAX
    if(isThisLostForCurrentPlayer(theGame, 1)):  # this means all the player's immediate moves are immediately losing
        return -MAX

    if(copyCatMakesCurrentPlayerLose(theGame)):
        return -MAX

    myThreats = theGame.threats()
    oppThreats = theGame.threats(False)
    ## We probably want to only count the lowest threat in each column....  That way we don't get overly excited about stacking garbage on top of threatening stuff!  Or we want to weight lower threats more?
    myThreatLevel = 0
    oppThreatLevel = 0
    for T in myThreats:
        myThreatLevel += 2**T[1]

    for T in oppThreats:
        oppThreatLevel -= 2**T[1]

    return myThreatLevel + oppThreatLevel


####  I'd like to add in a little bit of random noise somewhere to get things smoother...
##  Here, we consider what the score would be if current player made a random move to show how it's sort of precarious and to punish situations where there aren't many good moves
#    pRand = 0
#    if(plusRandom):
#      possibleMoves = theGame.legal_moves
#      for m in possibleMoves:
#        hypotheticalGame=copy.deepcopy(theGame)
#        hypotheticalGame.play_chip(m)
#        pRand -= crudeScore(hypotheticalGame, False)
#      pRand = pRand / (100*max(len(possibleMoves), 1))
#    return pRand + myThreatLevel + oppThreatLevel

def alphaBeta(theGame, depth, alpha, beta): 
    # Terminating condition. i.e 
    # leaf node is reached 
    guess = crudeScore(theGame)
    if((depth < 1) or (guess==MAX) or (guess == -MAX)):
        return guess

    best = -MAX
    possibleMoves = theGame.legal_moves

        # Recur for left and right children 
    for m in possibleMoves: 
        hypotheticalGame = copy.deepcopy(theGame)
        hypotheticalGame.play_chip(m)
        if(hypotheticalGame.is_won):
            val = MAX
        else:
            val = -alphaBeta(hypotheticalGame, depth - 1, -beta, -alpha) 
        best = max(best, val)
        alpha = max(alpha, best)
  
        # Alpha Beta Pruning 
        if beta <= alpha: 
            break 
       
    return best 
       
# Driver Code 
def alphaBetaPickMoves(theGame, depth, givenMoves=[]):
    possibleMoves = theGame.legal_moves
    if(givenMoves):
        possibleMoves = givenMoves

    best = -MAX
    movesWeLike = []
    for m in possibleMoves:
        hypotheticalGame = copy.deepcopy(theGame)
        hypotheticalGame.play_chip(m)
        if(hypotheticalGame.is_won):
            a = MAX
        else:
            a = -alphaBeta(hypotheticalGame, depth-1, -MAX, -best)
        if(best < a):
            movesWeLike = [m]
            best = a
        elif(best == a):
            movesWeLike.append(m)
    return [movesWeLike, best]


def abTest(theGame, depth):
    [ab, score] = alphaBetaPickMoves(theGame, depth)
    print("Crude game score is: ", crudeScore(theGame))
    print("Less crude score is: ", alphaBeta(theGame, depth, -MAX, MAX))
    print("I'm guessing this game has value: ", score, " and alphaBeta recommends the moves: ", [x+1 for x in ab])    




def think_ahead_complicated(theGame, depth, favorCenter = False, ownColumns = False, myTurn = True, withZug=False, considerThreats=False, abPruning=False):
    if(theGame.turn < 3):
        print("It's really early in the game, so I'm going in the center on principle")
        return 3

    instantWins = forcedWinningPlays(theGame, 1) # First check if you can win in one move
    if(instantWins):
        a = random.choice(instantWins)
        print("Good game!  The game is over, and I'll play ", a+1, " for the win!")
        return a

    # Then check if opponent can win in one move
    mustBlockNow = movesThatDontLose(theGame, 1)
    if(len(mustBlockNow) == 1):
        print("An easy choice!  I obviously play ", mustBlockNow[0]+1)
        return mustBlockNow[0]
    if(len(mustBlockNow) < 1):
        a = random.choice(theGame.legal_moves)
        print("Well!  Looks like the game is over and you have two ways to win!  I guess I'll just play ", a+1)
        return a


    winningPlays = forcedWinningPlays(theGame,depth, False, withZug)
    if(winningPlays):  # i.e., if this list is non-empty
        print("I spot a winning combination with any of the following moves!", [x+1 for x in winningPlays])
        return random.choice(winningPlays)

    okMoves = movesThatDontLose(theGame,depth, False, withZug)
    notBadMoves = copy.deepcopy(okMoves)

    if((ownColumns) and not(considerThreats)):
        foo=movesToMaximizeColumnsOwned(theGame, okMoves, depth)
#        bar = columnsIWantOpponentToPlay(theGame, depth, True)
#        if(bar):
#            print("I want you to play: ", [x+1 for x in bar])
        if (len(foo) < len(okMoves)):
            print("I wanna own columns, so I'm only considering ", [x+1 for x in foo], " as opposed to all ok-seeming moves: ", [y+1 for y in okMoves])
            okMoves = foo

    if(considerThreats):
        foobar = movesToOptimizeThreats(theGame, okMoves, 1, True, False)
        if(len(foobar) < len(okMoves)):
            print("Ok.  I'm thinking of optimizing threats, and I'm considering ", [x+1 for x in foobar])
            okMoves = foobar

    if(abPruning):
        [pruned, score] = alphaBetaPickMoves(theGame, depth, notBadMoves)
        print("alpha-beta thinks the score is: ", score, " and its moves are: ", [x+1 for x in pruned])
        okMoves = pruned

    if(okMoves): # i.e., if there is a move that doesn't obviously suck
        weightedMoves = copy.deepcopy(okMoves)

        if(favorCenter):
            if(3 in okMoves):
                weightedMoves.extend([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])
            if(2 in okMoves):
                weightedMoves.extend([2,2,2,2,2,2,2,2,2,2])
            if(4 in okMoves):
                weightedMoves.extend([4,4,4,4,4,4,4,4,4])
            if(1 in okMoves):
                weightedMoves.extend([1,1,1])
            if(5 in okMoves):
                weightedMoves.extend([5,5,5])
#            print([x+1 for x in weightedMoves])

        a = random.choice(weightedMoves)
        if(len(okMoves)==1):
           print("There's only one choice that looks good to me: ", a+1)
        elif(len(okMoves) < len(theGame.legal_moves)):
            print("You're not giving me too many choices...  I'm torn between: ", [x+1 for x in okMoves], ".  I guess I'll pick ", a+1)
        else:
            print("Well, I don't see anything great to play, so I'll just go with: ", a+1)
        return a
    else: ## If we get here, the bot knows that it lost, and now it's just going to delay the inevitable
        a = random.choice(delay_inevitable(theGame, depth+1))
        if(copyCatMakesCurrentPlayerLose(theGame)):
            print("Looks like I lose to a copy cat strategy!  Hopefully you don't notice...  I play: ", a+1)
        else:
            print("Oh shit.  Looks like I lose quite soon.  Oh, well.  I'll just delay the inevitable to the extent possible: ", a+1)
        return a











## LEVEL 0 BOT
# This is a function for a really stupid bot
# It takes a board and just returns a legal move at random
def level0_bot_move(theGame, myTurn = True):
    a = random.choice(theGame.legal_moves)
    print("Now time for LEVEL 0 BOT to move!  It chooses: ", a+1)
    return a


## LEVEL 1 BOT
# This bot first checks to see if it can win in one move (if so, it plays this)
# Then it checks if you can win in one move (if so, it blocks)
# Otherwise, it plays randomly
def level1_bot_move(theGame, myTurn = True):
    return think_ahead_complicated(theGame, 1, True, False, myTurn)  # Only thinks about the immediate win/loss.  Plays towards the center


def level2_bot_move(theGame, myTurn = True):
    return think_ahead_complicated(theGame, 2, True, False, myTurn)  # Thinks ahead a bit, cares about playing towards the center

## LEVEL 3 BOT
#  It does tactics like an L2 bot, but when it moves randomly, it favors the center

def level3_bot_move(theGame, myTurn = True):
    return think_ahead_complicated(theGame, 2, True, True, myTurn) # FavorCenter = True;;   ownColumns = True


def level4_bot_move(theGame, myTurn = True):
    return think_ahead_complicated(theGame, 2, True, True, myTurn, True, True) # FavorCenter = True;;   ownColumns = True;    withZug = True!   considerThreats=True


def get_human_move(theGame):
    current_player = str((theGame.turn % 2)+1)
    print('Player %s, it is your turn.\n' % current_player)
    foo = theGame.threats()
#    print('Your threats are: ', [x for x in foo])
#    print('Their threats are: ', [y for y in theGame.threats(False)])
#    abTest(theGame, 2)
#    if(copyCatMakesCurrentPlayerLose(theGame)):    #check this to debug it
#        print("You will lose by copy cat")
    foo = movesThatDontLose(theGame, 2, True)
#    print("Moves that don't immediately suck are: ", [x+1 for x in foo])
    next_move = int(input('Enter a column (1-7) to play a move: '))-1
    while (next_move not in theGame.legal_moves):
        print('Please enter a valid command.')
        next_move = int(input('Enter a column (1-7) to play a move: '))-1
    return next_move






def level5_bot_move(theGame, myTurn = True):
#    abTest(theGame, 3)
    return think_ahead_complicated(theGame, 2, True, True, myTurn, True, True, True) # FavorCenter = True;;   ownColumns = True;    withZug = True!   considerThreats=True;  abPruning = True







#Below is a game of connect four with user inputs.

game = ConnectFour()


opponent = int(input('Who would you like to play? Type -1 for human, else type 0, 1, 2, 3, 4, or 5 for bots of various levels (0 easiest, 3 hardest, [4 and 5 experimental]): '))

if (opponent == -1):
    humanGoFirst = 1
else:
    humanGoFirst = int(input('Would you like to play first (1) or second (2)?: '))


if(humanGoFirst == 1):
    silly_print(game.board_state)
    next_move=get_human_move(game)
    game.play_chip(next_move)
    clear_output()


while(game.is_over == False):
    # We sort of start with the machine, but this actually doesn't super matter since the IF right about this WHILE loop inserts a human first turn just in case they ask for it
    # The game internally keep track of which turn it is, so we don't have to worry about parity or anything

    silly_print(game.board_state)
    if(opponent==0):
        next_move=level0_bot_move(game)
    elif(opponent==1):
        next_move=level1_bot_move(game)
    elif(opponent==2):
        next_move=level2_bot_move(game)
    elif(opponent==3):
        next_move=level3_bot_move(game)
    elif(opponent==4):
        next_move=level4_bot_move(game)
    elif(opponent==5):
        next_move=level5_bot_move(game)
    else:
        next_move=get_human_move(game)
    
    game.play_chip(next_move)
    clear_output()

    if(game.is_over == False): # Then we keep going, and now it's the human's turn
      silly_print(game.board_state)
      next_move=get_human_move(game)
      game.play_chip(next_move)
      clear_output()




if (game.is_won == True):
    print('The game is over; %s wins!' % game.winner)
elif (game.is_won == False and game.legal_moves.size == 0):
    print('The game has ended in a draw.')

print('The final position:\n')
silly_print(game.board_state)

game.reset()
