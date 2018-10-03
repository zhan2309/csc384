# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # First, we initialize the new_food_list as the list that contains all food pos and also the current_position
        # since we will compare the current_posision and new_position to distinguish if we move towards a food.
        new_food_list = newFood.asList()
        score = 0
        current_pos = currentGameState.getPacmanPosition()
        i = 0
        # we need to avoid the ghost, therefore we have to make the evaluation score larger than eating food score.
        while i < len(newGhostStates):
            distance = manhattanDistance(newPos, newGhostStates[i].getPosition())
            if distance < 2:
                # When pacman eats a power pit, we increase the score as high as possible towards ghost; OtherWise, we
                # decrease the score as lower as possible towards ghost in order to avoid the ghosts.
                if newGhostStates[i].scaredTimer:
                    score = distance + 9999999
                else:
                    score = distance - 9999999
            i += 1
        # if ghosts are in scaredTime, we increase the score by it's left scaredTime.
        score += newScaredTimes[0]
        # Set both distance to food to infinite.
        min_dis_new = float("inf")
        min_dis_current = float("inf")
        while new_food_list != []:
            food = new_food_list.pop(0)
            if manhattanDistance(food, newPos) < min_dis_new:
                min_dis_new = manhattanDistance(food, newPos)
            if manhattanDistance(food, current_pos) < min_dis_current:
                min_dis_current = manhattanDistance(food, current_pos)
        # The pacman move towards food, therefore we need to add socre on it; OtherWise, we minus the evaluation score.
        if min_dis_new - min_dis_current < 0:
            score += 99999
        else:
            score -= 99999
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # build a minmax tree
        move, value = self.minmaxtree(0, 0, gameState)
        return move

    def minmaxtree(self, index, depth, gameState):
        # set best move to none first
        bs_move = None
        # check reach leaf or not
        if (depth == self.depth or gameState.isWin() or gameState.isLose()):
            return bs_move, self.evaluationFunction(gameState)
        # check pacman turn or ghost turn
        if (index == 0):
            value = -float("inf")
        else:
            value = float("inf")
        # if it is the last agent, move to next depth and set new index to 0
        if (index + 1 == gameState.getNumAgents()):
            new_index = 0
            depth += 1
        else:
            new_index = index + 1
        for act in gameState.getLegalActions(index):
            new_state = gameState.generateSuccessor(index, act)
            # build min max tree recursively
            new_move, new_value = self.minmaxtree(new_index, depth, new_state)
            if (index == 0 and value < new_value):
                value = new_value
                bs_move = act
            elif (index != 0 and value > new_value):
                value = new_value
                bs_move = act
        return bs_move, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # build alpha beta tree
        move, value = self.alphabetatree(-float("inf"), float("inf"), 0, 0, gameState)
        return move

    def alphabetatree(self, alpha, beta, index, depth, gameState):
        # set best move to none first
        bs_move = None
        # check reach leaf or not
        if (depth == self.depth or gameState.isWin() or gameState.isLose()):
            return bs_move, self.evaluationFunction(gameState)
        # check pacman turn or ghost turn
        if (index == 0):
            value = -float("inf")
        else:
            value = float("inf")
        # if it is the last agent, move to next depth and set new index to 0
        if (index + 1 == gameState.getNumAgents()):
            new_index = 0
            depth += 1
        else:
            new_index = index + 1
        for act in gameState.getLegalActions(index):
            new_state = gameState.generateSuccessor(index, act)
            # build alpha beta tree recursively
            new_move, new_value = self.alphabetatree(alpha, beta, new_index, depth, new_state)
            if (index == 0):
                if (value < new_value):
                    value = new_value
                    bs_move = act
                # check condition
                if (value >= beta):
                    return bs_move, value
                alpha = max(alpha, value)
            else:
                if (value > new_value):
                    value = new_value
                    bs_move = act
                # check condition
                if (value <= alpha):
                    return bs_move, value
                beta = min(beta, value)
        return bs_move, value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectMaxtree(index, depth, gameState):
            bs_move = None
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return bs_move, self.evaluationFunction(gameState)
            if index == 0:
                value = -float("inf")
            else:
                value = 0
            if index + 1 == gameState.getNumAgents():
                new_index = 0
                depth += 1
            else:
                new_index = index + 1
            for act in gameState.getLegalActions(index):
                new_state = gameState.generateSuccessor(index, act)
                new_move, new_value = expectMaxtree(new_index, depth, new_state)
                if index == 0 and value < new_value:
                    value = new_value
                    bs_move = act
                if index != 0:
                    length = len(gameState.getLegalActions(index))
                    value += float(new_value) / length
            return bs_move, value

        move, value = expectMaxtree(0, 0, gameState)
        return move


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    current_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    min_food_dis = float("inf")
    for food_pos in food_list:
        if manhattanDistance(food_pos, current_pos) < min_food_dis:
            min_food_dis = manhattanDistance(food_pos, current_pos)

    ghost_list = currentGameState.getGhostPositions()
    min_ghost_dis = float("inf")
    for ghost_pos in ghost_list:
        if manhattanDistance(ghost_pos, current_pos) < min_ghost_dis:
            min_ghost_dis = manhattanDistance(ghost_pos, current_pos)

    power_pit_list = currentGameState.getCapsules()
    if (len(power_pit_list) == 0):
        min_power_pit_dis = 0
    else:
        min_power_pit_dis = float("inf")
        for power_pit in power_pit_list:
            if manhattanDistance(power_pit, current_pos) < min_power_pit_dis:
                min_power_pit_dis = manhattanDistance(power_pit, current_pos)

    score = -(min_food_dis) * 0.6 + (min_ghost_dis) * 0.2 - (min_power_pit_dis) * 0.2 + currentGameState.getScore()

    if currentGameState.isWin():
        score = float("inf")
    if currentGameState.isLose():
        score = - float("inf")

    return score


# Abbreviation
better = betterEvaluationFunction
