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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = currentGameState.getGhostPositions()

        gridDist = newFood.width + newFood.height
        gridSize = newFood.width * newFood.height

        nearestFoodDistance = min([util.manhattanDistance(newPos, foodLoc) for foodLoc in oldFood.asList()])
        distancesToGhosts = [util.manhattanDistance(newPos, gPos) for gPos in ghostPositions]
        minDistToGhost = min(distancesToGhosts)
        numFood = len(newFood.asList())

        foodDistFactor = (gridDist * 1.0 / (nearestFoodDistance + 1))

        foodNumFactor = (gridSize * 1.0 / (numFood + 1))

        ghostFactor = ((minDistToGhost * 1.0) + 1) / gridSize

        evalFunc = foodDistFactor * foodNumFactor * ghostFactor

        if (minDistToGhost < 3):
            return ghostFactor

        return evalFunc

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        import sys

        def isMin(agent):
            return agent > 0

        def isMax(agent):
            return agent == 0

        def isTerminal(state):
            return state.isWin() or state.isLose()

        def minval(state, actionUtils):
            v = sys.maxint, None
            v = min([v] + actionUtils, key=lambda x: x[0])
            return v

        def maxval(state, actionUtils):
            v = -sys.maxint-1, None
            v = max([v] + actionUtils, key=lambda x: x[0])
            return v

        def value(state, agent, depth):
            if agent == state.getNumAgents():
                depth += 1
                agent = 0

            if depth > self.depth or isTerminal(state):
                return self.evaluationFunction(state), None

            actions = state.getLegalActions(agent)
            actionUtils = map(lambda x: (value(state.generateSuccessor(agent, x), agent + 1, depth)[0], x), actions)
            if isMax(agent):
                return maxval(state, actionUtils)

            if isMin(agent):
                return minval(state, actionUtils)

        return value(gameState, 0, 1)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        import sys

        def isMin(agent):
            return agent > 0

        def isMax(agent):
            return agent == 0

        def isTerminal(state):
            return state.isWin() or state.isLose()

        def minval(state, agent, depth, alpha, beta):
            v = (sys.maxint, None)
            actions = state.getLegalActions(agent)
            for a in actions:
                actionUtil = value(state.generateSuccessor(agent, a), agent + 1, depth, alpha, beta)
                v = min([v, (actionUtil[0], a)], key=lambda x: x[0])
                if v[0] < alpha:
                    return v
                beta = min([beta, v[0]])
            return v

        def maxval(state, agent, depth, alpha, beta):
            v = (-sys.maxint-1, None)
            actions = state.getLegalActions(agent)
            for a in actions:
                actionUtil = value(state.generateSuccessor(agent, a), agent + 1, depth, alpha, beta)
                v = max([v, (actionUtil[0], a)], key=lambda x: x[0])
                if v[0] > beta:
                    return v
                alpha = max([alpha, v[0]])
            return v

        def value(state, agent, depth, alpha, beta):
            if agent == state.getNumAgents():
                depth += 1
                agent = 0

            if depth > self.depth or isTerminal(state):
                return self.evaluationFunction(state), None

            if isMax(agent):
                return maxval(state, agent, depth, alpha, beta)

            if isMin(agent):
                return minval(state, agent, depth, alpha, beta)

        return value(gameState, 0, 1, -sys.maxint-1, sys.maxint)[1]

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
        import sys

        def isExp(agent):
            return agent > 0

        def isMax(agent):
            return agent == 0

        def isTerminal(state):
            return state.isWin() or state.isLose()

        def expval(state, actionUtils):
            expectedVal = reduce(lambda x, y: x + y, [au[0] for au in actionUtils]) * 1.0 / len(actionUtils)
            return expectedVal, None

        def maxval(state, actionUtils):
            v = -sys.maxint-1, None
            v = max([v] + actionUtils, key=lambda x: x[0])
            return v

        def value(state, agent, depth):
            if agent == state.getNumAgents():
                depth += 1
                agent = 0

            if depth > self.depth or isTerminal(state):
                return self.evaluationFunction(state), None

            actions = state.getLegalActions(agent)
            actionUtils = map(lambda x: (value(state.generateSuccessor(agent, x), agent + 1, depth)[0], x), actions)

            if isMax(agent):
                return maxval(state, actionUtils)

            if isExp(agent):
                return expval(state, actionUtils)

        return value(gameState, 0, 1)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Eval function crafted using factors that affect state value,
                    determining whether they are directly or inversely proportional,
                    and then optimizing powers to which they must be raised via trials

        Factors Added In The Denominator (Inverse Relationship):
            1. Amount of food left (Inversely proportional to 10th power)
            2. Distance to nearest food
            3. Number of capsules left
            4. Distance to nearest capsule
        Factors in Numerator (Direct Relationship):
            1. Distance to nearest ghost

        Special Cases:
            1. Distance to ghost drops below 2: Flee (Negative Utility)
            2. Ghosts are scared: Chase (Eval function only depends on distance to ghost)
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    gridDist = food.width + food.height

    nearestFoodDistance = min([util.manhattanDistance(pos, foodLoc) for foodLoc in food.asList()]) if food.asList() else 0
    distancesToGhosts = [util.manhattanDistance(pos, gPos) for gPos in ghostPositions]
    minDistToGhost = min(distancesToGhosts)
    numFood = len(food.asList())
    capsules = currentGameState.getCapsules()

    minCapsuleDist = min([util.manhattanDistance(pos, capPos) for capPos in capsules]) if capsules else 0
    numCapsules = len(capsules)

    if scaredTimes != [0]:
        return gridDist / (minDistToGhost + 1)
    if minDistToGhost < 2:
        return -1

    return ((minDistToGhost ** 0.5) * 1.0 / gridDist) / (0.1 + minCapsuleDist + nearestFoodDistance + numCapsules + (numFood ** 10.0))

# Abbreviation
better = betterEvaluationFunction

