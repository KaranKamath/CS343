# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint
import inference

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class DefinitelyOrganicAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

  def choose(self, agentStr, index):
    if agentStr == 'keys':
      global NUM_KEYBOARD_AGENTS
      NUM_KEYBOARD_AGENTS += 1
      if NUM_KEYBOARD_AGENTS == 1:
        return keyboardAgents.KeyboardAgent(index)
      elif NUM_KEYBOARD_AGENTS == 2:
        return keyboardAgents.KeyboardAgent2(index)
      else:
        raise Exception('Max of two keyboard agents supported')
    elif agentStr == 'offense':
      return OffensiveReflexAgent(index)
    elif agentStr == 'defense':
      return DefensiveReflexAgent(index)
    elif agentStr == 'smartoffense':
      return SmartOffenseAgent(index)
    elif agentStr == 'smartdefense':
        return SmartDefenseAgent(index)
    elif agentStr == 'minimax':
        return MiniMaxAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

class AllOffenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)

  def getAgent(self, index):
    return OffensiveReflexAgent(index)

class OffenseDefenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)
    self.offense = False

  def getAgent(self, index):
    self.offense = not self.offense
    if self.offense:
      return OffensiveReflexAgent(index)
    else:
      return DefensiveReflexAgent(index)

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

class IntelligentAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    """
    self.red = gameState.isOnRedTeam(self.index)
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)

    # comment this out to forgo maze distance computation and use manhattan distances
    self.distancer.getMazeDistances()

    self.pfilters = {}
    enemies = self.getOpponents(gameState)

    for e in enemies:
        import copy
        gsCopy = copy.deepcopy(gameState)
        self.pfilters[e] = inference.ParticleFilter(self.index, e, gameState.getInitialAgentPosition(e))
        self.pfilters[e].initialize(gsCopy)

    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display

  def updateInferenceUI(self, gameState):
    dList = [util.Counter()] * (2 * len(self.getOpponents(gameState)))
    for k in self.pfilters.keys():
        dList[k] = self.pfilters[k].getBeliefDistribution()

    self.displayDistributionsOverPositions(dList)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    self.updateParticleFilters(gameState)
    self.updateInferenceUI(gameState)

    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def updateParticleFilters(self, gameState):
    opponents = self.getOpponents(gameState)
    nds = gameState.getAgentDistances()
    for opponent in opponents:
        pfilter = self.pfilters[opponent]
        import copy
        gsCopy = copy.deepcopy(gameState)
        pfilter.observe(nds[opponent], gsCopy)
        pfilter.elapseTime(gsCopy)

  def getEnemyLocationGuesses(self, gameState):
    opponents = self.getOpponents(gameState)
    locDict = {}
    for opponent in opponents:
        pfilter = self.pfilters[opponent]
        locDict[opponent] = pfilter.getBeliefDistribution().argMax()
        locDict[opponent] = (int(locDict[opponent][0]), int(locDict[opponent][1]))

    return locDict

class SmartOffenseAgent(IntelligentAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class SmartDefenseAgent(IntelligentAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    enemyLocs = self.getEnemyLocationGuesses(gameState)

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [(i, successor.getAgentState(i)) for i in self.getOpponents(successor)]
    invaders = [(i, a) for (i, a) in enemies if a.isPacman and a.getPosition() != None]

    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, enemyLocs[i]) for (i, a) in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

class MiniMaxAgent(IntelligentAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def isARoguePacman(self, state, agent, agentPos):
        initEnemyY = state.getInitialAgentPosition(agent)[1]
        initFriendlyY = 0
        if agent > 0:
            initFriendlyY = state.getInitialAgentPosition(agent-1)[1]
        else:
            initFriendlyY = state.getInitialAgentPosition(agent+1)[1]

        if initFriendlyY == 0 and agentPos[1] < initEnemyY / 2:
            return True

        if initEnemyY == 0 and agentPos[1] > initFriendlyY / 2:
            return True

        return False

    def evaluationFunction(self, state):
        return self.getScore(state)

    def getCopyOfGameStateWithLikelyOpponentPositions(self, gameState):
        import copy
        gsCopy = copy.deepcopy(gameState)

        enemyLocs = self.getEnemyLocationGuesses(gsCopy)

        #print 'get copy ',enemyLocs

        for enemy in enemyLocs.keys():
            conf = game.Configuration(enemyLocs[enemy], game.Directions.STOP)
            gsCopy.data.agentStates[enemy] = \
                game.AgentState(conf, self.isARoguePacman(gsCopy, enemy, enemyLocs[enemy]))

        return gsCopy

    def chooseAction(self, originalGameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.updateParticleFilters(originalGameState)

        gameState = self.getCopyOfGameStateWithLikelyOpponentPositions(originalGameState)

        self.depth = 2

        import sys

        def isMin(agent):
            return agent in self.getOpponents(gameState)

        def isMax(agent):
            return agent not in self.getOpponents(gameState)

        def isTerminal(state):
            return state.getBlueFood().count() <= 2 or \
                   state.getRedFood().count() <= 2

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
            if agent == (len(self.getOpponents(state)) * 2):
                depth += 1
                agent = 0

            if depth > self.depth or isTerminal(state):
                return self.evaluationFunction(state), None

            if isMax(agent):
                return maxval(state, agent, depth, alpha, beta)

            if isMin(agent):
                return minval(state, agent, depth, alpha, beta)

        return value(gameState, self.index, 1, -sys.maxint-1, sys.maxint)[1]