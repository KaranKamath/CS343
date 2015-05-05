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
    elif agentStr == 'smartoffensev2':
      return SmartOffenseAgentV2(index)
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

    # for e in enemies:
    #     import copy
    #     gsCopy = copy.deepcopy(gameState)
    #     self.pfilters[e] = inference.ParticleFilter(self.index, e, gameState.getInitialAgentPosition(e))
    #     self.pfilters[e].initialize(gsCopy)
    #
    for e in enemies:
        import copy
        gsCopy = copy.deepcopy(gameState)
        self.pfilters[e] = inference.ExactInference(self.index, e, gameState.getInitialAgentPosition(e))
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

  def floodCorrectPfilter(self, pos):
      for pfilter in self.pfilters.values():
          pfilter.floodIfPossible(pos)

  def updateParticleFilters(self, gameState):
    self.floodFiltersOnHistory(gameState)
    opponents = self.getOpponents(gameState)
    nds = gameState.getAgentDistances()
    for opponent in opponents:
        pfilter = self.pfilters[opponent]
        import copy
        gsCopy = copy.deepcopy(gameState)
        pfilter.observe(nds[opponent], gsCopy)
        pfilter.elapseTime(gsCopy)

  def floodFiltersOnHistory(self, gameState):
    prevState = self.getPreviousObservation()
    if prevState is not None:
        prevPos = prevState.getAgentState(self.index).getPosition()
        currPos = gameState.getAgentState(self.index).getPosition()
        dist = self.getMazeDistance(currPos, prevPos)
        if dist > 1:
            self.floodCorrectPfilter(prevPos)

        foodEatenPos = None
        if not gameState.isOnRedTeam(self.index):
            prevFood = prevState.getBlueFood().asList()
            currFood = gameState.getBlueFood().asList()
            foodEatenPos = set(prevFood) - set(currFood)
        else:
            prevFood = prevState.getRedFood().asList()
            currFood = gameState.getRedFood().asList()
            foodEatenPos = set(prevFood) - set(currFood)

        if len(foodEatenPos) > 0:
            self.floodCorrectPfilter(list(foodEatenPos)[0])

  def getEnemyLocationGuesses(self, gameState):
    opponents = self.getOpponents(gameState)
    locDict = {}
    for opponent in opponents:
        pfilter = self.pfilters[opponent]
        locDict[opponent] = pfilter.getBeliefDistribution().argMax()
        locDict[opponent] = (int(locDict[opponent][0]), int(locDict[opponent][1]))

    return locDict

class SmartOffenseAgent(IntelligentAgent):
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

        enemyLocs = self.getEnemyLocationGuesses(gameState)
        minEnemyDist = min([self.getMazeDistance(myPos, p) for p in enemyLocs.values()])

        # if minEnemyDist <= 3 and gameState.getAgentState(self.index).isPacman:
        #     features['ghostDistance'] = minEnemyDist
        # elif minEnemyDist <= 3 and not gameState.getAgentState(self.index).isPacman:
        #     features['ghostDistance'] = -minEnemyDist

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1, 'ghostDistance': 1}

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

class SmartOffenseAgentV2(IntelligentAgent):
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
          myPos = successor.getAgentState(self.index).getPosition()
          minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
          bottomMostFood = min(foodList, key = lambda x: x[1])
          features['distanceToFood'] = self.getMazeDistance(myPos, bottomMostFood)

        teamIndices = self.getTeam(gameState)
        teamIndices.remove(self.index)
        minTeamDist = 0

        if teamIndices is not None:
            teamDistances = [self.getMazeDistance(myPos, gameState.getAgentState(p).getPosition()) \
                               for p in teamIndices]
            minTeamDist = min(teamDistances)
            if minTeamDist <= 10 & gameState.getAgentState(self.index).isPacman:
                features['minTeamDistance'] = minTeamDist

        enemyLocs = self.getEnemyLocationGuesses(gameState)
        minEnemyDist = min([self.getMazeDistance(myPos, p) for p in enemyLocs.values()])

        myState = gameState.getAgentState(self.index)
        #
        # if myState.isPacman:
        #     if minEnemyDist < 3:
        #         features['ghostDistance'] = minEnemyDist
        #         print "evader"
        # else:
        #     if myState.scaredTimer == 0 and minEnemyDist < 3:
        #         features['ghostDistance'] = -minEnemyDist
        #         print "Hunter"

        capsules = self.getCapsules(gameState)
        if len(capsules):
            features['minCapsuleDist'] = min([self.getMazeDistance(myPos, c)\
                for c in capsules])

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1, 'ghostDistance': 10,
                'minCapsuleDist': -1}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        import searchAgents, search
        #allFoodProblem = searchAgents.FoodSearchProblem(gameState, self.index)
        anyFood = searchAgents.AnyFoodSearchProblem(gameState, self.index)
        searchActions = search.uniformCostSearch(anyFood)
        print searchActions
        return searchActions[0]

        # self.updateParticleFilters(gameState)
        # self.updateInferenceUI(gameState)
        #
        # actions = gameState.getLegalActions(self.index)
        #
        # # You can profile your evaluation time by uncommenting these lines
        # # start = time.time()
        # values = [self.evaluate(gameState, a) for a in actions]
        # # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        #
        # maxValue = max(values)
        # bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        #
        # return random.choice(bestActions)