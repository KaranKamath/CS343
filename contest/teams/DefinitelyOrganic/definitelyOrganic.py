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

  def __init__(self, isRed, first='smartoffensev2', second='smartdefensev1', rest='smartoffensev2'):
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
    elif agentStr == 'smartoffensev2':
      return SmartOffenseAgentV2(index)
    elif agentStr == 'smartoffensev3':
      return SmartOffenseAgentV3(index)
    elif agentStr == 'smartdefensev1':
      return SmartDefenseAgentV1(index)
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

from game import Actions
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
    self.nextFood = None
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
    import copy
    pfilters = copy.deepcopy(self.pfilters)

    for k in pfilters.keys():
        beliefs = pfilters[k].getBeliefDistribution()
        dList[k] = beliefs

    self.displayDistributionsOverPositions(dList)

  def getNextPosition(self, currentPosition, action):
    x, y = currentPosition
    dx, dy = Actions.directionToVector(action)
    return (int(x + dx), int(y + dy))

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

  def getLegalTerritory(self, position, walls):
    surroundingCells = Actions.getLegalNeighbors(position, walls)
    surroundingCells.append(position)
    return surroundingCells

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

import searchAgents, search

class SmartOffenseAgentV2(IntelligentAgent):
    def getFeatures(self, gameState, action):

        features = util.Counter()
        successorState = self.getSuccessor(gameState, action)
        currentFood = self.getFood(gameState).asList()
        currentWalls = gameState.getWalls()
        currentAgentState = gameState.getAgentState(self.index)
        currentAgentPosition = currentAgentState.getPosition()
        nextAgentState = successorState.getAgentState(self.index)
        nextAgentPosition = self.getNextPosition(currentAgentState.getPosition(), action)

        isNextPacMan = nextAgentState.isPacman

        currentTeamPositions = [gameState.getAgentState(t).getPosition() \
                                for t in self.getTeam(gameState)]

        currentEnemyStates = [gameState.getAgentState(e)\
                                 for e in self.getOpponents(gameState)]

        enemyPacmenInRange = [e for e in currentEnemyStates \
                              if e.isPacman and e.getPosition() is not None]

        enemyGhostsInRange = [e for e in currentEnemyStates \
                              if not e.isPacman and e.getPosition() is not None]

        gridHeight = currentWalls.height

        teamSize = len(currentTeamPositions)

        agentFoodIndex = ((gridHeight + self.index) / 2) % teamSize
        foodDivider = gridHeight / teamSize

        for e in enemyGhostsInRange:
            enemyGhostTerritory = self.getLegalTerritory(e.getPosition(), currentWalls)
            if nextAgentPosition in enemyGhostTerritory:
                if e.scaredTimer:
                    features['enemyScaredGhost'] += 1
                else:
                    features['enemyGhost'] += 1

        for e in enemyPacmenInRange:
            enemyPacmanTerritory = self.getLegalTerritory(e.getPosition(), currentWalls)
            if nextAgentPosition in enemyPacmanTerritory:
                if currentAgentState.scaredTimer:
                    features['enemyScaryPacman'] += 1
                else:
                    features['enemyPacman'] += 1

        if (not features['enemyGhost']) and (not features['enemyScaryPacman']):
            if nextAgentPosition in currentFood:
                features['foodNext'] = 1

            agentFood = [f for f in currentFood if f[1] / foodDivider == agentFoodIndex]
            foodTarget = None
            if len(agentFood):
                features['minFoodDistance'] = min(\
                    [self.getMazeDistance(nextAgentPosition, x) for x in agentFood])
            else:
                features['minFoodDistance'] = min(\
                    [self.getMazeDistance(nextAgentPosition, x) for x in currentFood])

            features['minFoodDistance'] * 1.0 / (currentWalls.height + currentWalls.width)
        else:
            features['degreeOfFreedom'] = len(self.getLegalTerritory(nextAgentPosition, currentWalls))

        if action == Directions.STOP:
            features['stopPenalty'] = 1

        capsules = self.getCapsules(gameState)
        if nextAgentPosition in capsules:
            features['capsule'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'enemyGhost': -20, 'enemyScaredGhost': 5, 'enemyScaryPacman': -10,
                'enemyPacman': 20, 'foodNext': 5, 'minFoodDistance':-1, 'capsule': 10,
                'stopPenalty': -100, 'degreesOfFreedom': 10}

    def getAgentFood(self, gameState):
        gridHeight = gameState.getWalls().height
        teamSize = len(self.getTeam(gameState))
        currentFood = self.getFood(gameState).asList()
        agentFoodIndex = ((gridHeight + self.index) / 2) % teamSize
        foodDivider = gridHeight / teamSize
        return [f for f in currentFood if f[1] / foodDivider == agentFoodIndex]

    def isClosestToCapsule(self, gameState):
        capsules = self.getCapsules(gameState)

        if len(capsules) == 0:
            return False

        myPos = gameState.getAgentState(self.index).getPosition()
        teamPos = [gameState.getAgentState(i).getPosition() for i in self.getTeam(gameState)]
        teamPos.remove(myPos)

        myMinToCap = min([self.getMazeDistance(myPos, cPos) for cPos in capsules])

        for t in teamPos:
            tMinToCap = min([self.getMazeDistance(myPos, cPos) for cPos in capsules])
            if tMinToCap < myMinToCap:
                return False

        return True

    def isTeamPacman(self, gameState):
        team = self.getTeam(gameState)
        team.remove(self.index)
        for t in team:
            if not gameState.getAgentState(t).isPacman:
                return False
        return True

    def getClosestFoodLocation(self, gameState):
        food = self.getAgentFood(gameState)
        currentFood = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        agentFoodDistances = [(f, self.getMazeDistance(myPos, f)) for f in food]
        teamFoodDistances = [(f, self.getMazeDistance(myPos, f)) for f in currentFood]
        capsuleDistances = [(c, self.getMazeDistance(myPos, c)) for c in self.getCapsules(gameState)]

        foodDistances = agentFoodDistances
        if len(agentFoodDistances) == 0:
            if len(capsuleDistances) > 0 and self.isClosestToCapsule(gameState) and \
                    self.isTeamPacman(gameState):
                foodDistances = capsuleDistances
            else:
                foodDistances = teamFoodDistances

        if min(teamFoodDistances, key=lambda x: x[1])[1] == 1 or not self.isTeamPacman(gameState):
            return min(teamFoodDistances, key=lambda x: x[1])[0]

        return min(foodDistances, key=lambda x: x[1])[0]

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        closestFood = self.getClosestFoodLocation(gameState)
        positionSearchProblem = searchAgents.PositionSearchProblem(\
            gameState, self.index, food=self.getAgentFood(gameState), goal=closestFood)
        searchActions = search.uniformCostSearch(positionSearchProblem)

        searchActionFeatures = self.getFeatures(gameState, searchActions[0])
        if searchActionFeatures['enemyGhost'] or searchActionFeatures['enemyScaryPacman']\
                or searchActionFeatures['enemyScaredGhost'] or searchActionFeatures['enemyPacman']:
            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)

            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return random.choice(bestActions)

        return searchActions[0]

class SmartOffenseAgentV3(IntelligentAgent):

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successorState = self.getSuccessor(gameState, action)
        currentFood = self.getFood(gameState).asList()
        currentWalls = gameState.getWalls()
        currentAgentState = gameState.getAgentState(self.index)
        currentAgentPosition = currentAgentState.getPosition()
        nextAgentState = successorState.getAgentState(self.index)
        nextAgentPosition = self.getNextPosition(currentAgentState.getPosition(), action)

        isNextPacMan = nextAgentState.isPacman

        currentTeamPositions = [gameState.getAgentState(t).getPosition() \
                                for t in self.getTeam(gameState)]

        currentEnemyStates = [gameState.getAgentState(e)\
                                 for e in self.getOpponents(gameState)]

        enemyPacmenInRange = [e for e in currentEnemyStates \
                              if e.isPacman and e.getPosition() is not None]

        enemyGhostsInRange = [e for e in currentEnemyStates \
                              if not e.isPacman and e.getPosition() is not None]

        gridHeight = currentWalls.height

        teamSize = len(currentTeamPositions)

        agentFoodIndex = ((gridHeight + self.index) / 2) % teamSize
        foodDivider = gridHeight / teamSize

        for e in enemyGhostsInRange:
            enemyGhostTerritory = self.getLegalTerritory(e.getPosition(), currentWalls)
            if nextAgentPosition in enemyGhostTerritory:
                if e.scaredTimer:
                    features['enemyScaredGhost'] += 1
                else:
                    features['enemyGhost'] += 1

        for e in enemyPacmenInRange:
            enemyPacmanTerritory = self.getLegalTerritory(e.getPosition(), currentWalls)
            if nextAgentPosition in enemyPacmanTerritory:
                if currentAgentState.scaredTimer:
                    features['enemyScaryPacman'] += 1
                else:
                    features['enemyPacman'] += 1

        if (not features['enemyGhost']) and (not features['enemyScaryPacman']):
            if nextAgentPosition in currentFood:
                features['foodNext'] = 1

            agentFood = [f for f in currentFood if f[1] / foodDivider == agentFoodIndex]
            foodTarget = None
            if len(agentFood):
                features['minFoodDistance'] = min(\
                    [self.getMazeDistance(nextAgentPosition, x) for x in agentFood])
            else:
                features['minFoodDistance'] = min(\
                    [self.getMazeDistance(nextAgentPosition, x) for x in currentFood])

            features['minFoodDistance'] * 1.0 / (currentWalls.height + currentWalls.width)
        else:
            features['degreeOfFreedom'] = len(self.getLegalTerritory(nextAgentPosition, currentWalls))

        if action == Directions.STOP:
            features['stopPenalty'] = 1

        capsules = self.getCapsules(gameState)
        if nextAgentPosition in capsules:
            features['capsule'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'enemyGhost': -20, 'enemyScaredGhost': 5, 'enemyScaryPacman': -10,
                'enemyPacman': 20, 'foodNext': 5, 'minFoodDistance':-1, 'capsule': 10,
                'stopPenalty': -100, 'degreesOfFreedom': 10}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        #
        # # You can profile your evaluation time by uncommenting these lines
        # # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        #
        maxValue = max(values)

        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        #
        return random.choice(bestActions)

class SmartDefenseAgentV1(IntelligentAgent):
    def getFeatures(self, gameState, action):

        successor = self.getSuccessor(gameState, action)

        features = util.Counter()
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

        successorState = self.getSuccessor(gameState, action)
        currentFood = self.getFood(gameState).asList()
        currentWalls = gameState.getWalls()
        currentAgentState = gameState.getAgentState(self.index)
        currentAgentPosition = currentAgentState.getPosition()
        nextAgentState = successorState.getAgentState(self.index)
        nextAgentPosition = self.getNextPosition(currentAgentState.getPosition(), action)

        isNextPacMan = nextAgentState.isPacman

        currentTeamPositions = [gameState.getAgentState(t).getPosition() \
                                for t in self.getTeam(gameState)]

        currentEnemyStates = [gameState.getAgentState(e)\
                                 for e in self.getOpponents(gameState)]

        enemyPacmenInRange = [e for e in currentEnemyStates \
                              if e.isPacman and e.getPosition() is not None]

        enemyGhostsInRange = [e for e in currentEnemyStates \
                              if not e.isPacman and e.getPosition() is not None]

        gridHeight = currentWalls.height

        teamSize = len(currentTeamPositions)

        agentFoodIndex = ((gridHeight + self.index) / 2) % teamSize
        foodDivider = gridHeight / teamSize

        for e in enemyGhostsInRange:
            enemyGhostTerritory = self.getLegalTerritory(e.getPosition(), currentWalls)
            if nextAgentPosition in enemyGhostTerritory:
                if e.scaredTimer:
                    features['enemyScaredGhost'] += 1
                else:
                    features['enemyGhost'] += 1

        for e in enemyPacmenInRange:
            enemyPacmanTerritory = self.getLegalTerritory(e.getPosition(), currentWalls)
            if nextAgentPosition in enemyPacmanTerritory:
                if currentAgentState.scaredTimer:
                    features['enemyScaryPacman'] += 1
                else:
                    features['enemyPacman'] += 1

        if (not features['enemyGhost']) and (not features['enemyScaryPacman']):
            if nextAgentPosition in currentFood:
                features['foodNext'] = 1

            agentFood = [f for f in currentFood if f[1] / foodDivider == agentFoodIndex]
            foodTarget = None
            if len(agentFood):
                features['minFoodDistance'] = min(\
                    [self.getMazeDistance(nextAgentPosition, x) for x in agentFood])
            else:
                features['minFoodDistance'] = min(\
                    [self.getMazeDistance(nextAgentPosition, x) for x in currentFood])

            features['minFoodDistance'] * 1.0 / (currentWalls.height + currentWalls.width)
        else:
            features['degreeOfFreedom'] = len(self.getLegalTerritory(nextAgentPosition, currentWalls))

        if action == Directions.STOP:
            features['stopPenalty'] = 1

        capsules = self.getCapsules(gameState)
        if nextAgentPosition in capsules:
            features['capsule'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'enemyGhost': -20, 'enemyScaredGhost': 5, 'enemyScaryPacman': -10,
                'enemyPacman': 20, 'stopPenalty': -100, 'degreesOfFreedom': 10, 'invaderDistance': -1}

    def getAgentFood(self, gameState):
        gridHeight = gameState.getWalls().height
        teamSize = len(self.getTeam(gameState))
        currentFood = self.getFood(gameState).asList()
        agentFoodIndex = ((gridHeight + self.index) / 2) % teamSize
        foodDivider = gridHeight / teamSize
        return [f for f in currentFood if f[1] / foodDivider == agentFoodIndex]

    def isClosestToCapsule(self, gameState):
        capsules = self.getCapsules(gameState)

        if len(capsules) == 0:
            return False

        myPos = gameState.getAgentState(self.index).getPosition()
        teamPos = [gameState.getAgentState(i).getPosition() for i in self.getTeam(gameState)]
        teamPos.remove(myPos)

        myMinToCap = min([self.getMazeDistance(myPos, cPos) for cPos in capsules])

        for t in teamPos:
            tMinToCap = min([self.getMazeDistance(myPos, cPos) for cPos in capsules])
            if tMinToCap < myMinToCap:
                return False

        return True

    def isTeamPacman(self, gameState):
        team = self.getTeam(gameState)
        team.remove(self.index)
        for t in team:
            if not gameState.getAgentState(t).isPacman:
                return False
        return True

    def getMissingFoodLocation(self, gameState, missingFood):
        myPos = gameState.getAgentState(self.index).getPosition()
        missingFoodDistances = [(f, self.getMazeDistance(myPos, f)) for f in missingFood]

        return min(missingFoodDistances, key=lambda x: x[1])[0]

    def getNextTargetFood(self, gameState, closestFood):
        enemyFood = list(set(gameState.getBlueFood().asList()).union(set(gameState.getRedFood().asList())) \
                         - set(self.getFood(gameState).asList()))
        enemyPos = closestFood
        enemyFoodDistances = [(f, self.getMazeDistance(enemyPos, f)) for f in enemyFood]

        return min(enemyFoodDistances, key=lambda x: x[1])[0]

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        prevGameState = self.getPreviousObservation()

        if prevGameState is not None or self.nextFood is not None:
            missingFood = list(set(self.getFood(prevGameState).asList()) - set(self.getFood(gameState).asList()))

            if not len(missingFood) == 0:
                closestFood = self.getMissingFoodLocation(gameState, missingFood)
                nextFood = self.getNextTargetFood(gameState, closestFood)
                self.nextFood = nextFood

            if self.nextFood is not None:
                positionSearchProblem = searchAgents.PositionSearchProblem(\
                    gameState, self.index, goal=self.nextFood)
                searchActions = search.uniformCostSearch(positionSearchProblem)
                if len(searchActions) > 0:
                    searchActionFeatures = self.getFeatures(gameState, searchActions[0])
                    if not searchActionFeatures['enemyScaryPacman']:
                        return searchActions[0]
                    
            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)

            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return random.choice(bestActions)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)

        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)
