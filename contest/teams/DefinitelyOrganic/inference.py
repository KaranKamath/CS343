import util
import game

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, agentIndex, agentInitPosition):
        "Sets the ghost agent for later access"
        self.index = agentIndex
        self.initPosition = agentInitPosition
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return self.initPosition

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getAgentPosition(self.index) # The position you set

        def getDistribution(state):
            actions = state.getLegalActions(self.index)
            prob = 1.0 / len( actions )
            return [(action, prob ) for action in actions]

        actionDist = getDistribution(gameState)

        dist = util.Counter()
        for action, prob in actionDist:
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass

class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, trackingAgentIndex, agentIndex, agentInitPos, numParticles=300):
        InferenceModule.__init__(self, agentIndex, agentInitPos);
        self.setNumParticles(numParticles)
        self.trackingAgentIndex = trackingAgentIndex

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initializes a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        legalPositions = self.legalPositions

        from random import shuffle
        shuffle(legalPositions)

        excess = self.numParticles - len(legalPositions)

        if excess > 0:
            factor = (excess / len(legalPositions)) + 1
            legalPositions = legalPositions * factor

        self.particles = legalPositions[:self.numParticles]

    def observe(self, observation, gameState):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.getJailPosition()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeUniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattanDistance to calculate the
        distance between a particle and Pacman's position.
        """
        noisyDistance = observation

        myPosition = gameState.getAgentPosition(self.trackingAgentIndex)

        beliefs = self.getBeliefDistribution()
        nextBeliefs = util.Counter()

        if noisyDistance is None:
            self.particles = [self.getJailPosition()] * self.numParticles
            return

        else:
            for p in self.particles:
                trueDistance = util.manhattanDistance(p, myPosition)
                if gameState.getDistanceProb(trueDistance, noisyDistance) > 0:
                    nextBeliefs[p] = gameState.getDistanceProb(trueDistance, noisyDistance) * beliefs[p]

        nextBeliefs.normalize()

        if len(set(nextBeliefs.keys())) == 0:
            self.initializeUniformly(gameState)
        else:
            self.particles = util.nSample(nextBeliefs.values(), \
                                          nextBeliefs.keys(), self.numParticles)

    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        nextDist = util.Counter()
        beliefs = self.getBeliefDistribution()

        for p in set(self.particles):
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, p))

            for newPos, prob in newPosDist.items():
                nextDist[newPos] += prob * beliefs[p]

        nextDist.normalize()

        self.particles = util.nSample(nextDist.values(), nextDist.keys(), \
                                      self.numParticles)

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        beliefs = util.Counter()
        for particle in set(self.particles):
            beliefs[particle] = self.particles.count(particle) * 1.0 / self.numParticles
        return beliefs
