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
import sys

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
        if action == Directions.STOP:
          return -1000000
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        capsuleList = currentGameState.getCapsules()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        numAgents = currentGameState.getNumAgents()
        

        ## CHECK FOR NUMBER OF REMAINING FOOD PELLETS
        food_cnt_remaining = newFood.count()
        foodDistanceScalingFactor = 13
        foodDistances = [foodDistanceScalingFactor * util.manhattanDistance(newPos, foodPos) for foodPos in newFoodList]
        if foodDistances == []:
          minFoodDistance = 1000000
        else:
          minFoodDistance = min(foodDistances)
        capsuleDistances = [foodDistanceScalingFactor * util.manhattanDistance(newPos, capsulePos) for capsulePos in capsuleList]
        if capsuleDistances == []:
          minCapsuleDistance = 1000000
        else:
          minCapsuleDistance = min(capsuleDistances)
        ghostProperties = {}
        for ghostState in newGhostStates:
          ghostProperties[ghostState.getPosition()] = ghostState.scaredTimer
        closestGhostPos = None
        minGhostDistance = sys.maxint
        for ghostPos in ghostProperties.keys():
          dist = util.manhattanDistance(newPos, ghostPos)
          if dist < minGhostDistance:
            minGhostDistance = dist
            closestGhostPos = ghostPos
        #print ghostProperties
        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        ghostFeatureScale = 8
        foodFeatureScale = 10
        foodCntFeatureScale = 2
        capsuleFeatureScale = 7.5

        ghostFeature = 0
        closestGhostTimer = ghostProperties[closestGhostPos]
        if closestGhostTimer > 0:
          if minGhostDistance < 0.5 * closestGhostTimer:
            ghostFeature = ghostFeatureScale/float(minGhostDistance + 1)
        else:
          ghostFeature = -ghostFeatureScale/float(minGhostDistance + 1)
        evaluationValue = float(foodFeatureScale)/float(minFoodDistance + 1) - foodCntFeatureScale*food_cnt_remaining + float(capsuleFeatureScale)/float(minCapsuleDistance + 1) + ghostFeature
        #print action + ' ' + str(evaluationValue)
        return evaluationValue

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
    # (action, value)
    def value(self, gameState, action, numGhosts, depthSoFar, currentAgent, alpha, beta, q):
      # not sure about winning and losing states
      if gameState.isWin() or gameState.isLose():
        return (action, self.evaluationFunction(gameState))
      if depthSoFar == 0:
        return (action, self.evaluationFunction(gameState))
      if currentAgent == 0:
        return self.max_value(gameState, numGhosts, depthSoFar, currentAgent, alpha, beta, q)
      elif currentAgent != 0 and q == 1:
        return self.min_value(gameState, numGhosts, depthSoFar, currentAgent, alpha, beta, q)
      elif currentAgent != 0 and q == 2:
        return self.exp_value(gameState, numGhosts, depthSoFar, currentAgent, alpha, beta, q)

    def max_value(self, gameState, numGhosts, depthSoFar, currentAgent, alpha, beta, q):
      tmpValue = [None, -sys.maxint]
      count = 0
      for action in gameState.getLegalActions(0):
        count += 1
        successorGameState = gameState.generateSuccessor(0, action)
        successorValue = self.value(successorGameState, action, numGhosts, depthSoFar, currentAgent + 1, alpha, beta, q)
        if successorValue[1] > tmpValue[1]:
          tmpValue = (action, successorValue[1])
        if alpha != None and beta != None:
          if tmpValue[1] > beta:
            return tmpValue
          alpha = max(alpha, tmpValue[1])
      return tmpValue

    def min_value(self, gameState, numGhosts, depthSoFar, currentAgent, alpha, beta, q):
      tmpValue = [None, sys.maxint]
      for action in gameState.getLegalActions(currentAgent):
        successorGameState = gameState.generateSuccessor(currentAgent, action)
        if currentAgent == numGhosts:
          successorValue = self.value(successorGameState, action, numGhosts, depthSoFar - 1, 0, alpha, beta, q)
        else:
          successorValue = self.value(successorGameState, action, numGhosts, depthSoFar, currentAgent + 1, alpha, beta, q)
        if successorValue[1] < tmpValue[1]:
          tmpValue = (action, successorValue[1])
        if alpha != None and beta != None:
          if tmpValue[1] < alpha:
            return tmpValue
          beta = min(beta, tmpValue[1])
      return tmpValue

    def exp_value(self, gameState, numGhosts, depthSoFar, currentAgent, alpha, beta, q):
      tmpValue = [None, 0]
      legalActions = gameState.getLegalActions(currentAgent)
      for action in legalActions:
        successorGameState = gameState.generateSuccessor(currentAgent, action)
        if currentAgent == numGhosts:
          successorValue = self.value(successorGameState, action, numGhosts, depthSoFar - 1, 0, alpha, beta, q)
        else:
          successorValue = self.value(successorGameState, action, numGhosts, depthSoFar, currentAgent + 1, alpha, beta, q)
        tmpValue[1] += 1.0/float(len(legalActions)) * successorValue[1]
        tmpValue[0] = action
      return tmpValue

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        retVal = self.value(gameState, Directions.STOP, gameState.getNumAgents() - 1, self.depth, 0, None, None, 1)
        return retVal[0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        retVal = self.value(gameState, Directions.STOP, gameState.getNumAgents() - 1, self.depth, 0, -sys.maxint, sys.maxint, 1)
        return retVal[0]
        util.raiseNotDefined()

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
        retVal = self.value(gameState, Directions.STOP, gameState.getNumAgents() - 1, self.depth, 0, -sys.maxint, sys.maxint, 2)
        return retVal[0]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

