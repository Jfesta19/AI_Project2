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

        #print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostScore = 0
        ghostPos = successorGameState.getGhostPositions()
        pX, pY = newPos

        for p in ghostPos:
            gX, gY = p
            ghostScore = abs(pX - gX) + abs(pY - gY)

        foodScore = len(newFood.asList()) #doesnt help for itermediate steps
        cFood = currentGameState.getFood().asList()
        fScore = 999999999999
        for f in cFood:
            fX, fY = f
            fD = abs(pX - fX) + abs(pY - fY)
            if fD < fScore:
                fScore = fD

        #10/(.1 + fScore)
        n = newFood.asList()
        if ghostScore > 1:
            return 20/(.01 + len(n)) + 10/(.1 + fScore)
        else:
            return 20/(.01 + len(n)) + 10/(.1 + fScore) - 1000

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

    MAX_VALUE = 999999.00
    MIN_VALUE = -999999.00
    bestAction = None

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
        amount = self.miniMax(gameState, 0)
        action = self.bestAction
        return action

    def miniMax(self, gameState, depth):
        if self.terminalTest(gameState):
            return self.utility(gameState)
        if self.cutoffTest(gameState, depth):
            return self.evaluationFunction(gameState)
        agent = depth % gameState.getNumAgents()
        if self.playerIsMax(gameState, depth):
            maxValue = self.MIN_VALUE

            actions = gameState.getLegalActions(0)
            bestAction = actions[0]
            for action in actions:
                stateValue = self.miniMax(gameState.generateSuccessor(agent, action), depth + 1)
                if stateValue > maxValue:
                    maxValue = stateValue
                    bestAction = action
            self.bestAction = bestAction
            return maxValue
        else:
            minValue = self.MAX_VALUE
            actions = gameState.getLegalActions(agent)
            for action in actions:
                stateValue = self.miniMax(gameState.generateSuccessor(agent, action), depth + 1)
                if stateValue < minValue:
                    minValue = stateValue
            return minValue


    def cutoffTest(self, gameState, depth):
        plies = depth / gameState.getNumAgents()

        if plies == self.depth:
            return True
        return False

    def terminalTest(self, gameState):
        if gameState.isWin() or gameState.isLose():
            return True
        return False

    def utility(self, gameState):
        return gameState.getScore()

    def playerIsMax(self, gameState, depth):
        if depth % gameState.getNumAgents()  == 0:
            return True
        return False

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    MAX_VALUE = 999999.00
    MIN_VALUE = -999999.00
    bestAction = None

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        amount = self.miniMax(gameState, 0, self.MIN_VALUE, self.MAX_VALUE)
        action = self.bestAction
        return action

    def miniMax(self, gameState, depth, alpha, beta):
        if self.terminalTest(gameState):
            return self.utility(gameState)
        if self.cutoffTest(gameState, depth):
            return self.evaluationFunction(gameState)
        agent = depth % gameState.getNumAgents()
        if self.playerIsMax(gameState, depth):
            maxValue = self.MIN_VALUE
            actions = gameState.getLegalActions(0)
            bestAction = actions[0]
            for action in actions:
                stateValue = self.miniMax(gameState.generateSuccessor(agent, action), depth + 1, alpha, beta)
                if stateValue > maxValue:
                    maxValue = stateValue
                    bestAction = action
                if maxValue > beta:
                    return maxValue
                alpha = max(alpha, maxValue)
            self.bestAction = bestAction
            return maxValue
        else:
            minValue = self.MAX_VALUE
            actions = gameState.getLegalActions(agent)
            for action in actions:
                stateValue = self.miniMax(gameState.generateSuccessor(agent, action), depth + 1, alpha, beta)
                if stateValue < minValue:
                    minValue = stateValue
                if minValue < alpha:
                    return minValue
                beta = min(beta, minValue)
            return minValue


    def cutoffTest(self, gameState, depth):
        plies = depth / gameState.getNumAgents()

        if plies == self.depth:
            return True
        return False

    def terminalTest(self, gameState):
        if gameState.isWin() or gameState.isLose():
            return True
        return False

    def utility(self, gameState):
        return gameState.getScore()

    def playerIsMax(self, gameState, depth):
        if depth % gameState.getNumAgents()  == 0:
            return True
        return False

class ExpectimaxAgent(MultiAgentSearchAgent):

        MAX_VALUE = 999999.00
        MIN_VALUE = -999999.00
        bestAction = None

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
            amount = self.miniMax(gameState, 0)
            action = self.bestAction
            return action

        def miniMax(self, gameState, depth):
            if self.terminalTest(gameState):
                return self.utility(gameState)
            if self.cutoffTest(gameState, depth):
                return self.evaluationFunction(gameState)
            agent = depth % gameState.getNumAgents()
            if self.playerIsMax(gameState, depth):
                maxValue = self.MIN_VALUE

                actions = gameState.getLegalActions(0)
                bestAction = actions[0]
                for action in actions:
                    stateValue = self.miniMax(gameState.generateSuccessor(agent, action), depth + 1)
                    if stateValue > maxValue:
                        maxValue = stateValue
                        bestAction = action
                self.bestAction = bestAction
                return maxValue
            else:
                actions = gameState.getLegalActions(agent)
                totalStateValue = 0
                numberOfActions = 0
                for action in actions:
                    numberOfActions = numberOfActions + 1
                    stateValue = self.miniMax(gameState.generateSuccessor(agent, action), depth + 1)
                    totalStateValue = totalStateValue + stateValue

                averageValue = totalStateValue * 1.0 / numberOfActions
                return averageValue


        def cutoffTest(self, gameState, depth):
            plies = depth / gameState.getNumAgents()

            if plies == self.depth:
                return True
            return False

        def terminalTest(self, gameState):
            if gameState.isWin() or gameState.isLose():
                return True
            return False

        def utility(self, gameState):
            return gameState.getScore()

        def playerIsMax(self, gameState, depth):
            if depth % gameState.getNumAgents()  == 0:
                return True
            return False


def betterEvaluationFunction(currentGameState):

    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    ghostScore = 0
    ghostPos = currentGameState.getGhostPositions()
    pX, pY = newPos

    for p in ghostPos:
        gX, gY = p
        ghostScore = abs(pX - gX) + abs(pY - gY)

    foodScore = len(newFood.asList()) #doesnt help for itermediate steps
    cFood = currentGameState.getFood().asList()
    fScore = 999999999999
    for f in cFood:
        fX, fY = f
        fD = abs(pX - fX) + abs(pY - fY)
        if fD < fScore:
            fScore = fD

    #10/(.1 + fScore)
    n = newFood.asList()
    if ghostScore > 2:
        return 20/(.01 + len(n)) + 10/(.1 + fScore+3)
    else:
        return 20/(.01 + len(n)) + 10/(.1 + fScore) - 1000

# Abbreviation
better = betterEvaluationFunction
