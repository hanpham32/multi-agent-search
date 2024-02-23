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
import random
import util

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newPos = successorGameState.getPacmanPosition()  # new pacman pos
        newFood = successorGameState.getFood()  # remaining food
        newGhostStates = successorGameState.getGhostStates()  # a list
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  # a list

        print("TYPES:")
        print("newGhostStates: " + str(type(newGhostStates)))
        print("newScaredTimes: " + str(type(newScaredTimes)))

        print("VALUES:")
        print("newGhostStates: " + str(newGhostStates))
        print("newScaredTimes: " + str(newScaredTimes))

        "*** YOUR CODE HERE ***"
        # # successorGameState.getScore() + rewards associated with the state + penalty associated with the state
        score = successorGameState.getScore()

        # Calculate distance to nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 1.0 / min(foodDistances)  # to get closer food

        print("newGhostPos: " + str(newGhostPositions))

        # Calculate score to avoid ghosts
        ghostDistances = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPositions]
        print(ghostDistances)
        if ghostDistances:
            if min(ghostDistances) == 0.0:
                score -= 3.0
            else:
                score -= 2.0 / min(ghostDistances)

        # Calculate score to eat score ghost
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            if scaredTime > 0:
                distanceToScaredGhost = manhattanDistance(newPos, ghostState.getPosition())
                if distanceToScaredGhost > 0:
                    # Encourage Pacman to chase scared ghosts, more so if they are closer
                    score += 2.0 / distanceToScaredGhost
        # for scaredTime in newScaredTimes:
        #     if scaredTime > 0.0:
        #         score += 2.0 / scaredTime
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        result = self.get_value(gameState, 0, 0)

        # Return the action from result
        return result[1]

    def get_value(self, gameState, index, depth):

        # if Terminal states --> return scores, action
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return gameState.getScore(), ""

        # Max Agent: Pacman has index = 0
        if index == 0:
            return self.get_max(gameState, index, depth)

        # Min Agent: Ghost has index > 0
        else:
            return self.get_min(gameState, index, depth)
    # Max Agent
    def get_max(self, gameState, index, depth):

        legalMoves = gameState.getLegalActions(index)
        max_val = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current = self.get_value(successor, successor_index, successor_depth)[0]

            if current > max_val:
                max_val = current
                max_action = action

        return max_val, max_action
    # Min Agent
    def get_min(self, gameState, index, depth):

        legalMoves = gameState.getLegalActions(index)
        min_val = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current = self.get_value(successor, successor_index, successor_depth)[0]

            if current < min_val:
                min_val = current
                min_action = action

        return min_val, min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        result = self.get_max(gameState, 0, 0, -float("inf"), float("inf"))[0]

        return result
    # get value
    def alphabeta(self, gameState, index, depth, alpha, beta):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.get_max(gameState, index, depth, alpha, beta)[1]
        else:
            return self.get_min(gameState, index, depth, alpha, beta)[1]
    #Maxiumizer
    def get_max(self, gameState, index, depth, alpha, beta):
        bestAction = ("max", -float("inf"))
        for action in gameState.getLegalActions(index):
            successor = (action, self.alphabeta(gameState.generateSuccessor(index, action),
                                                (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            bestAction = max(bestAction, successor, key=lambda x: x[1])

            if bestAction[1] > beta:
                return bestAction
            else:
                alpha = max(alpha, bestAction[1])

        return bestAction
    #Minimizer
    def get_min(self, gameState, index, depth, alpha, beta):
        bestAction = ("min", float("inf"))
        for action in gameState.getLegalActions(index):
            successor = (action, self.alphabeta(gameState.generateSuccessor(index, action),
                                                (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            bestAction = min(bestAction, successor, key=lambda x: x[1])

            # Prunning
            if bestAction[1] < alpha:
                return bestAction
            else:
                beta = min(beta, bestAction[1])

        return bestAction


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
        # calling expectimax with the depth we are going to investigate
        Depth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", Depth, 0)[0]
        #Computes the action to take using the expectimax algorithm.
    def expectimax(self, gameState, action, depth, index):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        if index == 0:
            return self.get_max(gameState, action, depth, index)
        else:
            return self.expect_val(gameState, action, depth, index)
    #maxumizer
    def get_max(self, gameState, action, depth, agentIndex):
        bestAction = ("max", -(float('inf')))
        for legalMove in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, legalMove)
            current_action, current_val = self.expectimax(successor, legalMove, depth - 1, (agentIndex + 1) % gameState.getNumAgents())
            if current_val > bestAction[1]:
                bestAction = (legalMove, current_val)
        return bestAction
    #minimizer
    def expect_val(self, gameState, action, depth, index):
        legalMoves = gameState.getLegalActions(index)
        avg = 0
        probability = 1.0 / len(legalMoves)
        for legalMove in legalMoves:
            successor = gameState.generateSuccessor(index, legalMove)
            current_action, current_val = self.expectimax(successor, action, depth - 1, (index + 1) % gameState.getNumAgents())
            avg += current_val * probability
        return (action, avg)


def betterEvaluationFunction(currentGameState):
    """

    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Food distance
    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    minFoodDistance = min(foodDistances) if foodDistances else 1
    foodFeature = 1.0 / minFoodDistance

    # Scared ghosts distance
    scaredGhostDistances = [
        manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer > 0
    ]
    minScaredGhostDistance = min(scaredGhostDistances) if scaredGhostDistances else 0
    scaredGhostFeature = 2.0 / minScaredGhostDistance if minScaredGhostDistance > 0 else 0

    # Non-scared ghosts distance
    ghostDistances = [
        manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer == 0
    ]
    minGhostDistance = min(ghostDistances) if ghostDistances else 1
    ghostFeature = -2.0 / minGhostDistance if minGhostDistance > 0 else 0

    # Number of food pellets left
    foodCountFeature = -1.5 * currentGameState.getNumFood()

    score = currentGameState.getScore() + foodFeature + ghostFeature + \
        scaredGhostFeature + foodCountFeature

    return score


# Abbreviation
better = betterEvaluationFunction
