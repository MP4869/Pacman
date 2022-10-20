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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #chosenIndex = max(bestIndices)

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
        newFood = newFood.asList()
        ret = successorGameState.getScore()
        #store and store2 store distance of food and ghost respectivley
        store =[]
        store2 =[]
        #the two for loops store distance into the two arrays
        for i in newGhostStates:
            temp =(util.manhattanDistance(newPos, i.getPosition()))
            store2.append(temp)
        #modify ret
        if store2!=[]:
            ret =ret+max(store2)
        for i in newFood:
            store.append(util.manhattanDistance(newPos, i))
        #modify ret if not empty
        if store != []:
            ret = ret - max(store)
            # return ret
        return ret
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"

        def getMin(currState, depth, agent):
            ret = 99999
            if agent != 0:
                for curr in currState.getLegalActions(agent):
                    # if not all ghosts have been iterated
                    curAgent = ((agent + 1) % currState.getNumAgents())
                    if (curAgent != 0):
                        (output1, a1) = getMin(currState.generateSuccessor(agent, curr), depth, curAgent)
                    else:
                        # increase depth by one and go back to getMax
                        (output1, v1) = getMax(currState.generateSuccessor(agent, curr), depth + 1, curAgent)
                    # get the smallest
                    if (output1 < ret):
                        ret, smallest = output1, curr
            # if ret has changed over time that means there is a value to return
            if ret != 99999:
                return (ret, smallest)
            return (self.evaluationFunction(currState), None)

        def getMax(currState, depth, agent):

            # edge case
            if currState.isWin() or currState.isLose() or depth == self.depth or currState.getLegalActions(agent) == 0:
                # return (self.evaluationFunction(state), None)
                pass
            else:
                # smallest default value for ret
                ret = -99999
                if agent == 0:
                    for curr in currState.getLegalActions(agent):
                        # get min cost for all potental actionts
                        output1, output2 = getMin(currState.generateSuccessor(agent, curr), depth,(agent + 1) % currState.getNumAgents())
                        if (output1 > ret):
                            # check if its the largest value
                            ret, largest = output1, curr
                # return the max value
                return (ret, largest)
            # return base case value
            return (self.evaluationFunction(currState), None)

            ############################################################

        return getMax(gameState, 0, 0)[1]
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

        def getMin(currState, depth, agent, alpha, beta):
            ret = 99999
            if agent != 0:
                for curr in currState.getLegalActions(agent):
                    curAgent = ((agent + 1) % currState.getNumAgents())
                    # if not all ghosts have been iterated
                    if (curAgent != 0):
                        (output1, a1) = getMin(currState.generateSuccessor(agent, curr), depth, curAgent, alpha, beta)
                    else:
                        # increase depth by one and go back to getMax if all ghosts itterated
                        (output1, v1) = getMax(currState.generateSuccessor(agent, curr), depth + 1, curAgent, alpha,
                                               beta)
                    # get the smallest
                    if (output1 < ret):
                        ret, smallest = output1, curr
                    if ret < alpha:
                        return (ret, smallest)
                    beta = min(beta, ret)

            # if ret has changed over time that means there is a value to return
            if ret != 99999:
                return (ret, smallest)
            return (self.evaluationFunction(currState), None)

        def getMax(currState, depth, agent,alpha,beta):
            # edge case
            if currState.isWin() or currState.isLose() or depth == self.depth or currState.getLegalActions(agent) == 0:
                # return (self.evaluationFunction(state), None)
                pass
            else:
                #smallest default value for ret
                ret = -99999
                if agent == 0:
                    for curr in currState.getLegalActions(agent):
                        #get min cost for all potental actionts
                        output1, output2 = getMin(currState.generateSuccessor(agent, curr), depth, (agent + 1) % currState.getNumAgents(),alpha,beta)
                        if (output1 > ret):
                            #check if its the largest value
                            ret, largest = output1, curr
                        if ret > beta:
                            #if it is the largest
                            return (ret, largest)
                        alpha = max(alpha, ret)
                # return the max value
                return (ret, largest)
            #return base case value
            return (self.evaluationFunction(currState), None)

        ############################################################
        return getMax(gameState, 0, 0, -99999,99999)[1]
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
        def expectimax(currState, depth, agent):
            # edge case
            if currState.isWin() or currState.isLose() or depth == self.depth or currState.getLegalActions(agent) == 0:
                pass
            else:
                ret = -99999
                if (agent == 0):
                    ret, largest =pac(currState,depth,agent,ret, 0)
                # if the ret is cahnged that means ret is found
                if ret != -99999:
                    return (ret, largest)
                #reset values for if agent is ghost
                ret, count =  0.0, 0.0
                if (agent != 0):
                    #call ghost to minimize.
                    (ret, count,smallest) =ghost(currState,depth,agent,ret,count,0)
                #return values from ghost
                return (ret / count, smallest)
            #edge case
            return (self.evaluationFunction(currState), None)
        def pac(currState, depth, agent, ret, largest):
            for curr in currState.getLegalActions(agent):
                #itterate through all actions
                (output1, output2) = expectimax(currState.generateSuccessor(agent, curr), depth, (agent + 1) % currState.getNumAgents())
                if (output1 > ret):
                    #if value is the largest
                    ret, largest = output1, curr
            return ret, largest

        def ghost(currState, depth, agent,ret,count,smallest):
            for curr in currState.getLegalActions(agent):
                curAgent = ((agent + 1) % currState.getNumAgents())
                #itterate through all actions
                if ( curAgent!= 0):
                    #if it is not the first case
                    (output1, output2) = expectimax(currState.generateSuccessor(agent, curr), depth, curAgent)
                else:
                    #else we increase depth
                    (output1,output2) = expectimax(currState.generateSuccessor(agent, curr), depth + 1, curAgent)
                ret =ret + output1
                count =count + 1
                smallest = curr
            return ret,count,smallest


            ############################################################

        return expectimax(gameState, 0, 0)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    this program takes into consideration distance from ghost, scared ghost, and food and uses preset weights to determain a evaluation score
    if there is a scared ghost nearby, the score increases by: weight + (weightS/distance). if the ghost is not scared score decreases by: weight -(weight/distance)
    if there is still food left, score increases by foodweight/distance of closest food.
    Note that manhattan distance determines distance and that weights are 30,10,15 for scared ghost, food, and ghost respectivley.
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ret = currentGameState.getScore()
    weightS = 30.0
    weightF = 10.0
    weightG = 15.0  # 10

    nearstFood = []
    for x in newFood:
        # find all distance from foods
        nearstFood.append(util.manhattanDistance(newPos, x))
    # itterate through all foods to find how many are close to current position
    if len(nearstFood) != 0:
        # check if food list is empty
        largest = 0
        for i in nearstFood:
            largest = max(i, largest)
        # calculations for closest food
        ret += (weightF / largest)

    for x in newGhostStates:
        # Ghost X distance from packman
        length = util.manhattanDistance(newPos, x.getPosition())
        if length > 0:
            #if it is scared then do calculaitons
            if x.scaredTimer > 0:
                ret+=(weightS-(weightS/length))
            #regular ghosts
            else:
                ret-= (weightG-(weightG/length))
    #foodList = newFood.asList()

    return ret
    util.raiseNotDefined()
# Abbreviation
better = betterEvaluationFunction
