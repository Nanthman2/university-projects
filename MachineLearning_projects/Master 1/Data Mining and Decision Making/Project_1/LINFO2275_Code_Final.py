########################################################################################
#---LINFO2277 : SNAKES AND LADDER -----------------------------------------------------#
########################################################################################

# Required Package(s) :

import numpy as np


class BoardGame:

    def __init__(self, layout, circle):
        self.layout = layout
        self.circle = circle
        self.dice_options = {1: [0,1], 2: [0,1,2], 3: [0,1,2,3]}
        self.dice_prob = {1: 1/2, 2: 1/3, 3: 1/4}
        self.trap_prob = {1: 0, 2: 1/2, 3: 1}
        self.fast_line = [10, 11, 12, 13]
        self.fast_line_order = [0, 1, 2, 10, 11, 12, 13, 14] # order when you are taking the fast line
        self.slow_line = [3, 4, 5, 6, 7, 8, 9]
        self.slow_line_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14] # order when you are taking the slow line

    def ValueIteration(self, epsilon=10**-6):
        """
        The function iteratively computes the expected cost for each state and updates the optimal dice choice
        until the change in expected costs is less than a given threshold (epsilon). The stopping criterion is based
        on the maximum change between iterations being smaller than epsilon.

        Parameters:
            epsilon (float): The threshold for the maximum allowed change in the expected cost between iterations.
                             The iteration stops when the change is smaller than epsilon. Default is 1e-6.

        Returns:
            list: A list containing:
                            - Expec (numpy.ndarray): The final array of expected costs for each state.
                            - Dice (numpy.ndarray): The optimal dice choices for each state, based on the computed expected costs.
        """
        iteration = 0
        Dice = np.ones(14, dtype=int) # we suppose that the ideal policy is to always play 1
        Expec = np.array([20., 18., 16., 14., 12., 10., 8., 6., 4., 2., 8., 6., 4., 2.]) # in such a case, the expected cost is this one
        while True:
            delta = 0
            iteration += 1
            previous_expec = Expec.copy()
            for k in range(14):
                dice_k, expec_k = self.BellmanOptimum(k, Expec)
                Dice[k] = dice_k
                Expec[k] = expec_k
                delta = max(delta, np.abs(Expec - previous_expec).max())
            # convergence criterion
            if delta < epsilon:
                break
        return [Expec, Dice]

    def BellmanOptimum(self, state, expec):
        """
        This function evaluates the expected cost for each possible action in a given state based on the 
        transition probabilities and expected costs of successor states. It selects the action that minimizes 
        the expected cost, taking into account the effects of traps on the transition.

        Parameters:
            state (int): The current state for which the optimal action is being computed.
            expec (numpy.ndarray): Array of expected costs for each state, used to calculate the future costs.

        Returns:
            tuple: A tuple containing:
                - best_action (int): The optimal action (1, 2, or 3) for the given state.
                - min_value (float): The minimum expected cost associated with the best action.
        """
        c = 1
        best_action = None
        min_value = float('inf')

        for action in [1, 2, 3]:
            Sum = 0
            p = self.trap_prob[action]
            for successor, transition_prob in self.ApplyTraps(state, action).items():
                if successor < 14:
                    # bonus :
                    if self.layout[successor]==3 and action !=1 :
                        Sum+= (p*transition_prob * (expec[successor]+1)) + ((1-p)*transition_prob*expec[successor])
                    # prison :
                    elif self.layout[successor]==4 and action !=1:
                        Sum+= (p*transition_prob * (expec[successor]-1)) + ((1-p)*transition_prob*expec[successor])
                    else :
                        Sum += transition_prob * expec[successor]

            value = c + Sum

            if value < min_value:
                min_value = value
                best_action = action

        return best_action, min_value


    def ApplyTraps(self, state, dice):
        """
        This function computes the transition probabilities for a given state and dice choice, taking into account 
        traps (except Prison and Bonus) that may affect the transitions. If the dice roll is 1, no traps are applied. Otherwise, the 
        function updates the transitions by applying the trap effects based on the dice value, which modifies 
        the transition probabilities to reflect the changes due to the traps.

        Parameters:
            state (int): The current state from which transitions are evaluated.
            dice (int): The dice value (1, 2, or 3) that determines the effect of traps on the transition probabilities.

        Returns:
            dict: A dictionary where the keys are the successor states and the values are the corresponding transition probabilities.
        """
        transitions = self.TransitionProbabilities(state, dice)

        if dice == 1:
            return transitions # dice 1 = immunity to traps

        else:
            prob = self.trap_prob[dice]
            new_transitions = transitions.copy()

            for next_state, p in transitions.items():
                # Restart :
                if self.layout[next_state] == 1:
                    new_transitions[next_state] = p*(1-prob) # we modify the 'old' transition probabilities
                    new_transitions[0] = new_transitions.get(0, 0) + p*prob # then modify the transition probabilities to state 0
                # Minus 3 :
                elif self.layout[next_state] == 2:
                    trap_state = self.Move(next_state, -3) 
                    new_transitions[next_state] = p*(1-prob) # we modify the 'old' transition probabilities
                    new_transitions[trap_state] = new_transitions.get(trap_state, 0) + p*prob # then modify the transition probabilities to trap state

            return new_transitions


    def Move(self, start_state, move):
        """
        This function computes the next state based on the current state and the move applied. It takes into account 
        whether the current state is part of the fast or slow line and handles the specific movement rules (e.g., 
        special cases for the circle layout). If the move takes the state out of bounds, the function 
        ensures it returns a valid state based on the layout configuration.

        Parameters:
            start_state (int): The current state from which the move starts.
            move (int): The number of steps to move, which can be positive (forward) or negative (backward).

        Returns:
            int: The resulting state after applying the move. The value is constrained to the valid states, considering
            the circle layout and boundaries.
        """
        if start_state in self.fast_line:
            index = self.fast_line_order.index(start_state)
            next_index = index + move
            if next_index <= 0:
                return 0
            elif next_index >= len(self.fast_line_order):
                return 0 if self.circle else 14 # we restart if we are beyond the last state and if circle is True
            else:
                return self.fast_line_order[next_index]
        elif start_state in self.slow_line:
            index = self.slow_line_order.index(start_state)
            next_index = index + move
            if next_index <= 0:
                return 0
            elif next_index >= len(self.slow_line_order):
                return 0 if self.circle else 14
            else:
                return self.slow_line_order[next_index]
        else:
            next_state = start_state + move
            if next_state < 0 :
                return 0
            else :
                return next_state

    def TransitionProbabilities(self, state, dice):
        """
        This function computes the transition probabilities over the possible successor states.
        The transition probabilities depend on the current state, the chosen dice, and the move options available 
        for the dice roll. Special handling is included for specific states (such as state 2), where the transitions 
        might involve branching to multiple states with adjusted probabilities.

        Parameters:
            state (int): The current state from which transitions are evaluated.
            dice (int): The dice value (1, 2, or 3) that determines the movement options and associated probabilities.

        Returns:
            dict: A dictionary where the keys are the successor states and the values are the corresponding transition probabilities.
        """
        transitions = {}
        for move, p in zip(self.dice_options[dice], [self.dice_prob[dice]] * len(self.dice_options[dice])):
            # special case for state 2 (where we can take the fast line or the slow line with .5 probability)
            if state == 2:
                if move == 0:
                    transitions[state] = transitions.get(state, 0) + p
                elif move == 1:
                    transitions[3] = transitions.get(3, 0) + 0.5 * p
                    transitions[10] = transitions.get(10, 0) + 0.5 * p
                elif move == 2:
                    transitions[4] = transitions.get(4, 0) + 0.5 * p
                    transitions[11] = transitions.get(11, 0) + 0.5 * p
                elif move == 3:
                    transitions[5] = transitions.get(5, 0) + 0.5 * p
                    transitions[12] = transitions.get(12, 0) + 0.5 * p
            else:
                next_state = self.Move(state, move)
                transitions[next_state] = transitions.get(next_state, 0) + p
        return transitions


def markovDecision(layout, circle):
    """
    This function initializes a new game environment based on the provided layout and circle settings, 
    and then computes the optimal policy using value iteration. The optimal policy is determined by the 
    `ValueIteration` method, which evaluates the best dice choices and their expected costs across all states.

    Parameters:
        layout (list): The configuration of the board, specifying the arrangement of traps.
        circle (bool): A boolean indicating whether the board has a circular layout (True) or not (False).

    Returns:
        tuple: A list containing:
            - Expec (numpy.ndarray): The final array of expected costs for each state.
            - Dice (numpy.ndarray): The optimal dice choices for each state, based on the computed expected costs.
    """
    game = BoardGame(layout, circle)
    return game.ValueIteration()
