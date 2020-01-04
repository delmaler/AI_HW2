from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
import time
from enum import Enum
from scipy.spatial.distance import cityblock

"""
def heuristic(state: GameState, player_index: int) -> float:
    """"""
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """"""
    if not state.snakes[player_index].alive:
            return state.snakes[player_index].length
    discount_factor = 0.5
    max_possible_fruits = len(state.fruits_locations) + sum([s.length for s in state.snakes
                                                                 if s.index != player_index and s.alive])
    turns_left = (state.game_duration_in_turns - state.turn_number)
    snake_manhattan_dists_from_fruits = sorted([cityblock(state.snakes[player_index].head, trophy_i)
                                                    for trophy_i in state.fruits_locations])
    
    max_possible_fruits = min(max_possible_fruits, turns_left)
    optimistic_future_reward = discount_factor*(1 - discount_factor ** max_possible_fruits) / (1-discount_factor)
    return state.snakes[player_index].length + optimistic_future_reward+1/snake_manhattan_dists_from_fruits[0]+state.snakes[player_index].length+10*state.snakes[player_index].alive
    """


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    min_manhattan_distance_from_fruit = num_snakes_alive = num_snakes_alive_bigger_than_me = 0
    my_pos_row = state.snakes[player_index].head[0]
    my_pos_col = state.snakes[player_index].head[1]

    # TODO: return the utility value if the state is a final state
    if not state.snakes[player_index].alive:
        return 0

    max_possible_fruits = len(state.fruits_locations) + sum([s.length for s in state.snakes
                                                             if s.alive])
    weight_for_bonus_length = 500
    bonus_for_length = state.snakes[player_index].length / \
        max_possible_fruits * weight_for_bonus_length

    bonus_for_distance_from_closest_enemy = min(abs(my_pos_row - state.snakes[enemy].head[0]) + abs(
        my_pos_col - state.snakes[enemy].head[1])
        for enemy in state.get_opponents_alive(player_index)) if len(
        state.get_opponents_alive(player_index)) > 0 else 0
    bonus_for_distance_from_closest_enemy /= (
        state.board_size.height + state.board_size.width)
    weight_for_distance_from_closest_enemy = 0.5
    bonus_for_distance_from_closest_enemy *= weight_for_distance_from_closest_enemy

    bonus_for_air_dist_from_tail = np.sqrt((my_pos_row - state.snakes[player_index].tail_position[0]) ** 2 + (
        my_pos_col - state.snakes[player_index].tail_position[1]) ** 2) / np.sqrt(
        state.board_size.height ** 2 + state.board_size.width ** 2)
    length = state.snakes[player_index].length
    weight_for_air_dist_from_tail = 1 - 1 / length if length >= 10 else 0
    bonus_for_air_dist_from_tail *= weight_for_air_dist_from_tail

    if len(state.fruits_locations) > 0:
        min_manhattan_distance_from_fruit = min(
            abs(fruit_row - my_pos_row) + abs(fruit_col - my_pos_col) for fruit_row, fruit_col in
            state.fruits_locations) if len(state.fruits_locations) > 0 else 0
        bonus_for_manhattan_distance = state.board_size.height + \
            state.board_size.width - min_manhattan_distance_from_fruit
        bonus_for_manhattan_distance = bonus_for_manhattan_distance / (
            state.board_size.height + state.board_size.width)  # normalization
        weight_for_manhattan_distance = 1
        bonus_for_manhattan_distance *= weight_for_manhattan_distance
        return bonus_for_manhattan_distance + bonus_for_length + bonus_for_air_dist_from_tail + bonus_for_distance_from_closest_enemy
    else:
        weight_for_air_dist_from_tail = 10
        bonus_for_air_dist_from_tail *= weight_for_air_dist_from_tail
        return bonus_for_length * 2 + bonus_for_air_dist_from_tail + bonus_for_distance_from_closest_enemy





class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:

        #    This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        #   in the game (set agent_action=None to indicate that our agent has yet to pick an action).

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

        def set_action(self, agent_action):
            self.agent_action = agent_action

    def __RB_Minimax__(self, state: TurnBasedGameState, depth=2):
        if state.game_state.is_terminal_state:
            print("2")
            return heuristic(state.game_state, self.player_index)
        if depth <= 0:
            assert(depth == 0)
            print("1")
            return heuristic(state.game_state, self.player_index)
        if state.turn == self.Turn.AGENT_TURN:
            cur_max = -np.inf
            for action in state.game_state.get_possible_actions(self.player_index):
                state.agent_action=action
                cur_value = self.__RB_Minimax__(state, depth)
                cur_max = max(cur_max, cur_value)
            return cur_max
        else:
            assert state.turn == self.Turn.OPPONENTS_TURN
            cur_min = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action, self.player_index):
                next_state = get_next_state(
                    state.game_state, opponents_actions)
                next_state_with_turn = self.TurnBasedGameState(
                    next_state, None)
                cur_min = min(cur_min, self.__RB_Minimax__(
                    next_state_with_turn, depth-1))
            return cur_min

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        # cur_time=time.clock
        cur_max = -np.inf
        max_state = GameAction(1)
        for action in state.get_possible_actions(self.player_index):
            state_after_turn = MinimaxAgent.TurnBasedGameState(state, action)
            state_value = self.__RB_Minimax__(state_after_turn, 2)
            if state_value > cur_max:
                cur_max = state_value
                max_state = action
        return max_state


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        dep = 2
        start_time = time.clock()
        max_value = -np.inf
        maxi_action = GameAction(0)
        all_actions = state.get_possible_actions(self.player_index)
        for action in all_actions:
            curr_value = self.get_action_wrapper(
                MinimaxAgent.TurnBasedGameState(state, action), dep, -np.inf, np.inf)
            if curr_value > max_value:
                max_value = curr_value
                maxi_action = action
        stop_time = time.clock()
        #environment.time_elapsed += stop_time - start_time
        return maxi_action

    def get_action_wrapper(self, state: MinimaxAgent.TurnBasedGameState, dep: int, alpha: float, beta: float) -> float:
        if dep == 0 or state.game_state.is_terminal_state:
            return heuristic(state.game_state, self.player_index)
        turn = state.turn
        all_actions = state.game_state.get_possible_actions(self.player_index)
        if turn == MinimaxAgent.Turn.AGENT_TURN:
            curr_max = -np.inf
            for action in all_actions:
                state.agent_action = action
                temp_val = self.get_action_wrapper(state, dep-1, alpha, beta)
                curr_max = max(curr_max, temp_val)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return np.inf
            return curr_max
        else:
            curr_min = np.inf
            for action in all_actions:
                state.agent_action = action
                temp_val = self.get_action_wrapper(state, dep-1, alpha, beta)
                curr_min = min(curr_min, beta)
                beta = min(curr_min, temp_val)
                if curr_min <= alpha:
                    return -np.inf
            return curr_min


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """

    initial_actions_vector = get_random_action_list()
    to_print = ""
    for i in range(50):
        print("round num: " + str(i))
        curr_action = initial_actions_vector[i]
        for act in GameAction:
            if act != curr_action:
                test_actions_vector = initial_actions_vector.copy()
                test_actions_vector[i] = act  # put the new action here
                if get_fitness(initial_actions_vector) <= get_fitness(test_actions_vector):
                    initial_actions_vector[i] = test_actions_vector[i]
    for i in initial_actions_vector.__iter__():
        to_print = to_print + i.name + " "
    print(to_print)


def get_random_action_list():
    """
    randomizes a list of 50 actions, they are randomized so that there is:
    0.25 chances of a right turn
    0.25 chance of a left turn
    0.5 of going straight
    :return:
    """
    res = []
    for i in range(50):
        temp = np.random.randint(20)
        if 0 <= temp <= 4:
            res.append(GameAction.LEFT)
        if 5 <= temp <= 9:
            res.append(GameAction.RIGHT)
        else:
            res.append(GameAction.STRAIGHT)
    return res


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    to_print = ""
    max_fitness = 0
    max_actions_vector = []
    rounds = 0
    while rounds < 10:
        i = 0
        initial_actions_vector = get_random_action_list()
        print("round num:" + str(rounds))
        while i < 50:
            print("itr num:" + str(i))
            curr_action = initial_actions_vector[i]
            for act in GameAction:
                if act != curr_action:
                    test_actions_vector = initial_actions_vector.copy()
                    test_actions_vector[i] = act  # put the new action here
                    if get_fitness(initial_actions_vector) <= get_fitness(test_actions_vector):
                        initial_actions_vector[i] = test_actions_vector[i]
            i += 1
        if get_fitness(initial_actions_vector) > max_fitness:
            max_fitness = get_fitness(initial_actions_vector)
            max_actions_vector = initial_actions_vector.copy()
        rounds += 1
    print("max fitness: " + str(max_fitness))
    for i in max_actions_vector.__iter__():
        to_print = to_print + i.name + " "
    print(to_print)


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
