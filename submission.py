from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
import time
from enum import Enum
from scipy.spatial.distance import cityblock



def heuristic(state: GameState, player_index: int) -> float:

    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    #c
    if not state.snakes[player_index].alive:#we never want our snake to die
            return -500
    #setting weights
    too_long=8
    fruit_weight=1.4
    weight_for_length=500
    board_factor=np.sqrt(state.board_size.width**2+state.board_size.height**2)
    snake_length=state.snakes[player_index].length
    turns_left = (state.game_duration_in_turns - state.turn_number)
    possible_fruits=min(len(state.fruits_locations) + sum([s.length for s in state.snakes
                                                             if s.alive]) ,turns_left)
    if(possible_fruits>0):
        bonus_for_length = weight_for_length*snake_length / possible_fruits
    else:
        bonus_for_length=weight_for_length
    #calculating manheten distance and normalizing for board
    bonus_for_avoiding_tail=cityblock(state.snakes[player_index].head,state.snakes[player_index].tail_position)/np.sqrt(state.board_size.width**2+state.board_size.height**2)
    avoiding_tail_weight=1-1/snake_length if snake_length>too_long else 0
    bonus_for_avoiding_tail*=avoiding_tail_weight

    #distinguishing between two game modes eating fruits and surviving
    if len(state.fruits_locations)>0:
        nearest_fruits_weight = min([cityblock(state.snakes[player_index].head, trophy_i)
                                                    for trophy_i in state.fruits_locations])
        nearest_fruit_bonus=state.board_size.height+state.board_size.width-nearest_fruits_weight
        nearest_fruit_bonus/=(state.board_size.height+state.board_size.width)#normalize
        nearest_fruit_bonus*=fruit_weight
        return nearest_fruit_bonus+bonus_for_length+avoiding_tail_weight
    else:
        weight=1.8
        distance_from_enemy_bonus= min(cityblock(state.snakes[player_index].head,state.snakes[enemy].head)
        for enemy in state.get_opponents_alive(player_index)) if len(
        state.get_opponents_alive(player_index)) > 0 else 0
        distance_from_enemy_bonus/=board_factor #normalize
        return bonus_for_length*weight+bonus_for_avoiding_tail*weight+distance_from_enemy_bonus

class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    time=0
    num_played=0
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

    def __RB_Minimax__(self, state: TurnBasedGameState, depth):
        if state.game_state.is_terminal_state:
            return heuristic(state.game_state, self.player_index)
        if depth <= 0:
            assert(depth == 0)
            return heuristic(state.game_state, self.player_index)
        if state.turn == self.Turn.AGENT_TURN:
            cur_max = -np.inf
            for action in state.game_state.get_possible_actions(self.player_index):
                state.agent_action = action
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
        cur_time=time.clock()
        cur_max = -np.inf
        max_state = GameAction(1)
        for action in state.get_possible_actions(self.player_index):
            state_after_turn = MinimaxAgent.TurnBasedGameState(state, action)
            state_value = self.__RB_Minimax__(state_after_turn, 2)
            if state_value > cur_max:
                cur_max = state_value
                max_state = action
        end_time=time.clock()
        self.time+=end_time-cur_time
        self.num_played+=1
        return max_state


class AlphaBetaAgent(MinimaxAgent):
    time=0
    num_played=0
    dep=2
    def get_action(self, state: GameState) -> GameAction:
        start_time = time.clock()
        max_value = -np.inf
        maxi_action = GameAction(0)
        all_actions = state.get_possible_actions(self.player_index)
        for action in all_actions:
            curr_value = self.get_action_wrapper(
                MinimaxAgent.TurnBasedGameState(state, action), self.dep, -np.inf, np.inf)
            if curr_value > max_value:
                max_value = curr_value
                maxi_action = action
        stop_time = time.clock()
        #environment.time_elapsed += stop_time - start_time
        self.time+=stop_time-start_time
        self.num_played+=1
        avg_turn=self.time/self.num_played
        if((avg_turn) > 60/500 and self.dep>2):
            self.dep-=1
        elif((avg_turn)< 45/500): #if we have extra time we can allow ourselves to allow more depth
            self.dep+=1
        return maxi_action

    def get_action_wrapper(self, state: MinimaxAgent.TurnBasedGameState, dep: int, alpha: float, beta: float) -> float:
        if dep == 0 or state.game_state.is_terminal_state:
            return heuristic(state.game_state, self.player_index)
        turn = state.turn
        if turn == MinimaxAgent.Turn.AGENT_TURN:
            curr_max = -np.inf
            all_actions = state.game_state.get_possible_actions(
                self.player_index)
            for action in all_actions:
                state.agent_action = action
                temp_val = self.get_action_wrapper(state, dep, alpha, beta)
                curr_max = max(curr_max, temp_val)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return np.inf
            return curr_max
        else:
            assert(MinimaxAgent.Turn.OPPONENTS_TURN==turn)
            curr_min = np.inf
            for opponenets_action in state.game_state.get_possible_actions_dicts_given_action(state.agent_action, self.player_index):
                next_state = get_next_state(state.game_state,opponenets_action)
                next_state_with_turn = MinimaxAgent.TurnBasedGameState(
                    next_state, None)
                temp_val = self.get_action_wrapper(next_state_with_turn, dep-1, alpha, beta)
                curr_min = min(curr_min, temp_val)
                beta = min(curr_min, beta)
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


class TournamentAgent(AlphaBetaAgent):
    pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
