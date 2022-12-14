import random
import time
from typing import Dict, List
from pathlib import Path

import click
import joblib
import numpy as np
from blessed import Terminal

from poker_ai.ai import ai
from poker_ai.ai.agent import Agent

from poker_ai.games.short_deck.manualstate import ManualState, new_game
from poker_ai.games.short_deck.state import ShortDeckPokerState

from poker_ai.terminal.ascii_objects.card_collection import AsciiCardCollection
from poker_ai.terminal.ascii_objects.player import AsciiPlayer
from poker_ai.terminal.ascii_objects.logger import AsciiLogger
from poker_ai.terminal.render import print_footer, print_header, print_log, print_table
from poker_ai.terminal.results import UserResults
from poker_ai.utils.algos import rotate_list_once

# NOTE: hardcoded defaults for demo mode only.
def run_interactive_app(
    lut_path: str,
    pickle_dir: bool,
    agent: str = "offline",
    strategy_path: str = "",
    n_players: int = 3,
    low_card_rank: int = 10,
    initial_chips: int = 10000,
):
    """Start up terminal app to play as a bot.

    Example
    -------

    Usually you would call this from the `poker_ai` CLI. Alternatively you can
    call this method from this module directly from python.

    ```bash
    python -m poker_ai.terminal.runner                                       \
        --lut_path ./research/blueprint_algo                               \
        --agent offline                                                      \
        --pickle_dir True                                                   \
        --strategy_path ./agent.joblib                                       \
        --no_debug_quick_start
    ```
    """
    term = Terminal()
    log = AsciiLogger(term)
    assert Path(strategy_path).is_file(), f"Strategy path {strategy_path} does not exist."
    
    # Load strategy, LUT and stuff, one time operations
    # Load agent first since that takes time.
    print('Loading Agent...')
    start_time = time.time()
    if agent in {"offline", "online"}:
        offline_agent = Agent(agent_path=strategy_path, use_manager=False)
        # offline_strategy_dict = joblib.load(strategy_path)
        # offline_strategy = offline_strategy_dict['strategy']
        # offline_regrets = offline_strategy_dict["regret"]
        # Delete pre_flop_strategy since it is duplicate of strategy
        # del offline_strategy_dict["pre_flop_strategy"]
    else:
        offline_strategy = {}
    print(f'Successfully loaded Agent in {time.time() - start_time:.2f} seconds.')

    # Load the state which needs to load the very long LUT file
    state: ManualState = create_new_game(
        n_players=n_players,
        low_card_rank=low_card_rank, 
        initial_chips=initial_chips, 
        lut_path=lut_path, 
        pickle_dir=pickle_dir,)

    # This is to track the position of cursor/selection in the terminal app
    selected_action_i: int = 0
    num_players = None
    n_simulations = 5 # How many times to repeat CFR
    # Define player names, USER/BOT is always player 0
    names = {}
    names[f'player_0'] = "BOT"
    for i in range(n_players-1):
        names[f'player_{i+1}'] = f"HUMAN {i+1}"

    # Start main loop
    with term.cbreak(), term.hidden_cursor():
        # Loop until Human have no chips or win all chips, keep starting new HANDS
        while True:
            if num_players is not None and num_players != n_players:
                n_players = num_players
            # Construct ascii objects to be rendered later.
            ascii_players: List[AsciiPlayer] = []
            state_players = rotate_list_once(state.players)
            # state_players = state.players
            og_name_to_position = {}
            og_name_to_name = {}

            for player in state_players:
                player_name = player.name
                is_human = 'human' in names[player_name].lower() 
                # is_human = names[player_name].lower() == "human"

                ascii_players.append(AsciiPlayer(
                    *player.cards,
                    term=term,
                    name=names[player_name],
                    og_name=player.name,
                    hide_cards=is_human and not state.is_terminal,
                    # hide_cards=False,
                    folded=not player.is_active,
                    is_turn=player.is_turn,
                    chips_in_pot=player.n_bet_chips,
                    chips_in_bank=player.n_chips,
                    is_small_blind=player.is_small_blind,
                    is_big_blind=player.is_big_blind,
                    is_dealer=player.is_dealer,
                ))
                ascii_players.sort(key=lambda x: x.name, reverse=True)
                og_name_to_position[player.name] = player_name
                og_name_to_name[player.name] = names[player_name]
                if player.is_turn:
                    current_player_name = names[player_name]
            public_cards = AsciiCardCollection(*state.community_cards)

            # Check if hand is over
            if state.is_terminal:
                legal_actions = ["new hand", "new game", "quit"]
                human_should_interact = True
            else:
                og_current_name = state.current_player.name
                # human_should_interact = names[og_current_name].lower() == "human"
                human_should_interact = 'human' in names[og_current_name].lower()
                if human_should_interact:
                    legal_actions = state.legal_actions
                else:
                    legal_actions = []

            # Render game.
            print(term.home + term.white + term.clear)
            print_header(term, state, og_name_to_name)
            print_table(
                term,
                ascii_players,
                public_cards,
                n_chips_in_pot=state._table.pot.total,
            )
            print_footer(term, selected_action_i, legal_actions)
            print_log(term, log)
            # Make action of some kind.
            if human_should_interact:
                # Incase the legal_actions went from length 3 to 2 and we had
                # previously picked the last one.   
                selected_action_i %= len(legal_actions)
                key = term.inkey(timeout=None)
                if key.name == "q":
                    log.info(term.pink("quit"))
                    break
                elif key.name == "KEY_LEFT":
                    selected_action_i -= 1
                    if selected_action_i < 0:
                        selected_action_i = len(legal_actions) - 1
                elif key.name == "KEY_RIGHT":
                    selected_action_i = (selected_action_i + 1) % len(legal_actions)
                elif key.name == "KEY_ENTER":
                    action = legal_actions[selected_action_i]
                    if action == "quit":
                        log.info(term.pink("Quitting..."))
                        break
                    elif action == "new hand":
                        log.clear()
                        # Compute valid players based on chip count, use state_players to retain rotation info.
                        valid_players = [p for p in state_players if p.n_chips > 0]

                        if len(valid_players) < 2: # If one player left, create new game instead.
                            state = create_new_game(
                                n_players=n_players,
                                low_card_rank=low_card_rank, 
                                card_info_lut = state.card_info_lut, 
                                initial_chips=initial_chips,)
                            log.info(term.green(f"Game Over, Winner was {names[valid_players[0].name]}. Starting new game with fresh chips."))

                        else: # Create new state with previous state of players 
                            log.info(term.green("Dealing New Hand"))
                            state: ManualState = new_game(
                                n_players=n_players,
                                low_card_rank=low_card_rank, 
                                card_info_lut=state.card_info_lut, 
                                player_state = valid_players,)

                    elif action == "new game":
                        log.clear()
                        log.info(term.green("Starting new game with fresh chips."))
                                                
                        # Start user input block for new game
                        try: num_players = int(input("How many players?"))
                        except: num_players = n_players
                        if num_players >6 or num_players < 2:
                            print(f"Invalid number of players, defaulting previous: {n_players}")
                            num_players = n_players
                        try: initial_chips = int(input("How many chips? "))
                        except: initial_chips = 10000
                        if initial_chips < 1000 or initial_chips > 10000000:
                            print('Invalid number of chips, using default 10K')
                            initial_chips = 10000
                        try: n_simulations = int(input("How many simulations? "))
                        except: n_simulations = 5
                        if n_simulations < 1 or n_simulations > 1001:
                            print(f'Invalid number of simulations :{n_simulations} given, using default 5')
                            n_simulations = 5
                        # End user input block for new game
                        names = {}
                        names[f'player_0'] = "BOT"
                        for i in range(num_players-1):
                            names[f'player_{i+1}'] = f"HUMAN {i+1}"
                        state = create_new_game(
                            n_players=num_players,
                            low_card_rank=low_card_rank, 
                            card_info_lut = state.card_info_lut, 
                            initial_chips=initial_chips,)
                    else:
                        log.info(term.green(f"{current_player_name} chose {action}"))
                        state: ManualState = state.apply_action_interactive(action)
            else:
                if agent == "random":
                    action = random.choice(state.legal_actions)
                    time.sleep(0.8)
                elif agent == "offline":
                    default_strategy = {
                        action: 1 / len(state.legal_actions)
                        for action in state.legal_actions
                    }
                    this_state_strategy = offline_strategy.get(
                        state.info_set, default_strategy
                    )
                    # Normalizing strategy.
                    total = sum(this_state_strategy.values())
                    this_state_strategy = {
                        k: v / total for k, v in this_state_strategy.items()
                    }
                    actions = list(this_state_strategy.keys())
                    probabilties = list(this_state_strategy.values())
                    action = np.random.choice(actions, p=probabilties)
                    time.sleep(0.8)
                # 3) Play by blueprint for pre_flop. Then use CFR for flop onwards
                elif agent == "online":
                    i = state.player_i
                    # Create a copy of normal state so that search can be done
                    this_state_strategy = real_time_search(state, offline_agent, i, n_simulations)
                    # Normalizing strategy.
                    total = sum(this_state_strategy.values())
                    this_state_strategy = {
                        k: v / total for k, v in this_state_strategy.items()
                    }
                    actions = list(this_state_strategy.keys())
                    probabilties = list(this_state_strategy.values())
                    action = np.random.choice(actions, p=probabilties)

                    time.sleep(0.8)

                # Check if action is fold but state does not require betting, so change action to call
                biggest_bet = max(p.n_bet_chips for p in state.players)
                bet_not_required = state.current_player.n_bet_chips == biggest_bet
                if action == "fold" and bet_not_required:
                    log.info(f"{current_player_name} selected fold but defaulted to call")
                    action = "call"
                log.info(f"{current_player_name} chose {action}")
                # Do a manual printout in case apply action moves to next stage requiring user input..
                print(f"{current_player_name} chose {action}")
                state: ManualState = state.apply_action_interactive(action)

def real_time_search(state, agent, i, n_simulations):
    '''Performs real time search for the agent'''
    print(f'[INFO] Starting Real Time Search for player {i}')
    # use_pruning: bool = np.random.uniform() < 0.95
    use_pruning = False # Don't prune for real time search, takes longer but safer
    rts_iterations = n_simulations
    if use_pruning:
        for _ in range(rts_iterations):
            ai.cfrp(agent, state, i, t=0, c=-20000, locks=None)
    else:
        for _ in range(rts_iterations):
            ai.cfr(agent, state, i, t=0, locks=None)
    default_regret = state.initial_regret
    this_state_regret = agent.regret.get(state.info_set, default_regret)
    this_state_strategy = ai.calculate_strategy(this_state_regret)
    # ai.update_strategy(agent, state, i, t=0,locks=None)
    # print('Output from cfr', tmp)
    return this_state_strategy



def check_endgame(state):
    # Deprecated
    # helper to look through player chips and see if game has ended
    valid_players = 0
    for player in state.players:
        if player.n_chips > state.big_blind:
            valid_players += 1
    if valid_players == 1:
        return True
    else:
        return False

# Helper functions
def create_new_game(n_players, low_card_rank, initial_chips=10000, card_info_lut=None, lut_path=None, pickle_dir=True):
    '''Key function to creating a new game.
    Will generate a state: Manual State and also the player names.'''
    if lut_path: 
        state: ManualState = new_game(
            n_players=n_players,
            low_card_rank=low_card_rank, 
            initial_chips=initial_chips,
            lut_path=lut_path, 
            pickle_dir=pickle_dir,
        )
    else:
        assert card_info_lut is not None
        state: ManualState = new_game(
            n_players=n_players,
            low_card_rank=low_card_rank, 
            card_info_lut=card_info_lut, 
            initial_chips=initial_chips,
        )
    return state
