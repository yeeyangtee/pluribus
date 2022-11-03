import random
import time
from typing import Dict, List
from pathlib import Path

import click
import joblib
import numpy as np
from blessed import Terminal

from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState
from poker_ai.terminal.ascii_objects.card_collection import AsciiCardCollection
from poker_ai.terminal.ascii_objects.player import AsciiPlayer
from poker_ai.terminal.ascii_objects.logger import AsciiLogger
from poker_ai.terminal.render import print_footer, print_header, print_log, print_table
from poker_ai.terminal.results import UserResults
from poker_ai.utils.algos import rotate_list, rotate_list_once

from poker_ai.terminal.interactive import run_interactive_app

# NOTE: hardcoded defaults for demo mode only.
@click.command()
@click.option('--lut_path', required=False, default='/ebs_volume_shared/card_info', type=str)
@click.option('--pickle_dir', required=False, default=True, type=bool)
@click.option('--agent', required=False, default="offline", type=str)
@click.option('--strategy_path', required=False, default="/ebs_volume_shared/trained_agents/deliver_v1/agent.joblib", type=str)
@click.option('--n_players', required=True, type=int)
@click.option('--low_card_rank', required=True, type=int)
@click.option('--initial_chips', required=False, default=10000, type=int)
@click.option('--play_as_bot', type=bool, default=False)
@click.option('--debug_quick_start/--no_debug_quick_start', default=False)
def play(
    lut_path: str,
    pickle_dir: bool,
    agent: str,
    strategy_path: str,
    n_players: int,
    low_card_rank: int,
    initial_chips: int,
    debug_quick_start: bool,
    play_as_bot: bool = False,
):
    # Main function to run the game
    if play_as_bot == True:
        run_interactive_app(
            lut_path=lut_path,
            pickle_dir=pickle_dir,
            agent=agent,
            strategy_path=strategy_path,
            n_players=n_players,
            low_card_rank=low_card_rank,
            initial_chips=initial_chips,
        )
    else:
        run_terminal_app(
            lut_path=lut_path,
            pickle_dir=pickle_dir,
            agent=agent,
            strategy_path=strategy_path,
            n_players=n_players,
            low_card_rank=low_card_rank,
            initial_chips=initial_chips,
            debug_quick_start=debug_quick_start,
        )


def run_terminal_app(
    lut_path: str,
    pickle_dir: bool,
    agent: str = "offline",
    strategy_path: str = "",
    n_players: int = 3,
    low_card_rank: int = 10,
    initial_chips: int = 10000,
    debug_quick_start: bool = False
):
    """Start up terminal app to play against your poker AI.

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
        offline_strategy_dict = joblib.load(strategy_path)
        offline_strategy = offline_strategy_dict['strategy']
        # Using the more fine grained preflop strategy would be a good idea
        del offline_strategy_dict["pre_flop_strategy"]
        del offline_strategy_dict["regret"]
    else:
        offline_strategy = {}
    print(f'Successfully loaded Agent in {time.time() - start_time:.2f} seconds.')


    if debug_quick_start:
        state: ShortDeckPokerState = new_game(n_players, {}, load_card_lut=False)
    else:
        state: ShortDeckPokerState = create_new_game(
            n_players=n_players,
            low_card_rank=low_card_rank, 
            initial_chips=initial_chips, 
            lut_path=lut_path, 
            pickle_dir=pickle_dir,)

    selected_action_i: int = 0
    num_players= None
    # We hard code these for now, but we could make this more flexible in the future.
    names = {}
    names[f'player_0'] = "HUMAN"
    for i in range(n_players-1):
        names[f'player_{i+1}'] = f"BOT {i+1}"

    user_results: UserResults = UserResults()
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
                is_human = names[player_name].lower() == "human"

                ascii_players.append(AsciiPlayer(
                    *player.cards,
                    term=term,
                    name=names[player_name],
                    og_name=player.name,
                    hide_cards=not is_human and not state.is_terminal,
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
                human_should_interact = names[og_current_name].lower() == "human"
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
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.info(term.pink("Quitting..."))
                        break
                    elif action == "new hand":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.clear()
                        # Compute valid players based on chip count, use state_players to retain rotation info.
                        valid_players = [p for p in state_players if p.n_chips > 0]

                        if len(valid_players) < 2: # If one player left, create new game instead.
                            state = create_new_game(
                                n_players=n_players,
                                low_card_rank=low_card_rank, 
                                card_info_lut = state.card_info_lut, 
                                initial_chips=initial_chips,)
                            n_table_rotations = 0
                            log.info(term.green(f"Game Over, Winner was {names[valid_players[0].name]}. Starting new game with fresh chips."))

                        else: # Create new state with previous state of players 
                            log.info(term.green("Dealing New Hand"))
                            state: ShortDeckPokerState = new_game(
                                n_players=n_players,
                                low_card_rank=low_card_rank, 
                                card_info_lut=state.card_info_lut, 
                                player_state = valid_players,)

                    elif action == "new game":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.clear()
                        log.info(term.green("Starting new game with fresh chips."))

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
                        names = {}
                        names[f'player_0'] = "HUMAN"
                        for i in range(num_players-1):
                            names[f'player_{i+1}'] = f"BOT {i+1}"


                        state = create_new_game(
                            n_players=num_players,
                            low_card_rank=low_card_rank, 
                            card_info_lut = state.card_info_lut, 
                            initial_chips=initial_chips,)
                    else:
                        log.info(term.green(f"{current_player_name} chose {action}"))
                        state: ShortDeckPokerState = state.apply_action(action)
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
                log.info(f"{current_player_name} chose {action}")
                state: ShortDeckPokerState = state.apply_action(action)

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


def create_new_game(n_players, low_card_rank, initial_chips=10000, card_info_lut=None, lut_path=None, pickle_dir=True):
    # Helper function to create new game 
    if lut_path: 
        state: ShortDeckPokerState = new_game(
            n_players=n_players,
            low_card_rank=low_card_rank, 
            initial_chips=initial_chips,
            lut_path=lut_path, 
            pickle_dir=pickle_dir,
        )
    else:
        assert card_info_lut is not None
        state: ShortDeckPokerState = new_game(
            n_players=n_players,
            low_card_rank=low_card_rank, 
            card_info_lut=card_info_lut, 
            initial_chips=initial_chips,
        )
    return state


if __name__ == "__main__":
    play()
