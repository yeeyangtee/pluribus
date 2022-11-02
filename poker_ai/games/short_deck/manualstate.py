from __future__ import annotations

import collections
import copy
import json
import logging
import operator
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pickle
import time

from poker_ai import utils
from poker_ai.poker.card import Card
from poker_ai.poker.engine import PokerEngine
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.poker.actions import Action
from poker_ai.poker.pot import Pot
from poker_ai.poker.table import PokerTable
from poker_ai.poker.deck import get_all_suits
from poker_ai.games.short_deck.state import ShortDeckPokerState

logger = logging.getLogger("poker_ai.games.short_deck.state")
InfoSetLookupTable = Dict[str, Dict[Tuple[int, ...], str]]


def new_game(
    n_players: int, 
    card_info_lut: InfoSetLookupTable = {}, 
    initial_chips:int=10000,
    player_state: List[ShortDeckPokerPlayer]=None, 
    **kwargs,
) -> ManualState:
    """
    Create a new game of short deck poker.
    Used in `ai/worker.py` and terminal app.`
    ...

    Parameters
    ----------
    n_players : int
        Number of players.
    card_info_lut (Optional): InfoSetLookupTable
        Card information cluster lookup table.
    initial_chips (Optional): int
        How many chips to start the players with
    player_state (optional): List[ShortDeckPokerPlayer] 
        Player information containing chip info.
    Returns
    -------
    state : ManualState
        Current state of the game
    """
    pot = Pot()
    if player_state is None:
        players = [
            ShortDeckPokerPlayer(player_i=player_i, initial_chips=initial_chips, pot=pot)
            for player_i in range(n_players)
        ]

    else: # Only used in terminal app.
        players = [
            ShortDeckPokerPlayer(player_i=player.name.split('_')[1], initial_chips=player.n_chips,pot=pot)
            for player in player_state
            ]
        # Check chipcount if 0, then put to not active (folded). Shouldnt need anymore
        for player in players:
            if player.n_chips == 0:
                player.is_active = False
        logger.debug(f"Created new game state with players: {players}")
    
    if card_info_lut:
        # Don't reload massive files, it takes ages.
        state = ManualState(
            players=players,
            load_card_lut=False,
            **kwargs
        )
        state.card_info_lut = card_info_lut
    else:
        # Load massive files.
        state = ManualState(
            players=players,
            **kwargs
        )
    return state


class ManualState(ShortDeckPokerState):
    """The state of a Manual Input poker game at some given point in time.
    We only need to reimplement the initialization, apply action functions.
    This should only be used for interactive app mode.

    The class is immutable and new state can be instanciated from once an
    action is applied via the `ShortDeckPokerState.new_state` method.
    """

    def __init__(
        self,
        players: List[ShortDeckPokerPlayer],
        small_blind: int = 50,
        big_blind: int = 100,
        low_card_rank: int = 2,
        lut_path: str = ".",
        pickle_dir: bool = False,
        load_card_lut: bool = True,
    ):
    # TODO here need to figure how to subclass and do the inits from the parent class

        """Initialise state."""
        n_players = len(players)
        if n_players <= 1:
            raise ValueError(
                f"At least 2 players must be provided but only {n_players} "
                f"were provided."
            )
        self._pickle_dir = pickle_dir
        self.low_card_rank = low_card_rank 
        if load_card_lut:
            self.card_info_lut = self.load_card_lut(lut_path, self._pickle_dir)
        else:
            self.card_info_lut = {}

        # These 2 mappings are for parsing human input
        self._rank_mapping = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        self._suit_mapping = {'s': 'spades', 'h':'hearts', 'd':'diamonds', 'c':'clubs'}

        # Get a reference of the pot from the first player.
        self._table = PokerTable(
            players=players, pot=players[0].pot, include_ranks=list(range(self.low_card_rank, 15))
        )
        # Get a reference of the initial number of chips for the payout. 
        # YY: This only works if all have same initial chips
        self._initial_n_chips = players[0].n_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self._poker_engine = PokerEngine(
            table=self._table, small_blind=small_blind, big_blind=big_blind
        )
        # Reset the pot, assign betting order to players (might need to remove
        # this), assign blinds to the players.
        self._poker_engine.round_setup()

        # MEOW: manual dont need to deal cards
        # self._table.dealer.deal_private_cards(self._table.players)

        # Get hole card inputs for bot player, which is set to default to have name as f'player_{n_player-1}' by the terminal app
        n_players = len(self._table.players)

        # Here Loop through each player, if they are the bot player then we need to get the input from the terminal app, else assign aces for placeholder.
        for player in self._table.players:
            if player.name != f'player_0':
                player.add_private_card(Card(rank=14, suit='spades'))
                player.add_private_card(Card(rank=14, suit='hearts'))
            else:
                holecards = self.get_user_input('pre_flop')
                for holecard in holecards:
                    player.add_private_card(holecard)


        # Store the actions as they come in here.
        self._history: Dict[str, List[str]] = collections.defaultdict(list)
        self._betting_stage = "pre_flop"
        self._betting_stage_to_round: Dict[str, int] = {
            "pre_flop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3,
            "show_down": 4,
        }
        # Rotate the big and small blind to the final positions for the pre
        # flop round only.
        player_i_order: List[int] = [p_i for p_i in range(n_players)]
        self.players[0].is_small_blind = True
        self.players[1].is_big_blind = True
        self.players[-1].is_dealer = True
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop": player_i_order,
            "turn": player_i_order,
            "river": player_i_order,
            "show_down": player_i_order,
            "terminal": player_i_order,
        }
        self._skip_counter = 0
        self._first_move_of_current_round = True
        self._reset_betting_round_state()
        for player in self.players:
            player.is_turn = False
        self.current_player.is_turn = True

        self.cardlut = self.create_card_lut(self.low_card_rank)


    def apply_action(self, action_str: Optional[str]) -> ShortDeckPokerState:
        """Create a new state after applying an action.

        Parameters
        ----------
        action_str : str or None
            The description of the action the current player is making. Can be
            any of {"fold, "call", "raise"}, the latter two only being possible
            if the agent hasn't folded already.

        Returns
        -------
        new_state : ShortDeckPokerState
            A poker state instance that represents the game in the next
            timestep, after the action has been applied.
        """
        if action_str not in self.legal_actions:
            raise ValueError(
                f"Action '{action_str}' not in legal actions: " f"{self.legal_actions}"
            )
        # Deep copy the parts of state that are needed that must be immutable
        # from state to state.
        lut = self.card_info_lut
        self.card_info_lut = {}
        new_state = copy.deepcopy(self)
        new_state.card_info_lut = self.card_info_lut = lut
        # An action has been made, so alas we are not in the first move of the
        # current betting round.
        new_state._first_move_of_current_round = False
        if action_str is None:
            # Assert active player has folded already.
            assert (
                not new_state.current_player.is_active
            ), "Active player cannot do nothing!"
        elif action_str == "call":
            action = new_state.current_player.call(players=new_state.players)
            logger.debug("calling")
        elif action_str == "fold":
            action = new_state.current_player.fold()

        # MEOW important section to configure more than just limit betting.
        elif action_str == "raise_quarter":
            bet_n_chips = int(new_state._table.pot.total * 0.25)
            action = self.perform_raise(new_state, bet_n_chips, action_str.split('_')[1])
        elif action_str == "raise_half":
            bet_n_chips = int(new_state._table.pot.total * 0.5)
            action = self.perform_raise(new_state, bet_n_chips, action_str.split('_')[1])
        elif action_str == "raise_3quarter":
            bet_n_chips = int(new_state._table.pot.total * 0.75)
            action = self.perform_raise(new_state, bet_n_chips, action_str.split('_')[1])
        elif action_str == "raise_one":
            bet_n_chips = int(new_state._table.pot.total)
            action = self.perform_raise(new_state, bet_n_chips, action_str.split('_')[1])
        elif action_str == "raise_allin":
            bet_n_chips = new_state.current_player.n_chips
            action = self.perform_raise(new_state, bet_n_chips, action_str.split('_')[1])
        
        
        else:
            raise ValueError(
                f"Expected action to be derived from class Action, but found "
                f"type {type(action)}."
            )
        # Update the new state.
        skip_actions = ["skip" for _ in range(new_state._skip_counter)]
        new_state._history[new_state.betting_stage] += skip_actions
        new_state._history[new_state.betting_stage].append(str(action))
        new_state._n_actions += 1
        new_state._skip_counter = 0

        # Player has made move, increment the state.current_player to the next avail one.
        while True:
            new_state._move_to_next_player()
            
            # 1.1) If everyone finished betting, first check if all have folded but one (terminal). 
            finished_betting = not new_state._poker_engine.more_betting_needed
            if finished_betting and new_state._poker_engine.n_active_players == 1:
                new_state._betting_stage = "terminal"
                # Do deal flop here so that can compute winnings only. 
                # Also some lines to remove the placeholder holecards to avoid collision, in case a human wins by all fold.
                if not new_state._table.community_cards:
                    new_state._poker_engine.table.dealer.deck.remove(Card(rank=14, suit='spades'))
                    new_state._poker_engine.table.dealer.deck.remove(Card(rank=14, suit='hearts'))
                    new_state._poker_engine.table.dealer.deal_flop(new_state._table)
                new_state._poker_engine.compute_winners()

            #1.2) Everyone has finished betting, and there are 2 or more players left.
            elif finished_betting and new_state.all_players_have_actioned:
                new_state._increment_stage()
                new_state._reset_betting_round_state()
                new_state._first_move_of_current_round = True
            
            # 2) If this player has folded, skip them.
            if not new_state.current_player.is_active:
                new_state._skip_counter += 1
                assert not new_state.current_player.is_active

            # 3) If this player has not folded, do the following checks, if didnt hit any, we continue to the next action.
            elif new_state.current_player.is_active:
                # 3.1) If Everyone else has folded, go to terminal without doing anything.
                if new_state._poker_engine.n_active_players == 1 :
                    new_state._betting_stage = "terminal"
                    # Do deal flop here just for compute winnings only.
                    if not new_state._table.community_cards:
                        new_state._poker_engine.table.dealer.deal_flop(new_state._table)
                # 3.2) If everyone has all-ined include ownself, go straight to showdown
                elif new_state._poker_engine.n_players_with_moves == 0 :
                    new_state._betting_stage = "show_down"
                # 3.3) If everyone has all-ined except ownself, check if need more betting. Else go showdown
                elif new_state._poker_engine.n_players_with_moves == 1 and finished_betting:
                    logger.info(f'All players have all-ined except {new_state.current_player.name}, finished bet {finished_betting}')
                    new_state._betting_stage = "show_down"
                # 3.4) If current player has allin, skip also
                elif new_state.current_player.is_all_in:
                    new_state._skip_counter += 1
                    continue

                if new_state._betting_stage == "terminal":
                    # Distribute winnings.
                    new_state._poker_engine.compute_winners()
                elif new_state._betting_stage == "show_down":
                    # IF showdown, we need to first ask for the remaining community cards, Then each of the in-play player cards.
                    if len(new_state._table.community_cards) == 4:
                        for card in new_state.get_user_input('river'):
                            new_state._table.add_community_card(card)

                    elif len(new_state._table.community_cards) == 3:
                        for card in new_state.get_user_input('turn'):
                            new_state._table.add_community_card(card)
                        for card in new_state.get_user_input('river'):
                            new_state._table.add_community_card(card)

                    elif len(new_state._table.community_cards) == 0:
                        for card in new_state.get_user_input('flop'):
                            new_state._table.add_community_card(card)
                        for card in new_state.get_user_input('turn'):
                            new_state._table.add_community_card(card)
                        for card in new_state.get_user_input('river'):
                            new_state._table.add_community_card(card)
                    
                    # Now looop through all players
                    for player in new_state._table.players:
                        # If player has not folded and player is not the BOT/USER
                        if player.is_active and player.name != f'player_0':
                            # Clear the default cards
                            player.cards = []
                            playerid = player.name.split('_')[1]
                            print(f'Please enter holecards for HUMAN {playerid}')
                            for card in new_state.get_user_input('pre_flop'):
                                player.add_private_card(card)

                    # Finally, compute winnings
                    new_state._poker_engine.compute_winners()

                break # Those that can hit this, can do one action.
            
        for player in new_state.players:
            player.is_turn = False
        new_state.current_player.is_turn = True
        return new_state
    
    @staticmethod
    def get_sanitised_input(message):
        '''This function asks for input from user repeatedly until a properly format is achieved.
        Returns the obtained length 2 string depicting a card'''
        while True:
            card = input(message)
            if not isinstance(card, str):
                logger.info(f"Expected card to be a string, but found {type(card)}.")
                continue
            elif len(card) != 2:
                logger.info(f"Expected card to be a string of length 2, but found {len(card)}.")
                continue
            elif card[0] not in "23456789TJQKA":
                logger.info(f"Expected first character of card to be in '23456789TJQKA', but found {card[0]}.")
                continue
            elif card[1] not in "cdhs":
                logger.info(f"Expected second character of card to be in 'cdhs', but found {card[1]}.")
                continue
            else: # Passed all checks
                break
        print(f'\nReceived Input {card}')
        return card

    
    def get_user_input(self, stage)->List[Card]:
        '''Helper function that asks for user input depending on the stage of the game
        Returns a list of cards
        It should also check that the inputted cards are NOT already in the table'''
        cards = []
        if stage == "pre_flop":
            while len(cards) < 2:
                inp = self.get_sanitised_input(f'Enter hole card {len(cards)+1}')
                card = Card(self._rank_mapping[inp[0]],self._suit_mapping[inp[1]])
                if card not in self._poker_engine.table.dealer.deck._dealt_cards and card not in cards:
                    cards.append(card)
                else:
                    print(f'Card {card} already dealt, please try again')
        elif stage == "flop":
           while len(cards) < 3:
                inp = self.get_sanitised_input(f'Enter FLOP card {len(cards)+1}')
                card = Card(self._rank_mapping[inp[0]],self._suit_mapping[inp[1]])
                if card not in self._poker_engine.table.dealer.deck._dealt_cards and card not in cards:
                    cards.append(card)
                else:
                    print(f'Card {card} already dealt, please try again')
        elif stage == "turn":
            while len(cards) < 1:
                inp = self.get_sanitised_input(f'Enter TURN card {len(cards)+1}')
                card = Card(self._rank_mapping[inp[0]],self._suit_mapping[inp[1]])
                if card not in self._poker_engine.table.dealer.deck._dealt_cards:
                    cards.append(card)
                else:
                    print(f'Card {card} already dealt, please try again')
        elif stage == "river":
            while len(cards) < 1:
                inp = self.get_sanitised_input(f'Enter RIVER card {len(cards)+1}')
                card = Card(self._rank_mapping[inp[0]],self._suit_mapping[inp[1]])
                if card not in self._poker_engine.table.dealer.deck._dealt_cards:
                    cards.append(card)
                else:
                    print(f'Card {card} already dealt, please try again')
        else:
            raise ValueError(f"Stage {stage} is not valid.")
        
        # Remove the user-inputted cards from deck manually.
        for card in cards:
            self._poker_engine.table.dealer.deck.remove(card)
        return cards

    def _increment_stage(self):
        """Once betting has finished, increment the stage of the poker game."""
        # Progress the stage of the game.
        if self._betting_stage == "pre_flop":
            # Progress from private cards to the flop.
            self._betting_stage = "flop"
            for card in self.get_user_input('flop'):
                self._table.add_community_card(card)
        elif self._betting_stage == "flop":
            # Progress from flop to turn.
            self._betting_stage = "turn"
            for card in self.get_user_input('turn'):
                self._table.add_community_card(card)
        elif self._betting_stage == "turn":
            # Progress from turn to river.
            self._betting_stage = "river"
            for card in self.get_user_input('river'):
                self._table.add_community_card(card)

        elif self._betting_stage == "river":
            # Progress to the showdown.
            self._betting_stage = "show_down"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")
