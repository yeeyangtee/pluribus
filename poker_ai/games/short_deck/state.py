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

logger = logging.getLogger("poker_ai.games.short_deck.state")
InfoSetLookupTable = Dict[str, Dict[Tuple[int, ...], str]]


def new_game(
    n_players: int, 
    card_info_lut: InfoSetLookupTable = {}, 
    initial_chips:int=10000,
    player_state: List[ShortDeckPokerPlayer]=None, 
    **kwargs,
) -> ShortDeckPokerState:
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
    state : ShortDeckPokerState
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
        logger.debug(f"Created new game state (Hand) with players: {players}")
    
    if card_info_lut:
        # Don't reload massive files, it takes ages.
        state = ShortDeckPokerState(
            players=players,
            load_card_lut=False,
            **kwargs
        )
        state.card_info_lut = card_info_lut
    else:
        # Load massive files.
        state = ShortDeckPokerState(
            players=players,
            **kwargs
        )
    return state


class ShortDeckPokerState:
    """The state of a Short Deck Poker game at some given point in time.

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
        # Deal private cards to players.
        self._table.dealer.deal_private_cards(self._table.players)
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

    def __repr__(self):
        """Return a helpful description of object in strings and debugger."""
        return f"<ShortDeckPokerState player_i={self.player_i} betting_stage={self._betting_stage}>"
    
    @staticmethod
    def perform_raise(state, n_chips:int, amount:str)->Action:
        '''Helper function to abstract the code for performing 
        a raise, returns the action to keep in history.'''
        biggest_bet = max(p.n_bet_chips for p in state.players)
        n_chips_to_call = biggest_bet - state.current_player.n_bet_chips
        raise_n_chips = n_chips + n_chips_to_call
        logger.debug(f"Betting {raise_n_chips} chips")
        action = state.current_player.raise_to_varied(n_chips=raise_n_chips, amount=amount)
        state._n_raises += 1
        return action

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
        
        
        # MEOW backup of old raise apply action made on 20oct2022
        # elif action_str == "raise":
        #     bet_n_chips = new_state.big_blind
        #     if new_state._betting_stage in {"turn", "river"}:
        #         bet_n_chips *= 2
        #     biggest_bet = max(p.n_bet_chips for p in new_state.players)
        #     n_chips_to_call = biggest_bet - new_state.current_player.n_bet_chips
        #     raise_n_chips = bet_n_chips + n_chips_to_call
        #     logger.debug(f"betting {raise_n_chips} n chips")
        #     action = new_state.current_player.raise_to(n_chips=raise_n_chips)
        #     new_state._n_raises += 1
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
            
            # 1) If finished all betting, move to next betting stage. 
            finished_betting = not new_state._poker_engine.more_betting_needed
            if finished_betting and new_state.all_players_have_actioned:
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
                    if len(new_state._poker_engine.table.community_cards) < 5:
                        new_state._poker_engine.table.dealer.deal_all(new_state._table)
                # 3.3) If everyone has all-ined except ownself, check if need more betting. Else go showdown
                elif new_state._poker_engine.n_players_with_moves == 1 and finished_betting:
                    # logger.info(f'All players have all-ined except {new_state.current_player.name}, finished bet {finished_betting}')
                    new_state._betting_stage = "show_down"
                    if len(new_state._poker_engine.table.community_cards) < 5:
                        new_state._poker_engine.table.dealer.deal_all(new_state._table)
                # 3.4) If current player has allin, skip also
                elif new_state.current_player.is_all_in:
                    new_state._skip_counter += 1
                    continue



                if new_state._betting_stage in {"terminal", "show_down"}:
                    # Distribute winnings.
                    new_state._poker_engine.compute_winners()

                break # Those that can hit this, can do one action.
            
        for player in new_state.players:
            player.is_turn = False
        new_state.current_player.is_turn = True
        return new_state

    @staticmethod
    def create_card_lut(low_card_rank):
        '''LUT that transforms from tuple of int(evalcard) to string of integers
        sort order is VERY important, as needs to sync up with the stored card info lut.'''
        suits = sorted(list(get_all_suits()))
        ranks = sorted(list(range(low_card_rank, 14 + 1))) # hardcode high card 
        cards = [int(Card(rank, suit)) for suit in suits for rank in ranks]
        cards.sort(reverse=True)

        lut = {}
        for i, c in enumerate(cards):
            lut[c] = f'{i:02d}'
        return lut 

    @staticmethod
    def load_card_lut(
        lut_path: str = ".",
        pickle_dir: bool = False
    ) -> Dict[str, Dict[Tuple[int, ...], str]]:
        """
        Load card information lookup table.

        ...

        Parameters
        ----------
        lut_path : str
            Path to lookupkup table.
        pickle_dir : bool
            Whether the lut_path is a path to pickle files or not. Pickle files
            are deprecated for the lut.

        Returns
        -------
        cad_info_lut : InfoSetLookupTable
            Card information cluster lookup table.
        """
        if pickle_dir:
            card_info_lut = {}
            logger.info(f"Loading card information from pickle files at: {lut_path}")
            
            betting_stages = ["pre_flop", "flop", "turn", "river"]
            for street in betting_stages:
                card_info_lut_path = lut_path + f'/card_info_lut_{street}.pkl'
                print(f'Loading LUT file from {card_info_lut_path}...')
                start_time = time.time()
                with open(card_info_lut_path,'rb') as f: 
                    card_info_lut[street]= pickle.load(f)
                print(f'Loaded LUT file from {card_info_lut_path} in {time.time() - start_time:.2f} seconds.')
        elif lut_path:
            logger.info(f"Loading card from single file at path: {lut_path}")
            card_info_lut = joblib.load(lut_path + '/card_info_lut.joblib')
        else:
            card_info_lut = {}
        return card_info_lut

    def _move_to_next_player(self):
        """Ensure state points to next valid active player."""
        self._player_i_index += 1
        if self._player_i_index >= len(self.players):
            self._player_i_index = 0

    def _reset_betting_round_state(self):
        """Reset the state related to counting types of actions."""
        self._all_players_have_made_action = False
        self._n_actions = 0
        self._n_raises = 0
        self._player_i_index = 0
        self._n_players_started_round = self._poker_engine.n_active_players
        while not self.current_player.is_active:
            self._skip_counter += 1
            self._player_i_index += 1

    def _increment_stage(self):
        """Once betting has finished, increment the stage of the poker game."""
        # Progress the stage of the game.
        if self._betting_stage == "pre_flop":
            # Progress from private cards to the flop.
            self._betting_stage = "flop"
            self._poker_engine.table.dealer.deal_flop(self._table)
        elif self._betting_stage == "flop":
            # Progress from flop to turn.
            self._betting_stage = "turn"
            self._poker_engine.table.dealer.deal_turn(self._table)
        elif self._betting_stage == "turn":
            # Progress from turn to river.
            self._betting_stage = "river"
            self._poker_engine.table.dealer.deal_river(self._table)
        elif self._betting_stage == "river":
            # Progress to the showdown.
            self._betting_stage = "show_down"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")

    @property
    def community_cards(self) -> List[Card]:
        """Return all shared/public cards."""
        return self._table.community_cards

    @property
    def private_hands(self) -> Dict[ShortDeckPokerPlayer, List[Card]]:
        """Return all private hands."""
        return {p: p.cards for p in self.players}

    @property
    def initial_regret(self) -> Dict[str, float]:
        """Returns the default regret for this state."""
        return {action: 0 for action in self.legal_actions}

    @property
    def initial_strategy(self) -> Dict[str, float]:
        """Returns the default strategy for this state."""
        return {action: 0 for action in self.legal_actions}

    @property
    def betting_stage(self) -> str:
        """Return betting stage."""
        return self._betting_stage

    @property
    def all_players_have_actioned(self) -> bool:
        """Return whether all players have made atleast one action."""
        return self._n_actions >= self._n_players_started_round

    @property
    def n_players_started_round(self) -> bool:
        """Return n_players that started the round."""
        return self._n_players_started_round

    @property
    def player_i(self) -> int:
        """Get the index of the players turn it is."""
        return self._player_i_lut[self._betting_stage][self._player_i_index]

    @player_i.setter
    def player_i(self, _: Any):
        """Raise an error if player_i is set."""
        raise ValueError(f"The player_i property should not be set.")

    @property
    def betting_round(self) -> int:
        """Betting stagee in integer form."""
        try:
            betting_round = self._betting_stage_to_round[self._betting_stage]
        except KeyError:
            raise ValueError(
                f"Attemped to get betting round for stage "
                f"{self._betting_stage} but was not supported in the lut with "
                f"keys: {list(self._betting_stage_to_round.keys())}"
            )
        return betting_round

    @property
    def info_set(self) -> str:
        """Get the information set for the current player."""
        cards = sorted(
            self.current_player.cards,
            key=operator.attrgetter("eval_card"),
            reverse=True,
        )
        cards += sorted(
            self._table.community_cards,
            key=operator.attrgetter("eval_card"),
            reverse=True,
        )
        if self._pickle_dir:
            # lookup_cards = tuple([card.eval_card for card in cards])
            # MEOW hacky patch here to handle string representation of card info. Also, not sure where this obj is defined in the main agent loop
            lookup_cards = ''.join([self.cardlut[int(c)] for c in cards])
        else:
            # lookup_cards = tuple([int(card) for card in cards])
            # MEOW hacky patch here to handle string representation of card info. Also, not sure where this obj is defined in the main agent loop
            lookup_cards = ''.join([self.cardlut[int(c)] for c in cards])
        try:
            cards_cluster = self.card_info_lut[self._betting_stage][lookup_cards]
        except KeyError:
            if self.betting_stage not in {"terminal", "show_down"}:
                raise ValueError(f"Stage {self._betting_stage}. You should have these cards in your lut. {lookup_cards}")
            return "default info set, please ensure you load it correctly"
        # Convert history from a dict of lists to a list of dicts as I'm
        # paranoid about JSON's lack of care with insertion order.
        info_set_dict = {
            "cards_cluster": cards_cluster,
            "history": [
                {betting_stage: [str(action) for action in actions]}
                for betting_stage, actions in self._history.items()
            ],
        }
        return json.dumps(
            info_set_dict, separators=(",", ":"), cls=utils.io.NumpyJSONEncoder
        )

    @property
    def payout(self) -> Dict[int, int]:
        """Return player index to payout number of chips dictionary."""
        n_chips_delta = dict()
        for player_i, player in enumerate(self.players):
            n_chips_delta[player_i] = player.n_chips - self._initial_n_chips
        return n_chips_delta

    @property
    def is_terminal(self) -> bool:
        """Returns whether this state is terminal or not.

        The state is terminal once all rounds of betting are complete and we
        are at the show down stage of the game or if all players have folded.
        """
        return self._betting_stage in {"show_down", "terminal"}

    @property
    def players(self) -> List[ShortDeckPokerPlayer]:
        """Returns players in table."""
        return self._table.players

    @property
    def current_player(self) -> ShortDeckPokerPlayer:
        """Returns a reference to player that makes a move for this state."""
        return self._table.players[self.player_i]

    # MEOW modified this to code in the staged raises
    @property
    def legal_actions(self) -> List[Optional[str]]:
        """Return the actions that are legal for this game state."""
        actions: List[Optional[str]] = []
        if self.current_player.is_active:
            actions += ["fold", "call"]

            if self._betting_stage in {"pre_flop",'flop'}:
                if self._n_raises == 0:
                    actions += ["raise_quarter", "raise_half", "raise_3quarter","raise_one","raise_allin"]
                elif self._n_raises < 3:
                    actions += ["raise_one","raise_allin"]
            elif self._betting_stage in {'turn','river'}:
                if self._n_raises == 0:
                    actions += ["raise_half", "raise_one", "raise_allin"]
                elif self._n_raises < 3:
                    actions += ["raise_one", "raise_allin"]

        else:   
            actions += [None]
        return actions

    # MEOW backup of legal_actions property made on 20oct2022
    # @property
    # def legal_actions(self) -> List[Optional[str]]:
    #     """Return the actions that are legal for this game state."""
    #     actions: List[Optional[str]] = []
    #     if self.current_player.is_active:
    #         actions += ["fold", "call"]
    #         if self._n_raises < 3:
    #             # In limit hold'em we can only bet/raise if there have been
    #             # less than three raises in this round of betting, or if there
    #             # are two players playing.
    #             actions += ["raise"]
    #     else:
    #         actions += [None]
    #     return actions
