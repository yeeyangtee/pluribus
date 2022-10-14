from typing import List

import numpy as np
import random

from poker_ai.poker.evaluation import Evaluator


class GameUtility:
    """This class takes care of some game related functions."""

    def __init__(self, our_hand: np.ndarray, board: np.ndarray, cards: np.ndarray, evaluator: Evaluator):
        # MEOW try to use global
        # self._evaluator = Evaluator()
        self._evaluator = evaluator
        unavailable_cards = np.concatenate([board, our_hand], axis=0)
        self.available_cards = np.array(
            [c for c in cards if c not in unavailable_cards]
        )
        self.our_hand = our_hand
        self.board = board

    def evaluate_hand(self, hand: np.ndarray) -> int:
        """
        Evaluate a hand.

        Parameters
        ----------
        hand : np.ndarray
            Hand to evaluate.

        Returns
        -------
            Evaluation of hand
        """
        return self._evaluator.evaluate(
            board=self.board.astype(np.int).tolist(),
            cards=hand.astype(np.int).tolist(),
        )

    def get_winner(self) -> int:
        """Get the winner.

        Returns
        -------
            int of win (0), lose (1) or tie (2) - this is an index in the
            expected hand strength array
        """
        our_hand_rank = self.evaluate_hand(self.our_hand)
        opp_hand_rank = self.evaluate_hand(self.opp_hand)
        if our_hand_rank > opp_hand_rank:
            return 0
        elif our_hand_rank < opp_hand_rank:
            return 1
        else:
            return 2

    @property
    def opp_hand(self) -> List[int]:
        """Get random card.

        Returns
        -------
            Two cards for the opponent (Card)
        """
        return np.random.choice(self.available_cards, 2, replace=False)


class GameUtilityAbstract:
    """This class takes care of some game related functions.
    Abstracts cards into integers from the start."""

    def __init__(self, our_hand: list, board: list, cards: list, evaluator: Evaluator):
        # MEOW try to use global
        # self._evaluator = Evaluator()
        self._evaluator = evaluator
        unavailable_cards = [board+our_hand]
        self.available_cards = [c for c in cards if c not in unavailable_cards]        
        self.our_hand = our_hand
        self.board = board

    def evaluate_hand(self, hand: list) -> int:
        """
        Evaluate a hand.

        Parameters
        ----------
        hand : np.ndarray
            Hand to evaluate.

        Returns
        -------
            Evaluation of hand
        """
        return self._evaluator.evaluate(
            board=self.board,
            cards=hand,
        )

    def get_winner(self) -> int:
        """Get the winner.

        Returns
        -------
            int of win (0), lose (1) or tie (2) - this is an index in the
            expected hand strength array
        """
        our_hand_rank = self.evaluate_hand(self.our_hand)
        opp_hand_rank = self.evaluate_hand(self.opp_hand)
        if our_hand_rank > opp_hand_rank:
            return 0
        elif our_hand_rank < opp_hand_rank:
            return 1
        else:
            return 2

    @property
    def opp_hand(self) -> List[int]:
        """Get random card.

        Returns
        -------
            Two cards for the opponent (Card)
        """
        return random.sample(self.available_cards, 2)
