import logging
from typing import List
from itertools import combinations
import operator
import time

import numpy as np
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits

from multiprocessing import Manager,Process, Pool
import os


log = logging.getLogger("poker_ai.clustering.runner")


def process_combos(combos,publics):
    our_cards = []
    # Descending sort combos.
    sorted_combos: List[Card] = sorted(
        list(combos),
        key=operator.attrgetter("eval_card"),
        reverse=True,
    )
    out = []
    for public_combo in publics:
        # Descending sort public_combo.
        sorted_public_combo: List[Card] = sorted(
            list(public_combo),
            key=operator.attrgetter("eval_card"),
            reverse=True,
        )
        if not np.any(np.isin(sorted_combos, sorted_public_combo)):
            # Combine hand and public cards.
            hand: np.array = np.array(
                sorted_combos + sorted_public_combo
            )
            our_cards.append(hand)
    return our_cards

def process_publics(public_combo, sorted_combos):
    # print('PUBCOMBO',"="*100,'\n',public_combo)
    # print('SORTCOMBO',"="*100,'\n',sorted_combos)
    if not np.any(np.isin(sorted_combos, public_combo)):
        # Combine hand and public cards.
        hand: np.array = np.array(
            sorted_combos + public_combo
        )
        # print('HAND',"="*100,'\n', hand)
        return hand
    return None

class CardCombos:
    """This class stores combinations of cards (histories) per street."""

    def __init__(
        self, low_card_rank: int, high_card_rank: int,
    ):
        super().__init__()
        # Sort for caching.
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(low_card_rank, high_card_rank + 1)))
        self._cards = np.array(
            [Card(rank, suit) for suit in suits for rank in ranks]
        )
        self.starting_hands = self.get_card_combos(2)
        self.flop = self.create_info_combos(
            self.starting_hands, self.get_card_combos(3)
        )
        log.info("created flop")
        self.turn = self.create_info_combos(
            self.starting_hands, self.get_card_combos(4)
        )
        log.info("created turn")
        self.river = self.create_info_combos(
            self.starting_hands, self.get_card_combos(5)
        )
        log.info("created river")

    def get_card_combos(self, num_cards: int) -> np.ndarray:
        """
        Get the card combinations for a given street.

        Parameters
        ----------
        num_cards : int
            Number of cards you want returned

        Returns
        -------
            Combos of cards (Card) -> np.ndarray
        """
        return np.array([c for c in combinations(self._cards, num_cards)])


    

    def create_info_combos(
        self, start_combos: np.ndarray, publics: np.ndarray
    ) -> np.ndarray:
        """Combinations of private info(hole cards) and public info (board).

        Uses the logic that a AsKsJs on flop with a 10s on turn is the same
        as AsKs10s on flop and Js on turn. That logic is used within the
        literature as well as the logic where those two are different.

        Parameters
        ----------
        start_combos : np.ndarray
            Starting combination of cards (beginning with hole cards)
        publics : np.ndarray
            Public cards being added
        Returns
        -------
            Combinations of private information (hole cards) and public
            information (board)
        """
        if publics.shape[1] == 3:
            betting_stage = "flop"
        elif publics.shape[1] == 4:
            betting_stage = "turn"
        elif publics.shape[1] == 5:
            betting_stage = "river"
        else:
            betting_stage = "unknown"
        our_cards: List[Card] = []
        # COMMENT Multip on starting combos

        # with Pool(os.cpu_count()-1) as p:
        #     our_cards += p.starmap(process_combos, map(lambda x: (x,publics),start_combos))

        # print(np.array(our_cards).shape)
        # print(np.array(our_cards).reshape(-1, publics.shape[1]+2).shape)
        # return np.array(our_cards).reshape(-1, publics.shape[1]+2)

        # preprocess and do all sorting
        log.info(f"Preprocessing {betting_stage} combos")
        start_time = time.time()

        sorted_publics = []
        for public_combo in publics:
            sorted_public_combo: List[Card] = sorted(
            list(public_combo),
            key=operator.attrgetter("eval_card"),
            reverse=True)   
            sorted_publics.append(sorted_public_combo)
        sorted_publics = np.array(sorted_publics)
        end_time = time.time()
        log.info(f"Time to preprocess {betting_stage} combos: {end_time - start_time:.2f} seconds")


        # COMMENT multip on public combos
        start_time = time.time()
        with Pool(os.cpu_count()-1) as p:
            for combo in tqdm(start_combos,desc=betting_stage):

                sorted_combos: List[Card] = sorted(
                list(combo),
                key=operator.attrgetter("eval_card"),
                reverse=True,)

                tmp = p.starmap(process_publics, map(lambda x: (list(x),sorted_combos),sorted_publics))
                    
                tmp = [x for x in tmp if x is not None]
                our_cards.extend(tmp) 
            
        print(np.array(our_cards).shape)

        end_time = time.time()
        log.info(f"Time to create {betting_stage} combos: {end_time - start_time:.2f} seconds")
        return np.array(our_cards).reshape(-1, publics.shape[1]+2)

        # COMMENT Try with manager method
        # with Manager() as manager:
        #     # create the shared list
        #     our_cards = manager.list()
        #     # create many child processes

        #     max_jobs_running = os.cpu_count()-1
        #     jobs_running = 0
        #     processes = []
        #     for combos in tqdm(start_combos,desc=betting_stage):
        #         p = Process(target=process_combos, args=(combos,publics,our_cards))
        #         p.start()
        #         processes.append(p)


        #         jobs_running += 1

        #         if jobs_running >= max_jobs_running:
        #             while jobs_running >= max_jobs_running:
        #                 jobs_running = 0
        #                 for p in processes:
        #                     jobs_running += p.is_alive()

        #     for p in processes:
        #         p.join()
        # print(list(our_cards))
        # print(np.array(list(our_cards)).shape)
        # return np.array(our_cards)



        for combos in tqdm(
            start_combos,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            # Descending sort combos.
            sorted_combos: List[Card] = sorted(
                list(combos),
                key=operator.attrgetter("eval_card"),
                reverse=True,
            )
            for public_combo in publics:
                # Descending sort public_combo.
                sorted_public_combo: List[Card] = sorted(
                    list(public_combo),
                    key=operator.attrgetter("eval_card"),
                    reverse=True,
                )
                if not np.any(np.isin(sorted_combos, sorted_public_combo)):
                    # Combine hand and public cards.
                    hand: np.array = np.array(
                        sorted_combos + sorted_public_combo
                    )
                    our_cards.append(hand)
        print(np.array(our_cards).shape)

        end_time = time.time()
        log.info(f"Time to create {betting_stage} combos: {end_time - start_time:.2f} seconds")


        return np.array(our_cards)
