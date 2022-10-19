'''A single use script to convert an existing folder of card LUT info from tuple(int,int...) representations to
a string representation. I.e. "12525672"... Important to keep the order of the cards consitent, which is defined by 
the create_card_lut function in card_info_lut_builder.py'''

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits
import pickle
from pathlib import Path

def create_card_lut(low_card_rank):
    suits = sorted(list(get_all_suits()))
    ranks = sorted(list(range(low_card_rank, 14 + 1))) # hardcode high card 
    cards = [int(Card(rank, suit)) for suit in suits for rank in ranks]
    cards.sort(reverse=True)

    lut = {}
    for i, c in enumerate(cards):
        lut[c] = f'{i:02d}'
    return lut 

def convert_card_info(srcdir, low_card_rank):
    srcdir = Path(srcdir)
    dstdir = srcdir.parent / f'{srcdir.name}_stringmode'
    dstdir.mkdir()

    betting_stages = ["pre_flop", "flop", "turn", "river"]

    for street in betting_stages:
        card_info_lut_path = srcdir/f'card_info_lut_{street}.pkl'
        new_card_info = {} 
        
        with open(card_info_lut_path,'rb') as f: 
            card_info = pickle.load(f)

            # Here we start processing the card info into string style.
            cardlut = create_card_lut(low_card_rank)
            for key, value in card_info.items():
                new_key = ''.join([cardlut[int(c)] for c in key])
                new_card_info[new_key] = int(value)
            
        # Save the new card info
        new_card_info_path = dstdir/f'card_info_lut_{street}.pkl'
        with open(new_card_info_path, 'wb') as f:
            pickle.dump(new_card_info, f)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', type=str, required=True)
    parser.add_argument('--low_card_rank', type=int, required=True)
    args = parser.parse_args()
    convert_card_info(args.srcdir, args.low_card_rank)

