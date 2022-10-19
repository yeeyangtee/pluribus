"""
Usage: poker_ai cluster [OPTIONS]

  Run clustering.

Options:
  --low_card_rank INTEGER        The starting hand rank from 2 through 14 for
                                 the deck we want to cluster. We recommend
                                 starting small.
  --high_card_rank INTEGER       The starting hand rank from 2 through 14 for
                                 the deck we want to cluster. We recommend
                                 starting small.
  --n_river_clusters INTEGER     The number of card information buckets we
                                 would like to create for the river. We
                                 recommend to start small.
  --n_turn_clusters INTEGER      The number of card information buckets we
                                 would like to create for the turn. We
                                 recommend to start small.
  --n_flop_clusters INTEGER      The number of card information buckets we
                                 would like to create for the flop. We
                                 recommend to start small.
  --n_simulations_river INTEGER  The number of opponent hand simulations we
                                 would like to run on the river. We recommend
                                 to start small.
  --n_simulations_turn INTEGER   The number of river card hand simulations we
                                 would like to run on the turn. We recommend
                                 to start small.
  --n_simulations_flop INTEGER   The number of turn card hand simulations we
                                 would like to run on the flop. We recommend
                                 to start small.
  --save_dir TEXT                Path to directory to save card info lookup
                                 table and betting stage centroids.
  --help                         Show this message and exit.
"""
import click

from poker_ai.clustering.card_info_lut_builder import CardInfoLutProcessor, CardInfoLutStore


@click.command()
@click.option(
    "--low_card_rank",
    default=2,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--high_card_rank",
    default=14,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--n_river_clusters",
    default=200,
    help=(
        "The number of card information buckets we would like to create for "
        "the river. We recommend to start small."
    )
)
@click.option(
    "--n_turn_clusters",
    default=200,
    help=(
        "The number of card information buckets we would like to create for "
        "the turn. We recommend to start small."
    )
)
@click.option(
    "--n_flop_clusters",
    default=200,
    help=(
        "The number of card information buckets we would like to create for "
        "the flop. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_river",
    default=6,
    help=(
        "The number of opponent hand simulations we would like to run on the "
        "river. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_turn",
    default=6,
    help=(
        "The number of river card hand simulations we would like to run on the "
        "turn. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_flop",
    default=6,
    help=(
        "The number of turn card hand simulations we would like to run on the "
        "flop. We recommend to start small."
    )
)
@click.option(
    "--save_dir",
    default="",
    help=(
        "Path to directory to save card info lookup table and betting stage "
        "centroids."
    )
)
@click.option(
    "--save_mode",
    default='pickle',
    help=(
        "Whether to pickle the card lut info."
    )
)
@click.option(
    "--card_repr",
    default='string',
    help=(
        "What format to save the card info, |int|string|Card|"
    )
)

def cluster(n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        n_river_clusters: int,
        n_turn_clusters: int,
        n_flop_clusters: int,
        save_mode: str,
        save_dir: str,
        card_repr: str,):


    processor = CardInfoLutProcessor(n_simulations_river=n_simulations_river,
        n_simulations_turn=n_simulations_turn,
        n_simulations_flop=n_simulations_flop,
        low_card_rank=low_card_rank,
        high_card_rank=high_card_rank,
        save_dir=save_dir,
        card_repr=card_repr)
    storage = CardInfoLutStore(low_card_rank=low_card_rank,
        high_card_rank=high_card_rank,
        save_mode = save_mode,
        save_dir=save_dir)

    river = storage.get_unique_combos(5)
    turn = storage.get_unique_combos(4)
    flop = storage.get_unique_combos(3)

    preflop_lut = processor.compute_preflop(storage._starting_hands, card_repr)
    storage.save('pre_flop', preflop_lut)

    river_lut, river_centroids  = processor.compute_river(river, n_river_clusters)
    storage.save('river', river_lut, river_centroids)
    river_lut, river_centroids = None, None


    turn_lut, turn_centroids  = processor.compute_turn(turn, n_turn_clusters)
    storage.save('turn', turn_lut, turn_centroids)
    turn_lut, turn_centroids = None, None

    flop_lut, flop_centroids  = processor.compute_flop(flop, n_flop_clusters)
    storage.save('flop', flop_lut, flop_centroids)
    flop_lut, flop_centroids = None, None


if __name__ == "__main__":
    cluster()
