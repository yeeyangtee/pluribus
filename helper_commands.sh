poker_ai train start --lut_path ./cardinfos/20card_string --low_card_rank 2 --nickname tmp --pickle_dir True --sync_serialise


# Playing game on demo instance. Can change strategy path folder.
poker_ai play --lut_path /ebs_volume_shared/card_info/ --strategy_path /ebs_volume_shared/trained_agents/deliver_v1/agent.joblib --n_players 4 --low_card_rank 2 --initial_chips 10000


# Play with interactive
pip install .
poker_ai play --play_as_bot True --lut_path /ebs_volume_shared/card_info \
--strategy_path /ebs_volume_shared/trained_agents/deliver_v2/agent.joblib \
--low_card_rank 2 --n_players 6
