poker_ai train start --lut_path ./cardinfos/20card_string --nickname tmp --pickle_dir True --sync_serialise


# Playing game on demo instance
poker_ai play --lut_path /ebs_volume_shared/card_info/ --strategy_path /ebs_volume_shared/trained_agents/deliver_v1/agent.joblib --n_players 4 --low_card_rank 2 --initial_chips 10000

