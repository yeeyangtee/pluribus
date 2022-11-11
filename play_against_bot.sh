#!/bin/bash
pip install ./pluribus
poker_ai play --lut_path ./data --strategy_path ./data/deliver_v2/agent.joblib --n_players 4 --low_card_rank 2 --initial_chips 10000
