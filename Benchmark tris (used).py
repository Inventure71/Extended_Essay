import os
import random
import numpy as np
import pandas as pd
#from tensorflow.python.layers.normalization import BatchNormalization

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
from tensorflow.keras.models import load_model
import pandas as pd
import time

class NNPlayer:
    def __init__(self, name, num_layers, layer_size):
        self.name = name
        self.model_file = os.path.join("tris_models_2", f"{self.name}_layers_{num_layers}_size_{layer_size}_best_model.h5")
        if os.path.exists(self.model_file):
            print(f"Loading model from {self.model_file}")
            self.model = load_model(self.model_file)
        else:
            print(f"Model not found for: {self.model_file}")
        self.epsilon = 0


    def choose_action(self, state, possible_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(possible_actions)
        q_values = self.model.predict(np.array(state).reshape(-1, 9))
        for i in range(9):
            if i not in possible_actions:
                q_values[0][i] = -np.inf
        return np.argmax(q_values[0])

def get_possible_actions(board):
    return [i*3 + j for i in range(3) for j in range(3) if board[i][j] == 0]

def check_for_tris(board, player):
    # Check rows, columns and diagonals
    return any(all(cell == player for cell in row) for row in board) or \
           any(all(row[i] == player for row in board) for i in range(3)) or \
           all(board[i][i] == player for i in range(3)) or \
           all(board[i][2 - i] == player for i in range(3))

def check_for_draw(board):
    return all(cell != 0 for row in board for cell in row)

def play_game(player1, player2):
    board = [[0, 0, 0] for _ in range(3)]
    while True:
        for player in [player1, player2]:
            state = [cell for row in board for cell in row]
            action = player.choose_action(state, get_possible_actions(board))
            board[action // 3][action % 3] = 1 if player == player1 else -1
            if check_for_tris(board, 1 if player == player1 else -1):
                return player.name
            if check_for_draw(board):
                return "draw"


layers_sizes = [32, 64, 128, 256, 512, 1024]
num_layers = [2, 3]

# Exclude the model 3, 1024
excluded_models = [('Player1', 3, 1024), ('Player2', 3, 1024)]


def benchmark_models(layers_sizes, num_layers, num_games=10):
    results = []
    for n_layers in num_layers:
        for layer_size in layers_sizes:
            for player_num in ['Player1', 'Player2']:
                if (player_num, n_layers, layer_size) in excluded_models:
                    continue

                current_player = NNPlayer(player_num, n_layers, layer_size)
                for opp_n_layers in num_layers:
                    for opp_layer_size in layers_sizes:
                        for opp_player_num in ['Player1', 'Player2']:
                            # Modified condition to allow A vs A matches
                            if (player_num, n_layers, layer_size) > (opp_player_num, opp_n_layers, opp_layer_size):
                                continue

                            if (opp_player_num, opp_n_layers, opp_layer_size) in excluded_models:
                                continue

                            opponent = NNPlayer(opp_player_num, opp_n_layers, opp_layer_size)
                            wins, losses, draws = 0, 0, 0
                            for _ in range(num_games):
                                winner = play_game(current_player, opponent)
                                if winner == current_player.name:
                                    wins += 1
                                elif winner == "draw":
                                    draws += 1
                                else:
                                    losses += 1
                            results.append({
                                'Model': f"{player_num}_layers_{n_layers}_size_{layer_size}",
                                'Opponent': f"{opp_player_num}_layers_{opp_n_layers}_size_{opp_layer_size}",
                                'Wins': wins,
                                'Losses': losses,
                                'Draws': draws
                            })
    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)

benchmark_models(layers_sizes, num_layers)

