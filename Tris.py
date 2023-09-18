import numpy as np
import random
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

class Stats:
    def __init__(self, player, num_layers, layer_size):
        self.player = player
        self.stats_dir = "tris_stats"
        self.stats_file = os.path.join(self.stats_dir, f"{self.player.name}_layers_{num_layers}_size_{layer_size}_stats.csv")
        #self.stats_file = os.path.join(self.stats_dir, f"{self.player.name}_stats.csv")

        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        self.columns = ["Game", "Win/Loss", "Actions", "Rewards", "Epsilon", "Time"]
        if not os.path.exists(self.stats_file):
            self.data = pd.DataFrame(columns=self.columns)
            self.data.to_csv(self.stats_file, index=False)
        else:
            self.data = pd.read_csv(self.stats_file)

    def update_stats(self, game, win_loss, actions, rewards, epsilon, time_taken):
        new_row = pd.DataFrame([[game, win_loss, actions, rewards, epsilon, time_taken]], columns=self.columns)
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self.data.to_csv(self.stats_file, index=False)

class NNPlayer:
    def __init__(self, name, num_layers, layer_size, learning_rate=0.01, discount_rate=0.95, l2_reg=0.01, dropout_rate=0.2, epsilon=1.0, epsilon_decay=0.95, min_epsilon=0.01):
        self.name = name
        self.num_layers = num_layers
        self.layer_size = layer_size

        #self.model_file = f"{self.name}_best_model.h5"
        #self.model_file = os.path.join("tris_models", f"{self.name}_best_model.h5")
        self.model_file = os.path.join("tris_models", f"{self.name}_layers_{num_layers}_size_{layer_size}_best_model.h5")

        if os.path.exists(self.model_file):
            print(f"Loading model from {self.model_file}")
            self.model = load_model(self.model_file)
        else:
            self.model = self.create_model(learning_rate, l2_reg, dropout_rate)

        self.discount_rate = discount_rate
        self.memory = []

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def create_model(self, learning_rate, l2_reg, dropout_rate):
        model = Sequential()
        for _ in range(self.num_layers):
            model.add(Dense(self.layer_size, input_dim=9, activation='relu', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        model.add(Dense(9, activation='linear', kernel_regularizer=l2(l2_reg)))
        model.compile(loss='mse', optimizer=Adam(learning_rate))
        return model

    def choose_action(self, state, possible_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(possible_actions)
        q_values = self.model.predict(np.array(state).reshape(-1, 9))
        for i in range(9):
            if i not in possible_actions:
                q_values[0][i] = -np.inf
        return np.argmax(q_values[0])

    def remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

    def learn(self, batch_size, checkpoint):
        # decrement epsilon after each game
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)
        for current_state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.discount_rate * np.amax(self.model.predict(np.array(next_state).reshape(-1, 9))[0])
            target_f = self.model.predict(np.array(current_state).reshape(-1, 9))
            target_f[0][action] = target
            self.model.fit(np.array(current_state).reshape(-1, 9), target_f, epochs=1, verbose=0, callbacks=[lrate, checkpoint])

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

def get_possible_actions(board):
    possible_actions = []
    for row in range(0, 3):
        for cell in range(0, 3):
            if int(board[row][cell]) == 0:
                possible_actions.append(row * 3 + cell)
    return possible_actions

def check_horizontal(board, player):
    for k in range(0,3):
        if "".join(str(x) for x in board[k]) == (str(player) * 3):
            return True
    return False

def check_vertical(board, player):
    for i in range(0,3):
        array = [board[k][i] for k in range(0, 3)]
        if "".join(str(x) for x in array) == (str(player) * 3):
            return True
    return False

def check_trasversal(board, player):
    if (str(board[0][0]) + str(board[1][1]) + str(board[2][2]) == str(player) * 3) or \
        (str(board[2][0]) + str(board[1][1]) + str(board[0][2]) == str(player) * 3):
        return True
    return False

def check_for_tris(board, player):
    return check_horizontal(board, player) or check_vertical(board, player) or check_trasversal(board, player)

def check_for_draw(board):
    return len(get_possible_actions(board)) == 0

if __name__ == "__main__":

    layers_sizes = [32, 64, 128, 256, 512, 1024]
    num_layers = [2, 3, 4, 5]

    if not os.path.exists("tris_models"):
        os.makedirs("tris_models")

    # create learning rate schedule callback
    lrate = LearningRateScheduler(step_decay)

    for n_layers in num_layers:
        for layer_size in layers_sizes:
            player1 = NNPlayer('Player1', n_layers, layer_size)
            player2 = NNPlayer('Player2', n_layers, layer_size)

            stats1 = Stats(player1, n_layers, layer_size)
            stats2 = Stats(player2, n_layers, layer_size)
            num_games = 7500 #2500
            batch_size = 32

            # model checkpoint
            checkpoint1 = ModelCheckpoint(player1.model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
            checkpoint2 = ModelCheckpoint(player2.model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')

            for game in range(num_games):
                start_time = time.time()
                board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                win_loss = None
                actions = []
                rewards = []

                current_player = player1 if game % 2 == 0 else player2

                while True:
                    state = [item for sublist in board for item in sublist]  # flatten the board
                    possible_actions = get_possible_actions(board)
                    cell = current_player.choose_action(state, possible_actions)
                    actions.append(cell)

                    if cell in possible_actions:
                        board[cell // 3][cell % 3] = 1 if current_player == player1 else -1
                        if check_for_tris(board, 1 if current_player == player1 else -1):
                            print(f"Player {current_player.name} won the game!")
                            win_loss = 1
                            rewards.append(1)
                            current_player.remember(state, cell, 1, state, True)
                            break
                        elif check_for_draw(board):
                            print("Game is a draw!")
                            win_loss = 0
                            rewards.append(0)
                            current_player.remember(state, cell, 0, state, True)
                            break

                        else:
                            current_player.remember(state, cell, 0, state, False)
                            rewards.append(0)
                            current_player = player1 if current_player == player2 else player2

                    else:
                        current_player.remember(state, cell, -1, state, True)

                time_taken = time.time() - start_time

                # Update stats for Player 1
                if current_player == player1:
                    stats1.update_stats(game, win_loss, actions, rewards, current_player.epsilon, time_taken)
                else:
                    stats1.update_stats(game, -win_loss, actions, rewards, player1.epsilon, time_taken)  # Player 1 lost or drew the game

                # Update stats for Player 2
                if current_player == player2:
                    stats2.update_stats(game, win_loss, actions, rewards, current_player.epsilon, time_taken)
                else:
                    stats2.update_stats(game, -win_loss, actions, rewards, player2.epsilon, time_taken)  # Player 2 lost or drew the game


                if game % batch_size == 0:
                    player1.learn(batch_size, checkpoint1)
                    player2.learn(batch_size, checkpoint2)
                    # save final model weights
                    # Save the model

                    player1.model.save(player1.model_file)
                    player2.model.save(player2.model_file)
