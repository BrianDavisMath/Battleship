# Python 3.7

import warnings
warnings.simplefilter(action='ignore')
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from more_itertools import consecutive_groups

# subprocess.call("initialize.sh", shell=True)


board_side_length = 10
ship_lengths = {'carrier':5, 'battleship':4, 'destroyer':3, 'submarine':3, 'patrol boat':2}
points_to_win = sum(ship_lengths.values())


def string_to_coords(s):
    row = ord(s[0].upper()) - 65
    col = int(s[1:]) - 1
    return row, col


def coords_to_string(c):
    row = chr(c[0] + 65)
    col = str(c[1] + 1)
    return row + col


def random_guess(foggy_board):
    rows, cols = np.where(foggy_board == 0)
    index = np.random.choice(range(len(rows)))
    return rows[index], cols[index]


def get_random_placement():
    user_input = None
    while user_input is None:
        try:
            random_place = input("Do you want to place pieces randomly? ('Y' or 'N'): ").upper()
            if random_place == 'Y':
                user_input = True
            elif random_place == 'N':
                user_input = False
            else:
                print('Invalid input...')
                continue
        except:
            print('Invalid input...')
            continue
    return user_input


class Player:
    def __init__(self):
        self.board = - np.ones((board_side_length, board_side_length))
        self.opponent_board = np.zeros((board_side_length, board_side_length))
        self.opponent_guesses = np.zeros((board_side_length, board_side_length))

    def get_ship_placement(self, ship_type):
        ship_length = ship_lengths[ship_type]
        ship_placed = False
        while not ship_placed:
            try:
                square = input("Which square contains the top-left-most part of your {}? ".format(ship_type))
                orientation = input(
                    "Which way is your {} oriented? (e.g. 'Down' or 'Right'): ".format(ship_type)).upper()
                row, col = string_to_coords(square)
                if orientation == 'RIGHT':
                    support = np.sum(self.board[row, col: col + ship_length])
                elif orientation == 'DOWN':
                    support = np.sum(self.board[row: row + ship_length, col])
                else:
                    print("Invalid orientation! 'Right' or 'Down'...")
                    pass
                if support != (- ship_length):
                    print("Invalid placement! Ship leaves the board or hits another ship!")
                    pass
                else:
                    if orientation == 'RIGHT':
                        self.board[row, col: col + ship_length] *= -1
                    else:
                        self.board[row: row + ship_length, col] *= -1
                    ship_placed = True
            except:
                print('Invalid ship placement! A typo?')
                pass

    def place_ships(self, random):
        for ship in ship_lengths.keys():
            if not random:
                self.get_ship_placement(ship)
            else:
                ship_length = ship_lengths[ship]
                available_rows, available_cols = np.where(self.board == -1)
                ship_placed = False
                while not ship_placed:
                    orientation = np.random.choice(['RIGHT', 'DOWN'])
                    if orientation == 'RIGHT':
                        row = np.random.randint(0, board_side_length)
                        available_cols_tmp = available_cols[available_rows == row]
                        clumps = []
                        for clump in consecutive_groups(available_cols_tmp):
                            check = list(clump)
                            if len(check) >= ship_length:
                                clumps.append(check)
                        if len(clumps) == 0:
                            continue
                        else:
                            clump = clumps[np.random.choice(len(clumps))]
                            col = clump[0] + np.random.choice(range(len(clump) - ship_length + 1))
                            self.board[row, col: col + ship_length] *= -1
                            ship_placed = True
                    else:
                        col = np.random.randint(0, board_side_length)
                        available_rows_tmp = available_rows[available_cols == col]
                        clumps = []
                        for clump in consecutive_groups(available_rows_tmp):
                            check = list(clump)
                            if len(check) >= ship_length:
                                clumps.append(check)
                        if len(clumps) == 0:
                            continue
                        else:
                            clump = clumps[np.random.choice(len(clumps))]
                            row = clump[0] + np.random.choice(range(len(clump) - ship_length + 1))
                            self.board[row: row + ship_length, col] *= -1
                            ship_placed = True

    def get_guess(self):
        guess = None
        while guess is None:
            try:
                user_input = input("\nType your next guess (e.g. 'A2 <ENTER>'): ")
                x, y = string_to_coords(user_input)
                if self.opponent_guesses[x, y] != 0:
                    print('You already guessed that!')
                    continue
                elif (x not in range(board_side_length)) or (y not in range(board_side_length)):
                    print('Invalid guess!')
                    continue
                else:
                    guess = x, y
            except:
                continue
        return guess


def main():
    learningRate = 1e-4
    balanceWeight = 10
    # Network structure
    features = tf.placeholder(tf.float64, (None, 100))
    labels = tf.placeholder(tf.float64, (None, 100))
    hidden_layer = tf.contrib.layers.fully_connected(features, 100)
    predictedLabels = tf.contrib.layers.fully_connected(hidden_layer, 100, activation_fn=None)

    # Cost function

    BCE = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=(1 + labels) / 2,
                                                                  logits=predictedLabels,
                                                                  pos_weight=balanceWeight))
    l2RegTerm = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cost = BCE + l2RegTerm

    # Optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    new_game = True

    new_training_features = []
    new_training_labels = []
    while new_game:
        human = Player()
        computer = Player()
        human_turn = True
        num_computer_hits = 0
        num_human_hits = 0
        computer.place_ships(random=True)
        rand = get_random_placement()
        human.place_ships(random=rand)
        print('Here is your board: \n {}'.format((human.board + 1)/2))
        print("\n\nLet's get started...")
        # flat_human_board = np.reshape(human.board, (1,100))
        # flat_computer_board = np.reshape(computer.board, (1, 100))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "model/model.ckpt")
            while (num_computer_hits < points_to_win) and (num_human_hits < points_to_win):
                if human_turn:
                    human_guess = computer.get_guess()
                    guess_result = computer.board[human_guess]
                    human.opponent_board[human_guess] = guess_result
                    computer.opponent_guesses[human_guess] = 1
                    if guess_result == 1:
                        print('Hit!')
                    else:
                        print('Miss!')
                    human_turn = False
                    num_human_hits = np.sum(human.opponent_board[human.opponent_board == 1]).astype(int)
                    new_training_features.append(human.opponent_board.flatten())
                    new_training_labels.append(computer.board.flatten())
                if not human_turn:
                    if num_human_hits < points_to_win:
                        foggy_board = np.reshape(computer.opponent_board, (1, 100))
                        prediction = sess.run(predictedLabels, feed_dict={features: foggy_board})
                        max_value = np.max(prediction[foggy_board == 0])
                        index = np.where((foggy_board == 0) & (prediction == max_value))[1][0]
                        guess_row = index // 10
                        guess_col = index % 10
                        computer_guess = (guess_row, guess_col)
                        print('\nComputer guesses: {}'.format(coords_to_string(computer_guess)))
                        result = human.board[computer_guess]
                        computer.opponent_board[computer_guess] = result
                        if result == 1:
                            print('Hit!')
                        else:
                            print('Miss!')
                        human_turn = True
                        num_computer_hits = np.sum(computer.opponent_board[computer.opponent_board == 1]).astype(int)
                        new_training_features.append(computer.opponent_board.flatten())
                        new_training_labels.append(human.board.flatten())
                data_features = np.vstack(new_training_features)
                data_labels = np.vstack(new_training_labels)
        if num_computer_hits == 17:
            print('Computer wins!')
        else:
            print('You win!')
        play_again = input('New Game? (Y / N)').upper()
        if play_again == 'Y':
            new_game = True
        else:
            new_game = False

            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, "model/model.ckpt")
                sess.run(opt, feed_dict={features: data_features, labels: data_labels})
                saver.save(sess, "model/model.ckpt")


if __name__ == "__main__":
    main()