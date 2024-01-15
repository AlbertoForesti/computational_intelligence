import random
from game import Game, Move, Player
from typing import Tuple
from collections import defaultdict
from random import choice
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self._verbose = False
    
    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        if self.verbose:
            print('Random player')
            game.print()
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class QLPlayer(Player):
    def __init__(self, lr, df, epsilon, player_id) -> None:
        super().__init__()
        self._train = True
        self.lr = lr # Learning rate
        self.df = df # Discount factor
        self.player_id = player_id # Player id
        self.policy = defaultdict(lambda: defaultdict(lambda: np.random.uniform(low=-1, high=1))) # Q(s,a) table
        self.previous_board = None # Stores the previous board state
        self.previous_move = None # Stores the previous move
        self.epsilon = epsilon
        self.verbose = False
        self.p = 1
    
    def create_graph_embedding(board):
        # Initialize a new graph
        graph = nx.Graph()

        # Iterate over the board
        for i in range(len(board)):
            for j in range(len(board[i])):
                # If the cell is occupied by the player, add it as a node
                if board[i][j] != -1:
                    graph.add_node((i, j), player_id=board[i][j])

        # Add edges between all pairs of nodes with the Manhattan distance as the weight
        for node1 in graph.nodes:
            for node2 in graph.nodes:
                if node1 != node2:
                    weight = abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
                    graph.add_edge(node1, node2, weight=weight)

        return nx.weisfeiler_lehman_graph_hash(graph, node_attr='player_id', edge_attr='weight')
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, value):
        self._train = value
    
    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        self._verbose = value
    
    @property
    def player_id(self):
        return self._player_id
    
    @player_id.setter
    def player_id(self, value):
        self._player_id = value

    def make_random_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
    
    def make_best_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_board = tuple(game.get_board().flatten())
        action_dict = self.policy[current_board]
        try:
            return max(action_dict.items(), key=lambda x: x[1])[0]
        except:
            return self.make_random_move(game)
    
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        
        if self.train:
            self.update_policy(game)
            # epsilon greedy
            if np.random.uniform(low=0, high=1) < self.p:
                move = self.make_random_move(game)
            else:
                if self.previous_board == tuple(game.get_board().flatten()):
                    # Means that player did an illegal move
                    move = self.make_random_move(game)
                else:
                    move = self.make_best_move(game)
            self.p *= self.epsilon
            self.previous_board = tuple(game.get_board().flatten())
            self.previous_move = move
            return move
        else:
            if self.previous_board == tuple(game.get_board().flatten()):
                # Means that player did an illegal move
                move = self.make_random_move(game)
            else:
                move = self.make_best_move(game)
            self.previous_board = tuple(game.get_board().flatten())
            self.previous_move = move
            return move

    def update_policy(self, game: Game) -> None:
        if self.previous_board is not None:    
            optimal_estimate = 0 if len(self.policy[self.previous_move].values())==0 else max(self.policy[self.previous_move].values())
            self.policy[self.previous_board][self.previous_move] = (1-self.lr)*self.policy[self.previous_board][self.previous_move]+\
                self.lr*(self.compute_reward(game)+self.df*optimal_estimate)
    
    def reset_params(self) -> None:
        self.previous_board = None
        self.previous_move = None
        self.p = 1
    
    def compute_reward(self, game: Game) -> float:
        if game.check_winner() == self.player_id:
            return 1
        elif game.check_winner() == -1:
            if tuple(game.get_board().flatten()) == self.previous_board:
                # Punish illegal move or cycle
                return -1
            return self.count_lines(game.get_board(), self.player_id)
        else:
            return -1
    
    def count_lines(self, board, player_id):
        # Initialize count to 0
        count = 0

        # Check rows
        for row in board:
            if any(cell == player_id for cell in row):
                count += 1

        # Check columns
        for col in zip(*board):
            if any(cell == player_id for cell in col):
                count += 1

        # Check diagonals
        if any(board[i][i] == player_id for i in range(len(board))):
            count += 1
        if any(board[i][len(board)-i-1] == player_id for i in range(len(board))):
            count += 1

        # Calculate the ratio
        total_lines = 2 * len(board) + 2
        ratio = count / total_lines

        return ratio


class TrainTask:

    def __init__(self, lr=0.1, df=0.9, epsilon=None, n_sim=1, num_matches=10000, test_freq=1000, test_matches=100, test_agent='random') -> None:
        """
        Set parameters for the task
        """
        self.lr = lr # Learning rate
        self.df = df # Discount factor
        self.epsilon = epsilon # Decay of probability of selecting random move
        self.num_matches = num_matches
        self.test_freq = test_freq
        self.test_matches = test_matches
        self.test_agent = test_agent
        self.n_sim = n_sim
        self.metrics = [[[],[]],[[],[]]] # Stores win and losses for trained agent startinng first or second 

        if epsilon is None:
            self.epsilon = 0.1**(1/num_matches)
            print(f'Using epsilon={self.epsilon}')
    
    def test(self, player: Player) -> None:
        test_dict = defaultdict(int)
        test_dict_second_start = defaultdict(int)
        opponent = RandomPlayer()
        for _ in tqdm(range(self.test_matches)):
            g = Game()
            outcome = g.play(player, opponent)
            test_dict[outcome] += 1

            g = Game()
            outcome=g.play(opponent, player)
            test_dict_second_start[1-outcome] += 1
        for i in range(2):
            self.metrics[0][i].append(test_dict[i])
            self.metrics[1][i].append(test_dict_second_start[i])
        player.verbose = False
    
    def display_stats(self) -> None:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.metrics[0][0], label='Wins', color='g')
        ax1.axhline(y=max(self.metrics[0][0]), color='g', linestyle='--', label='Maximum wins')
        ax1.text(1, max(self.metrics[0][0])+0.1, f'{max(self.metrics[0][0])}')
        ax1.plot(self.metrics[0][1], label='Losses', color='r')
        ax1.set_title('Outcomes as trained agent starts first')
        ax1.legend()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Count')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.metrics[1][0], label='Wins', color='g')
        ax2.axhline(y=max(self.metrics[1][0]), color='g', linestyle='--', label='Maximum wins')
        ax2.text(1, max(self.metrics[1][0])+0.1, f'{max(self.metrics[1][0])}')
        ax2.plot(self.metrics[1][1], label='Losses', color='r')
        ax2.set_title('Outcomes as trained agent starts second')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Count')
        ax2.legend()

        plt.show()

    def train(self) -> None:
        """
        Execute training
        """

        player1 = QLPlayer(self.lr, self.df, self.epsilon, 0)
        player2 = RandomPlayer()
        player1.verbose = False

        print('Started training')

        for i in tqdm(range(self.num_matches)):
            g = Game()

            g.play(player1, player2)
            player1.update_policy(g)
            player1.reset_params()
            
            player1.player_id = 1
            g.play(player2, player1)
            player1.update_policy(g)
            player1.reset_params()
            
            if i%self.test_freq==0 and i > 0:
                print(f'Epoch {i}, testing')
                self.test(player1)
                print('Resuming training')
        self.test(player1)
        self.display_stats()

if __name__ == '__main__':
    task = TrainTask(num_matches=10000, test_freq=1000, test_matches=100, test_agent='random')
    task.train()