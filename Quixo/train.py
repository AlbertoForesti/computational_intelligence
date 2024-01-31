import random
from game import Game, Move, Player
from typing import Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy
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

class TransformDefaultDict(defaultdict):
    def __init__(self, default_factory=None, key_transform=None, make_hash=None, *args, **kwargs):
        self.key_transform = key_transform or (lambda x: x)
        self.make_hash = make_hash or (lambda x: x)
        self.transformed_keys = {}
        super().__init__(default_factory, *args, **kwargs)
    
    def get_and_set_transformed_key(self, key):
        hashable_key = self.make_hash(key)
        if hashable_key in self.transformed_keys:
            return self.transformed_keys[hashable_key]
        else:
            transformed_key = self.transformed_keys.get(hashable_key, self.key_transform(key))
            self.transformed_keys[hashable_key] = transformed_key
            return transformed_key

    def __getitem__(self, key):
        return super().__getitem__(self.get_and_set_transformed_key(key))

    def __setitem__(self, key, value):
        super().__setitem__(self.get_and_set_transformed_key(key), value)

    def __delitem__(self, key):
        super().__delitem__(self.get_and_set_transformed_key(key))
    
    def __call__(self, key) -> Any:
        return self.get_and_set_transformed_key(key)

class QLPlayer(Player):
    def __init__(self, lr, df, epsilon, player_id, embedding = None) -> None:
        super().__init__()
        self._train = True
        self.lr = lr # Learning rate
        self.df = df # Discount factor
        self.player_id = player_id # Player id
        # self.policy = defaultdict(lambda: defaultdict(lambda: np.random.uniform(low=-1, high=1))) # Q(s,a) table
        self.previous_board = None # Stores the previous board state
        self.previous_move = None # Stores the previous move
        self.epsilon = epsilon
        self.verbose = False
        self.p = 1

        if embedding is None:
            self.policy = TransformDefaultDict(lambda: defaultdict(lambda: np.random.uniform(low=-1, high=1)), lambda x: self.get_identity_embedding(x), lambda x: self.get_identity_embedding(x))
        elif embedding == 'graph':
            self.policy = TransformDefaultDict(lambda: defaultdict(lambda: np.random.uniform(low=-1, high=1)), lambda x: self.get_graph_embedding(x), make_hash=lambda x: self.get_identity_embedding(x))
        else:
            raise NotImplementedError(f'Embedding {embedding} not implemented')
    
    def get_identity_embedding(self, board):
        return tuple(np.array(board).flatten())

    def get_set_embedding(self, board):
        return tuple(sorted(set(np.array(board).flatten())))

    def get_graph_embedding(self, board):
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
        current_board = game.get_board()
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
                if self.previous_board == self.policy(game.get_board()):
                    # Means that player did an illegal move
                    move = self.make_random_move(game)
                else:
                    move = self.make_best_move(game)
            self.previous_board = self.policy(game.get_board())
            self.previous_move = move
            return move
        else:
            if self.previous_board == self.policy(game.get_board()):
                # Means that player did an illegal move
                move = self.make_random_move(game)
            else:
                move = self.make_best_move(game)
            self.previous_board = self.policy(game.get_board())
            self.previous_move = move
            return move

    def update_policy(self, game: Game, final=False) -> None:
        if self.previous_board is not None:
            q_values = self.policy[self.previous_board].values()
            if len(q_values) > 0:
                optimal_estimate = max(q_values)
            else:
                optimal_estimate = 0
            prev = self.policy[self.previous_board][self.previous_move]
            self.policy[self.previous_board][self.previous_move] = (1-self.lr)*self.policy[self.previous_board][self.previous_move]+\
                self.lr*(self.compute_reward(game)+self.df*optimal_estimate)
            succ = self.policy[self.previous_board][self.previous_move]
            if final:
                self.p *= self.epsilon
                self.policy[self.policy(game.get_board())][self.previous_move] = self.compute_reward(game)
                # print(self.previous_board,'\n|\nv\n',self.policy(game.get_board()))
                # print(list(self.policy[self.policy(game.get_board())].values()))
                # print(f'Previous: {prev}, Successor: {succ}, Reward: {self.compute_reward(game)}, update: {self.lr*(self.compute_reward(game)+self.df*optimal_estimate)}, lr: {self.lr}, df: {self.df}')
    
    def reset_params(self) -> None:
        self.previous_board = None
        self.previous_move = None
    
    def compute_reward(self, game: Game) -> float:
        if game.check_winner() == self.player_id:
            return 1
        elif game.check_winner() == -1:
            if self.policy(game.get_board()) == self.previous_board:
                # Punish illegal move or cycle
                return -1
            max_in_line_p1 = self.max_in_line(game.get_board(), self.player_id)
            max_in_line_p2 = self.max_in_line(game.get_board(), 1-self.player_id)
            if max_in_line_p1 > max_in_line_p2:
                return max_in_line_p1
            elif max_in_line_p1 < max_in_line_p2:
                return -max_in_line_p2
            else:
                return 0
        else:
            return -1
    
    def max_in_line(self, board, player_id):
        max_count = 0

        # Check rows
        for row in board:
            count = 0
            for cell in row:
                if cell == player_id:
                    count += 1
                    max_count = max(max_count, count)
                else:
                    count = 0

        # Check columns
        for col in zip(*board):
            count = 0
            for cell in col:
                if cell == player_id:
                    count += 1
                    max_count = max(max_count, count)
                else:
                    count = 0

        # Check diagonals
        diagonals = [board[::-1,:].diagonal(i) for i in range(-board.shape[0]+1,board.shape[1])]
        diagonals.extend(board.diagonal(i) for i in range(board.shape[1]-1,-board.shape[0],-1))

        for diag in diagonals:
            count = 0
            for cell in diag:
                if cell == player_id:
                    count += 1
                    max_count = max(max_count, count)
                else:
                    count = 0

        return max_count/5
    
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


class QLTask:

    def __init__(self, lr=0.1, df=0.9, epsilon=None, n_sim=1, num_matches=10000, test_freq=1000, test_matches=100, test_agent='random', embedding=None) -> None:
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
        self.embedding = embedding

        if epsilon is None:
            self.epsilon = 0.1**(1/num_matches)
            # self.epsilon = 0.99999
            print(f'Using epsilon={self.epsilon}')
    
    def test(self, player: Player) -> None:
        test_dict = defaultdict(int)
        test_dict_second_start = defaultdict(int)
        opponent = RandomPlayer()
        if isinstance(player, QLPlayer):
            player.train = False
        for _ in tqdm(range(self.test_matches), desc='Testing'):
            g = Game()
            outcome = g.play(player, opponent)
            test_dict[outcome] += 1

            g = Game()
            outcome=g.play(opponent, player)
            test_dict_second_start[1-outcome] += 1
        for i in range(2):
            self.metrics[0][i].append(test_dict[i])
            self.metrics[1][i].append(test_dict_second_start[i])
        if isinstance(player, QLPlayer):
            player.train = True
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

        player1 = QLPlayer(self.lr, self.df, self.epsilon, 0, embedding=self.embedding)
        player2 = RandomPlayer()
        player1.verbose = False

        policy_size = []
        ps = []

        print('Started training')

        for i in tqdm(range(self.num_matches), desc='Training'):
            g = Game()

            player1.player_id = 0
            g.play(player1, player2)
            player1.update_policy(g, final=True)
            ps.append(player1.p)
            policy_size.append(len(player1.policy))
            player1.reset_params()

            """g = Game()
            
            player1.player_id = 1
            g.play(player2, player1)
            player1.update_policy(g)
            player1.reset_params()"""
            
            if i%self.test_freq==0 and i > 0:
                print(f'Epoch {i}, testing')
                self.test(player1)
                print('Resuming training')
        # plt.plot(policy_size)
        plt.plot(ps)
        self.test(player1)
        self.display_stats()

class Agent:

    def __init__(self) -> None:
        self._genome = None
        self._fintness = None
    
    def __lt__(self, other: 'Agent'):
        return self._fitness < other._fitness

    def __hash__(self):
        return hash(str(self._genome))

    def __eq__(self, other):
        if isinstance(other, Agent):
            return str(self._genome) == str(other._genome)
        return False

    @property
    def genome(self):
        return self._genome

    @genome.setter
    def genome(self, p):
        self._genome = p
    
    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self._fitness = fitness
        self.explicit_fitness = True
    
    def set_implicit_fitness(self, fitness: float):
        self._fitness = fitness
        self.explicit_fitness = False
    
    def reset(self):
        self.fitness = 0
    
    def __iadd__(self, other) -> None:
        self._genome += other

class CrowdPlayer(Agent, Player):

    def __init__(self, players=None, path=None) -> None:
        super().__init__()
        print('Crowd player created')
        if path is not None:
            self.load(path)
        if players is not None:
            self.crowd = players
        if players is None and path is None:
            raise ValueError('Either players or path must be specified')
        self._id = 0

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        for player in self.crowd:
            player.id = value
        self._id = value

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        moves = [p.make_move(game) for p in self.crowd]
        m = Counter(moves).most_common()[0][0]
        return m

    def save(self, path):
        np.save(path, self.crowd, allow_pickle=True)

    def load(self, path):
        self.crowd = np.load(path, allow_pickle=True)


class EvolutionaryPlayer(Player, Agent):
    """
    A player that uses evolutionary algorithms to play
    """
    
    def __init__(self, genome=None, fitness=0, id=0, approach='total', path=None) -> None:
        Agent().__init__()
        if genome is not None:
            self.genome = genome
        else:
            self.genome = np.random.normal(size=(25,14))
        self.x = self.y = list(range(5))
        self.coords = list(set([(0,y) for y in self.y] + [(4,y) for y in self.y] + [(x,0) for x in self.x] + [(x,4) for x in self.x]))
        self.moves = [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]

        if approach == 'total':
            self.legal_moves = [(coord, move) for coord in self.coords for move in self.moves]
            if genome is not None:
                self.genome = genome
            else:
                self.genome = np.random.normal(size=(25,len(self.legal_moves)))

        if path is not None:
            self.genome = np.load(path)

        self.explicit_fitness = False
        self.fitness = fitness
        self.previous_board = None
        self.id=id
        self.approach = approach
    
    def save(self, path):
        np.save(path, self.genome)
    
    def load(self, path):
        self.genome.load(path)
    
    def make_random_move(self) -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        board = game.get_board().flatten()
        board = np.where([board == -1], board, 1-board) if id==1 else board
        if tuple(board) == self.previous_board:
            return self.make_random_move()
        else:
            self.previous_board = tuple(board)
        scores = np.dot(board, self.genome)

        if self.approach == 'total':
            probs = np.exp(scores)/np.sum(np.exp(scores))
            try:
                index = np.random.choice(len(self.legal_moves), p=probs)
                return self.legal_moves[index]
            except:
                return self.make_random_move()
        else:
            scores_x = scores[0:5]
            scores_y = scores[5:10]
            scores_from_pos = scores[10:14]
            prob_x = np.exp(scores_x)/np.sum(np.exp(scores_x))
            prob_y = np.exp(scores_y)/np.sum(np.exp(scores_y))
            prob_move = np.exp(scores_from_pos)/np.sum(np.exp(scores_from_pos))
            try:
                from_pos = (np.random.choice(self.x, p=prob_x), np.random.choice(self.y, p=prob_y))
                move = np.random.choice(self.moves, p=prob_move)
            except:
                return self.make_random_move()
            return from_pos, move

class EvolutionTask:

    def __init__(self, fitness, period_true_fitness: int = 5, sample_size_fitness: int = 10, novelty_factor: float = 0.01, test_freq = 10, test_matches = 100, self_play=False) -> None:
        self.is_island = False
        self.test_matches = test_matches
        self.fitness = fitness
        self.best_fitness = -np.inf
        self.average_fitness = -np.inf
        self.best_average_fitness = -np.inf
        self._best_agent = None
        self.fitness_list = []
        self.avg_fitness_list = []
        self.similarity_matrix = None
        self.test_freq = test_freq
        self.self_play = self_play
        self.metrics = [[[],[]],[[],[]]] # Stores win and losses for trained agent startinng first or second 
        self.set_fitness_calculation_params(period_true_fitness, sample_size_fitness, novelty_factor)
    
    def set_fitness_calculation_params(self, period_true_fitness: int = 5, sample_size_fitness: int = 10, novelty_factor: float = 1) -> None:
        self.period_true_fitness = period_true_fitness
        self.sample_size_fitness = sample_size_fitness
        self.novelty_factor = novelty_factor
    
    def mutate(self, agent: Agent, p: float = 0.001) -> Agent:
        """
        Mutates loci with random flipping and returns a new agent
        """
        new_genome = agent.genome
        new_genome += np.random.normal(loc=np.std(agent.genome)*p, size=agent.genome.shape)
        return EvolutionaryPlayer(genome=np.array(new_genome), fitness=np.random.normal(agent.fitness, scale=self.novelty_factor/(1+agent.fitness)))

    def calculate_fitness(self, explicit: bool=True) -> None:
        """
        Calculates the fitness of the current population
        """
        if explicit:
            best_in_gen = 0
            for agent in self.agents:
                agent.fitness = self.fitness(agent)
                if agent.fitness > best_in_gen:
                    best_in_gen = agent.fitness
                if agent.fitness > self.best_fitness:
                    self._best_agent = deepcopy(agent)
            if self.best_average_fitness is None or self.average_fitness > self.best_average_fitness:
                self.best_average_fitness = self.average_fitness
                self.best_crowd = deepcopy(self.agents)
            self.avg_fitness_list.append(self.average_fitness)
            self.fitness_list.append(best_in_gen)
        else:
            self.fitness_list.append(self.fitness_list[-1])
        self.average_fitness = np.mean([agent.fitness for agent in self.agents])

    
    def crossover(self, a1: Agent, a2: Agent, a: float=1, b: float=999, strategy='beta') -> Agent:
        """
        Given two agents it randomly selects the parameters between the two
        """
        if strategy=='beta':
            n = int(np.random.beta(a,b)*len(a1.genome))
            new_genome = np.concatenate((a1.genome[0:n], a2.genome[n:]))
            avg_fitness = (a1.fitness+a2.fitness)/2
            return EvolutionaryPlayer(genome=new_genome, fitness=np.random.normal(avg_fitness, scale=self.novelty_factor/(1+avg_fitness)))
        if strategy=='xor':
            return EvolutionaryPlayer(genome=np.logical_xor(a1.genome, a2.genome), fitness=np.random.normal(avg_fitness, scale=self.novelty_factor/(1+avg_fitness)))
    
    def es_iter(self, agents: Optional['np.array']=None, population_size: int=100, mu: int=30, strategy='comma', p=0.001) -> None:
        
        if agents is None:
            self.agents = np.partition(self.agents, population_size-mu)[population_size-mu:] # selective pressure, takes top mu agents
        else:
            self.agents = agents

        if strategy == 'comma':
            num_children = population_size
        else:
            num_children = population_size-mu
        
        parents = np.random.choice(self.agents, num_children)
        children = np.array([self.mutate(a, p) for a in parents])
        if strategy == 'comma':
            self.agents = children
        else:
            self.agents = np.concatenate((children, self.agents))
    
    def ga_iter(self, agents: Optional['np.array']=None, population_size: int=100, mu: int=30, strategy='comma', p=0.001, a_beta=1, b_beta=999) -> None:
        
        if agents is None:
            self.agents = np.partition(self.agents, population_size-mu)[population_size-mu:] # selective pressure, takes top mu agents
        else:
            self.agents = agents

        if strategy == 'comma':
            num_children = population_size
        else:
            num_children = population_size-mu
        
        parents = np.random.choice(self.agents, num_children)
        children = [self.crossover(a[0], a[1], a_beta, b_beta) for a in np.random.choice(parents, (num_children, 2))]
        children = np.array([self.mutate(a, p) for a in children])
        if strategy == 'comma':
            self.agents = children
        else:
            self.agents = np.concatenate((children, self.agents))
    
    def es(self, n_generations: int=100, population_size: int=100, mu: int=30, strategy='comma', p=0.001) -> None:
        self.is_island = False
        self.agents: 'np.array' = np.array([EvolutionaryPlayer() for _ in range(population_size)])
        progress_bar = tqdm(range(n_generations), desc='Training with evolutionary strategy.')
        for gen in progress_bar:
            self.calculate_fitness(explicit=gen%self.period_true_fitness==0)
            progress_bar.set_description('ES. Best fitness: {:.2f}. Average fitness: {:.2f}'.format(self.best_fitness, self.average_fitness))
            if self.test_freq is not None:
                if gen % self.test_freq == 0:
                    self.test(self.best_agent)
            self.es_iter(population_size=population_size, mu=mu, strategy=strategy, p=p)
        self.calculate_fitness()
    
    def genetic_algorithm(self, n_generations: int=100, population_size: int=100, mu: int=30, strategy='comma', p=0.001, a_beta=1, b_beta=999) -> None:
        self.is_island = False
        self.agents: 'np.array' = np.array([EvolutionaryPlayer() for _ in range(population_size)])
        flatness_factor = 1
        for gen in tqdm(range(n_generations), desc='Training with genetic algorithm'):
            self.calculate_fitness(explicit=gen%self.period_true_fitness==0)
            if gen > n_generations//10:
                flatness_factor = np.var(self.fitness_list[gen-n_generations//10:])+1e-3
            self.ga_iter(population_size=population_size, mu=mu, strategy=strategy, p=min(0.5,p/flatness_factor), a_beta=a_beta, b_beta=b_beta)
        self.calculate_fitness()
    
    def migrate(self, migration_size: int=1, hierarchical: bool=False) -> 'np.array':
        if hierarchical:
            population_size = len(self.agents)
            # migrants = np.partition(self.agents, population_size-migration_size)[population_size-migration_size:], np.partition(self.agents, migration_size)[:migration_size]
            migrants_indeces = np.concatenate([np.argpartition(self.agents, population_size-migration_size)[population_size-migration_size:], np.argpartition(self.agents, migration_size)[:migration_size]])
            best_indeces, worse_indeces = np.argpartition(self.agents, population_size-migration_size)[population_size-migration_size:], np.argpartition(self.agents, migration_size)[:migration_size]
            best_migrants = self.agents[best_indeces]
            worse_migrants = self.agents[worse_indeces]
            migrants = best_migrants, worse_migrants
            self.agents = self.agents[np.in1d(np.arange(len(self.agents)), np.concatenate([best_indeces, worse_indeces]), invert=True)]
        else:
            migrants_indeces = np.random.choice(np.arange(len(self.agents)), migration_size, replace=False)
            migrants = self.agents[migrants_indeces]
            self.agents = self.agents[np.in1d(np.arange(len(self.agents)), migrants_indeces, invert=True)]

        return migrants
    
    def receive_migrants(self, migrants: 'np.array'):
        self.agents = np.concatenate((self.agents, migrants))
    
    def island_model(self, n_generations: int=100, migration_period: int=1, migration_size: int=1, island_size: int=10, n_islands: int = 10, mu_island: int=3, strategy='comma', p=0.001, a_beta=1, b_beta=999, replace=False, hierarchical=False, mode='ga'):
        self.is_island = True
        self.islands: 'np.array' = [EvolutionTask(self.fitness, period_true_fitness=self.period_true_fitness, test_freq=None) for _ in range(n_islands)]
        flatness_factor = 1
        for i in tqdm(range(n_generations), desc='Training with island model'):
            if i > n_generations//10:
                flatness_factor = np.var(self.fitness_list[i-n_generations//10:])+1e-3
            for island in self.islands:
                if i == 0:
                    island.agents=np.array([EvolutionaryPlayer() for _ in range(island_size)])
                    if mode=='ga':
                        island.ga_iter(population_size=island_size, mu=mu_island, strategy=strategy, p=p, a_beta=a_beta, b_beta=b_beta)
                    else:
                        island.es_iter(population_size=island_size, mu=mu_island, strategy=strategy, p=p)
                else:
                    if mode=='ga':
                        island.ga_iter(population_size=island_size, mu=mu_island, strategy=strategy, p=p/flatness_factor, a_beta=a_beta, b_beta=b_beta)
                    else:
                        island.es_iter(population_size=island_size, mu=mu_island, strategy=strategy, p=p)
            if i%migration_period==0:
                if hierarchical:
                    for j, island in enumerate(self.islands):
                        best, worse = island.migrate(migration_size, hierarchical)
                        if j==0:
                            previous_worse = worse
                            previous_island = island
                            island.receive_migrants(best)
                        elif j==len(self.islands)-1:
                            island.receive_migrants(previous_worse)
                            island.receive_migrants(worse)
                            previous_island.receive_migrants(best)
                        else:
                            island.receive_migrants(previous_worse)
                            previous_island.receive_migrants(best)
                            previous_island=island
                else:
                    migrants = np.concatenate([island.migrate(migration_size) for island in self.islands])
                    migrants = np.random.choice(migrants, size=(n_islands, migration_size), replace=replace)
                    for j, island in enumerate(self.islands):
                        island.receive_migrants(migrants[j,:])
            for island in self.islands:
                if len(island.agents) != island_size:
                    raise UserWarning(f'Bug in code, wrong island size: {len(island.agents)} != {island_size}')
                island.calculate_fitness(explicit=i%self.period_true_fitness==0)
            if i % self.test_freq == 0:
                self.test(self.best_agent)
        for island in self.islands:
            island.calculate_fitness()
    
    def plot_fitness(self) -> None:
        if self.is_island:
            best_island_idx = np.argmax([isle.best_fitness for isle in self.islands])
            self.islands[best_island_idx].plot_fitness()
        else:
            plt.plot(self.fitness_list)
            plt.plot(self.avg_fitness_list)
            plt.legend(['Best fitness', 'Average fitness'])
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.show()
    
    @property
    def best_agent(self):
        if self.is_island:
            agent_list = [island.best_agent for island in self.islands]
            best = agent_list[np.argmax([island.best_fitness for island in self.islands])]
            return best
        return self._best_agent
    
    @property
    def best_fitness(self) -> float:
        if self.is_island:
            return max([island.best_fitness for island in self.islands])
        try:
            return max(self.fitness_list)
        except:
            return -np.inf
    
    @best_fitness.setter
    def best_fitness(self, f) -> None:
        self._best_fitness = f
    
    @property
    def fitness_calls(self) -> int:
        return self.fitness.calls
    
    def test(self, player: Player) -> None:
        test_dict = defaultdict(int)
        test_dict_second_start = defaultdict(int)
        opponent = RandomPlayer()
        for _ in range(self.test_matches):
            player.id=0
            g = Game()
            outcome = g.play(player, opponent)
            test_dict[outcome] += 1

            player.id=1
            g = Game()
            outcome=g.play(opponent, player)
            test_dict_second_start[1-outcome] += 1
        for i in range(2):
            self.metrics[0][i].append(test_dict[i])
            self.metrics[1][i].append(test_dict_second_start[i])
        player.verbose = False
    
    def get_wisdowm_of_crowds_agent(self) -> Agent:
        if self.is_island:
            agent_list = [island.get_wisdowm_of_crowds_agent() for island in self.islands]
            return CrowdPlayer(agent_list)
        return CrowdPlayer(self.best_crowd)
    
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

def fitness(player, n=10, opponent=RandomPlayer()):
    if not isinstance(player, Player):
        raise TypeError(f'{player} is not an instance of Player')
    total = 0
    for _ in range(n):
        g = Game()
        player.id=0
        outcome = g.play(player, opponent)
        total += (1-outcome)
        player.id=1
        g = Game()
        outcome = g.play(opponent, player)
        total += outcome
    return total/(2*n)

def test(player: Player, test_matches: int=100, opponent = RandomPlayer(), verbose: bool=False) -> tuple[int, int]:
    test_dict = defaultdict(int)
    test_dict_second_start = defaultdict(int)
    for _ in tqdm(range(test_matches), desc='Testing'):
        player.id=0
        g = Game()
        outcome = g.play(player, opponent)
        test_dict[outcome] += 1
        player.id=1
        g = Game()
        outcome=g.play(opponent, player)
        test_dict_second_start[1-outcome] += 1
    print(f'Wins starting first: {test_dict[0]/test_matches}')
    print(f'Wins starting second: {test_dict_second_start[0]/test_matches}')