import queue
from abc import ABC, abstractmethod
from typing import Callable, Optional, Type

class BaseState(ABC):
    
    @abstractmethod 
    def __init__(self, parent: Optional[Type['BaseState']] = None) -> None:
        pass
    
    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __eq__(self, other: 'BaseState') -> bool:
        if isinstance(other, BaseState):
            return other.__hash__()==self.__hash__()
        else:
            return False

    @abstractmethod 
    def explore(self) -> list[Type['BaseState']]:
        pass

class BaseSearch(ABC):

    def __init__(self) -> None:
        self.solution : Optional[Type[BaseState]] = None
        self.num_visited_states : int = 0
        self._priority : int = 0
    
    def _default_f_priority(self, s: Optional[type[BaseState]] = None) -> int:
        self._priority += 1
        return self._priority

    @abstractmethod
    def objective_search(self, initial_state: Type[BaseState], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[[Type[BaseState]], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
        """
        This function finds the first solution

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_goal: function that checks whether a state represents a possible solution
        :type f_goal: Callable[[Type[BaseState]], bool]
        :param f_stop: function that stops the search if a state respects some criteria
        :type f_stop: Optional[Callable[[Type[BaseState]], bool]]
        :param f_priority: function that decides the priority of the states in the frontier, determines the type of search
        :type f_priority: Optional[Callable[[Type[BaseState]], float]]
        """
        pass

    @abstractmethod
    def best_search(self, initial_state: Type[BaseState], f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float], f_goal: Callable[[Type[BaseState]], bool], f_stop: Optional[Callable[[Type[BaseState]], bool]] = None, f_priority: Optional[Callable[[Type[BaseState]], float]] = None) -> None:
        """
        This function finds the best solution under certain constraints

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_evaluate: function that compares two possible solutions
        :type f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float]
        :param f_goal: function that checks whether a state represents a possible solution
        :type f_goal: Callable[[Type[BaseState]], bool]
        :param f_stop: function that stops the search if a state respects some criteria
        :type f_stop: Optional[Callable[[Type[BaseState]], bool]]
        :param f_priority: function that decides the priority of the states in the frontier, determines the type of search
        :type f_priority: Optional[Callable[[Type[BaseState]], float]]
        """

        pass


class TreeSearch(BaseSearch):

    def __init__(self) -> None:
        super().__init__()

    def objective_search(self, initial_state: Type[BaseState], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[[Type[BaseState]], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
        """
        This function finds the first solution

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_goal: function that checks whether a state represents a possible solution
        :type f_goal: Callable[[Type[BaseState]], bool]
        :param f_stop: function that stops the search if a state respects some criteria
        :type f_stop: Optional[Callable[[Type[BaseState]], bool]]
        :param f_priority: function that decides the priority of the states in the frontier, determines the type of search
        :type f_priority: Optional[Callable[[Type[BaseState]], float]]
        """

        current_state : Type[BaseState] = initial_state
        frontier : queue.PriorityQueue[(int, Type[BaseState])] = queue.PriorityQueue()
        found_solution: bool = False

        if f_priority is None:
            f_priority = self._default_f_priority
        
        frontier.put((f_priority(current_state), current_state))

        while not found_solution and not frontier.empty():

            self.num_visited_states += 1
            _, current_state = frontier.get()

            if f_goal(current_state):
                found_solution = True
                self.solution = current_state
            else:
                if f_stop is not None:
                    stop = f_stop(current_state)
                    if not stop:
                        for state in current_state.explore():
                            frontier.put((f_priority(state), state))
                else:
                    for state in current_state.explore():
                        frontier.put((f_priority(state), state))
    
    def best_search(self, initial_state: Type[BaseState], f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float], f_goal: Callable[[Type[BaseState]], bool], f_stop: Optional[Callable[[Type[BaseState]], bool]] = None, f_priority: Optional[Callable[[Type[BaseState]], float]] = None) -> None:
        """
        This function finds the best solution under certain constraints

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_evaluate: function that compares two possible solutions
        :type f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float]
        :param f_goal: function that checks whether a state represents a possible solution
        :type f_goal: Callable[[Type[BaseState]], bool]
        :param f_stop: function that stops the search if a state respects some criteria
        :type f_stop: Optional[Callable[[Type[BaseState]], bool]]
        :param f_priority: function that decides the priority of the states in the frontier, determines the type of search
        :type f_priority: Optional[Callable[[Type[BaseState]], float]]
        """
        current_state : Type[BaseState] = initial_state
        stop: bool = False
        self.solution : Optional[Type[BaseState]] = None
        frontier : queue.PriorityQueue[(int, Type[BaseState])] = queue.PriorityQueue()
        
        if f_priority is None:
            f_priority = self._default_f_priority

        frontier.put((f_priority(current_state), current_state))

        while not frontier.empty() and not stop:

            self.num_visited_states += 1
            _, current_state = frontier.get()

            if f_goal(current_state):
                if self.solution is None:
                    self.solution = current_state
                else:
                    if f_evaluate(self.solution, current_state):
                        self.solution = current_state
            
            if f_stop is not None:
                stop = f_stop(current_state)
                if not stop:
                    for state in current_state.explore():
                        frontier.put((f_priority(state), state))
            else:
                for state in current_state.explore():
                    frontier.put((f_priority(state), state))


class GraphSearch(BaseSearch):

    def __init__(self) -> None:
        super().__init__()
        self.visited_states : list[Type[BaseState]] = list()

    def objective_search(self, initial_state: Type[BaseState], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[[Type[BaseState]], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
        """
        This function finds the first solution

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_goal: function that checks whether a state represents a possible solution
        :type f_goal: Callable[[Type[BaseState]], bool]
        :param f_stop: function that stops the search if a state respects some criteria
        :type f_stop: Optional[Callable[[Type[BaseState]], bool]]
        :param f_priority: function that decides the priority of the states in the frontier, determines the type of search
        :type f_priority: Optional[Callable[[Type[BaseState]], float]]
        """

        current_state : Type[BaseState] = initial_state
        frontier : queue.PriorityQueue[(int, Type[BaseState])] = queue.PriorityQueue()
        found_solution: bool = False

        if f_priority is None:
            f_priority = self._default_f_priority
        
        self.visited_states.append(current_state)
        frontier.put((f_priority(current_state), current_state))

        while not found_solution and not frontier.empty():

            self.num_visited_states += 1
            _, current_state = frontier.get()
            self.visited_states.append(current_state)

            if f_goal(current_state):
                found_solution = True
                self.solution = current_state
            else:
                if f_stop is not None:
                    stop = f_stop(current_state)
                    if not stop:
                        for state in current_state.explore():
                            self.visited_states.append(state)
                            frontier.put((f_priority(state), state))
                else:
                    for state in current_state.explore():
                        if state not in self.visited_states:
                            self.visited_states.append(state)
                            frontier.put((f_priority(state), state))
    
    def best_search(self, initial_state: Type[BaseState], f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float], f_goal: Callable[[Type[BaseState]], bool], f_stop: Optional[Callable[[Type[BaseState]], bool]] = None, f_priority: Optional[Callable[[Type[BaseState]], float]] = None) -> None:
        """
        This function finds the best solution under certain constraints

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_evaluate: function that compares two possible solutions
        :type f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float]
        :param f_goal: function that checks whether a state represents a possible solution
        :type f_goal: Callable[[Type[BaseState]], bool]
        :param f_stop: function that stops the search if a state respects some criteria
        :type f_stop: Optional[Callable[[Type[BaseState]], bool]]
        :param f_priority: function that decides the priority of the states in the frontier, determines the type of search
        :type f_priority: Optional[Callable[[Type[BaseState]], float]]
        """
        current_state : Type[BaseState] = initial_state
        stop: bool = False
        self.solution : Optional[Type[BaseState]] = None
        frontier : queue.PriorityQueue[(int, Type[BaseState])] = queue.PriorityQueue()
        
        if f_priority is None:
            f_priority = self._default_f_priority

        frontier.put((f_priority(current_state), current_state))
        self.visited_states.append(current_state)

        while not frontier.empty() and not stop:

            self.num_visited_states += 1
            _, current_state = frontier.get()

            if f_goal(current_state):
                if self.solution is None:
                    self.solution = current_state
                else:
                    if f_evaluate(self.solution, current_state):
                        self.solution = current_state
            
            if f_stop is not None:
                stop = f_stop(current_state)
                if not stop:
                    for state in current_state.explore():
                        self.visited_states.append(state)
                        frontier.put((f_priority(state), state))
            else:
                for state in current_state.explore():
                    if state not in self.visited_states:
                        self.visited_states.append(state)
                        frontier.put((f_priority(state), state))


