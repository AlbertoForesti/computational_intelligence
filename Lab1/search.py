import queue
from abc import ABC, abstractmethod
from typing import Callable, Optional, Type, Hashable

class BaseState(ABC):
    
    def __init__(self, state_identifier: Hashable, priority: float) -> None:
        self.state_identifier = state_identifier
        self.priority = priority

    def __gt__(self, other) -> bool:
        if isinstance(other, BaseState):
            return self.priority > other.priority
        else:
            # Handle the case where 'other' is not an instance of MyClass
            raise ValueError(f"Comparison not supported for {BaseState} and {type(other)}")
    
    def __hash__(self) -> int:
        return hash(self.state_identifier)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, BaseState):
            return self.state_identifier == other.state_identifier
        else:
            # Handle the case where 'other' is not an instance of MyClass
            raise ValueError(f"Comparison not supported for {BaseState} and {type(other)}")
    
    def __repr__(self) -> str:
        try:
            return str(self.state_identifier)
        except TypeError:
            raise TypeError(f'{type(self.state_identifier)} cannot be converted to a string')

    def __eq__(self, other: 'BaseState') -> bool:
        if isinstance(other, BaseState):
            return other.__hash__()==self.__hash__()
        else:
            return False

    @abstractmethod 
    def explore(self) -> list[Type['BaseState']]:
        raise NotImplementedError("This method should be implemented in a subclass.")

    @property
    def priority(self) -> float:
        return self._priority
    
    @priority.setter
    def priority(self, priority) -> None:
        try:
            self._priority = float(priority)
        except:
            raise(f'Expected {float} or {float}-castable but got {type(priority)}')
    
    @property
    def state_identifier(self) -> Hashable:
        return self._state_identifier

    @state_identifier.setter
    def state_identifier(self, state_identifier) -> None:
        try:
            hash(state_identifier)
        except TypeError:
            raise(f'{type(state_identifier)} is not hashable')
        self._state_identifier = state_identifier
    

class BaseSearch(ABC):

    def __init__(self) -> None:
        self.solution : Optional[Type[BaseState]] = None
        self.num_visited_states : int = 0
        self._priority : int = 0
        self.num_steps: int = 0
    
    def _default_f_priority(self, s: Optional[type[BaseState]] = None) -> int:
        self._priority += 1
        return self._priority

    @abstractmethod
    def objective_search(self, initial_state: Type[BaseState], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[['BaseState'], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
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
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def best_search(self, initial_state: Type[BaseState], f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[['BaseState'], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
        """
        This function finds the best solution under certain constraints

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_evaluate: function that compares two possible solutions, must return True if the first argument is better than the second argument
        :type f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float]
        :param f_goal: function that checks whether a state represents a possible solution
        :type f_goal: Callable[[Type[BaseState]], bool]
        :param f_stop: function that stops the search if a state respects some criteria
        :type f_stop: Optional[Callable[[Type[BaseState]], bool]]
        :param f_priority: function that decides the priority of the states in the frontier, determines the type of search
        :type f_priority: Optional[Callable[[Type[BaseState]], float]]
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


class TreeSearch(BaseSearch):

    def __init__(self) -> None:
        super().__init__()

    def objective_search(self, initial_state: Type[BaseState], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[['BaseState'], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
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
        
        current_state.priority = f_priority(current_state)
        frontier.put(current_state)

        while not found_solution and not frontier.empty():

            self.num_visited_states += 1
            self.num_steps += 1
            current_state = frontier.get()

            if f_goal(current_state):
                found_solution = True
                self.solution = current_state
            else:
                if f_stop is not None:
                    stop = f_stop(current_state)
                    if not stop:
                        for state in current_state.explore():
                            state.priority = f_priority(state)
                            frontier.put(state)
                else:
                    for state in current_state.explore():
                        state.priority = f_priority(state)
                        frontier.put(state)
    
    def best_search(self, initial_state: Type[BaseState], f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[['BaseState'], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
        """
        This function finds the best solution under certain constraints

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_evaluate: function that compares two possible solutions, must return True if the second argument is better than the first argument
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

        current_state.priority = f_priority(current_state)
        frontier.put(current_state)

        while not frontier.empty() and not stop:

            self.num_visited_states += 1
            self.num_steps += 1
            current_state = frontier.get()

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
                        state.priority = f_priority(state)
                        frontier.put(state)
            else:
                for state in current_state.explore():
                    state.priority = f_priority(state)
                    frontier.put(state)


class GraphSearch(BaseSearch):

    def __init__(self) -> None:
        super().__init__()
        self.visited_states : list[Type[BaseState]] = list()

    def objective_search(self, initial_state: Type[BaseState], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[['BaseState'], float]] = None,f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
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

        current_state.priority = f_priority(current_state)
        frontier.put(current_state)

        while not found_solution and not frontier.empty():

            self.num_steps += 1
            self.num_visited_states += 1

            current_state = frontier.get()
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
                            state.priority = f_priority(state)
                            frontier.put(state)
                else:
                    for state in current_state.explore():
                        if state not in self.visited_states:
                            self.visited_states.append(state)
                            state.priority = f_priority(state)
                            frontier.put(state)
    
    def best_search(self, initial_state: Type[BaseState], f_evaluate: Callable[[Type[BaseState], Type[BaseState]], float], f_goal: Callable[[Type[BaseState]], bool], f_priority: Optional[Callable[['BaseState'], float]] = None, f_stop: Optional[Callable[[Type[BaseState]], bool]] = None) -> None:
        """
        This function finds the best solution under certain constraints

        :param initial_state: state where the search starts
        :type initial_state: Type[BaseState]
        :param f_evaluate: function that compares two possible solutions, must return True if the second argument is better than the first argument
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

        current_state.priority = f_priority(current_state)
        frontier.put(current_state)
        self.visited_states.append(current_state)

        while not frontier.empty() and not stop:

            self.num_visited_states += 1
            self.num_steps += 1

            current_state = frontier.get()

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
                        state.priority = f_priority(state)
                        frontier.put(state)
            else:
                for state in current_state.explore():
                    if state not in self.visited_states:
                        self.visited_states.append(state)
                        state.priority = f_priority(state)
                        frontier.put(state)
