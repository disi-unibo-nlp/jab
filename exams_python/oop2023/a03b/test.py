# Utility Files
from typing import TypeVar, List, Iterable, Iterator
from abc import ABC, abstractmethod

X = TypeVar('X')

class InfiniteIterator(ABC, Iterator[X]):
    @abstractmethod
    def __next__(self) -> X:
        pass

    def next_list_of_elements(self, size: int) -> List[X]:
        return [next(self) for _ in range(size)]

class InfiniteIteratorsHelpers(ABC):
    @abstractmethod
    def of(self, x: X) -> InfiniteIterator[X]:
        pass

    @abstractmethod
    def cyclic(self, l: List[X]) -> InfiniteIterator[X]:
        pass

    @abstractmethod
    def incrementing(self, start: int, increment: int) -> InfiniteIterator[int]:
        pass

    @abstractmethod
    def alternating(self, i: InfiniteIterator[X], j: InfiniteIterator[X]) -> InfiniteIterator[X]:
        pass

    @abstractmethod
    def window(self, i: InfiniteIterator[X], n: int) -> InfiniteIterator[List[X]]:
        pass


# Solution
from itertools import cycle, chain, islice, tee

class InfiniteIteratorHelpersImpl(InfiniteIteratorsHelpers):

    def of(self, x: X) -> InfiniteIterator[X]:
        class ValueIterator(InfiniteIterator[X]):
            def __next__(self) -> X:
                return x
        return ValueIterator()

    def cyclic(self, l: List[X]) -> InfiniteIterator[X]:
        if not l:
            raise ValueError("Cannot create cyclic iterator from an empty list")
        class CyclicIterator(InfiniteIterator[X]):
            def __init__(self, input_list: List[X]):
                self._cycle = cycle(input_list)

            def __next__(self) -> X:
                return next(self._cycle)
        return CyclicIterator(l)

    def incrementing(self, start: int, increment: int) -> InfiniteIterator[int]:
        class IncrementingIterator(InfiniteIterator[int]):
            def __init__(self, start_val: int, increment_val: int):
                self.current = start_val
                self.increment = increment_val

            def __next__(self) -> int:
                value = self.current
                self.current += self.increment
                return value
        return IncrementingIterator(start, increment)

    def alternating(self, i: InfiniteIterator[X], j: InfiniteIterator[X]) -> InfiniteIterator[X]:
        class AlternatingIterator(InfiniteIterator[X]):
            def __init__(self, iterator1: InfiniteIterator[X], iterator2: InfiniteIterator[X]):
                self.current_iterator = iterator1
                self.next_iterator = iterator2

            def __next__(self) -> X:
                value = next(self.current_iterator)
                self.current_iterator, self.next_iterator = self.next_iterator, self.current_iterator
                return value
        return AlternatingIterator(i, j)

    def window(self, i: InfiniteIterator[X], n: int) -> InfiniteIterator[List[X]]:
        class WindowIterator(InfiniteIterator[List[X]]):
            def __init__(self, iterator: InfiniteIterator[X], window_size: int):
                self.iterator = iterator
                self.window_size = window_size
                self.window_cache: List[X] = []

            def __next__(self) -> List[X]:
                self.window_cache.append(next(self.iterator))
                if len(self.window_cache) > self.window_size:
                    self.window_cache.pop(0)
                return list(self.window_cache) if len(self.window_cache) == self.window_size else self.__next__() # Ensure window is full before returning

        return WindowIterator(i, n)


class InfiniteIteratorHelpersImpl2(InfiniteIteratorsHelpers):

    def of(self, x: X) -> InfiniteIterator[X]:
        class ValueIterator(InfiniteIterator[X]):
            def __next__(self) -> X:
                return x
        return ValueIterator()

    def cyclic(self, l: List[X]) -> InfiniteIterator[X]:
        class CyclicIterator(InfiniteIterator[X]):
            def __init__(self, input_list: List[X]):
                self.linked_list = list(input_list) # Using list as LinkedList
                if not self.linked_list:
                    raise ValueError("Cannot create cyclic iterator from an empty list")

            def __next__(self) -> X:
                e = self.linked_list.pop(0)
                self.linked_list.append(e)
                return e
        return CyclicIterator(l)

    def incrementing(self, start: int, increment: int) -> InfiniteIterator[int]:
        class IncrementingIterator(InfiniteIterator[int]):
            def __init__(self, start_val: int, increment_val: int):
                self.state = start_val
                self.increment = increment_val

            def __next__(self) -> int:
                itemp = self.state
                self.state += self.increment
                return itemp
        return IncrementingIterator(start, increment)

    def alternating(self, i: InfiniteIterator[X], j: InfiniteIterator[X]) -> InfiniteIterator[X]:
        class AlternatingIterator(InfiniteIterator[X]):
            def __init__(self, input1: InfiniteIterator[X], input2: InfiniteIterator[X]):
                self.i1 = input1
                self.i2 = input2

            def __next__(self) -> X:
                itemp = self.i1
                self.i1 = self.i2
                self.i2 = itemp
                return next(self.i2)
        return AlternatingIterator(i, j)

    def window(self, i: InfiniteIterator[X], n: int) -> InfiniteIterator[List[X]]:
        class WindowIterator(InfiniteIterator[List[X]]):
            def __init__(self, s: InfiniteIterator[X], window_size: int):
                self.cache: List[X] = []
                self.source_iterator = s
                self.window_size = window_size

            def __next__(self) -> List[X]:
                self.cache.append(next(self.source_iterator))
                while len(self.cache) < self.window_size:
                    self.cache.append(next(self.source_iterator))
                return list(self.cache[-self.window_size:]) # Simulate LinkedList subList behavior
        return WindowIterator(i, n)


# Test File
import unittest

class TestInfiniteIterators(unittest.TestCase):
    """
    Implement the InfiniteIteratorHelpers interface as indicated in the method
    init_factory below. Implement an interface of utility functions for an
    infinite iterator concept, captured by the InfiniteIterator interface:
    essentially it is an iterator that continues to provide values indefinitely.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the
    score:

    - implementation of all methods of the factory (i.e., in the
    mandatory part it is sufficient to implement all of them except one at will)
    - the good design of the solution, using design solutions that lead to
    succinct code that avoids repetitions

    Remove the comment from the init_factory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.iih: InfiniteIteratorsHelpers = InfiniteIteratorHelpersImpl() # or InfiniteIteratorHelpersImpl2()

    def test_value(self):
        # Test on sequences consisting of a single element
        self.assertEqual(self.iih.of("a").next_list_of_elements(5), ["a","a","a","a","a"])
        self.assertEqual(self.iih.of("a").next_list_of_elements(1), ["a"])
        self.assertEqual(self.iih.of("a").next_list_of_elements(10), ["a","a","a","a","a","a","a","a","a","a"])
        self.assertEqual(self.iih.of(1).next_list_of_elements(10), [1,1,1,1,1,1,1,1,1,1])

    def test_cyclic(self):
        # Test on cyclic sequences
        # sequence: a,b,a,b,a,b,a,b,a,b,....
        self.assertEqual(self.iih.cyclic(["a","b"]).next_list_of_elements(5), ["a","b","a","b","a"])
        # sequence: 1,2,3,1,2,3,1,2,3,1,2,3,1,2,....
        self.assertEqual(self.iih.cyclic([1,2,3]).next_list_of_elements(10), [1,2,3,1,2,3,1,2,3,1])

    def test_incrementing(self):
        # Test on constructing increment sequence
        self.assertEqual(self.iih.incrementing(1, 2).next_list_of_elements(5), [1,3,5,7,9])
        self.assertEqual(self.iih.incrementing(0, -3).next_list_of_elements(4), [0,-3,-6,-9])

    def test_alternating(self):
        # Test on constructing alternating sequences
        i1 = self.iih.incrementing(1, 1)
        i2 = self.iih.of(0)
        self.assertEqual(self.iih.alternating(i1, i2).next_list_of_elements(6), [1,0,2,0,3,0])

    def test_window(self):
        # Test on constructing "window" sequences
        self.assertEqual(self.iih.window(self.iih.incrementing(1, 1), 4).next_list_of_elements(3), [
                [1,2,3,4],
                [2,3,4,5],
                [3,4,5,6]])

if __name__ == '__main__':
    unittest.main()