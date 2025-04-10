# Utility Files
from typing import TypeVar, List, Iterable, Generator

X = TypeVar('X')

class InfiniteIterator:
    """
    Interface for an infinite iterator.
    """
    def next_element(self) -> X:
        """
        Returns the next element in the infinite sequence.
        """
        raise NotImplementedError()

    def next_list_of_elements(self, size: int) -> List[X]:
        """
        Returns a list of the next 'size' elements.
        """
        return [self.next_element() for _ in range(size)]

class InfiniteIteratorsHelpers:
    """
    Interface for utility functions for infinite iterators.
    """
    def of_single_value(self, x: X) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that always returns the same value.
        """
        raise NotImplementedError()

    def cyclic(self, l: List[X]) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that cycles through the elements of the given list.
        """
        raise NotImplementedError()

    def incrementing(self, start: int, increment: int) -> InfiniteIterator[int]:
        """
        Creates an infinite iterator that yields incrementing integers.
        """
        raise NotImplementedError()

    def alternating(self, i: InfiniteIterator[X], j: InfiniteIterator[X]) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that alternates between two input iterators.
        """
        raise NotImplementedError()

    def window(self, i: InfiniteIterator[X], n: int) -> InfiniteIterator[List[X]]:
        """
        Creates an infinite iterator that yields sliding windows of size 'n'.
        """
        raise NotImplementedError()


# Solution
from itertools import cycle, count, tee

class InfiniteIteratorsHelpersImpl(InfiniteIteratorsHelpers):

    def of_single_value(self, x: X) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that always returns the same value.
        """
        class SingleValueIterator(InfiniteIterator[X]):
            def next_element(self) -> X:
                return x
        return SingleValueIterator()

    def cyclic(self, l: List[X]) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that cycles through the elements of the given list.
        """
        if not l:
            raise ValueError("Cyclic list cannot be empty")
        iterator_cycle = cycle(l)

        class CyclicIterator(InfiniteIterator[X]):
            def next_element(self) -> X:
                return next(iterator_cycle)
        return CyclicIterator()

    def incrementing(self, start: int, increment: int) -> InfiniteIterator[int]:
        """
        Creates an infinite iterator that yields incrementing integers.
        """
        iterator_count = count(start, increment)
        class IncrementingIterator(InfiniteIterator[int]):
            def next_element(self) -> int:
                return next(iterator_count)
        return IncrementingIterator()

    def alternating(self, i: InfiniteIterator[X], j: InfiniteIterator[X]) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that alternates between two input iterators.
        """
        class AlternatingIterator(InfiniteIterator[X]):
            def __init__(self, iter1: InfiniteIterator[X], iter2: InfiniteIterator[X]):
                self.iter1 = iter1
                self.iter2 = iter2
                self.current_iter = self.iter1

            def next_element(self) -> X:
                element = self.current_iter.next_element()
                if self.current_iter == self.iter1:
                    self.current_iter = self.iter2
                else:
                    self.current_iter = self.iter1
                return element
        return AlternatingIterator(i, j)

    def window(self, i: InfiniteIterator[X], n: int) -> InfiniteIterator[List[X]]:
        """
        Creates an infinite iterator that yields sliding windows of size 'n'.
        """
        class WindowIterator(InfiniteIterator[List[X]]):
            def __init__(self, input_iter: InfiniteIterator[X], window_size: int):
                self.input_iter = input_iter
                self.window_size = window_size
                self.window_cache: List[X] = []

            def next_element(self) -> List[X]:
                while len(self.window_cache) < self.window_size:
                    self.window_cache.append(self.input_iter.next_element())
                window = list(self.window_cache) # Create a copy to avoid modification
                self.window_cache.pop(0) # Remove the oldest element for sliding window
                return window
        return WindowIterator(i, n)


class InfiniteIteratorsHelpersImpl2(InfiniteIteratorsHelpers):

    def of_single_value(self, x: X) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that always returns the same value.
        """
        class SingleValueIterator(InfiniteIterator[X]):
            def next_element(self) -> X:
                return x
        return SingleValueIterator()

    def cyclic(self, l: List[X]) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that cycles through the elements of the given list.
        """
        class CyclicIterator(InfiniteIterator[X]):
            def __init__(self, lst: List[X]):
                self.linked_list = list(lst) # Using list as LinkedList equivalent for simplicity

            def next_element(self) -> X:
                element = self.linked_list.pop(0)
                self.linked_list.append(element)
                return element
        return CyclicIterator(l)

    def incrementing(self, start: int, increment: int) -> InfiniteIterator[int]:
        """
        Creates an infinite iterator that yields incrementing integers.
        """
        class IncrementingIterator(InfiniteIterator[int]):
            def __init__(self, start_val: int, inc_val: int):
                self.state = start_val
                self.increment = inc_val

            def next_element(self) -> int:
                current_value = self.state
                self.state += self.increment
                return current_value
        return IncrementingIterator(start, increment)

    def alternating(self, i: InfiniteIterator[X], j: InfiniteIterator[X]) -> InfiniteIterator[X]:
        """
        Creates an infinite iterator that alternates between two input iterators.
        """
        class AlternatingIterator(InfiniteIterator[X]):
            def __init__(self, input1: InfiniteIterator[X], input2: InfiniteIterator[X]):
                self.i1 = input1
                self.i2 = input2
                self.current_iterator = self.i1

            def next_element(self) -> X:
                if self.current_iterator == self.i1:
                    self.current_iterator = self.i2
                    return self.i1.next_element()
                else:
                    self.current_iterator = self.i1
                    return self.i2.next_element()
        return AlternatingIterator(i, j)


    def window(self, i: InfiniteIterator[X], n: int) -> InfiniteIterator[List[X]]:
        """
        Creates an infinite iterator that yields sliding windows of size 'n'.
        """
        class WindowIterator(InfiniteIterator[List[X]]):
            def __init__(self, input_stream: InfiniteIterator[X], window_size: int):
                self.cache: List[X] = []
                self.source_iterator = input_stream
                self.window_dimension = window_size

            def next_element(self) -> List[X]:
                self.cache.append(self.source_iterator.next_element())
                while len(self.cache) < self.window_dimension:
                    self.cache.append(self.source_iterator.next_element())
                return list(self.cache[-self.window_dimension:]) # return a copy of the last n elements
        return WindowIterator(i, n)


# Test File
import unittest

class TestInfiniteIterators(unittest.TestCase):
    """
    Implement the InfiniteIteratorHelpers interface as indicated in the method
    initFactory below. Implement an interface of utility functions for an
    infinite iterator concept, captured by the InfiniteIterator interface:
    essentially it is an iterator that continues to provide values indefinitely.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the
    score:

    - implementation of all methods of the factory (i.e., in the
    mandatory part it is sufficient to implement all of them except one at will)
    - the good design of the solution, using design solutions that lead to
    succinct code that avoids repetitions

    Remove the comment from the initFactory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.iih = InfiniteIteratorsHelpersImpl() # or InfiniteIteratorsHelpersImpl2() to test the other implementation

    def test_value(self):
        # Test on sequences consisting of a single element
        self.assertEqual(self.iih.of_single_value("a").next_list_of_elements(5), ["a","a","a","a","a"])
        self.assertEqual(self.iih.of_single_value("a").next_list_of_elements(1), ["a"])
        self.assertEqual(self.iih.of_single_value("a").next_list_of_elements(10), ["a","a","a","a","a","a","a","a","a","a"])
        self.assertEqual(self.iih.of_single_value(1).next_list_of_elements(10), [1,1,1,1,1,1,1,1,1,1])

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
        i2 = self.iih.of_single_value(0)
        self.assertEqual(self.iih.alternating(i1, i2).next_list_of_elements(6), [1,0,2,0,3,0])

    def test_window(self):
        # Test on constructing "window" sequences
        self.assertEqual(self.iih.window(self.iih.incrementing(1, 1), 4).next_list_of_elements(3),
                         [[1,2,3,4],
                          [2,3,4,5],
                          [3,4,5,6]])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)