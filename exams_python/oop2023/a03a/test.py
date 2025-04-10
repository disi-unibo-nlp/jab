# Utility Files
from typing import TypeVar, Optional, Callable, List, Generic

X = TypeVar('X')
Y = TypeVar('Y')

class Windowing(Generic[X, Y]):
    """
    Python equivalent of the Windowing interface.
    """
    def process(self, x: X) -> Optional[Y]:
        """
        Processes an input element and returns an optional output.
        """
        raise NotImplementedError


class WindowingFactory:
    """
    Python equivalent of the WindowingFactory interface.
    """
    def trivial(self) -> Windowing[X, X]:
        """
        Returns a trivial windowing that outputs the input as is.
        """
        raise NotImplementedError

    def pairing(self) -> Windowing[X, tuple[X, X]]:
        """
        Returns a windowing that pairs consecutive inputs.
        """
        raise NotImplementedError

    def sum_last_four(self) -> Windowing[int, int]:
        """
        Returns a windowing that sums the last four integer inputs.
        """
        raise NotImplementedError

    def last_n(self, n: int) -> Windowing[X, List[X]]:
        """
        Returns a windowing that outputs the last N inputs as a list.
        """
        raise NotImplementedError

    def last_whose_sum_is_at_least(self, n: int) -> Windowing[int, List[int]]:
        """
        Returns a windowing that outputs the last inputs as a list whose sum is at least N.
        """
        raise NotImplementedError


# Solution
from collections import deque

class WindowingFactoryImpl(WindowingFactory):
    """
    Python equivalent of the WindowingFactoryImpl class.
    """

    def _generic_windowing(self, ready: Callable[[List[X]], bool], mapper: Callable[[List[X]], Y]) -> Windowing[X, Y]:
        """
        Generic windowing implementation.
        """
        class GenericWindowing(Windowing[X, Y]):
            def __init__(self):
                self._cache: deque[X] = deque()

            def process(self, x: X) -> Optional[Y]:
                self._cache.append(x)
                if not ready(list(self._cache)): # deque to list for slicing and compatibility with original logic
                    return None
                while self._cache and ready(list(self._cache)[1:]): # deque to list for slicing and compatibility with original logic
                    self._cache.popleft()
                return mapper(list(self._cache)) # deque to list for mapper

        return GenericWindowing()

    def trivial(self) -> Windowing[X, X]:
        """
        Implementation of trivial windowing.
        """
        return self._generic_windowing(lambda l: len(l) >= 1, lambda l: l[0])

    def pairing(self) -> Windowing[X, tuple[X, X]]:
        """
        Implementation of pairing windowing.
        """
        return self._generic_windowing(lambda l: len(l) >= 2, lambda l: (l[0], l[1]))

    def sum_last_four(self) -> Windowing[int, int]:
        """
        Implementation of sum_last_four windowing.
        """
        return self._generic_windowing(lambda l: len(l) >= 4, lambda l: self._list_sum(l))

    def last_n(self, n: int) -> Windowing[X, List[X]]:
        """
        Implementation of last_n windowing.
        """
        return self._generic_windowing(lambda l: len(l) >= n, lambda l: l)

    def last_whose_sum_is_at_least(self, n: int) -> Windowing[int, List[int]]:
        """
        Implementation of last_whose_sum_is_at_least windowing.
        """
        return self._generic_windowing(lambda l: self._list_sum(l) >= n, lambda l: l)

    @staticmethod
    def _list_sum(l: List[int]) -> int:
        """
        Calculates the sum of a list of integers.
        """
        return sum(l)


# Test File
import unittest

class Test(unittest.TestCase):
    """
    Implement the WindowingFactory interface as indicated in the init_factory method below.
    Create a factory for a Windowing concept, which is a particular type of transformation
    from sequences to sequences.

    The following are considered optional for the possibility of correcting
    the exercise, but still contribute to the achievement of the total
    score:

    - implementation of all methods of the factory (i.e., in the
      obligatory part it is sufficient to implement all of them except one at will)
    - good design of the solution, using design solutions that lead to
      succinct code that avoids repetitions

    Remove the comment from the init_factory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        """
        Initializes the factory before each test.
        """
        self.factory: WindowingFactory = WindowingFactoryImpl()

    def test_trivial(self):
        """
        Tests the trivial windowing.
        """
        windowing: Windowing[str, str] = self.factory.trivial()
        # the input corresponds to the output
        self.assertEqual(windowing.process("a"), "a")
        self.assertEqual(windowing.process("b"), "b")
        self.assertEqual(windowing.process("a"), "a")

    def test_pairing(self):
        """
        Tests the pairing windowing.
        """
        windowing: Windowing[int, tuple[int, int]] = self.factory.pairing()
        # the last two inputs provided to process form a pair, the first time Optional.empty
        self.assertIsNone(windowing.process(1))
        self.assertEqual(windowing.process(3), (1, 3))
        self.assertEqual(windowing.process(2), (3, 2))
        self.assertEqual(windowing.process(1), (2, 1))

    def test_sum_four(self):
        """
        Tests the sum_last_four windowing.
        """
        windowing: Windowing[int, int] = self.factory.sum_last_four()
        # the last four inputs provided to process produce their sum, the first 3 times Optional.empty
        self.assertIsNone(windowing.process(1))
        self.assertIsNone(windowing.process(10))
        self.assertIsNone(windowing.process(100))
        self.assertEqual(windowing.process(1000), 1111) #1+10+100+1000
        self.assertEqual(windowing.process(2), 1112) # 10+100+1000+2
        self.assertEqual(windowing.process(20), 1122) # 100+1000+2+20

    def test_last_n(self):
        """
        Tests the last_n windowing.
        """
        windowing: Windowing[int, List[int]] = self.factory.last_n(4)
        # the last N inputs provided to process produce a list, the first N-1 times Optional.empty
        self.assertIsNone(windowing.process(1))
        self.assertIsNone(windowing.process(10))
        self.assertIsNone(windowing.process(100))
        self.assertEqual(windowing.process(1000), [1, 10, 100, 1000])
        self.assertEqual(windowing.process(2), [10, 100, 1000, 2])
        self.assertEqual(windowing.process(20), [100, 1000, 2, 20])

    def test_sum_at_least(self):
        """
        Tests the last_whose_sum_is_at_least windowing.
        """
        windowing: Windowing[int, List[int]] = self.factory.last_whose_sum_is_at_least(10)
        # the list of the last elements whose sum is at least N (N=10 in this case) produce a list
        self.assertIsNone(windowing.process(5)) # not yet reached 10
        self.assertIsNone(windowing.process(3)) # not yet reached 10
        self.assertIsNone(windowing.process(1)) # not yet reached 10
        self.assertEqual(windowing.process(1), [5, 3, 1, 1]) # 5+3+1+1 >= 10
        self.assertEqual(windowing.process(2), [5, 3, 1, 1, 2]) # 5+3+1+1+2 >= 10, while 3+1+1+2 < 10
        self.assertEqual(windowing.process(4), [3, 1, 1, 2, 4]) # 3+1+1+2+4 >= 10, while 1+1+2+4 < 10
        self.assertEqual(windowing.process(8), [4, 8]) # etc...
        self.assertEqual(windowing.process(20), [20])


if __name__ == '__main__':
    unittest.main()