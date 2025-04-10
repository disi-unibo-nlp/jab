# Utility Files
from typing import TypeVar, Callable, List

T = TypeVar('T')
Replacer = Callable[[List[T], T], List[List[T]]]

class ReplacersFactory:
    def no_replacement(self) -> Replacer[T]:
        raise NotImplementedError()

    def duplicate_first(self) -> Replacer[T]:
        raise NotImplementedError()

    def translate_last_with(self, target: List[T]) -> Replacer[T]:
        raise NotImplementedError()

    def remove_each(self) -> Replacer[T]:
        raise NotImplementedError()

    def replace_each_from_sequence(self, sequence: List[T]) -> Replacer[T]:
        raise NotImplementedError()


# Solution
from typing import List, Callable

class ReplacersFactoryImpl(ReplacersFactory):

    def _replace_at_position(self, index: int, source: List[T], destination: List[T]) -> List[T]:
        l = list(source) # Create a copy to avoid modifying the original list
        l.pop(index)
        for item in reversed(destination): # Correct order of insertion
            l.insert(index, item)
        return l

    def _general_replacer(self, left: T, right_supplier: Callable[[], List[T]], where: Callable[[List[int]], List[int]]) -> Replacer[T]:
        def replacer_function(input_list: List[T], t: T) -> List[List[T]]:
            where_to_replace_indices = where(
                [i for i, val in enumerate(input_list) if val == t]
            )
            return [self._replace_at_position(index, input_list, right_supplier()) for index in where_to_replace_indices]
        return replacer_function

    def no_replacement(self) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return []
        def where_function(indices: List[int]) -> List[int]:
            return []
        return self._general_replacer(None, right_supplier, where_function) # None for left, as it's not used in this replacer

    def duplicate_first(self) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return [t, t]
        def where_function(indices: List[int]) -> List[int]:
            return indices[:1] if indices else []
        return self._general_replacer(None, right_supplier, where_function) # None for left, as it's not used directly in general_replacer

    def translate_last_with(self, target: List[T]) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return target
        def where_function(indices: List[int]) -> List[int]:
            return indices[-1:] if indices else []
        return self._general_replacer(None, right_supplier, where_function) # None for left, as it's not used directly in general_replacer

    def remove_each(self) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return []
        def where_function(indices: List[int]) -> List[int]:
            return indices
        return self._general_replacer(None, right_supplier, where_function) # None for left, as it's not used directly in general_replacer

    def replace_each_from_sequence(self, sequence: List[T]) -> Replacer[T]:
        sequence_iterator = iter(sequence)
        def right_supplier() -> List[T]:
            try:
                return [next(sequence_iterator)]
            except StopIteration:
                return [] # or handle exhaustion as needed, e.g., return empty list
        def where_function(indices: List[int]) -> List[int]:
            return indices[:len(sequence)]
        return self._general_replacer(None, right_supplier, where_function) # None for left, as it's not used directly in general_replacer


# Test File
import unittest
from typing import List

class TestReplacersFactory(unittest.TestCase):
    """
    Implement the ReplacerFactory interface as indicated in the initFactory method below.
    Create a factory for a Replacer concept, i.e., a functionality that, given a list and an element as input, modifies
    the list by replacing certain occurrences of the element with a new sub-list,
    depending on the implementation. It therefore provides in general n solutions,
    one for each replaced occurrence.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the
    score:

    - implementation of all factory methods (i.e., in the
    mandatory part it is sufficient to implement all of them except one of your choice)
    - the good design of the solution, using design solutions that lead to
    succinct code that avoids repetitions

    Remove the comment from the initFactory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (further factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.factory = ReplacersFactoryImpl()

    def test_none(self):
        # a replacer that does not replace anything, therefore does not provide solutions
        replacer = self.factory.no_replacement()
        self.assertEqual(replacer.replace([10,20,30], 10), [])
        self.assertEqual(replacer.replace([10,20,30], 11), [])

    def test_duplicate_first(self):
        # a replacer that duplicates the first occurrence of a certain number, therefore provides 0 or 1 solution
        replacer = self.factory.duplicate_first()
        # 10 is present, replaces its first occurrence with 10,20,30
        self.assertEqual(replacer.replace([10,20,30], 10), [[10, 10, 20, 30]])
        self.assertEqual(replacer.replace([0, 10,20,10], 10), [[0, 10, 10, 20, 10]])
        # 11 is not present: no solution
        self.assertEqual(replacer.replace([0, 10,20,10], 11), [])

    def test_translate_last_with(self):
        # a replacer that replaces the last occurrence of a certain number with a certain sublist (-1, -1)
        replacer = self.factory.translate_last_with([-1,-1])
        # 10 is present: its last (and only) occurrence is replaced with -1,-1
        self.assertEqual(replacer.replace([10,20,30], 10), [[-1, -1, 20, 30]])
        # 10 is present multiple times: its last occurrence is replaced with -1,-1
        self.assertEqual(replacer.replace([0, 10,20,10], 10), [[0, 10, 20, -1, -1]])
        # 11 is not present, no solution
        self.assertEqual(replacer.replace([0, 10,20,10], 11), [])

    def test_remove_each(self):
        # a replacer that removes each occurrence of a certain number one by one, giving n solutions in general
        replacer = self.factory.remove_each()
        # there is only one 10, there is one solution where it is removed
        self.assertEqual(replacer.replace([10,20,30], 10), [[20, 30]])
        # there are two 10s, there are two solutions where they are removed one at a time
        self.assertEqual(replacer.replace([0, 10,20,10], 10), [
            [0,20,10],
            [0, 10,20]
        ])
        # 11 is not present, no solution
        self.assertEqual(replacer.replace([0, 10,20,10], 11), [])

    def test_replace_each_from_sequence(self):
        # the first 3 occurrences of the number are replaced by -100, -101 and -102, respectively
        # giving therefore at most 3 solutions (but less if there are fewer occurrences)
        replacer = self.factory.replace_each_from_sequence([-100, -101, -102])
        # 1 single occurrence, replaced with -100
        self.assertEqual(replacer.replace([0,10,20,30], 10), [[0, -100, 20, 30]])
        # 2 occurrences, replaced with -100 and -101 respectively
        self.assertEqual(replacer.replace([0, 10,20,10], 10), [
            [0, -100, 20,10],
            [0, 10,20, -101]
        ])
        # 3 occurrences...
        self.assertEqual(replacer.replace([10, 10, 10, 10], 10), [
            [-100, 10, 10, 10],
            [10, -101, 10, 10],
            [10, 10, -102, 10]
        ])
        # 0 occurrences...
        self.assertEqual(replacer.replace([0, 10,20,10], 11), [])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)