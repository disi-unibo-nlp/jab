# Utility Files
from typing import List, Callable, TypeVar, Iterable, Iterator

T = TypeVar('T')

Replacer = Callable[[List[T], T], List[List[T]]]

class ReplacersFactory:
    def no_replacement(self) -> Replacer[T]:
        raise NotImplementedError

    def duplicate_first(self) -> Replacer[T]:
        raise NotImplementedError

    def translate_last_with(self, target: List[T]) -> Replacer[T]:
        raise NotImplementedError

    def remove_each(self) -> Replacer[T]:
        raise NotImplementedError

    def replace_each_from_sequence(self, sequence: List[T]) -> Replacer[T]:
        raise NotImplementedError


# Solution
from typing import List, Callable, TypeVar, Iterable, Iterator

T = TypeVar('T')

class ReplacersFactoryImpl(ReplacersFactory):

    def _replace_at_position(self, index: int, source: List[T], destination: List[T]) -> List[T]:
        l: List[T] = list(source) # LinkedList in Java is roughly list in Python for these operations
        l.pop(index)
        it = reversed(destination) # listIterator(destination.size()) in Java, reversed iteration in Python
        for item in it:
            l.insert(index, item) # add(index, it.previous()) in Java, insert in Python
        return l

    def _general_replacer(self, left: T, right_supplier: Callable[[], List[T]], where: Callable[[List[int]], List[int]]) -> Replacer[T]:
        def replacer_func(input_list: List[T], t: T) -> List[List[T]]:
            where_to_replace_indices: List[int] = where(
                [i for i, element in enumerate(input_list) if element == t] # Stream and filter in Java, list comprehension in Python
            )
            return [self._replace_at_position(index, input_list, right_supplier()) for index in where_to_replace_indices] # stream().map().toList() in Java, list comprehension in Python
        return replacer_func

    def no_replacement(self) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return []
        def where_func(indices: List[int]) -> List[int]:
            return []
        return self._general_replacer(None, right_supplier, where_func) # t is not used in general_replacer's lambda

    def duplicate_first(self) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return [None, None] # Placeholders, t will be used in the actual call
        def where_func(indices: List[int]) -> List[int]:
            return indices[:1] # l.isEmpty() ? l : l.subList(0, 1) in Java, slice in Python
        def replacer_func(input_list: List[T], t: T) -> List[List[T]]:
            right_supplier_t = lambda: [t, t] # Create supplier with t
            return self._general_replacer(t, right_supplier_t, where_func)(input_list, t)
        return replacer_func

    def translate_last_with(self, target: List[T]) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return target
        def where_func(indices: List[int]) -> List[int]:
            return indices[-1:] if indices else [] # l.isEmpty() ? l : l.subList(l.size()-1, l.size()) in Java, slice in Python, handle empty case
        return self._general_replacer(None, right_supplier, where_func) # t is not used in general_replacer's lambda

    def remove_each(self) -> Replacer[T]:
        def right_supplier() -> List[T]:
            return []
        def where_func(indices: List[int]) -> List[int]:
            return indices # l in Java, return all indices in Python
        return self._general_replacer(None, right_supplier, where_func) # t is not used in general_replacer's lambda

    def replace_each_from_sequence(self, sequence: List[T]) -> Replacer[T]:
        sequence_iterator = iter(sequence) # var it = sequence.iterator(); in Java, iter in Python
        def right_supplier() -> List[T]:
            try:
                return [next(sequence_iterator)] # it.next() in Java, next in Python
            except StopIteration:
                return [] # Handle case when sequence is exhausted
        def where_func(indices: List[int]) -> List[int]:
            return indices[:len(sequence)] # l.subList(0, Math.min(l.size(), sequence.size())) in Java, slice in Python
        def replacer_func(input_list: List[T], t: T) -> List[List[T]]:
            # Re-initialize iterator for each call to replace, to match Java behavior (important for test correctness)
            nonlocal sequence_iterator
            sequence_iterator = iter(sequence)
            return self._general_replacer(t, right_supplier, where_func)(input_list, t)
        return replacer_func


# Test File
import unittest
from typing import List

class TestReplacersFactoryImpl(unittest.TestCase):
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
        self.assertEqual([], replacer([10,20,30], 10))
        self.assertEqual([], replacer([10,20,30], 11))

    def test_duplicate_first(self):
        # a replacer that duplicates the first occurrence of a certain number, therefore provides 0 or 1 solution
        replacer = self.factory.duplicate_first()
        # 10 is present, replaces its first occurrence with 10,20,30
        self.assertEqual([[10, 10, 20, 30]], replacer([10,20,30], 10))
        self.assertEqual([[0, 10, 10, 20, 10]], replacer([0, 10,20,10], 10))
        # 11 is not present: no solution
        self.assertEqual([], replacer([0, 10,20,10], 11))

    def test_translate_last_with(self):
        # a replacer that replaces the last occurrence of a certain number with a certain sublist (-1, -1)
        replacer = self.factory.translate_last_with([-1,-1])
        # 10 is present: its last (and only) occurrence is replaced with -1,-1
        self.assertEqual([[-1, -1, 20, 30]], replacer([10,20,30], 10))
        # 10 is present multiple times: its last occurrence is replaced with -1,-1
        self.assertEqual([[0, 10, 20, -1, -1]], replacer([0, 10,20,10], 10))
        # 11 is not present, no solution
        self.assertEqual([], replacer([0, 10,20,10], 11))

    def test_remove_each(self):
        # a replacer that removes each occurrence of a certain number one by one, giving n solutions in general
        replacer = self.factory.remove_each()
        # there is only one 10, there is one solution where it is removed
        self.assertEqual([[20, 30]], replacer([10,20,30], 10))
        # there are two 10s, there are two solutions where they are removed one at a time
        self.assertEqual([
            [0,20,10],
            [0, 10,20]
        ], replacer([0, 10,20,10], 10))
        # 11 is not present, no solution
        self.assertEqual([], replacer([0, 10,20,10], 11))

    def test_replace_each_from_sequence(self):
        # the first 3 occurrences of the number are replaced by -100, -101 and -102, respectively
        # giving therefore at most 3 solutions (but less if there are fewer occurrences)
        replacer = self.factory.replace_each_from_sequence([-100, -101, -102])
        # 1 single occurrence, replaced with -100
        self.assertEqual([[0, -100, 20, 30]], replacer([0,10,20,30], 10))
        # 2 occurrences, replaced with -100 and -101 respectively
        self.assertEqual([
            [0, -100, 20,10],
            [0, 10,20, -101]
        ], replacer([0, 10,20,10], 10))
        # 3 occurrences...
        self.assertEqual([
            [-100, 10, 10, 10],
            [10, -101, 10, 10],
            [10, 10, -102, 10]
        ], replacer([10, 10, 10, 10], 10))
        # 0 occurrences...
        self.assertEqual([], replacer([0, 10,20,10], 11))

if __name__ == '__main__':
    unittest.main()