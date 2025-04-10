# Utility Files
import abc
from typing import TypeVar, Generic, Callable, List, Optional

X = TypeVar('X')
Y = TypeVar('Y')


class ListExtractorFactory(abc.ABC):
    """
    ListExtractorFactory interface.
    """

    @abc.abstractmethod
    def head(self) -> 'ListExtractor[X, Optional[X]]':
        """
        Creates a ListExtractor that extracts the head of a list.
        """
        pass

    @abc.abstractmethod
    def collect_until(self, mapper: Callable[[X], Y], stop_condition: Callable[[X], bool]) -> 'ListExtractor[X, List[Y]]':
        """
        Creates a ListExtractor that collects elements until a stop condition is met.
        """
        pass

    @abc.abstractmethod
    def scan_from(self, start_condition: Callable[[X], bool]) -> 'ListExtractor[X, List[List[X]]]':
        """
        Creates a ListExtractor that scans from a start condition.
        """
        pass

    @abc.abstractmethod
    def count_consecutive(self, x0: X) -> 'ListExtractor[X, int]':
        """
        Creates a ListExtractor that counts consecutive occurrences of an element.
        """
        pass


class ListExtractor(Generic[X, Y], abc.ABC):
    """
    ListExtractor interface.
    """

    @abc.abstractmethod
    def extract(self, list_: List[X]) -> Y:
        """
        Extracts data from a list.
        """
        pass


# Solution
from typing import List, Optional, Callable

class ListExtractorFactoryImpl(ListExtractorFactory):

    class AbstractListExtractor(ListExtractor[X, Y], abc.ABC):
        """
        Abstract base class for ListExtractors.
        """
        @abc.abstractmethod
        def initial_result(self) -> Y:
            pass

        @abc.abstractmethod
        def compute(self, output: Y, x: X) -> Y:
            pass

        @abc.abstractmethod
        def stop(self, output: Y, x: X) -> bool:
            pass

        @abc.abstractmethod
        def start(self, x: X) -> bool:
            pass

        def extract(self, list_: List[X]) -> Y:
            extracting = False
            output = self.initial_result()
            for x in list_:
                if not extracting and self.start(x):
                    extracting = True
                if extracting and self.stop(output, x):
                    break
                if extracting:
                    output = self.compute(output, x)
            return output

    def head(self) -> 'ListExtractor[X, Optional[X]]':
        """
        Implementation for head ListExtractor.
        """
        class HeadListExtractor(self.AbstractListExtractor[X, Optional[X]]):
            def initial_result(self) -> Optional[X]:
                return None

            def compute(self, output: Optional[X], x: X) -> Optional[X]:
                return x

            def stop(self, output: Optional[X], x: X) -> bool:
                return output is not None

            def start(self, x: X) -> bool:
                return True
        return HeadListExtractor()

    def collect_until(self, mapper: Callable[[X], Y], stop_condition: Callable[[X], bool]) -> 'ListExtractor[X, List[Y]]':
        """
        Implementation for collect_until ListExtractor.
        """
        class CollectUntilListExtractor(self.AbstractListExtractor[X, List[Y]]):
            def initial_result(self) -> List[Y]:
                return []

            def compute(self, output: List[Y], x: X) -> List[Y]:
                output.append(mapper(x))
                return output

            def stop(self, output: List[Y], x: X) -> bool:
                return stop_condition(x)

            def start(self, x: X) -> bool:
                return True
        return CollectUntilListExtractor()

    def scan_from(self, start_condition: Callable[[X], bool]) -> 'ListExtractor[X, List[List[X]]]':
        """
        Implementation for scan_from ListExtractor.
        """
        class ScanFromListExtractor(self.AbstractListExtractor[X, List[List[X]]]):
            def initial_result(self) -> List[List[X]]:
                return []

            def compute(self, output: List[List[X]], x: X) -> List[List[X]]:
                new_list = output[-1][:] if output else [] # Create a copy of the last list or start a new one
                new_list.append(x)
                output.append(new_list)
                return output

            def stop(self, output: List[List[X]], x: X) -> bool:
                return False

            def start(self, x: X) -> bool:
                return start_condition(x)
        return ScanFromListExtractor()

    def count_consecutive(self, x0: X) -> 'ListExtractor[X, int]':
        """
        Implementation for count_consecutive ListExtractor.
        """
        class CountConsecutiveListExtractor(self.AbstractListExtractor[X, int]):
            def initial_result(self) -> int:
                return 0

            def compute(self, output: int, x: X) -> int:
                return output + 1

            def stop(self, output: int, x: X) -> bool:
                return x != x0

            def start(self, x: X) -> bool:
                return x == x0

            def extract(self, list_: List[X]) -> int:
                count = 0
                if not list_ or list_[0] != x0:
                    return 0
                for x in list_:
                    if x == x0:
                        count += 1
                    else:
                        break
                return count
        return CountConsecutiveListExtractor()


# Test File
import unittest
from typing import List, Optional

class TestListExtractorFactory(unittest.TestCase):
    """
    Implement the ListExtractorFactory interface as indicated in the init_factory
    method below. Create a factory for a ListExtractor concept, captured by the
    interface of the same name: essentially it contains a pure function that,
    given a list, extracts a certain subsequence of it and uses it to produce,
    element by element, a result.
    In the complete exercise, reuse via inheritance must be applied.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the total score:

    - implementation of all factory methods (i.e., in the
    mandatory part it is sufficient to implement all but one at will)
    - the good design of the solution, using reuse via inheritance for all the various versions
    of the ListExtractor (i.e., in the mandatory part
    it is fine if reuse via inheritance is used for at least two ListExtractors)

    Remove the comment from the initFactory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.factory: ListExtractorFactory = ListExtractorFactoryImpl()

    def test_head(self):
        # an extractor that produces the first element of the list, if available
        le = self.factory.head()
        self.assertEqual(le.extract([10,20,30,40,50]), 10) # Optional.of(10) -> 10 because head returns Optional[X] and we are testing the value. If empty list, it returns None.
        self.assertEqual(le.extract([10]), 10) # Optional.of(10) -> 10
        self.assertEqual(le.extract([]), None) # Optional.empty() -> None

    def test_collect_until(self):
        le = self.factory.collect_until(mapper=lambda x: x + 1, stop_condition=lambda x: x >= 30)
        # collects the elements of the list until they reach 30, adding one to each
        self.assertEqual(le.extract([10,20,30,40,50]), [11, 21]) # List.of(11, 21) -> [11, 21]
        self.assertEqual(le.extract([10,50,20,40,50]), [11]) # List.of(11) -> [11]
        self.assertEqual(le.extract([30,50,20,40,50]), []) # List.of() -> []
        self.assertEqual(le.extract([]), []) # List.of() -> []

    def test_scan_from(self):
        le = self.factory.scan_from(start_condition=lambda x: x >= 30)
        # collects the elements from when there is one >= 30 to the end, producing lists of incremental lists
        self.assertEqual(le.extract([10,20,30,20,50]), [[30], [30, 20], [30, 20, 50]]) # List.of(List.of(30), List.of(30,20), List.of(30,20,50)) -> [[30], [30, 20], [30, 20, 50]]
        self.assertEqual(le.extract([30]), [[30]]) # List.of(List.of(30)) -> [[30]]
        self.assertEqual(le.extract([10,20,25]), []) # List.of() -> []

    def test_count(self):
        le = self.factory.count_consecutive("a")
        # counts how many consecutive "a"s there are starting from the first occurrence
        self.assertEqual(le.extract(["b", "a", "a","a","c","d","a"]), 0) # 0 -> 0
        self.assertEqual(le.extract(["a", "b", "a", "a","a","c","d","a"]), 1) # 1 -> 1
        self.assertEqual(le.extract(["b", "c", "d","a"]), 0) # 0 -> 0
        self.assertEqual(le.extract(["b", "c", "d"]), 0) # 0 -> 0
        self.assertEqual(le.extract([]), 0) # 0 -> 0


if __name__ == '__main__':
    unittest.main()