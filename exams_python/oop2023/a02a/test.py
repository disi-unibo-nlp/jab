# Utility Files
from typing import TypeVar, List, Callable, Generic, Iterable

T = TypeVar('T')


class ListBuilderFactory:
    """
    Factory interface for creating ListBuilder instances.
    """
    def empty(self) -> 'ListBuilder[T]':
        """
        Creates an empty ListBuilder.
        """
        raise NotImplementedError

    def from_element(self, element: T) -> 'ListBuilder[T]':
        """
        Creates a ListBuilder containing a single element.
        """
        raise NotImplementedError

    def from_list(self, list_val: List[T]) -> 'ListBuilder[T]':
        """
        Creates a ListBuilder from an existing list.
        """
        raise NotImplementedError

    def join(self, start: T, stop: T, builder_list: List['ListBuilder[T]']) -> 'ListBuilder[T]':
        """
        Joins a list of ListBuilders with a start and stop element.
        """
        raise NotImplementedError


class ListBuilder(Generic[T]):
    """
    Interface for building lists in a structured way.
    """
    def add(self, list_val: List[T]) -> 'ListBuilder[T]':
        """
        Adds a list to the current builder.
        """
        raise NotImplementedError

    def concat(self, lb: 'ListBuilder[T]') -> 'ListBuilder[T]':
        """
        Concatenates another ListBuilder to the current builder.
        """
        raise NotImplementedError

    def replace_all(self, t: T, lb: 'ListBuilder[T]') -> 'ListBuilder[T]':
        """
        Replaces all occurrences of an element with another ListBuilder.
        """
        raise NotImplementedError

    def reverse(self) -> 'ListBuilder[T]':
        """
        Reverses the elements in the builder.
        """
        raise NotImplementedError

    def build(self) -> List[T]:
        """
        Builds and returns the final list.
        """
        raise NotImplementedError


# Solution
from typing import TypeVar, List, Callable, Generic, Iterable

T = TypeVar('T')
R = TypeVar('R')


class ListBuilderFactoryImpl(ListBuilderFactory):

    def _flat_map(self, builder: 'ListBuilderImpl[T]', fun: Callable[[T, ], 'ListBuilderImpl[R]']) -> 'ListBuilderImpl[R]':
        """
        Utility function, essentially a flatMap for builders.
        """
        temp_list: List[R] = []
        for item in builder.build():
            temp_list.extend(fun(item).build())
        return ListBuilderFactoryImpl.ListBuilderImpl(temp_list)

    class ListBuilderImpl(ListBuilder[T]):
        """
        A builder that is an immutable wrapper of a list.
        """
        def __init__(self, initial_list: Iterable[T] = None):
            if initial_list is None:
                self._list: List[T] = []
            elif isinstance(initial_list, list):
                self._list: List[T] = list(initial_list) # defensive copy
            else:
                self._list: List[T] = list(initial_list) # consume iterator

        def add(self, list_val: List[T]) -> 'ListBuilderImpl[T]':
            """
            Adds a list to the current builder.
            """
            return ListBuilderFactoryImpl.ListBuilderImpl(self._list + list_val)

        def concat(self, lb: 'ListBuilderImpl[T]') -> 'ListBuilderImpl[T]':
            """
            Concatenates another ListBuilder to the current builder.
            """
            return self.add(lb.build())

        def replace_all(self, t: T, lb: 'ListBuilderImpl[T]') -> 'ListBuilderImpl[T]':
            """
            Replaces all occurrences of an element with another ListBuilder.
            """
            def replace_func(item: T) -> ListBuilderFactoryImpl.ListBuilderImpl[T]:
                return lb if item == t else ListBuilderFactoryImpl().from_element(item)
            return ListBuilderFactoryImpl()._flat_map(self, replace_func)

        def reverse(self) -> 'ListBuilderImpl[T]':
            """
            Reverses the elements in the builder.
            """
            reversed_list = self._list[::-1]
            return ListBuilderFactoryImpl.ListBuilderImpl(reversed_list)

        def build(self) -> List[T]:
            """
            Builds and returns the final list.
            """
            return list(self._list) # return a copy to maintain immutability


    def empty(self) -> 'ListBuilderImpl[T]':
        """
        Creates an empty ListBuilder.
        """
        return ListBuilderFactoryImpl.ListBuilderImpl()

    def from_element(self, element: T) -> 'ListBuilderImpl[T]':
        """
        Creates a ListBuilder containing a single element.
        """
        return ListBuilderFactoryImpl.ListBuilderImpl([element])

    def from_list(self, list_val: List[T]) -> 'ListBuilderImpl[T]':
        """
        Creates a ListBuilder from an existing list.
        """
        return ListBuilderFactoryImpl.ListBuilderImpl(list_val)

    def join(self, start: T, stop: T, builder_list: List['ListBuilderImpl[T]']) -> 'ListBuilderImpl[T]':
        """
        Joins a list of ListBuilders with a start and stop element.
        """
        joined_list: List[T] = [start]
        for lb in builder_list:
            joined_list.extend(lb.build())
        joined_list.append(stop)
        return ListBuilderFactoryImpl.ListBuilderImpl(joined_list)


# Test File
import unittest
from typing import List

class TestListBuilder(unittest.TestCase):
    """
    Implement the ListBuilderFactory interface as indicated in the init_factory method
    below. Create a factory for a ListBuilder concept,
    i.e., an immutable object that can be used to facilitate the creation of lists
    with an articulated structure (intent of the Builder pattern).

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the
    score:

    - implementation of all methods of the factory (i.e., in the
    mandatory part it is sufficient to implement all but one at will --
    the first 3 are still mandatory)
    - the good design of the solution, using design solutions
    that lead to
    succinct code that avoids repetition

    Remove the comment from the init_factory method.

    Score indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional method of the factory)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.factory = ListBuilderFactoryImpl()

    def test_empty(self):
        # empty() represents the builder of an empty list
        empty = self.factory.empty()
        self.assertEqual(list([]), empty.build())
        # if I add 10 and 20 it becomes the builder of a list (10, 20)
        self.assertEqual(list([10, 20]),
                         empty.add(list([10, 20]))
                         .build())
        # I can do two consecutive adds, concatenating the calls
        self.assertEqual(list([10, 20, 30]),
                         empty.add(list([10, 20]))
                         .add(list([30]))
                         .build())
        # with concat I get a builder that represents the concatenation of the lists
        self.assertEqual(list([10, 20, 30]),
                         empty.add(list([10, 20]))
                         .concat(self.factory.empty().add(list([30])))
                         .build())
        # another example with concat
        self.assertEqual(list([10, 20, 30]),
                         empty.add(list([10, 20]))
                         .concat(self.factory.empty().add(list([30])))
                         .build())

    def test_from_element(self):
        # from_element() represents the builder of a list with one element
        one = self.factory.from_element(1)
        # add and concat work as expected
        self.assertEqual(list([1]), one.build())
        self.assertEqual(list([1, 2, 3, 4]),
                         one.add(list([2, 3, 4])).build())
        self.assertEqual(list([1, 2, 1]),
                         one.concat(self.factory.from_element(2))
                         .concat(one)
                         .build())

    def test_basic_from_list(self):
        # fromList() represents the builder of a list with n elements
        l = self.factory.from_list(list([1, 2, 3]))
        self.assertEqual(list([1, 2, 3]), l.build())
        # concat work as expected
        self.assertEqual(list([1, 2, 3, 1, 2, 3]),
                         l.concat(l).build())
        # replaceAll here replaces the "1"s with lists [-1, -2]
        self.assertEqual(list([-1, -2, 2, 3, -1, -2, 2, 3]),
                         l.concat(l)
                         .replace_all(1, self.factory.from_list(list([-1, -2])))
                         .build())
        # if there is no match, replaceAll does nothing
        self.assertEqual(list([1, 2, 3, 1, 2, 3]),
                         l.concat(l)
                         .replace_all(10, self.factory.from_list(list([-1, -2])))
                         .build())

    def test_reverse_from_list(self):
        l = self.factory.from_list(list([1, 2, 3]))
        self.assertEqual(list([1, 2, 3]), l.build())
        # reverse makes you get a builder that represents the inverted list
        self.assertEqual(list([3, 2, 1]), self.factory.from_list(list([1, 2, 3])).reverse().build())
        self.assertEqual(list([1, 2, 3]), self.factory.from_list(list([1, 2, 3])).reverse().reverse().build())
        self.assertEqual(list([1, 2, 3, 3, 2, 1]),
                         self.factory.from_list(list([1, 2, 3])).reverse().reverse().concat(self.factory.from_list(list([1, 2, 3])).reverse()).build())

    def test_join(self):
        # join can be used to concatenate multiple builders, with an initial and a final element
        self.assertEqual(list(["(", "1", "2", "3", "4", ")"]),
                         self.factory.join("(", ")",
                                          list([self.factory.from_element("1"),
                                               self.factory.from_element("2"),
                                               self.factory.from_list(list(["3", "4"]))])).build())

if __name__ == '__main__':
    unittest.main()