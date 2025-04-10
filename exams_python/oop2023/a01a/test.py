# Utility Files
from typing import Set, Callable, Tuple

class TimetableFactory:
    """
    Factory for creating Timetable instances.
    """
    def empty(self) -> 'Timetable':
        """
        Creates an empty timetable.
        """
        raise NotImplementedError

    def single(self, activity: str, day: str) -> 'Timetable':
        """
        Creates a timetable with a single hour of activity on a given day.
        """
        raise NotImplementedError

    def join(self, table1: 'Timetable', table2: 'Timetable') -> 'Timetable':
        """
        Joins two timetables, summing up the hours for each activity and day.
        """
        raise NotImplementedError

    def cut(self, table: 'Timetable', bounds: Callable[[str, str], int]) -> 'Timetable':
        """
        Reduces the hours in a timetable based on the given bounds function.
        """
        raise NotImplementedError


class Timetable:
    """
    Represents a timetable, associating hours with activities and days.
    """
    def add_hour(self, activity: str, day: str) -> 'Timetable':
        """
        Adds an hour to the timetable for the given activity and day.
        """
        raise NotImplementedError

    def activities(self) -> Set[str]:
        """
        Returns the set of activities in the timetable.
        """
        raise NotImplementedError

    def days(self) -> Set[str]:
        """
        Returns the set of days in the timetable.
        """
        raise NotImplementedError

    def get_single_data(self, activity: str, day: str) -> int:
        """
        Returns the number of hours for a specific activity and day.
        """
        raise NotImplementedError

    def sums(self, activities: Set[str], days: Set[str]) -> int:
        """
        Returns the sum of hours for the given sets of activities and days.
        """
        raise NotImplementedError


# Solution
from typing import Set, Callable
from functools import reduce

class TimetableFactoryImpl(TimetableFactory):

    @staticmethod
    def _add_to_set(s: Set[str], t: str) -> Set[str]:
        return TimetableFactoryImpl._concat_set(s, {t})

    @staticmethod
    def _concat_set(s: Set[str], s2: Set[str]) -> Set[str]:
        return s.union(s2)

    class TimetableData(Timetable):
        def __init__(self, activities_set: Set[str], days_set: Set[str], data_func: Callable[[str, str], int]):
            self._activities: Set[str] = activities_set
            self._days: Set[str] = days_set
            self._data: Callable[[str, str], int] = data_func

        def get_single_data(self, activity: str, day: str) -> int:
            return self._data(activity, day)

        def add_hour(self, activity: str, day: str) -> 'TimetableFactoryImpl.TimetableData':
            return TimetableFactoryImpl.TimetableData(
                TimetableFactoryImpl._add_to_set(self._activities, activity),
                TimetableFactoryImpl._add_to_set(self._days, day),
                lambda a, d: self._data(a, d) + (1 if activity == a and day == d else 0)
            )

        def _statistics(self, predicate: Callable[[str, str], bool]) -> int:
            return sum(
                self.get_single_data(a, d)
                for a in self._activities
                for d in self._days
                if predicate(a, d)
            )

        def sums(self, activities_set: Set[str], days_set: Set[str]) -> int:
            return self._statistics(lambda a, d: a in activities_set and d in days_set)

        def activities(self) -> Set[str]:
            return self._activities

        def days(self) -> Set[str]:
            return self._days

    def empty(self) -> TimetableData:
        return TimetableFactoryImpl.TimetableData(set(), set(), lambda a, d: 0)

    def single(self, activity: str, day: str) -> TimetableData:
        return self.empty().add_hour(activity, day)

    def join(self, table1: Timetable, table2: Timetable) -> TimetableData:
        return TimetableFactoryImpl.TimetableData(
            TimetableFactoryImpl._concat_set(table1.activities(), table2.activities()),
            TimetableFactoryImpl._concat_set(table1.days(), table2.days()),
            lambda a, d: table1.get_single_data(a, d) + table2.get_single_data(a, d)
        )

    def cut(self, table: Timetable, bounds: Callable[[str, str], int]) -> TimetableData:
        return TimetableFactoryImpl.TimetableData(
            table.activities(),
            table.days(),
            lambda a, d: min(table.get_single_data(a, d), bounds(a, d))
        )


# Test File
import unittest

class TestTimetable(unittest.TestCase):
    """
    Implement the TimetableFactory interface as indicated in the init_factory
    method below. Create a factory for a timetable concept, captured by the
    Timetable interface: essentially it is a table that associates a number of
    hours spent (>=0) with each day and type of activity.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the score:

    - implementation of all factory methods (i.e., in the
    mandatory part it is sufficient to implement all of them except one at will --
    the first, empty, is mandatory)
    - the good design of the solution, using design solutions that lead to
    succinct code that avoids repetitions

    Remove the comment from the init_factory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.factory = TimetableFactoryImpl()

    def test_empty(self):
        # an empty table, without days and activities
        table = self.factory.empty()
        self.assertEqual(set(), table.activities())
        self.assertEqual(set(), table.days())
        self.assertEqual(0, table.get_single_data("act1", "day2"))
        self.assertEqual(0, table.sums(set(["act"]), set(["day"])))

        # now a table with one hour of activity "act" on day "day"
        table = table.add_hour("act", "day")
        self.assertEqual(set(["act"]), table.activities())
        self.assertEqual(set(["day"]), table.days())
        self.assertEqual(1, table.get_single_data("act", "day"))
        self.assertEqual(0, table.get_single_data("act0", "day0")) # no
        self.assertEqual(1, table.sums(set(["act"]), set(["day"]))) # hours of "act" on day "day"
        self.assertEqual(0, table.sums(set(["act"]), set(["day0"]))) # no

    def test_single(self):
        # single behaves like an empty one with the addition of an hour, as above
        table = self.factory.single("act1", "day1")
        self.assertEqual(set(["act1"]), table.activities())
        self.assertEqual(set(["day1"]), table.days())
        self.assertEqual(1, table.get_single_data("act1", "day1"))
        self.assertEqual(0, table.get_single_data("act0", "day0"))
        self.assertEqual(1, table.sums(set(["act1"]), set(["day1"])))
        self.assertEqual(0, table.sums(set(["act1"]), set()))

        # I add 3 hours to table, it becomes 4
        table = table.add_hour("act1", "day1") # note now I have 2 hours for act1 in day1
        table = table.add_hour("act1", "day2")
        table = table.add_hour("act2", "day2")
        self.assertEqual(set(["act1", "act2"]), table.activities())
        self.assertEqual(set(["day1", "day2"]), table.days())
        self.assertEqual(2, table.get_single_data("act1", "day1"))
        self.assertEqual(1, table.get_single_data("act1", "day2"))
        self.assertEqual(1, table.get_single_data("act2", "day2"))
        self.assertEqual(0, table.get_single_data("act2", "day1"))
        self.assertEqual(2, table.sums(set(["act1"]), set(["day1"])))
        self.assertEqual(0, table.sums(set(["act2"]), set(["day1"])))
        self.assertEqual(4, table.sums(set(["act1", "act2"]), set(["day1", "day2"]))) # total hours

    def test_join(self):
        table1 = self.factory.empty()
        table1 = table1.add_hour("act1", "day1")
        table1 = table1.add_hour("act1", "day1")
        table1 = table1.add_hour("act2", "day2")

        table2 = self.factory.empty()
        table2 = table2.add_hour("act2", "day1")
        table2 = table2.add_hour("act2", "day2")
        table2 = table2.add_hour("act1", "day3")
        table2 = table2.add_hour("act3", "day3")

        # I join the hours of two different tables: they are added up
        table = self.factory.join(table1, table2)
        self.assertEqual(set(["act1", "act2", "act3"]), table.activities())
        self.assertEqual(set(["day1", "day2", "day3"]), table.days())
        self.assertEqual(2, table.get_single_data("act1", "day1"))
        self.assertEqual(2, table.get_single_data("act2", "day2"))
        self.assertEqual(1, table.get_single_data("act1", "day3"))
        self.assertEqual(0, table.get_single_data("act2", "day3"))

        self.assertEqual(7, table.sums(set(["act1", "act2", "act3"]), set(["day1", "day2", "day3"])))
        self.assertEqual(6, table.sums(set(["act1", "act2"]), set(["day1", "day2", "day3"])))
        self.assertEqual(2, table.sums(set(["act1", "act2", "act3"]), set(["day3"])))

        # I can add a single hour to the usual
        table = table.add_hour("act1", "day1")
        self.assertEqual(8, table.sums(set(["act1", "act2", "act3"]), set(["day1", "day2", "day3"])))

    def test_bounds(self):
        table = self.factory.empty()
        table = table.add_hour("act1", "day1")
        table = table.add_hour("act1", "day1")
        table = table.add_hour("act1", "day1")
        table = table.add_hour("act1", "day2")
        table = table.add_hour("act1", "day3")
        table = table.add_hour("act1", "day3")
        table = table.add_hour("act2", "day1")
        table = table.add_hour("act2", "day1")
        table = table.add_hour("act3", "day2")
        table = table.add_hour("act3", "day3")

        # given a table with 10 hours as above, I remove them all
        table = self.factory.cut(table, lambda a, d: 0)
        self.assertEqual(set(["act1", "act2", "act3"]), table.activities())
        self.assertEqual(set(["day1", "day2", "day3"]), table.days())
        self.assertEqual(0, table.sums(set(["act1", "act2", "act3"]), set(["day1", "day2", "day3"])))

        table = self.factory.empty()
        table = table.add_hour("act1", "day1")
        table = table.add_hour("act1", "day1")
        table = table.add_hour("act1", "day1")
        table = table.add_hour("act1", "day2")
        table = table.add_hour("act1", "day3")
        table = table.add_hour("act1", "day3")
        table = table.add_hour("act2", "day1")
        table = table.add_hour("act2", "day1")
        table = table.add_hour("act3", "day2")
        table = table.add_hour("act3", "day3")

        # given a table with 10 hours as above, I allow a maximum of 1 per day per activity, they become 6
        table = self.factory.cut(table, lambda a, d: 1)
        self.assertEqual(set(["act1", "act2", "act3"]), table.activities())
        self.assertEqual(set(["day1", "day2", "day3"]), table.days())
        self.assertEqual(1, table.get_single_data("act1", "day1"))
        self.assertEqual(6, table.sums(set(["act1", "act2", "act3"]), set(["day1", "day2", "day3"])))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)