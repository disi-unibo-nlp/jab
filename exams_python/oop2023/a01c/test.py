# Utility Files
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set, Callable

class TimeSheetFactory(ABC):
    """
    Factory interface for creating TimeSheet instances.
    """
    @abstractmethod
    def of_raw_data(self, data: List[Tuple[str, str]]) -> 'TimeSheet':
        """
        Creates a TimeSheet from raw data.

        Args:
            data: List of pairs (activity, day).

        Returns:
            A TimeSheet instance.
        """
        pass

    @abstractmethod
    def with_bounds_per_activity(self, data: List[Tuple[str, str]], bounds_on_activities: Dict[str, int]) -> 'TimeSheet':
        """
        Creates a TimeSheet with bounds per activity.

        Args:
            data: List of pairs (activity, day).
            bounds_on_activities: Dictionary of activity bounds (activity -> max hours).

        Returns:
            A TimeSheet instance with activity bounds.
        """
        pass

    @abstractmethod
    def with_bounds_per_day(self, data: List[Tuple[str, str]], bounds_on_days: Dict[str, int]) -> 'TimeSheet':
        """
        Creates a TimeSheet with bounds per day.

        Args:
            data: List of pairs (activity, day).
            bounds_on_days: Dictionary of day bounds (day -> max hours).

        Returns:
            A TimeSheet instance with day bounds.
        """
        pass

    @abstractmethod
    def with_bounds(self, data: List[Tuple[str, str]], bounds_on_activities: Dict[str, int], bounds_on_days: Dict[str, int]) -> 'TimeSheet':
        """
        Creates a TimeSheet with bounds per activity and per day.

        Args:
            data: List of pairs (activity, day).
            bounds_on_activities: Dictionary of activity bounds (activity -> max hours).
            bounds_on_days: Dictionary of day bounds (day -> max hours).

        Returns:
            A TimeSheet instance with both activity and day bounds.
        """
        pass


class TimeSheet(ABC):
    """
    Interface for a TimeSheet, representing hours spent on activities per day.
    """
    @abstractmethod
    def activities(self) -> Set[str]:
        """
        Returns the set of activities in the timesheet.

        Returns:
            Set of activity names.
        """
        pass

    @abstractmethod
    def days(self) -> Set[str]:
        """
        Returns the set of days in the timesheet.

        Returns:
            Set of day names.
        """
        pass

    @abstractmethod
    def get_single_data(self, activity: str, day: str) -> int:
        """
        Gets the hours spent for a single activity on a single day.

        Args:
            activity: Activity name.
            day: Day name.

        Returns:
            Hours spent (>=0).
        """
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """
        Checks if the timesheet is valid according to defined bounds (if any).

        Returns:
            True if valid, False otherwise.
        """
        pass


# Solution
from typing import List, Tuple, Dict, Set, Callable


class TimeSheetFactoryImpl(TimeSheetFactory):
    """
    Implementation of TimeSheetFactory interface.
    """

    class TimeSheetData(TimeSheet):
        """
        Immutable implementation of TimeSheet.
        """
        def __init__(self, activities: Set[str], days: Set[str], fun: Callable[[str, str], int]):
            self._activities = activities
            self._days = days
            self._fun = fun

        def activities(self) -> Set[str]:
            return self._activities

        def days(self) -> Set[str]:
            return self._days

        def get_single_data(self, activity: str, day: str) -> int:
            return self._fun(activity, day)

        def is_valid(self) -> bool:
            return True

    class TimeSheetDecorator(TimeSheet):
        """
        Decorator class for TimeSheet to add validation policies.
        """
        def __init__(self, base: TimeSheet):
            self._base = base

        def activities(self) -> Set[str]:
            return self._base.activities()

        def days(self) -> Set[str]:
            return self._base.days()

        def get_single_data(self, activity: str, day: str) -> int:
            return self._base.get_single_data(activity, day)

        def is_valid(self) -> bool:
            return self._base.is_valid()

        def sum_per_activity(self, activity: str) -> int:
            return sum(self.get_single_data(activity, day) for day in self.days())

        def sum_per_day(self, day: str) -> int:
            return sum(self.get_single_data(act, day) for act in self.activities())

    def of_raw_data(self, data: List[Tuple[str, str]]) -> TimeSheet:
        """
        Creates a TimeSheet from raw data.
        """
        activities = sorted(set(d[0] for d in data))
        days = sorted(set(d[1] for d in data))

        def data_function(activity: str, day: str) -> int:
            return sum(1 for p in data if p[0] == activity and p[1] == day)

        return TimeSheetFactoryImpl.TimeSheetData(set(activities), set(days), data_function)

    def _decorate_with_bounds_per_activity(self, base: TimeSheet, bounds: Dict[str, int]) -> TimeSheet:
        """
        Decorates a TimeSheet with activity bounds.
        """
        class ActivityBoundsDecorator(TimeSheetFactoryImpl.TimeSheetDecorator):
            def is_valid(self) -> bool:
                return super().is_valid() and all(self.sum_per_activity(activity) <= bound for activity, bound in bounds.items())
        return ActivityBoundsDecorator(base)

    def _decorate_with_bounds_per_day(self, base: TimeSheet, bounds: Dict[str, int]) -> TimeSheet:
        """
        Decorates a TimeSheet with day bounds.
        """
        class DayBoundsDecorator(TimeSheetFactoryImpl.TimeSheetDecorator):
            def is_valid(self) -> bool:
                return super().is_valid() and all(self.sum_per_day(day) <= bound for day, bound in bounds.items())
        return DayBoundsDecorator(base)

    def _decorate_with_bounds(self, base: TimeSheet, bounds_per_activity: Dict[str, int], bounds_per_day: Dict[str, int]) -> TimeSheet:
        """
        Decorates a TimeSheet with both activity and day bounds.
        """
        return self._decorate_with_bounds_per_activity(self._decorate_with_bounds_per_day(base, bounds_per_day), bounds_per_activity)

    def with_bounds_per_activity(self, data: List[Tuple[str, str]], bounds: Dict[str, int]) -> TimeSheet:
        """
        Creates a TimeSheet with bounds per activity.
        """
        return self._decorate_with_bounds_per_activity(self.of_raw_data(data), bounds)

    def with_bounds_per_day(self, data: List[Tuple[str, str]], bounds: Dict[str, int]) -> TimeSheet:
        """
        Creates a TimeSheet with bounds per day.
        """
        return self._decorate_with_bounds_per_day(self.of_raw_data(data), bounds)

    def with_bounds(self, data: List[Tuple[str, str]], bounds_on_activities: Dict[str, int], bounds_on_days: Dict[str, int]) -> TimeSheet:
        """
        Creates a TimeSheet with bounds per activity and per day.
        """
        return self._decorate_with_bounds(self.of_raw_data(data), bounds_on_activities, bounds_on_days)


# Test File
import unittest
from typing import List, Tuple, Dict, Set
# Removed the incorrect import statement
# from timesheet_factory_impl import TimeSheetFactoryImpl, TimeSheet, TimeSheetFactory

class TestTimeSheetFactory(unittest.TestCase):
    """
    Implement the TimeSheetFactory interface as indicated in the init_factory
    method below. Creates a factory for a Timesheet concept, captured by the
    Timesheet interface: essentially it is a table with days (typically days of
    a month) in the columns, and work activities in the rows (for example:
    "teaching", "project research 1", "project research 2",...) where in each
    cell it reports how many hours (>=0) have been spent on a certain day for a
    certain activity.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the score:

    - implementation of all methods of the factory (i.e., in the mandatory part
    it is sufficient to implement all of them except one at will)
    - the good design of the solution, using design solutions that lead to
    succinct code that avoids repetitions

    Remove the comment from the init_factory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.factory: TimeSheetFactory = TimeSheetFactoryImpl()

    def basic_data(self) -> List[Tuple[str, str]]:
        """
        An example of data for a timesheet, used in the tests below
        """
        return [
            ("act1", "day1"),
            ("act1", "day2"),
            ("act1", "day2"),  # so two hours of activity act1 in day2
            ("act1", "day3"),
            ("act1", "day3"),
            ("act1", "day3"),  # so three hours of activity act1 in day2
            ("act2", "day3")
        ]

    def _assert_example_time_sheet(self, sheet: TimeSheet):
        """
        a set of tests for a timesheet obtained from basic_data()
        """
        self.assertEqual(set(["act1", "act2"]), sheet.activities())  # two activities extracted from the timesheet data
        self.assertEqual(set(["day1", "day2", "day3"]), sheet.days())  # three days extracted from the timesheet data
        self.assertEqual(1, sheet.get_single_data("act1", "day1"))
        self.assertEqual(2, sheet.get_single_data("act1", "day2"))  # 2 hours present in the data
        self.assertEqual(3, sheet.get_single_data("act1", "day3"))  # 3 hours present in the data
        self.assertEqual(0, sheet.get_single_data("act2", "day2"))
        self.assertEqual(1, sheet.get_single_data("act2", "day3"))

    def test_of_raw_data(self):
        """
        the timesheet obtained from an empty list has no activities or days
        """
        empty_sheet = self.factory.of_raw_data([])
        self.assertEqual(set(), empty_sheet.activities())
        self.assertEqual(set(), empty_sheet.days())

        # the timesheet obtained from basic_data() passes the test of _assert_example_time_sheet
        self._assert_example_time_sheet(self.factory.of_raw_data(self.basic_data()))

    def test_with_bounds_per_activity(self):
        """
        Tests timesheet with bounds per activity.
        """
        sheet = self.factory.with_bounds_per_activity(
            self.basic_data(),
            {"act1": 7, "act2": 7}  # max 7 hours on act1, and 7 on act2
        )
        self._assert_example_time_sheet(sheet)
        # it is a valid timesheet
        self.assertTrue(sheet.is_valid())

        # Same test below, but with stricter constraints, which do not make it valid
        sheet2 = self.factory.with_bounds_per_activity(
            self.basic_data(),
            {"act1": 4, "act2": 3}
        )
        self._assert_example_time_sheet(sheet2)
        self.assertFalse(sheet2.is_valid())

    def test_with_bounds_per_day(self):
        """
        Tests timesheet with bounds per day.
        """
        sheet = self.factory.with_bounds_per_day(
            self.basic_data(),
            {"day1": 8, "day2": 8, "day3": 8}  # max 8 hours on day,...
        )
        self._assert_example_time_sheet(sheet)
        # it is a valid timesheet
        self.assertTrue(sheet.is_valid())

        # Same test below, but with stricter constraints, which do not make it valid
        sheet2 = self.factory.with_bounds_per_day(
            self.basic_data(),
            {"day1": 2, "day2": 2, "day3": 2}
        )
        self._assert_example_time_sheet(sheet2)
        self.assertFalse(sheet2.is_valid())

    def test_with_bounds(self):
        """
        Tests timesheet with bounds on both activities and days.
        """
        sheet = self.factory.with_bounds(
            self.basic_data(),
            {"act1": 7, "act2": 4},  # max 7 hours on act1...
            {"day1": 8, "day2": 8, "day3": 8}  # max 8 hours on day1...
        )
        self._assert_example_time_sheet(sheet)
        # it is a valid timesheet
        self.assertTrue(sheet.is_valid())

        # Same test below, but with stricter constraints, which do not make it valid
        sheet2 = self.factory.with_bounds(
            self.basic_data(),
            {"act1": 4, "act2": 4},
            {"day1": 2, "day2": 2, "day3": 2}
        )
        self._assert_example_time_sheet(sheet2)
        self.assertFalse(sheet2.is_valid())

if __name__ == '__main__':
    unittest.main()