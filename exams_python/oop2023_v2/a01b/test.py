# Utility Files
from typing import List, Dict, Callable, Tuple

class TimeSheetFactory:
    """
    Factory for creating TimeSheet objects.
    """
    def flat(self, num_activities: int, num_days: int, hours: int) -> 'TimeSheet':
        """
        Creates a TimeSheet where every activity on every day has the same number of hours.
        """
        raise NotImplementedError()

    def of_lists_of_lists(self, activities: List[str], days: List[str], data: List[List[int]]) -> 'TimeSheet':
        """
        Creates a TimeSheet from lists of activities, days, and a 2D list of hours.
        """
        raise NotImplementedError()

    def of_raw_data(self, num_activities: int, num_days: int, data: List[Tuple[int, int]]) -> 'TimeSheet':
        """
        Creates a TimeSheet from raw data represented as a list of pairs of activity and day indices.
        """
        raise NotImplementedError()

    def of_partial_map(self, activities: List[str], days: List[str], data: Dict[Tuple[str, str], int]) -> 'TimeSheet':
        """
        Creates a TimeSheet from a partial map where keys are (activity, day) pairs and values are hours.
        """
        raise NotImplementedError()


class TimeSheet:
    """
    Interface representing a TimeSheet.
    """
    def activities(self) -> List[str]:
        """
        Returns the list of activities in the TimeSheet.
        """
        raise NotImplementedError()

    def days(self) -> List[str]:
        """
        Returns the list of days in the TimeSheet.
        """
        raise NotImplementedError()

    def get_single_data(self, activity: str, day: str) -> int:
        """
        Returns the hours spent on a specific activity on a specific day.
        Returns 0 if activity or day is not present.
        """
        raise NotImplementedError()

    def sums_per_activity(self) -> Dict[str, int]:
        """
        Returns a map of total hours per activity.
        """
        raise NotImplementedError()

    def sums_per_day(self) -> Dict[str, int]:
        """
        Returns a map of total hours per day.
        """
        raise NotImplementedError()


# Solution
class TimeSheetFactoryImpl(TimeSheetFactory):
    """
    Implementation of TimeSheetFactory interface.
    """

    class TimeSheetData(TimeSheet):
        """
        Implementation of TimeSheet as an immutable class.
        """
        def __init__(self, activities: List[str], days: List[str], fun: Callable[[str, str], int]):
            self._activities = list(activities)  # Create copies to ensure immutability
            self._days = list(days)
            self._fun = fun

        def activities(self) -> List[str]:
            return list(self._activities)  # Return a copy for immutability

        def days(self) -> List[str]:
            return list(self._days) # Return a copy for immutability

        def get_single_data(self, activity: str, day: str) -> int:
            return self._fun(activity, day) if activity in self._activities and day in self._days else 0

        def sums_per_activity(self) -> Dict[str, int]:
            return {
                act: sum(self._fun(act, day) for day in self._days)
                for act in self._activities
            }

        def sums_per_day(self) -> Dict[str, int]:
            return {
                day: sum(self._fun(act, day) for act in self._activities)
                for day in self._days
            }

    def _create_activities(self, num_activities: int) -> List[str]:
        return [f"act{i}" for i in range(1, num_activities + 1)]

    def _create_days(self, num_days: int) -> List[str]:
        return [f"day{i}" for i in range(1, num_days + 1)]

    def flat(self, num_activities: int, num_days: int, hours: int) -> 'TimeSheet':
        activities = self._create_activities(num_activities)
        days = self._create_days(num_days)
        return TimeSheetFactoryImpl.TimeSheetData(
            activities,
            days,
            lambda a, d: hours
        )

    def of_raw_data(self, num_activities: int, num_days: int, data: List[Tuple[int, int]]) -> 'TimeSheet':
        activities = self._create_activities(num_activities)
        days = self._create_days(num_days)
        return TimeSheetFactoryImpl.TimeSheetData(
            activities,
            days,
            lambda a, d: sum(1 for p in data if p[0] == activities.index(a) and p[1] == days.index(d))
        )

    def of_lists_of_lists(self, activities: List[str], days: List[str], data: List[List[int]]) -> 'TimeSheet':
        return TimeSheetFactoryImpl.TimeSheetData(
            list(activities),
            list(days),
            lambda a, d: data[activities.index(a)][days.index(d)]
        )

    def of_partial_map(self, activities: List[str], days: List[str], data: Dict[Tuple[str, str], int]) -> 'TimeSheet':
        return TimeSheetFactoryImpl.TimeSheetData(
            list(activities),
            list(days),
            lambda a, d: data.get((a, d), 0)
        )


# Test File
import unittest

class TimeSheetTest(unittest.TestCase):
    """
    Implement the TimeSheetFactory interface as indicated in the init_factory
    method below. Implement a factory for a Timesheet concept, captured by the
    Timesheet interface: essentially it is a table with days (typically days of
    a month) in the columns, and work activities in the rows (for example:
    "teaching", "research project 1", "research project 2",...) where each cell
    shows how many hours (>=0) have been spent on a certain day for a certain
    activity.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the score:

    - Implementation of all factory methods (i.e., in the mandatory part it is
    sufficient to implement all but one at will)
    - Good design of the solution, using design solutions that lead to
    succinct code that avoids repetitions

    Remove the comment from the init_factory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.factory = TimeSheetFactoryImpl()

    def test_flat(self):
        # a time sheet with 3 activities over 5 days, with one hour per day on each activity on each day
        sheet = self.factory.flat(3, 5, 1)
        self.assertEqual(["act1", "act2", "act3"], sheet.activities())
        self.assertEqual(["day1", "day2", "day3", "day4", "day5"], sheet.days())
        self.assertEqual(1, sheet.get_single_data("act1", "day2"))
        self.assertEqual(1, sheet.get_single_data("act2", "day3"))
        self.assertEqual(0, sheet.get_single_data("act22", "day30")) # activities/days not present, return 0
        self.assertEqual({"act1":5, "act2":5, "act3":5}, sheet.sums_per_activity()) # 5 hours per each activity
        self.assertEqual({"day1":3, "day2":3, "day3":3, "day4":3, "day5":3}, sheet.sums_per_day()) # 3 hours on each day

    def test_of_lists_of_lists(self):
        # a timesheet with 2 activities over 3 days, with provided names
        sheet = self.factory.of_lists_of_lists(
            ["a1","a2"],
            ["d1", "d2", "d3"],
            [
                [1,2,3], # activity a1: 1,2,3 hours in the 3 days d1, d2, d3, orderly
                [0,0,1]  # activity a2: 0,0,1 hours in the 3 days d1, d2, d3, orderly
            ])
        self.assertEqual(["a1", "a2"], sheet.activities())
        self.assertEqual(["d1", "d2", "d3"], sheet.days())
        self.assertEqual(2, sheet.get_single_data("a1", "d2"))
        self.assertEqual(3, sheet.get_single_data("a1", "d3"))
        self.assertEqual(0, sheet.get_single_data("a2", "d2"))
        self.assertEqual(1, sheet.get_single_data("a2", "d3"))
        self.assertEqual({"a1":6, "a2":1}, sheet.sums_per_activity()) # hours per activity
        self.assertEqual({"d1":1, "d2":2, "d3":4}, sheet.sums_per_day()) # hours per day

    def test_of_raw_data(self):
        # a timesheet with 2 activities over 3 days, with standard names
        sheet = self.factory.of_raw_data(2, 3, [(0,0), (0,1), (0,1), (0,2), (0,2), (0,2), (1,2)])
        self.assertEqual(["act1", "act2"], sheet.activities())
        self.assertEqual(["day1", "day2", "day3"], sheet.days())
        self.assertEqual(2, sheet.get_single_data("act1", "day2"))
        self.assertEqual(3, sheet.get_single_data("act1", "day3"))
        self.assertEqual(0, sheet.get_single_data("act2", "day2"))
        self.assertEqual(1, sheet.get_single_data("act2", "day3"))
        self.assertEqual({"act1":6, "act2":1}, sheet.sums_per_activity())
        self.assertEqual({"day1":1, "day2":2, "day3":4}, sheet.sums_per_day())

    def test_of_map(self):
        # a timesheet with 2 activities over 3 days, with provided names
        sheet = self.factory.of_partial_map(
            ["act1","act2"],
            ["day1", "day2", "day3"],
            { # map (activity, day) -> n_hours
                ("act1","day1"):1,
                ("act1","day2"):2,
                ("act1","day3"):3,
                ("act2","day3"):1})
        self.assertEqual(["act1", "act2"], sheet.activities())
        self.assertEqual(["day1", "day2", "day3"], sheet.days())
        self.assertEqual(2, sheet.get_single_data("act1", "day2"))
        self.assertEqual(3, sheet.get_single_data("act1", "day3"))
        self.assertEqual(0, sheet.get_single_data("act2", "day2"))
        self.assertEqual(1, sheet.get_single_data("act2", "day3"))
        self.assertEqual({"act1":6, "act2":1}, sheet.sums_per_activity())
        self.assertEqual({"day1":1, "day2":2, "day3":4}, sheet.sums_per_day())

if __name__ == '__main__':
    unittest.main()