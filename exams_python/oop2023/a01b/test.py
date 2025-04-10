# Utility Files
from typing import List, Dict, Tuple, Callable

class TimeSheetFactory:
    """
    TimeSheetFactory interface in Python.
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
        Creates a TimeSheet from raw data as a list of pairs of activity/day indices and hours.
        """
        raise NotImplementedError()

    def of_partial_map(self, activities: List[str], days: List[str], data: Dict[Tuple[str, str], int]) -> 'TimeSheet':
        """
        Creates a TimeSheet from a partial map of (activity, day) to hours.
        """
        raise NotImplementedError()


class TimeSheet:
    """
    TimeSheet interface in Python.
    """
    def activities(self) -> List[str]:
        """
        Returns the list of activities.
        """
        raise NotImplementedError()

    def days(self) -> List[str]:
        """
        Returns the list of days.
        """
        raise NotImplementedError()

    def get_single_data(self, activity: str, day: str) -> int:
        """
        Returns the hours for a given activity and day. Returns 0 if activity or day is not present.
        """
        raise NotImplementedError()

    def sums_per_activity(self) -> Dict[str, int]:
        """
        Returns a map of activity to the sum of hours for that activity across all days.
        """
        raise NotImplementedError()

    def sums_per_day(self) -> Dict[str, int]:
        """
        Returns a map of day to the sum of hours for that day across all activities.
        """
        raise NotImplementedError()


# Solution
class TimeSheetFactoryImpl(TimeSheetFactory):
    """
    Implementation of TimeSheetFactory.
    """

    class TimeSheetData(TimeSheet):
        """
        Implementation of TimeSheet as an immutable class.
        """
        def __init__(self, activities: List[str], days: List[str], fun: Callable[[str, str], int]):
            self._activities = list(activities) # Create copies to ensure immutability
            self._days = list(days)
            self._fun = fun

        def activities(self) -> List[str]:
            return list(self._activities) # Return a copy to maintain immutability

        def days(self) -> List[str]:
            return list(self._days) # Return a copy to maintain immutability

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

    def flat(self, num_activities: int, num_days: int, hours: int) -> TimeSheet:
        activities = [f"act{i}" for i in range(1, num_activities + 1)]
        days = [f"day{i}" for i in range(1, num_days + 1)]
        return TimeSheetFactoryImpl.TimeSheetData(
            activities,
            days,
            lambda a, d: hours
        )

    def of_raw_data(self, num_activities: int, num_days: int, data: List[Tuple[int, int]]) -> TimeSheet:
        activities = [f"act{i}" for i in range(1, num_activities + 1)]
        days = [f"day{i}" for i in range(1, num_days + 1)]

        def data_function(activity: str, day: str) -> int:
            activity_index = activities.index(activity)
            day_index = days.index(day)
            return sum(1 for p in data if p == (activity_index, day_index))

        return TimeSheetFactoryImpl.TimeSheetData(
            activities,
            days,
            data_function
        )


    def of_lists_of_lists(self, activities: List[str], days: List[str], data: List[List[int]]) -> TimeSheet:
        return TimeSheetFactoryImpl.TimeSheetData(
            activities,
            days,
            lambda a, d: data[activities.index(a)][days.index(d)]
        )

    def of_partial_map(self, activities: List[str], days: List[str], data: Dict[Tuple[str, str], int]) -> TimeSheet:
        return TimeSheetFactoryImpl.TimeSheetData(
            activities,
            days,
            lambda a, d: data.get((a, d), 0)
        )


# Test File
import unittest

class TestTimeSheet(unittest.TestCase):
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
        self.assertEqual(list(sheet.activities()), ["act1", "act2", "act3"])
        self.assertEqual(list(sheet.days()), ["day1", "day2", "day3", "day4", "day5"])
        self.assertEqual(sheet.get_single_data("act1", "day2"), 1)
        self.assertEqual(sheet.get_single_data("act2", "day3"), 1)
        self.assertEqual(sheet.get_single_data("act22", "day30"), 0) # activities/days not present, return 0
        self.assertEqual(sheet.sums_per_activity(), {"act1": 5, "act2": 5, "act3": 5}) # 5 hours per each activity
        self.assertEqual(sheet.sums_per_day(), {"day1": 3, "day2": 3, "day3": 3, "day4": 3, "day5": 3}) # 3 hours on each day

    def test_of_lists_of_lists(self):
        # a timesheet with 2 activities over 3 days, with provided names
        sheet = self.factory.of_lists_of_lists(
            ["a1","a2"],
            ["d1", "d2", "d3"],
            [
                [1,2,3], # activity a1: 1,2,3 hours in the 3 days d1, d2, d3, orderly
                [0,0,1]  # activity a2: 0,0,1 hours in the 3 days d1, d2, d3, orderly
            ])
        self.assertEqual(list(sheet.activities()), ["a1", "a2"])
        self.assertEqual(list(sheet.days()), ["d1", "d2", "d3"])
        self.assertEqual(sheet.get_single_data("a1", "d2"), 2)
        self.assertEqual(sheet.get_single_data("a1", "d3"), 3)
        self.assertEqual(sheet.get_single_data("a2", "d2"), 0)
        self.assertEqual(sheet.get_single_data("a2", "d3"), 1)
        self.assertEqual(sheet.sums_per_activity(), {"a1": 6, "a2": 1}) # hours per activity
        self.assertEqual(sheet.sums_per_day(), {"d1": 1, "d2": 2, "d3": 4}) # hours per day

    def test_of_raw_data(self):
        # a timesheet with 2 activities over 3 days, with standard names
        sheet = self.factory.of_raw_data(2, 3, [
            (0,0), # one hour on act1 and day1
            (0,1), # one hour on act1 and day2
            (0,1), # one hour on act1 and day2 (become two in total)
            (0,2), # one hour on act1 and day3
            (0,2), # one hour on act1 and day3 (become two in total)
            (0,2), # one hour on act1 and day3 (become three in total)
            (1,2)])# one hour on act2 and day3
        self.assertEqual(list(sheet.activities()), ["act1", "act2"])
        self.assertEqual(list(sheet.days()), ["day1", "day2", "day3"])
        self.assertEqual(sheet.get_single_data("act1", "day2"), 2)
        self.assertEqual(sheet.get_single_data("act1", "day3"), 3)
        self.assertEqual(sheet.get_single_data("act2", "day2"), 0)
        self.assertEqual(sheet.get_single_data("act2", "day3"), 1)
        self.assertEqual(sheet.sums_per_activity(), {"act1": 6, "act2": 1})
        self.assertEqual(sheet.sums_per_day(), {"day1": 1, "day2": 2, "day3": 4})

    def test_of_map(self):
        # a timesheet with 2 activities over 3 days, with provided names
        sheet = self.factory.of_partial_map(
            ["act1","act2"],
            ["day1", "day2", "day3"],
            { # map (activity, day) -> n_hours
                ("act1","day1"): 1,
                ("act1","day2"): 2,
                ("act1","day3"): 3,
                ("act2","day3"): 1})
        self.assertEqual(list(sheet.activities()), ["act1", "act2"])
        self.assertEqual(list(sheet.days()), ["day1", "day2", "day3"])
        self.assertEqual(sheet.get_single_data("act1", "day2"), 2)
        self.assertEqual(sheet.get_single_data("act1", "day3"), 3)
        self.assertEqual(sheet.get_single_data("act2", "day2"), 0)
        self.assertEqual(sheet.get_single_data("act2", "day3"), 1)
        self.assertEqual(sheet.sums_per_activity(), {"act1": 6, "act2": 1})
        self.assertEqual(sheet.sums_per_day(), {"day1": 1, "day2": 2, "day3": 4})

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)