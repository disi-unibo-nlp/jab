# Utility Files
from enum import Enum
from typing import Set, List, Dict, Optional

class Day(Enum):
    MON = "MON"
    TUE = "TUE"
    WED = "WED"
    THU = "THU"
    FRI = "FRI"

class TimetableFactory:
    def empty(self) -> 'Timetable':
        """
        Creates an empty timetable.
        """
        raise NotImplementedError()

class Timetable:
    def rooms(self) -> Set[str]:
        """
        Returns the set of rooms used in the timetable.
        """
        raise NotImplementedError()

    def courses(self) -> Set[str]:
        """
        Returns the set of courses in the timetable.
        """
        raise NotImplementedError()

    def hours(self) -> List[int]:
        """
        Returns a sorted list of all hours at which bookings are made.
        """
        raise NotImplementedError()

    def add_booking(self, room: str, course: str, day: Day, hour: int, duration: int) -> 'Timetable':
        """
        Adds a booking to the timetable. Returns a new Timetable instance.
        """
        raise NotImplementedError()

    def find_place_for_booking(self, room: str, day: Day, duration: int) -> Optional[int]:
        """
        Finds the first available hour in a given room and day for a booking of given duration.
        Returns the hour or None if no place is available.
        """
        raise NotImplementedError()

    def get_day_at_room(self, room: str, day: Day) -> Dict[int, str]:
        """
        Returns a map from hour to course for a given room and day.
        """
        raise NotImplementedError()

    def get_day_and_hour(self, day: Day, hour: int) -> Optional[tuple[str, str]]:
        """
        Returns a pair (course, room) if there is a booking at the given day and hour, otherwise None.
        """
        raise NotImplementedError()

    def get_course_table(self, course: str) -> Dict[Day, Dict[int, str]]:
        """
        Returns a map from day to a map from hour to room for a given course.
        """
        raise NotImplementedError()


# Solution
class TimetableFactoryImpl(TimetableFactory):
    def empty(self) -> 'Timetable':
        return TimetableImpl(set())

class TimetableImpl(Timetable):
    def __init__(self, data: Set[tuple[str, str, Day, int]]):
        self._data = data

    def rooms(self) -> Set[str]:
        return {booking[0] for booking in self._data}

    def courses(self) -> Set[str]:
        return {booking[1] for booking in self._data}

    def hours(self) -> List[int]:
        return sorted(list({booking[3] for booking in self._data}))

    def add_booking(self, room: str, course: str, day: Day, hour: int, duration: int) -> 'Timetable':
        new_bookings = set()
        for h in range(hour, hour + duration):
            new_bookings.add((room, course, day, h))
        return TimetableImpl(self._data.union(new_bookings))

    def find_place_for_booking(self, room: str, day: Day, duration: int) -> Optional[int]:
        booked_hours = {booking[3] for booking in self._data if booking[0] == room and booking[2] == day}
        for start_hour in range(9, 19): # Assuming hours are within 9-18, adjust as needed
            possible = True
            for h in range(start_hour, start_hour + duration):
                if h in booked_hours:
                    possible = False
                    break
            if possible:
                return start_hour
        return None

    def get_day_at_room(self, room: str, day: Day) -> Dict[int, str]:
        return {booking[3]: booking[1] for booking in self._data if booking[0] == room and booking[2] == day}

    def get_day_and_hour(self, day: Day, hour: int) -> Optional[tuple[str, str]]:
        for booking in self._data:
            if booking[2] == day and booking[3] == hour:
                return (booking[1], booking[0])
        return None

    def get_course_table(self, course: str) -> Dict[Day, Dict[int, str]]:
        course_table: Dict[Day, Dict[int, str]] = {}
        for booking in self._data:
            if booking[1] == course:
                day = booking[2]
                hour = booking[3]
                room = booking[0]
                if day not in course_table:
                    course_table[day] = {}
                course_table[day][hour] = room
        return course_table


# Test File
import unittest

class TestTimetable(unittest.TestCase):
    """
    Implement the TimetableFactory interface as indicated in the init_factory method below.
    Create a factory for a concept of "weekly university lesson timetable", captured by the
    Timesheet interface: essentially, it tracks on which days of the week lessons of which
    course are held, and in which classroom.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the total
    score:

    - implementation of all Timetable methods (i.e., in the
    mandatory part it is sufficient to implement all of them except for one of the three getXYZ methods, as desired)
    - the good design of the solution, using design solutions that lead to
    succinct code that avoids repetitions

    Remove the comment from the initFactory method.

    Scoring indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional Timetable method)
    - quality of the solution: 4 points (for good design)
    """

    def setUp(self):
        self.factory = TimetableFactoryImpl()

    def real(self) -> Timetable:
        # realistic lesson timetable (1st semester of the 2nd year, just finished, without labs)
        return self.factory.empty() \
            .add_booking("2.12", "OOP", Day.WED, 9, 3) \
            .add_booking("3.4", "MDP", Day.WED, 13, 3) \
            .add_booking("2.12", "SISOP", Day.THU, 9, 3) \
            .add_booking("2.12", "OOP", Day.THU, 13, 3) \
            .add_booking("2.12", "MDP", Day.FRI, 9, 2) \
            .add_booking("2.12", "SISOP", Day.FRI, 11, 3)

    def test_empty(self):
        # "empty" lesson timetable, let's say during the exam period...
        t = self.factory.empty()
        self.assertEqual(set(), t.courses())
        self.assertEqual(set(), t.rooms())
        self.assertEqual(list(), t.hours())
        self.assertEqual({}, t.get_course_table("OOP"))
        self.assertEqual(None, t.get_day_and_hour(Day.MON, 9))
        self.assertEqual({}, t.get_day_at_room("2.12", Day.WED))

    def test_bookings(self):
        # realistic lesson timetable, test of the methods courses, rooms and hours
        t = self.real()
        self.assertEqual(set({"OOP", "SISOP", "MDP"}), t.courses()) # the courses
        self.assertEqual(set({"2.12", "3.4"}), t.rooms()) # the rooms used
        self.assertEqual([9, 10, 11, 12, 13, 14, 15], t.hours()) # the hours of classroom used

    def test_find_place(self):
        # realistic lesson timetable
        t = self.real()
        # is there space in room 3.4 on Wednesday for 3 hours in a row? yes, at 9 (I provide the first possibility)
        self.assertEqual(9, t.find_place_for_booking("3.4", Day.WED, 3))
        # is there space in room 2.12 on Thursday for 2 hours in a row? no, completely full
        self.assertEqual(16, t.find_place_for_booking("2.12", Day.THU, 2))

    def test_course_table(self):
        # test of the getCourseTable method
        t = self.real()
        # test OOP timetable
        self.assertEqual({
            Day.WED: {9: "2.12", 10: "2.12", 11: "2.12"},
            Day.THU: {13: "2.12", 14: "2.12", 15: "2.12"}
        }, t.get_course_table("OOP"))
        # test MDP timetable
        self.assertEqual({
            Day.WED: {13: "3.4", 14: "3.4", 15: "3.4"},
            Day.FRI: {9: "2.12", 10: "2.12"}
        }, t.get_course_table("MDP"))

    def test_day_and_hour(self):
        # test of the getDayAndHour method
        t = self.real()
        # test of what is there on Wednesday at 9
        self.assertEqual(("OOP", "2.12"), t.get_day_and_hour(Day.WED, 9))
        # test of what is there on Thursday at 9
        self.assertEqual(("SISOP", "2.12"), t.get_day_and_hour(Day.THU, 9))
        # test of what is there on Wednesday at 12: nothing
        self.assertEqual(None, t.get_day_and_hour(Day.WED, 12))

    def test_day_at_room(self):
        # test of the getDayAtRoom method
        t = self.real()
        # test of what is there in 2.12 on Wednesday
        self.assertEqual({9: "OOP", 10: "OOP", 11: "OOP"}, t.get_day_at_room("2.12", Day.WED))
        # test of what is there in 2.12 on Thursday
        self.assertEqual(
            {9: "SISOP", 10: "SISOP", 11: "SISOP", 13: "OOP", 14: "OOP", 15: "OOP"},
            t.get_day_at_room("2.12", Day.THU))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)