# Utility Files
from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Tuple, Optional

T = TypeVar('T')
E1 = TypeVar('E1')
E2 = TypeVar('E2')


class RulesEngineFactory(ABC):
    @abstractmethod
    def apply_rule(self, rule: Tuple[T, List[T]], input_list: List[T]) -> List[List[T]]:
        raise NotImplementedError

    @abstractmethod
    def single_rule_engine(self, rule: Tuple[T, List[T]]) -> 'RulesEngine[T]':
        raise NotImplementedError

    @abstractmethod
    def cascading_rules_engine(self, base_rule: Tuple[T, List[T]], cascade_rule: Tuple[T, List[T]]) -> 'RulesEngine[T]':
        raise NotImplementedError

    @abstractmethod
    def conflicting_rules_engine(self, rule1: Tuple[T, List[T]], rule2: Tuple[T, List[T]]) -> 'RulesEngine[T]':
        raise NotImplementedError


class RulesEngine(ABC, Generic[T]):
    @abstractmethod
    def reset_input(self, input_list: List[T]) -> None:
        raise NotImplementedError

    @abstractmethod
    def has_other_solutions(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_solution(self) -> List[T]:
        raise NotImplementedError


# Solution
from typing import Iterator

class RulesEngineFactoryImpl(RulesEngineFactory):
    def _replace_at_position(self, index: int, source: List[T], new_elements: List[T]) -> List[T]:
        l = list(source)
        del l[index]
        l[index:index] = new_elements
        return l

    def apply_rule(self, rule: Tuple[T, List[T]], input_list: List[T]) -> List[List[T]]:
        solutions = []
        rule_head = rule[0]
        rule_body = rule[1]
        for i, element in enumerate(input_list):
            if element == rule_head:
                solutions.append(self._replace_at_position(i, input_list, rule_body))
        return solutions

    def _applicable(self, rules: List[Tuple[T, List[T]]], input_list: List[T]) -> bool:
        return any(rule[0] in input_list for rule in rules)

    def _apply_rules(self, rules: List[Tuple[T, List[T]]], input_list: List[T]) -> Iterator[List[T]]:
        if not self._applicable(rules, input_list):
            yield input_list
        else:
            generated_solutions = set() # use set to track generated solutions to ensure distinctness
            for rule in rules:
                rule_applications = self.apply_rule(rule, input_list)
                for next_input in rule_applications:
                    for solution in self._apply_rules(rules, next_input):
                        solution_tuple = tuple(solution) # lists are not hashable, convert to tuple for set
                        if solution_tuple not in generated_solutions:
                            generated_solutions.add(solution_tuple)
                            yield solution

    def _from_rules(self, rules: List[Tuple[T, List[T]]]) -> RulesEngine[T]:
        class AnonymousRulesEngine(RulesEngine[T]):
            def __init__(self_inner):
                self_inner._iterator: Optional[Iterator[List[T]]] = None
                self_inner._next_val: Optional[List[T]] = None

            def reset_input(self_inner, input_list: List[T]) -> None:
                self_inner._iterator = self._apply_rules(rules, input_list)
                self_inner._next_val = None

            def has_other_solutions(self_inner) -> bool:
                if self_inner._iterator is None:
                    return False
                if self_inner._next_val is not None:
                    return True
                try:
                    self_inner._next_val = next(self_inner._iterator)
                    return True
                except StopIteration:
                    self_inner._iterator = None
                    return False

            def next_solution(self_inner) -> List[T]:
                if self_inner._next_val is not None:
                    val = self_inner._next_val
                    self_inner._next_val = None
                    return val
                if self_inner._iterator is None:
                    raise StopIteration
                return next(self_inner._iterator)

        return AnonymousRulesEngine()

    def single_rule_engine(self, rule: Tuple[T, List[T]]) -> RulesEngine[T]:
        return self._from_rules([rule])

    def cascading_rules_engine(self, base_rule: Tuple[T, List[T]], cascade_rule: Tuple[T, List[T]]) -> RulesEngine[T]:
        return self._from_rules([base_rule, cascade_rule])

    def conflicting_rules_engine(self, rule1: Tuple[T, List[T]], rule2: Tuple[T, List[T]]) -> RulesEngine[T]:
        return self._from_rules([rule1, rule2])


# Test File
import unittest

class TestRulesEngine(unittest.TestCase):
    """
    Implement the RulesEngineFactory interface as indicated in the init_factory method
    below. Create a factory for a "rules engine" concept, captured by the RulesEngine
    interface: essentially it is a rewriting system of data sequences (lists), which
    starting from an input produces several outputs, applying rewriting rules.

    The following are considered optional for the purpose of being able to correct
    the exercise, but still contribute to achieving the totality of the score:

    - implementation of all factory methods (i.e., in the mandatory part it is
      sufficient to implement all but one at will)
    - the good design of the solution, using design solutions that lead to
      succinct code that avoids repetitions

    Remove the comment from the init_factory method.

    Score indications:
    - correctness of the mandatory part: 10 points
    - correctness of the optional part: 3 points (additional factory method)
    - quality of the solution: 4 points (for good design)
    """
    
    def setUp(self):
        self.factory = RulesEngineFactoryImpl()

    def test_apply_rule(self):
        # test of the method that implements the application of a single rule
        # rule: 10 --> (11,12)
        # if the input is (1,10,100) and therefore has only one 10, we get a single solution, i.e. (1,11,12,100)
        self.assertEqual(self.factory.apply_rule((10, [11, 12]), [1, 10, 100]),
                         [[1, 11, 12, 100]])

        # rule: a --> (b)
        # if the input is (a,a,a) we get 3 solutions, in each one we replace an a with b, in order
        self.assertEqual(self.factory.apply_rule(("a", ["b"]), ["a", "a", "a"]),
                         [['b', 'a', 'a'], ['a', 'b', 'a'], ['a', 'a', 'b']])

        # rule: 10 --> ()
        # if the input is (1,10,100) we have only one solution, where 10 is replaced by nothing, i.e. it is removed
        self.assertEqual(self.factory.apply_rule((10, []), [1, 10, 100]),
                         [[1, 100]])

        # rule: 10 --> (11,12,13)
        # if the input is (1,100) there is no solution
        self.assertEqual(self.factory.apply_rule((10, [11, 12, 13]), [1, 100]),
                         [])

    def test_single(self):
        # singleRule creates a RulesEngine, which works with the logic of the method above, only that:
        # 1 - applies the rule from left to right as long as it can, so there is always only one solution
        # 2 - must allow to iterate the different solutions (without replication),
        # 3 - it must be possible to reset the input to start again

        # rule: a --> (b,c)
        res = self.factory.single_rule_engine(("a", ["b", "c"]))
        # input: (a,z,z)
        res.reset_input(["a","z","z"])
        # only one solution
        self.assertTrue(res.has_other_solutions())
        self.assertEqual(res.next_solution(), ["b", "c", "z", "z"])
        self.assertFalse(res.has_other_solutions())

        # input: (a,z,a)
        res.reset_input(["a","z","a"])
        # only one solution, transforming both "a"s
        self.assertTrue(res.has_other_solutions())
        self.assertEqual(res.next_solution(), ["b", "c", "z", "b", "c"])
        self.assertFalse(res.has_other_solutions())

        # input: (z, z)
        res.reset_input(["z", "z"])
        # no possible transformation, so I have only one solution, which does not modify the input
        self.assertTrue(res.has_other_solutions())
        self.assertEqual(res.next_solution(), ["z", "z"])
        self.assertFalse(res.has_other_solutions())

    def test_cascading_rules(self):
        # This tests the case of two rules, where the result of the first one results in
        # a sequence that activates the second rule
        # a --> (b,c), and c-->d
        # in fact here, we have the same result that we would have with the rule: a-->b,d and c-->d
        res = self.factory.cascading_rules_engine(
            ("a", ["b", "c"]),
            ("c", ["d"])
        )
        res.reset_input(["a","z"])
        self.assertTrue(res.has_other_solutions())
        self.assertEqual(res.next_solution(), ["b", "d", "z"])
        self.assertFalse(res.has_other_solutions())

        # here the "a" becomes "b,d", and the "c" becomes "d"
        res.reset_input(["a","c","z"])
        self.assertTrue(res.has_other_solutions())
        self.assertEqual(res.next_solution(), ["b", "d", "d", "z"])
        self.assertFalse(res.has_other_solutions())

    def test_conflicting_rules(self):
        # This tests the case of two rules with the same "head", for each application of the rule
        # 2 cases are generated, whose solutions are then combined (eliminating duplicates).
        # That is, every time the rule has to be applied, one is applied with SingleRule,
        # the other with SingleRule, and the results are combined.

        # a --> b, and a-->c, d
        res = self.factory.conflicting_rules_engine(
            ("a", ["b"]),
            ("a", ["c", "d"])
        )
        res.reset_input(["a","a"])
        # 4 solutions, i.e. the combinations
        solutions = []
        while res.has_other_solutions():
            solutions.append(res.next_solution())
        expected_solutions = [['b', 'b'], ['b', 'c', 'd'], ['c', 'd', 'b'], ['c', 'd', 'c', 'd']]
        self.assertEqual(sorted(solutions, key=lambda x: tuple(x)), sorted(expected_solutions, key=lambda x: tuple(x)))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)