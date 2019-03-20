import pytest
from pytest import raises

from torchsim.core.models.expert_params import ExpertParams
from torchsim.gui.validators import *


# used in tests
GLOBAL_VARIABLE = 7


class TestValidator:
    @pytest.mark.parametrize("value, should_pass", [
        (None, False),
        (float(0), False),
        (float(1), True),
        (1.3, True),
        (float(-3), False),
        (-3.2, False),
        ('abc', False),
    ])
    def test_validate_positive_float(self, value, should_pass: bool):
        if should_pass:
            validate_positive_float(value)
        else:
            with raises(FailedValidationException, match="Strictly positive float expected"):
                validate_positive_float(value)

    @pytest.mark.parametrize("value, should_pass", [
        (None, True),
        (1, True),
        (0, False),
        (-3, False),
        ('abc', False),
    ])
    def test_validate_positive_optional_int(self, value, should_pass: bool):
        if should_pass:
            validate_positive_optional_int(value)
        else:
            with raises(FailedValidationException, match='Strictly positive int or None expected'):
                validate_positive_optional_int(value)

    @pytest.mark.parametrize("value, should_pass", [
        (None, False),
        (1, True),
        (0, True),
        (-1, False),
        ('abc', False),
    ])
    def test_validate_positive_with_zero_int(self, value, should_pass: bool):
        if should_pass:
            validate_positive_with_zero_int(value)
        else:
            with raises(FailedValidationException, match='Positive int or zero expected'):
                validate_positive_with_zero_int(value)

    @pytest.mark.parametrize("value, should_pass", [
        (None, False),
        (1, True),
        (0, False),
        (-1, False),
        ('abc', False),
    ])
    def test_validate_positive_int(self, value, should_pass: bool):
        if should_pass:
            validate_positive_int(value)
        else:
            with raises(FailedValidationException, match='Strictly positive int expected'):
                validate_positive_int(value)

    @pytest.mark.parametrize("value, should_pass", [
        ([[]], True),
        ([[0]], True),
        ([[1, 2], [3, 4], [5]], True),
        ([], True),  # empty list is allowed
        (None, False),
        (1, False),
        ('abc', False),
        ([1], False),
        ([[1.0]], False),
        ([['a']], False),
        ([[[1]]], False),
        ([[1], 2], False),
    ])
    def test_validate_list_list_int(self, value, should_pass: bool):
        if should_pass:
            validate_list_list_int(value)
        else:
            with raises(FailedValidationException):
                validate_list_list_int(value)

    @pytest.mark.parametrize("value, should_pass", [
        ([[]], True),
        ([[0.1]], True),
        ([[1.0, 2.1], [3.4, 4.0], [5.0]], True),  # accepts float and ints
        ([], True),  # empty list is allowed
        (None, False),
        (1, False),
        ('abc', False),
        ([1], False),
        ([['a']], False),
        ([[[1]]], False),
        ([[1], 2], False),
    ])
    def test_validate_list_list_float_or_int(self, value, should_pass: bool):
        if should_pass:
            validate_list_list_float_or_int(value)
        else:
            with raises(FailedValidationException):
                validate_list_list_float_or_int(value)

    @pytest.mark.parametrize("value, shape, should_pass", [
        (1, [], True),
        (-1, [], True),
        (0, [2, 4], True),
        (-1, [2, 5, 7], True),
        ('abc', [1, 2, 3], False),
        (3.0, [1, 2, 3], False),
        (2, [1], False)
    ])
    def test_validate_dimension_vs_shape(self, value, shape, should_pass: bool):
        if should_pass:
            validate_dimension_vs_shape(value, shape)
        else:
            with raises(FailedValidationException):
                validate_dimension_vs_shape(value, shape)

    @pytest.mark.parametrize("value, size, should_pass", [
        ([], 0, True),
        ([1], 1, True),
        (['a'], 1, True),
        ([0.5], 1, True),
        ([1, "abc", 0.5], 3, True),
        ([1, "abc", 0.5], 2, False),
        ([1, "abc", 0.5], 4, False),
        ([], 1, False),
        ([1], 0, False),
        ([1], 2, False),
        (0, 0, False),
        (0.5, 0, False),
        ((1,), 1, False),
    ])
    def test_validate_list_of_size(self, value, size, should_pass: bool):
        if should_pass:
            validate_list_of_size(value, size)
        else:
            with raises(FailedValidationException, match=f"Expected list of size {size}"):
                validate_list_of_size(value, size)

    # region test_validate_predicate

    class Params:
        class SubParams:
            c = 4

        a = 3
        b = 5
        sub_params = SubParams()

    def test_validate_predicate_valid(self):
        params = TestValidator.Params()
        validate_predicate(lambda: params.a < params.sub_params.c <= params.b < 17.4, "this error should never happen")

    def test_validate_predicate_invalid_user_message(self):
        params = self.Params()

        with raises(FailedValidationException, match="a 3 should be equal to c 4"):
            validate_predicate(lambda: params.a == params.sub_params.c <= params.b < 17.4,
                               f"a {params.a} should be equal to c {params.sub_params.c}")

    def test_validate_predicate_invalid_default_message(self):
        params = self.Params()

        with raises(FailedValidationException,
                    match=re.escape("assert params.a {3} == params.sub_params.c {4} <= params.b {5} < 17.4 {17.4}.")):
            validate_predicate(lambda: params.a == params.sub_params.c <= params.b < 17.4)

    def test_validate_predicate_invalid_tuple(self):
        params = self.Params()

        params.a = (1, 2, 3)
        params.b = ( 3,2,  1 )

        with raises(FailedValidationException,
                    match=re.escape("assert params.a {(1, 2, 3)} == params.sub_params.c {4} <= "
                                    "params.b {(3, 2, 1)} < (4,5,6,7,8) {(4, 5, 6, 7, 8)}.")):
            validate_predicate(lambda: params.a == params.sub_params.c <= params.b < (4, 5,6,   7, 8))

    def test_validate_predicate_invalid_list(self):
        params = self.Params()

        params.a = [1, 2, 3, [4, 5]]
        params.b = [   6  ,  [  [  7  ]  ]  ]

        with raises(FailedValidationException,
                    match=re.escape("assert params.a {[1, 2, 3, [4, 5]]} == params.sub_params.c {4} <= "
                                    "params.b {[6, [[7]]]} < [6,[[7]]] {[6, [[7]]]}.")):
            validate_predicate(lambda: params.a == params.sub_params.c <= params.b < [   6  ,  [  [  7  ]  ]  ])

    def test_validate_predicate_brackets(self):
        params = self.Params()

        with raises(FailedValidationException,
                    match=re.escape("assert ( ( params.a {3} ) - params.b {5} ) == params.sub_params.c {4}.")):
            validate_predicate(lambda: ((params.a) - params.b) == params.sub_params.c)

    def test_validate_predicate_additional_message_no_line_break(self):
        params = self.Params()

        with raises(FailedValidationException,
                    match=re.escape("assert params.a {3} == params.b {5}, some additional message.")):
            validate_predicate(lambda: params.a == params.b, additional_message=f"some additional message")

    def test_validate_predicate_additional_message_line_breaks(self):
        params = self.Params()

        with raises(FailedValidationException,
                    match=re.escape("assert params.a {3} == params.b {5}, "
                                    "some additional message containing params.sub_params.c 4.")):
            validate_predicate(lambda: params.a == params.b,
                               additional_message=f"some additional message containing params.sub_params.c"
                               f" {params.sub_params.c}")

    def test_validate_predicate_global_variable(self):
        local_variable = 6

        with raises(FailedValidationException,
                    match=re.escape("assert local_variable {6} == GLOBAL_VARIABLE {7}.")):
            validate_predicate(lambda: local_variable == GLOBAL_VARIABLE)

    # endregion
