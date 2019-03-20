import inspect
import re
from typing import Union, Optional, List, Callable

from torchsim.core.exceptions import FailedValidationException


def validate_positive_with_zero_int(value: int):
    if type(value) is not int or value < 0:
        raise FailedValidationException('Positive int or zero expected')


def validate_positive_int(value: int):
    if type(value) is not int or value <= 0:
        raise FailedValidationException('Strictly positive int expected')


def validate_float_in_range(value: float, min_range: float, max_range: float):
    if type(value) is not float or value < min_range or value > max_range:
        raise FailedValidationException(f'Float value {value} expected to be in range <{min_range},{max_range}>')


def validate_positive_float(value: Union[float, int]):
    if not (type(value) is float) or value <= 0:
        raise FailedValidationException("Strictly positive float expected")


def validate_positive_with_zero_float(value: Union[float, int]):
    if not (type(value) is float) or value < 0:
        raise FailedValidationException("Strictly positive float expected")


def validate_positive_optional_int(value: Optional[int]):
    if value is not None and (type(value) is not int or value <= 0):
        raise FailedValidationException('Strictly positive int or None expected')


def validate_list_str(value: List[str]):
    if type(value) is not list \
            or any({type(item) is not str for item in value}):
        raise FailedValidationException('List[str] object expected (e.g. [\'True\',\'False\'])')


def validate_list_list_int(value: List[List[int]]):
    if type(value) is not list \
            or any({type(item) is not list for item in value}) \
            or any({type(item) is not int for sub_list in value for item in sub_list}):
        raise FailedValidationException('List[List[int]] object expected (e.g. [[1,2],[3]])')


def validate_list_list_float_or_int(value: List[List[Union[float, int]]]):
    if type(value) is not list \
            or any({type(item) is not list for item in value}) \
            or any({not (type(item) is float or type(item) is int) for sub_list in value for item in sub_list}):
        raise FailedValidationException('List[List[float]] object expected (e.g. [[1,2],[3]])')


def validate_dimension_vs_shape(value: Optional[int], shape: List[int]):
    if value is not None and len(shape) > 0 and (type(value) is not int or value < -1 or value > len(shape)):
        raise FailedValidationException(f"Expected dimension to be -1 <= x <= {len(shape)}")


def validate_list_of_size(value: List[int], size: int):
    if type(value) is not list or len(value) != size:
        raise FailedValidationException(f"Expected list of size {size} but {value} was received")


def validate_predicate(predicate: Callable[[], bool], error_message: Optional[str] = None,
                       additional_message: Optional[str] = None):
    """Checks if predicate is True. If not, raises a FailedValidationException with either the specified message or an
    automatically generated message describing the problem. Optionally, one can add an additional message after the
    automatically generated one.

    It should be able to handle local and global variables, tuples and lists. But it does not handle function calls."""

    if error_message is None:
        _validate_predicate_automatic_message(predicate, additional_message)
    else:
        if not predicate():
            raise FailedValidationException(error_message)


def _validate_predicate_automatic_message(predicate: Callable[[], bool], additional_message: Optional[str] = None):
    """This method (or any method calling it) needs to have predicate as the last param,
    otherwise inspect.getsource does not work well."""

    if not predicate():
        # get the lambda assertion as str
        source = inspect.getsource(predicate).strip()
        #get rid of the additional message
        if additional_message is not None:
            source = re.sub(r',\s*additional_message(.|\s)*', '', source)
        else:
            # get rid of the ending bracket
            source = source[:-1]
        # get rid of the lambda
        lambda_pos = source.find('lambda')
        source = source[lambda_pos+8:]


        # split to individual expression and try to evaluate each

        # get rid of spaces in tuples and lists
        source = re.sub(r'\s*,\s*', ',', source)

        # get rid of spaces near brackets
        source = re.sub(r'\s*\)', ')', source)
        source = re.sub(r'\(\s*', '(', source)

        source = re.sub(r'\s*]', ']', source)
        source = re.sub(r'\[\s*', '[', source)

        # do not break lines
        source = source.replace('\n', ' ')

        chunks = re.split(r' ', source)
        message = ''
        for chunk in chunks:
            # remove beginning and ending brackets if the chunk does not contain commas
            if chunk.find(',') == -1:
                subchunks = re.split(r'([()])', chunk)
            else:
                subchunks = [chunk]

            for subchunk in subchunks:
                if subchunk == '':
                    continue

                variables = {**inspect.getclosurevars(predicate).nonlocals, **inspect.getclosurevars(predicate).globals}
                # noinspection PyBroadException
                try:
                    value = eval(subchunk, variables)
                    message += f"{subchunk} {{{value}}} "
                except Exception:
                    message += f"{subchunk} "

        message = message[:-1]

        if additional_message is not None:
            message += ', ' + additional_message

        message += '.'

        raise FailedValidationException(f"assert {message}")

