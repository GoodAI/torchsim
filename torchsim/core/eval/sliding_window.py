from itertools import islice


def sliding_window(seq, n=2):
    """Returns a sliding window (of width n) over data."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
