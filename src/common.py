
def ilog2(n):
    '''
    >>> ilog2(1.)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(-1)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(0)
    Traceback (most recent call last):
    AssertionError
    >>> ilog2(1)
    0
    >>> ilog2(2)
    1
    >>> ilog2(4)
    2
    >>> ilog2(8)
    3
    '''

    assert isinstance(n, int)
    assert n > 0
    assert n & (n - 1) == 0

    return n.bit_length() - 1
