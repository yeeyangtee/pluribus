from typing import Any, List


def rotate_list(l: List[Any], n: int):
    """Helper function for rotating lists, shifting n elements.

    Parameters
    ----------
    l : List[Any]
        List to rotate.
    n : int
        Integer index of where to rotate.
    """
    if n > len(l):
        raise ValueError
    return l[n:] + l[:n]

def rotate_list_once(l: List[Any]):
    """Helper function for rotating lists, simply puts the last element to the front.

    Parameters
    ----------
    l : List[Any]
        List to rotate.
    """
    return l[-1:] + l[:-1]
