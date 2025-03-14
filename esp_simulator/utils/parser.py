import re
from typing import List


def parse_str2intlist(s: str) -> List[int]:
    """Parse string into integer list

    Parameters
    ----------
    s : str
        string that contains integer and ranges, e.g. "1,2,3,4-5"

    Returns
    -------
    List[int]
        List of the numbers
    """
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        p = "".join(p.split())
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif p == "":
            continue
        else:
            ranges.append(int(p))
    return ranges
