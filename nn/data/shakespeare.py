import os
import nn

PREAMBLE_LEN = 10487
POSTAMBLE_LEN = 573

SPACE = " "
DOUBLE_SPACE = SPACE * 2

def get_data(trim=True, remove_double_spaces=True, lower_case=True):
    shakespeare_path = os.path.join(nn.data.CURRENT_DIR, "t8.shakespeare.txt")
    with open(shakespeare_path) as f:
        s = f.read()

    if trim:
        s = s[PREAMBLE_LEN:-POSTAMBLE_LEN]

    if remove_double_spaces:
        while DOUBLE_SPACE in s:
            s = s.replace(DOUBLE_SPACE, SPACE)

    if lower_case:
        s = s.lower()

    return s

def get_likely_chars(s):
    char_set = set(s)
    char_counts = {c: s.count(c) for c in char_set}
    most_unlikely_letter = min(
        (c for c in char_set if c.isalpha()),
        key=lambda c: char_counts[c]
    )
    likely_char_list = sorted(
        (
            c for c in char_set
            if char_counts[c] >= char_counts[most_unlikely_letter]
        ),
        key=lambda c: char_counts[c],
        reverse=True,
    )
    return likely_char_list
