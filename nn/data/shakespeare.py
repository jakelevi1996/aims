import os
import nn

PREAMBLE_LEN = 10526
POSTAMBLE_LEN = 573

def get_data():
    shakespeare_path = os.path.join(nn.data.CURRENT_DIR, "t8.shakespeare.txt")
    with open(shakespeare_path) as f:
        s = f.read()

    return s[PREAMBLE_LEN:-POSTAMBLE_LEN]

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
