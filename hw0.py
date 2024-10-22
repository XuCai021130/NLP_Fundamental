from typing import Generator, Iterable
from collections import Counter

def sorted_chars(s: str) -> list[str]:
    char_set = {c for c in s}
    return sorted(char_set)



def gen_sentences(path: str) -> Generator[list[str], None, None]:
    with open (path, encoding = "UTF-8") as f:
        for line in f:
            line = line.strip("\n")
            if line: 
                result = line.split(" ")    
                yield result



def n_most_frequent_tokens(sentences: Iterable[list[str]], n: int) -> list[str]:
    if n < 0:
        raise ValueError("n should be a positive number")

    c = Counter()
    for sentence in sentences:
        c.update(sentence)

    return [item[0] for item in c.most_common(n)]



def case_sarcastically(text: str) -> str:
    res = ""
    count = 0
    for char in text:
        if char.upper() == char.lower():
            res += char
        else:
            count += 1
            if count %2 == 1:
                res += char.lower()
            else:
                res += char.upper()
    return res


