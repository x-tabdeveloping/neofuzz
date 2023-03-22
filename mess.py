from fuzz_lightyear.search import create_matcher

with open("countries.txt") as in_file:
    options = [line.strip() for line in in_file]

match_country = create_matcher(options, ngram_range=(1, 3))

match_country("colu")
