import math
# Искусственные ранги в полностью перечисленном пространстве.
space = 1024
for rank in (1, 8, 128, 1024):
    exposure = math.log2(space) - math.log2(rank)
    print(rank, exposure)
assert math.log2(space) - math.log2(8) == 7
