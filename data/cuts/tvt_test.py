import random

begin = 1372636853
end = 1404172787

random.seed(42)
cuts = []
for i in range(500):
    cuts.append(random.randrange(begin, end))
