import sys

from src.lib.data.set import Dataset


set = Dataset(sys.argv[1])

x = {}
for i in range(50):
    x[i] = 0

pbi = set.point_by_id()
for id in pbi:
    x[len(id)] += 1

print(x)
