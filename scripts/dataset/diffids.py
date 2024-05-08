# Copy diff (id-level) of two datasets into new dataset
# Run from project root via python -m scripts.dataset.diff full existing_part new_part

import sys
from src.lib.data.set import Dataset


full_set = Dataset(sys.argv[1])
part1_set = Dataset(sys.argv[2])

new_ids = list(filter(lambda x: x not in list(part1_set.identities.keys()), full_set.identities.keys()))
part2_set = full_set.copy(only_ids=new_ids, newname=sys.argv[3])
