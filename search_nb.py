import json
import re
nb = json.load(open('notebooks/ab_analysis.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        for j, line in enumerate(c['source']):
            if 'print(f' in line:
                print('cell', i+1, 'line', j+1, repr(line))
