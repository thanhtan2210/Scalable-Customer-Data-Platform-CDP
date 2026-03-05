import json
nb = json.load(open('notebooks/ab_analysis.ipynb', encoding='utf-8'))
print('total cells', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    print('-' * 40)
    print('cell', i+1, 'type', c['cell_type'])
    for j, line in enumerate(c['source']):
        print(f'{j+1:02d}: {repr(line)}')
