import json
nb = json.load(open('notebooks/ab_analysis.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    print('cell', i, 'type', c['cell_type'])
    if i >= 5:
        print('--- content ---')
        print(''.join(c['source']))
        print('--------------')
