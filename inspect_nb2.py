import json
nb = json.load(open('notebooks/ab_analysis.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    print('cell', i, 'type', c['cell_type'], 'source length', len(c['source']))
    if len(c['source']) > 0:
        print(' first line:', c['source'][0].strip())
    else:
        print(' <empty>')
