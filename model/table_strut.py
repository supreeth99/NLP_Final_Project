import json
from prettytable import PrettyTable
from pathlib import Path
import numpy as np
import pandas as pd


p = Path(r'data.json')

with p.open('r', encoding='utf-8') as f:
    data = json.loads(f.read())
print(data.keys())
df = pd.json_normalize(data)
df.to_csv('structure.csv', index=False, encoding='utf-8')