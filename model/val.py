import pandas as pd 
from model import *
from sklearn.metrics import classification_report
df = pd.read_csv('./test.csv')
phrases = [x for x in df['phrase']]
prompts = [x for x in df['prompt']]
print(len(phrases))
pred = []
for i, data in enumerate(phrases):

    pred.append(intent_detector(data))
    print('File : ', i)
# print(classification_report(prompts, pred))
with open('results.txt', 'w') as f:
    f.write(classification_report(prompts, pred))
    # print(data)
