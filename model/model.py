import pandas as pd 
import numpy as np 
import torch
df_test = pd.read_csv('./test.csv')
print(df_test.head())
#### load the model and build the detector for deployment
# !pip install transformers
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
input_dir = './../model_files/'

loaded_model = BertForSequenceClassification.from_pretrained(input_dir)
loaded_model.eval()
loaded_tokenizer = BertTokenizer.from_pretrained(input_dir)
loaded_df_label = pd.read_pickle('./../model_files/df_label.pkl')

# test the model on an unseen example
# y_pred = []
def intent_detector(intent):

  pt_batch = loaded_tokenizer(
  intent,
  padding=True,
  truncation=True,
  return_tensors="pt")

  pt_outputs = loaded_model(**pt_batch)
  __, id = torch.max(pt_outputs[0], dim=1)
  prediction = loaded_df_label.iloc[[id.item()]]['intent'].item()
  # y_pred.append(prediction)
  # print(prediction)
  # print('You may have a medical condition: %s. Would you like me to transfer your call to your doctor?'%(prediction))
  return prediction
# y_test = df_test['prompt']
# for line in df_test['phrase']:
#   # print(line)
#   y_pred.append(intent_detector(line))
# # print(len(y_pred))
# # print(len (y_test))
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(30, 30))
# sns.heatmap(cm,annot=True, fmt='g')
# plt.xlabel('Prediction')
# plt.ylabel('Label')
# plt.show()
# input = "Where is germany ?"
# intent_detector(input)


