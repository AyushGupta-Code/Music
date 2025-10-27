# Twitter-roBERTa-base

This is a roBERTa-base model trained on ~58M tweets and finetuned for the emotion prediction task at Semeval 2018. 
For full description: [_TweetEval_ benchmark (Findings of EMNLP 2020)](https://arxiv.org/pdf/2010.12421.pdf). 
To evaluate this and other models on Twitter-specific data, please refer to the [Tweeteval official repository](https://github.com/cardiffnlp/tweeteval).

## Example of classification

```python
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    spamreader = csv.reader(html[:-1], delimiter='\t')
labels = [row[1] for row in spamreader]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

text = "Good night 😊"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# text = "Good night 😊"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

```

Output: 

```
1) 😘 0.2637
2) ❤️ 0.1952
3) 💕 0.1171
4) ✨ 0.0927
5) 😊 0.0756
6) 💜 0.046
7) 💙 0.0444
8) 😍 0.0272
9) 😉 0.0228
10) 😎 0.0198
11) 😜 0.0166
12) 😂 0.0132
13) 😁 0.0131
14) ☀ 0.0112
15) 🎄 0.009
16) 💯 0.009
17) 🔥 0.008
18) 📷 0.0057
19) 🇺🇸 0.005
20) 📸 0.0048
```
