'''
Example script of the API usage.
It shows a simple use case:
1. split your data as you like
2. call the API for this chain of actions: 
train -> evaluate -> save model -> load model -> classify -> (*bonus) classify single text

Obviously, it can be altered for any particular use case.

This script was also used for the API's sanity check (along with another script which cannot be shown
due to the fact that even its author can barely read it now; testing on the fly can be really messy).
'''

from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import MLClassifier as mlc

data = pd.read_csv("nlp_eng_task_train.csv")
print(data.shape)
#(1995, 28)
seen_data = data[:1900] #leave the last 95 to be unseen for future classification testing
train, test = train_test_split(seen_data, test_size=0.2, random_state=42)

classifier_api = mlc.MLClassifierAPI()

#call with train data only
model = classifier_api.train(train)

#evaluate the model
scores = classifier_api.evaluate(model, test)
print(scores)
#{'hamming loss': 0.007518796992481203, 'hamming score': 0.8425438596491227, 'accuracy': 0.8}

#you like the scores, save this model for future use with the current date/time
model_tosave = "model_" + datetime.today().strftime('%Y-%m-%d_%H:%M')
classifier_api.saveModel(name=model_tosave)

#you can exit now and load the same classifier later
classifier_api.loadModel(model_tosave)

#get unseen data
unseen_data = data[-80:] #leave 15 rows in between seen and unseen to hang out to dry :)

#classify the unseen data, ask to get text labels, not binarized vectors
#you can pass in the model or use None - then the same in-memory model will be used
predicted_labels = classifier_api.classify(None,unseen_data,textlabels=True)

#see the ground truth vs predicted labels with your eyes
df = pd.DataFrame({"true":unseen_data["Solution Type"], "predicted":predicted_labels})

df.to_csv("predictions.csv")
#print(df)
#looks good, though can be formatted for readability

#classify single text
single_text = str(data.iloc[-87]["Description"]) #lucky one
label = classifier_api.classify_single_text(None,single_text)
print("Label: {}\n Text: {}".format(label, single_text))

#Label: [('General Rice',)]

#Text: Top-of-stove directions: 1. In medium saucepan, bring 2 cups water,
# 1 tbsp. margarine (optional) and contents of package to a boil.
# (Try using I Can't believe it's not butter! spread or Country crock spread.)
# 2. Stir. Reduce heat & simmer covered 7 minutes or until rice is tender.
# 3. Let stand at least 2 minutes. Stir and serve. Microwave directions:
# In 2-quart microwave-safe bowl, combine 2-1/4 cups water, 1 tbsp. margarine(optional)
# and contents of package. Microwave uncovered at high about 12 minutes* or until rice is tender,
#  stirring once halfway through. Stir and serve. *Microwave ovens vary; adju




