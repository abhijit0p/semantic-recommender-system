import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib as joblib
# fake training data
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

X = []
y = []

for _ in range(300):
    similarity = random.uniform(0, 1)
    category_match = random.choice([0, 1])
    overlap = random.uniform(0, 1)


    #overlap alone can sometimes win
    #category is strong, but not absolute
    # similarity is a weaker signal on its own
    #      
    label = 1 if (
        category_match == 1 and (
            similarity > 0.6 or overlap > 0.6
        )
    ) else 0

    X.append([similarity, category_match, overlap])
    y.append(label)

model = LogisticRegression()
model.fit(X, y)

print("Learned weights:", model.coef_, model.intercept_)

joblib.dump(model, "model.pkl")