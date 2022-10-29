from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import random

calname_for_df = ["FIO","sex","random"]
df = pd.read_csv("x_train_g.csv", error_bad_lines=False, engine = "python", names = calname_for_df)
calname_for_df1 = ["Gender"]
df1 = pd.read_csv("y_train_g.csv", error_bad_lines=False, engine = "python", names = calname_for_df1)
df = df.replace(to_replace ='\t', value = ' ', regex = True)
df['gender'] = df1['Gender']

df['gender'] = df['gender'].replace(to_replace = "M", value = "1", regex = True)
df['gender'] = df['gender'].replace(to_replace = "F", value = "0", regex = True)

for i in range(len(df['random'])):
    df['random'][i] = random.uniform(0,1)

df['FIO'] = df['FIO'].str.lower()
Train = df[df['random'] >= 0.1]
Test = df[df['random'] < 0.1]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(Train['FIO'])
y = Train['gender']

clf = LogisticRegression(random_state=0).fit(X, y)
y_predict = clf.predict(vectorizer.transform(Test['FIO']))

print(classification_report(Test['gender'], y_predict))

Your_FIO = "Старунова Ольга Александоровна"
if clf.predict(vectorizer.transform([Your_FIO]))[0] == '1':
    print("Мужчина")
else:
    print("Женщина")