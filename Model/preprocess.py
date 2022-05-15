
import pandas as pd

tmp = pd.read_csv("Chatbot.csv")
tmp = tmp.drop(['Unnamed: 0'], axis=1)
tmp.to_csv("chatbot_preprocessed.csv", index=False)
