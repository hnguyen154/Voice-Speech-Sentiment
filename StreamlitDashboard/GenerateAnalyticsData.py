
import pandas as pd
import numpy as np
import random
random.seed(1234)
import datetime
from faker import Faker
num_calls = 150

df = pd.DataFrame({})
df["CustNo"] = np.random.randint(1, 9, num_calls)
df["CallNo"] = df.index

start_date = datetime.date(2020, 2, 1)
end_date = datetime.date(2020, 7, 1)

df["Date"] = np.random.choice(pd.date_range(start_date, end_date), len(df.CustNo))
df["Month"] = df["Date"].apply(lambda x: datetime.datetime.strftime(x, "%m"))

faker = Faker()
emotions = ['Positive', 'Neutral', 'Negative']

df["AudioSent"] = df.apply(lambda x: faker.words(1, emotions, True)[0], axis=1)
df["TextSent"] = df.apply(lambda x: faker.words(1, emotions, True)[0], axis=1)
df["TextSentNum"] = df.apply(lambda x: np.random.randn(), axis=1)

consignees = ['Urban Residential', 'Rural Residential', 'Commercial', 'Export']
df["Consignee"] = df.apply(lambda x: faker.words(1, consignees, True)[0], axis=1)

products = ['1DA','2DA','Ground']
df["Product"] = df.apply(lambda x: faker.words(1, products, True)[0], axis=1)

df["longitude"] = df.apply(lambda x: faker.location_on_land()[1], axis=1)
df["latitude"] = df.apply(lambda x: faker.location_on_land()[0], axis=1)

df.head()
df.to_csv(r'C:\Users\hongh\Documents\GitHub\Speech-Sentiment-Analysis-\StreamlitDashboard\input\data\AnalyticsData.csv')
transcripts = pd.read_csv(r'C:\Users\hongh\Documents\GitHub\Speech-Sentiment-Analysis-\StreamlitDashboard\input\data\text.csv')

out = pd.merge(df, transcripts, on=df.index)

out.head()
out.to_csv(r'C:\Users\hongh\Documents\GitHub\Speech-Sentiment-Analysis-\StreamlitDashboard\input\data\AnalyticsData.csv')
