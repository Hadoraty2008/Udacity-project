import pandas as pd
import numpy as np
import requests
import os
import json
twitter_archive = pd.read_csv('twitter_archive_enhanced.csv')
# رابط الملف (حسب موقع Udacity)
image_predictions_url = 'https://video.udacity-data.com/topher/2018/November/5bf52cc9_image-predictions/image-predictions.tsv'

# تحميل الملف
response = requests.get(image_predictions_url)
with open('image_predictions.tsv', mode='wb') as file:
    file.write(response.content)

# قراءة الملف
image_predictions = pd.read_csv('image_predictions.tsv', sep='\t')
# فتح الملف وقراءة التغريدات كسجلات JSON
tweets_data = []
with open('tweet_json.txt', encoding='utf-8') as file:
    for line in file:
        try:
            tweet = json.loads(line)
            tweets_data.append({
                'tweet_id': tweet['id'],
                'retweet_count': tweet['retweet_count'],
                'favorite_count': tweet['favorite_count']
            })
        except:
            continue

# تحويل البيانات إلى DataFrame
tweets_df = pd.DataFrame(tweets_data)
# معاينة أول 5 صفوف لكل DataFrame
print(twitter_archive.head())
print(image_predictions.head())
print(tweets_df.head())

# التقييم البصري (Visual Assessment)
# -----------------------

# عرض أول 5 صفوف من كل جدول
twitter_archive.head()
image_predictions.head()
tweets_df.head()

# -----------------------
# التقييم البرمجي (Programmatic Assessment)
# -----------------------

# معلومات عامة عن الجداول الثلاثة
twitter_archive.info()
image_predictions.info()
tweets_df.info()

# القيم الفارغة
twitter_archive.isnull().sum()
image_predictions.isnull().sum()
tweets_df.isnull().sum()

# الإحصائيات العامة
twitter_archive.describe()
image_predictions.describe()
tweets_df.describe()

# هل هناك تغريدات مع إعادة تغريد؟
twitter_archive.retweeted_status_id.notnull().sum()

# التغريدات التي ليس لديها صور
twitter_archive['expanded_urls'].isnull().sum()

# أنواع الأعمدة
twitter_archive.dtypes
image_predictions.dtypes
tweets_df.dtypes

# القيم الفريدة لبعض الأعمدة المهمة
twitter_archive['source'].value_counts()
twitter_archive['rating_numerator'].value_counts()
twitter_archive['rating_denominator'].value_counts()
twitter_archive['name'].value_counts()

# البحث عن قيم غير منطقية في الاسم (مثل "a", "an", "the", "None")
twitter_archive[twitter_archive['name'].str.match(r'^[a-z]+$') == True]['name'].value_counts()

# التأكد من عدم وجود تكرار في معرفات التغريدات
twitter_archive['tweet_id'].duplicated().sum()
image_predictions['tweet_id'].duplicated().sum()
tweets_df['tweet_id'].duplicated().sum()

# إنشاء نسخ من البيانات الأصلية
twitter_archive_clean = twitter_archive.copy()
image_predictions_clean = image_predictions.copy()
tweets_df_clean = tweets_df.copy()

twitter_archive_clean = twitter_archive_clean[twitter_archive_clean['retweeted_status_id'].isnull()]
twitter_archive_clean['retweeted_status_id'].notnull().sum()  # يجب أن يكون 0

twitter_archive_clean = twitter_archive_clean[twitter_archive_clean['expanded_urls'].notnull()]
twitter_archive_clean['expanded_urls'].isnull().sum()  # يجب أن يكون 0

import numpy as np

twitter_archive_clean['name'] = twitter_archive_clean['name'].replace(r'^[a-z]+$', np.nan, regex=True)
twitter_archive_clean['name'].value_counts().head(10)

twitter_archive_clean = twitter_archive_clean[twitter_archive_clean['rating_denominator'] == 10]
twitter_archive_clean['rating_denominator'].value_counts()

import re

twitter_archive_clean['source'] = twitter_archive_clean['source'].apply(lambda x: re.findall(r'>(.*?)<', x)[0])
twitter_archive_clean['source'].value_counts()

columns_to_drop = ['in_reply_to_status_id', 'in_reply_to_user_id',
                   'retweeted_status_id', 'retweeted_status_user_id',
                   'retweeted_status_timestamp']
twitter_archive_clean.drop(columns=columns_to_drop, inplace=True)

twitter_archive_clean.info()

df_merged = twitter_archive_clean.merge(image_predictions_clean, on='tweet_id', how='inner')
df_merged = df_merged.merge(tweets_df_clean, on='tweet_id', how='inner')
df_merged.info()

df_merged = df_merged.rename(columns={'p1': 'prediction', 'p1_conf': 'confidence', 'p1_dog': 'is_dog'})
df_merged = df_merged[['tweet_id', 'timestamp', 'source', 'text', 'rating_numerator',
                       'rating_denominator', 'name', 'prediction', 'confidence',
                       'is_dog', 'favorite_count', 'retweet_count', 'expanded_urls']]

df_merged.sample(5)

# حفظ البيانات المُنظّفة في ملف CSV
twitter_archive_master.to_csv('twitter_archive_master.csv', index=False)

import sqlite3

# إنشاء اتصال بقاعدة البيانات (أو إنشاء قاعدة جديدة إذا لم تكن موجودة)
conn = sqlite3.connect('twitter_archive_master.db')

# تخزين البيانات في جدول داخل قاعدة البيانات
twitter_archive_master.to_sql('twitter_archive_master', conn, index=False, if_exists='replace')

# إغلاق الاتصال بقاعدة البيانات
conn.close()

# تحليل التقييمات
top_ratings = twitter_archive_master['rating_numerator'].value_counts().sort_values(ascending=False)

# عرض أكثر 5 تقييمات شيوعًا
top_ratings.head()

import seaborn as sns
import matplotlib.pyplot as plt

# رسم العلاقة بين التقييم وعدد الإعجابات
plt.figure(figsize=(10, 6))
sns.regplot(x='rating_numerator', y='favorite_count', data=twitter_archive_master, scatter_kws={'alpha':0.4})
plt.title('العلاقة بين التقييم وعدد الإعجابات')
plt.xlabel('Rating')
plt.ylabel('Favorites')
plt.show()

# استبعاد أسماء الكلاب غير المعروفة أو الأخطاء مثل 'a', 'an', 'the'
clean_names = twitter_archive_master[~twitter_archive_master['name'].isin(['a', 'an', 'the', 'None'])]
top_dog_names = clean_names['name'].value_counts().head(10)

# عرض الأسماء الأكثر تكرارًا
top_dog_names

# تحليل: أي فصائل الكلاب حصلت على أكبر عدد من الإعجابات؟
top_breeds = twitter_archive_master.groupby('p1')['favorite_count'].mean().sort_values(ascending=False).head(10)

# رسم بياني
top_breeds.plot(kind='barh', figsize=(10,6), color='skyblue')
plt.title('أكثر سلالات الكلاب تفضيلًا (حسب متوسط الإعجابات)')
plt.xlabel('متوسط عدد الإعجابات')
plt.gca().invert_yaxis()
plt.show()

