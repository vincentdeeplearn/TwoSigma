
# coding: utf-8

# In[1]:

## kaggle Two Sigma Final subbmission

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import datetime
import gc
import time
import warnings
from itertools import chain

import lightgbm as lgb
import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


warnings.filterwarnings("ignore")

# Initialize the environment
if "env" not in globals():
    env = twosigmanews.make_env()
# Retrieve the data
mkt_train, news_train = env.get_training_data()

mkt_train["returnsClosePrevMktres1"].fillna(mkt_train["returnsClosePrevRaw1"], inplace=True)
mkt_train["returnsOpenPrevMktres1"].fillna(mkt_train["returnsOpenPrevRaw1"], inplace=True)
mkt_train["returnsClosePrevMktres10"].fillna(mkt_train["returnsClosePrevRaw10"], inplace=True)
mkt_train["returnsOpenPrevMktres10"].fillna(mkt_train["returnsOpenPrevRaw10"], inplace=True)
print("Missing values filled.")

log_ret = np.log(mkt_train["close"].values / mkt_train["open"].values)
outlier_idx = ((log_ret > 0.5).astype(int) + (log_ret < -0.5).astype(int)).astype(bool)
mkt_train = mkt_train.loc[~outlier_idx, :]

mkt_train = mkt_train.loc[mkt_train["assetName"] != "Unknown", :]

short_ret_cols = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1']
long_ret_cols = ['returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']

for col in short_ret_cols:
    mkt_train = mkt_train.loc[mkt_train[col].abs() < 1]

for col in long_ret_cols:
    mkt_train = mkt_train.loc[mkt_train[col].abs() < 2]

mkt_train = mkt_train.loc[mkt_train["time"].dt.date > datetime.date(2009, 1, 1)]
print("Outliers removed.")

del log_ret
del outlier_idx
gc.collect()

mkt_train["time"] = mkt_train["time"].dt.date
mkt_train.rename(columns={"time": "date"}, inplace=True)
mkt_train["returnsToday"] = np.log(mkt_train["close"].values / mkt_train["open"].values)
mkt_train["relVol"] = mkt_train.groupby(["date"])["volume"].transform(lambda x: (x - x.mean()) / x.std())
# mkt_train["raw1_10"] = (mkt_train["returnsClosePrevRaw1"] - mkt_train["returnsClosePrevRaw10"] +
#                         mkt_train["returnsOpenPrevRaw1"] - mkt_train["returnsOpenPrevRaw10"]) / 2
# mkt_train["res1_10"] = (mkt_train["returnsClosePrevMktres1"] - mkt_train["returnsClosePrevMktres10"] +
#                         mkt_train["returnsOpenPrevMktres1"] - mkt_train["returnsOpenPrevMktres10"]) / 2

news_train["sourceTimestamp"] = news_train["sourceTimestamp"].dt.date
news_train.rename(columns={"sourceTimestamp": "date"}, inplace=True)
news_train["rel1stMentionPos"] = news_train["firstMentionSentence"].values / news_train["sentenceCount"].values
news_train["relSentimentWord"] = news_train["sentimentWordCount"].values / news_train["wordCount"].values
news_train["relSentCnt"] = news_train.groupby(["date"])["sentenceCount"].transform(lambda x: (x - x.mean()) / x.std())
news_train["relWordCnt"] = news_train.groupby(["date"])["wordCount"].transform(lambda x: (x - x.mean()) / x.std())
news_train["relBodySize"] = news_train.groupby(["date"])["bodySize"].transform(lambda x: (x - x.mean()) / x.std())

# asset_codes = list(chain.from_iterable([list(eval(x)) for x in news_train["assetCodes"].values]))
# news_idx = list(chain.from_iterable([[i] * len(list(eval(news_train["assetCodes"][i]))) for i in range(news_train.shape[0])]))
asset_codes = []
news_idx = []
for i, value in news_train["assetCodes"].iteritems():
    
    asset_codes.extend(list(eval(value)))
    news_idx.extend([i] * len(eval(value)))
    
asset_codes = pd.DataFrame({"assetCode": asset_codes, "newsIndex": news_idx})
news_train["newsIndex"] = news_train.index
news_train = news_train.merge(asset_codes, how="left", on="newsIndex")
news_train.drop(["assetCodes", "newsIndex"], axis=1, inplace=True)
del asset_codes
del news_idx
print("New features added.")

mkt_train.drop(["assetName", "volume", "close", "open"], axis=1, inplace=True)

news_train.drop(["time", "sourceId", "headline", "provider", "subjects",
                 "audiences", "bodySize", "sentenceCount", "wordCount",
                 "assetName", "firstMentionSentence", "sentimentWordCount",
                 "headlineTag"], axis=1, inplace=True)
print("Useless features dropped.")
gc.collect()

news_train = news_train.groupby(["date", "assetCode"], as_index=False).mean()

data_train = pd.merge(mkt_train, news_train, how="left", left_on=["date", "assetCode"], right_on=["date", "assetCode"])
print("Data merged.")

del mkt_train
del news_train
gc.collect()

fillna_dict = {}

for col in data_train.columns:
    
    if col != "sentimentNeutral":
        fillna_dict[col] = 0
    else:
        fillna_dict[col] = 1
data_train.fillna(value=fillna_dict, inplace=True)
print("Missing values filled.")

feature_cols = [col for col in data_train.columns.values if col not in
                ["date", "assetCode", "returnsOpenNextMktres10", "universe"]]
                
feature_scalers = [StandardScaler() for i in range(len(feature_cols))]

for i in range(len(feature_cols)):
    data_train[feature_cols[i]] = feature_scalers[i].fit_transform(data_train[feature_cols[i]].values.reshape((-1, 1)))
    gc.collect()
print("Data normalized.")

data_train["y"] = (data_train["returnsOpenNextMktres10"] > 0).astype(int)

for col in data_train.select_dtypes(include="float64").columns:
    data_train[col] = data_train[col].astype("float16")

seed = np.random.randint(1, 100)
data_train, data_test = train_test_split(data_train, random_state=seed, test_size=0.2)
data_val, data_test = train_test_split(data_test, random_state=seed, test_size=0.5)
print("Data splitted.")

class LGBModel(lgb.LGBMClassifier):
    
    def evaluate(self, y_true, y_pred):
        
        y_true = y_true.astype(int).reshape((-1, 1))
        y_pred = y_pred.astype(int).reshape((-1, 1))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        
        print("Accuracy: {:.4f}.".format(accuracy))
        print("Precision: {:.4f}.".format(precision))
        print("Recall: {:.4f}.".format(recall))
        print("F1 score: {:.4f}".format(f1_score))
        
        return accuracy, precision, recall, f1_score

    def get_confidence(self, x_test, ref_scaler):
        
        pred_prob = self.predict_proba(x_test)
        confidence = (pred_prob[:, 1] - pred_prob[:, 0]).reshape((-1, 1))
        conf_scaler = StandardScaler()
        confidence = conf_scaler.fit_transform(confidence)
        confidence = ref_scaler.inverse_transform(confidence)
        confidence = np.clip(confidence, -0.99999, 0.99999)
        
        return confidence.flatten()
    
    def score(self, x_test, data_test, ref_scaler):
        
        confidence = self.get_confidence(x_test, ref_scaler)
        
        y_ret_pred = np.zeros(confidence.shape[0])
        for i in range(y_ret_pred.shape[0]):
            y_ret_pred[i] = confidence[i] * data_test["returnsOpenNextMktres10"].values[i] * data_test["universe"].values[i]
        pred_data = pd.DataFrame({"date": data_test["date"], "y_ret_pred": y_ret_pred})
        pred_data = pred_data.groupby(["date"])["y_ret_pred"].sum().values.flatten()
        score = np.mean(pred_data) / np.std(pred_data)
        print("Validation score: {:.4f}.".format(score))
        
        return score, confidence

seed = np.random.randint(1, 100)
lgbm_params = {
    "max_depth": 8,
    "num_leaves": 1000,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "boosting_type": "dart",
    "n_jobs": -1,
    "reg_lambda": 0.01,
    "random_state": seed
}

model = LGBModel(**lgbm_params)

x_train = data_train[feature_cols].values
y_train = data_train["y"].values
seed = np.random.randint(1, 100)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=seed, test_size=0.1)

iforest = IsolationForest(n_estimators=50, contamination=0.05, random_state=seed)
inlier_idx = iforest.fit_predict(x_train) == 1
x_train = x_train[inlier_idx, :]
y_train = y_train[inlier_idx]

start = time.clock()
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=20)
time_elapsed = int(time.clock() - start)
print("Total traninig time {} seconds.".format(time_elapsed))
del x_train
del y_train
del data_val
del x_val
del y_val

x_test = data_test[feature_cols].values
y_test = data_test["y"].values
y_pred = model.predict(x_test)
model.evaluate(y_test, y_pred)

ref_scaler = StandardScaler()
ref_scaler.fit(data_train["returnsOpenNextMktres10"].values.reshape((-1, 1)))
_, confidence = model.score(x_test, data_test, ref_scaler)
del data_train
del x_test
del y_test

plt.hist(confidence, bins="auto", label="Confidence")
plt.hist(data_test["returnsOpenNextMktres10"], bins="auto", alpha=0.8, label="True return")
plt.title("Confidence & True Return")
plt.legend(loc='best')
plt.xlim(-1,1)
plt.show()

del data_test
del y_pred
del confidence

feature_importance = pd.DataFrame()
feature_importance["feature"] = feature_cols
feature_importance["importance"] = model.feature_importances_
feature_importance.sort_values(by=["importance"], ascending=False, inplace=True)
feature_importance.reset_index(inplace=True, drop=True)

plt.barh(-np.arange(10), feature_importance.values[:10, 1])
plt.yticks(-np.arange(10), feature_importance.values[:10, 0])
plt.xlabel("importance")
plt.tight_layout()
plt.show()

def prepare_data(mkt, news, scalers):
    
    mkt["returnsClosePrevMktres1"].fillna(mkt["returnsClosePrevRaw1"], inplace=True)
    mkt["returnsOpenPrevMktres1"].fillna(mkt["returnsOpenPrevRaw1"], inplace=True)
    mkt["returnsClosePrevMktres10"].fillna(mkt["returnsClosePrevRaw10"], inplace=True)
    mkt["returnsOpenPrevMktres10"].fillna(mkt["returnsOpenPrevRaw10"], inplace=True)
    gc.collect()
    
    mkt["time"] = mkt["time"].dt.date
    mkt.rename(columns={"time": "date"}, inplace=True)
    mkt["returnsToday"] = np.log(mkt["close"].values / mkt["open"].values)
    mkt["relVol"] = mkt.groupby(["date"])["volume"].transform(lambda x: (x - x.mean()) / x.std())
    # mkt["raw1_10"] = (mkt["returnsClosePrevRaw1"] - mkt["returnsClosePrevRaw10"] +
    #                         mkt["returnsOpenPrevRaw1"] - mkt["returnsOpenPrevRaw10"]) / 2
    # mkt["res1_10"] = (mkt["returnsClosePrevMktres1"] - mkt["returnsClosePrevMktres10"] +
    #                         mkt["returnsOpenPrevMktres1"] - mkt["returnsOpenPrevMktres10"]) / 2
    gc.collect()
    
    news["sourceTimestamp"] = news["sourceTimestamp"].dt.date
    news.rename(columns={"sourceTimestamp": "date"}, inplace=True)
    news["rel1stMentionPos"] = news["firstMentionSentence"].values / news["sentenceCount"].values
    news["relSentimentWord"] = news["sentimentWordCount"].values / news["wordCount"].values
    news["relSentCnt"] = news.groupby(["date"])["sentenceCount"].transform(lambda x: (x - x.mean()) / x.std())
    news["relWordCnt"] = news.groupby(["date"])["wordCount"].transform(lambda x: (x - x.mean()) / x.std())
    news["relBodySize"] = news.groupby(["date"])["bodySize"].transform(lambda x: (x - x.mean()) / x.std())
    
    asset_codes = []
    news_idx = []
    for i, value in news["assetCodes"].iteritems():
        
        asset_codes.extend(list(eval(value)))
        news_idx.extend([i] * len(eval(value)))
    
    asset_codes = pd.DataFrame({"assetCode": asset_codes, "newsIndex": news_idx})
    news["newsIndex"] = news.index
    news = news.merge(asset_codes, how="left", on="newsIndex")
    news.drop(["newsIndex", "assetCodes"], axis=1, inplace=True)
    del asset_codes
    del news_idx
    
    mkt.drop(["assetName", "volume", "close", "open"], axis=1, inplace=True)

    news.drop(["time", "sourceId", "headline", "provider", "subjects",
               "audiences", "bodySize", "sentenceCount", "wordCount",
               "assetName", "firstMentionSentence", "sentimentWordCount",
               "headlineTag"], axis=1, inplace=True)
              
    gc.collect()
    
    news = news.groupby(["date", "assetCode"], as_index=False).mean()
    
    data = pd.merge(mkt, news, how="left", left_on=["date", "assetCode"], right_on=["date", "assetCode"])
    del mkt
    del news
    gc.collect()
    
    feature_cols = [col for col in data.columns.values if col not in
                    ["date", "assetCode", "returnsOpenNextMktres10", "universe", "y"]]
    
    fillna_dict = {}

    for col in feature_cols:

        if col != "sentimentNeutral":
            fillna_dict[col] = 0
        else:
            fillna_dict[col] = 1

    data.fillna(value=fillna_dict, inplace=True)
    
    for i in range(len(feature_cols)):
        data[feature_cols[i]] = feature_scalers[i].transform(data[feature_cols[i]].values.reshape((-1, 1)))
    
    gc.collect()

    return data

# Submission with the single LGBM model
if "days" not in globals():
    days = env.get_prediction_days()

fillna_dict = {}

for col in feature_cols:

    if col != "sentimentNeutral":
        fillna_dict[col] = 0
    else:
        fillna_dict[col] = 1
        
day_idx = 1

for (mkt, news, pred) in days:
    
    start = time.clock()
    data = prepare_data(mkt, news, feature_scalers)
    x_test = data[feature_cols].values
    confidence = model.get_confidence(x_test, ref_scaler)
    confidence = pd.DataFrame({"assetCode": data["assetCode"].values, "confidenceValue": confidence})
    pred.drop(["confidenceValue"], axis=1, inplace=True)
    pred = pd.merge(pred, confidence, how="left", left_on=["assetCode"], right_on=["assetCode"])
    pred.fillna(0, inplace=True)
    env.predict(pred)
    # lag_data = pd.concat([lag_data, data], axis=0)
    print("Day {}".format(day_idx))
    time_used = int(time.clock() - start)
    print("Time used: {} sec.".format(time_used))
    day_idx += 1
    
    del data
    del x_test
    del confidence
    gc.collect()

env.write_submission_file()
print("Finished.")


# In[ ]:



