

import pandas as pd
import datetime as dt
import  seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from  lifetimes import BetaGeoFitter
from  lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: '%.0f' % x)


# 1:  Veriyi Anlama ve Hazırlama

df_ = pd.read_csv("CRM/Miuul/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()

df.head(100)

def outlier_thresholds (dataframe, variable):
    quartile1 =dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range =quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquartile_range).round(0)
    low_limit = (quartile1 - 1.5 * interquartile_range).round(0)
    return low_limit, up_limit

#alt ve üst limitlerin aykırı değerlere atanması

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return low_limit, up_limit


df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()

# Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz

df.info()

date_col = [col for col in df.columns if "date" in col]

df[date_col] = df[date_col].apply(pd.to_datetime)

df.info()
df.head()

# 2: CLTV Veri Yapısının Oluşturulması


df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 2)


cltv_df = df[["master_id", "last_order_date", "first_order_date", "total_order", "total_value"]]

cltv_df.head()

cltv_df["recency"] =  ((cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days)
cltv_df["recency"] = cltv_df["recency"]  / 7

cltv_df["T"] =  (today_date - cltv_df["first_order_date"]).dt.days
cltv_df["T"] = cltv_df["T"] / 7

cltv_df["frequency"] = cltv_df["total_order"]

cltv_df.info()

cltv_df["monetary_cltv_avg"] = cltv_df["total_value"] / cltv_df["total_order"]

cltv_df.head()


# BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması


bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])


cltv_df["expected_purh_3_month"]= bgf.predict(12,
                                            cltv_df["frequency"],
                                            cltv_df["recency"],
                                            cltv_df["T"]
                                            )
cltv_df.sort_values("expected_purh_3_month", ascending=False)


cltv_df["expected_purh_6_month"]= bgf.predict(24,
                                            cltv_df["frequency"],
                                            cltv_df["recency"],
                                            cltv_df["T"]
                                            )
cltv_df.sort_values("expected_purh_6_month", ascending=False)



ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv_df["expected_average_profit"]=ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv_df.head()
cltv_df= cltv_df.drop("master_id", axis=1)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6, #buradaki değer aylıktır.
                                   freq="W", #frequency ve T değerlerinin haflık cinste olduğunu belritmiş olduk W yazarak
                                   discount_rate= 0.01 # indirim yapıldıysa bunu etkilendirmek için
                                    )

cltv.sort_values("")
cltv = cltv.reset_index()
cltv = cltv.reset_index()

cltv.sort_values("clv", ascending=False).head(20)


cltv_final = cltv_df.merge(cltv, on="master_id", how="left")

cltv_final.head()

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4,labels=["D", "C", "B", "A"])











