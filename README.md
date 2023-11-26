# FLO CLTV Tahmini -> BG-NBD ve GAMMA-GAMMA

![flo magaza](https://s3-eu-west-1.amazonaws.com/atlaspark/images/store_facade_image/SyvdQeYFQ-small.jpeg?1539869812527)

* ## İş Problemi

FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin **tahmin** edilmesi gerekmektedir.

* ## Veri Seti Hikayesi

Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (**hem online hem offline alışveriş yapan**) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
```
master_id: Unique customer number 
 
order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline) 
 
last_order_channel : The channel where the last purchase was made 
 
first_order_date : The date of the customer's first purchase
 
last_order_date : The last shopping date of the customer 
 
last_order_date_online : The customer's online shopping last purchase date on the platform 
 
last_order_date_offline : Date of the last purchase made by the customer on the offline platform 
 
order_num_total_ever_online : Total number of purchases made by the customer on the online platform 
 
order_num_total_ever_offline : Total number of purchases made by the customer 
 
offline customer_value_total_ever_offline : Total price paid by the customer in 
 
online shopping_value_total_ever_offline : List of categories the customer has shopped in the last 12 months
```



