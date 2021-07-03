################################### problem1 #######################################
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
airlines = pd.read_excel("C:\\Users\\DELL\\Downloads\\EastWestAirlines (1).xlsx", sheet_name= "data" )
airlines.describe()
airlines.info()
airlines.isna().sum()
airlines.isnull().sum()

#to check duplicates
dups = airlines.duplicated()
sum(dups)

airlines = airlines.drop_duplicates()
#univariate analysis
#boxplot for some features of df airlines
plt.boxplot(airlines.Balance)
plt.boxplot(airlines.Qual_miles)
plt.boxplot(airlines.cc1_miles)
plt.boxplot(airlines.cc2_miles)
plt.boxplot(airlines.cc3_miles)
plt.boxplot(airlines.Bonus_miles)
plt.boxplot(airlines.Bonus_trans)
plt.boxplot(airlines.Flight_miles_12mo)
plt.boxplot(airlines.Flight_trans_12)
plt.boxplot(airlines.Days_since_enroll)

#histogram for some features of df airlines
plt.hist(airlines.Balance)
plt.hist(airlines.Qual_miles)
plt.hist(airlines.cc1_miles)
plt.hist(airlines.cc3_miles)

#detection of outliers(find the RM based on IQR)
IQR = airlines["Balance"].quantile(0.75) - airlines["Balance"].quantile(0.25)
lower_limit_balance = airlines["Balance"].quantile(0.25) - (IQR * 1.5)
upper_limit_balance = airlines["Balance"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Balance"] > upper_limit_balance, upper_limit_balance,
                                                      np.where(airlines["Balance"] < lower_limit_balance,lower_limit_balance,airlines["Balance"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Bonus_miles"].quantile(0.75) - airlines["Bonus_miles"].quantile(0.25)
lower_limit_bonus_miles = airlines["Bonus_miles"].quantile(0.25) - (IQR * 1.5)
upper_limit_bonus_miles = airlines["Bonus_miles"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Bonus_miles"] > upper_limit_bonus_miles, upper_limit_bonus_miles,
                                                      np.where(airlines["Bonus_miles"] < lower_limit_bonus_miles,lower_limit_bonus_miles,airlines["Bonus_miles"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Bonus_trans"].quantile(0.75) - airlines["Bonus_trans"].quantile(0.25)
lower_limit_Bonus_trans = airlines["Bonus_trans"].quantile(0.25) - (IQR * 1.5)
upper_limit_Bonus_trans = airlines["Bonus_trans"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Bonus_trans"] > upper_limit_Bonus_trans, upper_limit_Bonus_trans,
                                                      np.where(airlines["Bonus_trans"] < lower_limit_Bonus_trans,lower_limit_Bonus_trans,airlines["Bonus_trans"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Flight_miles_12mo"].quantile(0.75) - airlines["Flight_miles_12mo"].quantile(0.25)
lower_limit_Flight_miles_12mo = airlines["Flight_miles_12mo"].quantile(0.25) - (IQR * 1.5)
upper_limit_Flight_miles_12mo = airlines["Flight_miles_12mo"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["airlines_replaced"] = pd.DataFrame(np.where(airlines["Flight_miles_12mo"] > upper_limit_Flight_miles_12mo, upper_limit_Flight_miles_12mo,
                                                      np.where(airlines["Flight_miles_12mo"] < lower_limit_Flight_miles_12mo,lower_limit_Flight_miles_12mo,airlines["Flight_miles_12mo"])))
sns.boxplot(airlines["airlines_replaced"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Flight_trans_12"].quantile(0.75) - airlines["Flight_trans_12"].quantile(0.25)
lower_limit_Flight_trans_12 = airlines["Flight_trans_12"].quantile(0.25) - (IQR * 1.5)
upper_limit_Flight_trans_12 = airlines["Flight_trans_12"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["Flight_trans_12"] = pd.DataFrame(np.where(airlines["Flight_trans_12"] > upper_limit_Flight_trans_12, upper_limit_Flight_trans_12,
                                                      np.where(airlines["Flight_trans_12"] < lower_limit_Flight_trans_12,lower_limit_Flight_trans_12,airlines["Flight_trans_12"])))
sns.boxplot(airlines["Flight_trans_12"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = airlines["Days_since_enroll"].quantile(0.75) - airlines["Days_since_enroll"].quantile(0.25)
lower_limit_Days_since_enroll = airlines["Days_since_enroll"].quantile(0.25) - (IQR * 1.5)
upper_limit_Days_since_enroll = airlines["Days_since_enroll"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
airlines["Days_since_enroll"] = pd.DataFrame(np.where(airlines["Days_since_enroll"] > upper_limit_Days_since_enroll, upper_limit_Days_since_enroll,
                                                      np.where(airlines["Days_since_enroll"] < lower_limit_Days_since_enroll,lower_limit_Days_since_enroll,airlines["Days_since_enroll"])))
sns.boxplot(airlines["Days_since_enroll"]);plt.title("Boxplot");plt.show()

#bivariate analysis
#scatterplot

sns.scatterplot(airlines["Balance"],airlines["Qual_miles"])
sns.scatterplot(airlines["cc1_miles"],airlines["cc2_miles"])
sns.scatterplot(airlines["cc3_miles"],airlines["Bonus_miles"])
sns.scatterplot(airlines["Bonus_trans"],airlines["Bonus_trans"])

#normalization function
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
    
#normalising data frame (conidering the numerical part of the data)
airlines_norm = norm_fun(airlines)
airlines_norm.describe()

airlines.columns

from sklearn.cluster import	KMeans

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(airlines_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines['clust'] = mb # creating a  new column and assigning it to new column 

airlines.head()
airlines_norm.head()

airlines = airlines.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
airlines.head()

airlines.iloc[:, 0:].groupby(airlines.clust).mean()

airlines.to_csv("Kmeans_airlines.csv", encoding = "utf-8")

import os
os.getcwd()

####################################### problem2 ###############################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import numpy as np
import os
import seaborn as sns

crime_data = pd.read_csv("C:\\Users\\DELL\\Downloads\\crime_data.csv")
crime_data.describe()
crime_data.info()
crime_data.isna().sum()
crime_data.isnull().sum()
crime_data.columns

#to check duplicates
dups = crime_data.duplicated()
sum(dups)

crime_data = crime_data.drop_duplicates()
#univariate analysis
#boxplot
plt.boxplot(crime_data.Murder)
plt.boxplot(crime_data.Assault)
plt.boxplot(crime_data.UrbanPop)
plt.boxplot(crime_data.Rape)

#bivariate analysis
#scatterplot
sns.scatterplot(crime_data["Assault"],crime_data["Murder"])
sns.scatterplot(crime_data["UrbanPop"],crime_data["Rape"])

#detection of outliers(find the RM based on IQR)
IQR = crime_data["Rape"].quantile(0.75) - crime_data["Rape"].quantile(0.25)
lower_limit_Rape = crime_data["Rape"].quantile(0.25) - (IQR * 1.5)
upper_limit_Rape = crime_data["Rape"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
crime_data["Rape"] = pd.DataFrame(np.where(crime_data["Rape"] > upper_limit_Rape, upper_limit_Rape,
                                                      np.where(crime_data["Rape"] < lower_limit_Rape,lower_limit_Rape,crime_data["Rape"])))
sns.boxplot(crime_data["Rape"]);plt.title("Boxplot");plt.show()

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
#normalised data frame considering numerical part of the data
crime_data_norm = norm_func(crime_data.iloc[:,1:])
crime_data_norm.describe()

from sklearn.cluster import	KMeans

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(crime_data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime_data['clust'] = mb # creating a  new column and assigning it to new column 

crime_data.head()
crime_data_norm.head()

crime_data = crime_data.iloc[:,[5,0,1,2,3,4]]
crime_data.head()

crime_data.iloc[:, 0:].groupby(crime_data.clust).mean()

crime_data.to_csv("Kmeans_crime_data.csv", encoding = "utf-8")

import os
os.getcwd()

############################################ problem3 ##################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import numpy as np
import seaborn as sns
import os

ins_data = pd.read_csv("C:\\Users\\DELL\\Downloads\\Insurance Dataset.csv")
ins_data.describe()
ins_data.info()
ins_data.isna().sum()
ins_data.columns
ins_data.isnull().sum()
#to check duplicates
dups =ins_data.duplicated()
sum(dups)

ins_data = ins_data.drop_duplicates()

#univariate analysis
#boxplot
plt.boxplot(ins_data["Premiums Paid"])
plt.boxplot(ins_data.Age)
plt.boxplot(ins_data["Days to Renew"])
plt.boxplot(ins_data["Claims made"])
plt.boxplot(ins_data.Income)

#detection of outliers(find the RM based on IQR)
IQR = ins_data["Premiums Paid"].quantile(0.75) - ins_data["Premiums Paid"].quantile(0.25)
lower_limit_ins_data = ins_data["Premiums Paid"].quantile(0.25) - (IQR * 1.5)
upper_limit_ins_data = ins_data["Premiums Paid"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
ins_data["Premiums Paid"] = pd.DataFrame(np.where(ins_data["Premiums Paid"] >upper_limit_ins_data, upper_limit_ins_data,
                                                      np.where(ins_data["Premiums Paid"] < lower_limit_ins_data,lower_limit_ins_data,ins_data["Premiums Paid"])))
sns.boxplot(ins_data["Premiums Paid"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = ins_data["Claims made"].quantile(0.75) - ins_data["Claims made"].quantile(0.25)
lower_limit_ins_data = ins_data["Claims made"].quantile(0.25) - (IQR * 1.5)
upper_limit_ins_data = ins_data["Claims made"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
ins_data["Claims made"] = pd.DataFrame(np.where(ins_data["Claims made"] >upper_limit_ins_data, upper_limit_ins_data,
                                                      np.where(ins_data["Claims made"] < lower_limit_ins_data,lower_limit_ins_data,ins_data["Claims made"])))
sns.boxplot(ins_data["Claims made"]);plt.title("Boxplot");plt.show()

#bivariate analysis
#scatterplot
sns.scatterplot(ins_data["Premiums Paid"],ins_data["Days to Renew"])
sns.scatterplot(ins_data["Claims made"],ins_data["Premiums Paid"])

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

ins_data_norm = norm_func(ins_data)

from sklearn.cluster import	KMeans

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(ins_data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(ins_data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
ins_data['clust'] = mb # creating a  new column and assigning it to new column 

ins_data.head()
ins_data_norm.head()

ins_data = ins_data.iloc[:,[5,0,1,2,3,4]]
ins_data.head()

ins_data.iloc[:, 0:].groupby(ins_data.clust).mean()

ins_data.to_csv("Kmeans_ins_data.csv", encoding = "utf-8")

import os
os.getcwd()


######################################### problem4 ###############################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import numpy as np
import seaborn as sns
import os

tel_com = pd.read_excel("C:\\Users\\DELL\\Downloads\\Telco_customer_churn (1).xlsx")
tel_com.describe()
tel_com.info()
tel_com.isna().sum()
tel_com.columns

#to check duplicates
dups = tel_com.duplicated()
sum(dups)

tel_com = tel_com.drop_duplicates()

#creat dummy variables on categorical data
tel_com_new = pd.get_dummies(tel_com)

#creating instance of one hot encoding
tel_com = pd.read_excel("C:\\Users\\DELL\\Downloads\\Telco_customer_churn (1).xlsx")
tel_com.describe()
tel_com.drop(['Count','Quarter'],axis = 1,inplace = True)
tel_com.dtypes

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
one_hot_tel_com = pd.DataFrame(one_hot.fit_transform(tel_com).toarray())

#creating instance of labelencoder
tel_com = pd.read_excel("C:\\Users\\DELL\\Downloads\\Telco_customer_churn.xlsx")
tel_com.describe()
tel_com.drop(['Count','Quarter'],axis = 1,inplace = True)
tel_com.dtypes

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x = tel_com.iloc[: , :]

x['Customer ID'] = label_encoder.fit_transform(x['Customer ID'])
x['Count'] = label_encoder.fit_transform(x['Count'])
x['Quarter'] = label_encoder.fit_transform(x['Quarter'])
x['Referred a Friend'] = label_encoder.fit_transform(x['Referred a Friend'])
x['Number of Referrals'] = label_encoder.fit_transform(x['Number of Referrals'])
x['Tenure in Months'] = label_encoder.fit_transform(x['Tenure in Months'])
x['Offer'] = label_encoder.fit_transform(x['Offer'])
x['Phone Service'] = label_encoder.fit_transform(x['Phone Service'])
x['Avg Monthly Long Distance Charges'] = label_encoder.fit_transform(x['Avg Monthly Long Distance Charges'])
x['Multiple Lines'] = label_encoder.fit_transform(x['Multiple Lines'])
x['Internet Service'] = label_encoder.fit_transform(x['Internet Service'])
x['Internet Type'] = label_encoder.fit_transform(x['Internet Type'])
x['Avg Monthly GB Download'] = label_encoder.fit_transform(x['Avg Monthly GB Download'])
x['Online Security'] = label_encoder.fit_transform(x['Online Security'])
x['Online Backup'] = label_encoder.fit_transform(x['Online Backup'])
x['Device Protection Plan'] = label_encoder.fit_transform(x['Device Protection Plan'])
x['Premium Tech Support'] = label_encoder.fit_transform(x['Premium Tech Support'])
x['Streaming TV'] = label_encoder.fit_transform(x['Streaming TV'])
x['Streaming Movies'] = label_encoder.fit_transform(x['Streaming Movies'])
x['Streaming Music'] = label_encoder.fit_transform(x['Streaming Music'])
x['Unlimited Data'] = label_encoder.fit_transform(x['Unlimited Data'])
x['Contract'] = label_encoder.fit_transform(x['Contract'])
x['Paperless Billing'] = label_encoder.fit_transform(x['Paperless Billing'])
x['Payment Method'] = label_encoder.fit_transform(x['Payment Method'])
x['Monthly Charge'] = label_encoder.fit_transform(x['Monthly Charge'])
x['Total Charges'] = label_encoder.fit_transform(x['Total Charges'])
x['Total Refunds'] = label_encoder.fit_transform(x['Total Refunds'])
x['Total Refunds'] = label_encoder.fit_transform(x['Total Refunds'])
x['Total Long Distance Charges'] = label_encoder.fit_transform(x['Total Long Distance Charges'])
x['Total Revenue'] = label_encoder.fit_transform(x['Total Revenue'])

#univariate analysis
#boxplot
plt.boxplot(tel_com["Avg Monthly Long Distance Charges"])
plt.boxplot(tel_com["Count"])
plt.boxplot(tel_com["Number of Referrals"]) 
plt.boxplot(tel_com["Tenure in Months"])
plt.boxplot(tel_com["Avg Monthly Long Distance"])
plt.boxplot(tel_com["Avg Monthly GB Download"])
plt.boxplot(tel_com["Monthly Charge"])
plt.boxplot(tel_com["Total Charges"])
plt.boxplot(tel_com["Total Refunds"])
plt.boxplot(tel_com["Total Extra Data Charges"])
plt.boxplot(tel_com["Total Long Distance Charges"])
plt.boxplot(tel_com["Total Revenue"])

#detection of outliers(find the RM based on IQR)
IQR = tel_com["Number of Referrals"].quantile(0.75) - tel_com["Number of Referrals"].quantile(0.25)
lower_limit_Number_of_Referrals = tel_com["Number of Referrals"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = tel_com["Number of Referrals"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
tel_com["Number of Referrals"] = pd.DataFrame(np.where(tel_com["Number of Referrals"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(tel_com["Number of Referrals"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,tel_com["Number of Referrals"])))
sns.boxplot(tel_com["Number of Referrals"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = tel_com["Monthly Charge"].quantile(0.75) - tel_com["Monthly Charge"].quantile(0.25)
lower_limit_Number_of_Referrals = tel_com["Monthly Charge"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = tel_com["Monthly Charge"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
tel_com["Monthly Charge"] = pd.DataFrame(np.where(tel_com["Monthly Charge"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(tel_com["Monthly Charge"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,tel_com["Monthly Charge"])))
sns.boxplot(tel_com["Monthly Charge"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = tel_com["Total Revenue"].quantile(0.75) - tel_com["Total Revenue"].quantile(0.25)
lower_limit_Number_of_Referrals = tel_com["Total Revenue"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = tel_com["Total Revenue"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
tel_com["Total Revenue"] = pd.DataFrame(np.where(tel_com["Total Revenue"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(tel_com["Total Revenue"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,tel_com["Total Revenue"])))
sns.boxplot(tel_com["Total Revenue"]);plt.title("Boxplot");plt.show()

#bivariate analysis
#scatterplot
sns.scatterplot(tel_com["Number of Referrals"],tel_com["Monthly Charge"])
sns.scatterplot(tel_com["Number of Referrals"],tel_com["Total Revenue"])

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

tel_com_norm = norm_func(tel_com_new)

from sklearn.cluster import	KMeans

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(tel_com_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(tel_com_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
tel_com['clust'] = mb # creating a  new column and assigning it to new column 

tel_com.head()
tel_com_norm.head()

tel_com = tel_com.iloc[:,[28,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]
tel_com.head()

tel_com.iloc[:, 0:].groupby(tel_com.clust).mean()

tel_com.to_csv("Kmeans_tel_com.csv", encoding = "utf-8")

import os
os.getcwd()

############################################ problem5 ##########################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import seaborn as sns
import numpy as np
import os

auto_ins = pd.read_csv("C:\\Users\\DELL\\Downloads\\AutoInsurance (1).csv")
auto_ins.describe()
auto_ins.info()
auto_ins.isna().sum()
auto_ins.columns

#to check duplicates
dups = auto_ins.duplicated()
sum(dups)

auto_ins = auto_ins.drop_duplicates()

#creat dummy variables on categorical data
auto_ins_new = pd.get_dummies(auto_ins)

#creating instance of one hot encoding
auto_ins = pd.read_csv("C:\\Users\\DELL\\Downloads\\AutoInsurance (1).csv")
auto_ins.describe()
auto_ins.drop(['Number of Open Complaints'],axis = 1,inplace = True)
auto_ins.dtypes

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
one_hot_auto_ins = pd.DataFrame(one_hot.fit_transform(auto_ins).toarray())

#creating instance of labelencoder
auto_ins = pd.read_csv("C:\\Users\\DELL\\Downloads\\AutoInsurance.csv")
auto_ins.describe()
auto_ins.drop(['Number of Open Complaints'],axis = 1,inplace = True)
auto_ins.dtypes

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x = auto_ins.iloc[: , :]

x['Customer'] = label_encoder.fit_transform(x['Customer'])
x['State'] = label_encoder.fit_transform(x['State'])
x['Customer Lifetime Value'] = label_encoder.fit_transform(x['Customer Lifetime Value'])
x['Response'] = label_encoder.fit_transform(x['Response'])
x['Coverage'] = label_encoder.fit_transform(x['Coverage'])
x['Education'] = label_encoder.fit_transform(x['Education'])
x['Effective To Date'] = label_encoder.fit_transform(x['Effective To Date'])
x['EmploymentStatus'] = label_encoder.fit_transform(x['EmploymentStatus'])
x['Gender'] = label_encoder.fit_transform(x['Gender'])
x['Income'] = label_encoder.fit_transform(x['Income'])
x['Location Code'] = label_encoder.fit_transform(x['Location Code'])
x['Marital Status'] = label_encoder.fit_transform(x['Marital Status'])
x['Monthly Premium Auto'] = label_encoder.fit_transform(x['Monthly Premium Auto'])
x['Months Since Last Claim'] = label_encoder.fit_transform(x['Months Since Last Claim'])
x['Months Since Policy Inception'] = label_encoder.fit_transform(x['Months Since Policy Inception'])
x['Number of Open Complaints'] = label_encoder.fit_transform(x['Number of Open Complaints'])
x['Number of Policies'] = label_encoder.fit_transform(x['Number of Policies'])
x['Policy Type'] = label_encoder.fit_transform(x['Policy Type'])
x['Policy'] = label_encoder.fit_transform(x['Policy'])
x['Renew Offer Type'] = label_encoder.fit_transform(x['Renew Offer Type'])
x['Sales Channel'] = label_encoder.fit_transform(x['Sales Channel'])
x['Total Claim Amount'] = label_encoder.fit_transform(x['Total Claim Amount'])
x['Vehicle Class'] = label_encoder.fit_transform(x['Vehicle Class'])
x['Vehicle Class'] = label_encoder.fit_transform(x['Vehicle Class'])


#univariate analysis
#boxplot
plt.boxplot(auto_ins["Customer Lifetime Value"])
plt.boxplot(auto_ins["Income"])
plt.boxplot(auto_ins["Monthly Premium Auto"])
plt.boxplot(auto_ins["Months Since Last Claim"])
plt.boxplot(auto_ins["Months Since Policy Inception"])
plt.boxplot(auto_ins["Number of Open Complaints"])
plt.boxplot(auto_ins["Number of Policies"])
plt.boxplot(auto_ins["Total Claim Amount"])

#detection of outliers(find the RM based on IQR)
IQR = auto_ins["Customer Lifetime Value"].quantile(0.75) - auto_ins["Customer Lifetime Value"].quantile(0.25)
lower_limit_Number_of_Referrals = auto_ins["Customer Lifetime Value"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = auto_ins["Customer Lifetime Value"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
auto_ins["Customer Lifetime Value"] = pd.DataFrame(np.where(auto_ins["Customer Lifetime Value"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(auto_ins["Customer Lifetime Value"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,auto_ins["Customer Lifetime Value"])))
sns.boxplot(auto_ins["Customer Lifetime Value"]);plt.title("Boxplot");plt.show()

#detection of outliers(find the RM based on IQR)
IQR = auto_ins["Monthly Premium Auto"].quantile(0.75) - auto_ins["Monthly Premium Auto"].quantile(0.25)
lower_limit_Number_of_Referrals = auto_ins["Monthly Premium Auto"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = auto_ins["Monthly Premium Auto"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
auto_ins["Monthly Premium Auto"] = pd.DataFrame(np.where(auto_ins["Monthly Premium Auto"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(auto_ins["Monthly Premium Auto"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,auto_ins["Monthly Premium Auto"])))
sns.boxplot(auto_ins["Monthly Premium Auto"]);plt.title("Boxplot");plt.show()


#detection of outliers(find the RM based on IQR)
IQR = auto_ins["Total Claim Amount"].quantile(0.75) - auto_ins["Total Claim Amount"].quantile(0.25)
lower_limit_Number_of_Referrals = auto_ins["Total Claim Amount"].quantile(0.25) - (IQR * 1.5)
upper_limit_Number_of_Referrals = auto_ins["Total Claim Amount"].quantile(0.75) + (IQR * 1.5)

#replace the outliers by the maximum and minimum limit
auto_ins["Total Claim Amount"] = pd.DataFrame(np.where(auto_ins["Total Claim Amount"] >upper_limit_Number_of_Referrals, upper_limit_Number_of_Referrals,
                                                      np.where(auto_ins["Total Claim Amount"] < lower_limit_Number_of_Referrals,lower_limit_Number_of_Referrals,auto_ins["Total Claim Amount"])))
sns.boxplot(auto_ins["Total Claim Amount"]);plt.title("Boxplot");plt.show()

#bivariate analysis
#scatter plot
sns.scatterplot(auto_ins["Total Claim Amount"],auto_ins["Monthly Premium Auto"])
sns.scatterplot(auto_ins["Total Claim Amount"],auto_ins["Customer Lifetime Value"])

#normalisation function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

auto_ins_norm = norm_func(auto_ins_new)

from sklearn.cluster import	KMeans

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(auto_ins_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(auto_ins_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
auto_ins['clust'] = mb # creating a  new column and assigning it to new column 

auto_ins.head()
auto_ins_norm.head()

auto_ins = auto_ins.iloc[:,[24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
auto_ins.head()

auto_ins.iloc[:, 0:].groupby(auto_ins.clust).mean()

auto_ins.to_csv("Kmeans_auto_ins.csv", encoding = "utf-8")

import os
os.getcwd()
