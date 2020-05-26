import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 1000)
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression




# LOAD DATA AND CREATE DATAFRAME
data = pd.read_csv('UNFCCC_v22.csv', encoding='latin1')
# print all column names 
print("Column names:")
print(data.columns.values)
print ()
#data shape()
print ("Data shape:")
print (data.shape)
print()
# print all data types
print("Data Types:") 
print(data.dtypes)
print()


#VERIFY DATA TYPE FOR COLUMNS "YEAR" 
#all unique values for "Year"
print("Values in column Year:")
print (data['Year'].unique())
print ()

#drop all the rows that value is equal '1985-1987'
data.drop(data[data['Year'] == '1985-1987'].index , inplace=True)

#Now we can change the data type for int
data = data.astype({'Year' : 'int64'})

# And drop all the values below 1990
data.drop( data[ data['Year'] < 1990].index , inplace=True)
#after changes
print ("Column Year after changes:") 
print(data["Year"].head())
print ()



#Descriptive statistics

print ("Descriptive statistics:")
print()
print (data.describe(include='all'))
print()


#MISSING VALUES

# Missing values in a dataset 
mv = data.isnull().sum()/len(data)*100
print()
print("Missing Values:")
print (mv)

#drop the rows when notation is null and emission is 0
data = data.drop(data[(data['Notation'].isnull()) & (data['emissions'] == 0)].index)

# Filling missing values using fillna()
data['Notation'] = data['Notation'].fillna('-')

# Missing values for Notation (should be 0)
mvnotation = data['Notation'].isnull().sum()/len(data)*100
print ()
print("Missing Values Notation:")
print (mvnotation)

# Drop Parent_sector_code (missing values)
data = data.drop(columns=['Parent_sector_code'])
#  Rename columns emissions
data = data.rename(columns={"emissions": "emissions in Gg"})

# SELECTIONG IMPORTANT ROWS
# Leave only 6 Main sectors in DF 
data = data.loc[(data['Sector_code'] == '1') | (data['Sector_code'] == '2') | (data['Sector_code'] == '3') | (data['Sector_code'] == '4') | (data['Sector_code'] == '5') | (data['Sector_code'] == '6')]  

# drop not relevant columns
data = data.drop(columns=['PublicationDate', 'DataSource', 'Sector_code', 'Country_code', 'Format_name', 'Unit'])

# Checking the pollutant names
#p = data.groupby('Pollutant_name')
#print()
#print("Pollutant names:")
#print (p.first())
# Leave only rows where string contains "All"
data = data[data['Pollutant_name'].str.contains("All")]

# Leave only rows that DON'T contain string EU
data = data[~data['Country'].str.contains("EU")]


# sort values by country name, then sector name and year
data = data.sort_values(by=['Country', 'Sector_name','Pollutant_name','Year'])

# reset indexes
data = data.reset_index(drop=True)
data = data.drop(columns=['Notation'])



# print final shape
print()
print("Final DF shape:")
print ("---------------")
print (data.shape)
print ("---------------")
print ()



#Additional Column "Emissions gr"
#Sort values 
data.sort_values('emissions in Gg', ascending=False)

data['emissionsgr'] = data['emissions in Gg']

data.loc[data['emissions in Gg'] < 0, 'emissionsgr'] = 'Negative emission'
data.loc[data['emissions in Gg'] == 0, 'emissionsgr'] = 'Zero emission'

data['emissionsgr'] = np.where((data['emissions in Gg'] > 0)
                           & (data['emissions in Gg'] < 1000), #Identifies the case to apply to
                           '1 - 999',      #This is the value that is inserted
                           data['emissionsgr'])      #This is the column that is affected

data['emissionsgr'] = np.where((data['emissions in Gg'] >= 1000)
                           & (data['emissions in Gg'] < 10000),
                           '1000 - 9999',      
                           data['emissionsgr'])   

data['emissionsgr'] = np.where((data['emissions in Gg'] >= 10000)
                           & (data['emissions in Gg'] < 100000), 
                           '10000-99999',    
                           data['emissionsgr'])     

data['emissionsgr'] = np.where((data['emissions in Gg'] >= 100000) , 
                           '100000+',    
                           data['emissionsgr'])

#SECTORS

#1.Which sector is the most likely to be in the highest emission group?


## {'100000+'} --> {sector(?)}
num_records = data.shape[0]
# X = {'100000+'}
# support {X} = Number of times that X apear/ total number of cases
X = data.loc[data['emissionsgr'] == '100000+'].shape[0]

# support{'100000+'}
supportX =  X/num_records
print ("Support for {'100000+'} is :", round(supportX * 100, 3), "%")

sector_names = data['Sector_name'].unique()
x = 0
for i in sector_names:
    # select the name from the sector names
    name = sector_names[x]
    # Support for {100000+, sector(?)}
    Y = data.loc[(data['emissionsgr']  == '100000+') & (data['Sector_name'] == name)]
    Y = Y.shape[0]
    supportY = Y/num_records
    
    # Support for {sector}
    Z = data.loc[(data['Sector_name'] == name)]
    Z = Z.shape[0]
    supportZ = Z/num_records
    
    
    #Calculate confidence {'100000+'} -> {sector(?)}
    def zero_division(n, d):
        return n/d if d else 0
    confid = zero_division (supportY,supportX)
    
    # Calculate Lift 
    lift = supportY/(supportX * supportZ)
    if  (confid > 0.2) and (lift > 1) and (supportY > 0.02):
        print ("Rule: {100000+} -> {", name , "}")
        print ("Support: ", round(supportY,2))
        print ("Confidence: ", round(confid,2))
        print ("Lift: ", round(lift,2))
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    x = x + 1
    
print ()
print ()    
    

# 2. Which sector is the most likely to generate Negative Emission?


# {'Negative emission'} --> {sector(?)}
# X = {Negative emission}
# support {X} = Number of times that X apear/ total number of cases

X = data.loc[data['emissionsgr'] == 'Negative emission'].shape[0]

# support{Negative emission'}
supportX =  X/num_records
print ("Support for {Negative emission} is :", round(supportX*100), "%")


x = 0
for i in sector_names:
    # select the name from the sector names
    name = sector_names[x]
    
    # Support for {Negative emission, sector(?)}
    Y = data.loc[(data['emissionsgr']  == 'Negative emission') & (data['Sector_name'] == name)]
    Y = Y.shape[0]
    
    # Support for {sector}
    Z = data.loc[(data['Sector_name'] == name)]
    Z = Z.shape[0]
    supportZ = Z/num_records
    supportY = Y/num_records
    
    #Calculate confidence {Negative emission} -> {sector(?)}
    def zero_division(n, d):
        return n/d if d else 0
    confid = zero_division (supportY,supportX)
    
    # Calculate Lift 
    lift = supportY/(supportX * supportZ)
    if  (confid > 0.2) and (lift > 1) and (supportY > 0.02):
        print ("Rule: {Negative emission} -> {", name , "}")
        print ("Support: ", round(supportY,2))
        print ("Confidence: ", round(confid,2))
        print ("Lift: ", round(lift,2))
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    x = x + 1

print ()
print ()


# CONUTRIES

### 1.Which country increased and decreased their emissions the most between 1990-2017? 

country_names = data['Country'].unique()
y = 0
for i in country_names:
    c_name = country_names[y]
    sum1 = data.loc[(data['Country'] == c_name) & (data['Year'] == 1990) , 'emissions in Gg'].sum()
    sum2 = data.loc[(data['Country'] == c_name) & (data['Year'] == 2017) , 'emissions in Gg'].sum()
    change = ((sum2 - sum1)/sum1)* 100
    print (c_name, " :")
    print ('Emission in 1990:', round (sum1,2))
    print ('Emission in 2017:', round (sum2,2))
    print ("The percentage change between 1990 and 2015: ", round(change), "%")
    print ()
    y = y + 1



#Compare Sweden and Turqay using chart 

years = data['Year'].unique()
sweden =[]
turkey = []
a=0
for i in years:
    y = years[a]
    sum_sw = data.loc[(data['Year'] == y) & (data['Country'] == 'Sweden') , 'emissions in Gg'].sum()
    sum_tq = data.loc[(data['Year'] == y) & (data['Country'] == 'Turkey') , 'emissions in Gg'].sum()
    sweden.append(sum_sw)
    turkey.append(sum_tq)
    a=a+1

plt.figure(figsize=(12, 8))    
plt.plot(years,turkey,  marker='', color = 'red', label='Turkey', linewidth=3)
plt.plot(years,sweden,  marker='', color = 'green', label='Sweden', linewidth=3)
plt.grid(color='blue', linestyle='-', linewidth=0.5, alpha=0.3)

plt.title ("Greenhouse gases emission Sweden vs Turkey (1990-2017)", fontsize=17)
plt.legend(loc=2, prop={'size': 26})
plt.show()

print()




#2. Which country / countries most likely to  have small CO2 emissions '1 - 999' Gg?


# X = {'1 - 999'}
# support {X} = Number of times that X apear/ total number of cases

X = data.loc[data['emissionsgr'] == '1 - 999'].shape[0]

# support{'1 - 999'}
supportX =  X/num_records
print ("Support for {1 - 999} is :", round (supportX * 100, 2),'%')


country_names = data['Country'].unique()
country_names

country_names = data['Country'].unique()
x = 0
for i in country_names:
    # select the name from the country names
    name = country_names[x]
    # Support for {1 - 999, country(?)}
    Y = data.loc[(data['emissionsgr']  == '1 - 999') & (data['Country'] == name)]
    Y = Y.shape[0]
    
    # Support for {Country}
    Z = data.loc[(data['Country'] == name)]
    Z = Z.shape[0]
    supportZ = Z/num_records
    supportY = Y/num_records
    
    #Calculate confidence  Support for {'1 - 999'} -> {Country(?)}
    def zero_division(n, d):
        return n/d if d else 0
    confid = zero_division (supportY,supportX)
    
    # Calculate Lift 
    lift = supportY/(supportX * supportZ)
    if  (confid > 0.1) and (lift > 1) and (supportY > 0.02):
        print ("Rule: {1 - 999} -> {", name , "}")
        print ("Support: ", round(supportY,2))
        print ("Confidence: ", round(confid,2))
        print ("Lift: ", round(lift,2))
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    x = x + 1
   
print ()
print ()    


# 3. Which country generates the most of negative emissions?

no_negemissions = []

#set figure size
plt.figure(figsize=(12, 8))
print ("These countries generate the highest negative  emissions:")   
x = 0
for i in country_names: 
    c_name = country_names[x]
    sum_ne = data.loc[(data['emissionsgr'] == 'Negative emission') &  (data['Country'] == c_name), 'emissions in Gg'].sum()        
    x = x + 1 
    plt.bar(c_name, sum_ne)
    if sum_ne == 0:
        no_negemissions.append(c_name)
    if sum_ne < -1000000:
        print (c_name, ":" , round(sum_ne), "Gg" )

print()        
print ("These countries don't generate any negative emissions:")
print (*no_negemissions, sep = ", ")
print()
plt.xticks(country_names,rotation=90, fontsize=12)
plt.ylabel("Emission(Gg)", fontsize=15)
plt.title("Total Negative emissions by country (1990-2017)", fontsize=25)
plt.show()

### 4. Which country has the highest emission?
x = 0
print ("These countries generate the highest emissions:") 

plt.figure(figsize=(12, 8))  

for i in country_names: 
    c_name = country_names[x]
    sum_e = data.loc[(data['Country'] == c_name), 'emissions in Gg'].sum()
    plt.bar(c_name, sum_e)
    if sum_e > 13000000:
        print (c_name, ":" , round(sum_e), "Gg" )
    x = x + 1 

total_e_mean = data['emissions in Gg'].mean()
plt.xticks(country_names,rotation=90, fontsize=12)
plt.title("Total Greenhouse gases emission (1990-2017)", fontsize=25)
plt.show()
print ()
print ()


#Only for 2017
x = 0
print ("These countries generated the highest emission in 2017:") 

plt.figure(figsize=(12, 8))  
for i in country_names: 
    c_name = country_names[x]
    sum_e = data.loc[(data['Country'] == c_name ) & (data['Year'] == 2017)  , 'emissions in Gg'].sum()
    plt.bar(c_name, sum_e)
    if sum_e > 100000:
        print (c_name, ":" , round(sum_e), "Gg" )
    x = x + 1 
mean = data['emissions in Gg'].mean()    
plt.xticks(country_names,rotation=90, fontsize=12)
plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
plt.title("Total Greenhouse gases emission (2017 only)", fontsize=25)
plt.axhline(y=mean, color='gray',linestyle='--')
plt.show()


## 3. EMISSON
# 1.Is negative emission decreasing or increasing? 

years = data["Year"].unique()
z=0
xyz=[]
for i in years:
    year = years[z]
    y = data.loc[(data['Sector_name'] == '4 - Land Use, Land-Use Change and Forestry') & (data['Year'] == year), 'emissions in Gg'].sum()
    print (year, y)
    xyz.append(y)
    z=z+1
    
x=xyz
plt.figure(figsize=(10, 10))  
plt.plot(years,x,'ko-',color="orange", linewidth=3, alpha=1)
plt.xlabel("Year", fontsize=15)
plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.3)
plt.title ("Negative greenhouse gases emissions between 1990-2017", fontsize=15)
plt.ylabel("Emission(Gg)", fontsize=15)
plt.show()

print ()
print ()

#4. What’s the relationship between year and total emission?
years = data['Year'].unique()
x = 0 
emission = []
for i in years:
    y = years[x]
    sum_e = data.loc[(data['Year'] == y), 'emissions in Gg'].sum()
    emission.append(sum_e)
    x = x+1

sns.jointplot(x=years, y=emission, data=data, kind = 'reg',fit_reg= True, size = 8)    
plt.show()
print()


# DATA MODELLING AND VISUALISATION

# 1. How Sweden managed to reduce the emission by 76%?
sec = data["Sector_name"].unique()

# style
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')

sweden =[]
sec_by_year= []

c = data.loc[(data['Country'] == 'Sweden')]
x=0

# VALUES FOR LINE CHART (total emission by year in Sweden)  
for i in years:
    year = years[x]
    l = c.loc[(data['Year'] == year) , 'emissions in Gg'].sum()
    sweden.append(l)
    x+=1

num = 0
no_plots = 1

fig = plt.figure(figsize=(15,20))

# VALUES FOR BAR CHARTS (subplot)
for i in sec:
    sector = sec[num]
    ax1 = plt.subplot(6, 1, num+1)
    num=num+1
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Emission for sector(Gg)', fontsize= 12)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total emission(Gg)', fontsize= 12)
    ax2.grid(b=False)
    count=0
    for i in years:
        year= years[count]
        s = c.loc[(data['Sector_name'] == sector)& (c['Year'] == year) , 'emissions in Gg'].sum()
        #bar charts
        ax1.bar (year,s, color=palette(num))
        count += 1
        plt.title (label=sector, fontsize=15)
    # line charts   
    ax2.plot(years, sweden, color='black', linewidth=3, alpha=0.6)
    fig.tight_layout()
    no_plots = no_plots+1
print ("SWEDEN:")    
plt.show ()


print ("------------------------------------------------")


#2. What caused such a dramatic increase in greenhouse gases emission in Turkey?

turkey =[]
sec_by_year= []

t = data.loc[(data['Country'] == 'Turkey')]
x=0

# VALUES FOR LINE CHART (total emission by year in Sweden)  
for i in years:
    year = years[x]
    l = t.loc[(data['Year'] == year) , 'emissions in Gg'].sum()
    turkey.append(l)
    x+=1

num = 0
no_plots = 1

fig = plt.figure(figsize=(15,20))

# VALUES FOR BAR CHARTS (subplot)
for i in sec:
    sector = sec[num]
    ax1 = plt.subplot(6, 1, num+1)
    num=num+1
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Emission for sector(Gg)', fontsize= 12)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total emission(Gg)', fontsize= 12)
    ax2.grid(b=False)
    count=0
    for i in years:
        year= years[count]
        s = c.loc[(data['Sector_name'] == sector)& (c['Year'] == year) , 'emissions in Gg'].sum()
        #bar charts
        ax1.bar (year,s, color=palette(num))
        count += 1
        plt.title (label=sector, fontsize=15)
    # line charts   
    ax2.plot(years, turkey, color='black', linewidth=3, alpha=0.6)
    fig.tight_layout()
    no_plots = no_plots+1
print ("TURKEY:")
plt.show ()


#3.So, does negative emission from sector 4 really have a positive impact on decreasing overall emissions?

import matplotlib.patches as mpatches

years = data["Year"].unique()

z=0
# total negative emissions by year (values)
xyz=[]
# total emissions by year (values)
pe = []
for i in years:
    year = years[z]
    y = data.loc[(data['Sector_name'] == '4 - Land Use, Land-Use Change and Forestry') & (data['Year'] == year) , 'emissions in Gg'].sum()
    p = data.loc[ (data['Year'] == year) , 'emissions in Gg'].sum()
    xyz.append(y)
    pe.append(p)
    z=z+1
    
neg=xyz
total=pe

#build the plot
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()  # set up the 2nd axis

#Negative emissions
ax2.plot(years,neg,'ko-',color='green', linewidth=3, alpha=1)
#Total emission 
ax1.plot(years,total,color='red', linewidth=3, alpha=1)
# turn off grid #2
ax1.grid(b=False) 
plt.xticks(np.arange(1990, 2017, step=2))

#set labels
ax1.set_xlabel('Years')
ax1.set_ylabel('Total Emission (Gg)', fontsize= 12)
ax2.set_ylabel('Negative emission only (Gg)', fontsize= 12)

#set legend
red_patch = mpatches.Patch(color='green', label='Negative emission (Gg)')
blue_patch = mpatches.Patch(color='red', label='Total emission (Gg)')
plt.legend(handles=[red_patch, blue_patch])
plt.title("Negative emission vs Total emission over the years.")
plt.show()


print ()
print ()
print ()


# Prediction 

x = 0 
emission = []
for i in years:
    y = years[x]
    sum_e = data.loc[(data['Year'] == y), 'emissions in Gg'].sum()
    emission.append(sum_e)
    x = x+1

# set target variable and feature
target = np.array(emission)
feature = np.array(years).reshape((-1, 1))


# Fit the model
model = LinearRegression().fit(feature, target)

# Check model scores
r_sq = model.score(feature, target)
print('Coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


# Make prediction 
x_new = range(2020, 2080)
x_new = np.array(x_new).reshape((-1, 1))
y_pred = model.predict(x_new)

print ("- - - - -")
print('Predicted response:', y_pred, sep='\n')
print ("- - - - -")

# Plot prediction
fig = plt.figure(figsize=(10,10))
plt.plot(x_new, y_pred, color='black', linewidth=3, alpha=0.6, label="Predicted emission")
plt.grid(alpha = 0.6)
plt.title("EU Greenhouse gases emission. Prediction for 2020-2080", fontsize=20, color='red')
plt.xticks(fontsize=12)
plt.legend(fontsize= 16)
plt.show()

######################################
# Which sector increased and decreased their emissions the most between 1990-2017?
y = 0
print ("Which sector increased and decreased their emissions the most between 1990-2017?")
sum_1990 = []
sum_2017 = []
for i in sector_names:
    c_name = sector_names[y]
    sum1 = data.loc[(data['Sector_name'] == c_name) & (data['Year'] == 1990) , 'emissions in Gg'].sum()
    sum2 = data.loc[(data['Sector_name'] == c_name) & (data['Year'] == 2017) , 'emissions in Gg'].sum()
    sum_1990.append(sum1)
    sum_2017.append(sum2)
    change = ((sum2 - sum1)/sum1)* 100
    print (c_name, " :")
    print ('Emission in 1990:', round (sum1,2))
    print ('Emission in 2017:', round (sum2,2))
    print ("The percentage change between 1990 and 2017: ", round(change), "%")
    print ()
    y = y + 1
    
print ()

    

# Sector 4 - subsectors emission


df = pd.read_csv('UNFCCC_v22.csv', encoding='latin1')
x = df[df['Sector_code'].str.startswith('4')]
all_sec = x['Sector_code'].unique()

s = 0
sec_4_em = []
plt.figure(figsize=(12, 8))    
for i in all_sec:
    s_code = all_sec[s]
    sum = df.loc[(df['Sector_code'] == s_code), 'emissions'].sum()
    plt.bar(s_code, sum)
    sec_4_em.append(sum)
    s=s+1
    
plt.xticks(all_sec,rotation=90, fontsize=12)
plt.title("Emissions for sector 4 - Land Use, Land-Use Change and Forestry and its subsectors", fontsize=20)
plt.show()

