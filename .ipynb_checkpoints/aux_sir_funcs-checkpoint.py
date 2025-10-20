import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import integrate, optimize
import matplotlib.pyplot as plt


def deal_country_data(population_, train_df, oname, nname):
    data_df = train_df[train_df["location"] == oname]
    data_df["location"] = nname
    
    df = pd.DataFrame()
    df["country"] = data_df["location"]
    df["date"] = data_df["date"]
    df["confirmed"] = data_df["new_cases"]
    df["fatalities"] = data_df["new_deaths"]
    # df["population"] = train_df["population"]
    df["population_density"] = data_df["population_density"]
    df["median_age"] = data_df["median_age"]
    df["aged_65_older"] = data_df["aged_65_older"]
    df["aged_70_older"] = data_df["aged_70_older"]
     
    country_df = df.merge(population_, how="left", on=['country']).drop_duplicates()
    
    country_df.confirmed.fillna(0, inplace=True)
    country_df.fatalities.fillna(0, inplace=True)
    country_df["confirmed"] = pd.to_numeric(country_df["confirmed"])
    country_df["fatalities"] = pd.to_numeric(country_df["fatalities"])
    country_df["population_density"] = pd.to_numeric(country_df["population_density"])
    country_df["median_age"] = pd.to_numeric(country_df["median_age"])
    country_df["aged_65_older"] = pd.to_numeric(country_df["aged_65_older"])
    country_df["aged_70_older"] = pd.to_numeric(country_df["aged_70_older"])
    
    country_df["confirmed"] = np.abs(country_df["confirmed"])
    country_df["fatalities"] = np.abs(country_df["fatalities"])
    
    return  country_df

def deal_global_df(population_, train_df, oname, nname):
    data_df = train_df[train_df["location"] == oname]
    data_df["location"] = nname
    
    country_df = pd.DataFrame()
    country_df["country"] = data_df["location"]
    country_df["date"] = data_df["date"]
    country_df["confirmed"] = data_df["new_cases"]
    country_df["fatalities"] = data_df["new_deaths"]
    
    country_df.confirmed.fillna(0, inplace=True)
    country_df.fatalities.fillna(0, inplace=True)
    country_df["confirmed"] = pd.to_numeric(country_df["confirmed"])
    country_df["fatalities"] = pd.to_numeric(country_df["fatalities"])
    
    country_df["confirmed"] = np.abs(country_df["confirmed"])
    country_df["fatalities"] = np.abs(country_df["fatalities"])
    
    list_cur = []
    list_cur.append({"country":nname,
                "population":population_["population"].mean(),
                "yearly_change":population_["yearly_change"].mean(),
                "net_change":population_["net_change"].mean(),
                "density":population_["density"].mean(),
                "land_area":population_["land_area"].mean(),
                "migrants":population_["migrants"].mean(),
                "rert_rate":population_["rert_rate"].mean(),
                "med_age":population_["med_age"].mean(),
                "urban_pop":population_["urban_pop"].mean(),
                "world_share":population_["world_share"].mean()})
    df = pd.DataFrame(list_cur)
    
    data_df = country_df.merge(df, how="left", on=['country']).drop_duplicates()
    
    
    return data_df

def log2_(df):
    df = df[df["confirmed"] >= 3]
    df = df[df["fatalities"] >= 3]
    df["confirmed"] = np.log(np.log(df["confirmed"]))
    df["fatalities"] = np.log(np.log(df["fatalities"]))
    idex = list(range(0, df.shape[0]))
    df["day"] = idex
    return df

def plot_data(t):

    plt.plot(range(0, t.shape[0]), t["confirmed"])
    plt.plot(range(0, t.shape[0]), t["fatalities"])
    plt.title('Confirmed & Fatalities Data')
    plt.ylabel('confirmed & Fatalities')
    plt.xlabel('Epoch')
    plt.show()
    
    
def addCsvDelNa(df): #累加
    temp_df = df.copy()
    temp_df = temp_df.dropna(axis=0,how='any')
    confirmed_ = temp_df.confirmed.fillna(0).astype(np.float32)
    fatalities_ = temp_df.fatalities.fillna(0).astype(np.float32)
    confirmed_ = np.array(confirmed_).reshape(-1, 1)
    fatalities_ = np.array(fatalities_).reshape(-1, 1)
    for i in range(confirmed_.shape[0] - 1):
        confirmed_[i+1] += confirmed_[i]
    for i in range(fatalities_.shape[0] - 1):
        fatalities_[i+1] += fatalities_[i]    
    temp_df['confirmed'] = confirmed_
    temp_df['fatalities'] = fatalities_
    return temp_df

def addCsvRepNa(df): #累加
    temp_df = df.copy()
    confirmed_ = temp_df.confirmed.fillna(0).astype(np.float32)
    fatalities_ = temp_df.fatalities.fillna(0).astype(np.float32)
    confirmed_ = np.array(confirmed_).reshape(-1, 1)
    fatalities_ = np.array(fatalities_).reshape(-1, 1)
    for i in range(confirmed_.shape[0] - 1):
        confirmed_[i+1] += confirmed_[i]
    for i in range(fatalities_.shape[0] - 1):
        fatalities_[i+1] += fatalities_[i]    
    temp_df['confirmed'] = confirmed_
    temp_df['fatalities'] = fatalities_
    return temp_df

#_________________________

def test(name, population_):
    total = 0
    if name == 'global':
        list1 = np.array(population_['r_population']).tolist()
        for ele in range(0, len(list1)):
            total = total + list1[ele]
    elif population_['country'].nunique()==1:
        total = population_['r_population'][0]
    else:
        total = population_[population_['country'] == name]['r_population'].values[0]
        
    return total


def SIR(country_df, population, names, N=0):
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    
    confirmed_  = country_df.confirmed.fillna(0)
    fatalities_ = country_df.fatalities.fillna(0)
    
    confirmed_ = np.array(confirmed_)
    fatalities_ = np.array(fatalities_)

    scaler1.fit(confirmed_.reshape(-1, 1))
    confirmed_ = scaler1.transform(confirmed_.reshape(-1, 1)).reshape(-1)
    
    scaler2.fit(fatalities_.reshape(-1, 1))
    fatalities_ = scaler2.transform(fatalities_.reshape(-1, 1)).reshape(-1)

    sir_df = pd.DataFrame()
    sir_df['confirmed'] = confirmed_ #country_df.confirmed.fillna(0)
    sir_df['fatalities'] = fatalities_ #country_df.fatalities.fillna(0)
    sir_df = sir_df[sir_df['confirmed'] >= 0]
    sir_df['day_count'] = list(range(1,len(sir_df)+1))
    
    ydata = [i for i in sir_df.confirmed]
    xdata = sir_df.day_count
    rdata = [i for i in sir_df.fatalities]
    
    rdata = np.array(rdata, dtype=float)
    ydata = np.array(ydata, dtype=float)
    xdata = np.array(xdata, dtype=float)
    
#     N = population[population['country'] == names]['r_population'].values[0]
    if N == 0:
        N = test(names, population)
    print(N)
    inf0 = ydata[0]
    rec0 = 0
    sus0 = N - inf0 - rec0
    
    def sir_model(y, x, beta, gamma):
        sus = -beta * y[0] * y[1] / N
        rec = gamma * y[1]
        inf = -(sus + rec)
        return sus, inf, rec
    
    def fit_odeint_c(x, beta, gamma):
        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]
    
    def fit_odeint_r(x, beta, gamma):
        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,2]
    
    popt, pcov = optimize.curve_fit(fit_odeint_c, xdata, ydata)
    fitted_c = fit_odeint_c(xdata, *popt)
    
    popt, pcov = optimize.curve_fit(fit_odeint_r, xdata, rdata)
    fitted_r = fit_odeint_r(xdata, *popt)
    
    
    fig = plt.figure(figsize=[20, 5])
    plt.subplot(121)
    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata, fitted_c)
    plt.title("Fit of SIR model for " + names + " infected cases")
    plt.ylabel("Population infected")
    plt.xlabel("Days")
    plt.legend(['groundtruth', 'predictions'], loc='best')
    
    plt.subplot(122)
    plt.plot(xdata, rdata, 'o')
    plt.plot(xdata, fitted_r)
    plt.title("Fit of SIR model for " + names + " infected cases")
    plt.ylabel("Population infected")
    plt.xlabel("Days")
    plt.legend(['groundtruth', 'predictions'], loc='best')
    plt.show()
    print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
    
    return fitted_c, fitted_r


def deal_SIR(df, name, population='', N=0):
    fc, fr = SIR(df, population, name, N)
    
    sirData = pd.DataFrame()
    sirData["confirmed"]  = fc
    sirData["fatalities"] = fr
    sirData["date"] = df["date"]
    idex = list(range(0, sirData.shape[0]))
    sirData["day"] = idex

#     sirData = log2_(sirData)
    plot_data(sirData)
    return sirData