# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:13:20 2023

@author: Manoj
"""
# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import err_ranges as err


# Reading input data files
def read_data(file_name):
    """
    Function to read datafile

    """

    df = pd.read_csv(file_name)

    df_change = df.drop(columns=["Series Name", "Country Name", "Country Code"])
    df_change = df_change.replace(np.nan, 0)
    df_transpose = np.transpose(df_change)
    df_transpose = df_transpose.reset_index()
    df_transpose = df_transpose.rename(columns={"index": "year", 0: "UK", 1: "India"})

    df_transpose = df_transpose.iloc[1:]
    df_transpose = df_transpose.dropna()

    df_transpose["year"] = df_transpose["year"].str[:4]
    df_transpose["year"] = pd.to_numeric(df_transpose["year"])
    df_transpose["India"] = pd.to_numeric(df_transpose["India"])
    df_transpose["UK"] = pd.to_numeric(df_transpose["UK"])
    print(df_transpose)
    return df_change, df_transpose


def curve_function(t, scale, growth):
    """

    Function to calculate curve fit values

    """

    c = scale * np.exp(growth * (t - 1960))
    return c


# Calling the file read function
df_access_to_elec, df_access_to_elect = read_data("Access to electricity.csv")
df_agri_methane, df_agri_methanet = read_data("Agricultural Methane Emission.csv")
df_non_agri_methane, df_non_agri_methanet = read_data("Non Agricultural Methane Emission.csv")


# Doing curve fit
param, cov = opt.curve_fit(
    curve_function, df_access_to_elect["year"], df_access_to_elect["India"], p0=[4e8, 0.1]
)
sigma = np.sqrt(np.diag(cov))

# Error
low, up = err.err_ranges(df_access_to_elect["year"], curve_function, param, sigma)
df_access_to_elect["fit_value"] = curve_function(df_access_to_elect["year"], *param)

# Plotting the Access to electricity (% of population) values for India
plt.figure()
plt.title("Access to electricity (% of population) - India")
plt.plot(df_access_to_elect["year"], df_access_to_elect["India"], label="data")
plt.plot(df_access_to_elect["year"], df_access_to_elect["fit_value"], c="red", label="fit")
plt.fill_between(df_access_to_elect["year"], low, up, alpha=0.5)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel("Access to electricity")
plt.savefig("Access to electricity.png", dpi=500, bbox_inches="tight")
plt.show()

# Curve ft for UK
param, cov = opt.curve_fit(
    curve_function, df_access_to_elect["year"], df_access_to_elect["UK"], p0=[4e8, 0.1]
)
sigma = np.sqrt(np.diag(cov))
print(*param)
lower, upper = err.err_ranges(df_access_to_elect["year"], curve_function, param, sigma)
df_access_to_elect["fit_value"] = curve_function(df_access_to_elect["year"], *param)

# Plotting
plt.figure()
plt.title("UK Access to electricity Predction For 2030")
predict_year = np.arange(1980, 2030)
predict_India = curve_function(predict_year, *param)
plt.plot(df_access_to_elect["year"], df_access_to_elect["UK"], label="data")
plt.plot(predict_year, predict_India, label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Access to electricity")
plt.savefig("Access to electricity UK_Predicted.png", dpi=500, bbox_inches="tight")
plt.show()

# Non Agricultural methane emissions (% of total) for India
param, cov = opt.curve_fit(
    curve_function, df_non_agri_methanet["year"], df_non_agri_methanet["India"], p0=[4e8, 0.1]
)
sigma = np.sqrt(np.diag(cov))
print(*param)
lower, upper = err.err_ranges(df_non_agri_methanet["year"], curve_function, param, sigma)

# Plotting
df_non_agri_methanet["fit_value"] = curve_function(df_non_agri_methanet["year"], *param)
plt.figure()
plt.title("Non Agricultural methane emissions (% of total) - India")
plt.plot(df_non_agri_methanet["year"], df_non_agri_methanet["India"], label="data")
plt.plot(df_non_agri_methanet["year"], df_non_agri_methanet["fit_value"], c="red", label="fit")
plt.fill_between(df_non_agri_methanet["year"], lower, upper, alpha=0.5)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel("Non Agricultural methane emissions")
plt.savefig("Non Agricultural methane emissions India.png", dpi=500, bbox_inches="tight")
plt.show()



# Agricultural methane emissions (% of total) for India
param, cov = opt.curve_fit(
    curve_function, df_agri_methanet["year"], df_agri_methanet["India"], p0=[4e8, 0.1]
)
sigma = np.sqrt(np.diag(cov))
print(*param)
lower, upper = err.err_ranges(df_agri_methanet["year"], curve_function, param, sigma)

# Plotting
df_agri_methanet["fit_value"] = curve_function(df_agri_methanet["year"], *param)
plt.figure()
plt.title("Agricultural methane emissions (% of total) - India")
plt.plot(df_agri_methanet["year"], df_agri_methanet["India"], label="data")
plt.plot(df_agri_methanet["year"], df_agri_methanet["fit_value"], c="red", label="fit")
plt.fill_between(df_agri_methanet["year"], lower, upper, alpha=0.5)
plt.legend()
plt.xlim(1990, 2019)
plt.xlabel("Year")
plt.ylabel("Agricultural methane emissions")
plt.savefig("Agricultural methane emissions India.png", dpi=500, bbox_inches="tight")
plt.show()


# Plotting the predicted values for India Access to electricity
plt.figure()
plt.title("India Access to electricity prediction")
predict_year = np.arange(1980, 2030)
predict_India = curve_function(predict_year, *param)
plt.plot(df_access_to_elect["year"], df_access_to_elect["India"], label="data")
plt.plot(predict_year, predict_India, label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Access to electricity")
plt.savefig("Access to electricity India_Predicted.png", dpi=500, bbox_inches="tight")
plt.show()

# Predicted values for UK Non Agricultural methane emissions
param, cov = opt.curve_fit(
    curve_function, df_non_agri_methanet["year"], df_non_agri_methanet["UK"], p0=[4e8, 0.1]
)
sigma = np.sqrt(np.diag(cov))
print(*param)
lower, upper = err.err_ranges(df_non_agri_methanet["year"], curve_function, param, sigma)

# Plotting the predicted values for UK total energy use
df_non_agri_methanet["fit_value"] = curve_function(df_non_agri_methanet["year"], *param)
plt.figure()
plt.title("Non Agricultural methane emissions prediction - UK")
predict_year = np.arange(1980, 2030)
predict_India = curve_function(predict_year, *param)
plt.plot(df_non_agri_methanet["year"], df_non_agri_methanet["UK"], label="data")
plt.plot(predict_year, predict_India, label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Non Agricultural methane emissions (% of total)")
plt.savefig("Non Agricultural methane emissions Prediction_UK.png", dpi=500, bbox_inches="tight")
plt.show()


# Predicted values for UK Agricultural methane emissions
param, cov = opt.curve_fit(
    curve_function, df_agri_methanet["year"], df_agri_methanet["UK"], p0=[4e8, 0.1]
)
sigma = np.sqrt(np.diag(cov))
print(*param)
lower, upper = err.err_ranges(df_agri_methanet["year"], curve_function, param, sigma)

# Plotting the predicted values for UK total energy use
df_agri_methanet["fit_value"] = curve_function(df_agri_methanet["year"], *param)
plt.figure()
plt.title("Non Agricultural methane emissions prediction - UK")
predict_year = np.arange(1980, 2030)
predict_India = curve_function(predict_year, *param)
plt.plot(df_agri_methanet["year"], df_agri_methanet["UK"], label="data")
plt.plot(predict_year, predict_India, label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Agricultural methane emissions (% of total)")
plt.savefig("Agricultural methane emissions Prediction_UK.png", dpi=500, bbox_inches="tight")
plt.show()

# Clustering
India = pd.DataFrame()
India["agri_methane"] = df_agri_methanet["India"]
India["non_agri_methane"] = df_non_agri_methanet["India"]
kmean = cluster.KMeans(n_clusters=2).fit(India)
label = kmean.labels_
plt.scatter(India["agri_methane"], India["non_agri_methane"], c=label, cmap="jet")
plt.title("Non Agricultural methane emissions vs Agricultural methane emissions -India")
c = kmean.cluster_centers_

# Plotting Scatter CO2 vs Renewable India
for t in range(2):
    xc, yc = c[t, :]
    plt.plot(xc, yc, "ok", markersize=8)
    plt.savefig("Scatter_UK_India_Agricultural methane emissions.png", dpi=500, bbox_inches="tight")
plt.figure()

# Plotting Scatter UK and India CO2 Agricultural methane emissions
df_agri_methanet = df_agri_methanet.iloc[:, 1:3]
kmean = cluster.KMeans(n_clusters=2).fit(df_access_to_elect)
label = kmean.labels_
plt.scatter(df_agri_methanet["UK"], df_agri_methanet["India"], c=label, cmap="jet")
plt.title("UK and India - Agricultural methane emissions")
c = kmean.cluster_centers_
plt.savefig("Scatter_UK_India_Agricultural methane emissions.png", dpi=500, bbox_inches="tight")
plt.show()



