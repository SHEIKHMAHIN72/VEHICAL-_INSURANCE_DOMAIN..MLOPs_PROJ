############################################################
# VEHICLE INSURANCE DATA - FULL EDA VISUALIZATION SCRIPT
############################################################

"""
Purpose:
Complete Exploratory Data Analysis visualization pipeline
for Vehicle Insurance Response Prediction dataset.
"""

############################################################
# IMPORT LIBRARIES
############################################################

print("\n========== IMPORTING LIBRARIES ==========\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

sns.set(style="whitegrid")

############################################################
# LOAD DATASET
############################################################

print("\n========== LOADING DATASET ==========\n")

df = pd.read_csv("notebook/data.csv")

print("Dataset Shape:", df.shape)

############################################################
# CREATE FOLDER TO SAVE GRAPHS (OPTIONAL)
############################################################

import os

if not os.path.exists("graphs"):
    os.makedirs("graphs")

############################################################
# TARGET VARIABLE DISTRIBUTION
############################################################

print("\n========== TARGET VARIABLE DISTRIBUTION ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(x="Response", data=df)

plt.title("Customer Response Distribution")
plt.savefig("graphs/response_distribution.png")

plt.show()


############################################################
# AGE DISTRIBUTION
############################################################

print("\n========== AGE DISTRIBUTION ==========\n")

plt.figure(figsize=(8,5))

sns.histplot(df["Age"], bins=25, kde=True)

plt.title("Age Distribution")
plt.savefig("graphs/age_distribution.png")

plt.show()


############################################################
# AGE VS PREMIUM SCATTER
############################################################

print("\n========== AGE VS PREMIUM ==========\n")

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Age",
    y="Annual_Premium",
    data=df,
    alpha=0.4
)

plt.title("Age vs Annual Premium")

plt.savefig("graphs/age_vs_premium.png")

plt.show()


############################################################
# GENDER DISTRIBUTION
############################################################

print("\n========== GENDER DISTRIBUTION ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(x="Gender", data=df)

plt.title("Gender Distribution")

plt.savefig("graphs/gender_distribution.png")

plt.show()


############################################################
# GENDER VS RESPONSE
############################################################

print("\n========== GENDER VS RESPONSE ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Gender",
    hue="Response",
    data=df
)

plt.title("Gender vs Response")

plt.savefig("graphs/gender_vs_response.png")

plt.show()


############################################################
# DRIVING LICENSE DISTRIBUTION
############################################################

print("\n========== DRIVING LICENSE DISTRIBUTION ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Driving_License",
    data=df
)

plt.title("Driving License Distribution")

plt.savefig("graphs/license_distribution.png")

plt.show()


############################################################
# PREVIOUSLY INSURED DISTRIBUTION
############################################################

print("\n========== PREVIOUSLY INSURED ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Previously_Insured",
    data=df
)

plt.title("Previously Insured Customers")

plt.savefig("graphs/previously_insured.png")

plt.show()


############################################################
# VEHICLE AGE DISTRIBUTION
############################################################

print("\n========== VEHICLE AGE DISTRIBUTION ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Vehicle_Age",
    data=df
)

plt.title("Vehicle Age Distribution")

plt.xticks(rotation=45)

plt.savefig("graphs/vehicle_age_distribution.png")

plt.show()


############################################################
# VEHICLE AGE VS RESPONSE
############################################################

print("\n========== VEHICLE AGE VS RESPONSE ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Vehicle_Age",
    hue="Response",
    data=df
)

plt.title("Vehicle Age vs Response")

plt.xticks(rotation=45)

plt.savefig("graphs/vehicle_age_vs_response.png")

plt.show()


############################################################
# VEHICLE DAMAGE DISTRIBUTION
############################################################

print("\n========== VEHICLE DAMAGE DISTRIBUTION ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Vehicle_Damage",
    data=df
)

plt.title("Vehicle Damage Distribution")

plt.savefig("graphs/vehicle_damage_distribution.png")

plt.show()


############################################################
# VEHICLE DAMAGE VS RESPONSE
############################################################

print("\n========== VEHICLE DAMAGE VS RESPONSE ==========\n")

plt.figure(figsize=(6,4))

sns.countplot(
    x="Vehicle_Damage",
    hue="Response",
    data=df
)

plt.title("Vehicle Damage vs Response")

plt.savefig("graphs/vehicle_damage_vs_response.png")

plt.show()


############################################################
# PREMIUM DISTRIBUTION
############################################################

print("\n========== PREMIUM DISTRIBUTION ==========\n")

plt.figure(figsize=(8,5))

sns.histplot(
    df["Annual_Premium"],
    bins=30,
    kde=True
)

plt.title("Annual Premium Distribution")

plt.savefig("graphs/premium_distribution.png")

plt.show()


############################################################
# PREMIUM BOXPLOT (OUTLIERS)
############################################################

print("\n========== PREMIUM BOXPLOT ==========\n")

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Annual_Premium"]
)

plt.title("Premium Outliers Detection")

plt.savefig("graphs/premium_boxplot.png")

plt.show()


############################################################
# CORRELATION HEATMAP
############################################################

print("\n========== CORRELATION HEATMAP ==========\n")

plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.savefig("graphs/correlation_heatmap.png")

plt.show()


############################################################
# VINTAGE DISTRIBUTION
############################################################

print("\n========== VINTAGE DISTRIBUTION ==========\n")

plt.figure(figsize=(8,5))

sns.histplot(
    df["Vintage"],
    bins=30,
    kde=True
)

plt.title("Customer Vintage Distribution")

plt.savefig("graphs/vintage_distribution.png")

plt.show()


############################################################
# AGE VS RESPONSE BOXPLOT
############################################################

print("\n========== AGE VS RESPONSE ==========\n")

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Response",
    y="Age",
    data=df
)

plt.title("Age vs Response")

plt.savefig("graphs/age_vs_response.png")

plt.show()


############################################################
# FINAL MESSAGE
############################################################

print("\n========== ALL VISUALIZATIONS COMPLETED ==========\n")
print("Graphs saved inside: /graphs folder 📊")