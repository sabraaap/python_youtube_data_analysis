import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


def import_data():
    data = {
        "Canada": pd.read_csv("data/CAvideos.csv"),
        "Germany": pd.read_csv("data/DEvideos.csv"),
        "France": pd.read_csv("data/FRvideos.csv"),
        "UK": pd.read_csv("data/GBvideos.csv"),
        "India": pd.read_csv("data/INvideos.csv"),
        "US": pd.read_csv("data/USvideos.csv"),
    }

    return data


def clean_data(data):
    # print(data["canada"].head(0))
    for key, value in data.items():
        # print(value.dtypes)
        # print(key, "\n", value.isnull().any(), "\n\n")
        value["trending_date"] = pd.to_datetime(
            value["trending_date"], format="%y.%d.%m"
        )
        value = value[
            (value["views"] >= 0)
            & (value["likes"] >= 0)
            & (value["dislikes"] >= 0)
            & (value["category_id"] >= 0)
        ]

    return data


def calculate(data):
    for key, value in data.items():
        # print(key)
        value["engagement_rate_1"] = (value["likes"] + value["dislikes"]) / (
            value["views"]
        )
        value["engagement_rate_2"] = value["comment_count"] / value["views"]
        value["year"] = value["trending_date"].dt.year
        value["month"] = value["trending_date"].dt.month_name()
        value["day_of_week"] = value["trending_date"].dt.day_name()
        value["country"] = key

    # print(data)
    return data


def analyse_statistics(data):
    statistics = {}
    metrics = [
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "engagement_rate_1",
        "engagement_rate_2",
    ]

    for key, value in data.items():
        statistics[key] = {}
        for column in metrics:
            statistics[key][column] = value[column].mean()
    print(statistics)


def analyse_views(data):
    data_grouped_views = {
        "Canada": data["Canada"][["views", "trending_date"]]
        .groupby(["trending_date"])
        .sum(),
        "Germany": data["Germany"][["views", "trending_date"]]
        .groupby(["trending_date"])
        .sum(),
        "France": data["France"][["views", "trending_date"]]
        .groupby(["trending_date"])
        .sum(),
        "UK": data["UK"][["views", "trending_date"]].groupby(["trending_date"]).sum(),
        "India": data["India"][["views", "trending_date"]]
        .groupby(["trending_date"])
        .sum(),
        "US": data["US"][["views", "trending_date"]].groupby(["trending_date"]).sum(),
    }

    for key, value in data_grouped_views.items():
        value["country"] = key

    combined = pd.DataFrame()
    for key, value in data_grouped_views.items():
        combined = pd.concat([combined, value])
    combined = combined.reset_index()

    print(combined)

    combined["country_cat"] = combined["country"].astype("category").cat.codes
    combined["trending_date_cat"] = (
        combined["trending_date"].astype("category").cat.codes
    )

    correlations_country = {}
    correlations_time = {}
    correlations_country['views'] = combined[["country_cat", "views"]].corr()
    correlations_time['views'] = combined[["trending_date_cat", "views"]].corr()

    print(correlations_country)
    print(correlations_time)

    anova = stats.f_oneway(
    combined["country_cat"],
    combined["trending_date_cat"],
    combined["views"]
    )

    print(anova)

    for key, value in data_grouped_views.items():
        value.sort_values(by=["trending_date"])
        value = value.reset_index()
        plt.plot(value["trending_date"], value["views"], label=key)

    plt.legend()
    plt.show()

 
    for key, value in data_grouped_views.items():
        value = value.reset_index()
        value.sort_values(by=["trending_date"])
        value["trending_date_cat"] =  value["trending_date"].astype("category").cat.codes
        linear_fit = np.polyfit(value["trending_date_cat"], value["views"],1)
        linear_fit_fn = np.poly1d(linear_fit)
        plt.plot(value["trending_date_cat"], value["views"], "rx", value["trending_date_cat"], linear_fit_fn(value["trending_date_cat"]))
        plt.show()
    

def analyse_engagement(data):

    metrics_engagement_rate = [
        "engagement_rate_1",
        "engagement_rate_2",
        "trending_date",
    ]
    data_grouped_engagement = {
        "Canada": data["Canada"][metrics_engagement_rate]
        .groupby(["trending_date"])
        .mean(),
        "Germany": data["Germany"][metrics_engagement_rate]
        .groupby(["trending_date"])
        .mean(),
        "France": data["France"][metrics_engagement_rate]
        .groupby(["trending_date"])
        .mean(),
        "UK": data["UK"][metrics_engagement_rate].groupby(["trending_date"]).mean(),
        "India": data["India"][metrics_engagement_rate]
        .groupby(["trending_date"])
        .mean(),
        "US": data["US"][metrics_engagement_rate].groupby(["trending_date"]).mean(),
    }

    for key, value in data_grouped_engagement.items():
        value["country"] = key

    combined = pd.DataFrame()
    for key, value in data_grouped_engagement.items():
        combined = pd.concat([combined, value])
    combined = combined.reset_index()

    combined["country_cat"] = combined["country"].astype("category").cat.codes
    combined["trending_date_cat"] = (
        combined["trending_date"].astype("category").cat.codes
    )

    correlations_country = {}
    correlations_time = {}
    for column in metrics_engagement_rate[0:2]:
        correlations_country[column] = combined[["country_cat", column]].corr()
        correlations_time[column] = combined[["trending_date_cat", column]].corr()
    correlations_engagement_1_2 = combined[["engagement_rate_1", "engagement_rate_2"]].corr()

    print(correlations_country)
    print(correlations_time)
    print(correlations_engagement_1_2)

    anova = stats.f_oneway(
    combined["country_cat"],
    combined["trending_date_cat"],
    combined["engagement_rate_1"]
    )
    print(anova)

    anova = stats.f_oneway(
    combined["country_cat"],
    combined["trending_date_cat"],
    combined["engagement_rate_2"]
    )
    print(anova)

    anova = stats.f_oneway(
    combined["country_cat"],
    combined["engagement_rate_1"],
    combined["engagement_rate_2"]
    )
    print(anova)

    for key, value in data_grouped_engagement.items():
        value.sort_values(by=["trending_date"])
        value = value.reset_index()
        plt.plot(value["trending_date"], value["engagement_rate_1"], label=key)

    plt.legend()
    plt.show()

    for key, value in data_grouped_engagement.items():
        value.sort_values(by=["trending_date"])
        value = value.reset_index()
        plt.plot(value["trending_date"], value["engagement_rate_2"], label=key)

    plt.legend()
    plt.show()

    for key, value in data_grouped_engagement.items():
        value = value.reset_index()
        value.sort_values(by=["trending_date"])
        value["trending_date_cat"] =  value["trending_date"].astype("category").cat.codes
        linear_fit = np.polyfit(value["trending_date_cat"], value["engagement_rate_1"],1)
        linear_fit_fn = np.poly1d(linear_fit)
        plt.plot(value["trending_date_cat"], value["engagement_rate_1"].rolling(20).mean(), "rx", value["trending_date_cat"], linear_fit_fn(value["trending_date_cat"]))
        plt.show()
        quadratic_fit = np.polyfit(value["trending_date_cat"], value["engagement_rate_1"],2)
        quadratic_fit_fn = np.poly1d(quadratic_fit)
        plt.plot(value["trending_date_cat"], value["engagement_rate_1"].rolling(20).mean(), "rx", value["trending_date_cat"], quadratic_fit_fn(value["trending_date_cat"]))
        plt.show()
        polyn_10_fit = np.polyfit(value["trending_date_cat"], value["engagement_rate_1"],10)
        polyn_10_fit_fn = np.poly1d(polyn_10_fit)
        plt.plot(value["trending_date_cat"], value["engagement_rate_1"].rolling(20).mean(), "rx", value["trending_date_cat"], polyn_10_fit_fn(value["trending_date_cat"]))
        plt.show()

    for key, value in data_grouped_engagement.items():
        value = value.reset_index()
        value.sort_values(by=["trending_date"])
        value["trending_date_cat"] =  value["trending_date"].astype("category").cat.codes
        linear_fit = np.polyfit(value["trending_date_cat"], value["engagement_rate_2"],1)
        linear_fit_fn = np.poly1d(linear_fit)
        plt.plot(value["trending_date_cat"], value["engagement_rate_2"].rolling(20).mean(), "rx", value["trending_date_cat"], linear_fit_fn(value["trending_date_cat"]))
        plt.show()

data = import_data()
data_2 = clean_data(data)
data_3 = calculate(data_2)
analyse_statistics(data_3)
analyse_views(data_3)
analyse_engagement(data_3)

