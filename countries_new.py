import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats


# ====================
# FUNCTION DEFINITIONS
# ====================
def import_data():
    # Create a dictionary of data frames
    raw_data = {
        "CA": pd.read_csv("data/CAvideos.csv"),
        "DE": pd.read_csv("data/DEvideos.csv"),
        "FR": pd.read_csv("data/FRvideos.csv"),
        "UK": pd.read_csv("data/GBvideos.csv"),
        "IN": pd.read_csv("data/INvideos.csv"),
        "US": pd.read_csv("data/USvideos.csv"),
    }

    return raw_data


def find_unclean_data(data):
    # Print key metrics to help identify cleaning and formatting requirements
    print(data["CA"].columns)
    print(data["CA"].dtypes)
    print(data["CA"].isnull().any())


def clean_data(data):
    # Iterate through each dataframe and clean it's data
    for df in data.values():
        # Filter out any unusable rows using heuristics
        df = df[
            (df["views"] >= 0)
            & (df["likes"] >= 0)
            & (df["dislikes"] >= 0)
            & (df["category_id"] >= 0)
        ]

        # Convert the trending date from a string to a datetime object
        df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m")

    return data


def create_additional_columns(data):
    # Iterate through each dataframe and add key columns
    for df in data.values():
        # Add columns for likes + dislikes and comments engagement rate
        df["engagement_rate_likes"] = (df["likes"] + df["dislikes"]) / (df["views"])
        df["engagement_rate_comments"] = df["comment_count"] / df["views"]

    return data


def generate_descriptive_statistics(data):
    statistics = {}
    metrics = [
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "engagement_rate_likes",
        "engagement_rate_comments",
    ]

    for country, df in data.items():
        statistics[country] = {}
        for column in metrics:
            statistics[country][column] = df[column].describe()
    print(statistics)


def analyse_views(data):
    # Calculate the total number of views on videos for each day
    views_by_date = {}
    for country, df in data.items():
        views_by_date[country] = df[["trending_date", "views"]].groupby("trending_date").sum()
        views_by_date[country]["country"] = country

    # Concatenate all dataframes into one large stacked data frame
    combined = pd.concat([df for df in views_by_date.values()])

    # Move trending_date data from the index into a column and generates autoincremented indexes
    combined = combined.reset_index()

    # Create numerical representations of country code and trending date for use in correlation and anova calculations
    combined["country_cat"] = combined["country"].astype("category").cat.codes
    combined["trending_date_cat"] = (
        combined["trending_date"].astype("category").cat.codes
    )

    # Calculate correlations between country/views and date/views
    correlations_country = combined[["country_cat", "views"]].corr()
    correlations_time = combined[["trending_date_cat", "views"]].corr()
    print(correlations_country)
    print(correlations_time)

    # Calculate one-way anova for views by date, using the country as the independent variable
    anova = stats.f_oneway(
        combined["country_cat"], combined["trending_date_cat"], combined["views"]
    )
    print(anova)

    # Visualize the views per day over time by country
    for country, df in views_by_date.items():
        df = df.reset_index()
        df.sort_values(by=["trending_date"])
        plt.plot(df["trending_date"], df["views"], label=country)
    plt.title("Video views-per-day, displayed by country")
    plt.xlabel("Date")
    plt.ylabel("Views")
    plt.legend()
    plt.show()

    # Visualise data with a linear regresssion
    for country, df in views_by_date.items():
        df = df.reset_index()
        df.sort_values(by=["trending_date"])
        df["trending_date_cat"] = df["trending_date"].astype("category").cat.codes
        linear_fit = np.polyfit(df["trending_date_cat"], df["views"], 1)
        linear_fit_fn = np.poly1d(linear_fit)
        plt.plot(
            df["trending_date"],
            df["views"],
            "rx",
            df["trending_date"],
            linear_fit_fn(df["trending_date_cat"]),
        )
        plt.title("Video views-per-day, {}".format(country))
        plt.xlabel("Date")
        plt.ylabel("Views")
        plt.show()


def analyse_engagement(data):
    metrics = ["trending_date", "engagement_rate_likes", "engagement_rate_comments"]

    # Calculate the mean engagement on videos for each day
    engagement_by_date = {}
    for country, df in data.items():
        engagement_by_date[country] = df[metrics].groupby(["trending_date"]).mean()
        engagement_by_date[country]["country"] = country

    # Concatenate all dataframes into one large stacked data frame
    combined = pd.concat([df for df in engagement_by_date.values()])

    # Move trending_date data from the index into a column and generates autoincremented indexes
    combined = combined.reset_index()

    # Create numerical representations of country code and trending date for use in correlation and anova calculations
    combined["country_cat"] = combined["country"].astype("category").cat.codes
    combined["trending_date_cat"] = (
        combined["trending_date"].astype("category").cat.codes
    )

    # Calculate correlations between country/engagement and date/engagement
    correlations_country = {}
    correlations_time = {}
    for column in metrics[1:]:
        correlations_country[column] = combined[["country_cat", column]].corr()
        correlations_time[column] = combined[["trending_date_cat", column]].corr()
    print(correlations_country)
    print(correlations_time)

    # Calculate correlations between likes/dislikes engagement and comments engagement
    correlations_engagement_1_2 = combined[
        ["engagement_rate_likes", "engagement_rate_comments"]
    ].corr()
    print(correlations_engagement_1_2)

    # Calculate one-way anova for likes/dislikes engagement by date, using the country as the independent variable
    anova_likes_by_date = stats.f_oneway(
        combined["country_cat"],
        combined["trending_date_cat"],
        combined["engagement_rate_likes"],
    )
    print(anova_likes_by_date)

    # Calculate one-way anova for comments engagement by date, using the country as the independent variable
    anova_comments_by_date = stats.f_oneway(
        combined["country_cat"],
        combined["trending_date_cat"],
        combined["engagement_rate_comments"],
    )
    print(anova_comments_by_date)

    # Calculate one-way anova for likes engagement against comments engagement, using the country as the independent variable
    anova_likes_vs_comments = stats.f_oneway(
        combined["country_cat"],
        combined["engagement_rate_likes"],
        combined["engagement_rate_comments"],
    )
    print(anova_likes_vs_comments)

    # Visualize the likes/dislikes engagement per day over time by country
    for country, df in engagement_by_date.items():
        df = df.reset_index()
        df.sort_values(by=["trending_date"])
        plt.plot(df["trending_date"], df["engagement_rate_likes"], label=country)
    plt.title("Video engagement rate per day (likes/dislikes), displayed by country")
    plt.xlabel("Date")
    plt.ylabel("Likes/dislikes engagement rate")
    plt.legend()
    plt.show()

    # Visualize the likes/dislikes engagement for each country individually, with a cubic regression line and moving average applied to the "raw" data
    for country, df in engagement_by_date.items():
        df = df.reset_index()
        df.sort_values(by=["trending_date"])
        df["trending_date_cat"] = df["trending_date"].astype("category").cat.codes
        
        cubic = np.polyfit(
            df["trending_date_cat"], df["engagement_rate_likes"], 3
        )
        cubic_fn = np.poly1d(cubic)

        plt.plot(
            df["trending_date"],
            df["engagement_rate_likes"].rolling(20).mean(),
            "rx",
            df["trending_date"],
            cubic_fn(df["trending_date_cat"])
        )
        plt.title("Video engagement rate per day (likes/dislikes), {}".format(country))
        plt.xlabel("Date")
        plt.ylabel("Likes/dislikes engagement rate")
        plt.show()

    # Visualize the comments engagement per day over time by country
    for country, df in engagement_by_date.items():
        df = df.reset_index()
        df.sort_values(by=["trending_date"])
        plt.plot(df["trending_date"], df["engagement_rate_comments"], label=country)
    plt.title("Video engagement rate per day (comments), displayed by country")
    plt.xlabel("Date")
    plt.ylabel("Comments engagement rate")
    plt.legend()
    plt.show()

    # Visualize the comments engagement for each country individually, with a cubic regression line and moving average applied to the "raw" data
    for country, df in engagement_by_date.items():
        df = df.reset_index()
        df.sort_values(by=["trending_date"])
        df["trending_date_cat"] = df["trending_date"].astype("category").cat.codes
        
        cubic = np.polyfit(
            df["trending_date_cat"], df["engagement_rate_comments"], 3
        )
        cubic_fn = np.poly1d(cubic)

        plt.plot(
            df["trending_date"],
            df["engagement_rate_comments"].rolling(20).mean(),
            "rx",
            df["trending_date"],
            cubic_fn(df["trending_date_cat"])
        )
        plt.title("Video engagement rate per day (comments), {}".format(country))
        plt.xlabel("Date")
        plt.ylabel("Comments engagement rate")
        plt.show()


# ==============
# FUNCTION CALLS
# ==============
raw_data = import_data()

find_unclean_data(raw_data)
clean_data = clean_data(raw_data)

base_data = create_additional_columns(clean_data)

generate_descriptive_statistics(base_data)

analyse_views(base_data)
analyse_engagement(base_data)
