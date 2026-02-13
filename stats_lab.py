import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1)")
    plt.show()
    return data


def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10)")
    plt.show()
    return data


def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5)")
    plt.show()
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    return np.sum(data) / len(data)


def sample_variance(data):
    mean = sample_mean(data)
    return np.sum((data - mean) ** 2) / (len(data) - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    data = np.sort(data)

    minimum = np.min(data)
    maximum = np.max(data)
    median = np.median(data)

    # Quartiles using "midpoint" method to satisfy test case
    q1 = np.percentile(data, 25, method='midpoint')
    q3 = np.percentile(data, 75, method='midpoint')

    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    x = np.array(x)
    y = np.array(y)

    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    return np.sum((x - mean_x) * (y - mean_y)) / (len(x) - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    return np.array([[var_x, cov_xy],
                     [cov_xy, var_y]])
