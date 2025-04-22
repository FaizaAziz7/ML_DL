import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. A fair six-sided dice is rolled twice. Define a random variable representing the sum of the outcomes.
#    What type of distribution does it follow?
#    --> It follows a discrete distribution with values from 2 to 12, modeled as a discrete uniform convolution.
dice_rolls = [i + j for i in range(1, 7) for j in range(1, 7)]
sum_distribution = np.bincount(dice_rolls)[2:]
print("1. Sum distribution of two dice rolls (2 to 12):", sum_distribution)
# np.bincount() counts how many times each number occurs


# 2. A weather station records the amount of rainfall each day.
#    What kind of probability distribution best models this data and why?
#    --> Gamma or Exponential distribution (positive, skewed data)
rainfall_data = stats.gamma.rvs(a=2.0, scale=3.0, size=1000)
print("2. Simulated rainfall data (gamma):", rainfall_data[:5])

# 3. A call center receives an average of 5 calls per hour.
#    Model this situation using an appropriate discrete distribution.
#    --> Poisson distribution
calls = stats.poisson.rvs(mu=5, size=10)
print("3. Poisson distributed call data:", calls)

# 4. The lifespan of a light bulb is measured in hours.
#    Which type of probability distribution can represent this data?
#    --> Exponential or Normal (based on context)
lifespan_data = stats.expon.rvs(scale=1000, size=1000)
print("4. Simulated light bulb lifespan data (exponential):", lifespan_data[:5])

# 5. A student updates belief from 70% to 90% after studying. What probability concept is used?
#    --> Bayesian updating
prior = 0.7
likelihood = 0.9
posterior = prior * likelihood / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))
print("5. Posterior probability using Bayesian update:", posterior)

# 6. A survey measures height of adult males. What distribution is suitable?
#    --> Normal distribution
height_data = np.random.normal(loc=175, scale=10, size=1000)
print("6. Simulated height data (normal):", height_data[:5])

# 7. In a game, 1 in 4 chance of winning. Played 10 times. Which distribution?
#    --> Binomial distribution
wins = np.random.binomial(n=10, p=0.25)
print("7. Number of wins out of 10 games (binomial):", wins)

# 8. A machine produces items with some defect probability.
#    --> Binomial distribution to model number of defectives
defects = np.random.binomial(n=50, p=0.1, size=1)
print("8. Number of defective items out of 50:", defects[0])

# 9. Monitoring network traffic (continuous flow). Which distribution?
#    --> Normal or Log-Normal depending on skew
traffic_data = np.random.lognormal(mean=3, sigma=0.5, size=1000)
print("9. Simulated network traffic data (log-normal):", traffic_data[:5])

# 10. A person updates rain chance from 30% to 60% after a report.
#     --> Bayesian updating
prior_rain = 0.3
evidence = 0.6
posterior_rain = prior_rain * evidence / ((prior_rain * evidence) + ((1 - prior_rain) * (1 - evidence)))
print("10. Posterior rain probability after update:", posterior_rain)

# 11. Hospital measures patient blood pressure (continuous).
#     --> Normal distribution
bp_data = np.random.normal(loc=120, scale=15, size=1000)
print("11. Simulated blood pressure data (normal):", bp_data[:5])

# 12. Company measures ad clicks per day.
#     --> Poisson distribution (count data)
ad_clicks = stats.poisson.rvs(mu=2, size=30)
print("12. Simulated ad clicks per day (poisson):", ad_clicks)
