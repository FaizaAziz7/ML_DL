# import numpy as np
#
# # 1. Create a NumPy array from a Python list containing integers 1 to 5.
# arr1 = np.array([1, 2, 3, 4, 5])
# print(arr1)
#
# # 2. Create a NumPy array using arange() from 0 to 10.
# arr2 = np.arange(0, 11)
# print(arr2)
#
# # 3. Generate a 3x3 array of random integers between 1 and 100.
# arr3 = np.random.randint(1, 101, size=(3, 3))
# print(arr3)
#
# # 4. Explain the difference between np.zeros(), np.ones(), and np.empty().
# zeros = np.zeros((2, 2))  # All elements are 0
# ones = np.ones((2, 2))    # All elements are 1
# empty = np.empty((2, 2))  # Uninitialized (random) values
# print("Zeros:\n", zeros)
# print("Ones:\n", ones)
# print("Empty:\n", empty)
#
# # 5. Create a NumPy array with 5 random floating-point numbers.
# arr5 = np.random.rand(5)
# print(arr5)
#
# # 6. What does reshape() do? Provide an example.
# arr6 = np.arange(9).reshape((3, 3))
# print(arr6)
#
# # 7. Find maximum value in an array.
# max_val = np.max(arr6)
# print(max_val)
#
# # 8. Find the index of the minimum value in a NumPy array.
# min_index = np.argmin(arr6)
# print(min_index)
#
# # 9. Use of shape attribute.
# print("Shape of arr6:", arr6.shape)
#
# # 10. Find number of elements in a NumPy array.
# print("Size of arr6:", arr6.size)
#
# # 11. ndim attribute tells number of dimensions.
# print("Dimensions of arr6:", arr6.ndim)
#
# # 12. dtype attribute shows data type of elements.
# print("Data type of arr6:", arr6.dtype)
#
# # 13. Create a deep copy of a NumPy array.
# arr_copy = arr6.copy()
# print("Copied array:\n", arr_copy)
#
# # 14. Append an element to the end of a NumPy array.
# arr7 = np.array([1, 2, 3])
# arr7_appended = np.append(arr7, 4)
# print(arr7_appended)
#
# # 15. Insert an element at index 2.
# arr7_inserted = np.insert(arr7, 2, 99)
# print(arr7_inserted)
#
# # 16. Sort a NumPy array in ascending order.
# arr8 = np.array([4, 1, 3, 2])
# arr8_sorted = np.sort(arr8)
# print(arr8_sorted)
#
# # 17. Sort a NumPy array in descending order.
# arr8_sorted_desc = np.sort(arr8)[::-1]
# print(arr8_sorted_desc)
#
# # 18. Remove the element at index 3.
# arr9 = np.array([10, 20, 30, 40, 50])
# arr9_removed = np.delete(arr9, 3)
# print(arr9_removed)
#
# # 19. Delete multiple elements from a NumPy array at once.
# arr9_deleted = np.delete(arr9, [1, 3])
# print(arr9_deleted)
#
# # 20. Combine two arrays using concatenate.
# a = np.array([1, 2])
# b = np.array([3, 4])
# combined = np.concatenate((a, b))
# print(combined)
#
# # 21. Vertically stack two arrays.
# v1 = np.array([[1, 2], [3, 4]])
# v2 = np.array([[5, 6]])
# v_stacked = np.vstack((v1, v2))
# print(v_stacked)
#
# # 22. Horizontally stack two arrays.
# h1 = np.array([[1], [2]])
# h2 = np.array([[3], [4]])
# h_stacked = np.hstack((h1, h2))
# print(h_stacked)
#
# # 23. Split a 1D array into 2 equal parts.
# arr10 = np.arange(10)
# split_1d = np.split(arr10, 2)
# print(split_1d)
#
# # 24. Split a 2D array into smaller arrays.
# arr11 = np.arange(16).reshape(4, 4)
# split_2d = np.vsplit(arr11, 2)
# print(split_2d)
#
# # 25. Check if two arrays are equal.
# arr12 = np.array([1, 2, 3])
# arr13 = np.array([1, 2, 3])
# are_equal = np.array_equal(arr12, arr13)
# print("Arrays equal?", are_equal)
#
# # 26. Broadcasting example: add scalar to array.
# arr14 = np.array([1, 2, 3])
# broadcasted = arr14 + 5
# print(broadcasted)
#
# # 27. Multiply two arrays element-wise.
# arr15 = np.array([1, 2, 3])
# arr16 = np.array([4, 5, 6])
# elementwise_product = arr15 * arr16
# print(elementwise_product)
#
# # 28. Reshape to incompatible shape (will raise an error).
# try:
#     arr17 = np.arange(10).reshape(3, 4)
# except ValueError as e:
#     print("Error:", e)
#
# # 29. Retrieve element at row 1, column 2.
# arr18 = np.array([[10, 20, 30], [40, 50, 60]])
# element = arr18[1, 2]
# print(element)
#
# # 30. Find both maximum and minimum values.
# arr19 = np.array([15, 25, 5, 35])
# max_val = np.max(arr19)
# min_val = np.min(arr19)
# print("Max:", max_val, "Min:", min_val)
#

#
# SECOND PART
#
import numpy as np

# 1. Determine which day was the hottest on average
temps = np.random.rand(7, 24) * 15 + 20
hottest_day = np.argmax(np.mean(temps, axis=1))
print("1. Hottest day index:", hottest_day)

# 2. Find which product had the highest total sales
sales = np.random.randint(50, 500, size=(6, 5))
highest_product = np.argmax(np.sum(sales, axis=0))
print("2. Product with highest total sales index:", highest_product)

# 3. Find the day with the fewest visits
visits = np.random.randint(50, 200, size=30)
fewest_day = np.argmin(visits)
print("3. Day with fewest visits index:", fewest_day)

# 4. Find the highest and lowest subject scores
marks = np.random.randint(40, 100, size=8)
highest = np.argmax(marks)
lowest = np.argmin(marks)
print("4. Highest score index:", highest, "Lowest score index:", lowest)

# 5. Find average likes per week
likes = np.random.randint(100, 500, size=28)
avg_likes_per_week = likes.reshape(4, 7).mean(axis=1)
print("5. Average likes per week:", avg_likes_per_week)

# 6. Increase all salaries by 10%
salaries = np.array([35000, 42000, 51000, 60000])
updated_salaries = salaries * 1.10
print("6. Updated salaries:", updated_salaries)

# 7. Group rainfall data by day
rainfall = np.random.rand(30 * 24)
daily_totals = rainfall.reshape(30, 24).sum(axis=1)
wettest_day = np.argmax(daily_totals)
print("7. Wettest day index:", wettest_day)

# 8. Combine two teams' scores
team1 = np.random.randint(0, 100, size=11)
team2 = np.random.randint(0, 100, size=11)
combined = np.vstack((team1, team2))
print("8. Combined teams' scores:\n", combined)

# 9. Identify the student with the highest average
scores = np.random.randint(50, 100, size=(5, 3))
best_student = np.argmax(scores.mean(axis=1))
print("9. Best student index:", best_student)

# 10. Split smart meter data into daily sections
data = np.random.rand(96 * 30)
daily_data = data.reshape(30, 96)
print("10. Smart meter daily data shape:", daily_data.shape)

# 11. Find max temp per day
temps = np.random.rand(30, 24) * 10 + 25
daily_max = temps.max(axis=1)
print("11. Max temperature per day:", daily_max)

# 12. Remove duplicate customer IDs
ids = np.array([1, 2, 3, 2, 1, 4])
unique_ids = np.unique(ids)
print("12. Unique customer IDs:", unique_ids)

# 13. Analyze improvements in weights lifted
weights = np.random.randint(50, 100, size=(7, 5))
improvement = weights[-1] - weights[0]
print("13. Improvement in weights lifted:", improvement)

# 14. Count how many transactions > 10,000
transactions = np.random.randint(5000, 20000, size=100)
high_value = np.sum(transactions > 10000)
print("14. Transactions > 10,000:", high_value)

# 15. Count full 5-star ratings
ratings = np.random.randint(1, 6, size=1000)
five_stars = np.sum(ratings == 5)
print("15. 5-star ratings count:", five_stars)

# 16. Find shortest and longest level durations
durations = np.random.rand(20) * 100
shortest = np.min(durations)
longest = np.max(durations)
print("16. Shortest duration:", shortest, "Longest duration:", longest)

# 17. Calculate attendance percentage
attendance = np.random.randint(0, 2, size=(30,))
attendance_percent = np.mean(attendance) * 100
print("17. Attendance percentage:", attendance_percent)

# 18. Organize visits into weeks and find busiest week
visits = np.random.randint(100, 500, size=60)
weekly_visits = visits.reshape(12, 5)
busiest_week = np.argmax(weekly_visits.sum(axis=1))
print("18. Busiest week index:", busiest_week)

# 19. Fastest, slowest, and average delivery times
deliveries = np.random.rand(100) * 60
fastest = np.min(deliveries)
slowest = np.max(deliveries)
average = np.mean(deliveries)
print("19. Fastest:", fastest, "Slowest:", slowest, "Average:", average)

# 20. Identify top 5 most active students
log_times = np.random.randint(20, 180, size=50)
top5 = np.argsort(log_times)[-5:]
print("20. Top 5 active student indices:", top5)

# 21. Calculate average call time
call_times = np.random.rand(200) * 30
avg_call = np.mean(call_times)
print("21. Average call time:", avg_call)

# 22. Determine best and worst selling books
sales = np.random.randint(0, 1000, size=10)
best = np.argmax(sales)
worst = np.argmin(sales)
print("22. Best book index:", best, "Worst book index:", worst)

# 23. Find most fuel-efficient vehicle
fuel_data = np.random.rand(10) * 10 + 5
efficient = np.argmin(fuel_data)
print("23. Most fuel-efficient vehicle index:", efficient)

# 24. Identify section with most and least stock
stocks = np.random.randint(100, 1000, size=20)
most_stock = np.argmax(stocks)
least_stock = np.argmin(stocks)
print("24. Most stock index:", most_stock, "Least stock index:", least_stock)

# 25. Compare average performance across semesters
performance = np.random.randint(50, 100, size=(100, 4))
semester_avg = np.mean(performance, axis=0)
print("25. Semester average performance:", semester_avg)

# 26. Group hourly foot traffic by day
traffic = np.random.randint(0, 200, size=24 * 30)
daily_traffic = traffic.reshape(30, 24)
peak_times = np.argmax(daily_traffic, axis=1)
print("26. Peak times per day:", peak_times)

# 27. Find most acidic and most alkaline regions
ph_levels = np.random.rand(50) * 7 + 3
acidic = np.argmin(ph_levels)
alkaline = np.argmax(ph_levels)
print("27. Most acidic index:", acidic, "Most alkaline index:", alkaline)

# 28. Summarize heart rate trends over 10 minutes
heart_rate = np.random.randint(60, 180, size=600)
trend_avg = heart_rate.reshape(10, 60).mean(axis=1)
print("28. Heart rate trend averages:", trend_avg)

# 29. Calculate avg delay ignoring on-time trains
delays = np.random.randint(0, 60, size=100)
delays[::10] = 0
avg_delay = np.mean(delays[delays > 0])
print("29. Average delay (excluding on-time):", avg_delay)

# 30. Find most active taxi
trips = np.random.randint(10, 100, size=(7, 20))
most_active = np.argmax(trips.sum(axis=0))
print("30. Most active taxi index:", most_active)
