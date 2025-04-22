# Q1
# In a school of 200 students, 120 like math and 80 like science. 50 like both math and science.
# What is the probability that a randomly selected student likes both subjects?
students_total = 200
both = 50
prob_both = both / students_total
print("Q1: Probability that a student likes both subjects:", prob_both)

# Q2
# A factory produces 60% of items from Machine A and 40% from Machine B.
# Machine A has a 5% defect rate, and Machine B has a 10% defect rate.
# If an item is found defective, what is the probability it came from Machine B?
P_A = 0.6
P_B = 0.4
P_D_given_A = 0.05
P_D_given_B = 0.10
P_D = P_A * P_D_given_A + P_B * P_D_given_B
P_B_given_D = (P_B * P_D_given_B) / P_D
print("Q2: Probability that a defective item came from Machine B:", P_B_given_D)

# Q3
# A basket contains 6 apples and 4 oranges. One fruit is picked at random.
# What is the probability it is an orange?
total_fruits = 6 + 4
prob_orange = 4 / total_fruits
print("Q3: Probability that the fruit is an orange:", prob_orange)

# Q4
# A test is 90% accurate. 1 in 100 people has a certain condition.
# A person tests positive. What is the probability they actually have the condition?
P_C = 0.01
P_Pos_given_C = 0.9
P_Pos_given_not_C = 0.1
P_not_C = 0.99
P_Pos = P_C * P_Pos_given_C + P_not_C * P_Pos_given_not_C
P_C_given_Pos = (P_C * P_Pos_given_C) / P_Pos
print("Q4: Probability that a person has the condition given a positive test:", P_C_given_Pos)

# Q5
# In a company of 300 employees, 180 can code in Python, 100 can code in Java, and 50 can code in both languages.
# What is the probability that an employee can code in Python or Java?
total_emp = 300
python = 180
java = 100
both_lang = 50
either = (python + java - both_lang) / total_emp
print("Q5: Probability that an employee can code in Python or Java:", either)

# Q6
# The probability that it rains today is 0.6. The probability that you carry an umbrella given it rains is 0.9.
# What is the probability that it rains, and you carry an umbrella?
P_rain = 0.6
P_umbrella_given_rain = 0.9
P_rain_and_umbrella = P_rain * P_umbrella_given_rain
print("Q6: Probability that it rains and you carry an umbrella:", P_rain_and_umbrella)

# Q7
# In a class, 70% of students passed the exam. Of those who passed, 80% studied the night before.
# What is the probability a student studied the night before given they passed the exam?
P_pass = 0.7
P_study_given_pass = 0.8
print("Q7: Probability that a student studied the night before given they passed:", P_study_given_pass)

# Q8
# You roll two fair dice. What is the probability that the sum is 7?
favorable = 6  # (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
total_outcomes = 6 * 6
prob_sum_7 = favorable / total_outcomes
print("Q8: Probability of getting a sum of 7:", prob_sum_7)

# Q9
# A bag contains 3 red, 2 green, and 5 blue balls. One ball is drawn randomly.
# What is the probability that it is either red or green?
total_balls = 3 + 2 + 5
prob_red_or_green = (3 + 2) / total_balls
print("Q9: Probability that the ball is red or green:", prob_red_or_green)

# Q10
# In a city, 30% of the population uses public transport. Of these, 60% are students.
# What is the probability that a randomly selected person is a student who uses public transport?
P_transport = 0.3
P_student_given_transport = 0.6
P_student_and_transport = P_transport * P_student_given_transport
print("Q10: Probability that a person is a student who uses public transport:", P_student_and_transport)


# Q11. In a survey, 40% of participants like tea, 50% like coffee, and 20% like both.
# What is the probability a person likes at least one of the two drinks?
P_tea = 0.40
P_coffee = 0.50
P_both = 0.20
P_at_least_one = P_tea + P_coffee - P_both
print("Q11 Answer:", P_at_least_one)

# Q12. A fair coin is flipped three times. What is the probability of getting exactly two heads?
from math import comb
total_outcomes = 2 ** 3
favorable = comb(3, 2)
prob = favorable / total_outcomes
print("Q12 Answer:", prob)

# Q13. If a customer made a second purchase, what is the probability that they provided positive feedback?
P_feedback = 0.50
P_second_given_feedback = 0.60
# Using Bayes' Theorem: P(feedback | second) = (P(second | feedback) * P(feedback)) / P(second)
P_second = P_second_given_feedback * P_feedback
P_feedback_given_second = (P_second_given_feedback * P_feedback) / P_second
print("Q13 Answer:", P_feedback_given_second)

# Q14. Probability that a person likes vegetarian or non-vegetarian food
P_veg = 0.40
P_nonveg = 0.35
P_both_foods = 0.20
P_either = P_veg + P_nonveg - P_both_foods
print("Q14 Answer:", P_either)

# Q15. Probability that a customer is both a repeat buyer and a member of the loyalty program
repeat_buyers = 300
loyalty_repeat = 200
total_customers = 1000
prob = loyalty_repeat / total_customers
print("Q15 Answer:", prob)

# Q16. Probability of lower heart disease risk
P_exercise = 0.60
P_lower_given_exercise = 0.90
P_lower_given_no_exercise = 0.10
P_no_exercise = 1 - P_exercise
P_lower = (P_exercise * P_lower_given_exercise) + (P_no_exercise * P_lower_given_no_exercise)
print("Q16 Answer:", P_lower)

# Q17. Probability student is enrolled in CS or Math
both = 150
only_cs = 250
only_math = 100
total_students = 500
either = (both + only_cs + only_math) / total_students
print("Q17 Answer:", either)

# Q18. Probability of promotion given employee exceeds expectations
P_exceeds = 0.30
P_promotion_given_exceeds = 0.50
# P(promotion ∩ exceeds) = P(promotion | exceeds) * P(exceeds)
joint = P_promotion_given_exceeds * P_exceeds
print("Q18 Answer:", joint)

# Q19. Probability that a person takes the quiz and shares results
P_survey_given_quiz = 0.25
P_share_given_survey = 0.40
P_quiz_and_share = P_survey_given_quiz * P_share_given_survey
print("Q19 Answer:", P_quiz_and_share)

# Q20. Probability that a person owns either a smartphone or tablet
P_phone = 0.60
P_tablet = 0.40
P_both_devices = 0.25
P_either_device = P_phone + P_tablet - P_both_devices
print("Q20 Answer:", P_either_device)

# Q21. Probability that a customer both visits the website and makes a purchase
P_visit_given_offer = 0.70
P_purchase_given_visit = 0.50
P_both_actions = P_visit_given_offer * P_purchase_given_visit
print("Q21 Answer:", P_both_actions)

# Q22. Probability that an employee is both satisfied and highly productive
satisfied = 700
satisfied_and_productive = 500
total_employees = 1000
P_both = satisfied_and_productive / total_employees
print("Q22 Answer:", P_both)

# Q23. Probability customer prefers product A or B
P_A = 0.50
P_B = 0.30
P_both = 0.20
P_either = P_A + P_B - P_both
print("Q23 Answer:", P_either)

# Q24. Probability student excels in Math or Science
P_math = 0.40
P_science = 0.30
P_both = 0.15
P_either = P_math + P_science - P_both
print("Q24 Answer:", P_either)
