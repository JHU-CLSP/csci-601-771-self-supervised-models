# read the csv file 'May 16,2024.csv"
# and aggregate the grades for each student


import csv

csv_file = open('Grades.csv')
csv_reader = csv.reader(csv_file)

import math


def normalize_latesness(lateness):
    """
    format: Hours:Minutes:Seconds
    turn it into days (integers)
    """
    lateness = lateness.split(':')
    days = math.ceil(int(lateness[0]) / 24)
    # normalize above
    return days

def discount(total_late_days, current_late_days, name=""):
    max_days = 7
    if name == "Keerthi Aluvala":
        max_days = 15
        print("Found Keerthi Aluvala!!")
    if total_late_days <= max_days:
        return 1.0
    return (1 - current_late_days * 0.05)

grade_count = {}

log = ""
# average the quiz columsn for each student: index 10 and 20
for idx, row in enumerate(csv_reader):

    # print(row)

    def num(int):
        if row[int] == '':
            return 0
        return float(row[int])


    quiz1 = 59
    quiz2 = 60
    quiz3 = 61

    hw1 = 3
    hw1_max = 4
    hw1_lateness = 5
    hw1_prog = 7
    hw1_prog_max = 8
    hw1_prog_lateness = 9

    hw2 = 10
    hw2_max = 11
    hw2_lateness = 13
    hw2_prog = 14
    hw2_prog_max = 15
    hw2_prog_lateness = 17

    hw3 = 18
    hw3_max = 19
    hw3_lateness = 21
    hw3_prog = 22
    hw3_prog_max = 23
    hw3_prog_lateness = 25

    hw4 = 27
    hw4_max = 28
    hw4_lateness = 30
    hw4_prog = 31
    hw4_prog_max = 32
    hw4_prog_lateness = 34

    hw5 = 35
    hw5_max = 36
    hw5_lateness = 38
    hw5_prog = 39
    hw5_prog_max = 40
    hw5_prog_lateness = 42

    hw6 = 43
    hw6_max = 44
    hw6_lateness = 45
    hw6_prog = 46
    hw6_prog_max = 47
    hw6_prog_lateness = 49

    hw7 = 50
    hw7_max = 51
    hw7_lateness = 53
    hw7_prog = 54
    hw7_prog_max = 55
    hw7_prog_lateness = 57

    proposal = 63
    proposal_max = 64
    proposal_lateness = 65

    midway = 67
    midway_max = 68
    midway_lateness = 69

    final = 71
    final_max = 72
    final_lateness = 73


    if idx == 0:
        assert row[quiz1] == 'Quiz 1', row[quiz1]
        assert row[quiz2] == 'Quiz 2'
        assert row[quiz3] == 'Quiz 3'

        assert row[hw1] == 'Homework 1', row[hw1]
        assert row[hw1_max] == 'Homework 1 - Max Points', row[hw1_max]
        assert row[hw1_lateness] == 'Homework 1 - Lateness (H:M:S)',row[hw1_lateness]
        assert row[hw1_prog_lateness] == 'Homework 1 - Programming - Lateness (H:M:S)'

        assert row[hw2] == 'Homework 2', row[hw2]
        assert row[hw2_max] == 'Homework 2 - Max Points', row[hw2_max]
        assert row[hw2_lateness] == 'Homework 2 - Lateness (H:M:S)'
        assert row[hw2_prog_lateness] == 'Homework 2 - Programming - Lateness (H:M:S)'
        assert row[hw2_prog_max] == 'Homework 2 - Programming - Max Points', row[hw2_prog_max]

        assert row[hw3] == 'Homework 3', row[hw3]
        assert row[hw3_max] == 'Homework 3 - Max Points', row[hw3_max]
        assert row[hw3_lateness] == 'Homework 3 - Lateness (H:M:S)'
        assert row[hw3_prog_lateness] == 'Homework  3 - Programming - Lateness (H:M:S)',f"`{row[hw3_prog_lateness]}`"
        assert row[hw3_prog_max] == 'Homework  3 - Programming - Max Points', row[hw3_prog_max]

        assert row[hw4] == 'Homework 4', row[hw4]
        assert row[hw4_max] == 'Homework 4 - Max Points', row[hw4_max]
        assert row[hw4_lateness] == 'Homework 4 - Lateness (H:M:S)'
        assert row[hw4_prog_lateness] == 'Homework  4 - Programming - Lateness (H:M:S)',f"`{row[hw4_prog_lateness]}`"
        assert row[hw4_prog_max] == 'Homework  4 - Programming - Max Points', row[hw4_prog_max]

        assert row[hw5] == 'Homework 5', row[hw5]
        assert row[hw5_max] == 'Homework 5 - Max Points', row[hw5_max]
        assert row[hw5_lateness] == 'Homework 5 - Lateness (H:M:S)'
        assert row[hw5_prog_lateness] == 'Homework 5- Programming - Lateness (H:M:S)'
        assert row[hw5_prog_max] == 'Homework 5- Programming - Max Points', row[hw5_prog_max]

        assert row[hw6] == 'Homework 6', row[hw6]
        assert row[hw6_max] == 'Homework 6 - Max Points', row[hw6_max]
        assert row[hw6_lateness] == 'Homework 6 - Lateness (H:M:S)'
        assert row[hw6_prog_lateness] == 'Homework 6 - Programming - Lateness (H:M:S)'
        assert row[hw6_prog_max] == 'Homework 6 - Programming - Max Points', row[hw6_prog_max]

        assert row[hw7] == 'Homework 7', row[hw7]
        assert row[hw7_max] == 'Homework 7 - Max Points', row[hw7_max]
        assert row[hw7_lateness] == 'Homework 7 - Lateness (H:M:S)'
        assert row[hw7_prog_lateness] == 'Homework 7 - Programming - Lateness (H:M:S)'

        assert row[proposal_lateness] == 'Project Proposals  - Lateness (H:M:S)',f"`{row[proposal_lateness]}`"
        assert row[midway_lateness] == 'Midway Project Reports - Lateness (H:M:S)'
        assert row[final_lateness] == 'Final Report - Lateness (H:M:S)'
    else:
        total_lateness = 0
        total_hw = 0
        total_project = 0

        quiz = (num(quiz1) + num(quiz2) + num(quiz3)) / 3

        name= row[0]
        if name == "Minghao Xue":
            quiz = (num(quiz1) + num(quiz2)) / 2
            print("found Minghao Xue!!")

        # assert num(hw1_prog_max) + num(hw1_max) == 100
        # assert num(hw2_prog_max) + num(hw2_max) == 100
        # assert num(hw3_prog_max) + num(hw3_max) == 100
        # assert num(hw4_prog_max) + num(hw4_max) == 100
        # assert num(hw5_prog_max) + num(hw5_max) == 100
        # assert num(hw6_prog_max) + num(hw6_max) == 100
        # assert num(hw7_prog_max) + num(hw7_max) == 100

        hw1_lateness = normalize_latesness(row[hw1_lateness])
        hw1_grade = num(hw1) * discount(total_lateness, hw1_lateness, name)
        hw1_prog_lateness = normalize_latesness(row[hw1_prog_lateness])
        hw1_prog_grade = num(hw1_prog) * discount(total_lateness, hw1_prog_lateness, name)
        total_lateness += max(hw1_lateness, hw1_prog_lateness)
        total_hw1 = (hw1_prog_grade + hw1_grade) / (num(hw1_max) + num(hw1_prog_max))
        total_hw += total_hw1

        hw2_lateness = normalize_latesness(row[hw2_lateness])
        hw2_grade = num(hw2) * discount(total_lateness, hw2_lateness, name)
        hw2_prog_lateness = normalize_latesness(row[hw2_prog_lateness])
        hw2_prog_grade = num(hw2_prog) * discount(total_lateness, hw2_prog_lateness, name)
        total_lateness += max(hw2_lateness, hw2_prog_lateness)
        total_hw2 = (hw2_prog_grade + hw2_grade) / (num(hw2_max) + num(hw2_prog_max))
        total_hw += total_hw2

        hw3_lateness = normalize_latesness(row[hw3_lateness])
        hw3_grade = num(hw3) * discount(total_lateness, hw3_lateness, name)
        hw3_prog_lateness = normalize_latesness(row[hw3_prog_lateness])
        hw3_prog_grade = num(hw3_prog) * discount(total_lateness, hw3_prog_lateness, name)
        total_lateness += max(hw3_lateness, hw3_prog_lateness)
        total_hw3 = (hw3_prog_grade + hw3_grade) / (num(hw3_prog_max) + num(hw3_max))
        total_hw += total_hw3

        hw4_lateness = normalize_latesness(row[hw4_lateness])
        hw4_grade = num(hw4)  * discount(total_lateness, hw4_lateness, name)
        hw4_prog_lateness = normalize_latesness(row[hw4_prog_lateness])
        hw4_prog_grade = num(hw4_prog) * discount(total_lateness, hw4_prog_lateness, name)
        total_lateness += max(hw4_lateness, hw4_prog_lateness)
        total_hw4 = (hw4_prog_grade + hw4_grade) / (num(hw4_max) + num(hw4_prog_max))
        total_hw += total_hw4

        hw5_lateness = normalize_latesness(row[hw5_lateness])
        hw5_grade = num(hw5) * discount(total_lateness, hw5_lateness, name)
        hw5_prog_lateness = normalize_latesness(row[hw5_prog_lateness])
        hw5_prog_grade = num(hw5_prog) * discount(total_lateness, hw5_prog_lateness, name)
        total_lateness += max(hw5_lateness, hw5_prog_lateness)
        total_hw5 = (hw5_prog_grade + hw5_grade) / (num(hw5_max) + num(hw5_prog_max))
        total_hw += total_hw5

        hw6_lateness = normalize_latesness(row[hw6_lateness])
        hw6_grade = num(hw6) * discount(total_lateness, hw6_lateness, name)
        hw6_prog_lateness = normalize_latesness(row[hw6_prog_lateness])
        hw6_prog_grade = num(hw6_prog) * discount(total_lateness, hw6_prog_lateness, name)
        total_lateness += max(hw6_lateness, hw6_prog_lateness)
        total_hw6 = (hw6_prog_grade + hw6_grade) / (num(hw6_max) + num(hw6_prog_max))
        total_hw += total_hw6

        proposal_lateness = normalize_latesness(row[proposal_lateness])
        proposal_grade = num(proposal) / num(proposal_max) * discount(total_lateness, proposal_lateness, name)

        hw7_lateness = normalize_latesness(row[hw7_lateness])
        hw7_grade = num(hw7) * discount(total_lateness, hw7_lateness, name)
        hw7_prog_lateness = normalize_latesness(row[hw7_prog_lateness])
        hw7_prog_grade = num(hw7_prog) * discount(total_lateness, hw7_prog_lateness, name)
        total_lateness += max(hw7_lateness, hw7_prog_lateness)
        total_hw7 = (hw7_prog_grade + hw7_grade) / (num(hw7_prog_max) + num(hw7_max))
        total_hw += total_hw7

        total_hw = 100 * total_hw / 7

        midway_lateness = normalize_latesness(row[midway_lateness])
        midway_grade = num(midway) / num(midway_max) * discount(total_lateness, midway_lateness, name)

        final_lateness = normalize_latesness(row[final_lateness])
        final_grade = num(final) / num(final_max) * discount(total_lateness, final_lateness, name)

        total_lateness += proposal_lateness

        total_project += 20.0 * proposal_grade + 20.0 * midway_grade + 40.0 * final_grade + 20.0

        total = 0.2 * total_project + 0.4 * total_hw + 0.4 * quiz
        # now print all the important information

        # map the grade into letter grade
        # =IFS(
        #     BZ2 > 97, "A+",
        #     BZ2 > 93, "A",
        #     BZ2 > 91, "A-",
        #     BZ2 > 83, "B+",
        #     BZ2 > 75, "B",
        #     BZ2 > 67, "B-",
        #     BZ2 > 57, "C+",
        #     BZ2 > 50, "C",
        #     BZ2 > 42, "C-",
        #     BZ2 > 32, "D+",
        #     BZ2 > 25, "D",
        #     BZ2 <= 25, "F"
        # )
        grade = ''
        if total >= 99:
            grade = 'A+'
        elif total >= 93:
            grade = 'A'
        elif total >= 91:
            grade = 'A-'
        elif total >= 83:
            grade = 'B+'
        elif total >= 75:
            grade = 'B'
        elif total >= 67:
            grade = 'B-'
        elif total >= 57:
            grade = 'C+'
        elif total >= 50:
            grade = 'C'
        elif total >= 42:
            grade = 'C-'
        elif total >= 32:
            grade = 'D'
        else:
            grade = 'F'

        grade_count[grade] = grade_count.get(grade, 0) + 1

        log += f"Grade: {row[0]}\t {grade} \t  total: {total} \t hw: {total_hw} \t project: {total_project} \t quiz: {quiz} \t total_lateness: {total_lateness} \n"
        print(f"Grade: {row[0]}\t {grade} \t  total: {total} \t hw: {total_hw} \t project: {total_project} \t quiz: {quiz} \t total_lateness: {total_lateness} \t total_hw1: {total_hw1} \t total_hw2: {total_hw2} \t total_hw3: {total_hw3} \t total_hw4: {total_hw4} \t total_hw5: {total_hw5} \t total_hw6: {total_hw6} \t total_hw7: {total_hw7} ")


with open("output.csv", "a") as file:
    file.write(log)

print(grade_count)
print(sum(grade_count.values()))