# read the csv file 'May 16,2024.csv"
# and aggregate the grades for each student


import csv

csv_file = open('May 16,2024.csv')
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

def discount(total_late_days, current_late_days):
    if total_late_days <= 10:
        return 1.0
    return (1 - current_late_days * 0.05)

grade_count = {}

# average the quiz columsn for each student: index 10 and 20
for idx, row in enumerate(csv_reader):

    # print(row)


    def num(int):
        if row[int] == '':
            return 0
        return float(row[int])


    if idx == 0:
        assert row[21] == 'Quiz 1'
        assert row[42] == 'Quiz 2'

        assert row[5] == 'Homework 1 - Lateness (H:M:S)'
        # assert row[8] == 'Homework 1 - Programming - Lateness (H:M:S)'

        assert row[11] == 'Homework 2 - Lateness (H:M:S)'
        # assert row[15] == 'Homework 2 - Programming - Lateness (H:M:S)'

        assert row[17] == 'Homework 3 - Lateness (H:M:S)'
        # assert row[20] == 'Homework 3 - Programming - Lateness (H:M:S)'

        assert row[26] == 'Homework 4 - Lateness (H:M:S)'
        # assert row[29] == 'Homework 4 - Programming - Lateness (H:M:S)'

        assert row[32] == 'Homework 5 - Lateness (H:M:S)'
        # assert row[35] == 'Homework 5- Programming - Lateness (H:M:S)'

        assert row[38] == 'Homework 6 - Lateness (H:M:S)'
        # assert row[41] == 'Homework 6 - Programming - Lateness (H:M:S)'

        assert row[50] == 'Homework 7 - Lateness (H:M:S)'
        # assert row[53] == 'Homework 7 - Programming - Lateness (H:M:S)'

        assert row[47] == 'Project Proposals - Lateness (H:M:S)'
        assert row[56] == 'Midway Project Reports - Lateness (H:M:S)'
        assert row[59] == 'Final Report - Lateness (H:M:S)'
    else:
        total_lateness = 0
        total_hw = 0
        total_project = 0

        quiz = (num(21) + num(42)) / 2


        hw1 = (num(3) + num(6))
        hw1_lateness = normalize_latesness(row[5])
        total_lateness += hw1_lateness
        total_hw += 100.0 / 130.0 * hw1 * discount(total_lateness, hw1_lateness)

        hw2 = (num(9) + num(12))
        hw2_lateness = normalize_latesness(row[11])
        total_lateness += hw2_lateness
        # max grade is 124
        # 9 extra credits, leading 124 - 9 = 115
        total_hw += 100.0 / 115.0 * hw2 * discount(total_lateness, hw2_lateness)

        hw3 = (num(15) + num(18))
        hw3_lateness = normalize_latesness(row[17])
        # max is 125.0
        # 5 extra credits, leading 15 - 5 = 120
        total_lateness += hw3_lateness
        total_hw += 100.0 / 120.0 * hw3 * discount(total_lateness, hw3_lateness)

        hw4 = (num(24) + num(27))
        hw4_lateness = normalize_latesness(row[26])
        total_lateness += hw4_lateness
        # max is 100.0
        total_hw += 100.0 / 100.0 * hw4 * discount(total_lateness, hw4_lateness)

        hw5 = (num(30) + num(33))
        hw5_lateness = normalize_latesness(row[32])
        total_lateness += hw5_lateness
        # the maximum grade is 100
        # 14 extra credits, leading 100 - 14 = 86
        total_hw += 100.0 * hw5 / 86.0  * discount(total_lateness, hw5_lateness)

        hw6 = (num(36) + num(39))
        hw6_lateness = normalize_latesness(row[38])
        total_lateness += hw6_lateness
        # the maximum grade is 129
        # 21 extra credits, leading 129 - 21 = 108
        total_hw += 100.0 / 108.0 * hw6 * discount(total_lateness, hw6_lateness)

        project_proposals = num(45)
        proposal_latness = normalize_latesness(row[47])
        total_lateness += proposal_latness
        total_project += 15.0 * project_proposals / 6.0 * discount(total_lateness, proposal_latness)

        hw7 = num(48)
        hw7_lateness = normalize_latesness(row[50])
        total_lateness += hw7_lateness
        # the maximum grade is 86
        total_hw += 100 / 86.0 * hw7 * discount(total_lateness, hw7_lateness)

        if row[0] == 'TRAVIS YOU':
            # registered late and asked for permission to distribute his hw1 grade across the other 6 homeworks
            total_hw = total_hw / 6
        else:
            total_hw = total_hw / 7

        midway_project_reports = num(54)
        midway_project_reports_lateness = normalize_latesness(row[56])
        total_lateness += midway_project_reports_lateness
        total_project += 25.0 * midway_project_reports / 60.0 * discount(total_lateness, midway_project_reports_lateness)

        final_report = num(57)
        final_report_lateness = normalize_latesness(row[59])
        total_lateness += final_report_lateness
        total_project += 50.0 * final_report / 60.0 * discount(total_lateness, final_report_lateness)

        total = 0.3 * total_project + 0.4 * total_hw + 0.3 * quiz
        # print(total)

        # map the grade into letter grade
        # A+	above 100
        # A	    100
        # A-	92.5
        # B+	82.5
        # B	    75
        # B-	67.5
        # C+	57.5
        grade = ''
        if total >= 96:
            grade = 'A+'
        elif total >= 90.5:
            grade = 'A'
        elif total >= 87:
            grade = 'A-'
        elif total >= 82:
            grade = 'B+'
        elif total >= 75:
            grade = 'B'
        elif total >= 67.5:
            grade = 'B-'
        elif total >= 57.5:
            grade = 'C+'
        else:
            grade = 'F'

        grade_count[grade] = grade_count.get(grade, 0) + 1

        print(f"{row[0]} --- {total} --- {grade}")

    # break

print(grade_count)