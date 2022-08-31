import csv
import re

fields = []
rows = []
with open("data/measurements/c1_measurement.txt)", 'r', encoding='utf16') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile, delimiter = '\t')
    fields = next(csvreader)



with open('data/example.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(fields)



