import csv
BBB = 0.10
HHH = 0.001
with open('./data/result.csv','a',newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|',quoting=csv.QUOTE_MINIMAL)
    writer.writerow([BBB] + [HHH])