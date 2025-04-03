import os

directories = os.listdir("organized_images")
for d in directories:
    files = os.listdir(f"organized_images/{d}")
    ages = []
    for f in files:
        age = f.split(".")[0]
        age = int(age)
        ages.append(age)
    data_points = []
    first_pointer = 0
    second_pointer = 1
    n = len(ages)
    while second_pointer < n:
        if abs(ages[first_pointer] - ages[second_pointer]) >= 18:
            data_points.append((ages[first_pointer], ages[second_pointer]))
            data_points.append((ages[second_pointer], ages[first_pointer]))
            first_pointer += 1
        else:
            second_pointer += 1
    print(data_points)
    break
