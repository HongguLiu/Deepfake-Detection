from collections import  Counter

from label import train_list


def check_data_distribution(txt_path) :
    with open(txt_path,'r') as file :
        labels = [int(line.strip().split()[-1])for line in file]


    label_counts = Counter(labels)


    print("Class distribution :")
    for label, count in label_counts.items() :
        print(f"Class {label}: {count} samples")


train_list = './last2_labels.txt'
check_data_distribution(train_list)