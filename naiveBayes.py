#!/usr/bin/python3
from collections import defaultdict
import sys

def train(train_filename):
    with open(train_filename,"r") as f:
        train_data = [ line.strip().split(',') for line in f.readlines() ]

    field_names, train_data = train_data[0], train_data[1:]
    field_values = { field_names[i] : { row[i] for row in train_data } for i in range(len(field_names)) }

    target_name = field_names[-1]
    target_values = sorted(list(field_values[target_name]))

    counts = defaultdict(int)
    for row in train_data:
        counts[("*","*",field_names[-1], row[-1])]+=1
        for i in range(len(field_names)-1):
            counts[(field_names[i], row[i], field_names[-1], row[-1])]+=1

    P=defaultdict(lambda: 0.0)
    for c,v in counts.items():
        if c[0]=="*" and c[1]=="*":
            P[c]=v/len(train_data)
        else:
            P[c]=v/counts[("*","*",c[2],c[3])]

    return (field_names, field_values, target_name, target_values, counts, P)

def report_counts(counts, counts_filename):
    # Note: if (X=x n Y=y) not in count, then count(X=x n Y=y)=0
    counts_txt = []
    for c,v in counts.items():
        counts_txt.append("{},{},{},{},{}".format(c[0],c[1],c[2],c[3],v))
    counts_txt.sort()

    f = open(counts_filename, "w")
    for line in counts_txt:
        f.write(line+"\n")
    f.close()

def report_probabilities(P, probabilities_filename):
    P_txt = []
    for c,v in P.items():
        P_txt.append("{},{},{},{},{}".format(c[0],c[1],c[2],c[3],v))
    P_txt.sort()

    f = open(probabilities_filename, "w")
    for line in P_txt:
        f.write(line+"\n")
    f.close()

'''
P(Y=y1|X=x) = P(X1=x1|Y=y1)P(X2=x2|Y=y1)... P(Xn=xn|Y=y1) P(Y=y1)/P(X=x)
P(Y=y2|X=x) = P(X1=x1|Y=y2)P(X2=x2|Y=y2)... P(Xn=xn|Y=y2) P(Y=y2)/P(X=x)
P(Y=y3|X=x) = P(X1=x1|Y=y3)P(X2=x2|Y=y3)... P(Xn=xn|Y=y3) P(Y=y3)/P(X=x)
...
P(Y=yk|X=x) = P(X1=x1|Y=yk)P(X2=x2|Y=yk)... P(Xn=xn|Y=yk) P(Y=yk)/P(X=x)
'''
def classify(observation, P, target_name, target_values):

    Ptop={} # the numerator of the Baysean classification probabilities
    for target_value in target_values:
        Ptop[target_value]=P[("*","*",target_name,target_value)]
        for o,v in observation.items():
            Ptop[target_value]*=P[(o,v,target_name,target_value)]
        if Ptop[target_value] == 0.0:
            print('Please run again and input using the format given!')
            sys.exit()

    return max(Ptop, key=Ptop.get), Ptop

def test(target_name, target_values, P, test_filename):
    '''
    Now that we have trained P, we have the target_name and
    its possible values, target_values, we run through each line
    of test_filename (which must have the same form as train_filename)
    and classify each line in test_filename, and record how well
    the classifier works.
    '''

    with open(test_filename,"r") as f:
        test_data = [ line.strip().split(',') for line in f.readlines() ]

    field_names, test_data = test_data[0], test_data[1:]


    # selected_fields=[ True, True, True, False, True, True, True, True, False, True, True, False, False ]


    for row in test_data:
        observation = { field_names[i]: row[i] for i in range(len(row)) }
        # observation = { field_names[i]: row[i] for i in range(len(row)-1) if selected_fields[i] }

        classed, Ptop = classify(observation, P, target_name, target_values)

        ptopString = ""
        for target_value in Ptop:
            ptopString += " Ptop({}={}|X=x)={}".format(target_name, target_value,Ptop[target_value])

    return classed, ptopString

if __name__ == "__main__":
    if len(sys.argv)!=1:
        print('''
Usage: naieveBayes.py 
       learning from train.csv
       testing on test.csv
       producing counts.csv
                 probabilities.csv
                 out.txt
       train consists of a header line and then rows, the last column is the class
       test consists of a header line and then rows, the last column is the class
''')
        sys.exit(1)
    temp = input('Would you like to test for Diabetes or Heart (diabetes/heart): ')
    if temp == "heart":
        (field_names, field_values, target_name, target_values, counts, P) = train("heart/train.csv")
        report_counts(counts, "heart/counts.csv")
        report_probabilities(P, "heart/probabilities.csv")
        store = []
        x = input('Have you smoked at least 100 cigarettes in your entire life? (Yes/No): ')
        store.append(x)
        x = input('Do you cosume alcohol? (Yes/No): ')
        store.append(x)
        x = input('Did you ever have a stroke? (Yes/No): ')
        store.append(x)
        x = input('Do you have serious difficulty walking or climbing stairs? (Yes/No): ')
        store.append(x)
        x = input('What is your sex? (Male/Female): ')
        store.append(x)
        x = input('Age? (18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80 or older): ')
        store.append(x)
        x = input('Imputed race/ethnicity value? (American Indian/Alaskan Native, Asian, Black, Other, Hispanic, White): ')
        store.append(x)
        x = input('Did you ever have diabetes? (No, Yes (during pregnancy), Yes, "No borderline diabetes"): ')
        store.append(x)
        x = input('Have you done any physical activity or exercise during the past 30 days other than your regular job? (Yes/No): ')
        store.append(x)
        x = input('Would you say that in general your health is... (Fair, Good, Poor, Very good, Excellent): ')
        store.append(x)
        x = input('Did you ever have asthma? (Yes/No): ')
        store.append(x)
        x = input('Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease? (Yes/No): ')
        store.append(x)
        x = input('Did you ever have skin cancer? (Yes/No): ')
        store.append(x)
        final = ','.join(store)
        f = open("heart/test.csv", "w")
        f.write("Smoking,AlcoholDrinking,Stroke,DiffWalking,Sex,AgeCategory,Race,Diabetic,PhysicalActivity,GenHealth,Asthma,KidneyDisease,SkinCancer,HeartDisease\n")
        f.write(final)
        f.close()

        result, ptop = test(target_name, target_values, P, "heart/test.csv")
        print('')
        print("Do you have any heart complications?")
        print("Based off the information given, our model states: {}".format(result))
        print("Probabilities: {}".format(ptop))

    else:
        (field_names, field_values, target_name, target_values, counts, P) = train("diabetes/train.csv")
        report_counts(counts, "diabetes/counts.csv")
        report_probabilities(P, "diabetes/probabilities.csv")
        store = []
        x = input('Have you EVER been told by a doctor, nurse or other health professional that your blood pressure is high? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Have you had a cholesterol check within past five years? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Did you ever have a stroke? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Have you ever reported having coronary heart disease (CHD) or myocardial infarction (MI)? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Have you done any physical activity or exercise during the past 30 days other than your regular job? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Do you consume fruit 1 or more times per day? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Do you consume vegetables 1 or more times per day? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Would you say that in general your health is: rate (1.0 ~ 5.0) (1.0, 2.0, 3.0, 4.0, 5.0): ')
        store.append(x)
        x = input('Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? (0.0 ~ 30.0): ')
        store.append(x)
        x = input('Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0.0 ~ 30.0): ')
        store.append(x)
        x = input('Do you have serious difficulty walking or climbing stairs? (1.0 = Yes/ 0.0 = No): ')
        store.append(x)
        x = input('Sex? (1.0 = Male/ 0.0 = Female): ')
        store.append(x)
        x = input('Age? (0.0 = 18-24, 1.0 = 25-29, 2.0 = 30-34, 3.0 = 35-39, 4.0 = 40-44, 5.0 = 45-49, 6.0 = 50-54, 7.0 = 55-59, 8.0 = 60-64, 9.0 = 65-69, 10.0 = 70-74, 11.0 = 75-79, 12.0 = 80 or older): ')
        store.append(x)
        x = input('Overall happiness? (1.0 - 6.0): ')
        store.append(x)
        x = input('Is your annual household income from all sources? (1.0 = less than 10,000; 5.0 = less than 35,000; 8.0 = $75,000 or more): ')
        store.append(x)
        final = ','.join(store)
        f = open("diabetes/test.csv", "w")
        f.write("HighBP,HighChol,CholCheck,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income\n")
        f.write(final)
        f.close()

        result, ptop = test(target_name, target_values, P, "diabetes/test.csv")
        if str(result) == '0.0':
            result = 'No'
        else:
            result = 'Yes'
        print('')
        print("Do you have diabetes?")
        print("Based off the information given, our model states: {}".format(result))
        print("Probabilities: {}".format(ptop))
