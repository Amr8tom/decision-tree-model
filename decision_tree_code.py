import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn import metrics

train = pd.read_csv("C:/Users/user/Downloads/house-votes-84.data.csv",names=["category",2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
for i in range(2,len(train.columns)+1):
    count_y=0
    count_n=0
    for j in train[i]:
        if(j=="y"):
            count_y=count_y+1
        elif(j=="n"):
            count_n=count_n+1
    if(count_y>count_n):
        train[i]=train[i].replace(to_replace="?",value="y")
    else:
        train[i]=train[i].replace(to_replace="?",value="n")
print(train)
train=train.replace(to_replace="y",value="1")
train=train.replace(to_replace="n",value="0")

x=train.drop(columns=["category"])
y=train["category"]
Train_size = [50,60,70,80]
Accuracy_list = []
minimum_list = []
maximum_list = []
mean_list =[]
calcmean = 0
nodes = []
calc_nodemean = 0
min_node_list = []
max_node_list = []
mean_node_list = []
for n in Train_size:
    i = 0
    while i < 5:
        ranint = random.randint(0,100)
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=n, random_state=ranint)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        Accuracy_list.append(accuracy)
        nodes.append(clf.tree_.node_count)
        i = i + 1
    minimum_list.append(min(Accuracy_list))
    maximum_list.append(max(Accuracy_list))
    calcmean=sum(Accuracy_list)/len(Accuracy_list)
    mean_list.append(calcmean)
    min_node_list.append(min(nodes))
    max_node_list.append(max(nodes))
    calc_nodemean = sum(nodes)/len(nodes)
    mean_node_list.append(calc_nodemean)
    nodes = []
    Accuracy_list = []

print("The list of minimum accuracy of each trace is :",minimum_list)
print("The list of maximum accuracy of each trace is :",maximum_list)
print("The list of mean accuracy of each trace is :",mean_list)
print("The list of minimum n. nodes of each trace is :",min_node_list)
print("The list of maximum n. nodes of each trace is :",max_node_list)
print("The list of mean n. nodes of each trace is :",mean_node_list)


plt.plot(Train_size, mean_list, color='red')

plt.xlabel('Train_Size')
plt.ylabel('Accuracy')
plt.title('Accuracy Varies With Training Set Size!')
plt.show()

plt.plot(Train_size, mean_node_list, color='green')

plt.xlabel('Train_Size')
plt.ylabel('N. Nodes')
plt.title('The N. of Nodes in The Final Tree Varies With Training Set Size!')
plt.show()

