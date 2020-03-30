import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold

fd=1
def get_data():
	dat=pd.read_csv('winequality-red.csv', sep=';')
	dat=np.array(dat.iloc[:,:])
	data1=np.copy(dat)
	data2=np.copy(dat)
	# print(dat[3:30])
	mx = [0,0,0,0,0,0,0,0,0,0,0]
	mn = [0,0,0,0,0,0,0,0,0,0,0]
	for i in range(11):
		mx[i]=np.max(dat[:,i])
		mn[i]=np.min(dat[:,i])


	for i in range(dat.shape[0]):
		for j in range(12):
			if j==11:
				if dat[i,j]<=6:
					data1[i,j]=0
				else:
					data1[i,j]=1
			else:
				data1[i,j]=(dat[i,j]-mn[j])/(mx[j]-mn[j])

	for j in range(11):
		data2[:,j] = (dat[:,j]-np.mean(dat[:,j]))/np.std(dat[:,j])
		# print(np.mean(dat[:,j]));
	
	# print(dat[3:30])
	for i in range(11):
		mx[i]=np.max(data2[:,i])
		mn[i]=np.min(data2[:,i])

	for i in range(dat.shape[0]):
		for j in range(12):
			if j==11:
				if dat[i,j]<6:
					data2[i,j]=0
				elif dat[i,j]<=6:
					data2[i,j]=1
				else:
					data2[i,j]=2
			else:
				if data2[i,j]<mn[j]+(mx[j]-mn[j])/4:
					data2[i,j]=0
				elif data2[i,j]<mn[j]+(mx[j]-mn[j])/2:
					data2[i,j]=1
				elif data2[i,j]<mn[j]+3*(mx[j]-mn[j])/4:
					data2[i,j]=2
				else:
					data2[i,j]=3

	data2 = data2.astype(int)

	return data1, data2

train_data1, train_data2 = get_data()
# train_data.reshape((-1,3))
# print(train_data1)
# print(train_data2)


#############################################  Part a COMPLETED  ################################################3




def get_entropy(data):
    zer = len(np.where(data[:,-1] == 0)[0])
    one = len(np.where(data[:,-1] == 1)[0])
    to = len(np.where(data[:,-1] == 2)[0])
    # print(zer)
    # print(one)
    # print(to)
    if data.shape[0]==0:
    	return 0
    prob0 = zer/(1.0 *(data.shape[0]))
    prob1 = one/(1.0 *(data.shape[0]))
    prob2 = to/(1.0 *(data.shape[0]))
    
    # if (prob0 == 0 || prob1 == 0 || prob2 == 0):
    #     return 0
    entropy=0
    if(prob0 == 0):
    	pass
    else:
    	entropy += -prob0 * math.log(prob0,2)

    if(prob1 == 0):
    	pass
    else:
    	entropy += -prob1 * math.log(prob1,2)

    if(prob2 == 0):
    	pass
    else:
    	entropy += -prob2 * math.log(prob2,2)

    
    return entropy

# get_entropy(train_data2)


def get_gain(data, attribute):
    entropy_final = 0
    entropy_beg = get_entropy(data)
    num_val = 0
    for val in range(4):
        data_div = data[np.where(data[:, attribute] == val)[0]]
        if data_div.shape[0] ==0:
            continue
        entropy_final += (data_div.shape[0])/(1.0 * data.shape[0])*get_entropy(data_div)
        num_val = num_val + 1
    return entropy_beg - entropy_final, num_val


def get_max_gain(data, attributes):
    max_gain = -1000000
    attribute = -1
    if data.shape[0]==0:
    	return 0
    for i in attributes:
        gain, num_val = get_gain(data, i)
        if num_val == 1:
            return -1
        if gain > max_gain:
            max_gain = gain
            attribute = i 
    
    return attribute

class Non_Leaf:
    def __init__(self ):
        # self.data = data
        # self.metadata = metadata
        # self.level = level
        # self.max_level = max_level
        self.child = []
        # self.div = div
        # self.attributes = attributes

    def set_child(self,data, attributes):
        self.attribute = get_max_gain(data, attributes)
        attrs = []
        for attr in attributes:
            if attr != self.attribute:
                attrs.append(attr)
        if(self.attribute == -1):
            return -1
        if data.shape[0] >= 10 :
            for val in range(4):
                node = Non_Leaf()  #
                
                check = node.set_child(data[np.where(data[:, self.attribute] == val)[0]], attrs)
                if check == -1:
                    # self.child.remove(child)
                    new_child = Leaf()
                    new_child.get_class(data[np.where(data[:, self.attribute] == val)[0]])
                    self.child.append(new_child)
                
                else:
                	self.child.append(node)
            
                
                
        else:
        	return -1
            

    def classify(self, sample):
        val = sample[self.attribute]
        temp=0
        for child in self.child:
            if temp == val:
                return child.classify(sample)
            temp +=1


class Leaf:
    def __init__(self):
        pass

    def get_class(self,data):
        zer = len(np.where(data[:,-1] == 0)[0])
        one = len(np.where(data[:,-1] == 1)[0])
        to = len(np.where(data[:,-1] == 2)[0])
        mx= max(zer,one,to)
        if zer == mx:
            self._class = 0
        elif one==mx:
        	self._class = 1
        else:
            self._class = 2

    

    def classify(self, sample):
        return self._class



Decision_Tree = Non_Leaf()
Decision_Tree.set_child(train_data2,[0,1,2,3,4,5,6,7,8,9,10])
count=0
ct=0
ind=0
clf = DecisionTreeClassifier(criterion="entropy",min_samples_split=10,random_state=0)
clf.fit(train_data2[:,:11],train_data2[:,11])
out = clf.predict(train_data2[:,:11])

for sample in train_data2:
    pred = Decision_Tree.classify(sample)
    if pred == sample[-1]:
        count = count + 1
    if pred == out[ind]:
    	ct +=1

    ind +=1

print('Training Accuracy: ', count * 100.0 /train_data2.shape[0])


print('Accuracy compared to scikit: ', ct * 100.0 /train_data2.shape[0])



kf = KFold(n_splits = 3)
saccd =0 
sprecd=0
srecd =0
accd =0 
precd=0
recd =0
for tri, tsi in kf.split(train_data2):
    xtr, ytr = train_data2[tri, :11], train_data2[tri, 11]
    xts, yts = train_data2[tsi, :11], train_data2[tsi, 11] 
    
    Decision_Tree = Non_Leaf()
    Decision_Tree.set_child(train_data2[tri,:],[0,1,2,3,4,5,6,7,8,9,10])
    DecPred=np.copy(yts)
    ind=0
    # print(DecPred.shape)
    for sample in train_data2[tsi,:]:
	    pred = Decision_Tree.classify(sample)
	    DecPred[ind]=(pred)
	    ind +=1
	
    precd += precision_score(yts, DecPred, average='macro')
    recd += recall_score(yts, DecPred, average='macro')
    accd += accuracy_score(yts, DecPred)

    clf = DecisionTreeClassifier(criterion="entropy", min_samples_split=10,random_state=0)
    clf.fit(xtr, ytr) 
    
    
    sDecPred = clf.predict(xts) 
    # print(sDecPred.shape)  
    saccd += accuracy_score(yts, sDecPred)
    sprecd += precision_score(yts, sDecPred, average='macro')
    srecd += recall_score(yts, sDecPred, average='macro')

saccd /= 3
sprecd /= 3
srecd /= 3
accd /= 3
precd /= 3
recd /= 3
print('Our Decision tree model mean Accuracy= ', accd,'Precision= ',precd,'Recall= ', recd, 'Scikit model Accuracy= ', saccd,'Precision= ', sprecd,'Recall= ',srecd)




######################################################################3








