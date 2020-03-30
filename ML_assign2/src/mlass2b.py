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


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

# print(sigmoid(train_data1))
# train_f=train_data1[:, 0:11]
# print(train_f.shape)


# Same as linear regression from last assignment just in one step we are using the above 
# sigmoid function to the the value of hypothesis function
class Logistic_Regression:
	def __init__(self, train_d,test_d):
		self.train_d=train_d
		self.test_d = test_d
		self.train_f=np.copy(train_d[:, 0:11])
		self.train_l=np.copy(train_d[:, 11])
		
		self.train_l = self.train_l.reshape((-1,1))
		
		self.train_fpow=np.ones([train_d.shape[0],1])
		self.train_f= np.hstack((self.train_fpow, self.train_f ))
		# self.test_fpow=self.train_f

		self.test_f=np.copy(test_d[:, 0:11])
		self.test_l=np.copy(test_d[:, 11])
		
		self.test_l = self.test_l.reshape((-1,1))
		
		self.test_fpow=np.ones([test_d.shape[0],1])
		self.test_f= np.hstack((self.test_fpow, self.test_f ))
		

		# self.parameter = np.random.normal(size=(n+1,1))
		self.parameter = np.zeros((12,1))
		# print(self.parameter)
		# print(self.train_f)
		# print(self.train_fpow)

	def grad_descent(self):
		global fd
		alpha = 0.05/self.train_l.shape[0]
		# print(self.parameter)
		# print(self.train_l.shape)
		# self.train_l = self.train_l.reshape((-1, 1))
		# print(self.parameter.shape)
		for i in range(100):
			self.parameter = self.parameter - alpha * np.matmul(self.train_f.transpose() , (sigmoid(np.matmul(self.train_f , self.parameter)) - self.train_l))
		# print(( self.train_l).shape)
		# print("Parameters for n = %d",self.n)
		# print(self.parameter)

		# sq_error = (np.matmul(self.train_fpow , self.parameter) - self.train_l)
		# sq_error = np.matmul(sq_error.transpose(),sq_error)
		# sq_error = sq_error/2/self.train_l.shape[0]
		# print("Squared error in training data is:")
		# print(sq_error[0][0])
		# return sq_error[0][0]



	
	def predict(self):
		global fd
		test_data_pred = np.matmul(self.test_f , self.parameter)
		
		fd += 1
		test_data_pred=(test_data_pred>=0.5).astype(int)
		# print("Predicted labels for test data are as follows")
		# print(test_data_pred )
		clf = LogisticRegression(random_state=0, solver= "saga").fit(self.train_d[:,:11], self.train_d[:,11])
		res = clf.predict(self.test_d[:,:11])
		# res.reshape((-1,1))
		# print(res)
		
		count = 0.0
		c11=0.0
		c10=0.0
		c01=0.0
		c00=0.0
		sc11=0.0
		sc10=0.0
		sc01=0.0
		sc00=0.0
		for i in range(test_data_pred.shape[0]):
			if(test_data_pred[i,0]==res[i]):
				count +=1
			if(self.test_l[i,0]==test_data_pred[i,0] and test_data_pred[i,0]==1):
				c11 +=1
			if(self.test_l[i,0]==test_data_pred[i,0] and test_data_pred[i,0]==0):
				c00 +=1
			if(self.test_l[i,0]!=test_data_pred[i,0] and test_data_pred[i,0]==0):
				c10 +=1
			if(self.test_l[i,0]!=test_data_pred[i,0] and test_data_pred[i,0]==1):
				c01 +=1
			if(self.test_l[i,0]==res[i] and res[i]==1):
				sc11 +=1
			if(self.test_l[i,0]==res[i] and res[i]==0):
				sc00 +=1
			if(self.test_l[i,0]!=res[i] and res[i]==0):
				sc10 +=1
			if(self.test_l[i,0]!=res[i] and res[i]==1):
				sc01 +=1
		    
		# in below variable c represents our models data and sc represents scikit's data also the digits 11, 10, 01, 00 
		# represents count of outputs and origional label correspnding to the digit. 
		ret =[c11 , c00, c10, c01, sc11 , sc00, sc10, sc01 ]
		# Here we are printing percentage by which sciKit and our models output are matching
		print('Accuracy compared to scikit learn: ', count * 100.0 /test_data_pred.shape[0])
		return ret

	


print("For entire data as test data too")
my_log = Logistic_Regression(train_data1,train_data1)
my_log.grad_descent()
my_ans = my_log.predict()
eak=(( train_data1.shape[0])/3)
do= ((2 * train_data1.shape[0])/3)
# print(eak)
eak=int(eak)
do= int(do)
acc=0.0
prec=0.0
rec=0.0
sacc=0.0
sprec=0.0
srec=0.0


# here we will find each needed term using values returned by predict
print("For a partition of 3 fold validation")

my_log1 = Logistic_Regression(train_data1[:do,:],train_data1[do:,:])
my_log1.grad_descent()
my_ans1 = my_log1.predict()
# print(my_ans1)
acc=(my_ans1[0]+my_ans1[1])/(my_ans1[0]+my_ans1[1]+my_ans1[2]+my_ans1[3])
sacc=(my_ans1[4+0]+my_ans1[4+1])/(my_ans1[4+0]+my_ans1[4+1]+my_ans1[4+2]+my_ans1[4+3])
prec +=my_ans1[0]/max(1,my_ans1[0]+my_ans1[3])
rec +=my_ans1[0]/max(1,my_ans1[0]+my_ans1[2])
sprec +=my_ans1[4]/max(1,my_ans1[4]+my_ans1[7])
srec +=my_ans1[4]/max(1,my_ans1[4]+my_ans1[6])


print("For a partition of 3 fold validation")
my_log2 = Logistic_Regression(train_data1[eak:,:],train_data1[:eak,:])
my_log2.grad_descent()
my_ans2 = my_log2.predict()
acc +=(my_ans2[0]+my_ans2[1])/(my_ans2[0]+my_ans2[1]+my_ans2[2]+my_ans2[3])
sacc +=(my_ans2[4+0]+my_ans2[4+1])/(my_ans2[4+0]+my_ans2[4+1]+my_ans2[4+2]+my_ans2[4+3])
prec +=my_ans2[0]/max(1,my_ans2[0]+my_ans2[3])
rec +=my_ans2[0]/max(1,my_ans2[0]+my_ans2[2])
sprec +=my_ans2[4]/max(1,my_ans2[4]+my_ans2[7])
srec +=my_ans2[4]/max(1,my_ans2[4]+my_ans2[6])


print("For a partition of 3 fold validation")
my_log3 = Logistic_Regression(np.concatenate((train_data1[do:,:],train_data1[:eak,:])),train_data1[eak:do,:])
my_log3.grad_descent()
my_ans3 = my_log3.predict()
acc +=(my_ans3[0]+my_ans3[1])/(my_ans3[0]+my_ans3[1]+my_ans3[2]+my_ans3[3])
sacc +=(my_ans3[4+0]+my_ans3[4+1])/(my_ans3[4+0]+my_ans3[4+1]+my_ans3[4+2]+my_ans3[4+3])
prec +=my_ans3[0]/max(1,my_ans3[0]+my_ans3[3])
rec +=my_ans3[0]/max(1,my_ans3[0]+my_ans3[2])
sprec +=my_ans3[4]/max(1,my_ans3[4]+my_ans3[7])
srec +=my_ans3[4]/max(1,my_ans3[4]+my_ans3[6])
acc /=3
sacc /=3
prec /=3
rec /=3
sprec /=3
srec /=3
print('Our Logistic Regression model mean Accuracy= ', acc,'Precision= ',prec,'Recall= ', rec, 'Scikit model Accuracy= ', sacc,'Precision= ', sprec,'Recall= ',srec)

##############################################################################################







