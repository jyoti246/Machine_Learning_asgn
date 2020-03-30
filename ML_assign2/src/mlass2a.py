import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

	# Data for Part B is created above by normalising and bounding data


	for j in range(11):
		data2[:,j] = (dat[:,j]-np.mean(dat[:,j]))/np.std(dat[:,j])
		print(np.mean(dat[:,j]));
	
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
	# Data for part C is extracted above as per the instructions
	
	return data1, data2

train_data1, train_data2 = get_data()
# train_data.reshape((-1,3))
print("Data for logistic regression: ")
print(train_data1)

print("Data for decision tree: ")
print(train_data2)


