import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA



def get_data():
	dat=pd.read_csv('AllBooks_baseline_DTM_Labelled.csv')
	dat=np.array(dat.iloc[:,:])
	data1 = dat[:,0]
	data1 = data1.reshape((-1,1))
	data2 = np.copy(dat[:,1:])
	# data1 = np.char.split(data1[:,0], sep = '_')
	for i in range(dat.shape[0]):
		dat[i,0]= dat[i,0].split('_')[0]
	# data1 = data1[:,0][:-3]
	# idf= np.zeros((dat.shape[1]-1,1))
	# print(idf.shape)
	print(data2.shape)
	for i in range(dat.shape[1]):
		if i!=0:
			x=0
			for j in range(dat.shape[0]):
				if dat[j,i]!=0:
					x=x+1
			# idf[i-1,0]=x
			x= math.log((1.0+dat.shape[0])/(1.0+x))
			for j in range(dat.shape[0]):
				data2[j,i-1]=x*data2[j,i-1]

	# set_printoptions(precision=2)
	Data_normalizer = Normalizer(norm='l2').fit(data2)
	data2 = Data_normalizer.transform(data2)
	# for i in range(data2.shape[1]):
	# 	for j in range(data2.shape[0]):
	# 		if(data2[j,i]>0):
	# 			print(data2[j,i])
	
	print(data1.shape)
	return data1, data2

data1, data2  = get_data()
# train_data.reshape((-1,3))
print("Data sd DTM")
print(data2)





# def entropy_class_labels(data,labels):
# 	(name,counts) = np.unique(labels,return_counts=True)
# 	frequencies = np.asarray((name,counts)).T
# 	P = {x[0]:x[1]/data.shape[0] for x in frequencies}
# 	H = 0
# 	for i in P.values():
# 		H = H - i*np.log(i)
# 	return H

# def entropy_cluster_labels(clusters, m):
# 	H = 0
# 	for i in range(8):
# 		x = len(clusters[i])/m
# 		H = H - x*np.log(x)
# 	return H


# def conditional_entropy(clusters, m, labels):

# 	H = 0
# 	for i in range(8):
# 		p = len(clusters[i])/m
# 		dic = {x:0 for x in labels}
# 		for j in clusters[i]:
# 			x=labels[int(j)]
# 			# print(x)
# 			dic[x] = dic[x] + 1
		
# 		h_yc = 0
# 		for j in dic.values():
# 			if(j/len(clusters[i])!=0):
# 				h_yc = h_yc + (j/len(clusters[i]))*np.log(j/len(clusters[i]))

# 		h_yc = h_yc * -p
# 		H = H + h_yc
# 	return H


def calculate_NMI(file,labels,data):
	labels = labels.flatten()
	(name,counts) = np.unique(labels,return_counts=True)
	frequencies = np.asarray((name,counts)).T
	P = {x[0]:x[1]/data.shape[0] for x in frequencies}
	H_class = 0
	for i in P.values():
		H_class = H_class - i*np.log(i)
	
	# H_class = entropy_class_labels(data,labels)
		
	f = open(file,'r')
	clusters_ = f.read().split('\n\n\n')
	clusters = [clusters_[i].split(', ') for i in range(8)]
	H_cluster = 0
	for i in range(8):
		x = len(clusters[i])/data.shape[0]
		H_cluster = H_cluster - x*np.log(x)
	
	# H_cluster = entropy_cluster_labels(clusters, data.shape[0])

	# Mutual Info I = H_class - (entropy of class labels within each cluster)

	I = 0
	for i in range(8):
		p = len(clusters[i])/data.shape[0]
		dic = {x:0 for x in labels}
		for j in clusters[i]:
			x=labels[int(j)]
			# print(x)
			dic[x] = dic[x] + 1
		
		h_yc = 0
		for j in dic.values():
			if(j/len(clusters[i])!=0):
				h_yc = h_yc + (j/len(clusters[i]))*np.log(j/len(clusters[i]))

		h_yc = h_yc * -p
		I = I + h_yc
	



	I = H_class - I #conditional_entropy(clusters, data.shape[0], labels)

	NMI = 2*I / (H_class + H_cluster)
	print("NMI score for '%s' is %f"%(file,NMI))


calculate_NMI("agglomerative_reduced.txt",data1, data2)
calculate_NMI("kmeans.txt",data1, data2)
calculate_NMI("agglomerative.txt",data1, data2)
calculate_NMI("kmeans_reduced.txt",data1, data2)


