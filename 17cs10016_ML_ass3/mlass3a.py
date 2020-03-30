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



# -------------------------------Part b -----------------------------------------------

def allog_clustering(data):
	x=(data.shape[0]*data.shape[0]-data.shape[0])/2
	edges = np.zeros((int(x), 3))
	k=0
	# print(data[0])
	# print(data[1])
	for i in range(data.shape[0]):
		for j in range(i,data.shape[0]):
			if i!=j:
				if (np.dot(data[i],data[i]) * np.dot(data[j],data[j])) ==0:
					edges[k,0]=0
				else:
					edges[k,0]= np.dot(data[i],data[j])/math.sqrt(np.dot(data[i],data[i]) * np.dot(data[j],data[j]))
				# edges[k,0]= np.dot(data[i],data[j])
				edges[k,1]= i
				edges[k,2]= j
				# print(edges[k,0])
				k=k+1
				

	# print(1000000)
	tree = np.zeros((2,data.shape[0]))
	for i in range(data.shape[0]):
		tree[0,i]=i
		tree[1,i]=1
	edges = edges[edges[:,0].argsort()]
	k=0
	for i in range(edges.shape[0]):
		x = int(edges[i,1])
		y = int(edges[i,2])

		while tree[0,int(x)]!= x:
			x= tree[0,int(x)]
		while tree[0,int(y)]!= y:
			y= tree[0,int(y)]
		if x!=y:
			k=k+1
			if tree[1,int(x)]>=tree[1,int(x)]:
				tree[0,int(y)]=x
				tree[1,int(x)]=tree[1,int(x)]+tree[1,int(y)]
			else:
				tree[0,int(x)]=y
				tree[1,int(y)]=tree[1,int(x)]+tree[1,int(y)]
			if (k>=data.shape[0]-8):
				break
	# print(2000000)
	cluster = np.zeros((data.shape[0],2))
	for i in range(data.shape[0]):
		x=int(i)
		while tree[0,int(x)]!=x :
			x= tree[0,int(x)]
		cluster[i,0]=x
		cluster[i,1]=i

	cluster = cluster[cluster[:,0].argsort()]
	ans=[[],[],[],[],[],[],[],[]]
	i=0
	j=0
	# print(3000000)
	while 1:

		if(j+1 == data.shape[0]):
			ans[7].append(int(cluster[j,1]))
			break
		else:
			ans[i].append(int(cluster[j,1]))
			if (cluster[j,0]!=cluster[j+1,0]):
				
				i=i+1
		j=j+1
	# print(4000000)
	finans= sorted(ans, key=lambda x: x[0])
	# print(5000000)

	

	return finans




edge = allog_clustering(data2)
print(edge)
f = open("agglomerative.txt",'w')
for i in range(8):
	f.write(str(edge[i])[1:-1])
	f.write("\n\n\n")
f.close()

# --------------------------------Part c --------------------------------------

def kmeans_clustering(data):
	cluster_centroids = np.random.random((8,data.shape[1]))

	# # repeat
	t=0
	while(t<15):

		# cluster assignment
		clusters = [[] for x in range(8)]
		err = 0
		for i in range(data.shape[0]):
			min_ = 1*math.inf
			for j in range(8):
				temp = np.exp(-np.dot(cluster_centroids[j],data[i]))
				if(min_ > temp):
					min_ = temp
					index = j
			clusters[index].append(i)
			err = err + np.linalg.norm(data[i] - cluster_centroids[j])

		# error (SSE)
		print("SSE is  -  %.5f"%(err))

		# move centroid
		for i in range(8):
			sum_ = np.zeros((1,data.shape[1]))
			for j in range(len(clusters[i])):
				sum_ = sum_ + data[clusters[i][j]]
			cluster_centroids[i] = sum_/len(clusters[i])

		t = t+1
	finans= sorted(clusters, key=lambda x: x[0])
	
	return finans

clusters = kmeans_clustering(data2)


f = open("kmeans.txt",'w')
for i in range(8):
	f.write(str(clusters[i])[1:-1])
	f.write("\n\n\n")
f.close()



#----------------------- Part d ---------------------------------------------


pca = PCA(n_components=100)
data3 = pca.fit_transform(data2)
# data3 = pca.transform(data2)

edge1 = allog_clustering(data3)
print(edge1)
f = open("agglomerative_reduced.txt",'w')
for i in range(8):
	f.write(str(edge1[i])[1:-1])
	f.write("\n\n\n")
f.close()


clusters1 = kmeans_clustering(data3)


f = open("kmeans_reduced.txt",'w')
for i in range(8):
	f.write(str(clusters1[i])[1:-1])
	f.write("\n\n\n")
f.close()





