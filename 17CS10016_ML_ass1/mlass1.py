import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fd=1
def get_data():
	dat=pd.read_csv('train.csv')
	dat=np.array(dat.iloc[:,:])
	tdat=pd.read_csv('test.csv')
	tdat=np.array(tdat.iloc[:,:])
	# print(dat)
	return dat,tdat

train_data , test_data = get_data()


def plot():
	#to plot data vs label
	global fd
	plt.scatter(train_data[:, 0], train_data[:, 1])
	plt.ylabel('Label')
	plt.xlabel('Feature')
	plt.title("Training data label vs Feature")
	
	plt.show()
	fd += 1
	plt.scatter(test_data[:, 0], test_data[:, 1])
	plt.ylabel('Label')
	plt.xlabel('Feature')
	plt.title("Test data label vs Feature")
	
	plt.show()
	fd += 1

plot()

class Linear_Regression:
	def __init__(self,n):
		self.train_f=train_data[:, 0]
		self.train_l=train_data[:, 1]
		self.test_f=test_data[:, 0]
		self.test_l=test_data[:, 1]
		self.train_l.reshape((-1,1))
		self.test_l.reshape((-1,1))
		self.n=n
		self.train_fpow=np.power(self.train_f,0).reshape((-1,1))
		self.test_fpow=np.power(self.test_f,0).reshape((-1,1))
		for i in range(n+1):
			if i is 0:
				pass
				# print(self.train_fpow.shape)
			else:
				self.train_fpow = np.hstack((self.train_fpow, np.power(self.train_f,i).reshape((-1,1))))


		for i in range(n+1):
			if i is 0:
				pass
				# print(self.test_fpow.shape)
			else:
				self.test_fpow = np.hstack((self.test_fpow, np.power(self.test_f,i).reshape((-1,1))))


		# self.parameter = np.random.normal(size=(n+1,1))
		self.parameter = np.zeros((n+1,1))
		# print(self.parameter)
		# print(self.train_f)
		# print(self.train_fpow)

	def grad_descent(self):
		global fd
		alpha = 0.05/self.train_l.shape[0]
		# print(self.parameter)
		# print(self.train_l.shape)
		self.train_l = self.train_l.reshape((-1, 1))
		for i in range(7000):
			self.parameter = self.parameter - alpha * np.matmul(self.train_fpow.transpose() , (np.matmul(self.train_fpow , self.parameter) - self.train_l))

		print("Parameters for n = %d",self.n)
		print(self.parameter)

		sq_error = (np.matmul(self.train_fpow , self.parameter) - self.train_l)
		sq_error = np.matmul(sq_error.transpose(),sq_error)
		sq_error = sq_error/2/self.train_l.shape[0]
		print("Squared error in training data is:")
		print(sq_error[0][0])
		return sq_error[0][0]



	
	def predict(self):
		global fd
		test_data_pred = np.matmul(self.test_fpow , self.parameter)
		self.test_l = self.test_l.reshape((-1, 1))
		print("Predicted labels for test data are as follows")
		print(test_data_pred[:,0] )
		sq_error = (test_data_pred - self.test_l)
		sq_error = np.matmul(sq_error.transpose(),sq_error)
		sq_error = sq_error/2/self.test_l.shape[0]
		print("Squared Error in test data is : ")
		print(sq_error[0][0])
		# plt.scatter(test_data[:, 0], test_data_pred[:,0])
		# plt.scatter(test_data[:, 0], test_data[:, 1])
		# plt.legend(['predicted labels', 'origional labels'], loc='upper left')
		# plt.ylabel('Label')
		# plt.xlabel('Feature')
		# plt.title("Test data label vs Feature for origional and predicted curves")
		
		# plt.show()
		fd += 1
		return sq_error[0][0]

	def predict_train(self):
		global fd
		train_data_pred = np.matmul(self.train_fpow , self.parameter)
		self.train_l = self.train_l.reshape((-1, 1))
		print("Predicted labels for train data are as follows")
		print(train_data_pred[:,0] )
		sq_error = (train_data_pred - self.train_l)
		sq_error = np.matmul(sq_error.transpose(),sq_error)
		sq_error = sq_error/2/self.train_l.shape[0]
		print("Squared Error in train data is : ")
		print(sq_error[0][0])
		plt.scatter(train_data[:, 0], train_data_pred[:,0])
		plt.scatter(train_data[:, 0], train_data[:, 1])
		plt.legend(['predicted labels', 'origional labels'], loc='upper left')
		plt.ylabel('Label')
		plt.xlabel('Feature')
		plt.title("train data label vs Feature for origional and predicted values")
		
		plt.show()
		fd += 1






model1 = Linear_Regression(1)
model2 = Linear_Regression(2)
model3 = Linear_Regression(3)
model4 = Linear_Regression(4)
model5 = Linear_Regression(5)
model6 = Linear_Regression(6)
model7 = Linear_Regression(7)
model8 = Linear_Regression(8)
model9 = Linear_Regression(9)

squr_error = np.zeros((9,2))
x = np.zeros((9,1))
for i in range(9):
	x[i][0]=i+1
squr_error[0][0] = model1.grad_descent()
squr_error[0][1] = model1.predict()
squr_error[1][0] = model2.grad_descent()
squr_error[1][1] = model2.predict()
squr_error[2][0] = model3.grad_descent()
squr_error[2][1] = model3.predict()
squr_error[3][0] = model4.grad_descent()
squr_error[3][1] = model4.predict()
squr_error[4][0] = model5.grad_descent()
squr_error[4][1] = model5.predict()
squr_error[5][0] = model6.grad_descent()
squr_error[5][1] = model6.predict()
squr_error[6][0] = model7.grad_descent()
squr_error[6][1] = model7.predict()
squr_error[7][0] = model8.grad_descent()
squr_error[7][1] = model8.predict()
squr_error[8][0] = model9.grad_descent()
squr_error[8][1] = model9.predict()

