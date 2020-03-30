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



	def lasso_regularised_descent(self, lmda):
		global fd
		alpha = 0.05/self.train_l.shape[0]
		# print(self.parameter)
		# print(self.train_l.shape)
		self.train_l = self.train_l.reshape((-1, 1))
		self.test_l = self.test_l.reshape((-1,1))
		for i in range(7000):
			self.parameter = self.parameter - alpha* lmda/2- alpha * np.matmul(self.train_fpow.transpose() , (np.matmul(self.train_fpow , self.parameter) - self.train_l))


		# print(self.parameter)
		sq_error_train = (np.matmul(self.train_fpow , self.parameter) - self.train_l)
		sq_error_train = np.power(sq_error_train ,2).reshape((-1,1))
		# print(self.test_l.shape)
		# print(self.test_l.shape)
		sq_error_test = (np.matmul(self.test_fpow , self.parameter) - self.test_l)
		# print(sq_error_test.shape)
		sq_error_test = np.power(sq_error_test ,2).reshape((-1,1))
		# print(sq_error_train.shape)
		# print(train_data.shape)
		# print(sq_error_test.shape)
		# print(test_data.shape)
		plt.scatter(train_data[:, 0], sq_error_train[:,0])
		plt.scatter(test_data[:, 0], sq_error_test[:, 0])
		plt.legend(['train error', 'test error'], loc='upper left')
		plt.ylabel('Squared error')
		plt.xlabel('Feature')
		plt.title("Errors in train and test data predicted values for lasso_regularised_descent ")
		
		plt.show()
		fd += 1
		sq_error_train = np.sum(sq_error_train)/2/self.train_l.shape[0]
		sq_error_test = np.sum(sq_error_test)/2/self.train_l.shape[0]
		print("For lasso_regularised_descent with n = ", self.n ,"and lambda = ",lmda ,"train error = " ,sq_error_train," and test error = " ,sq_error_test)


	def ridge_regularised_descent(self, lmda):
		global fd
		alpha = 0.05/self.train_l.shape[0]
		# print(self.parameter)
		# print(self.train_l.shape)
		self.train_l = self.train_l.reshape((-1, 1))
		self.test_l = self.test_l.reshape((-1,1))
		for i in range(7000):
			self.parameter = self.parameter - alpha* lmda *self.parameter - alpha * np.matmul(self.train_fpow.transpose() , (np.matmul(self.train_fpow , self.parameter) - self.train_l))


		# print(self.parameter)
		sq_error_train = (np.matmul(self.train_fpow , self.parameter) - self.train_l)
		sq_error_train = np.power(sq_error_train ,2).reshape((-1,1))
		# print(self.test_l.shape)
		# print(self.test_l.shape)
		sq_error_test = (np.matmul(self.test_fpow , self.parameter) - self.test_l)
		# print(sq_error_test.shape)
		sq_error_test = np.power(sq_error_test ,2).reshape((-1,1))
		# print(sq_error_train.shape)
		# print(train_data.shape)
		# print(sq_error_test.shape)
		# print(test_data.shape)
		plt.scatter(train_data[:, 0], sq_error_train[:,0])
		plt.scatter(test_data[:, 0], sq_error_test[:, 0])
		plt.legend(['train error', 'test error'], loc='upper left')
		plt.ylabel('Squared error')
		plt.xlabel('Feature')
		plt.title("Errors in train and test data predicted values for ridge_regularised_descent ")
		
		plt.show()
		fd += 1
		sq_error_train = np.sum(sq_error_train)/2/self.train_l.shape[0]
		sq_error_test = np.sum(sq_error_test)/2/self.train_l.shape[0]
		print("For ridge_regularised_descent with n = ", self.n ,"and lambda = ",lmda ,"train error = " ,sq_error_train," and test error = " ,sq_error_test)
		

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
		plt.scatter(test_data[:, 0], test_data_pred[:,0])
		plt.scatter(test_data[:, 0], test_data[:, 1])
		plt.legend(['predicted labels', 'origional labels'], loc='upper left')
		plt.ylabel('Label')
		plt.xlabel('Feature')
		plt.title("Test data label vs Feature for origional and predicted curves")
		
		plt.show()
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







modellreg1 = Linear_Regression(2)
modellreg2 = Linear_Regression(2)
modellreg3 = Linear_Regression(2)
modellreg4 = Linear_Regression(2)
modellreg1.lasso_regularised_descent(0.25)
modellreg2.lasso_regularised_descent(0.5)
modellreg3.lasso_regularised_descent(0.75)
modellreg4.lasso_regularised_descent(1)


modellreg5 = Linear_Regression(9)
modellreg6 = Linear_Regression(9)
modellreg7 = Linear_Regression(9)
modellreg8 = Linear_Regression(9)
modellreg5.lasso_regularised_descent(0.25)
modellreg6.lasso_regularised_descent(0.5)
modellreg7.lasso_regularised_descent(0.75)
modellreg8.lasso_regularised_descent(1)


modelrreg1 = Linear_Regression(2)
modelrreg2 = Linear_Regression(2)
modelrreg3 = Linear_Regression(2)
modelrreg4 = Linear_Regression(2)
modelrreg1.ridge_regularised_descent(0.25)
modelrreg2.ridge_regularised_descent(0.5)
modelrreg3.ridge_regularised_descent(0.75)
modelrreg4.ridge_regularised_descent(1)


modelrreg5 = Linear_Regression(9)
modelrreg6 = Linear_Regression(9)
modelrreg7 = Linear_Regression(9)
modelrreg8 = Linear_Regression(9)
modelrreg5.ridge_regularised_descent(0.25)
modelrreg6.ridge_regularised_descent(0.5)
modelrreg7.ridge_regularised_descent(0.75)
modelrreg8.ridge_regularised_descent(1)
# model.predict()
# model.predict_train()




