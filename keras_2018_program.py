# %%
import tensorflow as tf
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import itertools
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import layers


def get_data():

    values=[]
    with open(r"C:\Users\Crystal\Desktop\Programs\Neural-Network-MxNx1\model_parameters.txt") as fp:
        external_file_params=dict() 
        for line in fp:
            a=line.split('-')
            if (a[0]=='external'):
                external_file_params['external_file']=a[3]
                external_file_params['feature list']=a[4]
                external_file_params['target']=a[5]
                # print(external_file_params)
                break
            else:
                return None

    ext_target=external_file_params['target'].rstrip('\n')
    the_file=external_file_params['external_file'] 
    my_features=external_file_params['feature list'].split(',') 
    usecols=my_features.insert(len(my_features),ext_target)
    my_data=pd.read_csv(the_file,usecols=usecols)
    my_features.pop()

    my_data=my_data.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))

    
    X_data=my_data[my_features]
    Y_data=my_data[ext_target]

    return X_data,Y_data
        




# %%
# data,labels=make_moons(40,noise=.1)
# data,labels=make_circles(40,noise=.1)

data,labels=get_data()

print(data)
print(len(data))
print(data.shape[1])

# %%
data, x_test, labels, y_test=train_test_split(data,labels,test_size=0.20,random_state=42)

print(data.shape[1])
# %%
##PLOT TRAINING & TEST DATA
if (data.shape[1]==2):
    xx_1=pd.DataFrame(data[:,0])
    xx_2=pd.DataFrame(data[:,1])
    y=pd.DataFrame(labels)

    # xx_1=data[:,0]
    # xx_2=data[:,1]
    # y=labels


    print(xx_1.head(),'\n',xx_2.head(),'\n',y.head())
    print(list(xx_1[y==0]))
    print(list(xx_1[y==1]))
    plt.figure(figsize=(15,10))

    plt.scatter(xx_1[y==0],xx_2[y==0],color='b')
    plt.scatter(xx_1[y==1],xx_2[y==1],color='r')

    # plt.scatter(data[:,0],data[:,1],s=40,c=labels,cmap=plt.cm.Spectral)
    plt.scatter(x_test[:,0],x_test[:,1],color='black')

    # plt.xlabel()
    # plt.ylabel()
    # plt.xlim(xx_1.min(),xx_1.max())
    # plt.ylim(xx_2.min(),xx_2.max())
    plt.grid()
    plt.show()
# %%

##MODEL (NEURAL NETWORK TOPOGRAPHY)

num_inputs=data.shape[1]
# model = Sequential([
#     Dense(6, input_shape=(num_inputs,)),Activation('tanh'),
#     Dense(1),Activation('sigmoid')])

# %%
model=Sequential()
model.add(layers.Dense(7,activation='relu',input_shape=(num_inputs,)))
model.add(layers.Dense(7,activation='relu'))
model.add(layers.Dense(7,activation='relu'))
model.add(layers.Dense(7,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


# %%
sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

              ##loss='binary_crossentropy'
# %%
batch_size=1
epochs=100
# %%
model.summary()

# %%
model.fit(x=data,y=labels,validation_split=0.0,epochs=epochs,batch_size=batch_size)
# %%
score = model.evaluate(x_test, y_test, batch_size=batch_size)
# %%
print(score)
# %%
predict=model.predict(x=x_test,batch_size=batch_size)
# %%
for i in predict:
    print(i)
# %%
rounded_predict=model.predict_classes(x=x_test,batch_size=batch_size)
# %%
for i in rounded_predict:
    print(i)
# %%
print(x_test)

y_predict=pd.DataFrame(predict)
y_round_predict=pd.DataFrame(rounded_predict)

# %%
if (data.shape[1]==2):
    xx_1=pd.DataFrame(data[:,0])
    xx_2=pd.DataFrame(data[:,1])
    y=pd.DataFrame(labels)

    xx_test_1=pd.DataFrame(x_test[:,0])
    xx_test_2=pd.DataFrame(x_test[:,1])

    # y_predict=pd.DataFrame(predict)
    # y_round_predict=pd.DataFrame(rounded_predict)

    plt.figure(figsize=(15,10))
    #cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])¶
    cmap_light = ListedColormap(['#FFAAAA','#AAAAFF'])

    #cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])¶
    cmap_bold = ListedColormap(['#FF0000','#0000FF'])

    h=.02

    ##Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].¶
    x1=data[:,0]
    x2=data[:,1]
    x_min,x_max = x1.min()-1,x1.max()+1
    y_min,y_max = x2.min()-1,x2.max()+1 
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    ##the_predict = feedforward(np.c[xx.ravel(),yy.ravel()],syn0,syn1,activation_function)¶
    the_predict=model.predict(x=np.c_[xx.ravel(),yy.ravel()],batch_size=1)

    ##Put the result into a color plot¶
    Z = the_predict.reshape(xx.shape)
    plt.figure(figsize=(15,15))
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())

    #plt.title('e={} alpha={} n={} accuracy={} activation={}'.format(num_epoch,Alpha,n_hidden,model_accuracy,activation_function))¶
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

    plt.scatter(xx_1[y==0],xx_2[y==0],color='b')
    plt.scatter(xx_1[y==1],xx_2[y==1],color='r')

    ##plt.scatter(data[:,0],data[:,1],s=40,c=labels,cmap=plt.cm.Spectral)¶
    cm=plt.cm.get_cmap('RdYlBu_r')


    plt.scatter(x_test[:,0],x_test[:,1],cmap=cm,vmin=0,vmax=1,c=predict,marker='D')
    plt.scatter(xx_test_1[y_round_predict==0],xx_test_2[y_round_predict==0],cmap=cm,vmin=0,vmax=1,c=y_predict,marker='D')
    plt.scatter(xx_test_1[y_round_predict==1],xx_test_2[y_round_predict==1],cmap=cm,vmin=0,vmax=1,c=y_predict,marker='D')
    plt.colorbar()

    # plt.xlabel()
    # plt.ylabel()
    # plt.xlim(xx_1.min(),xx_1.max())
    # plt.ylim(xx_2.min(),xx_2.max())

    plt.grid()
    plt.show()


# %%
model.summary()
model.output_shape
model.get_config()
##CONFUSION MATRIX
con_mat=confusion_matrix(y_test,y_round_predict)
print(con_mat)
# %%
print(score)
