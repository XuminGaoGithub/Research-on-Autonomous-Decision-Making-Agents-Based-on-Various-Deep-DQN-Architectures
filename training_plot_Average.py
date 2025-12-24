# coding=utf-8
import csv
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import logging
import matplotlib.ticker as mticker

#plot the Average training curves of different agents

def plot1(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)

    plt.xticks(fontname="Times New Roman", fontsize=15)
    plt.yticks(fontname="Times New Roman", fontsize=15)
    plt.xlabel('Episode', fontdict={"family": "Times New Roman", "size": 20})
    plt.ylabel('Ave_Reward', fontdict={"family": "Times New Roman", "size": 20})

    #axs.set(xlabel='Episode', ylabel='Ave_Reward')
    #axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 3], 'tab:green')

    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()

def plot2(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)

    plt.xticks(fontname="Times New Roman", fontsize=15)
    plt.yticks(fontname="Times New Roman", fontsize=15)
    plt.xlabel('Episode', fontdict={"family": "Times New Roman", "size": 20})
    plt.ylabel('Ave_Steps', fontdict={"family": "Times New Roman", "size": 20})

    #axs.set(xlabel='Episode', ylabel='Ave_Steps')
    #axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 4], 'tab:green')
    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()

def plot3(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)

    plt.xticks(fontname="Times New Roman", fontsize=15)
    plt.yticks(fontname="Times New Roman", fontsize=15)
    plt.xlabel('Episode', fontdict={"family": "Times New Roman", "size": 20})
    plt.ylabel('Ave_Loss', fontdict={"family": "Times New Roman", "size": 20})

    #axs.set(xlabel='Episode', ylabel='Ave_Loss')
    #axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 5], 'tab:green')
    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()

def plot4(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)

    plt.xticks(fontname="Times New Roman", fontsize=15)
    plt.yticks(fontname="Times New Roman", fontsize=15)
    plt.xlabel('Episode', fontdict={"family": "Times New Roman", "size": 20})
    plt.ylabel('Ave_Q', fontdict={"family": "Times New Roman", "size": 20})

    #axs.set(xlabel='Episode', ylabel='Ave_Q')
    #axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 6], 'tab:green')
    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()


# when generates log after training, needs to modify the name of log file to the corresponding name manually
# which is shown as below
#
#Read dataset to numpy.array and remove the first row of the dataset (variable name)
data1 = genfromtxt("log_nature_dqn.csv") #read dataset by genfromtxt
#print('type(data):',type(data))
data1 = data1[1:3000][:] # Remove the first row of the array (variable name)

data2 = genfromtxt("log_double_dqn.csv") #read dataset by genfromtxt
data2 = data2[1:3000][:] # Remove the first row of the array (variable name)

data3 = genfromtxt("log_dueling_dqn.csv") #read dataset by genfromtxt
data3 = data3[1:3000][:] # Remove the first row of the array (variable name)

data4 = genfromtxt("log_nature_dqn_rnn.csv") #read dataset by genfromtxt
data4 = data4[1:3000][:] # Remove the first row of the array (variable name)

data5 = genfromtxt("log_nature_dqn_transformer.csv") #read dataset by genfromtxt
data5 = data5[1:3000][:] # Remove the first row of the array (variable name)

Ave_data = (data1+data2+data3+data4+data5)/float(5)


plot1(Ave_data)
plot2(Ave_data)
plot3(Ave_data)
plot4(Ave_data)
