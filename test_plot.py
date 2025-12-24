# coding=utf-8
import csv
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import logging
import matplotlib.ticker as mticker

def plot1(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)
    axs.set(xlabel='Episode', ylabel='Ave_Reward')
    axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 3], 'tab:green')

    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()

def plot2(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)
    axs.set(xlabel='Episode', ylabel='Ave_Steps')
    axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 4], 'tab:green')
    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()

def plot3(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)
    axs.set(xlabel='Episode', ylabel='Ave_Loss')
    axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 5], 'tab:green')
    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()

def plot4(logged_data):
    logged_data = np.array(logged_data)
    fig, axs = plt.subplots(1)
    axs.set(xlabel='Episode', ylabel='Ave_Q')
    axs.set_yscale('log')
    axs.plot(logged_data[:, 0], logged_data[:, 6], 'tab:green')
    x_major_locator = MultipleLocator(1000)  #
    axs.xaxis.set_major_locator(x_major_locator)  #

    plt.show()



#Read dataset to numpy.array and remove the first row of the dataset (variable name)
data = genfromtxt("log.csv") #read dataset by genfromtxt
data = data[1:][:] # Remove the first row of the array (variable name)


#plot1(data)
#plot2(data)
#plot3(data)
#plot4(data)

print("Ave_Reward,Ave_Steps,Ave_Loss,Ave_Q:",sum(data[:, 3])/ float(10),sum(data[:, 4])/ float(10),sum(data[:, 5])/ float(10),sum(data[:, 6])/ float(10))
