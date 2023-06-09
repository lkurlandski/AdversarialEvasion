#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:40:13 2023

@author: jc4303
"""

import matplotlib.pyplot as plt  # mod - for line plot of 2b
import numpy as np

#using: output/False/False/None/18.pth
 #python3 graph_data.py output/False/False/None/18.pth output/False/True/FGSM FGSM



#### graph for aversarial examples against non-adv trained model

epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

fgsm_y = [0.9875, 0.7175, 0.26, 0.11, 0.0975, 0.1, 0.1, 0.1, 0.1, 0.1]

# output/False/False/None/18.pth
# [0.94, 0.73, 0.32, 0.22, 0.1, 0.11, 0.08, 0.1]


pgd_y = [0.9875, 0.64, 0.2125, 0.12, 0.12, 0.12, 0.1125, 0.125, 0.1125, 0.115]
# output/False/False/None/18.pth
# [0.93, 0.66, 0.34, 0.18, 0.15, 0.12, 0.11, 0.13]

igsm_y = [0.99, 0.4825, 0.1075, 0.09, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# output/False/False/None/18.pth
# [0.84, 0.57, 0.24, 0.09, 0.09, 0.09, 0.09, 0.09]


plt.figure(figsize=(6, 4.5))
string_title_acc = "ae_acc" + ".eps"
string_title_acc2 = "ae_acc" + ".png"
plt.plot(epsilons, fgsm_y)
plt.plot(epsilons, pgd_y)
plt.plot(epsilons, igsm_y)
# plt.title('model accuracy')
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Epsilon", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM", "PGD", "IGSM"], loc="right")
plt.grid(linewidth=0.7)
plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(0, 1)
plt.savefig(string_title_acc, format="eps")
plt.savefig(string_title_acc2, format="png")
plt.show()
plt.clf()


#### ae training models agains benigin

std_model = 98.5

fgsm_new_y = 98.75




fgsm_pre_y = 98.25







pgd_y = 99.5

igsm_y = 99.75

benign_val = [std_model, fgsm_new_y, fgsm_pre_y, pgd_y, igsm_y]

models = ("Clean", "FGSM/n", "FGSM/p", "PGD", "IGSM")
x_pos = np.arange(len(models))
plt.figure(figsize=(6, 4.5))
string_title_acc = "at_benign" + ".eps"
string_title_acc2 = "at_benign" + ".png"
# plt.title('model accuracy')
plt.bar(x_pos, benign_val, color=["black", "red", "green", "blue", "orange"])
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.xlabel("Model", fontsize=16)
plt.xticks(x_pos, models)
plt.yticks(np.arange(80,105, 5))
plt.gca().set_ylim(bottom=80, top=100)
plt.savefig(string_title_acc, format="eps")
plt.savefig(string_title_acc2, format="png")
plt.show()
plt.clf()




#### at_adv graph- graph for aversarial trained models against same attacks
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # placeholder


# output/False/True/FGSM/18.pth
# [0.987, 0.985, 0.981, 0.978, 0.976, 0.976, 0.768, 0.338]
fgsm_new_y = [0.9875, 0.987, 0.985, 0.981, 0.978, 0.976, 0.976, 0.768, 0.338] # placeholder
# output/False/True/FGSM/1.pth
# [0.964, 0.965, 0.963, 0.953, 0.946, 0.941, 0.878, 0.666]
# output/False/True/FGSM/10.pth
# [0.977, 0.986, 0.973, 0.965, 0.969, 0.962, 0.594, 0.179]
# output/False/True/FGSM/11.pth
# [0.985, 0.988, 0.982, 0.973, 0.98, 0.976, 0.797, 0.294]
# output/False/True/FGSM/12.pth
# [0.989, 0.989, 0.98, 0.978, 0.975, 0.97, 0.504, 0.202]
# output/False/True/FGSM/13.pth
# [0.987, 0.992, 0.981, 0.979, 0.978, 0.965, 0.518, 0.21]
# output/False/True/FGSM/14.pth
# [0.987, 0.99, 0.98, 0.98, 0.983, 0.971, 0.564, 0.222]
# output/False/True/FGSM/15.pth
# [0.981, 0.973, 0.972, 0.962, 0.967, 0.978, 0.845, 0.318]
# output/False/True/FGSM/16.pth
# [0.988, 0.993, 0.982, 0.979, 0.977, 0.973, 0.745, 0.259]
# output/False/True/FGSM/17.pth
# [0.989, 0.991, 0.98, 0.979, 0.979, 0.965, 0.57, 0.235]
# output/False/True/FGSM/18.pth
# [0.987, 0.985, 0.981, 0.978, 0.976, 0.976, 0.768, 0.338]
# output/False/True/FGSM/19.pth
# [0.99, 0.991, 0.981, 0.982, 0.975, 0.975, 0.68, 0.225]
# output/False/True/FGSM/2.pth
# [0.979, 0.978, 0.972, 0.967, 0.954, 0.948, 0.571, 0.32]
# output/False/True/FGSM/20.pth
# [0.988, 0.993, 0.98, 0.981, 0.981, 0.978, 0.588, 0.249]
# output/False/True/FGSM/3.pth
# [0.978, 0.98, 0.975, 0.969, 0.973, 0.967, 0.603, 0.3]
# output/False/True/FGSM/4.pth
# [0.978, 0.983, 0.98, 0.979, 0.968, 0.962, 0.531, 0.142]
# output/False/True/FGSM/5.pth
# [0.99, 0.987, 0.977, 0.98, 0.973, 0.976, 0.709, 0.232]
# output/False/True/FGSM/6.pth
# [0.983, 0.987, 0.974, 0.968, 0.969, 0.972, 0.75, 0.327]
# output/False/True/FGSM/7.pth
# [0.99, 0.988, 0.98, 0.981, 0.974, 0.96, 0.48, 0.241]
# output/False/True/FGSM/8.pth
# [0.985, 0.987, 0.974, 0.98, 0.976, 0.956, 0.527, 0.265]
# output/False/True/FGSM/9.pth
# [0.989, 0.987, 0.98, 0.975, 0.979, 0.982, 0.644, 0.253]



# output/True/True/FGSM/18.pth
# [0.987, 0.988, 0.991, 0.982, 0.983, 0.983, 0.848, 0.373]
fgsm_pre_y = [0.9825, 0.987, 0.988, 0.991, 0.982, 0.983, 0.983, 0.848, 0.373] # placeholder
# output/True/True/FGSM/1.pth
# [0.973, 0.969, 0.966, 0.955, 0.95, 0.936, 0.892, 0.727]
# output/True/True/FGSM/10.pth
# [0.983, 0.984, 0.993, 0.982, 0.981, 0.978, 0.817, 0.302]
# output/True/True/FGSM/11.pth
# [0.982, 0.984, 0.99, 0.984, 0.984, 0.98, 0.827, 0.346]
# output/True/True/FGSM/12.pth
# [0.986, 0.988, 0.993, 0.983, 0.983, 0.975, 0.747, 0.274]
# output/True/True/FGSM/13.pth
# [0.986, 0.988, 0.992, 0.987, 0.987, 0.978, 0.717, 0.288]
# output/True/True/FGSM/14.pth
# [0.99, 0.987, 0.992, 0.989, 0.982, 0.974, 0.685, 0.273]
# output/True/True/FGSM/15.pth
# [0.984, 0.986, 0.991, 0.983, 0.98, 0.981, 0.762, 0.263]
# output/True/True/FGSM/16.pth
# [0.986, 0.988, 0.992, 0.987, 0.987, 0.98, 0.706, 0.271]
# output/True/True/FGSM/17.pth
# [0.986, 0.99, 0.994, 0.988, 0.98, 0.97, 0.616, 0.239]
# output/True/True/FGSM/18.pth
# [0.987, 0.988, 0.991, 0.982, 0.983, 0.983, 0.848, 0.373]
# output/True/True/FGSM/19.pth
# [0.99, 0.988, 0.996, 0.983, 0.987, 0.967, 0.651, 0.264]
# output/True/True/FGSM/2.pth
# [0.98, 0.977, 0.977, 0.969, 0.97, 0.961, 0.756, 0.293]
# output/True/True/FGSM/20.pth
# [0.985, 0.986, 0.992, 0.987, 0.981, 0.977, 0.7, 0.309]
# output/True/True/FGSM/3.pth
# [0.985, 0.981, 0.981, 0.969, 0.974, 0.971, 0.804, 0.34]
# output/True/True/FGSM/4.pth
# [0.985, 0.982, 0.986, 0.972, 0.979, 0.959, 0.702, 0.275]
# output/True/True/FGSM/5.pth
# [0.988, 0.985, 0.987, 0.984, 0.986, 0.96, 0.564, 0.17]
# output/True/True/FGSM/6.pth
# [0.985, 0.984, 0.99, 0.982, 0.98, 0.957, 0.58, 0.182]
# output/True/True/FGSM/7.pth
# [0.982, 0.982, 0.991, 0.986, 0.983, 0.967, 0.576, 0.28]
# output/True/True/FGSM/8.pth
# [0.983, 0.99, 0.991, 0.984, 0.981, 0.948, 0.59, 0.266]
# output/True/True/FGSM/9.pth
# [0.983, 0.986, 0.992, 0.981, 0.984, 0.964, 0.653, 0.31]







# output/True/True/PGD/18.pth
# [1.0, 1.0, 0.99, 0.98, 1.0, 0.96, 0.43, 0.23]
pgd_y = [0.995, 1.0, 1.0, 0.99, 0.98, 1.0, 0.96, 0.43, 0.23] # placeholder


# output/True/True/PGD/1.pth
# [1.0, 0.98, 0.97, 0.91, 0.98, 0.95, 0.92, 0.54]
# output/True/True/PGD/10.pth
# [1.0, 1.0, 0.99, 0.97, 0.99, 0.97, 0.39, 0.16]
# output/True/True/PGD/11.pth
# [1.0, 1.0, 0.99, 0.98, 0.99, 0.97, 0.42, 0.17]
# output/True/True/PGD/12.pth
# [1.0, 1.0, 0.99, 0.98, 0.99, 0.97, 0.37, 0.13]
# output/True/True/PGD/13.pth
# [1.0, 1.0, 0.99, 0.98, 1.0, 0.97, 0.41, 0.16]
# output/True/True/PGD/14.pth
# [1.0, 0.99, 0.98, 0.97, 0.99, 0.98, 0.41, 0.24]
# output/True/True/PGD/15.pth
# [1.0, 1.0, 0.99, 0.96, 0.97, 0.97, 0.65, 0.28]
# output/True/True/PGD/16.pth
# [1.0, 1.0, 0.99, 0.98, 0.99, 0.97, 0.45, 0.19]
# output/True/True/PGD/17.pth
# [1.0, 1.0, 0.98, 0.97, 0.96, 0.97, 0.43, 0.17]
# output/True/True/PGD/18.pth
# [1.0, 1.0, 0.99, 0.98, 1.0, 0.96, 0.43, 0.23]
# output/True/True/PGD/19.pth
# [1.0, 1.0, 0.99, 0.98, 1.0, 0.97, 0.35, 0.19]
# output/True/True/PGD/2.pth
# [1.0, 0.98, 0.98, 0.95, 0.96, 0.86, 0.37, 0.16]
# output/True/True/PGD/20.pth
# [1.0, 1.0, 1.0, 0.98, 0.99, 0.97, 0.49, 0.26]
# output/True/True/PGD/3.pth
# [1.0, 0.98, 0.96, 0.95, 0.98, 0.97, 0.87, 0.37]
# output/True/True/PGD/4.pth
# [1.0, 1.0, 0.98, 0.97, 0.99, 0.96, 0.57, 0.26]
# output/True/True/PGD/5.pth
# [0.99, 1.0, 1.0, 0.98, 0.99, 0.98, 0.43, 0.17]
# output/True/True/PGD/6.pth
# [1.0, 0.99, 0.99, 0.96, 1.0, 0.98, 0.47, 0.22]
# output/True/True/PGD/7.pth
# [1.0, 1.0, 0.99, 0.98, 0.99, 0.95, 0.37, 0.16]
# output/True/True/PGD/8.pth
# [1.0, 0.99, 1.0, 0.98, 0.99, 0.94, 0.47, 0.29]
# output/True/True/PGD/9.pth
# [1.0, 1.0, 0.98, 0.96, 0.98, 0.97, 0.45, 0.25]









# output/True/True/IGSM/18.pth
# [0.98, 0.99, 0.99, 1.0, 0.96, 0.95, 0.52, 0.15]
igsm_y = [0.9975, 0.98, 0.99, 0.99, 1.0, 0.96, 0.95, 0.52, 0.15]  # placeholder
# output/True/True/IGSM/1.pth
# [0.96, 0.98, 0.95, 0.97, 0.93, 0.96, 0.86, 0.45]
# output/True/True/IGSM/10.pth
# [0.99, 0.99, 1.0, 0.99, 0.97, 0.95, 0.44, 0.21]
# output/True/True/IGSM/11.pth
# [0.97, 0.99, 1.0, 0.99, 0.97, 0.98, 0.57, 0.25]
# output/True/True/IGSM/12.pth
# [0.98, 0.98, 1.0, 0.97, 0.96, 0.96, 0.53, 0.19]
# output/True/True/IGSM/13.pth
# [0.98, 0.99, 1.0, 1.0, 0.96, 0.93, 0.43, 0.15]
# output/True/True/IGSM/14.pth
# [0.99, 0.99, 1.0, 0.97, 0.95, 0.97, 0.59, 0.24]
# output/True/True/IGSM/15.pth
# [0.97, 0.98, 1.0, 0.99, 0.94, 0.97, 0.44, 0.2]
# output/True/True/IGSM/16.pth
# [1.0, 1.0, 1.0, 0.98, 0.96, 0.9, 0.28, 0.11]
# output/True/True/IGSM/17.pth
# [0.99, 0.99, 1.0, 1.0, 0.98, 0.96, 0.42, 0.12]
# output/True/True/IGSM/18.pth
# [0.98, 0.99, 0.99, 1.0, 0.96, 0.95, 0.52, 0.15]
# output/True/True/IGSM/19.pth
# [0.98, 1.0, 0.99, 0.99, 0.98, 0.95, 0.39, 0.14]
# output/True/True/IGSM/2.pth
# [0.96, 0.95, 0.95, 0.96, 0.96, 0.92, 0.47, 0.15]
# output/True/True/IGSM/20.pth
# [0.98, 0.99, 1.0, 0.99, 0.96, 0.96, 0.37, 0.16]
# output/True/True/IGSM/3.pth
# [0.95, 0.95, 0.95, 0.96, 0.95, 0.97, 0.83, 0.24]
# output/True/True/IGSM/4.pth
# [0.97, 0.97, 0.99, 0.97, 0.97, 0.96, 0.62, 0.17]
# output/True/True/IGSM/5.pth
# [0.98, 0.97, 0.99, 0.97, 0.99, 0.94, 0.66, 0.13]
# output/True/True/IGSM/6.pth
# [0.98, 0.99, 0.99, 0.98, 0.96, 0.87, 0.41, 0.1]
# output/True/True/IGSM/7.pth
# [0.98, 0.99, 1.0, 0.98, 0.96, 0.89, 0.34, 0.15]
# output/True/True/IGSM/8.pth
# [1.0, 1.0, 0.99, 0.99, 0.96, 0.96, 0.49, 0.2]
# output/True/True/IGSM/9.pth
# [0.98, 1.0, 1.0, 0.97, 0.97, 0.96, 0.6, 0.22]


plt.figure(figsize=(6, 4.5))
string_title_acc = "at_adv" + ".eps"
string_title_acc2 = "at_adv" + ".png"
plt.plot(epsilons, fgsm_new_y)
plt.plot(epsilons, fgsm_pre_y)
plt.plot(epsilons, pgd_y)
plt.plot(epsilons, igsm_y)
# plt.title('model accuracy')
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Epsilon", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM - new model", "FGSM - pre-trained", "PGD", "IGSM"], loc="lower left")
plt.grid(linewidth=0.7)
plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(0, 0.4)
plt.savefig(string_title_acc, format="eps")
plt.savefig(string_title_acc2, format="png")
plt.show()
plt.clf()






#### at_cross_Attac
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # placeholder

# output/True/True/PGD/18.pth
# [0.9933333333333333, 0.9966666666666667, 0.99, 0.9833333333333333, 0.9833333333333333, 0.9833333333333333, 0.5833333333333334, 0.22333333333333333]
fgsm_pgd =  [0.9875, 0.9975, 0.9925, 0.985, 0.9925, 0.975, 0.97, 0.5875, 0.265] # fgsm attacks pgd
# output/True/True/PGD/1.pth
# [0.97, 0.98, 0.9433333333333334, 0.93, 0.9266666666666666, 0.9, 0.8, 0.47333333333333333]
# output/True/True/PGD/10.pth
# [0.9866666666666667, 0.9933333333333333, 0.98, 0.98, 0.97, 0.9533333333333334, 0.49333333333333335, 0.19]
# output/True/True/PGD/11.pth
# [0.9833333333333333, 0.9933333333333333, 0.98, 0.9833333333333333, 0.9766666666666667, 0.97, 0.48333333333333334, 0.17666666666666667]
# output/True/True/PGD/12.pth
# [0.98, 0.9966666666666667, 0.9866666666666667, 0.9866666666666667, 0.98, 0.9633333333333334, 0.46, 0.18666666666666668]
# output/True/True/PGD/13.pth
# [0.9933333333333333, 0.99, 0.99, 0.98, 0.98, 0.9633333333333334, 0.5533333333333333, 0.19]
# output/True/True/PGD/14.pth
# [0.9933333333333333, 0.9966666666666667, 0.9933333333333333, 0.9833333333333333, 0.9833333333333333, 0.96, 0.5466666666666666, 0.25]
# output/True/True/PGD/15.pth
# [0.9833333333333333, 0.9966666666666667, 0.9766666666666667, 0.9833333333333333, 0.98, 0.9566666666666667, 0.7666666666666667, 0.26666666666666666]
# output/True/True/PGD/16.pth
# [0.9866666666666667, 0.99, 0.9933333333333333, 0.98, 0.9833333333333333, 0.97, 0.5633333333333334, 0.19666666666666666]
# output/True/True/PGD/17.pth
# [0.9866666666666667, 0.9933333333333333, 0.99, 0.99, 0.9833333333333333, 0.9566666666666667, 0.5166666666666667, 0.19]
# output/True/True/PGD/18.pth
# [0.9933333333333333, 0.9966666666666667, 0.99, 0.9833333333333333, 0.9833333333333333, 0.9833333333333333, 0.5833333333333334, 0.22333333333333333]
# output/True/True/PGD/19.pth
# [0.9966666666666667, 0.9966666666666667, 0.9833333333333333, 0.98, 0.99, 0.9733333333333334, 0.46, 0.18666666666666668]
# output/True/True/PGD/2.pth
# [0.98, 0.9866666666666667, 0.9666666666666667, 0.95, 0.94, 0.84, 0.44, 0.15]
# output/True/True/PGD/20.pth
# [0.9966666666666667, 0.9933333333333333, 0.9833333333333333, 0.98, 0.99, 0.9633333333333334, 0.6533333333333333, 0.3233333333333333]
# output/True/True/PGD/3.pth
# [0.9766666666666667, 0.99, 0.9533333333333334, 0.9666666666666667, 0.9466666666666667, 0.9366666666666666, 0.8566666666666667, 0.5233333333333333]
# output/True/True/PGD/4.pth
# [0.9833333333333333, 0.9866666666666667, 0.9666666666666667, 0.9733333333333334, 0.9633333333333334, 0.9433333333333334, 0.68, 0.32666666666666666]
# output/True/True/PGD/5.pth
# [0.99, 0.9866666666666667, 0.9733333333333334, 0.9866666666666667, 0.9766666666666667, 0.9533333333333334, 0.5933333333333334, 0.19666666666666666]
# output/True/True/PGD/6.pth
# [0.9833333333333333, 0.99, 0.9833333333333333, 0.98, 0.97, 0.9733333333333334, 0.6333333333333333, 0.22666666666666666]
# output/True/True/PGD/7.pth
# [0.9833333333333333, 0.9866666666666667, 0.98, 0.9866666666666667, 0.97, 0.9633333333333334, 0.47, 0.19333333333333333]
# output/True/True/PGD/8.pth
# [0.99, 0.9833333333333333, 0.9733333333333334, 0.98, 0.95, 0.9333333333333333, 0.5433333333333333, 0.31666666666666665]
# output/True/True/PGD/9.pth
# [0.98, 0.9933333333333333, 0.9866666666666667, 0.9833333333333333, 0.9733333333333334, 0.9633333333333334, 0.5433333333333333, 0.25]


# output/True/True/FGSM/18.pth
# [0.9966666666666667, 0.9766666666666667, 0.98, 0.99, 0.97, 0.9733333333333334, 0.7433333333333333, 0.3333333333333333]
pgd_fgsm =  [0.99, 0.9925, 0.985, 0.985, 0.99, 0.9725, 0.98, 0.7575, 0.33] # pgd attacks fgsm


# output/True/True/FGSM/1.pth
# [0.99, 0.9766666666666667, 0.9733333333333334, 0.9566666666666667, 0.9466666666666667, 0.9566666666666667, 0.8966666666666666, 0.6966666666666667]
# output/True/True/FGSM/10.pth
# [0.9966666666666667, 0.9866666666666667, 0.98, 0.9833333333333333, 0.98, 0.9733333333333334, 0.7366666666666667, 0.25333333333333335]
# output/True/True/FGSM/11.pth
# [1.0, 0.9866666666666667, 0.98, 0.9733333333333334, 0.9766666666666667, 0.9733333333333334, 0.6666666666666666, 0.31]
# output/True/True/FGSM/12.pth
# [0.9933333333333333, 0.9833333333333333, 0.9866666666666667, 0.9833333333333333, 0.99, 0.97, 0.6666666666666666, 0.23666666666666666]
# output/True/True/FGSM/13.pth
# [0.9966666666666667, 0.9866666666666667, 0.9866666666666667, 0.9866666666666667, 0.98, 0.9666666666666667, 0.56, 0.24]
# output/True/True/FGSM/14.pth
# [0.9933333333333333, 0.9833333333333333, 0.99, 0.9833333333333333, 0.9766666666666667, 0.9566666666666667, 0.5366666666666666, 0.24333333333333335]
# output/True/True/FGSM/15.pth
# [0.99, 0.99, 0.9833333333333333, 0.98, 0.99, 0.97, 0.62, 0.24]
# output/True/True/FGSM/16.pth
# [1.0, 0.98, 0.9933333333333333, 0.9833333333333333, 0.9833333333333333, 0.97, 0.55, 0.21333333333333335]
# output/True/True/FGSM/17.pth
# [0.9966666666666667, 0.9766666666666667, 0.9933333333333333, 0.9866666666666667, 0.9833333333333333, 0.9466666666666667, 0.4766666666666667, 0.24]
# output/True/True/FGSM/18.pth
# [0.9966666666666667, 0.9766666666666667, 0.98, 0.99, 0.97, 0.9733333333333334, 0.7433333333333333, 0.3333333333333333]
# output/True/True/FGSM/19.pth
# [0.99, 0.9866666666666667, 0.99, 0.9933333333333333, 0.9866666666666667, 0.9733333333333334, 0.55, 0.23333333333333334]
# output/True/True/FGSM/2.pth
# [0.9866666666666667, 0.9866666666666667, 0.9833333333333333, 0.9666666666666667, 0.98, 0.9633333333333334, 0.6966666666666667, 0.23]
# output/True/True/FGSM/20.pth
# [1.0, 0.98, 0.9933333333333333, 0.99, 0.9833333333333333, 0.96, 0.5833333333333334, 0.29]
# output/True/True/FGSM/3.pth
# [0.9833333333333333, 0.9766666666666667, 0.9833333333333333, 0.98, 0.9766666666666667, 0.97, 0.7, 0.27]
# output/True/True/FGSM/4.pth
# [0.99, 0.9866666666666667, 0.99, 0.9733333333333334, 0.9766666666666667, 0.9633333333333334, 0.6033333333333334, 0.22333333333333333]
# output/True/True/FGSM/5.pth
# [0.9866666666666667, 0.9866666666666667, 0.9833333333333333, 0.9833333333333333, 0.9866666666666667, 0.95, 0.49, 0.17666666666666667]
# output/True/True/FGSM/6.pth
# [0.99, 0.99, 0.99, 0.9833333333333333, 0.98, 0.9533333333333334, 0.52, 0.16333333333333333]
# output/True/True/FGSM/7.pth
# [0.9933333333333333, 0.98, 0.9866666666666667, 0.9866666666666667, 0.9833333333333333, 0.95, 0.5033333333333333, 0.2633333333333333]
# output/True/True/FGSM/8.pth
# [1.0, 0.99, 0.9866666666666667, 0.9733333333333334, 0.98, 0.94, 0.5066666666666667, 0.24]
# output/True/True/FGSM/9.pth
# [0.9966666666666667, 0.9866666666666667, 0.9933333333333333, 0.9833333333333333, 0.9833333333333333, 0.9633333333333334, 0.5666666666666667, 0.2733333333333333]





plt.figure(figsize=(6, 4.5))
string_title_acc = "pgd_cross_attack" + ".eps"
string_title_acc2 = "pgd_cross_attack" + ".png"
plt.plot(epsilons, fgsm_pgd)
plt.plot(epsilons, pgd_fgsm)
# plt.title('model accuracy')
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Epsilon", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM->PGD", "PGD->FGSM"], loc="lower left")
plt.grid(linewidth=0.7)
plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(0, 0.4)
plt.savefig(string_title_acc, format="eps")
plt.savefig(string_title_acc2, format="png")
plt.show()
plt.clf()

#### at_cross_Attac
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # placeholder


# output/True/True/IGSM/18.pth
# [0.985, 0.99, 0.9925, 0.975, 0.98, 0.985, 0.7475, 0.225]
fgsm_igsm = [0.985, 0.94, 0.7275, 0.3975, 0.2525, 0.1425, 0.11, 0.0975, 0.0975]
 #fgsm attacks igsm
# output/True/True/IGSM/1.pth
# [0.9675, 0.9625, 0.9575, 0.9625, 0.9325, 0.9325, 0.8675, 0.545]
# output/True/True/IGSM/10.pth
# [0.985, 0.9875, 0.9875, 0.98, 0.98, 0.9725, 0.6675, 0.23]
# output/True/True/IGSM/11.pth
# [0.9775, 0.9825, 0.995, 0.9775, 0.965, 0.9625, 0.7825, 0.4125]
# output/True/True/IGSM/12.pth
# [0.9825, 0.9875, 0.985, 0.975, 0.9725, 0.9725, 0.805, 0.3]
# output/True/True/IGSM/13.pth
# [0.9875, 0.9825, 0.9875, 0.9825, 0.985, 0.9675, 0.7125, 0.24]
# output/True/True/IGSM/14.pth
# [0.9825, 0.9875, 0.9975, 0.9775, 0.985, 0.975, 0.8075, 0.3725]
# output/True/True/IGSM/15.pth
# [0.98, 0.98, 0.9875, 0.9775, 0.98, 0.98, 0.745, 0.2875]
# output/True/True/IGSM/16.pth
# [0.9825, 0.9875, 0.9875, 0.9775, 0.9825, 0.9675, 0.515, 0.135]
# output/True/True/IGSM/17.pth
# [0.985, 0.99, 0.9925, 0.975, 0.9775, 0.965, 0.725, 0.2375]
# output/True/True/IGSM/18.pth
# [0.985, 0.99, 0.9925, 0.975, 0.98, 0.985, 0.7475, 0.225]
# output/True/True/IGSM/19.pth
# [0.9875, 0.9875, 0.995, 0.9825, 0.9875, 0.96, 0.6375, 0.1725]
# output/True/True/IGSM/2.pth
# [0.9575, 0.9675, 0.9725, 0.965, 0.97, 0.9525, 0.685, 0.235]
# output/True/True/IGSM/20.pth
# [0.985, 0.9825, 0.9875, 0.9725, 0.9775, 0.9625, 0.695, 0.2375]
# output/True/True/IGSM/3.pth
# [0.955, 0.9575, 0.975, 0.9575, 0.96, 0.96, 0.8625, 0.5275]
# output/True/True/IGSM/4.pth
# [0.975, 0.9775, 0.99, 0.975, 0.9725, 0.9725, 0.815, 0.29]
# output/True/True/IGSM/5.pth
# [0.9775, 0.985, 0.9925, 0.975, 0.9675, 0.9675, 0.8375, 0.29]
# output/True/True/IGSM/6.pth
# [0.9825, 0.99, 0.99, 0.985, 0.98, 0.9725, 0.665, 0.2025]
# output/True/True/IGSM/7.pth
# [0.98, 0.9875, 0.99, 0.985, 0.98, 0.955, 0.6025, 0.22]
# output/True/True/IGSM/8.pth
# [0.9825, 0.99, 0.9925, 0.985, 0.98, 0.97, 0.7075, 0.295]
# output/True/True/IGSM/9.pth
# [0.985, 0.9825, 0.985, 0.98, 0.98, 0.9725, 0.795, 0.345]



igsm_fgsm = [0.9875, 0.98, 0.9875, 0.985, 0.98, 0.975, 0.985, 0.7625, 0.31]

 # placeholder
# [0.965, 0.9625, 0.9625, 0.9675, 0.96, 0.965, 0.8875, 0.64]
# output/True/True/FGSM/10.pth
# [0.9875, 0.985, 0.98, 0.9875, 0.9825, 0.9825, 0.705, 0.2425]
# output/True/True/FGSM/11.pth
# [0.9825, 0.9925, 0.9775, 0.99, 0.9825, 0.9725, 0.68, 0.29]
# output/True/True/FGSM/12.pth
# [0.9725, 0.9925, 0.9875, 0.9875, 0.9875, 0.9775, 0.64, 0.2425]
# output/True/True/FGSM/13.pth
# [0.98, 0.99, 0.9825, 0.9875, 0.9875, 0.9725, 0.545, 0.2525]
# output/True/True/FGSM/14.pth
# [0.9825, 0.995, 0.985, 0.99, 0.9925, 0.98, 0.5125, 0.2425]
# output/True/True/FGSM/15.pth
# [0.975, 0.9925, 0.98, 0.9825, 0.985, 0.9775, 0.625, 0.2425]
# output/True/True/FGSM/16.pth
# [0.985, 0.985, 0.9775, 0.985, 0.99, 0.975, 0.5375, 0.23]
# output/True/True/FGSM/17.pth
# [0.9775, 0.9925, 0.99, 0.985, 0.985, 0.9575, 0.4475, 0.2225]
# output/True/True/FGSM/18.pth
# [0.9825, 0.9925, 0.9875, 0.985, 0.9825, 0.9825, 0.74, 0.32]
# output/True/True/FGSM/19.pth
# [0.99, 0.995, 0.99, 0.9925, 0.985, 0.9675, 0.5075, 0.2525]
# output/True/True/FGSM/2.pth
# [0.98, 0.9675, 0.9675, 0.9775, 0.9775, 0.9675, 0.5975, 0.24]
# output/True/True/FGSM/20.pth
# [0.985, 0.9975, 0.98, 0.9875, 0.9925, 0.975, 0.5375, 0.2925]
# output/True/True/FGSM/3.pth
# [0.9775, 0.98, 0.9675, 0.9775, 0.97, 0.96, 0.66, 0.2875]
# output/True/True/FGSM/4.pth
# [0.975, 0.9875, 0.985, 0.985, 0.9775, 0.965, 0.585, 0.2175]
# output/True/True/FGSM/5.pth
# [0.975, 0.9925, 0.98, 0.985, 0.9825, 0.9575, 0.4575, 0.1525]
# output/True/True/FGSM/6.pth
# [0.9775, 0.9825, 0.9775, 0.9875, 0.98, 0.9675, 0.4775, 0.1575]
# output/True/True/FGSM/7.pth
# [0.98, 0.99, 0.98, 0.985, 0.9825, 0.9625, 0.45, 0.2375]
# output/True/True/FGSM/8.pth
# [0.98, 0.9875, 0.9725, 0.9825, 0.9825, 0.9525, 0.47, 0.245]
# output/True/True/FGSM/9.pth
# [0.9825, 0.99, 0.9725, 0.9875, 0.985, 0.97, 0.535, 0.275]


plt.figure(figsize=(6, 4.5))
string_title_acc = "igsm_cross_attack" + ".eps"
string_title_acc2 = "igsm_cross_attack" + ".png"
plt.plot(epsilons, fgsm_igsm)
plt.plot(epsilons, igsm_fgsm)
# plt.title('model accuracy')
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Epsilon", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM->IGSM", "IGSM->FGSM"], loc="lower left")
plt.grid(linewidth=0.7)
plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(0, 0.4)
plt.savefig(string_title_acc, format="eps")
plt.savefig(string_title_acc2, format="png")
plt.show()
plt.clf()




epochs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

fgsm_new_y = [35.4428775832057, 11.4066710174084, 8.55659718811512, 7.03692476078868, 5.92836035601795, 5.23106892034411, 4.68606476671994, 4.46186952106655, 4.09713917411864, 3.57374563254416, 3.46683517564088, 3.20501520112157, 2.98658007755876, 2.77816785406321, 2.6565207913518, 2.49598759459332, 2.34465308859944, 2.35000560712069, 2.22104105912149, 2.22356179729104]

fgsm_pre_y = [11.6571532774251, 9.0815023444593, 7.15373789519072, 6.22127731516957, 5.3753024674952, 4.8114389013499, 4.45906460657716, 4.25882803555578, 3.95982400514185, 3.59942827094346, 3.41326364502311, 3.18753960821778, 3.07877715304494, 3.01085077039897, 2.77816768968478, 2.6035786177963, 2.39224099460989, 2.43865782953799, 2.29154802020639, 2.27396757621318]


pgd_y = [5.29068638081662, 5.0256267413497, 4.55464823357761, 4.21436089463532, 3.58061544410884, 3.37977678515017, 2.8926353668794, 2.82201458513737, 2.72778925253078, 2.45039569027722, 2.23939340095967, 2.22661769995466, 2.09365646773949, 2.07927615614608, 2.10609243856743, 1.72303819749504, 1.66006362461485, 1.74952309951186, 1.58220762852579, 1.48631179472432 ] 

igsm_y = [7.64140543411486, 6.78314277902246, 5.42207394912839, 4.89320301264524, 4.08926278911531, 3.84817663114518, 3.59777331724763, 3.40438659954816, 2.99596772249788, 2.73706432152539, 2.74830358661711, 2.50071657821536, 2.37408613576554, 2.38404576666653, 2.07340772077441, 2.08203464793041, 1.85650866152719, 2.00875262194313, 1.82504641381092, 1.67087604128756]






plt.figure(figsize=(6, 4.5))
string_title_acc = "t_loss" + ".eps"
string_title_acc2 = "t_loss" + ".eps"
plt.plot(epochs, fgsm_new_y)
plt.plot(epochs, fgsm_pre_y)
plt.plot(epochs, pgd_y)
plt.plot(epochs, igsm_y)
# plt.title('model accuracy')
plt.ylabel("Training loss", fontsize=16)
plt.xlabel("Epoch", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM - new model", "FGSM - pre-trained", "PGD", "IGSM"], loc="right")
plt.grid(linewidth=0.7)
plt.yticks(np.arange(0,40, 10))
#plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(1, 20)
plt.gca().set_ylim(0,50)
plt.savefig(string_title_acc, format="eps")
plt.savefig(string_title_acc2, format="png")
plt.show()
plt.clf()
