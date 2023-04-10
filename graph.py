#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:40:13 2023

@author: jc4303
"""

import matplotlib.pyplot as plt  # mod - for line plot of 2b
import numpy as np

epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

fgsm_y = [0.99, 0.70, 0.26, 0.12, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1]

pgd_y = [0.95, 0.76, 0.42, 0.16, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

igsm_y = [0.95, 0.76, 0.42, 0.16, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


plt.figure(figsize=(6, 4.5))
string_title_acc = "ae_acc" + ".eps"
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
plt.show()
plt.clf()


#### ae training models agains benigin

std_model = 98.80

fgsm_new_y = 80.5

fgsm_pre_y = 80.5

pgd_y = 80.2

igsm_y = 80.2

benign_val = [std_model, fgsm_new_y, fgsm_pre_y, pgd_y, igsm_y]

models = ("Clean", "FGSM/n", "FGSM/p", "PGD", "IGSM")
x_pos = np.arange(len(models))
plt.figure(figsize=(6, 4.5))
string_title_acc = "at_benign" + ".eps"
# plt.title('model accuracy')
plt.bar(x_pos, benign_val, color=["black", "red", "green", "blue", "yellow"])
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Model", fontsize=16)
plt.xticks(x_pos, models)
plt.gca().set_ylim(bottom=0, top=100)
plt.savefig(string_title_acc, format="eps")
plt.show()
plt.clf()

#### at_adv graph
epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # placeholder


fgsm_new_y = [0.95, 0.76, 0.42, 0.16, 0.1, 0.1, 0.1, 0.1]  # placeholder

fgsm_pre_y = [0.99, 0.73, 0.22, 0.12, 0.1, 0.1, 0.1, 0.1]  # placeholder

pgd_y = [0.95, 0.76, 0.42, 0.16, 0.1, 0.1, 0.1, 0.1]  # placeholder
igsm_y = [0.95, 0.76, 0.42, 0.16, 0.1, 0.1, 0.1, 0.1]  # placeholder


plt.figure(figsize=(6, 4.5))
string_title_acc = "at_adv" + ".eps"
plt.plot(epsilons, fgsm_new_y)
plt.plot(epsilons, fgsm_pre_y)
plt.plot(epsilons, pgd_y)
plt.plot(epsilons, igsm_y)
# plt.title('model accuracy')
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Epsilon", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM - new model", "FGSM - pre-trained", "PGD", "IGSM"], loc="right")
plt.grid(linewidth=0.7)
plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(0, 0.4)
plt.savefig(string_title_acc, format="eps")
plt.show()
plt.clf()


#### at_cross_Attac
epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # placeholder


fgsm_pgd = [0.95, 0.76, 0.42, 0.16, 0.1, 0.1, 0.1, 0.1]  # placeholder

pgd_fgsm = [0.99, 0.73, 0.22, 0.12, 0.1, 0.1, 0.1, 0.1]  # placeholder

plt.figure(figsize=(6, 4.5))
string_title_acc = "pgd_cross_attack" + ".eps"
plt.plot(epsilons, fgsm_pgd)
plt.plot(epsilons, pgd_fgsm)
# plt.title('model accuracy')
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Epsilon", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM->PGD", "PGD->FGSM"], loc="right")
plt.grid(linewidth=0.7)
plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(0, 0.4)
plt.savefig(string_title_acc, format="eps")
plt.show()
plt.clf()

#### at_cross_Attac
epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # placeholder


fgsm_igsm = [0.95, 0.76, 0.42, 0.16, 0.1, 0.1, 0.1, 0.1]  # placeholder

igsm_fgsm = [0.99, 0.73, 0.22, 0.12, 0.1, 0.1, 0.1, 0.1]  # placeholder

plt.figure(figsize=(6, 4.5))
string_title_acc = "igsm_cross_attack" + ".eps"
plt.plot(epsilons, fgsm_igsm)
plt.plot(epsilons, igsm_fgsm)
# plt.title('model accuracy')
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Epsilon", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.tick_params(axis="both", labelsize=14)
plt.legend(["FGSM->IGSM", "IGSM->FGSM"], loc="right")
plt.grid(linewidth=0.7)
plt.gca().set_ylim(bottom=0, top=1)
plt.gca().set_xlim(0, 0.4)
plt.savefig(string_title_acc, format="eps")
plt.show()
plt.clf()
