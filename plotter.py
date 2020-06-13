#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.optimize as opt

def expfit(data):
    n = len(data)
    n_start = round(n*0.8)
    x1 = n_start
    x2 = n-1
    y1 = np.log10(data[x1])
    y2 = np.log10(data[x2])
    a = (y2 - y1)/(x2- x1)
    b = data[x1]
    f = lambda X: b * 10**(a*(X-n_start))
    print(a)
    return f

whole_data = {}
for prefix in ["nesterov", "grad"]:
    for n in [32, 64, 128]:
        filename = "json/" + prefix + "_n" + str(n) + ".json"
        with open(filename, 'rb') as f:
            data = json.load(f)
        key = (prefix, n)
        whole_data[key] = data

#for n in [32, 64, 128]:
for n in [32, 64, 128]:
    data_nest = whole_data[("nesterov", n)]
    data_grad = whole_data[("grad", n)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 4.5)

    ax1.plot(data_grad['err_seq'], c='r', linewidth=2.0)
    f = expfit(data_grad['err_seq'])
    n_end = len(data_grad['err_seq'])
    X = np.linspace(0, n_end)
    Y_approx = f(X)
    ax1.plot(X, Y_approx, c='gray', linewidth=3.0, linestyle=':')

    ax1.plot(data_nest['err_seq'], c='b', linewidth=2.0)
    f = expfit(data_nest['err_seq_processed'])
    n_end = len(data_nest['err_seq'])
    X = np.linspace(0, n_end)
    Y_approx = f(X)
    ax1.plot(X, Y_approx, c='chocolate', linewidth=3.0, linestyle=':')

    ax1.set_yscale('log', basey=10)
    ax1.legend(["GD", "GD-approx", "NAG", "NAG-approx"], loc=0)
    ax1.set_xlabel("iteration", fontsize=12)
    ax1.set_ylabel("error", fontsize=12)

    ax2.plot(data_grad['err_seq_processed'], data_grad['time_seq'], c='r', linewidth=1)
    ax2.plot(data_nest['err_seq_processed'], data_nest['time_seq'], c='b', linewidth=1)
    ax2.legend(["GD", "NAG"], loc=2)
    ax2.set_xlabel("accuracy", fontsize=12)
    ax2.set_ylabel("time [sec]", fontsize=12)

    err_seqs = data_grad['err_seq_processed'] + data_nest['err_seq_processed']
    ax2.set_xlim(max(err_seqs), min(err_seqs)) # inverted
    ax2.set_xscale('log', basex=10)
    #fig.subplots_adjust(bottom=0.2)
    plt.savefig("figs/n"+str(n)+".png", format="png", dpi=300)
    plt.show()


