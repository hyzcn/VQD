import argparse
import os
import utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', help='save folder name',default='0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()      
    savefolder = args.save    
    info_json = os.path.join(savefolder,'infos.json')
    js = utils.parsejson(info_json)   
    colors = iter('rbgkcm')
    print (js.keys())
    sns.set()
    sns.set_context("paper")
    plt.figure(figsize=(8,4))
    for key in js:
        L = len(js[key])
        c = next(colors)
        if 'loss' in key:
            plt.subplot(1,2,1)
            plt.ylabel('Loss')
        else:
            plt.subplot(1,2,2)
            plt.ylabel('Accuracy')
        plt.plot(range(L),js[key],'o',color=c)
        plt.plot(range(L),js[key],label=key,color=c)
        plt.xlabel("Epochs")
        plt.legend()
    plt.show()
