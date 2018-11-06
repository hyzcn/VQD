#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:56:13 2018

@author: manoj
"""

from flask import Flask 
from flask import render_template
import random
import numpy as np
import os
import glob
import json
from PIL import Image
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import parsejson

#%%

STATIC_FOLDER = 'example'
app = Flask(__name__,static_folder=STATIC_FOLDER)

html = """<!DOCTYPE html>
<html lang="en">
<head>
  <title>VQD</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
</head>
<body>

<div class="container"> 
<br><br>
{}
</div>

</body>
</html>"""


htmls = """<table class="table table-bordered">
<thead><tr><th>Dataset</th><th>Model</th><th>Eval Details</th><th>Performance</th></tr></thead>
<tbody>"""

row = """<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>"""

htmle = """
</tbody>
</table>"""
    
#%%

@app.route('/')
def hello_world():
    folders = glob.glob('refcoco*/')
    folders = sorted(folders)
    rows = ''
    for folder in folders:
        name = ' '.join(folder.split("_")[2:])[:-1]
        ds = folder.split("_")[0]
        model = folder.split("_")[1]
        try:
            js = parsejson( os.path.join(folder,'infos.json'))
            acc = max(js['test acc'])
            description = '{:.2f}%'.format(acc)
        except:
            description = 'Not found'
        rows += row.format(ds,model,name,description)
    table = htmls + rows + htmle
    htmlstr = html.format(table)
    return htmlstr

#        plt.figure()
#        plt.imshow(npimg)
#               plt.plot(rect[:,0],rect[:,1],'r',linewidth=4.0)
#        plt.axis('off')
#        fullpath = os.path.join("/home/manoj/count_baselines/VQA2/example/tmp",imgq['img'])
#        plt.savefig(fullpath)
#    return render_template('imshow.html' , images = imgs_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)