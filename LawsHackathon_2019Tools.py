import os
import numpy as np
import pandas as pd
import jieba
from os import listdir
import matplotlib.pyplot as plt
from os.path import isfile, isdir, join
import math
from pandas.io.json import json_normalize
import json

# for read json (before 7/1 use)
def readJson_chouCrawler(path,reason):
    lines = ""
    # 打開指定的檔案
    with open(path + "/" + reason + "judgements.json","r", encoding='utf8') as json_file:
        # 一行一行讀取json的字串 並串成完整的jsonStr
        for row in json_file.readlines(): 
            lines+=row
        # jsonStr轉成dict
        lines = json.loads(lines)
        # dict轉成dataframe 
        data = json_normalize(lines, 'judgements')
        # 把dataframe中的judgement與no兩個column[1, 2]取出來
        numpy_matrix = data.iloc[:, [1,2]].values
        data_list = []
        # print(numpy_matrix.shape)
        # print(numpy_matrix[0][1][0])
        for row in numpy_matrix:
            # print(row[1][0][-1])
            # 檢查no中的最後一個字[-1]是否為判"決"
            if row[1][0][-1] == '決':
                # 把 judgement中判決主文取出加到lsit
                # (在元素裡面的還是是lsit
                # 所以用row[0][0]拿出str)
                temp = row[0][0]
                data_list.append(temp)
        print('判決書數量', len(data_list))

        return data_list

# for read json (after 7/1 use ,Fit lowsnote format)
# def readJson(path,reason):

# def nameRecognition():

# def segmentation():

# def wordsTokenization():

# def topicModel(): 

# def resultShow():



