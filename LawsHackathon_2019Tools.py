import os
import json
import math
import jieba
import random
import numpy as np
import pandas as pd
from os import listdir
from scipy import sparse
import matplotlib.pyplot as plt
from os.path import isfile, isdir, join
from pandas.io.json import json_normalize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



def readJson_chouCrawler(path,reason):
    # for read json (before 7/1 use)
    print('案由:', reason)
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


# def readJson(path,reason):
    # for read json (after 7/1 use ,Fit lowsnote format)

# def nameRecognition():


def segmentation(path, corpus):
    ## 使用jieba來處理斷詞-----------------------------------

    # 存停用詞
    stopWords=[]
    # 讀入停用詞檔
    with open(path + '/jieba_dict/stopWords.txt', 'r', encoding='utf-8-sig') as file:
        for data in file.readlines():
            data = data.strip()
            stopWords.append(data)
    # print('停用詞', stopWords)
    

    # jieba的斷詞字典
    jieba.set_dictionary(path + '/jieba_dict/dict.txt.big')
    # 自訂的斷詞字典
    jieba.load_userdict(path + '/jieba_dict/my.dict.txt')

    # 準備裝斷詞的結果
    seg_corpus = []
    # 對每個判決書斷詞並移除停用詞
    for i in range(len(corpus)):
    # for i in range(1):
        words = jieba.cut(corpus[i], cut_all=False)

        # 精確模式輸出
        # print("Default Mode: " + "/ ".join(words)) 
        
        # remainderWords=[]
        # 移除停用詞及跳行符號
        remainderWords = list(filter(lambda a: a not in stopWords and a != '\n', words))
        
        # 合併判決書的所有分詞為一個str:a
        seg_lsit = []
        for word in remainderWords:
            # print(word)
            seg_lsit.append(word)
        # print("seg_list", seg_lsit)
        seg_symbol = ' '
        a = seg_symbol.join(seg_lsit)
        # 看斷詞後且剔除停用詞後的結果
        # print('a', a)
        seg_corpus.append(a)

    print('斷詞結束, 檢查文本篇數', len(seg_corpus))
    ## 斷詞結束檢查長度

    return seg_corpus


def wordsTokenization(path, seg_corpus, core_factor, tfidf_func):
    ## 使用CountVectorizer()來將詞編碼（給予id）--------------------------
    # vectorizer = CountVectorizer(max_features=5000)
    vectorizer = CountVectorizer()
    
    # 開啟TF-IDF與否
    if tfidf_func == True:
        print('TF-IDF on !!!')
        tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
        bow = tfidf.fit_transform(vectorizer.fit_transform(seg_corpus))
        print('bow shape', bow.shape)
        np.set_printoptions(precision=2)
        # print(bow.toarray()) 
        # print(bow.toarray().shape)
        word_list = vectorizer.get_feature_names()
        print('word_list shape', len(word_list))

    else:
        print('TF-IDF off !!!')
        bow = vectorizer.fit_transform(seg_corpus)
        print('bow shape', bow.shape)
        # print(bow.toarray()) 
        # print(bow.toarray().shape)
        word_list = vectorizer.get_feature_names()
        print('word_list shape', len(word_list))
    
    # print(vectorizer.vocabulary_)

    ## 核心詞的加權處理--------------------------
    # 存核心詞
    coreWords=[]
    # 讀入核心詞檔
    with open(path + '/jieba_dict/coreWords.txt', 'r', encoding='utf-8-sig') as file:
        for data in file.readlines():
            data = data.strip()
            coreWords.append(data)
    # print('核心詞', coreWords)

    # bow 稀疏矩陣轉成一般矩陣
    bow_array = bow.A

    # 把核心詞找出來對應id的column乘上倍率
    for word in word_list:
        if word in coreWords:
            word_id = vectorizer.vocabulary_[word]
            # print('word_id', word_id)
            # print(bow_array)
            # print(bow_array[:, word_id])
            # print(bow_array[:, word_id]*core_factor)
            bow_array[:, word_id] = bow_array[:, word_id]*core_factor
            # print(bow_array)
    
    # bow_array 一般矩陣轉回稀疏矩陣
    bow = sparse.csr_matrix(bow_array)

    # 觀察高出現率的詞是什麼？
    # for i in range(bow.shape[0]):
    #     for j in range(bow.shape[1]):
    #         if bow.toarray()[i][j] >= 10:
    #             print(bow.toarray()[i][j], word_table[j])

    # print(bow.toarray()) 
    # print(bow) 

    return bow, word_list


def topicModelLDA(bow, word_table, topic_k, n_top_words):
    # LDA主題模型歸類
    lda = LatentDirichletAllocation(n_components=topic_k, random_state=0)
    lad_result = lda.fit_transform(bow) 
    # 歸類結果對應轉換
    # print(lda.components_.shape)
    lad_result = np.argmax(lad_result, axis=1)
    print('歸類結果') 
    print(lad_result) 

    feature_names = word_table
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d:" % (topic_idx + 1))
        print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))

    return lad_result

def checkJudgement(model_result, corpus):

    topic = input("隨機印出Topic?")
    topic = int(topic)
    assert type(topic)==int

    number_paper = input("?篇判決書")
    number_paper = int(number_paper)
    assert type(number_paper)==int

    find = topic
    paper_list = [i for i,v in enumerate(model_result) if v==find]
    print(paper_list)

    if (len(paper_list) == 0):
        print('沒有結果')

    elif (len(paper_list) == 1):
        paper_index = random.sample(paper_list, 1)
        print(paper_index)
        print('Topic', topic)
        print(corpus[paper_index[0]])

    else:
        paper_index = random.sample(paper_list, number_paper)
        print(paper_index)
        for t in range(len(paper_index)):
            print('Topic', topic)
            print(corpus[paper_index[t]])

    os.system('pause')
        

    



