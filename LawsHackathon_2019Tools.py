import os
import re
import json
import math
import time
import jieba
import random
import numpy as np
import pandas as pd
from os import listdir
from scipy import sparse
import matplotlib.pyplot as plt
from os.path import isfile, isdir, join
from pandas.io.json import json_normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



def readJsonChouChouCrawler(path,reason):
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


def readJson(txtpath, josnpath, reason):
    # for read json (after 7/1 use ,Fit lowsnote format)
    print('案由:', reason)
    
    # 打開「特定案由檔名清單」
    json_name_list = open(txtpath + "/" + reason + ".txt", 'r')
    # print(type(json_name_list))

    # 存放判決書轉成str的 list
    data_list = []

    # 儲存\uXXXX讀取錯誤清單
    # escape_error_list = []
    
    # 儲存文末標記區域清單
    # tail_list = []

    # 從「特定案由檔名清單」裡依序撈出「json檔名」
    for json_name in json_name_list:
        # 去掉換行符號'\n'
        json_name = json_name.strip('\n')
        # print(json_name)

        # 讀取json內容

        # lines = ""
        # with open(josnpath + "/" + json_name, "r", encoding='utf8') as json_file:
        #     print(json_name)
        #     # 一行一行讀取json的字串 並串成完整的jsonStr
        #     for row in json_file.readlines():
        #         # print(row)
        #         #消除數字
        #         lines+=re.sub(r'\d+','',row)
        #         # lines+=row
        #     # 把json轉成dict
        #     try:
        #         lines = json.loads(lines)
        #     except:
        #         escape_error_list.append(json_name)

        #     judgement_str = lines['judgement']

        json_file = open(josnpath + "/" + json_name, "r", encoding='UTF-8-sig')
        json_dict = json.load(json_file)
        # try:
        #     json_dict = json.load(json_file)
        # except:
        #     escape_error_list.append(json_name)
        
        #把dict中的judgement部份取出
        judgement_str = json_dict['judgement']
        # print(judgement_str)
                
        ## 消除數字
        digi_out = re.sub(r'\d+','',judgement_str)
        ## 消除\n \r \u3000
        trash_out = "".join(digi_out.split())
        #print(trash_out)
        #print('\n')
        
        ##消除 上訴人 被上訴人 訴訟代理人
        # print(type(lines['party']))
        # print(len(lines['party']))
        # print(lines['party']) 
        special_symbol = '?'  
        for i in range(len(json_dict['party'])):
            #print(lines['party'][i]['value'])
            # 取出人名
            person_name = json_dict['party'][i]['value']
            # 消除特殊字元
            intersection = (person_name and special_symbol)
            if intersection == '?':
                person_name = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", person_name)
            # print(person_name)
            # 消除人名
            trash_out = re.sub(person_name, '', trash_out)
        # print(trash_out)
        # print('\n')
        
        ##消除 書記官 法官
        mark_point_str_list = ["訟費用負擔", "判決如主文", "本院判決如下", "判決如下"]
        for mark_point_str in mark_point_str_list:
            mark_point = trash_out.find(mark_point_str)
            # print(mark_point)
            if mark_point != -1:
                # print(mark_point_str)
                break
        
        '''
        # 判決書末尾抓不到上列關鍵片段，儲存檔名待查
        if mark_point == -1:
            tail_list.append(json_name)
        '''

        detect_area = trash_out[mark_point:]
        # print(detect_area)
        # print('\n')
        # print(trash_out)
        judge = "法官"
        clerk = "書記官"

        #書記官
        clerk_mark_point = detect_area.find(clerk)
        #姓名長度三 #若姓名長度二會圈到後面的字 就一起砍了
        clerk_name = detect_area[clerk_mark_point:clerk_mark_point+6]
        # print(clerk_name)
        detect_area = re.sub(clerk_name, '', detect_area)
        # print(detect_area)
        # print('\n')
        # print(trash_out)
        trash_out = re.sub(clerk_name, '', trash_out)
        
        # 法官
        while(1):
            judge_mark_point = detect_area.find(judge)
            #姓名長度三
            judge_name = detect_area[judge_mark_point:judge_mark_point+5]
            # print(judge_name)
            if judge_name == '':
                # print('byebye!')
                break
            #姓名長度二
            if judge_name[-1]=='法':
                judge_name = detect_area[judge_mark_point:judge_mark_point+4]
                # print(judge_name)
            detect_area = re.sub(judge_name, '', detect_area)
            trash_out = re.sub(judge_name, '', trash_out)
            # print(detect_area)
            # print('\n')
            # print(trash_out)
        # print(trash_out)

        # 塞進存判決書的list
        data_list.append(trash_out)
        # print('OK')
    '''
    # 錯誤紀錄轉存
    escape_error = np.asarray(escape_error_list).T
    df_escape_error = pd.DataFrame(escape_error)
    df_escape_error.to_excel("escape_error_list.xlsx", index=False)

    tail = np.asarray(tail_list).T
    df_tail = pd.DataFrame(tail)
    df_tail.to_excel("tail_list.xlsx", index=False)

    print('Save right!')
    '''
    return data_list


# def nameRecognition():

class TextRepresatation:

    def __init__(self, path, corpus):
        self.path = path
        self.corpus = corpus


    def segmentation(self, dict_name, stop_dict_name):
        ## 使用jieba來處理斷詞-----------------------------------

        self.jieba_result = []
        # jieba的斷詞字典
        jieba.set_dictionary(self.path + '/jieba_dict/dict.txt.big')
        # 自訂的斷詞字典
        jieba.load_userdict(self.path + '/jieba_dict/' + dict_name)

        # 存停用詞
        stopWords=[]
        # 讀入停用詞檔
        with open(self.path + '/jieba_dict/'+ stop_dict_name, 'r', encoding='utf-8-sig') as file:
            for data in file.readlines():
                data = data.strip()
                stopWords.append(data)
        # print('停用詞', stopWords)

        # 準備裝斷詞的結果
        judgement_length = []
        seg_corpus = []
        tic = time.time()
        # 對每個判決書斷詞並移除停用詞
        for i in range(len(self.corpus)):
        # for i in range(10):
            # 初始化jieba斷詞生成器
            words = jieba.cut(self.corpus[i], cut_all=False)
            words_copy = jieba.cut(self.corpus[i], cut_all=False)
        
            # 精確模式輸出
            # print("Default Mode: " + "/ ".join(words)) 

            # 留下jieba斷詞的副本以供查詢
            combine_words = ("/".join(words_copy))
            self.jieba_result.append(combine_words)
            
            # remainderWords=[]
            # 移除停用詞及跳行符號
            remainderWords = list(filter(lambda a: a not in stopWords and a != '\n', words))
            # print('判決書字數', len(remainderWords))   
            judgement_length.append(len(remainderWords))

            # 合併判決書的所有分詞為一個str
            seg_symbol = ' '
            seg_str = seg_symbol.join(remainderWords)
            # 看斷詞後且剔除停用詞後的結果
            # print('seg_str', seg_str)
            seg_corpus.append(seg_str)

        # print('jieba_result', self.jieba_result)

        toc = time.time()
        ## 斷詞結束檢查長度
        print('斷詞結束, 檢查文本篇數', len(seg_corpus))
        print('斷詞花費時間: ' + str((toc - tic)) + 'sec')
        
        return seg_corpus, judgement_length
    

    def wordsTokenization(self, seg_corpus, core_factor, tfidf_func):
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
        '''
        ## 核心詞的加權處理--------------------------
        # 存核心詞
        coreWords=[]
        # 讀入核心詞檔
        with open(self.path + '/jieba_dict/coreWords.txt', 'r', encoding='utf-8-sig') as file:
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
        '''
        # 觀察高出現率的詞是什麼？
        # for i in range(bow.shape[0]):
        #     for j in range(bow.shape[1]):
        #         if bow.toarray()[i][j] >= 10:
        #             print(bow.toarray()[i][j], word_table[j])

        # print(bow.toarray()) 
        # print(bow) 

        return bow, word_list


def topicModelCluster(bow, word_table, topic_k, n_top_words, model_type, hierarchy):
    if model_type == 'LDA':
        # LDA主題模型歸類
        lda = LatentDirichletAllocation(n_components=topic_k, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
        cluster_result = lda.fit_transform(bow) 
        # 歸類結果對應轉換
        # print(lda.components_.shape)
        cluster_result = np.argmax(cluster_result, axis=1)
        print('歸類結果') 
        print(cluster_result)

        feature_names = word_table
        for topic_idx, topic in enumerate(lda.components_):
            print('\n')
            if hierarchy == -1:
                print("Topic %d:" % (topic_idx),'篇數', cluster_result.tolist().count(topic_idx))
                print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))        
            if hierarchy != -1:
                print("Topic %d.%d:" % (hierarchy, topic_idx),'篇數', cluster_result.tolist().count(topic_idx))
                print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))
    
    if model_type == 'NMF':
        # NMF主題模型歸類
        nmf = NMF(n_components=topic_k, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5)
        cluster_result = nmf.fit_transform(bow) 
        # 歸類結果對應轉換
        cluster_result = np.argmax(cluster_result, axis=1)
        print('歸類結果') 
        print(cluster_result)

        feature_names = word_table
        for topic_idx, topic in enumerate(nmf.components_):
            print('\n')
            if hierarchy == -1:
                print("Topic %d:" % (topic_idx),'篇數', cluster_result.tolist().count(topic_idx))
                print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))        
            if hierarchy != -1:
                print("Topic %d.%d:" % (hierarchy, topic_idx),'篇數', cluster_result.tolist().count(topic_idx))
                print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))
    
    '''
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
        print('\n')
        print("Topic %d:" % (topic_idx),'篇數', lad_result.tolist().count(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))

    return lad_result
    '''
    return cluster_result

def checkJudgement(model_result, corpus, judgement_length):

    try:
        topic = input("隨機印出Topic? ")
        topic = int(topic)
        assert type(topic)==int
    except ValueError:
        print('請輸入小於主題數量的整數')

    try:
        number_paper = input("?篇判決書 ")
        number_paper = int(number_paper)
        assert type(number_paper)==int
    except ValueError:
        print('請輸入小於文章數量的整數')
    find = topic
    paper_list = [i for i,v in enumerate(model_result) if v==find]
    # print(paper_list)

    if (len(paper_list) == 0):
        print('\n')
        print('沒有結果')

    elif (len(paper_list) == 1):
        
        paper_index = random.sample(paper_list, 1)
        print(paper_index)
        print('\n')
        print('Topic', topic, '第', paper_index, '篇', '判決書字數', judgement_length[paper_index[0]])
        print(corpus[paper_index[0]])

    else:
        paper_index = random.sample(paper_list, number_paper)
        print(paper_index)
        for t in range(len(paper_index)):
            print('\n')
            print('Topic', topic, '第', paper_index[t], '篇', '判決書字數', judgement_length[paper_index[t]])
            print(corpus[paper_index[t]])

    os.system('pause')
        
def checkJudgement_all(model_result, corpus):
    print('NotYet!')
    



