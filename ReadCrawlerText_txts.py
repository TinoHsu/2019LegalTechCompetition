import numpy as np
import pandas as pd
import jieba
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, isdir, join
import os
from gensim.models import word2vec
from mittens import GloVe
import matplotlib.pyplot as plt


## 撈出資料夾中檔案名稱-----------------------------------
# 給予路徑
now_path = os.getcwd()
mypath = now_path + "/judgement_txt/"
# 拿出路徑
files = listdir(mypath)
# 存檔案名稱
dir_list = []
# 搜索全部路徑內容
for f in files:
  # 產生檔案的絕對路徑
  fullpath = join(mypath, f)
  # 判斷 fullpath 是檔案還是目錄
  if isfile(fullpath):
    # print("檔案：", f)
    # 把檔案名稱加進list
    dir_list.append(f)

## 開始讀檔整理內文為可分割資料-----------------------------------
# 裝主文部份
corpus = []
# 裝分割}後成為str部份
y_tataol = []
for excute_dir in dir_list:
    print(excute_dir)
    df = pd.read_csv(open(mypath + excute_dir, encoding='utf8'), 'r',header=None)
    X = df.iloc[:, :].values
    X = np.ravel(X)
    y = np.sum(X)
    y_list = y.split("}")
    # print(type(y_list))
    # print(len(y_list))
    # print('500=',y_list[500])
    # print('501=',y_list[501])
    del y_list[500]
    del y_list[500]
    # print(len(y_list))

    onetxt_temp = []
    for y_str in y_list:
        y_tataol.append(y_str)
        start = y_str.find("judgement:")
        end = y_str.find("中　　華　　民　　國")

        # print('raw',end)
        if end == -1:
            end = y_str.find("中    華    民    國")
            # print('update',end) 
        keystr = y_str[start+10:end]
        
        corpus.append(keystr)

print(len(corpus))
## 讀檔結束檢查長度


## 使用jieba來處理斷詞-----------------------------------
# 存停用詞, 分詞, 過濾後分詞的list
stopWords=[]
segments=[]
remainderWords=[]

# 讀入停用詞檔
with open(now_path + '/jieba_dict/stopWords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)
print(stopWords)

# jieba的斷詞字典
jieba.set_dictionary(now_path + '/jieba_dict/dict.txt.big')
# 自訂的斷詞字典
jieba.load_userdict(now_path + '/jieba_dict/my.dict.txt')

# 準備裝斷詞的結果
seg_corpus = []
for i in range(len(corpus)):
    words = jieba.cut(corpus[i], cut_all=False)
    # 精確模式輸出
    # print("Default Mode: " + "/ ".join(words))  
    # 移除停用詞及跳行符號
    remainderWords = list(filter(lambda a: a not in stopWords and a != '\n', words))

    seg_lsit = []
    for word in remainderWords:
        # print(word)
        seg_lsit.append(word)
    # print("seg_list", seg_lsit)
    seg = ' '
    a = seg.join(seg_lsit)
    # print('a', a)
    seg_corpus.append(a)

print(len(seg_corpus))
## 斷詞結束檢查長度

## 模型部份-----------------------------------
# Word2Vec 嘗試(待研究)
# sentences = word2vec.LineSentence(seg_corpus)
# model = word2vec.Word2Vec(sentences, size=5, min_count=1, negative=10)
# # model.save('word2vec.model    model.wv.save_word2vec_format('word2vec.model.2inary = False)

# # sentences = word2vec.LineSentence("wiki_seg.txt")
# # model = word2vec.Word2Vec(sentences, size=250)
# model.save("word2vec.model")


## 前處理部份-----------------------------------
# 一般word embbading
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000)
# X = vectorizer.fit_transform(seg_corpus)
# # print(vectorizer.get_feature_names())
# print(X.toarray())

# TF-IDF處理
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X = tfidf.fit_transform(vectorizer.fit_transform(seg_corpus))
print(vectorizer.get_feature_names())
print(X.toarray()) 
print(X.shape) 

## 主題模型部份-----------------------------------
# LDA主題模型歸類
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=7, random_state=0)
lad_result = lda.fit_transform(X) 

print(lda.components_.shape)
lad_result = np.argmax(lad_result, axis=1)
print(lad_result)

# 各Topic 前20關鍵字
n_top_words = 20
feature_names = vectorizer.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))

## 主題模型歸類結果對應檢查-----------------------------------
for i in range(0, 1500, 100):
    y_str = y_tataol[i]
    start = y_str.find("why:")
    end = y_str.find("judgement:")    
    key_reson = y_str[start+20:end-4]
    print('模型推論Topic:', lad_result[i]+1, '   案由:', key_reson, )
    # print(corpus)

# count = CountVectorizer()
# bag = count.fit_transform(seg_corpus)

glove_model = GloVe(n=2, max_iter=1000)
cooccurrence_matrix = np.dot(X.toarray().transpose(), X.toarray())
embeddings = glove_model.fit(cooccurrence_matrix)
print(embeddings.shape)

plt.scatter(embeddings[:,0], embeddings[:,1], marker="o")
for i in range(0,len(feature_names)):
    plt.text(embeddings[i,0], embeddings[i,1], feature_names[i])

plt.show()
