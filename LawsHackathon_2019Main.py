from LawsHackathon_2019Tools import *

## 基礎設定區-----------------------
## 設定案由
case = "侵權行為損害賠償"

## 核心詞與法條加權的係數
coreWeighting = 50

## 主題歸類完後顯示前幾?筆詞彙
show_top_words = 20


## 開啟TF-IDF與否
# tfidf = True
tfidf = False
## 基礎設定區-----------------------


## 獲取工作路徑
work_path = os.getcwd()

# 檔案存放路徑
data_path = work_path + "/judgement_json/"
# data_path = work_path + "/judgement_json_check_performance/"


## 開始讀檔並整理為字串存成文本raw_text
raw_text = readJson_chouCrawler(data_path, reason=case)

# 隨機印出一篇檢查內容
# rd = random.randint(0,len(raw_text))
# print('隨機印出一篇檢查')
# print(raw_text[rd])


## 除去人名
# corpus = nameRecognition()


## 分割文本中的字串
seg_corpus = segmentation(work_path, raw_text)


## 文本轉化成文件矩陣
docVector, words_list = wordsTokenization(work_path, seg_corpus, core_factor=coreWeighting, tfidf_func = tfidf)


## 依照文件矩陣中詞頻進行主題分類
model_result = topicModelLDA(docVector, words_list, topic_k=5, n_top_words=show_top_words)


# 隨機印出?篇?Topic檢查內容
os.system('pause')
while(1):
    checkJudgement(model_result, raw_text)