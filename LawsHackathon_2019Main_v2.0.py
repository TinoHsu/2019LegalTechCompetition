from LawsHackathon_2019Tools import *

## 基礎設定區-----------------------
## 設定案由
case = "侵權行為損害賠償"
# case = "確認所有權存在"


## 核心詞與法條加權的係數
coreWeighting = 1

## 主題歸類完後顯示前幾?筆詞彙
show_top_words = 50


## 開啟TF-IDF與否
# tfidf = True
tfidf = False
## 基礎設定區
# -----------------------


## 獲取工作路徑
work_path = os.getcwd()

## 存放路徑
#「特定案由檔名清單」
reson_list_path = work_path + "/file_name_list_wReaon"
#「json檔案」
json_path = "D:/LAW_HACH/CivilCourtsJudgment"

## 開始讀檔並整理為字串存成文本raw_text
raw_text = readJson(reson_list_path, json_path, reason=case)
# raw_text = readJsonChouChouCrawler(data_path, reason=case)


# 隨機印出一篇檢查內容
# rd = random.randint(0,len(raw_text))
# print('隨機印出一篇檢查')
# print(raw_text[rd])


## 除去人名
# corpus = nameRecognition()

## 文本向量化
bow1 = TextRepresatation(work_path, raw_text)
## 分割文本中的字串
seg_corpus, judgement_length = bow1.segmentation('my.dict.txt', 'stopWords.txt')

# 判決書字數
# print(judgement_length)

## 文本轉化成文件矩陣
docVector, words_list = bow1.wordsTokenization(seg_corpus, core_factor=coreWeighting, tfidf_func = tfidf)

# seg_corpus = segmentation(work_path, raw_text)
# docVector, words_list = wordsTokenization(work_path, seg_corpus, core_factor=coreWeighting, tfidf_func = tfidf)
print('\n')
print('執行1st聚類')
## 依照文件矩陣中詞頻進行主題分類
model_result = topicModelLDA(docVector, words_list, topic_k=5, n_top_words=show_top_words)

## 執行2rd聚類
for i in range(np.unique(model_result).shape[0]):
    ## 取出目標Topic=?的文章在list中的index
    result_index = np.squeeze(np.argwhere(model_result==i).T)
    print('\n')
    print('Topic', i, '執行2rd聚類')
    print('目標文本的索引', result_index)
    result_index_list = result_index.tolist()

    ## 依照index重新合成2rd文本
    seg_corpus_for_2rd = []
    for index in result_index_list:
        # print(index)
        seg_corpus_for_2rd.append(seg_corpus[index])

    ## 2rd聚類的過程
    print('2rd聚類結果')
    docVector_2rd, words_list_2rd = bow1.wordsTokenization(seg_corpus_for_2rd, core_factor=coreWeighting, tfidf_func = tfidf)
    model_result_2rd = topicModelLDA(docVector_2rd, words_list_2rd, topic_k=3, n_top_words=show_top_words)


# 隨機印出?篇?Topic檢查內容
os.system('pause')
while(1):
    checkJudgement(model_result, bow1.jieba_result, judgement_length)