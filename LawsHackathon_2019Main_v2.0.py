from LawsHackathon_2019Tools import *

## 基礎設定區-----------------------
## 設定案由
case = "侵權行為損害賠償"
# case = "給付租金"

## 核心詞與法條加權的係數
coreWeighting = 1

## 主題歸類完後顯示前幾?筆詞彙
show_top_words = 100

## 開啟TF-IDF與否
# tfidf = True
tfidf = False

## 聚類模型表單
model_type_list = ['LDA', 'NMF']

## 聚類主題Topic k
topic_k_list = [[3, 3], [4, 5], [3, 4], [5, 6], [4, 4], [2, 5]]

## 基礎設定區-----------------------


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

for topic_k_choice in topic_k_list:
    for model_type in model_type_list:
        print('\n')
        print('執行1st聚類 模型', model_type, ' 一階k=', topic_k_choice[0], ' 二階k=', topic_k_choice[1], )
        ## 依照文件矩陣中詞頻進行主題分類
        model_result = topicModelCluster(docVector, words_list, topic_k=topic_k_choice[0], 
                                        n_top_words=show_top_words, model_type=model_type, 
                                        hierarchy = -1)

        ## 執行2nd聚類
        for i in range(np.unique(model_result).shape[0]):
            ## 取出目標Topic=?的文章在list中的index
            result_index = np.squeeze(np.argwhere(model_result==i).T)
            print('\n')
            print('Topic', i, '執行2nd聚類')
            print('目標文本的索引', result_index)
            result_index_list = result_index.tolist()

            ## 依照index重新合成2nd文本
            seg_corpus_for_2nd = []
            for index in result_index_list:
                # print(index)
                seg_corpus_for_2nd.append(seg_corpus[index])

            ## 2nd聚類的過程
            print('2nd聚類結果')
            docVector_2nd, words_list_2nd = bow1.wordsTokenization(seg_corpus_for_2nd, core_factor=coreWeighting, tfidf_func = tfidf)
            model_result_2nd = topicModelCluster(docVector_2nd, words_list_2nd, topic_k=topic_k_choice[1], 
                                                n_top_words=show_top_words, model_type=model_type, 
                                                hierarchy=i)


# 隨機印出?篇?Topic檢查內容
os.system('pause')
while(1):
    checkJudgement(model_result, bow1.jieba_result, judgement_length)