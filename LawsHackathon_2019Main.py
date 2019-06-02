from LawsHackathon_2019Tools import *

# 獲取工作路徑
now_path = os.getcwd()
# 檔案存放路徑
mypath = now_path + "/judgement_json/"

# 開始讀檔並整理為字串存成文本raw_text
raw_text = readJson_chouCrawler(mypath, reason="清償借款")
print(type(raw_text[0]))
print(raw_text[0])

# 除去人名與法條
# corpus = nameRecognition()

# 分割文本中的字串
# seg_corpus = segmentation()

# 字串轉化成id
# X = wordsTokenization()

# 把依照文本中id進行主題分類
# result = topicModel()

# 顯示主題分類結果
# resultShow()