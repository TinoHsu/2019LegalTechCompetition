import json
from os import listdir
from os.path import isfile, isdir, join
from pandas.io.json import json_normalize
import os

mypath = "D:/LAW_HACH/testzone"
# jsonpath = "D:/LAW_HACH/testzone"
jsonpath = "D:/LAW_HACH/CivilCourtsJudgment"

need_reson_1 = '侵權行為損害賠償'
need_reson_2 = '損害賠償'

print('開始歷遍資料夾！')
files = listdir(jsonpath)
dir_list = []
for f in files:
  # 產生檔案的絕對路徑
  fullpath = join(jsonpath, f)
  # 判斷 fullpath 是檔案還是目錄
  if isfile(fullpath):
    #print("檔案：", f)
    dir_list.append(f)
# print(dir_list)
print('歷遍所有檔名！')

need_json_list = []
for excute_dir in dir_list:
    #print(excute_dir)
    lines = ""
    with open(jsonpath + "/" + excute_dir ,"r", encoding='UTF-8-sig') as json_file:
        for row in json_file.readlines():
        #print(row)
            lines+=row
        lines = json.loads(lines)
        #print(lines['reason'])
        
        if lines['reason'] == need_reson_1:
            need_json_list.append(excute_dir)
        elif lines['reason'] == need_reson_2:
            need_json_list.append(excute_dir)

            
# print(need_json_list)
print('共幾筆：', len(need_json_list))

with open(mypath + "/" + need_reson_1 + "_list.txt",'a') as text2:
    for i in need_json_list:
        #print(i)
        w = (i,'\n')#輸出換行
        text2.writelines(w)

print('輸出txt！')