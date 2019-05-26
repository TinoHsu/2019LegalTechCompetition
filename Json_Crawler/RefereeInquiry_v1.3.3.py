# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
from collections import defaultdict
import os
import sys


driver = webdriver.Chrome()
main_handle = driver.current_window_handle
driver.execute_script('window.open("https://www.google.com");')
handles = driver.window_handles
if handles[1] != main_handle:
    sub_handle = handles[1]
driver.switch_to.window(main_handle)

refereeDict = defaultdict(list)
preDict={}

page = 1
maxPage = 2
amountOfBatch = 39454
amount = 0
y = 0
m = 0
d = 0
y1=107
y2=85
m1=2
m2=12
d1=1
d2=31
reason = u'給付違約金'
# reason = u'返還借款'
# reason = u'侵權行為損害賠償'
# reason = u'給付贈與'


def main(): 
    driver.get("https://law.judicial.gov.tw/FJUD/default.aspx")
    time.sleep(2)
    driver.find_element_by_link_text(u"更多條件查詢").click()
    global y, m, d
    for y in range(y1, y2-1, -1):
        for m in range(m1, m2+1):
            for d in range(d1,d2+1):
                test_referee_inquiry(str(y), str(m), str(d))
                try:
                    driver.switch_to.frame(driver.find_elements_by_tag_name("iframe")[0])
                except:
                    driver.refresh()
                    continue
                print(str(y) + '/' + str(m) + '/' + str(d))
                print('page: '+(str(page)) + " start")
                RefereeLinks()
                print('page: '+(str(page)) + " completed")
                try:
                    driver.execute_script('arguments[0].scrollIntoView(false);',driver.find_element_by_id('hlNext'))
                    while driver.find_element_by_id('hlNext') and page < maxPage:
                        nextPage()
                except:
                    driver.back()

    
    
def test_referee_inquiry(y,m,d):
        
    checkBox_V = driver.find_element_by_css_selector('#vtype_V > input[type=checkbox]')
    checkBox_M = driver.find_element_by_css_selector("#vtype_M > input[type=checkbox]")
    time.sleep(1)
    if not checkBox_V.is_selected():
        checkBox_V.click()
    if not checkBox_M.is_selected():
        checkBox_M.click()
    driver.find_element_by_id("dy1").click()
    driver.find_element_by_id("dy1").clear()
    driver.find_element_by_id("dy1").send_keys(y)
    driver.find_element_by_id("dm1").click()
    driver.find_element_by_id("dm1").clear()
    driver.find_element_by_id("dm1").send_keys(m)
    driver.find_element_by_id("dd1").click()
    driver.find_element_by_id("dd1").clear()
    driver.find_element_by_id("dd1").send_keys(d)
    driver.find_element_by_id("dy2").click()
    driver.find_element_by_id("dy2").clear()
    driver.find_element_by_id("dy2").send_keys(y)
    driver.find_element_by_id("dm2").click()
    driver.find_element_by_id("dm2").clear()
    driver.find_element_by_id("dm2").send_keys(m)
    driver.find_element_by_id("dd2").click()
    driver.find_element_by_id("dd2").clear()
    driver.find_element_by_id("dd2").send_keys(d)
    driver.find_element_by_id("jud_title").click()
    driver.find_element_by_id("jud_title").clear()
    driver.find_element_by_id("jud_title").send_keys(reason)
    driver.find_element_by_id("btnQry").click()

def nextPage():
    global page

    driver.execute_script('arguments[0].scrollIntoView(false);',driver.find_element_by_id('hlNext'))
    driver.find_element_by_id('hlNext').click()
    page+=1
    print('page: '+(str(page)) + " start")
    RefereeLinks()
    print('page: '+(str(page)) + " completed")

def read_json():
    lines = ""
    with open("./output/" + reason + "judgements.json","r", encoding='utf8') as json_file:
        for row in json_file.readlines(): 
            lines+=row
        lines = json.loads(lines) 
        print("讀取json完成...")
        return lines
    
def to_json(refereeDict):
    global y, m, d, preDict
    filePath = "./output/" + reason +"judgements.json"
    if os.path.isfile(filePath):
        preDict = read_json()
        # print(preDict)
        for i in range(0, refereeDict['judgements'].__len__()):
            preDict['judgements'].append(refereeDict['judgements'][i])
        json_str = json.dumps(preDict, indent = 4, ensure_ascii=False)
    else:
        json_str = json.dumps(refereeDict, indent = 4, ensure_ascii=False)
    # print(json_str)
    path = "./output"
    if not os.path.isdir(path):
        os.mkdir(path)
    with open("./output/" + reason +"judgements.json","w", encoding='utf8') as json_file:      # 有輸入案由時使用
        json_file.write(json_str)
        print("共" + str(amount) + "筆，" +"儲存json完成...")

def add2Dict(referee):
    global refereeDict, amountOfBatch, amount 
    refereeDict['judgements'].append(referee)
    amount += 1
    print('amount = ' + str(amount))
    if amount % 10 == 0 and amount != 0 and amount != amountOfBatch:
        to_json(refereeDict)
        refereeDict.clear()
    if amount == amountOfBatch:
        to_json(refereeDict)
        os._exit(0)
    

def readReferee():
    try:
        soup = BeautifulSoup(driver.page_source, 'lxml')
        no = soup.select('#jud div:nth-of-type(1) div.col-td')[0].text # 不能用nth-child
        date = soup.select('#jud > div:nth-of-type(2) > div.col-td')[0].text
        reason = soup.select('#jud > div:nth-of-type(3) > div.col-td')[0].text
        judgement = soup.select('#jud > div:nth-of-type(4) > div > table > tbody > tr > td:nth-of-type(1) > div')[0].text
        judgement = filter(str.isalnum, judgement)
        judgement = ''.join(list(judgement))
    except: 
        print('page:' + str(page) + 'can not read a referee')
        driver.get("https://www.google.com")
        return
    referee = defaultdict(list)
    referee['no'].append(no)
    referee['date'].append(date)
    referee['reason'].append(reason)
    referee['judgement'].append(judgement)
    add2Dict(referee)


def RefereeLinks():
    soup = BeautifulSoup(driver.page_source, 'lxml')
    referees = soup.select('#hlTitle')
    for i in range(0, referees.__len__()):
        time.sleep(3)
        # print(driver.page_source)
        ele = driver.find_element_by_css_selector('#jud > tbody > tr:nth-of-type('+ str(2+2*i) +') > td:nth-of-type(2) > a')
        location = ele.location
        y = location['y']
        driver.execute_script('arguments[0].scrollIntoView(false);',ele)
        ele.click()
        readReferee()
        time.sleep(2)
        driver.back()
        try:
            driver.switch_to.frame(driver.find_elements_by_tag_name("iframe")[0])
        except:
            driver.refresh()
            return

main()