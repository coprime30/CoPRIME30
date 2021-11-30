#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re  # 정규표현식

# load data
import glob
import os

# 전처리 시간
import math
import time
import datetime


# In[2]:


# 맞춤법 전처리 : Hanspell
from hanspell import spell_checker

# sentence = spell_checker.check("문장") -> completed = sentence.checked

# 형태소 분석기 : Mecab
from konlpy.tag import Mecab

Mecab = Mecab()


# In[4]:


# CONSTANTS

violence_types = ["기타", "직장", "갈취", "협박"]
violence_nums = [1, 2, 3, 4]


# ##### 파일들 불러오기

# In[5]:


def load_filename(DIR_PATH, violence_type):
    with open(DIR_PATH + "/" + violence_type + "_파일목록.txt") as f:
        lines = f.read().splitlines()
        base_dirs = [line[:-1] for line in lines if line[:2] == "./" and line[-1:] == ":"]
        file_names = [line for line in lines if line[:2] == violence_type and line[-4:] == ".txt"]
    return base_dirs, file_names


DIR_PATH = "/tf/text-data/파일목록"
base_dirs, file_names = dict(), dict()

for violence_type in violence_types:
    base_dirs[violence_type], file_names[violence_type] = load_filename(DIR_PATH, violence_type)
    print(f"{violence_type} : {len(file_names[violence_type])}")


# In[6]:


def load_txt(DIR_PATH, file_name):
    try:
        try:
            with open(DIR_PATH.strip() + "/" + file_name.strip(), encoding="euc-kr") as f:
                lines = f.read().splitlines()
                return lines
        except:
            try:
                with open(DIR_PATH.strip() + "/" + file_name.strip(), encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    return lines
            except:
                #                 print(f"\t[ERROR]\t {file_name}")  # 인코딩 문제
                return 0
    except:
        return -1


def search_every_files():
    dialog_dict, error_list = dict(), list()
    for violence_type in violence_types:  # 폭력상황 분류 loop
        dialog_dict[violence_type] = []  # 결과 담을 빈 리스트 생성
        for base_dir in base_dirs[violence_type]:  # 폭력상황별 base_dir / e.g. ./대화작성-OOO/1차/갈취
            for file_name in file_names[violence_type]:  # base_dir 하위의 파일명
                result = load_txt(base_dir, file_name)
                if not load_txt(base_dir, file_name) == 0:  # 제대로 불러와지는 파일을 read & splitlines
                    dialog_dict[violence_type].append(result)
                elif load_txt(base_dir, file_name) == 0:
                    error_list.append(f"{base_dir}/{file_name}")
            break
    return dialog_dict, error_list


dialog_dict, error_list = search_every_files()
print("=" * 20, "dialog_dict", "=" * 20)
for violence_type in violence_types:  # 폭력상황 분류 loop
    print(violence_type, " ", len(dialog_dict[violence_type]))
print(dialog_dict["갈취"][0])
print("=" * 20, "error_list", "=" * 20)
print("error cnt : ", len(error_list))


# ##### 정규화 : 띄어쓰기, 맞춤법

# In[7]:


def hanspell(sentence):
    checking = spell_checker.check(sentence)
    completed = checking.checked
    return completed


# In[8]:


def txt_write(dialog_list):
    # 일단 원본 파일에 덮어썼음
    with open(f"{violence_num}_{violence_type}_sentences_mecab_hanspell.txt", "w") as f:
        f.write("".join(dialog_list) + "\n")


# In[9]:


def preprocessing_mecab(preprocessed_list, dialog_list):
    dialog_list_2_str = "[CLS] "  # 한 대화의 시작을 CLS로 연다.
    for sentence in dialog_list:

        # 발화자 부분 제거
        sentence = sentence.split(":")[-1].strip()

        # Hanspell로 맞춤법 확인
        checking2 = hanspell(sentence)

        # mecab으로 형태소 분석 후 반영
        try:
            checked = Mecab.morphs(checking2)
        except:  # 빈 문장은 못잡아내서 try-except로 예외처리
            print(f"\t[EXCEPT]\t {checking2}")
            continue

        dialog_list_2_str += " ".join(checked)
        dialog_list_2_str += " [SEP] "  # 한 발화(턴)의 마지막에 SEP을 단다.

    dialog_list_2_str += f"\t{violence_num}\n\n"
    preprocessed_list.append(dialog_list_2_str)
    return preprocessed_list


# In[11]:


def make_dialog_list(violence_type, violence_num):
    preprocessed_list = []
    LENG = len(dialog_dict[violence_type])

    # log
    print("=" * 20, f"{violence_type} ({len(dialog_dict[violence_type])})", "=" * 20)

    for i in range(LENG):
        dialog_list = dialog_dict[violence_type][i]

        ## --------------- MECAB --------------- ##
        preprocessing_mecab(preprocessed_list, dialog_list)
        txt_write(preprocessed_list)

        # log
        print(f"[{violence_type} | PROCEED]\t{i}/{LENG} 번째 대화 파일 전처리 중, txt에 추가 완료")
        if i % 10 == 0:
            print(f"\t sentence : {preprocessed_list[-1]}")

    return preprocessed_list


def check_time(start, end):
    time = str(datetime.timedelta(seconds=(end - start)))
    return time.split(".")[0]


# In[12]:

# 전처리 파트 함수 분리 전 결과
for violence_type, violence_num in zip(violence_types, violence_nums):

    start = time.time()  # 시간 측정 시작
    print("=" * 14, "\t", violence_type, violence_num, "\t", "=" * 14)
    dialog_list = make_dialog_list(violence_type, violence_num)
    end = time.time()  # 시간 측정 끝

    print(f"[{check_time(start,end)}]\t {violence_num}_{violence_type}_sentences_mecab.txt")
