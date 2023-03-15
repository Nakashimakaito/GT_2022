import os
import pandas as pd
import numpy as np
import torch
import copy
import itertools
import collections
import random
from collections import Counter

def get_speaker(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    speakers=[]
    speaker=[]
    juge=False
    for indx, tag in enumerate(seq["ner_tags"]):
        if not isinstance(tag, str):
            tag = id2label[int(tag)]
        if tag.startswith("B-S"):
            if chunk[2] != -1:
                chunks.append(chunk)
                speakers.append(speaker)
            chunk = [-1, -1, -1]
            speaker=[]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            speaker.append(seq["tokens"][indx])
            if indx == len(seq["ner_tags"]) - 1:
                chunks.append(chunk)
                speakers.append(speaker)
        elif tag.startswith('I-S') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            speaker.append(seq["tokens"][indx])
            if indx == len(seq["ner_tags"]) - 1:
                chunks.append(chunk)
                speakers.append(speaker)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
                speakers.append(speaker)
            chunk = [-1, -1, -1]
    if len(speakers)>0:
        juge=True
    return speakers,chunks,juge


def get_all_tokens_and_ner_tags(directory):
    return pd.concat([get_tokens_and_ner_tags(directory) ]).reset_index().drop('index', axis=1)
    
def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split(' ')[0] for x in y] for y in split_list]
        entities = [[x.split(' ')[1][:-1] for x in y] for y in split_list] 
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
  
def get_un_token_dataset(directory):
    df = get_all_tokens_and_ner_tags(directory)
    for i in range(len(df)):
        df.ner_tags[i]=[ "O" if d =="Out" else d for d in df.ner_tags[i]]
    return df

def Speaker_List(d,id2label):
    Counter_List=[]
    Speaker_J_list=[]
    name_list_1=[]
    name_list_over=[]
    for i in range(len(d)):
        sp,ch,j=get_speaker(d.loc[i],id2label)
        if len(sp)>0:
            for sp1 in [" ".join(s1) for s1 in sp]:
                Counter_List.append(sp1.lower())
    c = collections.Counter(Counter_List)
    for word in c.most_common():
        if word[1] == 1:
            name_list_1.append(word[0])
        else:
            name_list_over.append(word[0])
    Speaker_J_list=[n.split(' ') for n in name_list_1]+[n.split(' ') for n in name_list_over]
    return name_list_1 , name_list_over , Speaker_J_list

def get_Data_Agument(FILE_PATH):
    SP_JUGE=[]
    LOG={}
    LOG2=[]
    df2= get_un_token_dataset(FILE_PATH)
    df= get_un_token_dataset(FILE_PATH)
    id2label = {
    0:'O',
    1:'B-RightSpeaker',
    2:'B-Speaker',
    3:'B-LeftSpeaker',
    4:'B-Unknown',
    5:'I-RightSpeaker',
    6:'I-Speaker',
    7:'I-LeftSpeaker',
    8:'I-Unknown'
    }
    #creat name list
    BEFORE_NAME_LIST_APPEAR_1 , BEFORE_NAME_LIST__OVER_1 , BEFORE_TOTAL_SPEAKER_LIST = Speaker_List(df,id2label)
    for i in range(len(df2)):
        sp,ch,j=get_speaker(df2.loc[i],id2label)
        if len(sp) == 0:
            continue
        else:
            Be_len=0
            Af_len=0
            log_len=0
            for SP,CH in zip([" ".join(S) for S in sp],ch):
                SP=SP.lower()
                Inset_Speaker = random.choice(BEFORE_NAME_LIST_APPEAR_1).split(" ")
                assert  Inset_Speaker in BEFORE_TOTAL_SPEAKER_LIST,'未知の言語です[{0}]'.format(Inset_Speaker)

                Add_la=["B-Speaker"]
                if SP in SP_JUGE and SP in BEFORE_NAME_LIST__OVER_1:
                    if CH[1]==CH[2]:
                        if len(Inset_Speaker)>1:
                            for I in range(len(Inset_Speaker)-1):
                                Add_la.append("I-Speaker")
                        assert len(Add_la) == len(Inset_Speaker),'話者ラベルの長さ[{0}], 話者の長さ[{1}]'.format(len(Add_la),len(Inset_Speaker))

                        Be_len=len(df2.loc[i]["tokens"])
                        df2.loc[i]["tokens"][CH[1]+log_len:CH[1]+1+log_len]=[]
                        df2.loc[i]["ner_tags"][CH[1]+log_len:CH[1]+1+log_len]=[]
                        df2.loc[i]["tokens"][CH[1]+log_len:CH[1]+log_len] = Inset_Speaker
                        df2.loc[i]["ner_tags"][CH[1]+log_len:CH[1]+log_len]= Add_la
                        Af_len=len(df2.loc[i]["tokens"])
                        log_len += Af_len - Be_len
                        assert len(df2.loc[i]["ner_tags"]) == len(df2.loc[i]["tokens"]),'変換後話者ラベルの長さ[{0}], 変換後話者の長さ[{1}]'.format(len(Add_la),len(Inset_Speaker))

                        LOG[i]=CH[1]

                        LOG2.append(i)

                    else:
                        if len(Inset_Speaker)>1:
                            for I in range(len(Inset_Speaker)-1):
                                Add_la.append("I-Speaker")
                        assert len(Add_la) == len(Inset_Speaker),'話者ラベルの長さ[{0}], 話者の長さ[{1}]'.format(len(Add_la),len(Inset_Speaker))

                        #print(Inset_Speaker,Add_la)
                        Be_len=len(df2.loc[i]["tokens"])
                        df2.loc[i]["tokens"][CH[1]+log_len:CH[2]+1+log_len]= []
                        df2.loc[i]["ner_tags"][CH[1]+log_len:CH[2]+1+log_len]= []
                        df2.loc[i]["tokens"][CH[1]+log_len:CH[1]+log_len]=Inset_Speaker
                        df2.loc[i]["ner_tags"][CH[1]+log_len:CH[1]+log_len]=Add_la
                        Af_len=len(df2.loc[i]["tokens"])
                        log_len+=Af_len-Be_len
                        assert len(df2.loc[i]["ner_tags"]) == len(df2.loc[i]["tokens"]),'変換後話者ラベルの長さ[{0}], 変換後話者の長さ[{1}]'.format(len(Add_la),len(Inset_Speaker))

                        LOG[i]=CH[1]
                        LOG2.append(i)

                elif SP not in SP_JUGE and SP in BEFORE_NAME_LIST__OVER_1:
                    SP_JUGE.append(SP)
                else:
                    continue
    AFTER_NAME_LIST_APPEAR_1 , AFTER_NAME_LIST__OVER_1 , AFTER_TOTAL_SPEAKER_LIST = Speaker_List(df2,id2label)
    assert len(AFTER_TOTAL_SPEAKER_LIST) == len(BEFORE_TOTAL_SPEAKER_LIST),'変換後話者数[{0}], 変換前話者数[{1}]'.format(len(AFTER_TOTAL_SPEAKER_LIST),len(BEFORE_TOTAL_SPEAKER_LIST))
    assert len(BEFORE_NAME_LIST_APPEAR_1) != len(AFTER_NAME_LIST_APPEAR_1),'変換後話者数[{0}], 変換前話者数[{1}]'.format(len(AFTER_NAME_LIST__OVER_1),len(AFTER_NAME_LIST_APPEAR_1))

    return df2 , AFTER_NAME_LIST_APPEAR_1 , AFTER_NAME_LIST__OVER_1

if __name__ == '__main__':
    df2 , AFTER_NAME_LIST_APPEAR_1 , AFTER_NAME_LIST__OVER_1 = get_Data_Agument("./DirectQuote/data/truecased.txt")
    print(len(AFTER_NAME_LIST__OVER_1),len(AFTER_NAME_LIST_APPEAR_1))