#!/usr/bin/env python
# coding=utf-8

# @file classfication.py
# @brief classfication
# @author Anemone95,x565178035@126.com
# @version 1.0
# @date 2018-11-11 19:30

from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
import time

def translate_str(func):
    """ 1. 全部转换成小写，2，过滤非str类型, 3. 删除空格"""
    def handle(s):
        if type(s)==str:
            s=s.lower()
            s.replace(" ","")
            s=func(s)
        else:
            s="Unknown"
        return s
    return handle

@translate_str
def translate_server(server):
    origin2class={
            "Apache":"apache",
            "Microsoft":"microsoft",
            "nginx": "nginx",
            "Nginx": "nginx",
            "lighttpd":"lighttpd",
            "squid":"squid",
            "www.lexisnexis.com": "lexisnexis",
            "Pagely Gateway": "Pagely",
            "marrakesh":"marrakesh",
            "ATS":"ATS",
            "PWS":"PWS",
            "CherryPy":"CherryPy",
            "openresty": "openresty",
            }
    for origin in origin2class:
        if server.startswith(origin):
            return origin2class[origin]

    if server.endswith("codfw.wmnet"):
        return "codfw.wmnet"
    else:
        return server

@translate_str
def translate_charset(charset):
    if charset.startswith("iso"):
        return "iso"
    if charset.startswith("ISO"):
        return "iso"
    else:
        return charset

@translate_str
def translate_country(country):
    if "uk" in country:
        return "uk"
    if "United Kingdom" in country:
        return "uk"
    else:
        return country

@translate_str
def translate_statepro(state):
    return state


def translate_time(t_str):
    # 28/09/2007 16:06
    # 2002-03-20T23:59:59.0Z
    res=-1
    # filter illegal value
    if t_str=='None' or t_str=='b' or t_str=='0':
        return -1
    try:
        res = time.mktime(time.strptime(t_str, "%d/%m/%Y %H:%M"))
    except ValueError:
        res = time.mktime(time.strptime(t_str, "%Y-%m-%dT%H:%M:%S.0Z"))
    return res


def translate_int(_int):
    if type(_int)==float:
        return float(_int)
    return _int

def translate_ip(ip):
    return ip

def classfication():
    dataframe = pd.read_csv('./dataset.csv')
    '''
    有如下字段
    URL,
    URL_LENGTH, *
    NUMBER_SPECIAL_CHARACTERS, *
    CHARSET,
    SERVER,
    CONTENT_LENGTH,
    WHOIS_COUNTRY,
    WHOIS_STATEPRO,
    WHOIS_REGDATE,
    WHOIS_UPDATED_DATE,
    TCP_CONVERSATION_EXCHANGE,
    DIST_REMOTE_TCP_PORT,
    REMOTE_IPS,
    APP_BYTES,
    SOURCE_APP_PACKETS,
    REMOTE_APP_PACKETS,
    SOURCE_APP_BYTES,
    REMOTE_APP_BYTES,
    APP_PACKETS,
    DNS_QUERY_TIMES,
    Type
    '''

    ''' 数据预处理 '''
    # URL只是ID，不作为属性
    dataframe.drop(['URL'], axis=1, inplace=True)

    # 编码 将ISO-8859、ISO-8859-1、iso-8859-1应为一类，UTF-8和utf-8应为一类， 字符型的空值使用人工填写（filling manually）的方式（填写UnKnown）
    # from:
    # ['None', 'ISO-8859', 'ISO-8859-1', 'UTF-8', 'iso-8859-1', 'windows-1251', 'windows-1252', 'us-ascii', 'utf-8']
    # to:
    # ['None', 'iso', 'utf-8', 'windows-1252', 'us-ascii', 'windows-1251']
    dataframe['CHARSET'] = dataframe['CHARSET'].apply(translate_charset)
    class_label = LabelEncoder()
    dataframe["CHARSET"] = class_label.fit_transform(dataframe["CHARSET"])
    #  col=set(dataframe.CHARSET)
    #  print(col)

    # 服务器 按格式归类，去除版本信息，有的服务器类型只有一个，则不必要去除，
    # to：
    # set(['fbs', 'Aeria Games & Entertainment', 'Varnish', 'nxfps', 'CherryPy', 'My Arse', 'squid', nan, '294', 'GSE', 'ATS', 'AmazonS3', 'LiteSpeed', 'Proxy Pandeiro UOL', 'codfw.wmnet', 'Play', 'Scratch Web Server', 'YouTubeFrontEnd', 'ebay server', 'SSWS', 'IdeaWebServer/v0.80', 'MediaFire', 'DPS/1.1.8', 'KHL', 'tsa_c', 'Oracle-iPlanet-Web-Server/7.0', 'Heptu web server', 'cloudflare-nginx', 'marrakesh', 'Sucuri/Cloudproxy', 'Zope/(2.13.16; python 2.6.8; linux2) ZServer/1.1', 'None', 'Jetty(9.0.z-SNAPSHOT)', 'Cowboy', 'Pagely', 'lighttpd', 'HTTPDaemon', 'Server', 'Virtuoso/07.20.3217 (Linux) i686-generic-linux-glibc212-64  VDB', 'nginx', 'lexisnexis', 'Tengine', 'apache', '.V01 Apache', 'AkamaiGHost', 'Roxen/5.4.98-r2', 'DMS/1.0.42', 'Application-Server', 'Yippee-Ki-Yay', 'PWS', 'XXXXXXXXXXXXXXXXXXXXXX', 'ECD (fll/0790)', 'Boston.com Frontend', 'MI', 'Squeegit/1.2.5 (3_sir)', 'Resin/3.1.8', 'Pizza/pepperoni', 'openresty', 'gunicorn/19.7.1', 'Pepyaka/1.11.3', 'Sun-ONE-Web-Server/6.1', 'DOSarrest', 'microsoft', 'barista/5.1.3'])
    dataframe['SERVER'] = dataframe['SERVER'].apply(translate_server)
    class_label = LabelEncoder()
    dataframe["SERVER"] = class_label.fit_transform(dataframe["SERVER"])
    # 分类后结果：
    #  server=set(dataframe.SERVER)
    #  print(server)

    # 国家字段 UK过滤，详细看代码
    dataframe['WHOIS_COUNTRY'] = dataframe['WHOIS_COUNTRY'].apply(translate_country)
    class_label = LabelEncoder()
    dataframe["WHOIS_COUNTRY"] = class_label.fit_transform(dataframe["WHOIS_COUNTRY"])

    # 洲省
    dataframe['WHOIS_STATEPRO'] = dataframe['WHOIS_STATEPRO'].apply(translate_statepro)
    class_label = LabelEncoder()
    dataframe["WHOIS_STATEPRO"] = class_label.fit_transform(dataframe["WHOIS_STATEPRO"])

    # 域名创建时间，有删除无效值，转换两种日期格式为时间戳
    dataframe['WHOIS_REGDATE'] = dataframe['WHOIS_REGDATE'].apply(translate_time)

    # 域名更新时间
    dataframe['WHOIS_UPDATED_DATE'] = dataframe['WHOIS_UPDATED_DATE'].apply(translate_time)

    # 服务器与我们的蜜罐客户端的tcp连接包个数
    dataframe['TCP_CONVERSATION_EXCHANGE'] = dataframe['TCP_CONVERSATION_EXCHANGE'].apply(translate_int)

    # 端口
    dataframe['DIST_REMOTE_TCP_PORT'] = dataframe['DIST_REMOTE_TCP_PORT'].apply(translate_int)

    # IP个数
    dataframe['REMOTE_IPS'] = dataframe['REMOTE_IPS'].apply(translate_int)

    # 网络流量
    dataframe['APP_BYTES'] = dataframe['APP_BYTES'].apply(translate_int)

    # 发送至服务器的包
    dataframe['SOURCE_APP_PACKETS'] = dataframe['SOURCE_APP_PACKETS'].apply(translate_int)

    # 接收服务器的包
    dataframe['REMOTE_APP_PACKETS'] = dataframe['REMOTE_APP_PACKETS'].apply(translate_int)

    # 整个通讯中IP包个数
    dataframe['APP_PACKETS'] = dataframe['APP_PACKETS'].apply(translate_int)

    # DNS 包个数
    dataframe['DNS_QUERY_TIMES'] = dataframe['DNS_QUERY_TIMES'].apply(translate_int)

    # content长度
    dataframe['CONTENT_LENGTH'] = dataframe['CONTENT_LENGTH'].apply(translate_int)

    dataframe['Type'] = dataframe['Type'].apply(translate_int)
    col=set(dataframe['NUMBER_SPECIAL_CHARACTERS'])
    print(col)

    '''检查空值, 数值型的空值使用平均值填充'''
    print(dataframe.isnull().any())
    dataframe.fillna(dataframe.mean(), inplace=True)
    print(dataframe.isnull().any())

    '''计算相关性'''
    print("Pearson Correlation")
    print(dataframe.corr())

    ''' 分类算法比较 '''
    x=dataframe.drop(['Type'], axis=1)
    y=dataframe.Type

    ''' 10折交叉验证 '''
    print("===Mix===")
    print("Random Forest:")
    t1=time.time()
    model = RandomForestClassifier()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision:", np.mean(cores['test_precision']))
    print("Recall:", np.mean(cores['test_recall']))
    print("F1:", np.mean(cores['test_f1']))

    # 朴素贝叶斯效果很差，因为很多属性存在相关性
    print("Naive Bayes:")
    t1=time.time()
    model = GaussianNB()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    print("Support Vector:")
    t1=time.time()
    model = SVC()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    # 神经网络效果差，原因是样本不充分
    # 参数：
    # hidden_layer_sizes：隐藏层个数
    # max_iter：最大迭代次数
    # alpha：误差
    # tol：每次进步阈值
    print("Neural Network:")
    t1=time.time()
    model=MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4, solver='sgd', verbose=10, tol=1e-6, random_state=1)
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    x=dataframe.loc[:,['URL_LENGTH','NUMBER_SPECIAL_CHARACTERS','CHARSET','SERVER','CONTENT_LENGTH','WHOIS_COUNTRY','WHOIS_STATEPRO','WHOIS_REGDATE','WHOIS_UPDATED_DATE']]

    ''' 10折交叉验证 '''
    print("===Application===")
    print("Random Forest:")
    t1=time.time()
    model = RandomForestClassifier()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision:", np.mean(cores['test_precision']))
    print("Recall:", np.mean(cores['test_recall']))
    print("F1:", np.mean(cores['test_f1']))

    # 朴素贝叶斯效果很差，因为很多属性存在相关性
    print("Naive Bayes:")
    t1=time.time()
    model = GaussianNB()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    print("Support Vector:")
    t1=time.time()
    model = SVC()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    print("Neural Network:")
    t1=time.time()
    model=MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4, solver='sgd', verbose=10, tol=1e-6, random_state=1)
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    x=dataframe.loc[:,['TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS','APP_BYTES',
        'SOURCE_APP_PACKETS','REMOTE_APP_PACKETS','APP_PACKETS','DNS_QUERY_TIMES']]

    ''' 10折交叉验证 '''
    print("===Network===")
    print("Random Forest:")
    t1=time.time()
    model = RandomForestClassifier()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision:", np.mean(cores['test_precision']))
    print("Recall:", np.mean(cores['test_recall']))
    print("F1:", np.mean(cores['test_f1']))

    # 朴素贝叶斯效果很差，因为很多属性存在相关性
    print("Naive Bayes:")
    t1=time.time()
    model = GaussianNB()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    print("Support Vector:")
    t1=time.time()
    model = SVC()
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

    print("Neural Network:")
    t1=time.time()
    model=MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4, solver='sgd', verbose=10, tol=1e-6, random_state=1)
    cores = cross_validate(model, x, y, cv=10,
                           scoring=('precision', 'recall', 'f1'),
                           return_train_score=False)
    t2=time.time()
    print("Time:",(t2-t1)*100)
    print("Precision", np.mean(cores['test_precision']))
    print("Recall", np.mean(cores['test_recall']))
    print("F1", np.mean(cores['test_f1']))

if __name__ == '__main__':
    classfication()

