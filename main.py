# coding: utf8
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
#import httplib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from tkinter import Button
import webbrowser
#import twitter
import json
import time
import nltk
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import numpy as np
import pandas as pd
import re
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities 
import gensim
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
#implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import requests
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import webbrowser
LW = 0.3
#import tktable
#wordpunct_tokenize("That's thirty minutes away. I'll be there in ten.")
#print (stopwords.words('english')[0:10])
def polar2xy(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)])

def hex2rgb(c):
    return tuple(int(c[i:i+2], 16)/256.0 for i in (1, 3 ,5))

def IdeogramArc(start=0, end=60, radius=1.0, width=0.2, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    inner = radius*(1-width)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end),
        polar2xy(inner, end),
        polar2xy(inner, end) + polar2xy(opt*(1-width), end-0.5*np.pi),
        polar2xy(inner, start) + polar2xy(opt*(1-width), start+0.5*np.pi),
        polar2xy(inner, start),
        polar2xy(radius, start),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.LINETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CLOSEPOLY,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)


def ChordArc(start1=0, end1=60, start2=180, end2=240, radius=1.0, chordwidth=0.7, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi/180.
    end1 *= np.pi/180.
    start2 *= np.pi/180.
    end2 *= np.pi/180.
    opt1 = 4./3. * np.tan((end1-start1)/ 4.) * radius
    opt2 = 4./3. * np.tan((end2-start2)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start1),
        polar2xy(radius, start1) + polar2xy(opt1, start1+0.5*np.pi),
        polar2xy(radius, end1) + polar2xy(opt1, end1-0.5*np.pi),
        polar2xy(radius, end1),
        polar2xy(rchord, end1),
        polar2xy(rchord, start2),
        polar2xy(radius, start2),
        polar2xy(radius, start2) + polar2xy(opt2, start2+0.5*np.pi),
        polar2xy(radius, end2) + polar2xy(opt2, end2-0.5*np.pi),
        polar2xy(radius, end2),
        polar2xy(rchord, end2),
        polar2xy(rchord, start1),
        polar2xy(radius, start1),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)

def selfChordArc(start=0, end=60, radius=1.0, chordwidth=0.7, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end),
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)

def chordDiagram(X, ax, colors=None, width=0.1, pad=2, chordwidth=0.7):
    """Plot a chord diagram
    Parameters
    ----------
    X :
        flux data, X[i, j] is the flux from i to j
    ax :
        matplotlib `axes` to show the plot
    colors : optional
        user defined colors in rgb format. Use function hex2rgb() to convert hex color to rgb color. Default: d3.js category10
    width : optional
        width/thickness of the ideogram arc
    pad : optional
        gap pad between two neighboring ideogram arcs, unit: degree, default: 2 degree
    chordwidth : optional
        position of the control points for the chords, controlling the shape of the chords
    """
    # X[i, j]:  i -> j
    x = X.sum(axis = 1) # sum over rows
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    if colors is None:
    # use d3.js category10 https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if len(x) > 10:
            print('x is too large! Use x smaller than 10')
        colors = [hex2rgb(colors[i]) for i in range(len(x))]

    # find position for each start and end
    y = x/np.sum(x).astype(float) * (360 - pad*len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5*(start+end)
        #print(start, end, angle)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(tuple(polar2xy(1.1, 0.5*(start+end)*np.pi/180.)) + (angle,))
        z = (X[i, :]/x[i].astype(float)) * (end - start)
        ids = np.argsort(z)
        z0 = start
        for j in ids:
            pos[(i, j)] = (z0, z0+z[j])
            z0 += z[j]
        start = end + pad

    for i in range(len(x)):
        start, end = arc[i]
        IdeogramArc(start=start, end=end, radius=1.0, ax=ax, color=colors[i], width=width)
        start, end = pos[(i,i)]
        selfChordArc(start, end, radius=1.-width, color=colors[i], chordwidth=chordwidth*0.7, ax=ax)
        for j in range(i):
            color = colors[i]
            if X[i, j] > X[j, i]:
                color = colors[j]
            start1, end1 = pos[(i,j)]
            start2, end2 = pos[(j,i)]
            ChordArc(start1, end1, start2, end2,
                     radius=1.-width, color=colors[i], chordwidth=chordwidth, ax=ax)

    #print(nodePos)
    return nodePos
#---------------------------------------------------------------отражение резюме соискателя

def scoring ():
    window3 = tk.Toplevel(root)
    window3.minsize(800,800)
    window3.title(u"Оценка и рекомендованные курсы")
    #fig2 = plt.figure(figsize=(7,5))
    #photo=PhotoImage(file="22.png")
    c33 = Button(window3, text=u"Udemy", command=online_course)
    c33.place(relx=0.2, rely=0.99, anchor=SE)
    c44 = Button(window3, text=u"Cousera", command=online_course2)
    c44.place(relx=0.4, rely=0.99, anchor=SE)
    c55 = Button(window3, text=u"EdX", command=online_course3)
    c55.place(relx=0.6, rely=0.99, anchor=SE)
    #for i in range(4):
        #for j in range(2):
            #tk.Entry(window3).grid(row=i, column=j)
    #c33.config(image=photo,compound=RIGHT)
    #img2 = ImageTk.PhotoImage(Image.open("2.png"))
    #im = plt.imshow(img)
    #canvas = FigureCanvasTkAgg(fig, master=window3)
    #canvas.show()
    #canvas.get_tk_widget().place(relx=0.54, rely=0.03)#pack(side=TOP, fill=BOTH, expand=1)
    #canvas._tkcanvas.place(relx=0.52, rely=0.03)#pack(side=TOP, fill=BOTH, expand=1)
    #window3()
    """
    img2 = ImageTk.PhotoImage(Image.open("2.png"))
    optimized_canvas = Canvas(window3)
    optimized_canvas.pack(fill=BOTH, expand=1)
    optimized_image = optimized_canvas.create_image(0, 0, anchor=NW, image=img2)
    """
    #---------------------------------------------------------------отражение обучения
    w88=Label(window3,text=u"ХОРДОВАЯ ДИАГРАММА", font = "Times")
    w88.place(relx=0.3, rely=0.01)
    w99=Label(window3,text=u"ПРОЙДИТЕ КУРСЫ СО СКИДКОЙ 30%", font = "Times")
    w99.place(relx=0.25, rely=0.9)
    fig = plt.figure(figsize=(6,6))
    flux = np.array([[11975,  5871, 8916, 2868],
      [ 1951, 10048, 2060, 6171],
      [ 8010, 16145, 8090, 8045],
      [ 1013,   990,  940, 6907]
    ])

    ax = plt.axes([0,0,1,1])

    #nodePos = chordDiagram(flux, ax, colors=[hex2rgb(x) for x in ['#666666', '#66ff66', '#ff6666', '#6666ff']])
    nodePos = chordDiagram(flux, ax)
    ax.axis('off')
    prop = dict(fontsize=16*0.8, ha='center', va='center')
    nodes = [u'Компетенции', u'Опыт', u'Образование', u'Пройденные курсы']
    for i in range(4):
        ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)
    #im = plt.imshow(wc)
    canvas = FigureCanvasTkAgg(fig, master=window3)
    canvas.show()
    canvas.get_tk_widget().place(relx=0.1, rely=0.1)#pack(side=TOP, fill=BOTH, expand=1)
    #canvas._tkcanvas.place(relx=0.1, rely=0.1)#pack(side=TOP, fill=BOTH, expand=1)

    #plt.savefig("example2.png", dpi=600,
            #transparent=True,
            #bbox_inches='tight', pad_inches=0.02)

def online_course():
    
    text_file = open("Output.txt", "r")
    worl=text_file.read()
    text_file.close()
    new = 1
    url_udemy = 'https://www.udemy.com/courses/search/?ref=home&q='+str(worl)
    url_courer='https://ru.coursera.org/courses?query='+str(worl)
    webbrowser.open(url_udemy,new=new)
def online_course2():
    
    text_file = open("Output.txt", "r")
    worl=text_file.read()
    text_file.close()
    new = 1
    url_udemy = 'https://www.udemy.com/courses/search/?ref=home&q='+str(worl)
    url_courer='https://ru.coursera.org/courses?query='+str(worl)
    webbrowser.open(url_courer,new=new)
def online_course3():
    
    text_file = open("Output.txt", "r")
    worl=text_file.read()
    text_file.close()
    new = 1
    url_edx= 'https://www.edx.org/course?search_query='+str(worl)
    #url_courer='https://ru.coursera.org/courses?query='+str(worl)
    webbrowser.open(url_edx,new=new)
def testing():
    window2 = tk.Toplevel(root)
    window2.minsize(1300,800)
    window2.title(u"Тест")
    c = Button(window2, text=u"Оценка", font = "Times 14 bold", command=scoring)
    c.place(relx=0.9, rely=0.95, anchor=SE)
def show_entry_fields():
    url='http://api.hh.ru/vacancies?text='+(e1.get())+'&page=0&per_page=100'
    data = requests.get(url).json()
    print ("Поиск вакансий")
    p = json.dumps(data)
    res2 = json.loads(p)
    i=0
    texts = []
    total_word=[]
    window = tk.Toplevel(root)
    window.minsize(1300,1000)
    window.title(u"Вывод данных")
    #webbrowser.open("index.html")
    w00=Label(window,text=u"ВАКАНСИИ", font = "Times")
    w00.place(relx=0.2, rely=0.01)
    t1=Text(window, height=60, width=75)
    t1.place(relx=0.01, rely=0.03)
    w11=Label(window,text=u"НАПИСАТЬ СОПРОВОДИТЕЛЬНОЕ ПИСЬМО", font = "Times")
    w11.place(relx=0.64, rely=0.57)
    t2=Text(window, height=20, width=70)
    t2.place(relx=0.52, rely=0.6)
    while i<len(res2['items']):
            a=((res2['items'][i]['id']))#['requirement']
            #print (a)
            #print ((res2['items'][i]['name']))
            aa=((res2['items'][i]['snippet']['requirement']))
            #aa=(res2['items'][i]['snippet']['requirement']).replace('<highlighttext>', '')
            #patt = re.compile('(\s*)aa(\s*)')
            print (aa)
            
            
            texts.append(aa)
            #wordpunct_tokenize(str(aa))
            tokenizer = RegexpTokenizer(r'\w+')
            #print (stopwords.words('english'))
            (total_word.extend(tokenizer.tokenize(str(aa))))

            aaa=str(i+1)+') '+str(res2['items'][i]['name'])+ ' | '+str(res2['items'][i]['area']['name'])+'\n'
            t1.insert(END, (aaa))
            i=i+1





    #----------------------------------------------------------------------формирование окна выдачи результатов
    stopwords = nltk.corpus.stopwords.words('english')
    en_stop = get_stop_words('en')
    stemmer = SnowballStemmer("english")
    #print stopwords[:10]
    


    


      


    #--------------------------------------------------------------------------скрытое размещение дирихле
    #w8=Label(window,text=u"ОСНОВНЫЕ ТЕМЫ И СЛОВА", font = "Times")
    #w8.place(relx=0.17, rely=0.53)
    #t8=Text(window, height=24, width=75)
    #t8.place(relx=0.01, rely=0.57)
    texts = []
    stopped_tokens = [i for i in total_word if not i in en_stop]
    #print le(stopped_tokens)
    p_stemmer = PorterStemmer()
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    #print len(stemmed_tokens), stemmed_tokens
    texts.append(stemmed_tokens)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.LdaModel(corpus, num_topics=100, id2word = dictionary, passes=20)
    a=ldamodel.print_topics(num_topics=10, num_words=7)
    #print ldamodel.print_topics(num_topics=4, num_words=7)[0][1]
    #print a
    num_topics = 5
    topic_words = []
    for i in range(num_topics):
        tt = ldamodel.get_topic_terms(i,10)
        topic_words.append([dictionary[pair[0]] for pair in tt])
    #print topic_words[0]
    jj=0
    while jj<len(topic_words):
        topic11=((u"Тема #%d:" % (jj+1))+"\n"+"-".join(topic_words[jj])+"\n")
        #t8.insert(END, topic11)
        #print(u"Тема #%d:" % (jj+1))
        #print("-".join(topic_words[jj]))
        jj=jj+1    
    #--------------------------------------------------------------------------определение основных компетенций
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=.5)
    tfv = vec.fit_transform(stopped_tokens)
    terms = vec.get_feature_names()
    result=list(set(list_skills) & set(terms))
    print (result)
    text_file = open("Output.txt", "w")
    text_file.write(result[2])
    text_file.close()
    wc = WordCloud(height=1000, width=1000, max_words=1000).generate(" ".join(terms))
    nmf = NMF(n_components=11).fit(tfv)
    #for idx, topic in enumerate(nmf.components_):
        #print(u"Тема #%d:" % (idx+1))
        #print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))
    #--------------------------------------------------------------------------рисунок распределения терминов
    w8=Label(window,text=u"РАСПРЕДЕЛЕНИЕ НАВЫКОВ", font = "Times")
    w8.place(relx=0.66, rely=0.01)
    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(wc)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.show()
    canvas.get_tk_widget().place(relx=0.54, rely=0.03)#pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.place(relx=0.52, rely=0.03)#pack(side=TOP, fill=BOTH, expand=1)
    #--------------------------------------------------------------------------оцека тональности
    c = Button(window, text=u"Подтвердить квалификацию", font = "Times 14 bold", command=scoring, bg="deep sky blue")
    c.place(relx=0.95, rely=0.97, anchor=SE)
    c1 = Button(window, text=u"Откликнуться", font = "Times 14 bold", command=testing, bg="lime green")
    c1.place(relx=0.7, rely=0.97, anchor=SE)
list_skills=['1с', '3х', 'android', 'api', 'asp', 'company', 'cpc', 'cpi', 'degree', 'developer', 'digital', 'disc', 'dropbox', 'elco', 'experience', 'fast', 'gaap', 'goodle', 'growing', 'hiring', 'hr', 'html', 'iiko', 'ios', 'java', 'keeper', 'knowledge', 'linux', 'master', 'net', 'nodejs', 'npm', 'objective', 'oilon', 'photoshop', 'php', 'quot', 'restful', 'roles', 'rx', 'sdks', 'software', 'stl', 'storehouse', 'swift', 'technical', 'typescript', 'viessmann', 'weishaupt', 'work', 'xcode', 'xml', 'xpath', 'xslt', 'angular', 'ansible', 'api', 'applications', 'asyncio', 'backbone', 'backend', 'bash', 'batch', 'bootstrap', 'caffe', 'celery', 'ci', 'communication', 'css', 'css3', 'data', 'databases', 'developer', 'development', 'devops', 'different', 'django', 'docker', 'elasticsearch', 'excellent', 'experience', 'expertise', 'familiarity', 'flask', 'formset', 'framework', 'frameworks', 'frontend', 'git', 'gnu', 'golang', 'good', 'gpu', 'grep', 'highlighttext', 'html', 'html5', 'java', 'javascript', 'jenkins', 'jquery', 'js', 'keras', 'knowledge', 'kubernetes', 'library', 'linux', 'marionette', 'mastery', 'middleware', 'mongodb', 'mvc', 'mysql', 'nix', 'nodejs', 'nosql', 'numpy', 'opencv', 'orm', 'ot', 'pep8', 'perl', 'postgesql', 'postgres', 'postgresql', 'postrgesql', 'preferably', 'professional', 'programming', 'proven', 'pytest', 'python', 'rabbitmq', 'react', 'redis', 'relational', 'rest', 'restful', 'science', 'scipy', 'sed', 'shell', 'skills', 'sklearn', 'sql', 'sqlalchemy', 'standard', 'strong', 'tastypie', 'tdd', 'tensorfflow', 'tensorflow', 'theano', 'tornado', 'twisted', 'ui', 'verbal', 'web']
root = Tk()
root.minsize(900,580)
root.title("UpYourSkills")
img = ImageTk.PhotoImage(Image.open("1.png"))
panel = Label(root, image = img)
w1=Label(root, text=u"Поиск работы", font = "Times 16 bold")
w1.place(relx=0.45, rely=0.95, anchor=SE)
e1 = Entry(root)
e1.place(relx=0.6, rely=0.94, anchor=SE)
panel.pack(side = "bottom", fill = "both", expand = "yes")
b = Button(root, text=u"Перейти к поиску", font = "Times 14 bold",  bg="deep sky blue", command=show_entry_fields)
b.place(relx=0.9, rely=0.95, anchor=SE)
#root.geometry("500x500")
#app = App(root)   
root.mainloop()
