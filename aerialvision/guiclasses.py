#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Wilson W. L. Fung, Tor M. Aamodt, Andrew 
# Turner and the University of British Columbia, Vancouver, 
# BC V6T 1Z4, All Rights Reserved.
# 
# THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
# TERMS AND CONDITIONS.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 
# NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
# are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
# (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
# benchmarks/template/ are derived from the CUDA SDK available from 
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
# src/intersim/ are derived from Booksim (a simulator provided with the 
# textbook "Principles and Practices of Interconnection Networks" available 
# from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# which is distributed seperately by NVIDIA under separate terms and 
# conditions.  To use this version of GPGPU-Sim with OpenCL requires a
# recent version of NVIDIA's drivers which support OpenCL.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the University of British Columbia nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
#  
# 5. No nonprofit user may place any restrictions on the use of this software,
# including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4

"""
[한국어 설명] AerialVision GUI 및 matplotlib 플로팅 클래스 (guiclasses.py)

=== 파일의 역할 ===
AerialVision의 핵심 GUI 구성 요소와 matplotlib 기반 시각화 로직을 담는다.
- formEntry: Time Lapse View의 그래프 설정 폼(파일/X축/Y축/그래프종류/서브플롯)
- subplotInstance: 서브플롯 설정 폼 (formEntry 상속)
- PlotFormatInfo: 플롯 제목/레이블/폰트/색상맵 포맷 정보
- graphManager: matplotlib Figure 생성, 데이터 타입별 플로팅, 사용자 인터랙션
- NaviPlotInfo: Source Code View 폰트 설정
- newTextTab: Source Code View의 CUDA/PTX 소스코드 및 라인별 통계 표시

=== 전체 아키텍처에서의 위치 ===
  aerialvision/startup.py
        ↓
  aerialvision/guiclasses.py (이 파일)
        ↓
  matplotlib / Tk / Pmw

=== 타 모듈과의 연결 ===
- startup.py: formEntry, newTextTab 인스턴스 생성
- lexyacc.py: parseMe()로부터 vars[파일명] 데이터 획득
- organizedata.py: 데이터 재배열 후 형태(type 1~5)에 따라 시각화 분기
- lexyaccbookmark.py: 즐겨찾기 파싱
- lexyacctexteditor.py: Source Code View에서 .stats/.ptx 파싱
- configs.py: PlotFormatInfo/NaviPlotInfo에서 avconfig.get_value()로 폰트/회전 옵션 조회

=== 주요 함수/구조체 요약 ===
- formEntry: 그래프 탭 폼, 파일/변수/그래프 선택, 서브플롯 관리
- subplotInstance: formEntry 상속, 서브플롯별 설정
- PlotFormatInfo: 플롯 레이블/폰트/색상맵 관리
- graphManager: matplotlib Figure, type1~5Variable, dy/dx, 커널 경계 표시, 인터랙션
- NaviPlotInfo: Source Code View 폰트 정보
- newTextTab: 소스코드 뷰 텍스트/히스토그램 UI
"""

import time
import os
import array
import tkinter as Tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib.colors import colorConverter
from matplotlib import pyplot
import Pmw
import numpy
import startup
import lexyaccbookmark
import copy
import lexyacc
import organizedata
import lexyacctexteditor
import variableclasses
from configs import avconfig

# [한국어] Time Lapse View 그래프 설정 폼 클래스.
# 하나의 탭 안에 배치되어 사용자가 파일/X축/Y축/그래프 종류/서브플롯을 선택할 수 있게 한다.
class formEntry:
  
  #This class is essentially a form placed inside a tab. It collects all the data from the user required for graphing. It then instantiates a new object that takes care of all the graphing
  # [한국어] 변수 타입 -> 가능한 그래프 인덱스 매핑.
  # possGraphs 순서: ['Line','Histogram','Bar Chart','Parallel Intensity Plot',
  #                    'Stacked Bar Chart','Parallel Intensity Plot (Sum)','Scatter Plot']
  graphForVarType = { 1:[0,3], 2:[0,3], 3:[4], 4:[0,3,5], 5:[6] }
  
  # [한국어]
  # 생성자: 탭 내 폼 UI를 구성한다.
  # @graphTabs: 부모 Pmw.NoteBook
  # @numb: 탭 번호 문자열
  # @vars: {파일명: {변수명: variable}} 데이터 딕셔너리
  # @res: 해상도 문자열 ('small', 'medium', 'big')
  # @entry: 탭 제목 입력 Entry 위젯
  def __init__(self, graphTabs, numb, vars, res, entry):
    
    #Variable Initializations
    self.data = vars
    self.subBool = 0
    self.possGraphs = ['Line', 'Histogram', 'Bar Chart', 'Parallel Intensity Plot', 'Stacked Bar Chart', 'Parallel Intensity Plot (Sum)', 'Scatter Plot']
    self.res = res
    self.subplots = []
    self.dataChosenX = "globalCycle"
    self.dataChosenY = ""
    self.dydx = 0
    self.fileChosen = ""
    self.graphChosen = ''
    self.num = 0
     
    #Setting a title to this tab
    # [한국어] 탭 제목 설정. 기본값 "TabTitle?"이면 "Page N"으로 대체
    if entry.get() == "TabTitle?":
        self.tabnum = "Page " + numb
    else:
        self.tabnum = entry.get()
    self.page = graphTabs.add(self.tabnum)
    
    
    #Size of the self.background depending on how large the user screen is
    # [한국어] 해상도에 따른 폼 배경 크기 선택
    if res == "small":
        self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 700, width = 1200);
        self.formArea = Tk.Frame(self.background, bg = "white")
    elif res == 'medium':
        self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
        self.formArea = Tk.Frame(self.background, bg = "white")
    else:
        self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
        self.formArea = Tk.Frame(self.background, bg = "white")
    chosenVars = Tk.Frame(self.background, bg = 'white')
    self.background.pack()
    # self.background.pack_propagate(0)
    self.formArea.pack(side = Tk.LEFT, anchor = Tk.N)
    chosenVars.pack(side = Tk.LEFT, anchor = Tk.N, padx = 15)
    
  
    # Frame and listboxes for choosing what file you want to use data from 
    whichFileFrame = Tk.Frame(self.formArea, bg= 'white')
    whichFileFrame.pack(side = Tk.TOP, anchor = Tk.W, pady = 5)
    lwhichFileFrame = Tk.Label(whichFileFrame, text = 'Choose a File:', font = ("Gills Sans MT", 12), bg = "white")
    lwhichFileFrame.pack(side = Tk.LEFT)
    self.cWhichFile= Tk.Listbox(whichFileFrame, width = 85, height = 4)
    self.cWhichFile.pack(side =Tk.LEFT)
    self.cWhichFile.bind("<Double-Button-1>", self.chooseFile)
    
    #Placing the available filenames in the self.cWhichFileFrame Listbox
    # [한국어] 사용 가능한 파일 이름을 정렬하여 Listbox에 삽입
    for files in sorted(self.data):
      self.cWhichFile.insert(Tk.END, files)

    # move to end of lines to actually see the filenames
    self.cWhichFile.xview_moveto(1)

    #Favourites Button
    self.bFavourites = Tk.Button(whichFileFrame, text = "Favourites", state = Tk.DISABLED, command = self.chooseFavourite )
    self.bFavourites.pack(side = Tk.LEFT, padx = 5)
    
 
    #Frame for choosing what data you are going to plot
    chooseDataFrame = Tk.Frame(self.formArea, bg = "white")
    chooseDataFrame.pack(side = Tk.TOP, anchor = Tk.W)
    lchooseDataFrame = Tk.Label(chooseDataFrame, text= 'Data to be Plotted:', font = ("Gills Sans MT", 12), bg = "white")
    lchooseDataFrame.pack(side= Tk.LEFT);
    bQchooseDataFrameHelp = Tk.Button(chooseDataFrame, text = " ? -->", command = (lambda: self.helpMsg('Double-Click on the metric to select it for plotting.')))
    bQchooseDataFrameHelp.pack(side =Tk.LEFT, padx = 5)
    XListbox = Tk.Frame(chooseDataFrame, bg= 'white')
    XListbox.pack(side = Tk.LEFT)
    XTitle = Tk.Label(XListbox, text = "X Vars", bg= 'white')
    XTitle.pack(side = Tk.TOP)
    self.cXAxisData = Tk.Listbox(XListbox, width = 19, height = 5, selectmode = Tk.MULTIPLE)
    self.cXAxisData.bind("<Double-Button-1>", self.chooseDataX)
    YListbox = Tk.Frame(chooseDataFrame, bg = 'white')
    YListbox.pack(side = Tk.LEFT)
    YFrameForScrollbar = Tk.Frame(YListbox, bg= 'white')
    YFrameForScrollbar.pack(side = Tk.BOTTOM)
    scrollYAxisData = Tk.Scrollbar(YFrameForScrollbar, orient = Tk.VERTICAL)
    YTitle = Tk.Label(YListbox, text = 'Y Vars', bg= 'white')
    YTitle.pack(side = Tk.TOP)
    self.cYAxisData = Tk.Listbox(YFrameForScrollbar,width = 19, height = 5, yscrollcommand=scrollYAxisData.set)
    self.cYAxisData.bind("<Double-Button-1>", self.chooseDataY)
    scrollYAxisData.config(command=self.cYAxisData.yview)
    self.cXAxisData.pack(side = Tk.BOTTOM)
    self.cYAxisData.pack(side = Tk.LEFT)
    scrollYAxisData.pack(side=Tk.LEFT, fill = 'y')
    
    # The Take Derivative Checkbutton
    self.var0 = Tk.IntVar()
    checkbDyDx = Tk.Checkbutton(chooseDataFrame, text= "dy/dx", variable= self.var0, bg = 'white', command = (lambda: self.checkDyDx()))
    checkbDyDx.pack(side= Tk.LEFT, padx = 5)
    
    #Frame for choosing what type of graph
    typeGraph = Tk.Frame(self.formArea, bg = "white")
    typeGraph.pack(side = Tk.TOP, anchor = Tk.W, pady = 10)
    lTypeGraph = Tk.Label(typeGraph, text= 'Type of Graph:', font = ("Gills Sans MT", 12), bg = "white")
    lTypeGraph.pack(side= Tk.LEFT);
    bQTypeGraphHelp = Tk.Button(typeGraph, text = " ? -->", command = (lambda: self.helpMsg('Double-Click on the graph type to select it.')))
    bQTypeGraphHelp.pack(side =Tk.LEFT, padx = 5)
    self.cTypeGraph = Tk.Listbox(typeGraph, width = 50, height = 3)
    self.cTypeGraph.bind("<Double-Button-1>", self.chooseGraph)
    self.cTypeGraph.pack(side = Tk.LEFT)
    
    subplotWindow = Tk.Frame(self.formArea, bg = "white")
    subplotWindow.pack(side = Tk.TOP, anchor = Tk.W)
    lSubplot = Tk.Label(subplotWindow, bg = "white", text = "Add Subplot:",  font = ("Gills Sans MT", 12))
    lSubplot.pack(side =Tk.LEFT, anchor = Tk.N)
    lnumSubplot = Tk.Label(subplotWindow, bg= 'white', text= "# Subplots -")
    lnumSubplot.pack(side = Tk.LEFT, anchor = Tk.S)
    subplotSlider = Tk.Scale(subplotWindow, from_=1, to=5, orient = Tk.HORIZONTAL, bg= 'white')
    subplotSlider.pack(side = Tk.LEFT, anchor = Tk.N)
    bSubplotSlider = Tk.Button(subplotWindow, text = "Submit", command = lambda: (self.addSubplot(subplotSlider.get())))
    bSubplotSlider.pack(side = Tk.LEFT)
    bcancelSubplot = Tk.Button(subplotWindow, text= "Cancel", command = lambda: self.removeSubplotWindow())
    bcancelSubplot.pack(side = Tk.LEFT)
    
    #Button for graphing  
    graphButton = Tk.Frame(chosenVars, bg = "white")
    graphButton.pack(side = Tk.TOP, fill = Tk.X)
    bGraphButton = Tk.Button(graphButton,borderwidth = 5, bg = 'green', text = "GraphMe!",font = ("Gills Sans MT", 14), command = (lambda: self.setupPlotData()))
    bGraphButton.pack(side = Tk.RIGHT, pady = 5)
    
    #Frame that will have the textbox that lists everything the user has chosen to do
    ChosenVarsTitle = Tk.Label(chosenVars, text = 'Options Chosen', font = ("Gills Sans MT", 12), bg = "white")
    ChosenVarsTitle.pack(side = Tk.TOP)
    innerChosenVarsFrame = Tk.Frame(chosenVars, bg= 'white')
    innerChosenVarsFrame.pack(side = Tk.TOP, pady = 10, fill = Tk.Y)
    ChosenVarsTextboxScrollbar = Tk.Scrollbar(innerChosenVarsFrame, orient = Tk.VERTICAL)
    ChosenVarsTextboxScrollbar.pack(side = Tk.RIGHT, fill = Tk.Y)
    self.ChosenVarsTextbox = Tk.Text(innerChosenVarsFrame, height = 41, width = 45, yscrollcommand = ChosenVarsTextboxScrollbar.set)
    self.ChosenVarsTextbox.pack(fill = Tk.Y)
    ChosenVarsTextboxScrollbar.config(command = self.ChosenVarsTextbox.yview)
    
    
    #Setting up subplot stuff
    # [한국어] 서브플롯 영역 초기화. Pmw.NoteBook으로 서브플롯 탭 관리
    self.subplotWindow = Tk.Frame(self.formArea, bg='green', height = 300, width = 700)
    self.subplotWindow.pack(side = Tk.TOP, pady = 10)
    self.subplotWindow.pack_propagate(0)
    self.subplotTabs = Pmw.NoteBook(self.subplotWindow)
    self.subplotTabs.pack(fill='both',expand = 'True')
    tmpInnerSubplotWindow = self.subplotTabs.add('No Subplots Chosen Yet')
    tmpInnerSubplotWindow.pack()
    tmpInnerSubplotWindow1 = Tk.Frame(tmpInnerSubplotWindow, bg = 'white', height = 300, width = 300)
    tmpInnerSubplotWindow1.pack(fill = 'both')
    tmpInnerSubplotWindow1.pack_propagate(0)
    tmpInnerSubplotWindow1Title = Tk.Label(tmpInnerSubplotWindow1, text = 'You have not chosen to have any subplots', bg= 'white')
    tmpInnerSubplotWindow1Title.pack()
    
    self.updateChosen()
    
    
  # [한국어]
  # chooseFile: 파일 Listbox에서 더블클릭 시 호출.
  # 선택한 파일의 X축(globalCycle)과 Y축 변수들을 Listbox에 채운다.
  def chooseFile(self, *event):
    try:
      self.bFavourites.config( state = Tk.NORMAL )
    except:
      pass
    self.fileChosen = self.cWhichFile.get('active')
    #Initialize both 'cXAxisData' and 'self.cYAxisData' to empty listboxes
    self.cXAxisData.delete(0, Tk.END)
    self.cYAxisData.delete(0, Tk.END)
    
      
    #filling in xAxis vars
    # [한국어] X축으로는 globalCycle만 제공
    for keys in list(self.data[self.fileChosen].keys()):
        if keys == 'globalCycle':
            self.cXAxisData.insert(Tk.END, keys)
            
    #filling in yAxis vars
    #Need to fill up list alphabetically
    keysAlpha = []
    for key in list(self.data[self.fileChosen].keys()):
        if key not in ['globalCycle','CFLOG','EXTVARS']:#exclude hacks from list
            keysAlpha.append(key)
    #keysAlpha.sort(key=lambda x, y: cmp(x.lower(),y.lower()))
    for keys in keysAlpha:
        self.cYAxisData.insert(Tk.END, keys)
            
    self.updateChosen()
    
  
  # [한국어]
  # chooseDataX: X축 변수 선택. 선택 후 Y축 변수 타입에 따라 가능한 그래프 종류 갱신.
  def chooseDataX(self, *event):
    self.dataChosenX = self.cXAxisData.get('active')
    self.cTypeGraph.delete(0,Tk.END)


    if ((self.dataChosenX != "") and (self.dataChosenY != "")):
      if self.data[self.fileChosen][self.dataChosenY].type == 1 or self.data[self.fileChosen][self.dataChosenY].type == 2 or self.data[self.fileChosen][self.dataChosenY].type == 4:
          self.cTypeGraph.insert(Tk.END, self.possGraphs[0])
          self.cTypeGraph.insert(Tk.END, self.possGraphs[3])
      else:
          self.cTypeGraph.insert(Tk.END, self.possGraphs[4])
           

    
    
    self.updateChosen()
  
  # [한국어]
  # chooseDataY: Y축 변수 선택. 변수 타입에 따라 그래프 종류 Listbox 갱신.
  def chooseDataY(self, *event):
    self.dataChosenY = self.cYAxisData.get('active')
    self.cTypeGraph.delete(0,Tk.END)
    
    
    if ((self.dataChosenX != "") and (self.dataChosenY != "")):
      varType = self.data[self.fileChosen][self.dataChosenY].type
      for g in formEntry.graphForVarType[varType]:
          self.cTypeGraph.insert(Tk.END, self.possGraphs[g])
          
    self.updateChosen()
                
                
  
  # [한국어]
  # checkDyDx: dy/dx 체크박스 상태를 self.dydx에 반영.
  def checkDyDx(self):
    self.dydx = self.var0.get()

        
    self.updateChosen()

  # [한국어]
  # chooseGraph: 그래프 종류 선택.
  def chooseGraph(self,num):
    self.graphChosen = self.cTypeGraph.get('active')
    self.updateChosen()
  
  # [한국어]
  # updateChosen: 우측 'Options Chosen' 텍스트 박스에 현재 선택 상태를 업데이트.
  # 완료된 항목은 녹색, 미완료는 빨간색 배경으로 표시.
  def updateChosen(self):
    self.ChosenVarsTextbox.tag_config('title', font = ("Gills Sans MT", 12), justify = 'center', spacing1 = 0.5, underline = 1)
    self.ChosenVarsTextbox.tag_config('complete', background = 'green')
    self.ChosenVarsTextbox.tag_config('incomplete', background= 'red')
    if self.dydx == 1:
      check = 'YES'
    else:
      check = 'NO'
    if self.num == 0:
      self.ChosenVarsTextbox.delete(0.0, Tk.END)
      self.ChosenVarsTextbox.insert(Tk.END, 'Plot: \n', ("title"))
      if self.fileChosen == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'FileChosen: ' + self.fileChosen + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'FileChosen: ' + self.fileChosen + '\n', ('complete'))
      if self.dataChosenX == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'XAxis: ' + self.dataChosenX + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'XAxis: ' + self.dataChosenX + '\n', ('complete'))
      if self.dataChosenY == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'YAxis: ' + self.dataChosenY + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'YAxis: ' + self.dataChosenY + '\n', ('complete'))
      if self.graphChosen == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'graphChosen: ' + self.graphChosen + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'graphChosen: ' + self.graphChosen + '\n', ('complete'))
      self.ChosenVarsTextbox.insert(Tk.END, 'dydx: ' + check + '\n')
    else:
      if self.ChosenVarsTextbox.search('Subplot0', index = 0.0) != '':
        self.ChosenVarsTextbox.delete(self.ChosenVarsTextbox.search('Subplot0', index = 0.0), Tk.END)
    for iter in self.subplots:
      if iter.dydx == 1:
        check = 'YES'
      else:
        check = 'NO'
      self.ChosenVarsTextbox.insert(Tk.END, '\n')
      self.ChosenVarsTextbox.insert(Tk.END, 'Subplot' + str(self.subplots.index(iter)) + ': \n', ("title"))
      if iter.fileChosen == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'FileChosen: ' + iter.fileChosen + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'FileChosen: ' + iter.fileChosen + '\n', ('complete'))
      if iter.dataChosenX == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'XAxis: ' + iter.dataChosenX + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'XAxis: ' + iter.dataChosenX + '\n', ('complete'))
      if iter.dataChosenY == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'YAxis: ' + iter.dataChosenY + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'YAxis: ' + iter.dataChosenY + '\n', ('complete'))
      if iter.graphChosen == '':
        self.ChosenVarsTextbox.insert(Tk.END, 'graphChosen: ' + iter.graphChosen + '\n', ('incomplete'))
      else:
        self.ChosenVarsTextbox.insert(Tk.END, 'graphChosen: ' + iter.graphChosen + '\n', ('complete'))
      
      self.ChosenVarsTextbox.insert(Tk.END, 'dydx: ' + check + '\n')
      

    
  
  # [한국어]
  # addSubplot: 서브플롯 개수를 지정하여 새 서브플롯 폼을 생성.
  def addSubplot(self, subNum):
    self.removeSubplotWindow()
    self.subplots = []
    if self.subBool == 1:
      pass
    else:
      self.subplotTabs.delete(Pmw.SELECT)
      subplotFrames = []
      for subplotForms in range(0,subNum):
        subplotFrames.append(self.subplotTabs.add('Subplot' + str(subplotForms)))
        self.subplots.append(subplotInstance(subplotFrames[-1], self.data, self, subplotForms + 1))
    self.updateChosen()
        
  # [한국어]
  # modSubplot: 기존 서브플롯 개수를 oldNum에서 subNum으로 조정(추가/삭제).
  def modSubplot(self, oldNum, subNum):
    if (oldNum > subNum): #trucate
      self.subplots = self.subplots[:subNum]
      return
    if (oldNum == subNum): #nothing needs doing
      return
    if (oldNum > 0): #some subplots have been specified add remaining
      for subplotForms in range(oldNum,subNum):
        subplotFrames = []
        subplotFrames.append(self.subplotTabs.add('Subplot' + str(subplotForms)))
        self.subplots.append(subplotInstance(subplotFrames[-1], self.data, self, subplotForms + 1))
      self.updateChosen()
      return
    #there are no subplots, add them:
    self.removeSubplotWindow()
    self.subplots = []
    self.subplotTabs.delete(Pmw.SELECT)
    subplotFrames = []
    for subplotForms in range(0,subNum):
      subplotFrames.append(self.subplotTabs.add('Subplot' + str(subplotForms)))
      self.subplots.append(subplotInstance(subplotFrames[-1], self.data, self, subplotForms + 1))
    self.updateChosen()


  # [한국어]
  # removeSubplotWindow: 서브플롯 영역을 초기 상태로 복원.
  def removeSubplotWindow(self):
    self.subplotWindow.destroy()
    #Setting up subplot stuff
    self.subplotWindow = Tk.Frame(self.formArea, bg='green', height = 300, width = 700)
    self.subplotWindow.pack(side = Tk.TOP, pady = 10)
    self.subplotWindow.pack_propagate(0)
    self.subplotTabs = Pmw.NoteBook(self.subplotWindow)
    self.subplotTabs.pack(fill='both',expand = 'True')
    tmpInnerSubplotWindow = self.subplotTabs.add('No Subplots Chosen Yet')
    tmpInnerSubplotWindow.pack()
    tmpInnerSubplotWindow1 = Tk.Frame(tmpInnerSubplotWindow, bg = 'white', height = 300, width = 300)
    tmpInnerSubplotWindow1.pack(fill = 'both')
    tmpInnerSubplotWindow1.pack_propagate(0)
    tmpInnerSubplotWindow1Title = Tk.Label(tmpInnerSubplotWindow1, text = 'You have not chosen to have any subplots', bg= 'white')
    tmpInnerSubplotWindow1Title.pack()
    self.subplots = []
    self.updateChosen()
    
    
  # [한국어]
  # setupPlotData: 모든 필드가 선택되었는지 확인하고, graphManager를 생성하여 플롯을 그린다.
  def setupPlotData(self):
    bool = 1
    error = 0
    for iter in self.subplots:
      if iter.dataChosenX == '' or iter.dataChosenY == '' or iter.graphChosen == '' or iter.fileChosen == '':
        bool = 0
        error = 1
    
    if self.dataChosenX == '' or self.dataChosenY == '' or self.graphChosen == '' or self.fileChosen == '':
      bool = 0
      error = 1
      
    if error == 1:
      self.errorMsg('You must choose from all fields before you may graph')
  
    if bool == 1:
      self.background.destroy()
      self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
      self.background.pack()
      self.background.pack_propagate(0)
      THEGRAPH = graphManager(self.background, self.data, self.res, [self, self.subplots])
    
  # [한국어]
  # favouriteDescription: 즐겨찾기 항목 선택 시 설명 텍스트 표시.
  def favouriteDescription(self):
    self.tDescription.delete(1.0, Tk.END)
    favouriteChosen = self.cFavourites.get('active')
    for count in range(0,len(self.bookmarks)):
      if favouriteChosen == self.bookmarks[count].title:
        bookmarkNum = count
    description =  self.bookmarks[bookmarkNum].description
    
    self.tDescription.insert(1.0, description )
    
    
  # [한국어]
  # chooseFavourite: 즐겨찾기 선택 팝업창을 띄우고 bookmarks.txt 파싱.
  def chooseFavourite(self):
    self.favouriteWindow = Tk.Toplevel(bg = 'white')
    self.favouriteWindow.title("Favourites")
    above = Tk.Frame(self.favouriteWindow, bg = 'white')
    above.pack(side = Tk.TOP, padx = 100)
    topLabel = Tk.Label(above, text = "Available Options", font = ("Gills Sans MT", 20, "underline", "bold"), bg = 'white')
    topLabel.pack(pady = 20, padx = 20)
    subLabel = Tk.Label(above, text = "*Note: to display favourites that have subplots, you must first choose a subplot file", bg = 'white')
    subLabel.pack(side = Tk.TOP, padx = 20)
    subLabel1 = Tk.Label(above, text = "*Currently only displaying favourites that are composed of " + str(len(self.subplots) + 1) + ' plots', bg = 'white')
    subLabel1.pack(side = Tk.TOP, padx = 20)
    listboxFrames = Tk.Frame(self.favouriteWindow, bg = 'white')
    listboxFrames.pack(side = Tk.TOP, anchor = Tk.W)
    self.cFavourites = Tk.Listbox(listboxFrames, width = 30, height = 10)
    self.cFavourites.pack(side = Tk.LEFT, padx = 20, pady = 20)
    self.cFavourites.bind("<Double-Button-1>", self.chooseFavourites1)
    bDescription = Tk.Button(listboxFrames, text = "Description -->", command = self.favouriteDescription)
    bDescription.pack(side = Tk.LEFT)
    self.tDescription = Tk.Text(listboxFrames, bg = 'white', width = 50, height = 10)
    self.tDescription.pack(side = Tk.LEFT, padx = 20)

      
    self.bQuit = Tk.Button(self.favouriteWindow, text= "Quit Window", command = (lambda: self.favouriteWindow.destroy()))
    self.bQuit.pack(side = Tk.TOP, padx = 10, pady = 10)
    self.bookmarks = lexyaccbookmark.parseMe()
    for iter in self.bookmarks:
      self.cFavourites.insert(Tk.END, iter.title)
     
  # [한국어]
  # chooseFavourites1: 즐겨찾기 항목을 더블클릭하면 해당 설정으로 바로 플롯 생성.
  def chooseFavourites1(self, *event):
    self.favouriteChosen = self.cFavourites.get('active')
    for count in range(0,len(self.bookmarks)):
      if self.favouriteChosen == self.bookmarks[count].title:
        bookmarkNum = count
    
    
      
    self.dataChosenX = self.bookmarks[bookmarkNum].dataChosenX[0]
    self.dataChosenY = self.bookmarks[bookmarkNum].dataChosenY[0]
    self.graphChosen = self.bookmarks[bookmarkNum].graphChosen[0]
    self.dydx = int(self.bookmarks[bookmarkNum].dydx[0])
  
    self.modSubplot(len(self.subplots),len(self.bookmarks[bookmarkNum].graphChosen)-1)

    if self.subplots > 0:
      for iter in range(0, len(self.subplots)):
        if self.subplots[iter].fileChosen == '':
          self.subplots[iter].fileChosen = self.fileChosen
        self.subplots[iter].dataChosenX = self.bookmarks[bookmarkNum].dataChosenX[iter+1]
        self.subplots[iter].dataChosenY = self.bookmarks[bookmarkNum].dataChosenY[iter+1]
        self.subplots[iter].graphChosen = self.bookmarks[bookmarkNum].graphChosen[iter+1]
        self.subplots[iter].dydx = int(self.bookmarks[bookmarkNum].dydx[iter+1])
        
    self.favouriteWindow.destroy()
    self.background.destroy()
    self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
    self.background.pack()
    self.background.pack_propagate(0)
    THEGRAPH = graphManager(self.background, self.data, self.res, [self, self.subplots])
    
  # [한국어]
  # errorMsg: 오류 메시지 팝업창 표시.
  def errorMsg(self, string):
    error = Tk.Toplevel(bg = 'white')
    error.title("Error Message")
    tError = Tk.Label(error, text = "Error", font = ("Gills Sans MT", 20, "underline", "bold"), bg = "red")
    tError.pack(side = Tk.TOP, pady = 20)
    lError = Tk.Label(error, text = string, font = ("Gills Sans MT", 15, "bold"), bg = 'white')
    lError.pack(pady = 10, padx = 10)
    bError = Tk.Button(error, text = "OK", font = ("Times New Roman", 14), command = (lambda: error.destroy()))
    bError.pack(pady = 10)
  

  # [한국어]
  # helpMsg: 도움말 팝업창 표시.
  def helpMsg(self, string):
    helpTip = Tk.Toplevel(bg = 'white')
    helpTip.title("Helping Tip")
    lHelpTip = Tk.Label(helpTip, text = string, font = ("Gills Sans MT", 15, "bold"), bg = 'white')
    lHelpTip.pack(pady = 10, padx = 10)
    bHelpTip = Tk.Button(helpTip, text = "OK", font = ("Times New Roman", 14), command = (lambda: helpTip.destroy()))
    bHelpTip.pack(pady = 10)
    
        
# [한국어] 서브플롯 설정 폼 클래스. formEntry를 상속받는다.
class subplotInstance(formEntry):
  
  # [한국어]
  # 생성자: 서브플롯 탭 내 폼 UI 구성. 부모 formEntry와 유사하지만
  # ChosenVarsTextbox는 부모의 것을 공유한다.
  def __init__(self, master,data,plotInstance, subNum):
    frame = Tk.Frame(master, width = 700, height = 300, bg ='white')
    frame.pack()
    frame.pack_propagate(0)
    
    #Variable Initializations
    self.fileChosen = ''
    self.dataChosenX = 'globalCycle'
    self.dataChosenY = ''
    self.graphChosen = ''
    self.data = data
    self.var0 = Tk.IntVar()
    self.possGraphs = ['Line', 'Histogram', 'Bar Chart', 'Parallel Intensity Plot', 'Stacked Bar Chart', 'Parallel Intensity Plot (Sum)', 'Scatter Plot']
    self.dydx = 0
    self.ChosenVarsTextbox = plotInstance.ChosenVarsTextbox
    self.num = subNum
    self.subplots = plotInstance.subplots
    
    
    # Frame and listboxes for choosing what file you want to use data from
    whichFileFrame = Tk.Frame(frame, bg= 'white')
    whichFileFrame.pack(side = Tk.TOP, anchor = Tk.W, pady = 5)
    lwhichFileFrame = Tk.Label(whichFileFrame, text = 'Choose a File:', font = ("Gills Sans MT", 12), bg = "white")
    lwhichFileFrame.pack(side = Tk.LEFT)
    self.cWhichFile= Tk.Listbox(whichFileFrame, width = 85, height = 4)
    self.cWhichFile.pack(side =Tk.LEFT)
    self.cWhichFile.bind("<Double-Button-1>", self.chooseFile)
    
    #Placing the available filenames in the self.cWhichFileFrame Listbox
    for files in sorted(self.data):
      self.cWhichFile.insert(Tk.END, files)
    
    # move to end of lines to actually see the filenames
    self.cWhichFile.xview_moveto(1)

    #Frame for choosing what data you are going to plot
    chooseDataFrame = Tk.Frame(frame, bg = "white")
    chooseDataFrame.pack(side = Tk.TOP, anchor = Tk.W)
    lchooseDataFrame = Tk.Label(chooseDataFrame, text= 'Data to be Plotted:', font = ("Gills Sans MT", 12), bg = "white")
    lchooseDataFrame.pack(side= Tk.LEFT);
    bQchooseDataFrameHelp = Tk.Button(chooseDataFrame, text = " ? -->", command = (lambda: self.helpMsg('Double-Click on the metric to select it for plotting.')))
    bQchooseDataFrameHelp.pack(side =Tk.LEFT, padx = 5)
    XListbox = Tk.Frame(chooseDataFrame, bg= 'white')
    XListbox.pack(side = Tk.LEFT)
    XTitle = Tk.Label(XListbox, text = "X Vars", bg= 'white')
    XTitle.pack(side = Tk.TOP)
    self.cXAxisData = Tk.Listbox(XListbox, width = 19, height = 5, selectmode = Tk.MULTIPLE)
    self.cXAxisData.bind("<Double-Button-1>", self.chooseDataX)
    YListbox = Tk.Frame(chooseDataFrame, bg = 'white')
    YListbox.pack(side = Tk.LEFT)
    YFrameForScrollbar = Tk.Frame(YListbox, bg= 'white')
    YFrameForScrollbar.pack(side = Tk.BOTTOM)
    scrollYAxisData = Tk.Scrollbar(YFrameForScrollbar, orient = Tk.VERTICAL)
    YTitle = Tk.Label(YListbox, text = 'Y Vars', bg= 'white')
    YTitle.pack(side = Tk.TOP)
    self.cYAxisData = Tk.Listbox(YFrameForScrollbar,width = 19, height = 5, yscrollcommand=scrollYAxisData.set)
    self.cYAxisData.bind("<Double-Button-1>", self.chooseDataY)
    scrollYAxisData.config(command=self.cYAxisData.yview)
    self.cXAxisData.pack(side = Tk.BOTTOM)
    self.cYAxisData.pack(side = Tk.LEFT)
    scrollYAxisData.pack(side=Tk.LEFT, fill = 'y')
    
    # The Take Derivative Checkbutton
    self.var0 = Tk.IntVar()
    checkbDyDx = Tk.Checkbutton(chooseDataFrame, text= "dy/dx", variable= self.var0, bg = 'white', command = (lambda: self.checkDyDx()))
    checkbDyDx.pack(side= Tk.LEFT, padx = 5)
    
    #Frame for choosing what type of graph
    typeGraph = Tk.Frame(frame, bg = "white")
    typeGraph.pack(side = Tk.TOP, anchor = Tk.W, pady = 10)
    lTypeGraph = Tk.Label(typeGraph, text= 'Type of Graph:', font = ("Gills Sans MT", 12), bg = "white")
    lTypeGraph.pack(side= Tk.LEFT);
    bQTypeGraphHelp = Tk.Button(typeGraph, text = " ? -->", command = (lambda: self.helpMsg('Double-Click on the graph type to select it.')))
    bQTypeGraphHelp.pack(side =Tk.LEFT, padx = 5)
    self.cTypeGraph = Tk.Listbox(typeGraph, width = 50, height = 3)
    self.cTypeGraph.bind("<Double-Button-1>", self.chooseGraph)
    self.cTypeGraph.pack(side = Tk.LEFT)
    
    
# Class holding all the format information of a plot
# [한국어] 플롯 포맷 정보 클래스. 제목, 축 레이블, 폰트 크기, 색상맵 등을 관리.
class PlotFormatInfo:

    # names of available colormaps
    cmapOptions = ['gist_heat_r', 'gist_heat', 'hot_r', 'hot', 'gray', 'gray_r', 'spectral', 'bone', 'bone_r', 'GnBu', 'GnBu_r', 'gist_earth', 'gist_earth_r', 'zero_dark', 'zero_white']

    # custom color maps
    zero_dark = {'red':  ((0.0, 0.0, 0.0), (0.0001, 1.0, 1.0), (1.0, 0.0, 1.0)),
                 'green':((0.0, 0.0, 0.0), (0.0001, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 1.0)),
                 'blue': ((0.0, 0.0, 0.0), (0.0001, 0.0, 0.0), (0.25, 0.0, 0.0), (1.0, 1.0, 1.0))}
    zero_dark_cmap = mpl.colors.LinearSegmentedColormap('zero_dark', zero_dark, 1024)

    zero_white = {'red':  ((0.0, 1.0, 1.0), (0.0001, 1.0, 1.0), (1.0, 0.0, 1.0)),
                 'green':((0.0, 1.0, 1.0), (0.0001, 1.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 1.0)),
                 'blue': ((0.0, 1.0, 1.0), (0.0001, 1.0, 0.0), (0.25, 0.0, 0.0), (1.0, 1.0, 1.0))}
    zero_white_cmap = mpl.colors.LinearSegmentedColormap('zero_white', zero_white, 1024)

    custom_cmaps = {'zero_dark': zero_dark_cmap, 'zero_white': zero_white_cmap }

    strNoDisplay = 'NULL:0x0000'

    optionSection = 'TimeLapseView'

    # [한국어]
    # 생성자: 플롯 포맷 초기값 설정. avconfig에서 폰트/눈금 회전 옵션을 읽어옴.
    def __init__(self, plotID,
                 title = strNoDisplay,
                 xlabel = strNoDisplay, 
                 ylabel = strNoDisplay, 
                 cbarlabel = strNoDisplay, 
                 rotation = avconfig.get_value(optionSection, 'xTicksRotation', 'horizontal'),
                 labelFontSize = avconfig.get_value(optionSection, 'labelFontSize', 13), 
                 cticksFontSize = avconfig.get_value(optionSection, 'cTicksFontSize', 10), 
                 xticksFontSize = avconfig.get_value(optionSection, 'xTicksFontSize', 10), 
                 yticksFontSize = avconfig.get_value(optionSection, 'yTicksFontSize', 10)):
       
        self.plotID = plotID   # for debugging purpose

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cbarlabel = cbarlabel
        self.rotation = rotation
        self.labelFontSize = labelFontSize
        self.cticksFontSize = cticksFontSize
        self.xticksFontSize = xticksFontSize
        self.yticksFontSize = yticksFontSize

        self.cmap = Tk.StringVar()
        self.cmap.set(PlotFormatInfo.cmapOptions[0])

        self.norm = mpl.colors.Normalize()

    # [한국어]
    # InitLabels: 값이 strNoDisplay인 레이블에만 실제 값을 채워넣는다.
    def InitLabels(self, xlabel, ylabel, cbarlabel, title = ''):
        
        if (self.xlabel == PlotFormatInfo.strNoDisplay):
            self.xlabel = xlabel

        if (self.ylabel == PlotFormatInfo.strNoDisplay):
            self.ylabel = ylabel

        if (self.cbarlabel == PlotFormatInfo.strNoDisplay):
            self.cbarlabel = cbarlabel

        if (self.title == PlotFormatInfo.strNoDisplay):
            self.title = title

    # Obtain the matplotlib colormap that will be used for the parallel intensity plot
    # [한국어]
    # GetColorMap: 선택된 색상맵 이름에 해당하는 matplotlib colormap 객체 반환.
    def GetColorMap(self):
        cmapName = self.cmap.get()
        if (cmapName in PlotFormatInfo.custom_cmaps):
            cmap = PlotFormatInfo.custom_cmaps[cmapName]
        else:
            cmap = mpl.cm.get_cmap(name=cmapName) 
        return cmap
        
# [한국어] matplotlib Figure 관리 클래스. 실제 플롯 생성과 인터랙션 버튼을 담당.
class graphManager:
  
    # [한국어]
    # 생성자: matplotlib Figure, Canvas, Toolbar, 인터랙션 버튼을 구성하고 plotData() 호출.
    # @master: 부모 Tk Frame
    # @data: {파일명: {변수명: variable}} 데이터
    # @res: 해상도
    # @dataChosen: [주 formEntry 객체, 서브플롯 리스트]
    def __init__(self, master, data, res, dataChosen):
    
        self.normalizePlotColors = ''
        self.master = master
        #Variable initializations
        self.cycleStep = 0
        self.wilLeft = ''
        self.res =res
        self.dataChosen = dataChosen
        self.data = data
        self.disconnect = 0
        self.possGraphs = ['Line', 'Histogram', 'Bar Chart', 'Parallel Intensity Plot', 'Stacked Bar Chart', 'Parallel Intensity Plot (Sum)', 'Scatter Plot']
        self.xlim = 0
        self.xAxisStepsWilStack = {}
        self.cbarAxes = {}
        self.plotRef = {}
        self.plotFormatInfo = {}
        self.displayData = {}

        # [한국어] 해상도별 그래프 하단 컨트롤 프레임 크기 선택
        if self.res == "small":
            self.underneathGraph = Tk.Frame(master, borderwidth = 5, relief = Tk.GROOVE, height = 100, width = 1225);
        elif self.res == 'medium':
            self.underneathGraph = Tk.Frame(master, borderwidth = 5, relief = Tk.GROOVE, height = 100, width = 1572);
        else:
            self.underneathGraph = Tk.Frame(master, borderwidth = 5, relief = Tk.GROOVE, height = 100, width = 1572);
        self.underneathGraph.pack(side = Tk.BOTTOM, fill = Tk.X )
        
        # [한숙어] 해상도별 그래프 캔버스 크기 선택
        if self.res == "small":
            self.graphArea = Tk.Canvas(master, bg = "black", borderwidth = 5, relief = Tk.GROOVE, width = 1200);
        elif self.res == 'medium':
            self.graphArea = Tk.Canvas(master, bg = "black", borderwidth = 5, relief = Tk.GROOVE, width = 1513, height = 800);
        else:
            self.graphArea = Tk.Canvas(master, bg = "black", borderwidth = 5, relief = Tk.GROOVE, width = 1513, height = 800);
        self.graphArea.pack(side = Tk.BOTTOM, fill = Tk.BOTH, expand = 1)
        
        ## Allow user to choose specific graph options
        
        self.leftMostUnderneath = Tk.Frame(self.underneathGraph);
        self.leftMostUnderneath.pack(side = Tk.LEFT, anchor = Tk.N)
        self.toolbarArea = Tk.Frame(self.leftMostUnderneath, borderwidth = 1, relief = Tk.GROOVE, bg = 'black')
        self.toolbarArea.pack(anchor = Tk.N)
        # [한국어] 인터랙션 버튼들
        bDyDx = Tk.Button(self.underneathGraph, text = 'dy/dx', command = self.takeDerivativeButton)
        bDyDx.pack(side = Tk.RIGHT, anchor = Tk.N)
        bAddToFavourites = Tk.Button(self.underneathGraph, text = "Add to Favourites", command = self.addToFavourites)
        bAddToFavourites.pack(side = Tk.RIGHT, anchor = Tk.N)
        bRefreshInputFiles = Tk.Button(self.underneathGraph, text= "Refresh Input Files", command = self.refreshInputs)
        bRefreshInputFiles.pack(side = Tk.RIGHT, anchor = Tk.N)
        bChangeColorMapMaxMin = Tk.Button(self.underneathGraph, text = "Change Colormap Max/Min", command = self.changeColorMapMaxMin)
        bChangeColorMapMaxMin.pack(side = Tk.RIGHT, anchor = Tk.N)
        bChangeBinning = Tk.Button(self.underneathGraph, text = 'Change Binning', command = self.changeBinning)
        bChangeBinning.pack(side = Tk.RIGHT, anchor = Tk.N)
        bEditLabels = Tk.Button(self.underneathGraph, text = 'Edit Labels', command = self.editLabelsButton)
        bEditLabels.pack(side = Tk.RIGHT, anchor = Tk.N)
        bZoom = Tk.Button(self.underneathGraph, text = 'Zoom', command = self.zoomButton)
        bZoom.pack(side = Tk.RIGHT, anchor = Tk.N)

        # [한국어] 해상도별 Figure 크기
        if self.res == "small":
          self.figure = Figure(figsize=(17,9), dpi=96)
        elif self.res == 'medium':
          self.figure = Figure(figsize=(20,9), dpi=96)  
        else:
          self.figure = Figure(figsize=(22,13), dpi=96)
        
        #self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graphArea)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarArea)
        self.toolbar.update()
        self.plotData()
        
    # [한국어]
    # addToFavourites: 현재 플롯 구성을 bookmarks.txt에 추가하기 위한 팝업창 생성.
    def addToFavourites(self):
      self.addFavourite = Tk.Toplevel(bg = 'white')
      self.addFavourite.title("Add Favourite")
      fTitle = Tk.Frame(self.addFavourite, bg = 'white')
      fTitle.pack(side = Tk.TOP, padx= 100, pady = 10)
      lTitle = Tk.Label(fTitle, text = "Title of new favourite:  ", bg = 'white')
      lTitle.pack(side = Tk.LEFT)
      self.eTitle = Tk.Entry(fTitle, width = 50, bg = 'white')
      self.eTitle.pack(side = Tk.LEFT, padx = 3)
      fDescription = Tk.Frame(self.addFavourite, bg = 'white')
      fDescription.pack(side = Tk.TOP, padx= 100, pady = 10)
      lDescription = Tk.Label(fDescription, text = "Description of favourite: ", bg = 'white')
      lDescription.pack(side = Tk.LEFT)
      self.eDescription = Tk.Text(fDescription, width = 50, height = 10, bg = 'white')
      self.eDescription.pack(side = Tk.LEFT)
      fSubCanc = Tk.Frame(self.addFavourite, bg = 'white')
      fSubCanc.pack(side = Tk.BOTTOM)
      bSubmit = Tk.Button(fSubCanc, text = "Submit", command = self.addFavouriteTitDesc)
      bSubmit.pack(side = Tk.LEFT)
      bCancel = Tk.Button(fSubCanc, text = "Cancel", command = (lambda: self.addFavourite.destroy()))
        
    # [한국어]
    # addFavouriteTitDesc: 제목/설명을 입력받아 bookmarks.txt에 기록.
    def addFavouriteTitDesc(self):
      bool = 1
      self.favouriteTitle = self.eTitle.get()
      self.favouriteDesc = self.eDescription.get(1.0,Tk.END)
      self.favouriteDesc = str(self.favouriteDesc)
      test = []
      test.append(self.favouriteDesc)
      self.addFavourite.destroy()
      self.favouriteDesc = self.favouriteDesc.rstrip()
      test.append(self.favouriteDesc)
      
      if self.favouriteTitle == "":
          self.errorMsg("You need to at least submit a title, try again...")
          bool = 0
      
      numPlots = len(self.dataChosen[1]) + 1
      self.dataPointer = self.dataChosen[0]
      
      if bool:
        # [한국어] bookmarks.txt에 START 블록과 함께 현재 플롯 구성 기록
        file = open(os.environ['HOME'] + "/.gpgpu_sim/aerialvision/bookmarks.txt", 'a')
        file.write('START = "TRUE"\n')
        file.write('title = "' + self.favouriteTitle + '"\n')
        file.write('description = "' + self.favouriteDesc + '"\n')
        for self.currPlot in range(1,numPlots + 1):
          file.write('dataChosenX = "' + self.dataPointer.dataChosenX + '"\n')
          file.write('dataChosenY = "' + self.dataPointer.dataChosenY + '"\n')
          file.write('graphChosen = "' + self.dataPointer.graphChosen + '"\n')
          file.write('dydx = "' +  str(self.dataPointer.dydx) + '"\n')
          
          if self.dataChosen[1] != [] and self.currPlot != numPlots:
            self.dataPointer = self.dataChosen[1][self.currPlot - 1]

        file.close()
   
   
    # [한국어]
    # format_coordWilson: Parallel Intensity Plot 마우스 오버 시 좌표 문자열 생성.
    def format_coordWilson(self, x, y):
        col = int(x)*(self.simplerName[self.dataPointer.dataChosenX].data[1])
        row = int(y+0.5)

        numrows = len(self.simplerName[self.dataPointer.dataChosenY].data)
        try:
          numcols = len(self.simplerName[self.dataPointer.dataChosenY].data[0])
        except:
          numcols = 1

        if x>=0 and x<numcols and y>=0 and y<numrows:
            return 'x=%d, y=%d'%(col, row)
        else:
            return 'x=%d, y=%d'%(col, row)


    # [한국어]
    # plotData: 전체 플롯을 생성하는 메인 함수.
    # 모든 서브플롯을 순회하며 xlim을 결정하고, 변수 타입(1~5)에 따라 플로팅 함수 분기.
    def plotData(self):
        
        #Variable initializations
        self.currPlot = 1
        numPlots = len(self.dataChosen[1]) + 1
        self.xAxisStepsWilStack = ['null']
        self.yAxisStepsWilStack = ['null']
        #need to scale all plots to the same length
        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        self.colorbars = {}
        
        # [한국어] 1단계: 모든 플롯의 xlim과 cycleStep을 계산
        for self.currPlot in range(1,numPlots + 1):
          self.xAxisStepsWilStack.append(8)
          self.yAxisStepsWilStack.append('null')
          self.findKernalLocs()
          tmp = self.updateVarKernal(self.simplerName[self.dataPointer.dataChosenX].data)
          if tmp[-1] > self.xlim:
            self.xlim = tmp[-1]
            self.cycleStep = tmp[1]
          else:
            pass

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
            self.dataPointer = self.dataChosen[1][self.currPlot - 1]
            self.simplerName = self.data[self.dataPointer.fileChosen]
       
          # initialize format info of this plot
          if self.currPlot not in self.plotFormatInfo:
            self.plotFormatInfo[self.currPlot] = PlotFormatInfo(self.currPlot)
        
        
        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        
        # [한국어] 2단계: 각 플롯을 실제로 그림
        for self.currPlot in range(1,numPlots + 1):
          self.findKernalLocs()
          if (self.currPlot in self.plotRef):
              self.figure.delaxes(self.plotRef[self.currPlot])
          self.plot = self.figure.add_subplot(numPlots,1,self.currPlot)
          self.plotRef[self.currPlot] = self.plot 
          self.plotFormatInfo[self.currPlot].graphType = self.dataPointer.graphChosen
          if self.dataPointer.graphChosen == 'Parallel Intensity Plot':
              self.plot.format_coord = self.format_coordWilson
          # [한국어] 변수 타입에 따라 플로팅 함수 분기
          yDataType = self.simplerName[self.dataPointer.dataChosenY].type
          if yDataType == 1:
              self.type1Variable(self.simplerName[self.dataPointer.dataChosenX].data , self.dataPointer.dataChosenX, self.simplerName[self.dataPointer.dataChosenY].data, self.dataPointer.dataChosenY, self.simplerName[self.dataPointer.dataChosenY].bool, self.currPlot)
          elif yDataType == 2:
              self.type2Variable(self.simplerName[self.dataPointer.dataChosenX].data,self.dataPointer.dataChosenX, self.simplerName[self.dataPointer.dataChosenY].data,self.dataPointer.dataChosenY, self.currPlot)
          elif yDataType == 3:
              self.type3Variable(self.simplerName[self.dataPointer.dataChosenX].data,self.dataPointer.dataChosenX,self.simplerName[self.dataPointer.dataChosenY].data,self.dataPointer.dataChosenY, self.currPlot)
          elif yDataType == 4:
              self.type4Variable(self.simplerName[self.dataPointer.dataChosenX].data,self.dataPointer.dataChosenX,self.simplerName[self.dataPointer.dataChosenY].data,self.dataPointer.dataChosenY, self.currPlot)
          elif yDataType == 5:
              self.type5Variable(self.simplerName[self.dataPointer.dataChosenX].data,self.dataPointer.dataChosenX,self.simplerName[self.dataPointer.dataChosenY].data,self.dataPointer.dataChosenY, self.currPlot)
          else:
              raise Exception("Unknown data type!\n")

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
    

    # [한국어]
    # type1Variable: scalar(1차원 시계열) 데이터를 Line 또는 Parallel Intensity Plot으로 그림.
    def type1Variable(self, x, xAxis, y, yAxis, boolK, plotID):
    
        graphOption = 'NULL'
        
        
        
        if ('globalTotInsn' in self.simplerName) == 'False':
            graphOption = 1
            
        if (graphOption == 1):  
            if (self.dataPointer.dydx >= 1):
                for iter in range(0, self.dataPointer.dydx):
                    y = self.takeDerivative(x, y)
    
            yAxis = yAxis + '/Cycle'
                
            if (self.graphChosen == self.possGraphs[3]):
                self.plotParallelIntensity(x, xAxis, [y], yAxis, yAxis, [1], plotID)
            else:
                self.plot2VarLine(x, xAxis, y, yAxis)
    
            
        else:     
          x = self.updateVarKernal(x)
          
          if boolK:
              y = self.updateVarKernal(y)
    
          if (self.dataPointer.dydx >= 1):
              for iter in range(0, self.dataPointer.dydx):
                  y = self.takeDerivative(x, y)
              yAxis = yAxis + '/Cycle'
    
               
          if (self.dataPointer.graphChosen == self.possGraphs[3]):
              self.plotParallelIntensity(x, xAxis, [y], yAxis, yAxis, [1], plotID)
          else:           
              #Label and plot Line Graph
              self.labelKernals(x,y)
              self.plot2VarLine(x, xAxis, y, yAxis)
              
  
      
    # [한국어]
    # type2Variable: vector(2차원, SM/뱅크별) 데이터를 Line 또는 Parallel Intensity Plot으로 그림.
    def type2Variable(self, x, xAxis, y, yAxis, plotID):

        graphOption = "NULL"
            
        if ('globalTotInsn' in self.simplerName) == 'False':
            graphOption = 1
    
        if (graphOption == 1):
            
          if (self.dataPointer.dydx >= 1):
              for iter in range(0, self.dataPointer.dydx):
                  y = self.takeDerivativeMult(x,y)
    
          yAxis = yAxis + '/Cycle' 
    
          if (self.dataPointer.graphChosen == self.possGraphs[3]):
              yTicks = [n for n in range(len(y))]
              self.plotParallelIntensity(x, xAxis, y, yAxis, yAxis, yTicks, plotID)
          else:
              self.plotMultVarLine(x, xAxis, y, yAxis)
    
        
        else:
    
          x = self.updateVarKernal(x)
         
          if (self.dataPointer.dydx >= 1):
              for iter in range(0, self.dataPointer.dydx):
                  y = self.takeDerivativeMult(x,y)
    
          yAxis = yAxis + '/Cycle'
    
    
          if (self.dataPointer.graphChosen == self.possGraphs[3]):
              yTicks = [n for n in range(len(y))]
              self.plotParallelIntensity(x, xAxis, y, yAxis, yAxis, yTicks, plotID)
          
          else:           
              #Label and Plot
              self.labelKernalsMult(x, y)
              self.plotMultVarLine(x,xAxis, y,yAxis)




    # [한국어]
    # type3Variable: stackedbar(3차원) 데이터를 Stacked Bar Plot으로 그림.
    def type3Variable(self, x, xAxis, y, yAxis, plotID):
        #Type 3 variables are currently those that are used for STACKED BAR PLOTS
    
        #if there are kernals.. we need to adjust the x axis for proper labelling
        #Need to make changes here.. works for now though
        if 'globalTotInsn' in self.simplerName:
            x = self.updateVarKernal(x)

        concentrationFactor = len(x) // 512 + 1
        
        #Scalar Vars
        numCols = len(y[0]) #the number of columns in the stacked bar plot
        width = 1.0 #Our bars will occupy 100% of the space allocated to them
        numRows = len(y) #The number of stacks
        colours = mpl.cm.get_cmap('RdBu', numRows) #discretizing a matplotlib color scheme to serve as the various colors of our stacked bar plot
        
    
        #Labelling the xAxis with the name of the variable and also the file that the data was chosen from
        self.plot.set_xlabel(xAxis)
        
        #Non-Scalar Vars
        ind = [tmp for tmp in range(0,numCols)] #the location of the bar and the x axis labels        
        yoff = numpy.array([0.0] * numCols) #variable use to remember the last top location of a bar so that we may stack the proceeding bar on top of it
        # [한국어] 기본 범례 이름. 일부 하드코딩된 메모리 파이프라인 단계명.
        Legendname = ['N/A', 'N/A','N/A','IcntInpBuf','N/A','Icnt2DRAM','N/A','N/A','N/A','DRAM','2Sh_IcntInpBuf','N/A','Icnt2shd','N/A','N/A']; 
        BarSequence = list(range(numRows-1,-1,-1))

        # [한국어] 변수별로 맞춤형 범례 이름과 쌓기 순서 설정
        if yAxis == 'WarpDivergenceBreakdown':
            Legendname = []
            Legendname.append('Idle')
            Legendname.append('Data Hazard')
            Legendname.append('Stall')
            for c in range(2, numRows):
                Legendname.append('W' + repr(4*(c-2)+1) +  ':' + repr(4*(c-1)))
            BarSequence = list(range(0,numRows))

        if yAxis == 'WarpIssueSlotBreakdown':
            Legendname = []
            for c in range(0, numRows):
                Legendname.append('W' + repr(c))
            BarSequence = list(range(0,numRows))

        dynamic_warp_resolution = 32
        if yAxis == 'WarpIssueDynamicIdBreakdown':
            Legendname = []
            for c in range(0, numRows):
                Legendname.append('W' + repr(dynamic_warp_resolution*c) + ":" + repr(dynamic_warp_resolution*(c+1)))
            BarSequence = list(range(0,numRows))

        yoff_max = numpy.array([0.0] * numCols)
        for row in range(numRows-1,-1,-1):
            yoff_max += y[row]
    
        # [한국어] 누적 막대 그래프 그리기
        for row in BarSequence:
            row1 = float(row) #Used to select a new color from the colormap.. need to be a float thats why a new variable was made
            yoff = yoff + y[row] #updating the yoff variable
            self.plot.bar(ind, y[row], width, bottom = yoff_max-yoff, color=colours(row1/numRows), edgecolor=colours(row1/numRows), label = Legendname[row]) #plotting each set of bar plots individually
        
        
        plotFormat = self.plotFormatInfo[plotID]
        plotFormat.InitLabels(xlabel = xAxis, ylabel = yAxis, cbarlabel = '', 
                              title = yAxis + ' vs ' + xAxis + ' ...' + self.dataPointer.fileChosen[-80:])
        self.plot.set_title(plotFormat.title)

        # More Labelling 
        self.plot.set_xlabel(plotFormat.xlabel, fontsize = plotFormat.labelFontSize)
        self.plot.set_ylabel(plotFormat.ylabel, fontsize = plotFormat.labelFontSize)
       
        matplotlib.rcParams['legend.fontsize'] = plotFormat.yticksFontSize
        self.plot.legend(loc=(1.01,0.1))
            
        labelValues = []
        labelPos = []
        try:
            for count in range(0,len(x)/concentrationFactor,len(x)/8/concentrationFactor):
                labelValues.append(x[count * concentrationFactor])
                labelPos.append(ind[count])
        except:
            pass

        
        xlim = self.type3findxlim(ind[-1], x[1])
        self.plot.set_xlim(0,xlim / concentrationFactor)
        
        self.plot.set_xticklabels(labelValues, rotation = plotFormat.rotation, fontsize = plotFormat.xticksFontSize)
        self.plot.set_xticks(labelPos)  
        for label in self.plot.get_yticklabels():
            label.set_fontsize(plotFormat.yticksFontSize)
     
        self.canvas.draw()
        
    # [한국어]
    # type4Variable: vector2d(4차원, chip.bank별) 데이터를 Parallel Intensity Plot으로 그림.
    def type4Variable(self, x, xAxis, y, yAxis, plotID):
        keys = list(y.keys())
        keys.sort()
            
        if (self.dataPointer.graphChosen == self.possGraphs[3]):
            image = []
            for iter in keys:
                image.append(y[iter])
              
            if self.dataPointer.dydx >= 1:
                for iter in range(0, self.dataPointer.dydx):
                    image = self.takeDerivativeMult(x, image)

        elif (self.dataPointer.graphChosen == self.possGraphs[5]):
            # collect a set of majorkeys (assuming format m.n in key)
            majorKeys = set()
            dataLength = 0
            for k in keys:
                majork, minork = k.split('.')
                majorKeys.add(majork)
                dataLength = len(y[k])
            majorKeys = list(majorKeys)
            majorKeys.sort()
           
            # define a (dataLength x #MajorKeys) array
            image = [[0 for t in range(dataLength)] for s in range(len(majorKeys))]
            for k in keys:
                majork, minork = k.split('.')
                imageArray = image[int(majork)]
                dataArray = y[k]
                for pt in range(dataLength):
                    imageArray[pt] += dataArray[pt]
              
            if self.dataPointer.dydx >= 1:
                for iter in range(0, self.dataPointer.dydx):
                    image = self.takeDerivativeMult(x, image)

            keys = majorKeys
        else: 
            pass

        self.plotParallelIntensity(x, xAxis, image, yAxis, yAxis, keys, plotID)
      

    # [한국어]
    # type5Variable: sparse(5차원) 데이터를 Scatter Plot으로 그림.
    def type5Variable(self, x, xAxis, y, yAxis, plotID):
            
        assert (self.dataPointer.graphChosen == self.possGraphs[6])

        xScatter = y[2][0:] * (x[1] - x[0])
        self.plotScatter(xScatter, xAxis, y[1], yAxis, plotID)

        del xScatter
      

    # [한국어]
    # updateVarKernal: 커널 경계에서 globalCycle이 리셋되는 경우,
    # 누적값을 이전 커널의 마지막 값에 더해 연속적인 X축을 만든다.
    def updateVarKernal(self,var):
          var = [val for val in var]
          if self.disconnect == 0:
              for cycleNum in range(0,len(self.simplerName[self.dataPointer.dataChosenX].data)):
                  if (self.xIsKernal(cycleNum,self.kernalLocs)):
                      for cycleSum in range(self.kernalLocs[self.kernalLocs.index(cycleNum) + 1] - 1, cycleNum - 1, -1):
                          if cycleNum == 0:
                              continue
                          var[cycleSum] += var[cycleNum - 1]
              return var
          else:
              return var
          

    # [한국어]
    # xIsKernal: 주어진 사이클 번호가 커널 시작 위치인지 확인.
    def xIsKernal(self,x,kernalStarts):
        bool = 0
        for y in kernalStarts:
            if (y == x):
                bool = 1
        return bool
    
    # [한국어]
    # findKernalLocs: globalCycle 데이터에서 커널 경계(감소 지점)를 찾아 self.kernalLocs에 저장.
    def findKernalLocs(self):
        
        self.kernalLocs = []
        prevCycle = -1
        countIter = 0
        for cycle in self.simplerName[self.dataPointer.dataChosenX].data:
            if (prevCycle >= cycle):
                self.kernalLocs.append(countIter)
            prevCycle = cycle
            countIter += 1
        self.kernalLocs.append(len(self.simplerName[self.dataPointer.dataChosenX].data))
    
    # [한국어]
    # labelKernalsMult: 다중 변수 라인 그래프에서 커널 경계에 수직 점선과 라벨 추가.
    def labelKernalsMult(self,x, y):
        countKernal = 0
        label = ""
        sum = []
        maximum = 0
        sumAve = []
        
        for values in range(0,len(y[0])):
            sum.append(0)
            for chip in range(0,len(y)):
                sum[values] += y[chip][values]
                sumAve.append(sum[values]/len(y))
                
        for vectors in range(0,len(y)):
            if max(y[vectors]) > max(y[maximum]):
                maximum = vectors
    
        for a in self.kernalLocs:
            a = a - 1
            countKernal += 1
            label = " EndKern" + str(countKernal)
            self.plot.text(x[a],sum[a]/len(y), label, fontsize = 10)
            tmpx = [x[a], x[a]]
            tmpy = [0, max(y[maximum])]
            self.plot.plot(tmpx, tmpy, 'k:' )
        
        
    # [한국어]
    # labelKernals: 단일 변수 라인 그래프에서 커널 경계에 수직 점선과 라벨 추가.
    def labelKernals(self,x, y):
        countKernal = 0
        label = ""
    
        for cycleNum in range(0,len(x)):
            if(self.xIsKernal(cycleNum,self.kernalLocs)):
                countKernal += 1
                label = " EndKern" + str(countKernal)
                kernelBound = self.kernalLocs[self.kernalLocs.index(cycleNum)] - 1
                self.plot.text(x[kernelBound], y[kernelBound], label, fontsize = 10)

                kernelBoundx = x[kernelBound]
                
                tmpx = [kernelBoundx, kernelBoundx]
                tmpy = [0, max(y)]
                self.plot.plot(tmpx, tmpy, 'k:' )
        countKernal += 1
        label = " EndKern" + str(countKernal)
        self.plot.text(x[-1], y[-1], label, fontsize = 10)

        kernelBoundx = x[-1]
        tmpx = [kernelBoundx, kernelBoundx]
        tmpy = [0, max(y)]
        self.plot.plot(tmpx, tmpy, 'k:' )
    
    
    # [한국어]
    # plot2VarLine: 단일 변수 Line 그래프 그리기.
    def plot2VarLine(self, x, xAxis, y, yAxis):
      self.plot.plot(x, y)
      self.plot.set_xlim(0, self.xlim)
      self.plotFormatInfo[self.currPlot].InitLabels(xlabel = xAxis, ylabel = yAxis, cbarlabel = '', title = xAxis + ' vs ' + yAxis + ' ...' + self.dataPointer.fileChosen[-80:])
      self.plot.set_title(self.plotFormatInfo[self.currPlot].title)
      self.plot.set_xlabel(self.plotFormatInfo[self.currPlot].xlabel, fontsize = self.plotFormatInfo[self.currPlot].labelFontSize)
      self.plot.set_ylabel(self.plotFormatInfo[self.currPlot].ylabel, fontsize = self.plotFormatInfo[self.currPlot].labelFontSize)
      self.canvas.draw()
    
    
    # [한국어]
    # plotMultVarLine: 다중 변수 Line 그래프 그리기.
    def plotMultVarLine(self, x, xAxis, y, yAxis):
      for num in range(0,len(y)):
          self.plot.plot(x, y[num])
      self.plot.set_xlim(0, self.xlim)
      self.plotFormatInfo[self.currPlot].InitLabels(xlabel = xAxis, ylabel = yAxis, cbarlabel = '', title = '')
      self.plot.set_xlabel(self.plotFormatInfo[self.currPlot].xlabel, fontsize = self.plotFormatInfo[self.currPlot].labelFontSize)
      self.plot.set_ylabel(self.plotFormatInfo[self.currPlot].ylabel, fontsize = self.plotFormatInfo[self.currPlot].labelFontSize)
      self.canvas.draw()


    # [한국어]
    # plotScatter: Scatter Plot 그리기.
    def plotScatter(self, x, xAxis, y, yAxis, plotID):
        plotFormat = self.plotFormatInfo[self.currPlot]
        self.plot.scatter(x, y, s=5, marker='s', edgecolors='none')
        self.plot.set_xlim(0, self.xlim)
        ymax = max(y)
        ymin = min(y)
        self.plot.set_ylim(ymin, ymax)
        plotFormat.InitLabels(xlabel = xAxis, ylabel = yAxis, cbarlabel = '', title = xAxis + ' vs ' + yAxis + ' ...' + self.dataPointer.fileChosen[-80:])
        self.plot.set_title(plotFormat.title, fontsize = plotFormat.labelFontSize)
        self.plot.set_xlabel(plotFormat.xlabel, fontsize = plotFormat.labelFontSize)
        self.plot.set_ylabel(plotFormat.ylabel, fontsize = plotFormat.labelFontSize)
        self.canvas.draw()
      
    
    # [한국어]
    # takeDerivativeMult: 2D 데이터의 각 행에 대해 dy/dx 계산.
    def takeDerivativeMult(self,x,y):
        multDerivative = []

        cycleStep = self.simplerName[self.dataPointer.dataChosenX].data[1] - self.simplerName[self.dataPointer.dataChosenX].data[0]
       
        
        for num in range(0,len(y)):
            multDerivative.append(self.takeDerivative(x,y[num]))
        return multDerivative                
    
    
    # [한국어]
    # takeDerivative: 1D 시계열 y에 대해 x 간격으로 차분(derivative) 계산.
    # @x: X축 데이터 (globalCycle)
    # @y: Y축 데이터
    # @b_acc: 누적값이 감소하면 prevY 리셋 (커널 리셋 대응)
    def takeDerivative(self,x,y,b_acc=1): #both variables have to already be organized for this to work!!!
        derivative = array.array('f')
        prevY = 0
        
        cycleStep = x[1] - x[0]
        if (cycleStep < 0):
            prevCycle = 0
            for cycle in self.simplerName[self.dataPointer.dataChosenX].data:
                cycleStep = cycle - prevCycle
                if cycleStep == 0:
                    continue
                else:
                    break
                prevCycle = cycle
        # fill up self.globalIPC list
        for yNum in y:
            if (b_acc == 1 and yNum < prevY):
                prevY = 0	
            derivative.append(float(yNum - prevY)/cycleStep)
            prevY = yNum
        return derivative
    
    
    # [한국어]
    # plotParallelIntensity: Parallel Intensity Plot(heatmap) 그리기.
    def plotParallelIntensity(self, x, xAxis, y, yAxis, colorAxis, yTicks, plotID):
        # Obtain plot format info
        plotFormat = self.plotFormatInfo[plotID]

        # Obtain the matplotlib colormap that we will use for the parallel intensity plot
        cmap = plotFormat.GetColorMap()
        
        #The yAxis/xAxis Labels and their corresponding positions
        yticks, yticksPos =  self.updateWilTicks(y)
        #currently not using the variable xticks. we want to use values from the x input parameter and place them at 'xticksPos'
        xticks, xticksPos = self.updateWilTicks(x) 

        #Limits the number of xAxis labels to 8 otherwise the labels will become too cluttered
        xlabelValues = []
        xlabelPos = []
        ylabelValues = []
        ylabelPos = []
        
        # [한국어] X축 binning 개수 클리핑
        if self.xAxisStepsWilStack[self.currPlot] < 1:
            self.xAxisStepsWilStack[self.currPlot] = 1
        if self.xAxisStepsWilStack[self.currPlot] > len(x):
            self.xAxisStepsWilStack[self.currPlot] = len(x)
        
        # [한국어] Y축 binning 개수 클리핑
        if self.yAxisStepsWilStack[self.currPlot] == 'null':
            self.yAxisStepsWilStack[self.currPlot] = 8
        if self.yAxisStepsWilStack[self.currPlot] < 1:
            self.yAxisStepsWilStack[self.currPlot] = 1
        if self.yAxisStepsWilStack[self.currPlot] > len(y):
            self.yAxisStepsWilStack[self.currPlot] = len(y)
            
        
        # put number on axis if there are more than one ticks 
        if (self.xAxisStepsWilStack[self.currPlot] != 1):
            for count in range(0,len(x),int(len(x)/self.xAxisStepsWilStack[self.currPlot])):
                xlabelValues.append(x[count])
                xlabelPos.append(xticksPos[count])
        
        print(self.yAxisStepsWilStack[self.currPlot])
        for count in range(0,len(y),int(len(y)/self.yAxisStepsWilStack[self.currPlot])):
            ylabelValues.append(yTicks[count])
            ylabelPos.append(yticksPos[count])            

        #Now that we have all of our axis labels, lets set them
        self.plot.set_yticklabels(ylabelValues, fontsize = plotFormat.yticksFontSize)
        self.plot.set_yticks(ylabelPos)
        self.plot.set_xticklabels(xlabelValues, rotation = plotFormat.rotation, fontsize = plotFormat.xticksFontSize)
        self.plot.set_xticks(xlabelPos)
        
        interpolation = 'nearest'
        norm = plotFormat.norm
        im = self.plot.imshow(y, cmap = cmap, interpolation = interpolation, aspect = 'auto', norm = norm )
        # tmp = im.get_axes().get_position().get_points()
        tmp = im.get_window_extent().get_points()

        # [한국어] 기존 colorbar 축 제거 후 새로 추가
        if (plotID in self.cbarAxes):
            self.figure.delaxes(self.cbarAxes[plotID])
        cax = self.figure.add_axes([0.91, tmp[0][1], 0.01, tmp[1][1] - tmp[0][1]])
        cbar = self.figure.colorbar(im, cax = cax, orientation = 'vertical')
        self.cbarAxes[plotID]= cax  # use for cleanup
        self.colorbars[self.currPlot] = cbar

        self.plotFormatInfo[plotID].InitLabels( cbarlabel = 'Scale: ' + colorAxis, xlabel = xAxis + ' ...' + self.dataPointer.fileChosen[-80:], ylabel = yAxis )
        cbar.set_label( plotFormat.cbarlabel, fontsize = plotFormat.labelFontSize)
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(plotFormat.cticksFontSize)
        self.plot.set_xlabel(plotFormat.xlabel, fontsize = plotFormat.labelFontSize)
        self.plot.set_ylabel(plotFormat.ylabel, fontsize = plotFormat.labelFontSize)
        #self.plot.set_title(self.dataChosenY)

        # provide default scaling to match all plots on figure 
        xtickStep = x[1] - x[0]
        self.plot.set_xlim(0 / xtickStep - 0.5, self.xlim / xtickStep + 0.5)

        self.canvas.draw()
        
    # [한국어]
    # updateWilTicks: 데이터 길이에 따라 0..N-1 눈금 위치 생성.
    def updateWilTicks(self, z):
        x= []
        pos = []
        for y in range(0,len(z)):
            x.append(y)
            pos.append(y)
        return x, pos
    
    def updateWilScaleTicks(self, z):
        x= []
        pos = []
        for y in range(0,len(z)):
            x.append(y)
            pos.append(y)
        return x, pos   
    
    # [한국어]
    # refreshInputs: 입력 파일을 다시 파싱/재배열하고 graphManager를 재초기화.
    def refreshInputs(self):
        
        numPlots = len(self.dataChosen[1]) + 1
        self.dataPointer = self.dataChosen[0]
        for self.currPlot in range(1,numPlots + 1):
          del self.data[self.dataPointer.fileChosen]
          self.data[self.dataPointer.fileChosen] = lexyacc.parseMe(self.dataPointer.fileChosen)
          
          markForDel = []
          for variables in self.data[self.dataPointer.fileChosen]:
              if self.checkEmpty(self.data[self.dataPointer.fileChosen][variables].data) == 0:
                  markForDel.append(variables)
      
          for variables in markForDel:
              del self.data[self.dataPointer.fileChosen][variables]
          
          self.data[self.dataPointer.fileChosen] = organizedata.organizedata(self.data[self.dataPointer.fileChosen])  
          
          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
    
        self.graphArea.destroy()
        self.underneathGraph.destroy()
        self.currPlot = 1
        self.__init__(self.master, self.data, self.res, self.dataChosen)
    
    # [한국어]
    # checkEmpty: 리스트에 0/NULL이 아닌 값이 있는지 확인.
    def checkEmpty(self,list):
      bool = 0
      for x in list:
          if ((x != 0) and (x != 'NULL')):
              bool = 1
      return bool
      
    # [한국어]
    # type3findxlim: stacked bar에서 x축 상한을 cycleStep으로 정규화.
    def type3findxlim(self, max, cycleStep):
      cycleStep = float(cycleStep)
      factor = 1.0/cycleStep
      newXlim = self.xlim*factor
      return newXlim
  
    # [한국어]
    # wilsonScaleX: Parallel Intensity Plot용 데이터를 xlim 길이에 맞춰 0 패딩.
    def wilsonScaleX(self, yshort):
      numPixels = self.xlim/self.cycleStep
      if len(yshort[0]) < numPixels:
        length  = len(yshort[0])
        for rows in range(0,len(yshort)):
          for count in range(0, numPixels - length):
              yshort[rows].append(0)
      return yshort
      
    
    # [한국어]
    # changeColorMapMaxMin: colorbar 최소/최대값 및 colormap 변경 팝업.
    def changeColorMapMaxMin(self):
        #Variable initializations
        NEWFRAME = Tk.Toplevel(self.master, bg = 'white') 
        self.currPlot = 1
        numPlots = len(self.dataChosen[1]) + 1
        absoluteMax = 0
        
        root = []
        entry = {}
        #need to scale all plots to the same length
        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        
        for self.currPlot in range(1,numPlots + 1):
          self.findKernalLocs()
          if 'Parallel Intensity Plot' in self.dataPointer.graphChosen:
              if self.simplerName[self.dataPointer.dataChosenY].type == 5:
                  locMax = self.simplerName[self.dataPointer.dataChosenY].data[0,1][0]
                  locMin = 0
              elif self.simplerName[self.dataPointer.dataChosenY].type != 1:
                  if self.dataPointer.dydx == 0:
                      y = self.simplerName[self.dataPointer.dataChosenY].data
                  else:
                      for iter in range(0, self.dataPointer.dydx):
                        if self.simplerName[self.dataPointer.dataChosenY].type == 4:
                          keys = list(self.simplerName[self.dataPointer.dataChosenY].data.keys())
                          keys.sort()
                          y = []
                          for iter in keys:
                            y.append(self.simplerName[self.dataPointer.dataChosenY].data[iter])
                          y = self.takeDerivativeMult(self.simplerName[self.dataPointer.dataChosenX].data, y)
                        else:
                          y = self.takeDerivativeMult(self.simplerName[self.dataPointer.dataChosenX].data, self.simplerName[self.dataPointer.dataChosenY].data)
                  locMax = 0
                  locMin = 99999999999999999999999999999
                  for iter in y:
                      if max(iter) > locMax:
                          locMax = max(iter)
                          if locMax > absoluteMax:
                              absoluteMax = locMax
                  for iter in y:
                      if min(iter) < locMin:
                          locMin = min(iter)
              else:
                  if self.dataPointer.dydx == 0:
                      y = self.simplerName[self.dataPointer.dataChosenY].data
                  else:
                      for iter in range(0, self.dataPointer.dydx):
                          y = self.takeDerivative(self.simplerName[self.dataPointer.dataChosenX].data, self.simplerName[self.dataPointer.dataChosenY].data)
                  locMax = max(y)
                  locMin = min(y)
                  if locMax > absoluteMax:
                      absoluteMax = locMax
              root.append(Tk.Frame(NEWFRAME, bg = 'white'))
              root[-1].pack(side = Tk.TOP, anchor = Tk.W, pady = 10)
              plotLabel = Tk.Label(root[-1], text= '\n' + self.dataPointer.dataChosenY + ' vs ' + self.dataPointer.dataChosenX  + '\t Current Max: ' + str(locMax) + ' \t New Max: ', bg= 'white')
              plotLabel.pack(side = Tk.LEFT)
              maxEntry = Tk.Entry(root[-1])
              maxEntry.insert(0, str(locMax))
              maxEntry.pack(side = Tk.LEFT)
              plotLabel1 = Tk.Label(root[-1], text = '\t Current Min: ' + str(locMin) + '\t New Min: ', bg= 'white')
              plotLabel1.pack(side = Tk.LEFT)
              minEntry = Tk.Entry(root[-1])
              minEntry.insert(0, str(locMin))
              minEntry.pack(side = Tk.LEFT, padx = 15)
              entry[self.currPlot] = (maxEntry, minEntry)
        
              cmap = self.plotFormatInfo[self.currPlot].cmap
              plotCMap = Tk.OptionMenu(*(root[-1], cmap) + tuple(PlotFormatInfo.cmapOptions)) 
              plotCMap.pack(side = Tk.LEFT, padx = 5)
          

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
        
        bottomFrame = Tk.Frame(NEWFRAME, bg = 'white')
        bottomFrame.pack(side = Tk.BOTTOM, pady = 15)
        note = Tk.Label(bottomFrame, text = '*Leave with default values to reset', bg= 'white')
        note.pack(side = Tk.BOTTOM)
        bDONE = Tk.Button(bottomFrame, text= 'Submit Changes', bg = 'green', command = lambda: self.collectDataChangeColormaps(entry,NEWFRAME))
        bDONE.pack(side = Tk.BOTTOM)
        bSetToGlobalMax = Tk.Button(bottomFrame, text = 'Normalize all Subplots', command = lambda: self.normalizeSubplots(absoluteMax, entry, NEWFRAME))
        bSetToGlobalMax.pack(side = Tk.BOTTOM)
        bCancel = Tk.Button(bottomFrame, text = 'Cancel', command = lambda: NEWFRAME.destroy())
        bCancel.pack(side = Tk.BOTTOM)
        
    # [한국어]
    # normalizeSubplots: 모든 서브플롯을 동일한 vmax(전역 최대값)로 정규화.
    def normalizeSubplots(self, absoluteMax, dict, master):
        listKeys = list(dict.keys())
        for plotID in listKeys:
            self.plotFormatInfo[plotID].norm.vmax = int(absoluteMax)
            self.plotFormatInfo[plotID].norm.vmin = 0
        master.destroy()
        self.plotData()
        
    # [한국어]
    # collectDataChangeColormaps: 사용자가 입력한 max/min 값을 norm에 반영하고 replot.
    def collectDataChangeColormaps(self, dict, master):
        listKeys = list(dict.keys())
        for plotID in listKeys:
            maxEntry, minEntry = dict[plotID]
            maxValue = maxEntry.get()
            if maxValue != '':
                self.plotFormatInfo[plotID].norm.vmax = float(maxValue)
            minValue = minEntry.get()
            if minValue != '':
                self.plotFormatInfo[plotID].norm.vmin = float(minValue)
        master.destroy()
        self.plotData() # Now replot with changes.....
        
    # [한국어]
    # takeDerivativeButton: dy/dx 적용 여부를 선택하는 팝업.
    def takeDerivativeButton(self):
        #Variable initializations
        NEWFRAME = Tk.Toplevel(self.master, bg = 'white') 
        self.currPlot = 1
        numPlots = len(self.dataChosen[1]) + 1
        
        root = []
        checkButton = {}
        vars = {}
        #need to scale all plots to the same length
        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        
        for self.currPlot in range(1,numPlots + 1):
          self.findKernalLocs()
          if self.dataPointer.graphChosen == 'Parallel Intensity Plot' or self.dataPointer.graphChosen == 'Line':
              root.append(Tk.Frame(NEWFRAME, bg = 'white'))
              root[-1].pack(side = Tk.TOP, anchor = Tk.W, pady = 10)
              plotLabel = Tk.Label(root[-1], text= '\n' + self.dataPointer.dataChosenY + ' vs ' + self.dataPointer.dataChosenX + '\t \t TakeDiv:', bg= 'white')
              plotLabel.pack(side = Tk.LEFT)
              vars[str(self.currPlot)] = Tk.IntVar()
              checkButton[str(self.currPlot)] = Tk.Checkbutton(root[-1], bg = 'white', variable = vars[str(self.currPlot)])
              checkButton[str(self.currPlot)].pack(side = Tk.LEFT, padx = 10)
              
        
              
          

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
        
        bottomFrame = Tk.Frame(NEWFRAME, bg = 'white')
        bottomFrame.pack(side = Tk.BOTTOM, pady = 15)
        bDONE = Tk.Button(bottomFrame, text= 'Submit Changes', bg = 'green', command = lambda: self.collectDataChangeDiv(vars,NEWFRAME))
        bDONE.pack(side = Tk.BOTTOM)
        bCancel = Tk.Button(bottomFrame, text = 'Cancel', command = lambda: NEWFRAME.destroy())
        bCancel.pack(side = Tk.BOTTOM)
        
        
    # [한국어]
    # collectDataChangeDiv: 체크된 플롯의 dydx를 1 증가시키고 replot.
    def collectDataChangeDiv(self, vars,master):

        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        numPlots = len(self.dataChosen[1]) + 1
        
        for self.currPlot in range(1,numPlots + 1):
          self.findKernalLocs()
          
          if str(self.currPlot) in vars:
              if vars[str(self.currPlot)].get() == 1:
                  self.dataPointer.dydx += 1

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
        
        master.destroy()
        ## Now replot with changes.....
        self.plotData()
      
    # [한국어]
    # changeBinning: Parallel Intensity Plot의 X/Y축 binning 개수 조정 팝업.
    def changeBinning(self):
        #Variable initializations
        NEWFRAME = Tk.Toplevel(self.master, bg = 'white') 
        self.currPlot = 1
        numPlots = len(self.dataChosen[1]) + 1
        root = Tk.Frame(NEWFRAME, bg = 'white')
        root.pack(side = Tk.TOP, anchor = Tk.W, pady = 10)
        plotListbox = Tk.Listbox(root, width = 100, height = 6)
        plotListbox.grid(row = 0, column = 0, rowspan = 2)
        bIncreaseBinningX = Tk.Button(root, text = 'Increase Binning X-Axis', command = lambda: self.collectDataIncreaseXBinning(plotListbox.get('active')))
        bIncreaseBinningX.grid(row = 0, column = 1, padx = 10)
        bDecreaseBinningX = Tk.Button(root, text = 'Decrease Binning X-Axis', command = lambda: self.collectDataDecreaseXBinning(plotListbox.get('active')))
        bDecreaseBinningX.grid(row = 0, column = 2, padx = 10)
        bDecreaseBinningX = Tk.Button(root, text = 'Remove Binning X-Axis', command = lambda: self.collectDataDecreaseXBinning(plotListbox.get('active'), remove = True))
        bDecreaseBinningX.grid(row = 0, column = 3, padx = 10)
        bIncreaseBinningY = Tk.Button(root, text = 'Increase Binning Y-Axis', command = lambda: self.collectDataIncreaseYBinning(plotListbox.get('active')))
        bIncreaseBinningY.grid(row = 1, column = 1, padx = 10)
        bDecreaseBinningY = Tk.Button(root, text = 'Decrease Binning Y-Axis', command = lambda: self.collectDataDecreaseYBinning(plotListbox.get('active')))
        bDecreaseBinningY.grid(row = 1, column = 2, padx = 10)
        bDecreaseBinningY = Tk.Button(root, text = 'Remove Binning Y-Axis', command = lambda: self.collectDataDecreaseYBinning(plotListbox.get('active'), remove = True))
        bDecreaseBinningY.grid(row = 1, column = 3, padx = 10)
        
        bCancel = Tk.Button(root, text = 'Finished' , command = lambda: NEWFRAME.destroy())
        bCancel.grid(row = 0, column = 4, padx = 10)

        #need to scale all plots to the same length
        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        
        for self.currPlot in range(1,numPlots + 1):
          self.findKernalLocs()
          if 'Parallel Intensity Plot' in self.dataPointer.graphChosen:
              plotListbox.insert(Tk.END,str(self.currPlot) + ' ' + self.dataPointer.dataChosenY + ' vs ' + self.dataPointer.dataChosenX)

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]

    # [한국어]
    # collectDataIncreaseXBinning: X축 binning 개수 증가.
    def collectDataIncreaseXBinning(self, currPlot):
        plotToIncrease = int(currPlot[0])
        self.xAxisStepsWilStack[plotToIncrease] = self.xAxisStepsWilStack[plotToIncrease] + 20
        self.plotDataForNewBinning(plotToIncrease)

    # [한국어]
    # collectDataDecreaseXBinning: X축 binning 개수 감소 또는 제거.
    def collectDataDecreaseXBinning(self, currPlot, remove = False):
        plotToDecrease = int(currPlot[0])
        if (remove == True):
            self.xAxisStepsWilStack[plotToDecrease] = 1
        else:
            self.xAxisStepsWilStack[plotToDecrease] = self.xAxisStepsWilStack[plotToDecrease] - 5
        self.plotDataForNewBinning(plotToDecrease)
        
  
    # [한국어]
    # collectDataIncreaseYBinning: Y축 binning 개수 증가.
    def collectDataIncreaseYBinning(self, currPlot):
        plotToIncrease = int(currPlot[0])
        if (self.yAxisStepsWilStack[plotToIncrease] == 1):
            self.yAxisStepsWilStack[plotToIncrease] = 2
        self.yAxisStepsWilStack[plotToIncrease] = int(float(self.yAxisStepsWilStack[plotToIncrease])*1.50)
        print(self.yAxisStepsWilStack[plotToIncrease])
        self.plotDataForNewBinning(plotToIncrease)

    # [한국어]
    # collectDataDecreaseYBinning: Y축 binning 개수 감소 또는 제거.
    def collectDataDecreaseYBinning(self, currPlot, remove = False):
        plotToDecrease = int(currPlot[0])
        print(self.yAxisStepsWilStack[plotToDecrease])
        if (remove == True):
            self.yAxisStepsWilStack[plotToDecrease] = 1
        else:
            self.yAxisStepsWilStack[plotToDecrease] = int(float(self.yAxisStepsWilStack[plotToDecrease])/1.50)
        self.plotDataForNewBinning(plotToDecrease)
        

    # [한국어]
    # plotDataForNewBinning: binning 변경 후 지정한 플롯만 다시 그림.
    def plotDataForNewBinning(self, plotToChange):
        self.currPlot = 1
        numPlots = len(self.dataChosen[1]) + 1
        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        
        for self.currPlot in range(1,numPlots + 1):
          if self.currPlot == plotToChange:
              self.findKernalLocs()
              self.plot = self.figure.add_subplot(numPlots,1,self.currPlot)
              if self.simplerName[self.dataPointer.dataChosenY].type == 1:
                  self.type1Variable(self.simplerName[self.dataPointer.dataChosenX].data , self.dataPointer.dataChosenX, self.simplerName[self.dataPointer.dataChosenY].data, self.dataPointer.dataChosenY, self.simplerName[self.dataPointer.dataChosenY].bool, self.currPlot)
              elif self.simplerName[self.dataPointer.dataChosenY].type == 2:
                  self.type2Variable(self.simplerName[self.dataPointer.dataChosenX].data,self.dataPointer.dataChosenX, self.simplerName[self.dataPointer.dataChosenY].data,self.dataPointer.dataChosenY, self.currPlot)
              elif self.simplerName[self.dataPointer.dataChosenY].type == 3:
                  self.type3Variable(self.simplerName[self.dataPointer.dataChosenX].data,self.dataPointer.dataChosenX,self.simplerName[self.dataPointer.dataChosenY].data,self.dataPointer.dataChosenY, self.currPlot)
              else:
                  self.type4Variable(self.simplerName[self.dataPointer.dataChosenX].data,self.dataPointer.dataChosenX,self.simplerName[self.dataPointer.dataChosenY].data,self.dataPointer.dataChosenY, self.currPlot)
              
          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
              
              
    # [한국어]
    # editLabelsButton: 축 레이블/제목/폰트 크기 수정 팝업.
    def editLabelsButton(self):
        #Variable initializations
        NEWFRAME = Tk.Toplevel(self.master, bg = 'white') 
        self.currPlot = 1
        numPlots = len(self.dataChosen[1]) + 1
        
        root = Tk.Frame(NEWFRAME, bg = 'white')
        root.pack(side = Tk.TOP, anchor = Tk.W, pady = 10)
        entries = {}
        #need to scale all plots to the same length
        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        numPlots = len(self.dataChosen[1]) + 1
        currentRow = 0
        
        for self.currPlot in range(1,numPlots + 1):
          self.plot = self.figure.add_subplot(numPlots,1,self.currPlot)
          plotFormat = self.plotFormatInfo[self.currPlot]
          self.findKernalLocs()
          entries[self.currPlot] = []
          
          plotLabel = Tk.Label(root, text= self.dataPointer.dataChosenY + ' vs ' + self.dataPointer.dataChosenX, bg= 'white')
          plotLabel.grid(row = currentRow, column = 0, pady = 10, padx = 15, sticky=Tk.W)
          plotLabel1 = Tk.Label(root, text = 'Y Axis: ', bg = 'white')
          plotLabel1.grid(row = currentRow, column = 1)
          entries[self.currPlot].append(Tk.Entry(root))
          entries[self.currPlot][-1].grid(row = currentRow, column = 2, padx = 10)
          entries[self.currPlot][-1].insert(0, self.plot.get_ylabel())
          plotLabel2 = Tk.Label(root, text = 'X Axis: ', bg = 'white')
          plotLabel2.grid(row = currentRow, column = 3)
          entries[self.currPlot].append(Tk.Entry(root, width = 50))
          entries[self.currPlot][-1].grid(row = currentRow, column = 4, padx = 10)
          entries[self.currPlot][-1].insert(0, self.plot.get_xlabel())
          if self.currPlot in self.colorbars:
              plotLabel3 = Tk.Label(root, text = 'Colorbar: ', bg = 'white')
              plotLabel3.grid(row = currentRow, column = 5)
              entries[self.currPlot].append(Tk.Entry(root, width = 20))
              entries[self.currPlot][-1].grid(row = currentRow, column = 6, padx = 10)
              entries[self.currPlot][-1].insert(0, plotFormat.cbarlabel)
          else:
              plotLabel3 = Tk.Label(root, text = 'Title: ', bg = 'white')
              plotLabel3.grid(row = currentRow, column = 5)
              entries[self.currPlot].append(Tk.Entry(root, width = 20))
              entries[self.currPlot][-1].grid(row = currentRow, column = 6, padx = 10)
              entries[self.currPlot][-1].insert(0, plotFormat.title)

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
        
          plotFontLabel = Tk.Label(root, text = 'Label \nFont Size: ', bg = 'white')
          plotFontLabel.grid(row = currentRow, column = 7)
          entries[self.currPlot].append(Tk.Entry(root, width = 5))
          entries[self.currPlot][-1].grid(row = currentRow, column = 8, padx = 10)
          entries[self.currPlot][-1].insert(0, plotFormat.labelFontSize)

          plotFontLabel = Tk.Label(root, text = 'X Ticks \nFont Size: ', bg = 'white')
          plotFontLabel.grid(row = currentRow, column = 9)
          entries[self.currPlot].append(Tk.Entry(root, width = 5))
          entries[self.currPlot][-1].grid(row = currentRow, column = 10, padx = 10)
          entries[self.currPlot][-1].insert(0, plotFormat.xticksFontSize)

          plotFontLabel = Tk.Label(root, text = 'Y Ticks \nFont Size: ', bg = 'white')
          plotFontLabel.grid(row = currentRow, column = 11)
          entries[self.currPlot].append(Tk.Entry(root, width = 5))
          entries[self.currPlot][-1].grid(row = currentRow, column = 12, padx = 10)
          entries[self.currPlot][-1].insert(0, plotFormat.yticksFontSize)

          currentRow += 1

        bottomFrame = Tk.Frame(NEWFRAME, bg = 'white')
        bottomFrame.pack(side = Tk.BOTTOM, pady = 15)
        bDONE = Tk.Button(bottomFrame, text= 'Submit Changes', bg = 'green', command = lambda: self.collectDataEditLabels(entries,NEWFRAME))
        bDONE.pack(side = Tk.BOTTOM)
        bCancel = Tk.Button(bottomFrame, text = 'Cancel', command = lambda: NEWFRAME.destroy())
        bCancel.pack(side = Tk.BOTTOM)
        
    # [한국어]
    # collectDataEditLabels: 입력된 레이블/폰트를 plotFormat에 반영하고 다시 그림.
    def collectDataEditLabels(self, entries, master):

        self.dataPointer = self.dataChosen[0]
        self.simplerName = self.data[self.dataPointer.fileChosen]
        numPlots = len(self.dataChosen[1]) + 1
        
        for self.currPlot in range(1,numPlots + 1):
          self.findKernalLocs()
          self.plot = self.figure.add_subplot(numPlots,1,self.currPlot)
          plotFormat = self.plotFormatInfo[self.currPlot]
         
          labelFontOptionIndex = 3
          if entries[self.currPlot][labelFontOptionIndex].get() != '':
              plotFormat.labelFontSize = int(entries[self.currPlot][labelFontOptionIndex].get())
          ticksFontOptionIndex = labelFontOptionIndex + 1
          if entries[self.currPlot][ticksFontOptionIndex].get() != '':
              plotFormat.xticksFontSize = int(entries[self.currPlot][ticksFontOptionIndex].get())
          ticksFontOptionIndex = ticksFontOptionIndex + 1
          if entries[self.currPlot][ticksFontOptionIndex].get() != '':
              plotFormat.yticksFontSize = int(entries[self.currPlot][ticksFontOptionIndex].get())
              plotFormat.cticksFontSize = plotFormat.yticksFontSize

          plotFormat.ylabel = entries[self.currPlot][0].get()
          self.plot.set_ylabel(plotFormat.ylabel, fontsize=plotFormat.labelFontSize)
          plotFormat.xlabel = entries[self.currPlot][1].get()
          self.plot.set_xlabel(plotFormat.xlabel, fontsize=plotFormat.labelFontSize)
          if self.currPlot in self.colorbars:
              plotFormat.cbarlabel = entries[self.currPlot][2].get()
              self.colorbars[self.currPlot].set_label(plotFormat.cbarlabel, fontsize=plotFormat.labelFontSize)
          else:
              plotFormat.title = entries[self.currPlot][2].get()
              self.plot.set_title(plotFormat.title)
          
          # change xtick label fontsize
          if (plotFormat.xticksFontSize == 0):
              self.plot.set_xticklabels([])
          xtickslabels = self.plot.get_xmajorticklabels()
          for n in range(0,len(xtickslabels)):
              xtickslabels[n].set_fontsize(plotFormat.xticksFontSize)
          
          # change ytick label fontsize
          ytickslabels = self.plot.get_ymajorticklabels()
          for n in range(0,len(ytickslabels)):
              ytickslabels[n].set_fontsize(plotFormat.yticksFontSize)

          # change colorbar ticks label fontsize
          if self.currPlot in self.colorbars:
              for label in self.colorbars[self.currPlot].ax.get_yticklabels():
                  label.set_fontsize(plotFormat.cticksFontSize)

          if self.dataChosen[1] != [] and self.currPlot != numPlots:
              self.dataPointer = self.dataChosen[1][self.currPlot - 1]
              self.simplerName = self.data[self.dataPointer.fileChosen]
        
        master.destroy()
        ## Now replot with changes.....
        self.canvas.draw()

    # [한국어]
    # zoomButton: X/Y축 범위 조정 팝업.
    def zoomButton(self):
        #Variable initializations
        NEWFRAME = Tk.Toplevel(self.master, bg = 'white') 
        numPlots = len(self.dataChosen[1]) + 1
        
        root = Tk.Frame(NEWFRAME, bg = 'white')
        root.pack(side = Tk.TOP, anchor = Tk.W, pady = 10)

        numPlots = len(self.dataChosen[1]) + 1
        currentRow = 0
        
        # create text label for fields
        labelYmin = Tk.Label(root, text = 'Y min: ', bg = 'white')
        labelYmin.grid(row = currentRow, column = 1)
        labelYmax = Tk.Label(root, text = 'Y max: ', bg = 'white')
        labelYmax.grid(row = currentRow, column = 2)
        labelXmin = Tk.Label(root, text = 'X min: ', bg = 'white')
        labelXmin.grid(row = currentRow, column = 3)
        labelXmax = Tk.Label(root, text = 'X max: ', bg = 'white')
        labelXmax.grid(row = currentRow, column = 4)
        currentRow += 1

        # for x-axis scaling
        xData = self.simplerName[self.dataPointer.dataChosenX].data
        xtickStep = xData[1] - xData[0]

        # create entry fields for each plot
        entries = {}
        for plotID in range(1,numPlots + 1):
            plot = self.plotRef[plotID]
            plotFormat = self.plotFormatInfo[plotID]
            ymin, ymax = plot.get_ylim()
            xmin, xmax = plot.get_xlim()
            if ('Parallel Intensity Plot' in plotFormat.graphType):
                ymin = (ymin - 0.5)
                ymax = (ymax + 0.5)
                xmin = (xmin + 0.5) * xtickStep
                xmax = (xmax - 0.5) * xtickStep
            elif ('Bar Chart' in plotFormat.graphType):
                xmin = (xmin) * xtickStep
                xmax = (xmax) * xtickStep
            
            plotLabel = Tk.Label(root, text= self.dataPointer.dataChosenY + ' vs ' + self.dataPointer.dataChosenX, bg= 'white')
            plotLabel.grid(row = currentRow, column = 0, pady = 10, padx = 15, sticky=Tk.W)

            entries[plotID] = []
            # Y min
            entries[plotID].append(Tk.Entry(root, width = 12))
            entries[plotID][-1].grid(row = currentRow, column = 1, padx = 10)
            entries[plotID][-1].insert(0, ymin)
            # Y max
            entries[plotID].append(Tk.Entry(root, width = 12))
            entries[plotID][-1].grid(row = currentRow, column = 2, padx = 10)
            entries[plotID][-1].insert(0, ymax)
            # X min
            entries[plotID].append(Tk.Entry(root, width = 12))
            entries[plotID][-1].grid(row = currentRow, column = 3, padx = 10)
            entries[plotID][-1].insert(0, xmin)
            # X max
            entries[plotID].append(Tk.Entry(root, width = 12))
            entries[plotID][-1].grid(row = currentRow, column = 4, padx = 10)
            entries[plotID][-1].insert(0, xmax)
          
            currentRow += 1

        bottomFrame = Tk.Frame(NEWFRAME, bg = 'white')
        bottomFrame.pack(side = Tk.BOTTOM, pady = 15)
        bDONE = Tk.Button(bottomFrame, text = 'Apply Changes', bg = 'green', command = lambda: self.zoomCollect(entries,NEWFRAME))
        bDONE.pack(side = Tk.BOTTOM)
        bCancel = Tk.Button(bottomFrame, text = 'Cancel', command = lambda: NEWFRAME.destroy())
        bCancel.pack(side = Tk.BOTTOM)
        
    # [한국어]
    # zoomCollect: 사용자가 입력한 범위를 각 플롯에 적용.
    def zoomCollect(self, entries, master):
 
        numPlots = len(self.dataChosen[1]) + 1
        zoomLimit = [ [] for plotID in range(numPlots + 1) ]

        # for xtick generation
        xData = self.simplerName[self.dataPointer.dataChosenX].data
        xtickStep = xData[1] - xData[0]
        xLength = len(xData)

        # obtain entered x and y limit
        for plotID in range(1,numPlots + 1):
            plot = self.plotRef[plotID]
           
            ylim = plot.get_ylim()
            xlim = plot.get_xlim()
            zoomLimit[plotID] = [ylim[0], ylim[1], xlim[0], xlim[1]] 
            for idx in range(len(zoomLimit[plotID])):
                limitStr = entries[plotID][idx].get()
                if limitStr != '':
                    zoomLimit[plotID][idx] = float(limitStr)
            
        # apply change of x and y limit to each plot
        for plotID in range(1,numPlots + 1):
            plot = self.plotRef[plotID]
            plotFormat = self.plotFormatInfo[plotID]
            if ('Parallel Intensity Plot' in plotFormat.graphType):
                zoomLimit[plotID][2] /= xtickStep
                zoomLimit[plotID][3] /= xtickStep
                zoomLimit[plotID][0] += 0.5
                zoomLimit[plotID][1] -= 0.5
                zoomLimit[plotID][2] -= 0.5
                zoomLimit[plotID][3] += 0.5
            elif ('Bar Chart' in plotFormat.graphType):
                zoomLimit[plotID][2] /= xtickStep
                zoomLimit[plotID][3] /= xtickStep
            plot.set_ylim( ymin = zoomLimit[plotID][0], ymax = zoomLimit[plotID][1])
            plot.set_xlim( xmin = zoomLimit[plotID][2], xmax = zoomLimit[plotID][3])

            # regenerate the xtick labels if the new maximum is exceeds the data length
            if ('Parallel Intensity Plot' in plotFormat.graphType or 'Bar Chart' in plotFormat.graphType):
                xmaxNew = zoomLimit[plotID][3]
                if (xmaxNew > xLength):
                    xlabelValues = []
                    xlabelPos = []
                    if (self.xAxisStepsWilStack[plotID] != 1):
                        xNticks = self.xAxisStepsWilStack[plotID]
                        for count in range(0, int(xmaxNew + 1), int(xmaxNew + 1) / xNticks):
                            xlabelValues.append(count * xtickStep)
                            xlabelPos.append(count)
                    
                    plot.set_xticklabels(xlabelValues, rotation = plotFormat.rotation, fontsize = plotFormat.xticksFontSize)
                    plot.set_xticks(xlabelPos)

        master.destroy()
        self.canvas.draw()
    

# [한국어] Source Code View 폰트 정보 클래스.
class NaviPlotInfo:

    srcViewSection = 'SourceCodeView'

    # [한국어]
    # 생성자: config.rc의 SourceCodeView 섹션에서 폰트 크기 읽기.
    def __init__(self):
        self.titleFontSize = avconfig.get_value(self.__class__.srcViewSection, 'titleFontSize', 12)
        self.ylabelFontSize = avconfig.get_value(self.__class__.srcViewSection, 'yLabelFontSize', 12)
        self.xlabelFontSize = avconfig.get_value(self.__class__.srcViewSection, 'xLabelFontSize', 12)
        self.yticksFontSize = avconfig.get_value(self.__class__.srcViewSection, 'yTicksFontSize', 10)
        self.xticksFontSize = avconfig.get_value(self.__class__.srcViewSection, 'xTicksFontSize', 10)

# [한국어] Source Code View 탭 클래스.
class newTextTab:
    
    srcViewSection = 'SourceCodeView'

    # [한국어]
    # 생성자: CUDA/PTX 파일 선택 UI와 통계 선택 UI를 구성.
    def __init__(self, textTabs, numb, res, TEFILES):  
        
        tabnum = "self.page " + numb
        self.page = textTabs.add(tabnum)
        self.res = res
        self.TEFILES = TEFILES
        self.fileChosen = ''
        self.typeFileChosen = ''
        self.chosenStat1 = ''
        self.chosenStat2 = ''
        self.key2bool = 0
        self.chosenMethod = ''
        self.showLineStatName = 0
        self.first_draw = 1
        self.textFont = ('courier', avconfig.get_value(self.__class__.srcViewSection, 'codeFontSize', 8))
        self.naviPlotInfo = NaviPlotInfo()
        
        
        # [한국어] 해상도별 배경 프레임 크기
        if self.res == "small":
            self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 700, width = 1200);
        elif self.res == 'medium':
            self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
        else:
            self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
        self.background.pack()
        self.background.pack_propagate(0)
        
        # [한국어] CUDA 소스 파일 선택 Listbox
        chooseFileFrame = Tk.Frame(self.background, bg = 'white')
        chooseFileFrame.pack(side = Tk.TOP, anchor = Tk.W, pady = 10, padx = 5)
        lChooseFile = Tk.Label(chooseFileFrame, text = 'Choose a Text File to Display:    ', font = ("Gills Sans MT", 12), bg = 'white' )
        lChooseFile.pack(side= Tk.LEFT)
        
        fileListboxOuterFrame = Tk.Frame(chooseFileFrame, bg = 'white')
        fileListboxOuterFrame.pack(side = Tk.LEFT)
        
        cAvailableCudaFilesFrame = Tk.Frame(fileListboxOuterFrame, bg = 'white')
        cAvailableCudaFilesFrame.pack(side = Tk.TOP)
        cAvailableCudaFilesTitle = Tk.Label(cAvailableCudaFilesFrame, text= 'Cuda C', bg= 'white')
        cAvailableCudaFilesTitle.pack(side = Tk.TOP)
        self.cAvailableCudaFiles = Tk.Listbox(cAvailableCudaFilesFrame, width = 100, height = 4)
        self.cAvailableCudaFiles.pack(side = Tk.BOTTOM)
        for keys in TEFILES[0]:
            self.cAvailableCudaFiles.insert(Tk.END, keys)
            
        self.cAvailableCudaFiles.bind("<Double-Button-1>", self.chooseFileCuda)
            
        lOr = Tk.Label(chooseFileFrame, text = 'OR', font = ("Gills Sans MT", 20), bg= 'white')
        lOr.pack(side = Tk.LEFT)
        
        # [한국어] PTX 파일 선택 Listbox
        cAvailablePTXFilesFrame = Tk.Frame(fileListboxOuterFrame, bg = 'white')
        cAvailablePTXFilesFrame.pack(side = Tk.BOTTOM)
        cAvailablePTXFilesTitle = Tk.Label(cAvailablePTXFilesFrame, text= 'PTX', bg= 'white')
        cAvailablePTXFilesTitle.pack(side = Tk.TOP)
        self.cAvailablePTXFiles = Tk.Listbox(cAvailablePTXFilesFrame, width = 100, height = 4)
        self.cAvailablePTXFiles.pack(side = Tk.BOTTOM)
        for keys in TEFILES[1]:
            self.cAvailablePTXFiles.insert(Tk.END, keys)
            
        self.cAvailablePTXFiles.bind("<Double-Button-1>", self.chooseFilePTX)
            
        # [한국어] 통계/함수 선택 영역
        chooseStatsFrame = Tk.Frame(self.background, bg = 'white')
        chooseStatsFrame.pack(side = Tk.TOP, anchor = Tk.W, pady = 5, padx =5)
        lChooseStats = Tk.Label(chooseStatsFrame, text = "Choose Data to be Shown:  ", font = ("Gills Sans MT", 12), bg= 'white')
        lChooseStats.pack(side = Tk.LEFT)
        availMethodsListboxFrame = Tk.Frame(chooseStatsFrame, bg = 'white')
        availMethodsListboxFrame.pack(side = Tk.LEFT)
        lavailMethodsTitle = Tk.Label(availMethodsListboxFrame, text = 'Available Functions', bg= 'white')
        lavailMethodsTitle.pack(side = Tk.TOP)
        self.cAvailableMethodsOnStats = Tk.Listbox(availMethodsListboxFrame, width = 25, height = 10)
        self.cAvailableMethodsOnStats.pack(side = Tk.BOTTOM, anchor = Tk.W)
        self.cAvailableMethodsOnStats.bind("<Double-Button-1>", self.chooseMethod)
        
        # [한국어] 사용 가능한 통계 변수 리스트박스 1, 2
        availStatsListboxFrame1 = Tk.Frame(chooseStatsFrame, bg = 'white')
        availStatsListboxFrame1.pack(side = Tk.LEFT)
        lavailStatsTitle1 = Tk.Label(availStatsListboxFrame1, text = 'Available Stats', bg= 'white')
        lavailStatsTitle1.pack(side = Tk.TOP)
        self.cAvailableStats1 = Tk.Listbox(availStatsListboxFrame1, width = 25, height = 10)
        self.cAvailableStats1.pack(side = Tk.BOTTOM, anchor = Tk.W, padx = 5)
        self.cAvailableStats1.bind("<Double-Button-1>", self.chooseStats1)
        
        
        availStatsListboxFrame2 = Tk.Frame(chooseStatsFrame, bg = 'white')
        availStatsListboxFrame2.pack(side = Tk.LEFT)
        lavailStatsTitle2 = Tk.Label(availStatsListboxFrame2, text = 'Available Stats', bg= 'white')
        lavailStatsTitle2.pack(side = Tk.TOP)
        self.cAvailableStats2 = Tk.Listbox(availStatsListboxFrame2, width = 25, height = 10)
        self.cAvailableStats2.pack(side = Tk.BOTTOM, anchor = Tk.W, padx = 5)
        self.cAvailableStats2.bind("<Double-Button-1>", self.chooseStats2)

        
        
        
        chosenDataFrame = Tk.Frame(chooseStatsFrame, bg = 'white')
        chosenDataFrame.pack(side = Tk.LEFT, padx = 5)
        lChosenData = Tk.Label(chosenDataFrame, text = 'Chosen Data', bg = 'white')
        lChosenData.pack(side = Tk.TOP)
        self.cChosenData = Tk.Text(chosenDataFrame, width = 45, height = 10)
        self.cChosenData.pack(side = Tk.TOP)
        self.cChosenData.tag_config('complete', background = 'green')
        self.cChosenData.tag_config('incomplete', background= 'red')
        
        fShowDataFrame = Tk.Frame(self.background, bg = 'white')
        fShowDataFrame.pack(side = Tk.BOTTOM)
        self.bShowData = Tk.Button(fShowDataFrame, text = 'Show Data', bg = 'green',font = ("Gills Sans MT", 14), command = lambda: self.showData(),borderwidth = 5)
        self.bShowData.pack(side = Tk.RIGHT, pady = 10)
        self.updateChosen()
        

        
    # [한국어]
    # statString: Source Code View 하단 통계 텍스트박스에 표시할 한 행 문자열 생성.
    def statString(self, lineNum, statData):
        if (self.showLineStatName == 1):
            statName = self.chosenStat1
            if (self.chosenStat2 != ''):
                statName += '/' + self.chosenStat2
            finalString = 'Line#: ' + str(lineNum) + ' ' + statName + ': '  + str(statData) + '\n'
        else:
            finalString = 'Line#: ' + str(lineNum) + ' : '  + str(statData) + '\n'
        return finalString
        
    # [한국어]
    # showData: 선택한 소스코드와 통계를 보여주는 메인 Source Code View UI 구성.
    def showData(self):
        self.background.destroy()
        # [한국어] 해상도별 프레임 크기
        if self.res == "small":
            self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 725, width = 1225);
            topFrame = Tk.Frame(self.background, bg = 'white', height = 442, width = 1225)
            textFrame = Tk.Frame(topFrame, bg = 'green', height = 442, width = 860)
            outStatsFrame = Tk.Frame(topFrame, bg= 'white', height = 442, width = 352)
            statsFrame = Tk.Frame(outStatsFrame, bg = 'red', height = 413, width = 352)
            bottomFrame = Tk.Frame(self.background, bg= 'purple', height = 284, width = 1225)
            toolbarFrame = Tk.Frame(outStatsFrame, bg = 'gray', height = 38, width = 352)
        elif self.res == 'medium':
            self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
            topFrame = Tk.Frame(self.background, bg = 'white', height = 575, width = 1530)
            textFrame = Tk.Frame(topFrame, bg = 'green', height = 575, width = 1075)
            outStatsFrame = Tk.Frame(topFrame, bg= 'white', height = 575, width = 440)
            statsFrame = Tk.Frame(outStatsFrame, bg = 'red', height = 537, width = 440)
            bottomFrame = Tk.Frame(self.background, bg= 'purple', height = 370, width = 1530)
            toolbarFrame = Tk.Frame(outStatsFrame, bg = 'black', height = 38, width = 440)
        else:
            self.background = Tk.Frame(self.page, bg = "white", borderwidth = 5, relief = Tk.GROOVE, height = 943, width = 1530);
            topFrame = Tk.Frame(self.background, bg = 'white', height = 575, width = 1530)
            textFrame = Tk.Frame(topFrame, bg = 'green', height = 575, width = 1075)
            outStatsFrame = Tk.Frame(topFrame, bg= 'white', height = 575, width = 440)
            statsFrame = Tk.Frame(outStatsFrame, bg = 'red', height = 537, width = 440)
            bottomFrame = Tk.Frame(self.background, bg= 'purple', height = 370, width = 1530)
            toolbarFrame = Tk.Frame(outStatsFrame, bg = 'gray', height = 38, width = 440)
            
        self.background.pack_propagate(0)
        self.background.pack()
        topFrame.pack_propagate(0)
        topFrame.pack(side = Tk.TOP)
        outStatsFrame.pack_propagate(0)
        outStatsFrame.pack(side = Tk.LEFT)
        statsFrame.pack_propagate(0)
        statsFrame.pack(side = Tk.TOP)
        toolbarFrame.pack_propagate(0)
        toolbarFrame.pack(side = Tk.TOP)
        textFrame.pack_propagate(0)
        textFrame.pack(side = Tk.LEFT)
        bottomFrame.pack_propagate(0)
        bottomFrame.pack(side = Tk.TOP)
        btoolbox = Tk.Button(toolbarFrame, text = 'Toolbox', command = self.toolboxTopLevel)
        btoolbox.pack(side = Tk.RIGHT)
        
        
        
        # [한국어] 소스코드 텍스트박스와 통계 텍스트박스, 공통 스크롤바 연결
        scrollbar = Tk.Scrollbar(statsFrame, orient = Tk.VERTICAL )
        scrollbar.pack(side = Tk.RIGHT, fill = 'y')
        self.textbox = Tk.Text(textFrame, height = 36, width = 150,yscrollcommand = scrollbar.set, wrap = Tk.NONE)
        self.textbox.pack(side = Tk.TOP, anchor = Tk.W, padx =10, pady = 5)
        self.statstextbox = Tk.Text(statsFrame, height = 36, width = 55, yscrollcommand = scrollbar.set, wrap = Tk.NONE)
        self.statstextbox.pack(padx = 10, pady = 5)
        self.statstextbox.tag_config('normal', background = 'white', font = self.textFont)
        self.statstextbox.tag_config('highlight',background = 'lightblue', font = self.textFont)
        scrollbar.config(command = self.yview)
        self.textbox.tag_config('highlight', background = 'lightblue', font = self.textFont)
        self.textbox.tag_config('normal', background = 'white', font = self.textFont)
        
        
        
    

        # [한국어] 소스 파일과 대응 stat 파일 열기
        self.file = open(self.fileChosen, 'r')
        self.Lines = {}
        if self.typeFileChosen == 'cuda':
            self.statFile = self.TEFILES[2][self.TEFILES[0].index(self.fileChosen)]
        else:
            self.statFile = self.TEFILES[2][self.TEFILES[1].index(self.fileChosen) ] 
        
        
        self.stats = lexyacctexteditor.textEditorParseMe(self.statFile)
        
        
        # [한국어] CUDA 파일이면 PTX->CUDA 매핑을 이용해 cudaLineNo 객체 생성
        if self.typeFileChosen == 'cuda':
            self.map = lexyacctexteditor.ptxToCudaMapping(self.TEFILES[1][self.TEFILES[2].index(self.statFile) ] )
            for keys in self.map:
                tmp = []
                for ptxLines in self.map[keys]:
                    try:
                        tmp.append(self.stats[ptxLines])
                    except:
                        noDataArray = []
                        for nStats in variableclasses.lineStatName:
                            noDataArray.append("Null")
                        tmp.append(noDataArray)
                self.Lines[keys] = variableclasses.cudaLineNo(self.map[keys], tmp)
        else:
            for keys in self.stats:
                self.Lines[keys] = variableclasses.ptxLineNo(self.stats[keys])
        
        
        
        
        # [한국어] 소스코드 텍스트박스에 라인 번호와 함께 내용 삽입
        countLines = 1
        for lines in self.file.readlines():
            lines = lines.decode()
            self.textbox.insert(Tk.END, str(countLines) + '.   ' + lines, ('normal'))
            countLines += 1
        countLines -= 1
        self.countLines = countLines
        
        # [한국어] 하단 히스토그램(라인별 통계 막대 그래프) 생성
        figure = Figure(figsize=(22,5), dpi = 70)
        self.histArea = FigureCanvasTkAgg(figure, master= bottomFrame)
        self.histArea.get_tk_widget().pack()
        toolbar  = NavigationToolbar2Tk(self.histArea, toolbarFrame)
        toolbar.update()
        self.histogram = figure.add_subplot(111)
        cid = figure.canvas.mpl_connect('button_press_event',self.onclick)
        
        # [한국어] 라인별 통계값 계산
        self.lineCounts = [0.1]
        for count in range(1,countLines):
            if count in self.Lines:
                if self.typeFileChosen == 'cuda':
                    if self.chosenMethod == 'Ratio':
                        self.lineCounts.append(float(self.Lines[count].takeRatioSums(self.chosenStat1, self.chosenStat2)))
                    elif self.chosenMethod == 'Max':
                        self.lineCounts.append(int(self.Lines[count].takeMax(self.chosenStat1)))
                    else:
                        self.lineCounts.append(int(self.Lines[count].sum(self.chosenStat1)))
                else:
                    if self.chosenMethod == 'Ratio':
                        self.lineCounts.append(float(self.Lines[count].returnRatio(self.chosenStat1, self.chosenStat2)))
                    else:
                        self.lineCounts.append(int(self.Lines[count].returnStat(self.chosenStat1)))
            else:
                if count == countLines - 1:
                    self.lineCounts.append(0.1)
                else:
                    self.lineCounts.append(0)
                    
        if (self.first_draw == 1): 
            self.xlabelfreq = countLines/30
            self.first_draw = 0
        self.countLines = countLines
        width = 0.4
        
        self.generate_xticklabels()

        self.histogram.set_xlim(0, countLines)
        
        self.data_x_posn = [x for x in range(0, countLines)]
        rects1 = self.histogram.bar(self.data_x_posn, self.lineCounts, width, color = 'blue', edgecolor = 'blue' )
        if self.chosenStat2 == '':
            self.histogram.set_title(self.chosenStat1)
        else:
            self.histogram.set_title(self.chosenStat1 + '/' + self.chosenStat2)
        self.histArea.show()
        
        # [한국어] 통계 텍스트박스에 라인별 값 삽입
        count = 0
        for iter in (self.lineCounts + [0]):
            if count == 0:
                pass
            else:
              self.statstextbox.insert(Tk.END, self.statString(count, iter), ('normal'))
            count += 1

    # [한국어]
    # yview: 소스코드 텍스트박스와 통계 텍스트박스를 동시에 스크롤.
    def yview(self, *args):
        self.textbox.yview(*args)
        self.statstextbox.yview(*args)
        
    # [한국어]
    # onclick: 히스토그램에서 마우스 오른쪽 버튼 클릭 시 해당 라인을 소스/통계 텍스트박스에서 하이라이트.
    def onclick(self, event):
      if event.button == 3:
        
        shiftFactor = float(15.25)/float(self.countLines)
        args = ('moveto', str(float(event.xdata)/float(self.countLines) - shiftFactor))
        
        countLines = 1
        self.textbox.delete(0.0, Tk.END)
        self.file = open(self.fileChosen, 'r')
        for lines in self.file.readlines():
          lines=lines.decode()
          if (countLines < event.xdata - 1) or (countLines > event.xdata + 1):
            self.textbox.insert(Tk.END, str(countLines) + '.   ' + lines, ('normal'))
          else:
            self.textbox.insert(Tk.END, str(countLines) + '.   ' + lines, ('highlight'))
          countLines += 1
        
        self.statstextbox.delete(0.0, Tk.END)
        countLines = 1
        count = 1
        for iter in (self.lineCounts[1:] + [0]):
            if (countLines < event.xdata - 1) or (countLines > event.xdata + 1):
                self.statstextbox.insert(Tk.END, self.statString(count, iter), ('normal'))
            else:
                self.statstextbox.insert(Tk.END, self.statString(count, iter), ('highlight'))
            countLines += 1
            count += 1
          
        
        
        self.textbox.yview(*args)
        self.statstextbox.yview(*args)
      
    # [한국어]
    # chooseFileCuda: CUDA 소스 파일 선택.
    def chooseFileCuda(self, *event):
      self.fileChosen = self.cAvailableCudaFiles.get('active')
      self.typeFileChosen = 'cuda'
      self.chosenStat1 = ''
      self.chosenStat2 = ''
      self.key2bool = 0
      
      self.cAvailableStats1.delete(0, Tk.END)
      self.cAvailableStats2.delete(0, Tk.END)
      self.cAvailableMethodsOnStats.delete(0, Tk.END)
      
      
      self.cAvailableMethodsOnStats.insert(Tk.END, 'Sum')
      self.cAvailableMethodsOnStats.insert(Tk.END, 'Max')
      self.cAvailableMethodsOnStats.insert(Tk.END, 'Ratio')


      self.updateChosen()
      
      
      
    # [한국어]
    # chooseFilePTX: PTX 파일 선택.
    def chooseFilePTX(self, *event):
      self.fileChosen = self.cAvailablePTXFiles.get('active')
      self.typeFileChosen = 'ptx'
      self.chosenMethod = ''
      self.chosenStat1 = ''
      self.chosenStat2 = ''
      self.key2bool = 0

      self.cAvailableStats1.delete(0, Tk.END)
      self.cAvailableStats2.delete(0, Tk.END)
      self.cAvailableMethodsOnStats.delete(0, Tk.END)
      
      self.cAvailableMethodsOnStats.delete(0, Tk.END)
      self.cAvailableMethodsOnStats.insert(Tk.END, 'Stat')
      self.cAvailableMethodsOnStats.insert(Tk.END, 'Ratio')
      
      self.updateChosen()
      


    # [한국어]
    # chooseStats1: 첫 번째 통계 변수 선택.
    def chooseStats1(self, *event):
        self.chosenStat1 = self.cAvailableStats1.get('active')
        self.updateChosen()
        
    # [한국어]
    # chooseStats2: 두 번째 통계 변수 선택(Ratio 메서드용).
    def chooseStats2(self, *event):
        self.chosenStat2 = self.cAvailableStats2.get('active')
        self.updateChosen()
      

    # [한국어]
    # chooseMethod: 통계 처리 메서드(Sum/Max/Ratio/Stat) 선택 후 통계 변수 리스트 채움.
    def chooseMethod(self, *event):
        self.chosenMethod = self.cAvailableMethodsOnStats.get('active')
            
        self.key2bool = 0
        self.cAvailableStats1.delete(0, Tk.END)
        self.cAvailableStats2.delete(0, Tk.END)

        if self.typeFileChosen == 'cuda':
            statFile = self.TEFILES[2][self.TEFILES[0].index(self.fileChosen)]
        else:
            statFile = self.TEFILES[2][self.TEFILES[1].index(self.fileChosen)] 

        variableclasses.loadLineStatName(statFile)
       
        for statName in variableclasses.lineStatName:
            self.cAvailableStats1.insert(Tk.END, statName)

        if self.chosenMethod == 'Ratio':
            self.key2bool = 1
            try:
                self.cAvailableStats2.delete(0, Tk.END)
            except:
                pass

            for statName in variableclasses.lineStatName:
                self.cAvailableStats2.insert(Tk.END, statName)

        self.updateChosen()
            
          
    
    
    # [한국어]
    # updateChosen: Source Code View의 'Chosen Data' 텍스트 박스 업데이트.
    def updateChosen(self):
        self.cChosenData.delete(0.0, Tk.END)
        if self.fileChosen == '':
          self.cChosenData.insert(Tk.END, 'File: \n', ('incomplete'))
          self.cChosenData.insert(Tk.END, 'Type File: \n', ('incomplete'))
        else:
          self.cChosenData.insert(Tk.END, 'File: ' + self.fileChosen + '\n', ('complete'))
          self.cChosenData.insert(Tk.END, 'Type File: ' + self.typeFileChosen + '\n', ('complete'))
        if self.chosenStat1 == '':
          self.cChosenData.insert(Tk.END, 'Stat1: \n', ('incomplete'))
        else:
          self.cChosenData.insert(Tk.END, 'Stat1: ' + self.chosenStat1 + '\n', ('complete'))
        if self.key2bool == 1:
          if self.chosenStat2 == '':
              self.cChosenData.insert(Tk.END, 'Stat2: \n', ('incomplete'))
          else:
              self.cChosenData.insert(Tk.END, 'Stat2: ' + self.chosenStat2 + '\n', ('complete'))
          
          
          
        if self.chosenMethod == '':
          self.cChosenData.insert(Tk.END, 'Method: \n', ('incomplete'))
        else:
          self.cChosenData.insert(Tk.END, 'Method: ' + self.chosenMethod + '\n', ('complete'))
        
    # [한국어]
    # toolboxTopLevel: Source Code View 히스토그램용 도구상자 팝업.
    def toolboxTopLevel(self):
        NEWFRAME = Tk.Toplevel(self.background, bg = 'white')
        changefontsize = Tk.Button(NEWFRAME, text = 'Edit Font Size', command = lambda: ( self.editPlotFontSizes(NEWFRAME)))
        changefontsize.pack(side = Tk.LEFT, padx = 5, pady = 5)
        changeLabels = Tk.Button(NEWFRAME, text = 'Edit Label Names', command = lambda:( self.editPlotLabels(NEWFRAME)))
        changeLabels.pack(side = Tk.LEFT, padx = 5, pady = 5)
        changeBinning = Tk.Button(NEWFRAME, text = 'Edit X-Axis Binning', command = lambda: (self.changePlotBinning(NEWFRAME)))
        changeBinning.pack(side = Tk.LEFT, padx = 5, pady = 5)
          
    # [한국어]
    # editPlotLabels: 히스토그램 제목/축 레이블 수정 팝업.
    def editPlotLabels(self, oldframe):
        oldframe.destroy()
        NEWFRAME = Tk.Toplevel(self.background, bg = 'white')
        TopFrame = Tk.Frame(NEWFRAME, bg = 'white')
        TopFrame.pack(side = Tk.TOP)
        lTitle = Tk.Label(TopFrame, text = 'Title: ', bg= 'white')
        lTitle.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eTitle = Tk.Entry(TopFrame)
        eTitle.insert(0, self.histogram.get_title())
        eTitle.pack(side = Tk.LEFT, padx = 5, pady = 5)
        lYAxis = Tk.Label(TopFrame, text = 'Y Axis ', bg = 'white')
        lYAxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eYAxis = Tk.Entry(TopFrame)
        eYAxis.insert(0, self.histogram.get_ylabel())
        eYAxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        lXaxis = Tk.Label(TopFrame, text = 'X Axis', bg = 'white')
        lXaxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eXaxis = Tk.Entry(TopFrame)
        eXaxis.insert(0, self.histogram.get_xlabel())
        eXaxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        bottomFrame = Tk.Frame(NEWFRAME, bg = 'white')
        bottomFrame.pack(side = Tk.BOTTOM, pady = 10)
        bCancel = Tk.Button(bottomFrame, text = 'Cancel', command = lambda: (NEWFRAME.destroy()))
        bCancel.pack(side = Tk.TOP, pady = 5)
        bSubmit = Tk.Button(bottomFrame, text = 'Submit', bg = 'green', command = lambda: self.editPlotLabelsSubmit(NEWFRAME, {'title': eTitle.get(), 'xlabel': eXaxis.get(), 'ylabel': eYAxis.get()}))
        bSubmit.pack(side = Tk.TOP, pady = 5)
        
    # [한국어]
    # editPlotLabelsSubmit: 입력된 레이블을 히스토그램에 반영.
    def editPlotLabelsSubmit(self, oldFrame, entries):
        oldFrame.destroy()
        self.histogram.set_title(entries['title'], fontsize = self.naviPlotInfo.titleFontSize)
        self.histogram.set_xlabel(entries['xlabel'], fontsize = self.naviPlotInfo.xlabelFontSize)
        self.histogram.set_ylabel(entries['ylabel'], fontsize = self.naviPlotInfo.ylabelFontSize)
        self.histArea.show()
        
        
    # [한국어]
    # editPlotFontSizes: 히스토그램 폰트 크기 수정 팝업.
    def editPlotFontSizes(self, oldFrame):
        oldFrame.destroy()
        NEWFRAME = Tk.Toplevel(self.background, bg = 'white')
        TopFrame = Tk.Frame(NEWFRAME, bg = 'white')
        TopFrame.pack(side = Tk.TOP)
        lTitle = Tk.Label(TopFrame, text = 'Title Size: ', bg= 'white')
        lTitle.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eTitle = Tk.Entry(TopFrame)
        eTitle.pack(side = Tk.LEFT, padx = 5, pady = 5)
        lYAxis = Tk.Label(TopFrame, text = 'Y Axis Size: ', bg = 'white')
        lYAxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eYAxis = Tk.Entry(TopFrame)
        eYAxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        lXaxis = Tk.Label(TopFrame, text = 'X Axis Size: ', bg = 'white')
        lXaxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eXaxis = Tk.Entry(TopFrame)
        eXaxis.pack(side = Tk.LEFT, padx = 5, pady = 5)
        lYBinning = Tk.Label(TopFrame, text = 'Y Axis Binning Size: ', bg = 'white')
        lYBinning.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eYBinning = Tk.Entry(TopFrame)
        eYBinning.pack(side = Tk.LEFT, padx = 5, pady = 5)
        lXBinning = Tk.Label(TopFrame, text = 'X Axis Binning Size: ', bg= 'white')
        lXBinning.pack(side = Tk.LEFT, padx = 5, pady = 5)
        eXBinning = Tk.Entry(TopFrame)
        eXBinning.pack(side = Tk.LEFT, padx = 5, pady = 5)
        
        
        
        bottomFrame = Tk.Frame(NEWFRAME, bg = 'white')
        bottomFrame.pack(side = Tk.BOTTOM, pady = 10)
        bCancel = Tk.Button(bottomFrame, text = 'Cancel', command = lambda: (NEWFRAME.destroy()))
        bCancel.pack(side = Tk.TOP, pady = 5)
        bSubmit = Tk.Button(bottomFrame, text = 'Submit', bg = 'green', command = lambda: self.editPlotFontSizesSubmit(NEWFRAME, {'title': eTitle.get(), 'xlabel': eXaxis.get(), 'ylabel': eYAxis.get(), 'ybinning' : eYBinning.get(), 'xbinning': eXBinning.get()}))
        bSubmit.pack(side = Tk.TOP, pady = 5)
    
    # [한국어]
    # editPlotFontSizesSubmit: 입력된 폰트 크기를 히스토그램에 반영.
    def editPlotFontSizesSubmit(self, oldframe, entries):
        oldframe.destroy()
        if entries['title'] != '':
            self.histogram.set_title(self.histogram.get_title(), fontsize = int(entries['title']))
            self.naviPlotInfo.titleFontSize = int(entries['title'])
        if entries['xlabel'] != '':
            self.histogram.set_xlabel(self.histogram.get_xlabel(), fontsize = int(entries['xlabel']))
            self.naviPlotInfo.xlabelFontSize = int(entries['xlabel'])
        if entries['ylabel'] != '':
            self.histogram.set_ylabel(self.histogram.get_ylabel(), fontsize = int(entries['ylabel']))
            self.naviPlotInfo.ylabelFontSize = int(entries['ylabel'])
        if entries['ybinning'] != '':
            for label in self.histogram.get_yticklabels():
                label.set_fontsize(int(entries['ybinning']))
            self.naviPlotInfo.yticksFontSize = int(entries['ybinning'])
        if entries['xbinning'] != '':
            for label in self.histogram.get_xticklabels():
                label.set_fontsize(int(entries['xbinning']))
            self.naviPlotInfo.xticksFontSize = int(entries['xbinning'])
            
        self.histArea.show()
    
    # [한국어]
    # changePlotBinning: X축 눈금 빈도 조정 팝업.
    def changePlotBinning(self,oldframe):
        oldframe.destroy()
        master = Tk.Toplevel(self.background, bg = 'white')
        topFrame = Tk.Frame(master, bg = 'white')
        topFrame.pack(side = Tk.TOP)
        bIncrease = Tk.Button(topFrame, text = 'Increase Frequency', command = lambda: self.increaseBinning())
        bIncrease.pack(side = Tk.LEFT, padx = 5, pady = 10)
        bDecrease = Tk.Button(topFrame, text = 'Decrease Frequency', command = lambda: self.decreaseBinning())
        bDecrease.pack(side = Tk.LEFT, padx = 5)
        bottomFrame = Tk.Frame(master, bg = 'white')
        bottomFrame.pack(side = Tk.BOTTOM, pady = 10)
        bCancel = Tk.Button(bottomFrame, text = 'Cancel', command = lambda: (master.destroy()))
        bCancel.pack(side = Tk.TOP, pady = 5)
        bSubmit = Tk.Button(bottomFrame, text = 'Submit', bg = 'green', command = lambda: master.destroy())
        bSubmit.pack(side = Tk.TOP, pady = 5)

    # [한국어]
    # generate_xticklabels: x축 눈금 라벨 생성.
    def generate_xticklabels(self, fontsize = -1):
        ind = [x * self.xlabelfreq for x in range(0, self.countLines / self.xlabelfreq)]
        self.histogram.set_xticks(ind)

        labels = []
        for x in range(0, self.countLines / self.xlabelfreq):
            labels.append(x * self.xlabelfreq)
        if (fontsize >= 0):
            self.histogram.set_xticklabels(labels, fontsize = fontsize)
        else:
            self.histogram.set_xticklabels(labels)
        
    # [한국어]
    # increaseBinning: x축 눈금 빈도 증가(간격 축소).
    def increaseBinning(self):
        self.xlabelfreq = int(self.xlabelfreq / 1.5)
        if self.xlabelfreq < 1:
            self.xlabelfreq = 1
        self.generate_xticklabels()
        self.histArea.show()        

    # [한국어]
    # decreaseBinning: x축 눈금 빈도 감소(간격 확대).
    def decreaseBinning(self):
        self.xlabelfreq = int(self.xlabelfreq * 1.5)
        self.generate_xticklabels()
        self.histArea.show()        
      
            
            
            
        
