#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel
# and the University of British Columbia, Vancouver, 
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
[한국어 설명] AerialVision 변수/통계 데이터 클래스 (variableclasses.py)

=== 파일의 역할 ===
GPGPU-Sim 시뮬레이션 출력 로그의 통계 변수를 표현하는 데이터 클래스를 정의한다.
변수의 로그 파일 태그, 플롯 타입, 데이터 조직 방식, 데이터 타입 등 메타데이터를
캡슐화하고, 사용자 정의 변수(variables.txt) 파싱을 위한 팩토리 기능을 제공한다.
또한 소스코드 뷰에서 PTX/CUDA 라인별 통계를 다루는 cudaLineNo, ptxLineNo 클래스와
즐겨찾기(bookmark) 클래스도 포함한다.

=== 전체 아키텍처에서의 위치 ===
  aerialvision/lexyacc.py       - 로그 파싱 시 vc.variable 인스턴스 생성
  aerialvision/lexyaccbookmark.py - vc.bookmark 인스턴스 생성
        ↓
  aerialvision/variableclasses.py (이 파일)
        ↓
  aerialvision/organizedata.py   - variable.organize 기반 변환
  aerialvision/guiclasses.py     - variable.data 시각화

=== 타 모듈과의 연결 ===
- lexyacc.py: 로그 행을 파싱하여 variable.data 리스트에 수치를 append.
- organizedata.py: variable.organize(scalar/impVec/idxVec/stackbar/idx2DVec/sparse)
                   값에 따라 데이터 재배열.
- guiclasses.py: variable.type(1~5)에 따라 Line/Bar/Parallel Intensity Plot 등 선택.
- lexyacctexteditor.py: lineStatName 전역 리스트를 사용하여 .stats 파일 헤더 파싱.

=== 주요 함수/구조체 요약 ===
- variable: 통계 변수 메타데이터 및 원본 데이터 컨테이너
- bookmark: 사용자 즐겨찾기 설정 저장
- lineStatName: PTX 라인 통계 파일의 기본 컬럼명 전역 리스트
- loadLineStatName(statFile): 'kernel line :' 헤더에서 컬럼명 오버라이드
- cudaLineNo: CUDA 소스 라인에 대응하는 PTX 라인 집단 통계
- ptxLineNo: 단일 PTX 라인 통계
"""

# [한국어] 통계 변수 클래스.
# 로그 파일의 한 메트릭을 표현하며, 파싱 후 organizedata에서 데이터 형태를 변환한다.
class variable:
   
    # [한국어]
    # 생성자: 통계 변수의 메타데이터를 초기화한다.
    # @lookup_tag: 로그 파일 내 실제 태그 이름 (GUI 표시명과 다를 수 있음)
    # @type: 플롯 타입 (1=scalar, 2=vector, 3=stackedbar, 4=vector2d, 5=sparse)
    # @bool: 커널 시작 시 리셋 여부 (organizedata/kernel 경계 처리에서 사용)
    # @organize: 데이터 조직 방식 ('scalar','impVec','idxVec','stackbar','idx2DVec','sparse','custom')
    # @datatype: Python 타입 (int 또는 float)
    def __init__(self, lookup_tag, type, bool, organize = 'custom', datatype = int):
        self.data = []
        self.lookup_tag = lookup_tag  # the stat name in the log file (can be different from the GUI)
        self.type = type          # plot type 
        self.bool = bool          # wheither to expect reset at the end of a kernel
        self.organize = organize  # how is the data organize in the log, see organizedata.py for options
        self.datatype = datatype  # int or float or other custom type?
        self.initialized = 0
        self.sampleNum = 0

    # [한국어]
    # importFromString: variables.txt 또는 커스텀 헤더의 한 행 문자열로부터
    # 통계 변수 설정을 파싱한다.
    # 입력 형식: <name>, <plot type>, <reset at kernel launch>, <organization>, <data type>
    # 예: "myStat, scalar, 0, scalar, int"
    # @string_spec: 쉼표로 구분된 변수 정의 문자열
    def importFromString(self, string_spec):
        data_type_str = {'int':int, 'float':float}
        plot_type_str = {'scalar': 1, 'vector': 2, 'stackedbar': 3, 'vector2d': 4, 'sparse': 5}
        organize_str = {'scalar': 'scalar', 'implicit': 'impVec', 'index': 'idxVec', 'index2d': 'idx2DVec', 'sparse': 'sparse'} #skip custom

        try:
            # initialize new stat variable with info from input string
            self.data = []
            self.lookup_tag = ''
            spec = [token.strip().lower() for token in string_spec.split(",")]
            self.lookup_tag = spec[0]
            self.type = plot_type_str[spec[1]]
            self.bool = int(spec[2])
            self.organize = organize_str[spec[3]]
            self.datatype = data_type_str[spec[4]]
           
            # guard against bogus entries
            # [한국어] 플롯 타입과 조직 방식의 조합이 유효한지 검증
            if (self.type == 1):
                assert(self.organize == 'scalar')
            elif (self.type == 2):
                assert(self.organize in ['impVec', 'idxVec'])
            elif (self.type == 3):
                assert(self.organize == 'stackbar')
            elif (self.type == 4):
                assert(self.organize == 'idx2DVec')
            elif (self.type == 5):
                assert(self.organize == 'sparse')
        except Exception as xxx_todo_changeme:
            (e) = xxx_todo_changeme
            print("Error in creating new stat variable from string: %s" % string_spec)
            raise e

    # [한국어]
    # initSparseMatrix: sparse(type==5) 변수의 데이터 구조를 초기화한다.
    # data를 [data_values, row_indices, col_indices] 형태의 3개 빈 리스트로 구성.
    def initSparseMatrix(self):
        if (self.initialized == 0):
            if (self.type != 5): raise Exception("initSparseMatrix called from wrong variable type")
            self.data = [[], [], []]
            self.initialized = 1
            self.sampleNum = 1

# [한국어] 즐겨찾기(bookmark) 클래스.
# ~/.gpgpu_sim/aerialvision/bookmarks.txt에 저장된 사용자 정의 플롯 설정을 표현.
class bookmark:

    # [한국어]
    # 생성자: 즐겨찾기 항목의 모든 필드를 빈 값으로 초기화.
    def __init__(self):
        self.title = ""
        self.fileChosen = []
        self.dataChosenX = []
        self.dataChosenY = []
        self.graphChosen = []
        self.dydx = []
        self.description = ""

# [한국어] PTX 라인 통계 파일(.stats)의 기본 컬럼명 전역 리스트.
# 'kernel line :' 헤더가 있으면 loadLineStatName()에서 오버라이드됨.
global lineStatName
lineStatName = ['count', 'latency', 'dram_traffic', 'smem_bk_conflicts', 'smem_warp', 
                'gmem_access_generated', 'gmem_warp', 'exposed_latency', 'warp_divergence', 
                'warp_issued']

# [한국어]
# loadLineStatName: PTX 라인 통계 파일의 첫 'kernel line :' 행을 읽어
# lineStatName 전역 리스트를 업데이트한다.
# @filename: .stats 파일 경로
# @return: 없음 (전역 lineStatName 수정)
def loadLineStatName(filename):
    global lineStatName
    file = open(filename, 'r')
    while file:
        line = file.readline().decode()
        if not line : break
        if (line.startswith('kernel line :')) :
            line = line.strip()
            ptxLineStatName = line.split(' ')
            ptxLineStatName = ptxLineStatName[3:]
            lineStatName = ptxLineStatName
            break


# [한국어] CUDA 소스 라인별 통계 집계 클래스.
# 하나의 CUDA 라인에 매핑된 여러 PTX 라인의 통계를 합산/최대/비율 연산한다.
class cudaLineNo:

    debug = 0
    
    # [한국어]
    # 생성자: CUDA 라인 번호에 해당하는 PTX 라인 번호 목록과
    # 각 PTX 라인의 통계 데이터를 받아 라인별 통계 맵을 구성.
    # @ptxLines: 이 CUDA 라인에 해당하는 PTX 라인 번호 리스트
    # @ptxStats: 각 PTX 라인에 대한 통계 리스트의 리스트
    def __init__(self, ptxLines, ptxStats):
        self.stats = {}
        self.ptxLines = ptxLines
        for statName in lineStatName:
            self.stats[statName] = []

        #Filling up count appropriately
        # [한국어] 모든 PTX 라인의 통계를 현재 CUDA 라인의 stats 딕셔너리에 누적.
        for iter in ptxStats:
            for statID in range(0, len(iter)):
                if (iter[statID] != "Null"):
                    self.stats[lineStatName[statID]].append(int(iter[statID]))
               
    # [한국어]
    # sum: 지정한 통계 키의 모든 PTX 라인 값을 합산.
    # @key: lineStatName에 속하는 통계 이름
    # @return: 합산된 정수 값
    def sum(self,key):
        sum = 0
        for iter in self.stats[key]:
            sum += int(iter)    
        return sum
    
    # [한국어]
    # takeMax: 지정한 통계 키의 PTX 라인 값 중 최대값.
    # @key: 통계 이름
    # @return: 최대값(데이터 없으면 0)
    def takeMax(self,key):
        try:
            tmp = max(self.stats[key])
        except:
            tmp = 0
            if cudaLineNo.debug:
                print('Exception in cudaLineNo.takeMax()', self.stats[key])
        return tmp
        
    # [한국어]
    # takeRatioSums: 두 통계 키의 합산값 비율을 반환.
    # @key1: 분자 통계 이름
    # @key2: 분모 통계 이름
    # @return: key1 합 / key2 합 (분모가 0이면 0)
    def takeRatioSums(self, key1,key2):
        tmp1 = float(self.sum(key1))
        tmp2 = float(self.sum(key2))

        try:
            return tmp1/tmp2
        except:
            if cudaLineNo.debug:
                print(tmp1, tmp2)
            if tmp2 == 0 and cudaLineNo.debug:
                print('infinite')
            return 0
    
        

# [한국어] 단일 PTX 라인 통계 클래스.
class ptxLineNo:

    debug = 0

    # [한국어]
    # 생성자: 한 PTX 라인의 통계 배열로부터 stats 딕셔너리 구성.
    # @ptxStats: lineStatName 순서대로 배염된 통계 값 리스트
    def __init__(self, ptxStats):
        self.stats = {}

        for statID in range(0, len(ptxStats)):
            self.stats[lineStatName[statID]] = int(ptxStats[statID])

    # [한국어]
    # returnStat: 지정 키의 통계 값 반환.
    def returnStat(self, key):
        return self.stats[key]
        
    # [한국어]
    # returnRatio: 두 통계 값의 비율 반환.
    # @key1: 분자 통계 이름
    # @key2: 분모 통계 이름
    # @return: key1/key2 (분모가 0이면 0)
    def returnRatio(self, key1, key2):
        tmp1 = float(self.stats[key1])
        tmp2 = float(self.stats[key2])
        try:
            return tmp1/tmp2
        except:
            if tmp2 == 0 and ptxLineNo.debug:
                print('infinite')
            return 0
            
