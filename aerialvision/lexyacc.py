#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt, Andrew Turner, Wilson W. L.
# Fung, Ali Bakhoda and the University of British Columbia, Vancouver, 
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

"""
[한국어 설명] AerialVision GPGPU-Sim 로그 파서 (lexyacc.py)

=== 파일의 역할 ===
GPGPU-Sim이 출력하는 시뮬레이션 로그 파일(gzipped 또는 plain text)을 파싱하여
AerialVision에서 그래프로 그릴 수 있는 통계 변수 딕셔너리를 생성한다.
난 내장된 변수 목록(shaderInsn, globalCycle, L1/L2 미스, DRAM 통계 등)을 포함하고,
사용자 정의 변수(variables.txt)를 추가로 로드할 수 있다.
CFLOG(Control Flow LOG) 섹션도 선택적으로 파싱한다.

=== 전체 아키텍처에서의 위치 ===
  GPGPU-Sim 시뮬레이션 출력 (stats/log 파일)
        ↓
  aerialvision/lexyacc.py (이 파일) - 로그 파싱
        ↓
  aerialvision/organizedata.py - 데이터 재배열
        ↓
  aerialvision/guiclasses.py - 시각화

=== 타 모듈과의 연결 ===
- organizedata.py: organizedata()가 반환된 variables 딕셔너리를 후처리
- guiclasses.py: formEntry/graphManager가 variables[파일명][변수명].data 접근
- variableclasses.py: variable 클래스 정의
- startup.py: skipCFLOGParsing 플래그 설정

=== 주요 함수/구조체 요약 ===
- import_user_defined_variables(variables): ~/.gpgpu_sim/aerialvision/variables.txt 로드
- parseMe(filename): 로그 파일 파싱, {이름: variable} 딕셔너리 반환
- Lex/Yacc 토큰: WORD, NUMBERSEQUENCE
"""

import os
import os.path
import sys
sys.path.insert(0,"Lib/site-packages/ply-3.2/ply-3.2")
import ply.lex as lex
import ply.yacc as yacc
import gzip
import gc

import variableclasses as vc

# [한국어] CFLOG 파싱 스킵 플래그. startup.py의 "Skip CFLog parsing" 체크박스로 제어.
global skipCFLOGParsing
skipCFLOGParsing = 0

# [한국어] AerialVision 사용자 설정 디렉토리 경로.
userSettingPath = os.path.join(os.environ['HOME'], '.gpgpu_sim', 'aerialvision')


# Import user-defined statistic variables on top of the default ones built into AerialVision
# Here is the format to specify a new variable: 
#   <name>, <plot type>, <reset at kernel launch>, <organization>, <data type>
# <plot type> can be one of: scalar, vector, stackedbar, vector2d
# <organization> can be: 
#   scalar (for scalar plot), 
#   implicit or index (for vector plot or stackedbar), 
#   index2d (for vector2d plot)
# <data type> can be: int or float

# [한국어]
# import_user_defined_variables: 사용자 설정 파일 variables.txt에서
# 추가 통계 변수 정의를 읽어 기본 variables 딕셔너리에 병합한다.
# @variables: parseMe() 내에서 생성될 variables 딕셔너리(참조)
def import_user_defined_variables(variables):
    # attempt to open the user defined variables definition file
    try:
        file = open(os.path.join(userSettingPath, 'variables.txt'),'r')
    except:
        print("No variables.txt file found.")
        return

    #this can be replaced with a proper lex-yacc parser later
    for line in file:
        try:
            # strip out trailing whitespaces and skip comment lines
            line = line.strip()
            if len(line) == 0: # skip empty line
                continue
            if line[0] == '#': # skip comment
                continue

            # parse the line containing definition of a stat variable
            s = line.split(',')
            statName = s[0]
            statVar = vc.variable('', 1, 0)
            statVar.importFromString(line)

            # add parsed stat variable to the searchable map
            variables[statName] = statVar
            
        except Exception as xxx_todo_changeme:
            (e) = xxx_todo_changeme
            print("error:",e,", in variables.txt line:",line)

# Parses through a given log file for data
# [한국어]
# parseMe: GPGPU-Sim 시뮬레이션 로그 파일을 파싱하여 통계 변수 딕셔너리를 반환.
# @filename: 로그 파일 경로 (.gz 압축도 지원)
# @return: {변수이름: variable 객체} 딕셔너리
def parseMe(filename):
    
    #The lexer
        
    # List of token names.   This is always required
    tokens = ['WORD', 
       'NUMBERSEQUENCE',
    ]
    
    # Regular expression rules for tokens
    # [한국어] Lex 토큰: WORD=메트릭 이름, NUMBERSEQUENCE=공백 구분 숫자열
    
    
    def t_WORD(t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        return t
    
    def t_NUMBERSEQUENCE(t):
        r'([-]{0,1}[0-9]+([\.][0-9]+){0,1}[ ]*)+'
        return t
    
        
    t_ignore = '[\t: ]+'
    
    def t_newline(t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")
        
    def t_error(t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1) 

    lex.lex()

    # Creating holder for CFLOG
    CFLOG = {}
    
    # Declaring the properties of supported stats in a single dictionary
    # FORMAT: <stat name in GUI>:vc.variable(<Stat Name in Log>, <type>, <reset@kernelstart>, [datatype]) 
    # [한국어] AerialVision 기본 지원 통계 변수 테이블.
    # key: GUI 표시명, value: variable(로그태그, 타입, 리셋플래그, 조직방식, 데이터타입)
    variables = {
        'shaderInsn':vc.variable('shaderinsncount', 2, 0, 'impVec'), 
        'globalInsn':vc.variable('globalinsncount', 1, 1, 'scalar'), 
        'globalCycle':vc.variable('globalcyclecount', 1, 1, 'scalar'), 
        'shaderWarpDiv':vc.variable('shaderwarpdiv', 2, 0, 'impVec'), 
        'L1TextMiss' :vc.variable('lonetexturemiss', 1, 0, 'scalar'), 
        'L1ConstMiss':vc.variable('loneconstmiss',   1, 0, 'scalar'),
        'L1ReadMiss' :vc.variable('lonereadmiss',    1, 0, 'scalar'),
        'L1WriteMiss':vc.variable('lonewritemiss',   1, 0, 'scalar'), 
        'L2ReadMiss' :vc.variable('ltworeadmiss',    1, 0, 'scalar'),
        'L2WriteMiss':vc.variable('ltwowritemiss',   1, 0, 'scalar'),
        'L2WriteHit' :vc.variable('ltwowritehit',    1, 0, 'scalar'),
        'L2ReadHit'  :vc.variable('ltworeadhit',     1, 0, 'scalar'),
        'globalTotInsn':vc.variable('globaltotinsncount', 1,0, 'scalar'), 
        'dramCMD' :vc.variable('', 2, 0, 'idxVec'),
        'dramNOP' :vc.variable('', 2, 0, 'idxVec'),
        'dramNACT':vc.variable('', 2, 0, 'idxVec'),
        'dramNPRE':vc.variable('', 2, 0, 'idxVec'),
        'dramNREQ':vc.variable('', 2, 0, 'idxVec'),
        'dramMaxMRQS':vc.variable('', 2, 0, 'idxVec'),
        'dramAveMRQS':vc.variable('', 2, 0, 'idxVec'),
        'dramUtil':vc.variable('', 2, 0, 'idxVec'),
        'dramEff' :vc.variable('', 2, 0, 'idxVec'), 
        'globalCompletedThreads':vc.variable('gpucompletedthreads', 1, 1, 'scalar'),
        'globalSentWrites':vc.variable('gpgpunsentwrites', 1, 0, 'scalar'), 
        'globalProcessedWrites':vc.variable('gpgpunprocessedwrites', 1, 0, 'scalar'), 
        'averagemflatency' :vc.variable('', 1, 0, 'custom'), 
        'LDmemlatdist':vc.variable('', 3, 0, 'stackbar'), 
        'STmemlatdist':vc.variable('', 3, 0, 'stackbar'), 
        'WarpDivergenceBreakdown':vc.variable('', 3, 0, 'stackbar'), 
        'WarpIssueSlotBreakdown':vc.variable('', 3, 0, 'stackbar'), 
        'WarpIssueDynamicIdBreakdown':vc.variable('', 3, 0, 'stackbar'), 
        'dram_writes_per_cycle':vc.variable('', 1, 0, 'scalar', float),
        'dram_reads_per_cycle' :vc.variable('', 1, 0, 'scalar', float),
        'gpu_stall_by_MSHRwb':vc.variable('', 1, 0, 'scalar'),
        'dramglobal_acc_r' :vc.variable('', 4, 0, 'idx2DVec'), 
        'dramglobal_acc_w' :vc.variable('', 4, 0, 'idx2DVec'), 
        'dramlocal_acc_r'  :vc.variable('', 4, 0, 'idx2DVec'),
        'dramlocal_acc_w'  :vc.variable('', 4, 0, 'idx2DVec'), 
        'dramconst_acc_r'  :vc.variable('', 4, 0, 'idx2DVec'), 
        'dramtexture_acc_r':vc.variable('', 4, 0, 'idx2DVec'), 
        'cacheMissRate_globalL1_all'    :vc.variable('cachemissrate_globallocall1_all',    2, 0, 'impVec', float),
        'cacheMissRate_textureL1_all'   :vc.variable('cachemissrate_texturel1_all',        2, 0, 'impVec', float),
        'cacheMissRate_constL1_all'     :vc.variable('cachemissrate_constl1_all',          2, 0, 'impVec', float),
        'cacheMissRate_globalL1_noMgHt' :vc.variable('cachemissrate_globallocall1_nomght', 2, 0, 'impVec', float),
        'cacheMissRate_textureL1_noMgHt':vc.variable('cachemissrate_texturel1_nomght',     2, 0, 'impVec', float),
        'cacheMissRate_constL1_noMgHt'  :vc.variable('cachemissrate_constl1_nomght',       2, 0, 'impVec', float),
        'shdrctacount': vc.variable('shdrctacount', 2, 0, 'impVec'),
        'CFLOG' : CFLOG 
    }

    # import user defined stat variables from variables.txt - adds on top of the defaults
    import_user_defined_variables(variables)

    # generate a lookup table based on the specified name in log file for each stat
    # [한국어] 로그 파일 내 실제 태그 이름으로 variable을 빠르게 찾기 위한 룩업 테이블 구성
    stat_lookuptable = {}
    for name, var in variables.items():
        if (name == 'CFLOG'):
            continue;
        if (var.lookup_tag != ''):
            stat_lookuptable[var.lookup_tag] = var 
        else:
            stat_lookuptable[name.lower()] = var

    inputData = 'NULL'

    # a table containing all the metrics that has received the missing data warning 
    stat_missing_warned = {}
        
    def p_sentence(p):
        '''sentence : WORD NUMBERSEQUENCE'''
        #print p[0], p[1],p[2]
        num = p[2].split(' ')  
        
        # detect empty data entry for particular metric and print a warning 
        if p[2] == '': 
            if not p[1] in stat_missing_warned: 
                print("WARNING: Sample entry for metric '%s' has no data. Skipping..." % p[1])
                stat_missing_warned[p[1]] = True
            return

        lookup_input = p[1].lower()
        if (lookup_input  in stat_lookuptable):
            if (lookup_input == "globalcyclecount") and (int(num[0]) % 10000 == 0):
                # [한국어] 글로벌 사이클 10000단위마다 진행 상황 출력
                print("Processing global cycle %s" % num[0])
                
            stat = stat_lookuptable[lookup_input]
            # [한국어] 변수 타입별로 데이터 append 방식 분기
            if (stat.type == 1):
                for x in num:
                    stat.data.append(stat.datatype(x))
                
            elif (stat.type == 2):
                for x in num:
                    stat.data.append(stat.datatype(x))
                stat.data.append("NULL")
                
            elif (stat.type == 3):
                for x in num:
                    stat.data.append(stat.datatype(x))
                stat.data.append("NULL")

            elif (stat.type == 4):
                for x in num:
                    stat.data.append(stat.datatype(x))
                stat.data.append("NULL")

            elif (stat.type == 5):
                stat.initSparseMatrix()
                for entry in num:
                    row, value = entry.split(',')
                    row = stat.datatype(row)
                    value = stat.datatype(value)
                    stat.data[0].append(value)
                    stat.data[1].append(row)
                    stat.data[2].append(stat.sampleNum)
                stat.sampleNum += 1

        elif (lookup_input[0:5] == 'cflog'):
            # [한국어] CFLOG 섹션 파싱. skipCFLOGParsing==1이면 무시.
            if (skipCFLOGParsing == 1): 
                return
            count = 0
            pc = []
            threadcount = []
            for x in num:
                if (count % 2) == 0:
                    pc.append(int(x))
                else:
                    threadcount.append(int(x))
                count += 1

            if (p[1] not in CFLOG):
                CFLOG[p[1]] = vc.variable('',2,0)
                CFLOG[p[1]].data.append([]) # pc[]
                CFLOG[p[1]].data.append([]) # threadcount[]
                CFLOG[p[1]].maxPC = 0

            CFLOG[p[1]].data[0].append(pc)
            CFLOG[p[1]].data[1].append(threadcount)
            MaxPC = max(pc)
            CFLOG[p[1]].maxPC = max(MaxPC, CFLOG[p[1]].maxPC)
        
        else:
            pass
        


    def p_error(p):
        if p:
            print(("Syntax error at '%s'" % p.value))
        else:
            print("Syntax error at EOF")
    
    yacc.yacc()

    # detect for gzip'ed log file and gunzip on the fly
    # [한국어] .gz 확장자면 gzip으로, 아니면 일반 파일로 읽기
    if (filename.endswith('.gz')):
        file = gzip.open(filename, 'r')
    else:
        file = open(filename, 'r')
    while file:
        line = file.readline().decode()

        if not line : break
        nameNdata = line.split(':')
        if (len(nameNdata) != 2): 
            print(("Syntax error at '%s'" % line)) 
        namePart = nameNdata[0].strip()
        dataPart= nameNdata[1].strip()
        parts = [' ', namePart, dataPart]
        # [한국어] yacc 대신 직접 p_sentence 호출: parts[1]=WORD, parts[2]=NUMBERSEQUENCE
        p_sentence(parts)
        # yacc.parse(line[0:-1])
    file.close()    

    return variables
  



