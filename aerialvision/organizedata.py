#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt, Wilson W. L. Fung
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
[한국어 설명] AerialVision 데이터 재배열 모듈 (organizedata.py)

=== 파일의 역할 ===
lexyacc.py가 파싱한 원시 통계 데이터를 그래프 종류에 맞는 낮은 수준 구조로 변환한다.
scalar/vector/stackedbar/vector2d/sparse 등 variable.organize 방식별로 데이터를 재배열하고,
선택적으로 CFLOG(Control Flow LOG) 데이터를 PTX 라인 단위 및 CUDA 소스 라인 단위로 집계한다.

=== 전체 아키텍처에서의 위치 ===
  aerialvision/lexyacc.py - 로그 파싱 (variable.data에 원본 축적)
        ↓
  aerialvision/organizedata.py (이 파일) - 데이터 재배열
        ↓
  aerialvision/guiclasses.py - matplotlib 시각화

=== 타 모듈과의 연결 ===
- lexyacc.py: parseMe()가 반환한 variables 딕셔너리를 입력으로 사용
- lexyacctexteditor.py: CFLOG→CUDA 변환 시 ptxToCudaMapping, textEditorParseMe 호출
- variableclasses.py: variable 객체 및 lineStatName 사용
- startup.py: skipCFLog, convertCFLog2CUDAsrc 플래그 설정

=== 주요 함수/구조체 요약 ===
- organizedata(fileVars): 메인 데이터 재배열 진입점
- OrganizeScalar: 1차원 스칼라 시계열 → array
- nullOrganizedShader: NULL 구분자를 기준으로 SM별/뱅크별 2D 배열 생성
- nullOrganizedStackedBar: vector 데이터를 512개 bin으로 그룹화
- nullOrganizedDram: (memNum, value) 쌍 → 뱅크별 배열
- nullOrganizedDramV2: (chip, bank, value) 쌍 → (chip.bank) 키별 배열
- OrganizeSparse: sparse 데이터 → [data, row, col]
- CFLOGOrganizePTX/CFLOGOrganizeCuda: CFLOG 집계
"""

import os
import array
#from numpy import array
import numpy
import lexyacctexteditor
import variableclasses as vc

# [한국어] CFLOG→CUDA 소스 라인 변환 플래그. startup.py의 "Convert CFLog to CUDA source line" 체크박스로 제어.
global convertCFLog2CUDAsrc
# [한국어] CFLOG 파싱 스킵 플래그. startup.py의 "Skip CFLog parsing" 체크박스로 제어.
global skipCFLog

convertCFLog2CUDAsrc = 0
skipCFLog = 1 

# [한국어] CFLOG 관련 보조 파일 경로. setCFLOGInfoFiles()에서 설정.
CFLOGInsnInfoFile = ''
CFLOGptxFile = ''

# [한국어]
# setCFLOGInfoFiles: 소스코드 뷰 탭에서 입력된 파일 리스트로부터
# PTX 파일과 stat 파일 경로를 전역 변수에 저장한다.
# @sourceViewFileList: [CUDA소스파일목록, PTX파일목록, stat파일목록]
def setCFLOGInfoFiles(sourceViewFileList):

    global CFLOGInsnInfoFile 
    global CFLOGptxFile

    if CFLOGInsnInfoFile == '' and len(sourceViewFileList[2]) > 0:
        CFLOGInsnInfoFile = sourceViewFileList[2][0]
    if CFLOGptxFile == '' and len(sourceViewFileList[1]) > 0:
        CFLOGptxFile = sourceViewFileList[1][0]

# [한국어]
# organizedata: 파싱된 통계 변수 딕셔너리를 그래프에 적합한 형태로 재배열한다.
# @fileVars: {변수이름: variable 객체} 딕셔너리
# @return: 재배엻된 fileVars
# 
# 주요 흐름:
# 1) globalCycle을 먼저 scalar로 변환 (다른 변수의 길이 기준 참조)
# 2) 나머지 변수를 organize 방식별로 변환
# 3) averagemflatency 커스텀 처리 (globalCycle 길이에 맞춰 0 패딩)
# 4) skipCFLog==0이면 CFLOG 데이터를 PTX/CUDA 라인 단위로 집계
def organizedata(fileVars):

    # [한국어] organize 방식별 처리 함수 매핑
    organizeFunction = {
        'scalar':OrganizeScalar,        # Scalar data
        'impVec':nullOrganizedShader,   # Implicit vector data for multiple units (used by Shader Core stats)
        'stackbar':nullOrganizedStackedBar, # Stacked bars 
        'idxVec':nullOrganizedDram,     # Vector data with index  (used by DRAM stats)
        'idx2DVec':nullOrganizedDramV2, # Vector data with 2D index  (used by DRAM access stats)
        'sparse':OrganizeSparse,        # Vector data with 2D index  (used by DRAM access stats)
        'custom':0
    }
    # [한국어] array.array 타입 문자 매핑 (int->'I', float->'f')
    data_type_char = {int:'I', float:'f'}

    print("Organizing data into internal format...")

    # Organize globalCycle in advance because it is used as a reference
    # [한국어] globalCycle은 다른 변수의 길이/인덱스 기준으로 사용되므로 먼저 변환
    if ('globalCycle' in fileVars):
        statData = fileVars['globalCycle']
        fileVars['globalCycle'].data = organizeFunction[statData.organize](statData.data, data_type_char[statData.datatype])

    # Organize other stat data into internal format
    # [한국어] CFLOG와 globalCycle, custom 조직 방식을 제외한 변수들 변환
    for statName, statData in fileVars.items():
        if (statName != 'CFLOG' and statName != 'globalCycle' and statData.organize != 'custom'):
            fileVars[statName].data = organizeFunction[statData.organize](statData.data, data_type_char[statData.datatype])
  
    # Custom routines to organize stat data into internal format
    # [한국어] averagemflatency는 길이가 globalCycle보다 짧을 수 있으므로 앞에 0 패딩
    if 'averagemflatency' in fileVars:
        zeros = []
        for count in range(len(fileVars['averagemflatency'].data),len(fileVars['globalCycle'].data)):
            zeros.append(0)
        fileVars['averagemflatency'].data = zeros + fileVars['averagemflatency'].data

    # [한국어] CFLOG 파싱이 활성화된 경우 PTX/CUDA 라인 단위 집계 수행
    if (skipCFLog == 0) and 'CFLOG' in fileVars:
        ptxFile = CFLOGptxFile
        statFile = CFLOGInsnInfoFile
        
        print("PC Histogram to CUDA Src = %d" % convertCFLog2CUDAsrc)
        parseCFLOGCUDA = convertCFLog2CUDAsrc

        if parseCFLOGCUDA == 1:
            print("Obtaining PTX-to-CUDA Mapping from %s..." % ptxFile)
            map = lexyacctexteditor.ptxToCudaMapping(ptxFile.rstrip())
            print("Obtaining Program Range from %s..." % statFile)
            maxStats = max(lexyacctexteditor.textEditorParseMe(statFile.rstrip()).keys())

        if parseCFLOGCUDA == 1:
            # [한국어] {CUDA라인: [PTX라인]} -> {PTX라인: CUDA라인} 역방향 매핑 생성
            newMap = {}
            for lines in map:
                for ptxLines in map[lines]:
                    newMap[ptxLines] = lines
            print("    Total number of CUDA src lines = %s..." % len(newMap))
            
            # [한국어] stat 파일 범위를 벗어나는 PTX 라인 제거
            markForDel = []
            for ptxLines in newMap:
                if ptxLines > maxStats:
                    markForDel.append(ptxLines)
            for lines in markForDel:
                del newMap[lines]
            print("    Number of touched CUDA src lines = %s..." % len(newMap))
    
        # [한국어] 전역 CFLOG 집계 변수 초기화
        fileVars['CFLOGglobalPTX'] = vc.variable('',2,0)
        fileVars['CFLOGglobalCUDA'] = vc.variable('',2,0)
        
        count = 0
        for iter in fileVars['CFLOG']:

            print("Organizing data for %s" % iter)

            fileVars[iter + 'PTX'] = fileVars['CFLOG'][iter]
            fileVars[iter + 'PTX'].data = CFLOGOrganizePTX(fileVars['CFLOG'][iter].data, fileVars['CFLOG'][iter].maxPC)
            if parseCFLOGCUDA == 1:
                fileVars[iter + 'CUDA'] = vc.variable('',2,0)
                fileVars[iter + 'CUDA'].data = CFLOGOrganizeCuda(fileVars[iter + 'PTX'].data, newMap)

            try:
                if count == 0:
                    # [한국어] 첫 번째 CFLOG 항목을 전역 집계의 초기값으로 사용
                    fileVars['CFLOGglobalPTX'] = fileVars[iter + 'PTX']
                    if parseCFLOGCUDA == 1:
                        fileVars['CFLOGglobalCUDA'] = fileVars[iter + 'CUDA']
                else:
                    # [한국어] 이후 CFLOG 항목들을 누적 합산
                    for rows in range(0, len(fileVars[iter + 'PTX'].data)):
                        for columns in range(0, len(fileVars[iter + 'PTX'].data[rows])):
                            fileVars['CFLOGglobalPTX'].data[rows][columns] += fileVars[iter + 'PTX'].data[rows][columns]
                    if parseCFLOGCUDA == 1:
                        for rows in range(0, len(fileVars[iter + 'CUDA'].data)):
                            for columns in range(0, len(fileVars[iter + 'CUDA'].data[rows])): 
                                fileVars['CFLOGglobalCUDA'].data[rows][columns] += fileVars[iter + 'CUDA'].data[rows][columns]
            except:
                print("Error in generating globalCFLog data")

            count += 1
        del fileVars['CFLOG']


    return fileVars

# [한국어]
# OrganizeScalar: 스칼라 시계열 데이터 앞에 0을 추가하고 array.array로 변환.
# @data: 원본 스칼라 값 리스트
# @datatype_c: array.array 타입 문자 ('I' 또는 'f')
def OrganizeScalar(data, datatype_c):
    organized = [0] + data;
    organized = array.array(datatype_c, organized)
    return organized;

# [한국어]
# nullOrganizedShader: NULL 구분자를 기준으로 여러 유닛(예: SM)의 데이터를 2D 배열로 변환.
# @nullVar: 값과 NULL이 교차하는 1D 리스트
# @datatype_c: array.array 타입 문자
# @return: [유닛0값배열, 유닛1값배열, ...]
def nullOrganizedShader(nullVar, datatype_c):
    #need to organize this array into usable information
    count = 0
    organized = []
    
    #determining how many shader cores are present
    # [한국어] 마지막 NULL 뒤의 값 개수로 유닛 수 결정
    for x in reversed(nullVar):
        if x != 'NULL':
            count += 1
        elif count != 0:
            break
    numPlots = count
    count = 0
    
    #initializing 2D list
    for x in range(0, numPlots):
        organized.append(array.array(datatype_c, [0]))
    
    #filling up list appropriately
    # [한국어] NULL을 만나면 유닛 인덱스를 0으로 리셋, 아니면 해당 유닛 배열에 append
    for x in range(0,(len(nullVar))):
        if nullVar[x] == 'NULL':
            while count < numPlots:
                organized[count].append(0)
                count += 1
            count=0
        else:
            organized[count].append(nullVar[x])
            count += 1

    #for x in range(0,len(organized)):
    #    organized[x] = [0] + organized[x]
    
    return organized

# [한국어]
# nullOrganizedStackedBar: nullOrganizedShader 결과를 받아
# 데이터 포인트가 512개를 초과하면 binning으로 그룹화하여 표시 속도 향상.
def nullOrganizedStackedBar(nullVar, datatype_c):
    organized = nullOrganizedShader(nullVar, datatype_c)

    # group data points to improve display speed
    if len(organized[0]) > 512:
        n_data = len(organized[0]) // 512 + 1 
        newLen = 512
        for row in range (0,len(organized)):
            newy = array.array(datatype_c, [0 for col in range(newLen)])
            for col in range(0, len(organized[row])):
                newcol = int(col / n_data)
                newy[newcol] += organized[row][col]
            for col in range(0, len(newy)):
                newy[col] = int(newy[col]/n_data) 
            organized[row] = newy

    return organized
    
# [한국어]
# nullOrganizedDram: (memNum, value) 쌍으로 구성된 데이터를
# 메모리 파티션 번호별 배열로 변환.
def nullOrganizedDram(nullVar, datatype_c):
    organized = [array.array(datatype_c, [0])]
    mem = 1
    for iter in nullVar:
        if iter == 'NULL':
            mem = 1
            continue
        elif mem == 1:
            memNum = iter
            mem = 0
            continue
        else:
            try:
                organized[memNum].append(iter)
            except:
                organized.append(array.array(datatype_c, [0]))
                organized[memNum].append(iter)
    return organized

# [한국어]
# nullOrganizedDramV2: (chip, bank, value) 트리플릿으로 구성된 데이터를
# "chip.bank" 키별 배열 딕셔너리로 변환.
def nullOrganizedDramV2(nullVar, datatype_c):
    organized = {}
    mem = 1
    for iter in nullVar:
        if iter == 'NULL':
            mem = 1
            continue
        elif mem == 1:
            ChipNum = iter
            mem += 1
            continue
        elif mem == 2:
            BankNum = iter
            mem = 0
            continue
        else:
            try:
                key = str(ChipNum) + '.' + str(BankNum)
                organized[key].append(iter)
            except:
                organized[key] = array.array(datatype_c, [0])
                organized[key].append(iter)

    return organized

# [한국어]
# OrganizeSparse: sparse(type==5) 데이터를 [data, row, col] 형태로 재구성.
def OrganizeSparse(variable, datatype_c):
    data = numpy.array(variable[0], dtype=numpy.int32)
    row = numpy.array(variable[1], dtype=numpy.int32)
    col = numpy.array(variable[2], dtype=numpy.int32)
    del variable[0:]
    #organized = sparse.coo_matrix((data, (row, col)))
    organized = [data, row, col]

    return organized

# [한국어]
# CFLOGOrganizePTX: CFLOG 원본 데이터를 [PC][cycle] 형태의 2D 히스토그램으로 변환.
# @list: [pc리스트, threadcount리스트]
# @maxPC: 최대 PC 값
# @return: array.array('I') 2D 배열 [PC][cycle]
def CFLOGOrganizePTX(list, maxPC):
    count = 0
    
    organizedThreadCount = list[1]
    organizedPC = list[0]

    nCycles = len(organizedPC)
    final_template = [0 for cycle in range(nCycles)]
    final = [array.array('I', final_template) for pc in range(maxPC + 1)] # fill the 2D array with zeros

    # [한국어] 각 사이클의 PC 리스트를 순회하며 해당 [PC][cycle] 셀에 thread count 기록
    for cycle in range(0, nCycles):
        pcList = organizedPC[cycle]
        threadCountList = organizedThreadCount[cycle] 
        for n in range(0, len(pcList)):
            final[pcList[n]][cycle] = threadCountList[n]
    
    return final

# [한국어]
# CFLOGOrganizeCuda: PTX 라인 단위 CFLOG 데이터를 CUDA 소스 라인 단위로 합산.
# @list: CFLOGOrganizePTX 결과 [PC][cycle]
# @ptx2cudamap: {PTX라인: CUDA라인} 매핑
# @return: CUDA 라인 단위 2D 배열
def CFLOGOrganizeCuda(list, ptx2cudamap):
    #We need to aggregate lines of PTX together
    cudaMaxLineNo = max(ptx2cudamap.keys())
    tmp = {}
    #need to fill up the final matrix appropriately

    nSamples = len(list[0])

    # create a dictionary of empty data array (one array per cuda source line)
    for ptxline, cudaline in ptx2cudamap.items():
        if cudaline in tmp:
            pass
        else:
            tmp[cudaline] = [0 for lengthData in range(nSamples)]


    # [한국어] 동일 CUDA 라인에 매핑된 모든 PTX 라인의 값을 누적
    for cudaline in tmp:
        for ptxLines, mapped_cudaline in ptx2cudamap.items():
            if mapped_cudaline == cudaline:
                for lengthData in range(nSamples):
                    tmp[cudaline][lengthData] += list[ptxLines][lengthData]

    
    # [한국어] 누락된 CUDA 라인은 0으로 채워 연속된 리스트 반환
    final = []           
    for iter in range(min(tmp.keys()),max(tmp.keys())):
        if iter in tmp:
            final.append(tmp[iter])            
        else:
            final.append([0 for lengthData in range(nSamples)])

    return final
                

#def stackedBar(nullVar):
#    #Need to initialize organize ar
#    organized = [[]]
#    for iter in nullVar:
#        if iter != 'NULL':
#            organized[-1].append(iter)
#        else:
#            organized.append([])
#    organized.remove([])
#    return organized



