#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Wilson W. L. Fung
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
[한국어 설명] AerialVision 소스코드 뷰 PTX 통계 파서 (lexyacctexteditor.py)

=== 파일의 역할 ===
"Source Code View" 탭에서 사용하는 PTX/CUDA 소스 라인별 통계 파일(.stats)과
PTX 파일(.ptx)을 파싱한다.
- textEditorParseMe: .stats 파일에서 라인 번호별 통계 배열을 딕셔너리로 변환
- ptxToCudaMapping: .ptx 파일의 .loc 디버그 지시어를 읽어 PTX 라인→CUDA 라인 매핑 생성

=== 전체 아키텍처에서의 위치 ===
  aerialvision/guiclasses.py - newTextTab.showData()
        ↓
  aerialvision/lexyacctexteditor.py (이 파일)
        ↓
  aerialvision/variableclasses.py - cudaLineNo/ptxLineNo 클래스

=== 타 모듈과의 연결 ===
- guiclasses.py: showData()에서 textEditorParseMe, ptxToCudaMapping 호출
- organizedata.py: setCFLOGInfoFiles()로부터 PTX/stat 파일 경로 수신

=== 주요 함수/구조체 요약 ===
- textEditorParseMe(filename): .stats 파일 파싱, {라인번호: [통계값]} 반환
- ptxToCudaMapping(filename): .ptx의 .loc 정보로 {CUDA라인: [PTX라인]} 반환
"""


import sys
import re
sys.path.insert(0,"Lib/site-packages/ply-3.2/ply-3.2")
import ply.lex as lex
import ply.yacc as yacc
import variableclasses as vc

# [한국어]
# textEditorParseMe: .stats 파일을 파싱하여 {라인번호: [통계값 리스트]} 딕셔너리를 반환.
# 파일 형식: "<filename.ptx>: <라인번호>: <통계값1> <통계값2> ..."
# 예: "/path/kernel.ptx: 100: 5 120 0 ..."
def textEditorParseMe(filename):
    
    tokens = ['FILENAME', 'NUMBERSEQUENCE']
    
    def t_FILENAME(t):
        r'[a-zA-Z_/.][a-zA-Z0-9_/.]*\.ptx'
        return t

    def t_NUMBERSEQUENCE(t):
        r'[0-9 :]+'
        return t
        
    t_ignore = '\t: '
    
    def t_newline(t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")
        
    def t_error(t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)
        
    lex.lex()
    
    count = []
    latency = []
    organized = {}

    def p_sentence(p):
      '''sentence : FILENAME NUMBERSEQUENCE'''
      # [한국어] NUMBERSEQUENCE를 ':'로 분리하여 라인번호와 통계값 리스트 추출
      tmp1 = []
      tmp = p[2].split(':')
      for x in tmp:
        x = x.strip()
        tmp1.append(x)
      organized[int(tmp1[0])] = tmp1[1].split(' ')
          

    def p_error(p):
      if p:
          print(("Syntax error at '%s'" % p.value))
          print(p)
      else:
          print("Syntax error at EOF")

        
    yacc.yacc()
    
    file = open(filename, 'r')
    while file:
        line = file.readline()
        if not line : break
        if (line.startswith('kernel line :')) :
            # [한국어] 'kernel line :' 헤더 행은 컬럼명 정보이므로 파싱하지 않고 건다.
            line = line.strip()
            ptxLineStatName = line.split(' ')
            ptxLineStatName = ptxLineStatName[3:]
        else: 
            yacc.parse(line[0:-1])
        
        
    return organized
  
  
# [한국어]
# ptxToCudaMapping: PTX 파일의 .loc 지시어를 분석하여
# CUDA 소스 라인 번호 -> 해당 PTX 라인 번호 리스트 매핑을 반환.
# .loc 형식: ".loc <파일번호> <CUDA라인> <컬럼>"
# @filename: .ptx 파일 경로
# @return: {CUDA라인: [PTX라인번호, ...]}
def ptxToCudaMapping(filename):
  map = {}
  file = open(filename, 'r')
  bool = 0
  count = 0
  loc = 0
  while file:
    line = file.readline()
    if not line: break
    try:
      # [한국어] 현재 loc(CUDA 라인)에 현재 PTX 라인 번호(count)를 추가
      map[loc].append(count)
    except:
      map[loc] = []
      map[loc].append(count)

    # [한국어] .loc 지시어를 만나면 현재 CUDA 소스 라인 갱신
    m = re.search(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)', line)
    if (m != None):
      loc = int(m.group(2))

    count += 1
  x = list(map.keys())
  return map
    

#Unit test / playground
# [한국어] 단위 테스트: 명령행 인자로 .stats 파일을 받아 100번 라인 통계 출력.
def main():
    data = textEditorParseMe(sys.argv[1])
    print(data[100])
   
if __name__ == "__main__":
    main()
  
  
  
