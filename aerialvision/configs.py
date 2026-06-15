#!/usr/bin/env python

# Copyright (C) 2009 by Wilson W. L. Fung
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
[한국어 설명] AerialVision 사용자 설정 로더 (configs.py)

=== 파일의 역할 ===
AerialVision GUI의 런타임 사용자 설정을 로드하고 접근하는 진입점이다.
$HOME/.gpgpu_sim/aerialvision/config.rc INI 형식 설정 파일을 파싱하여
폰트 크기, 눈금 회전, 색상맵 기본값 등 GUI 외관과 관련된 옵션을 제공한다.

=== 전체 아키텍처에서의 위치 ===
  aerialvision/startup.py   - 메인 GUI 초기화
        ↓
  aerialvision/configs.py   - 설정 로드 (이 파일)
        ↓
  aerialvision/guiclasses.py - PlotFormatInfo, NaviPlotInfo 등에서
                               avconfig.get_value()로 값 조회

=== 타 모듈과의 연결 ===
- guiclasses.py: PlotFormatInfo, NaviPlotInfo 생성 시 avconfig.get_value() 호출
- startup.py:    사용자 설정 경로(recentfiles.txt, bookmarks.txt)와 공유

=== 주요 함수/구조체 요약 ===
- AerialVisionConfig: SafeConfigParser 기반 설정 래퍼
  * __init__(): ~/.gpgpu_sim/aerialvision/config.rc 읽기
  * print_all(): 현재 섹션/옵션 출력
  * get_value(section, option, default): 옵션 조회, 없으면 default 반환
- avconfig: 전역 AerialVisionConfig 인스턴스
"""

import configparser, os

# [한국어] AerialVision 사용자 설정 디렉토리 경로.
# $HOME/.gpgpu_sim/aerialvision 아래에 config.rc, bookmarks.txt, recentfiles.txt 등이 위치.
userSettingPath = os.path.join(os.environ['HOME'], '.gpgpu_sim', 'aerialvision')

# Globally available configuration options for AerialVision
# [한국어] AerialVision 전역 설정 객체. 이 클래스를 통해 config.rc 값을 읽는다.
class AerialVisionConfig:

    # [한국어]
    # 생성자: ~/.gpgpu_sim/aerialvision/config.rc 파일을 읽어 설정 객체를 초기화한다.
    # 파일이 없으면 빈 설정으로 동작하며, get_value() 호출 시 default값이 반환된다.
    def __init__(self):
        self.config = configparser.SafeConfigParser()
        self.config.read( os.path.join(userSettingPath, 'config.rc') )

    # [한국어]
    # print_all: config.rc에 포함된 모든 섹션과 옵션 값을 stdout에 출력한다.
    # 단위 테스트/디버깅용 메인 진입점에서 사용.
    def print_all(self):
        for section in self.config.sections():
            for option in self.config.options(section):
                value = self.config.get(section, option)
                print("\t%s.%s = %s" % (section, option, value));

    # [한국어]
    # get_value: 지정한 섹션/옵션 값을 조회한다.
    # @section: INI 섹션 이름 (예: 'TimeLapseView', 'SourceCodeView')
    # @option:  섹션 내 옵션 이름 (예: 'labelFontSize', 'xTicksRotation')
    # @default: 설정 파일에 옵션이 없을 때 반환할 기본값
    # @return:  설정 파일의 문자열 값 또는 default
    def get_value(self, section, option, default):
        if (self.config.has_option(section, option)):
            return self.config.get(section, option)
        else:
            return default

# This is the object containing all the options
# [한국어] 모듈 수준 전역 인스턴스. 다른 AerialVision 모듈이 직접 임포트하여 사용.
avconfig = AerialVisionConfig()


#Unit test / configviewer
# [한국어] 단위 테스트: 직접 실행 시 config.rc의 모든 설정을 출력.
def main():
    print("AerialVision Options:")
    avconfig.print_all()
    print("");

if __name__ == "__main__":
    main()
