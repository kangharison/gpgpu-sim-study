/*****************************************************************************
 *                                McPAT
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/
/********************************************************************
 *      Modified by:
 * Jingwen Leng, University of Texas, Austin
 * Syed Gilani, University of Wisconsin–Madison
 * Tayler Hetherington, University of British Columbia
 * Ahmed ElTantawy, University of British Columbia
 * Vijay Kandiah, Northwestern University
 ********************************************************************/

#include "XML_Parse.h"
#include <stdio.h>
#include <string>
#include "xmlParser.h"

using namespace std;

/*
 * [한국어 설명] McPAT/AccelWattch XML 설정 파싱 구현 (XML_Parse.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 XML_Parse.h의 ParseXML 클래스 구현체이다.
 * xmlParser 라이브러리로 AccelWattch XML 파일을 DOM 트리로 읽은 뒤,
 * <component>/<param>/<stat> 노드를 순회하며 root_system(sys)의 각 필드를 채운다.
 * GPU 아키텍처, SM/L2/NoC/MC 파라미터, 스케일링 계수, 정적 누설 전력 계수 등을 파싱한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpgpu_sim_wrapper 생성자 → ParseXML::parse(xml_filename)
 * → XMLNode::openFileHelper()로 XML 로드
 * → recursive node traversal → sys.* 필드 채움
 * → Processor 생성자/compute()가 sys를 전력 계산에 사용
 *
 * === 타 모듈과의 연결 ===
 * 의존: xmlParser.h/cc (XML 파싱), XML_Parse.h (구조체 정의)
 * 사용: gpgpu_sim_wrapper.cc (카운터 주입), processor.h (McPAT)
 *
 * === 주요 함수/구조체 요약 ===
 * ParseXML::parse() — XML 파일을 열어 root_system을 채우는 메인 파싱 함수
 * ParseXML::initialize() — sys 구조체를 0/기본값으로 초기화
 * perf_count_label[] — 성능 카운터 이름 문자열 배열 (트레이스 파일 헤더용)
 */

const char* perf_count_label[] = {
    "TOT_INST,",      "FP_INT,",      "IC_H,",        "IC_M,",
    "DC_RH,",         "DC_RM,",       "DC_WH,",       "DC_WM,",
    "TC_H,",          "TC_M,",        "CC_H,",        "CC_M,",
    "SHRD_ACC,",      "REG_RD,",      "REG_WR,",      "NON_REG_OPs,",
    "INT_ACC,",       "FPU_ACC,",     "DPU_ACC,",     "INT_MUL24_ACC,",
    "INT_MUL32_ACC,", "INT_MUL_ACC,", "INT_DIV_ACC,", "FP_MUL_ACC,",
    "FP_DIV_ACC,",    "FP_SQRT_ACC,", "FP_LG_ACC,",   "FP_SIN_ACC,",
    "FP_EXP_ACC,",    "DP_MUL_ACC,",  "DP_DIV_ACC,",  "TENSOR_ACC,",
    "TEX_ACC,",       "MEM_RD,",      "MEM_WR,",      "MEM_PRE,",
    "L2_RH,",         "L2_RM,",       "L2_WH,",       "L2_WM,",
    "NOC_A,",         "PIPE_A,",      "IDLE_CORE_N,", "constant_power"};

/*
 * [한국어] ParseXML::parse - XML 파일 파싱 메인 함수
 *
 * @filepath 파싱할 XML 파일 경로
 * @return 없음
 *
 * 호출 체인:
 *   gpgpu_sim_wrapper() → ParseXML::parse()
 */
void ParseXML::parse(char* filepath) {
  unsigned int i, j, k, m, n;
  unsigned int NumofCom_4;
  unsigned int itmp;
  // Initialize all structures
  ParseXML::initialize();
  string strtmp;
  char chtmp[60];
  char chtmp1[60];
  chtmp1[0] = '\0';
  // this open and parse the XML file:
  XMLNode xMainNode = XMLNode::openFileHelper(
      filepath, "component");  // the 'component' in the first layer

  XMLNode xNode2 = xMainNode.getChildNode(
      "component");  // the 'component' in the second layer
  // get all params in the second layer
  itmp = xNode2.nChildNode("param");
  for (i = 0; i < itmp; i++) {
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "GPU_Architecture") == 0) {
      sys.GPU_Architecture =  // [한국어] sys.GPU_Architecture 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_of_cores") == 0) {
      sys.number_of_cores =  // [한국어] sys.number_of_cores 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "architecture") == 0) {
      sys.architecture =  // [한국어] sys.architecture 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_of_L1Directories") == 0) {
      sys.number_of_L1Directories =  // [한국어] sys.number_of_L1Directories 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_of_L2Directories") == 0) {
      sys.number_of_L2Directories =  // [한국어] sys.number_of_L2Directories 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_of_L2s") == 0) {
      sys.number_of_L2s =  // [한국어] sys.number_of_L2s 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "Private_L2") == 0) {
      sys.Private_L2 =  // [한국어] sys.Private_L2 에 정수형 XML 파라미터/속성 값 저장
          (bool)atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_of_L3s") == 0) {
      sys.number_of_L3s =  // [한국어] sys.number_of_L3s 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_of_NoCs") == 0) {
      sys.number_of_NoCs =  // [한국어] sys.number_of_NoCs 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_of_dir_levels") == 0) {
      sys.number_of_dir_levels =  // [한국어] sys.number_of_dir_levels 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "domain_size") == 0) {
      sys.domain_size =  // [한국어] sys.domain_size 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "first_level_dir") == 0) {
      sys.first_level_dir =  // [한국어] sys.first_level_dir 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "homogeneous_cores") == 0) {
      sys.homogeneous_cores =  // [한국어] sys.homogeneous_cores 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "core_tech_node") == 0) {
      sys.core_tech_node =  // [한국어] sys.core_tech_node 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "target_core_clockrate") == 0) {
      sys.target_core_clockrate =  // [한국어] sys.target_core_clockrate 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "modeled_chip_voltage_ref") == 0) {
      sys.modeled_chip_voltage_ref =  // [한국어] sys.modeled_chip_voltage_ref 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat1_flane") == 0) {
      sys.static_cat1_flane =  // [한국어] sys.static_cat1_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat2_flane") == 0) {
      sys.static_cat2_flane =  // [한국어] sys.static_cat2_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat3_flane") == 0) {
      sys.static_cat3_flane =  // [한국어] sys.static_cat3_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat4_flane") == 0) {
      sys.static_cat4_flane =  // [한국어] sys.static_cat4_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat5_flane") == 0) {
      sys.static_cat5_flane =  // [한국어] sys.static_cat5_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat6_flane") == 0) {
      sys.static_cat6_flane =  // [한국어] sys.static_cat6_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_shared_flane") == 0) {
      sys.static_shared_flane =  // [한국어] sys.static_shared_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_l1_flane") == 0) {
      sys.static_l1_flane =  // [한국어] sys.static_l1_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_l2_flane") == 0) {
      sys.static_l2_flane =  // [한국어] sys.static_l2_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_light_flane") == 0) {
      sys.static_light_flane =  // [한국어] sys.static_light_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_intadd_flane") == 0) {
      sys.static_intadd_flane =  // [한국어] sys.static_intadd_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_intmul_flane") == 0) {
      sys.static_intmul_flane =  // [한국어] sys.static_intmul_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_geomean_flane") == 0) {
      sys.static_geomean_flane =  // [한국어] sys.static_geomean_flane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat1_addlane") == 0) {
      sys.static_cat1_addlane =  // [한국어] sys.static_cat1_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat2_addlane") == 0) {
      sys.static_cat2_addlane =  // [한국어] sys.static_cat2_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat3_addlane") == 0) {
      sys.static_cat3_addlane =  // [한국어] sys.static_cat3_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat4_addlane") == 0) {
      sys.static_cat4_addlane =  // [한국어] sys.static_cat4_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat5_addlane") == 0) {
      sys.static_cat5_addlane =  // [한국어] sys.static_cat5_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_cat6_addlane") == 0) {
      sys.static_cat6_addlane =  // [한국어] sys.static_cat6_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_shared_addlane") == 0) {
      sys.static_shared_addlane =  // [한국어] sys.static_shared_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_l1_addlane") == 0) {
      sys.static_l1_addlane =  // [한국어] sys.static_l1_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_l2_addlane") == 0) {
      sys.static_l2_addlane =  // [한국어] sys.static_l2_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_light_addlane") == 0) {
      sys.static_light_addlane =  // [한국어] sys.static_light_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_intadd_addlane") == 0) {
      sys.static_intadd_addlane =  // [한국어] sys.static_intadd_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_intmul_addlane") == 0) {
      sys.static_intmul_addlane =  // [한국어] sys.static_intmul_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "static_geomean_addlane") == 0) {
      sys.static_geomean_addlane =  // [한국어] sys.static_geomean_addlane 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "target_chip_area") == 0) {
      sys.target_chip_area =  // [한국어] sys.target_chip_area 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "temperature") == 0) {
      sys.temperature =  // [한국어] sys.temperature 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "number_cache_levels") == 0) {
      sys.number_cache_levels =  // [한국어] sys.number_cache_levels 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "L1_property") == 0) {
      sys.L1_property =  // [한국어] sys.L1_property 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "L2_property") == 0) {
      sys.L2_property =  // [한국어] sys.L2_property 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "homogeneous_L2s") == 0) {
      sys.homogeneous_L2s =  // [한국어] sys.homogeneous_L2s 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "homogeneous_L1Directories") == 0) {
      sys.homogeneous_L1Directories =  // [한국어] sys.homogeneous_L1Directories 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "homogeneous_L2Directories") == 0) {
      sys.homogeneous_L2Directories =  // [한국어] sys.homogeneous_L2Directories 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "L3_property") == 0) {
      sys.L3_property =  // [한국어] sys.L3_property 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "homogeneous_L3s") == 0) {
      sys.homogeneous_L3s =  // [한국어] sys.homogeneous_L3s 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "homogeneous_ccs") == 0) {
      sys.homogeneous_ccs =  // [한국어] sys.homogeneous_ccs 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "homogeneous_NoCs") == 0) {
      sys.homogeneous_NoCs =  // [한국어] sys.homogeneous_NoCs 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "Max_area_deviation") == 0) {
      sys.Max_area_deviation =  // [한국어] sys.Max_area_deviation 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "Max_power_deviation") == 0) {
      sys.Max_power_deviation =  // [한국어] sys.Max_power_deviation 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "device_type") == 0) {
      sys.device_type =  // [한국어] sys.device_type 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "longer_channel_device") == 0) {
      sys.longer_channel_device =  // [한국어] sys.longer_channel_device 에 정수형 XML 파라미터/속성 값 저장
          (bool)atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "opt_dynamic_power") == 0) {
      sys.opt_dynamic_power =  // [한국어] sys.opt_dynamic_power 에 정수형 XML 파라미터/속성 값 저장
          (bool)atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "opt_lakage_power") == 0) {
      sys.opt_lakage_power =  // [한국어] sys.opt_lakage_power 에 정수형 XML 파라미터/속성 값 저장
          (bool)atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "opt_clockrate") == 0) {
      sys.opt_clockrate =  // [한국어] sys.opt_clockrate 에 정수형 XML 파라미터/속성 값 저장
          (bool)atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "opt_area") == 0) {
      sys.opt_area =  // [한국어] sys.opt_area 에 정수형 XML 파라미터/속성 값 저장
          (bool)atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "Embedded") == 0) {
      sys.Embedded =  // [한국어] sys.Embedded 에 정수형 XML 파라미터/속성 값 저장
          (bool)atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "interconnect_projection_type") == 0) {
      sys.interconnect_projection_type =  // [한국어] sys.interconnect_projection_type 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value")) == 0 ? 0
                                                                           : 1;
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "machine_bits") == 0) {
      sys.machine_bits =  // [한국어] sys.machine_bits 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "virtual_address_width") == 0) {
      sys.virtual_address_width =  // [한국어] sys.virtual_address_width 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "physical_address_width") == 0) {
      sys.physical_address_width =  // [한국어] sys.physical_address_width 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "virtual_memory_page_size") == 0) {
      sys.virtual_memory_page_size =  // [한국어] sys.virtual_memory_page_size 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "idle_core_power") == 0) {
      sys.idle_core_power =  // [한국어] sys.idle_core_power 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "TOT_INST") == 0) {
      sys.scaling_coefficients[TOT_INST] =  // [한국어] sys.scaling_coefficients[TOT_INST] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_INT") == 0) {
      sys.scaling_coefficients[FP_INT] =  // [한국어] sys.scaling_coefficients[FP_INT] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "IC_H") ==
        0) {
      sys.scaling_coefficients[IC_H] =  // [한국어] sys.scaling_coefficients[IC_H] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "IC_M") ==
        0) {
      sys.scaling_coefficients[IC_M] =  // [한국어] sys.scaling_coefficients[IC_M] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "DC_RH") ==
        0) {
      sys.scaling_coefficients[DC_RH] =  // [한국어] sys.scaling_coefficients[DC_RH] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "DC_RM") ==
        0) {
      sys.scaling_coefficients[DC_RM] =  // [한국어] sys.scaling_coefficients[DC_RM] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "DC_WH") ==
        0) {
      sys.scaling_coefficients[DC_WH] =  // [한국어] sys.scaling_coefficients[DC_WH] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "DC_WM") ==
        0) {
      sys.scaling_coefficients[DC_WM] =  // [한국어] sys.scaling_coefficients[DC_WM] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "TC_H") ==
        0) {
      sys.scaling_coefficients[TC_H] =  // [한국어] sys.scaling_coefficients[TC_H] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "TC_M") ==
        0) {
      sys.scaling_coefficients[TC_M] =  // [한국어] sys.scaling_coefficients[TC_M] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "CC_H") ==
        0) {
      sys.scaling_coefficients[CC_H] =  // [한국어] sys.scaling_coefficients[CC_H] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "CC_M") ==
        0) {
      sys.scaling_coefficients[CC_M] =  // [한국어] sys.scaling_coefficients[CC_M] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "SHRD_ACC") == 0) {
      sys.scaling_coefficients[SHRD_ACC] =  // [한국어] sys.scaling_coefficients[SHRD_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "REG_RD") == 0) {
      sys.scaling_coefficients[REG_RD] =  // [한국어] sys.scaling_coefficients[REG_RD] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "REG_WR") == 0) {
      sys.scaling_coefficients[REG_WR] =  // [한국어] sys.scaling_coefficients[REG_WR] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "NON_REG_OPs") == 0) {
      sys.scaling_coefficients[NON_REG_OPs] =  // [한국어] sys.scaling_coefficients[NON_REG_OPs] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "INT_ACC") == 0) {
      sys.scaling_coefficients[INT_ACC] =  // [한국어] sys.scaling_coefficients[INT_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_ACC") == 0) {
      sys.scaling_coefficients[FP_ACC] =  // [한국어] sys.scaling_coefficients[FP_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "DP_ACC") == 0) {
      sys.scaling_coefficients[DP_ACC] =  // [한국어] sys.scaling_coefficients[DP_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "INT_MUL24_ACC") == 0) {
      sys.scaling_coefficients[INT_MUL24_ACC] =  // [한국어] sys.scaling_coefficients[INT_MUL24_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "INT_MUL32_ACC") == 0) {
      sys.scaling_coefficients[INT_MUL32_ACC] =  // [한국어] sys.scaling_coefficients[INT_MUL32_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "INT_MUL_ACC") == 0) {
      sys.scaling_coefficients[INT_MUL_ACC] =  // [한국어] sys.scaling_coefficients[INT_MUL_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "INT_DIV_ACC") == 0) {
      sys.scaling_coefficients[INT_DIV_ACC] =  // [한국어] sys.scaling_coefficients[INT_DIV_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_MUL_ACC") == 0) {
      sys.scaling_coefficients[FP_MUL_ACC] =  // [한국어] sys.scaling_coefficients[FP_MUL_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_DIV_ACC") == 0) {
      sys.scaling_coefficients[FP_DIV_ACC] =  // [한국어] sys.scaling_coefficients[FP_DIV_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_SQRT_ACC") == 0) {
      sys.scaling_coefficients[FP_SQRT_ACC] =  // [한국어] sys.scaling_coefficients[FP_SQRT_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_LG_ACC") == 0) {
      sys.scaling_coefficients[FP_LG_ACC] =  // [한국어] sys.scaling_coefficients[FP_LG_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_SIN_ACC") == 0) {
      sys.scaling_coefficients[FP_SIN_ACC] =  // [한국어] sys.scaling_coefficients[FP_SIN_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "FP_EXP_ACC") == 0) {
      sys.scaling_coefficients[FP_EXP_ACC] =  // [한국어] sys.scaling_coefficients[FP_EXP_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "DP_MUL_ACC") == 0) {
      sys.scaling_coefficients[DP_MUL_ACC] =  // [한국어] sys.scaling_coefficients[DP_MUL_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "DP_DIV_ACC") == 0) {
      sys.scaling_coefficients[DP_DIV_ACC] =  // [한국어] sys.scaling_coefficients[DP_DIV_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "TENSOR_ACC") == 0) {
      sys.scaling_coefficients[TENSOR_ACC] =  // [한국어] sys.scaling_coefficients[TENSOR_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "TEX_ACC") == 0) {
      sys.scaling_coefficients[TEX_ACC] =  // [한국어] sys.scaling_coefficients[TEX_ACC] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "MEM_RD") == 0) {
      sys.scaling_coefficients[MEM_RD] =  // [한국어] sys.scaling_coefficients[MEM_RD] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "MEM_WR") == 0) {
      sys.scaling_coefficients[MEM_WR] =  // [한국어] sys.scaling_coefficients[MEM_WR] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "MEM_PRE") == 0) {
      sys.scaling_coefficients[MEM_PRE] =  // [한국어] sys.scaling_coefficients[MEM_PRE] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "L2_RH") ==
        0) {
      sys.scaling_coefficients[L2_RH] =  // [한국어] sys.scaling_coefficients[L2_RH] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "L2_RM") ==
        0) {
      sys.scaling_coefficients[L2_RM] =  // [한국어] sys.scaling_coefficients[L2_RM] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "L2_WH") ==
        0) {
      sys.scaling_coefficients[L2_WH] =  // [한국어] sys.scaling_coefficients[L2_WH] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "L2_WM") ==
        0) {
      sys.scaling_coefficients[L2_WM] =  // [한국어] sys.scaling_coefficients[L2_WM] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"), "NOC_A") ==
        0) {
      sys.scaling_coefficients[NOC_A] =  // [한국어] sys.scaling_coefficients[NOC_A] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "PIPE_A") == 0) {
      sys.scaling_coefficients[PIPE_A] =  // [한국어] sys.scaling_coefficients[PIPE_A] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "IDLE_CORE_N") == 0) {
      sys.scaling_coefficients[IDLE_CORE_N] =  // [한국어] sys.scaling_coefficients[IDLE_CORE_N] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("param", i).getAttribute("name"),
               "constant_power") == 0) {
      sys.scaling_coefficients[constant_power] =  // [한국어] sys.scaling_coefficients[constant_power] 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("param", i).getAttribute("value"));
      continue;
    }

    /*
                    if
       (strcmp(xNode2.getChildNode("param",i).getAttribute("name"),"scaling_coefficients")==0)
                    {
                            strtmp.assign(xNode2.getChildNode("param",i).getAttribute("value"));
                            m=0;
                            for(n=0; n<strtmp.length(); n++)
                            {
                                    if (strtmp[n]!=',')
                                    {
                                            sprintf(chtmp,"%c",strtmp[n]);
                                            strcat(chtmp1,chtmp);
                                    }
                                    else{
                                            sys.scaling_coefficients[m]=atof(chtmp1);  // [한국어] sys.scaling_coefficients[m] 에 XML에서 읽은 값 저장
                                            m++;
                                            chtmp1[0]='\0';
                                    }
                            }
                            sys.scaling_coefficients[m]=atof(chtmp1);  // [한국어] sys.scaling_coefficients[m] 에 XML에서 읽은 값 저장
                            m++;
                            chtmp1[0]='\0';
                            continue;
                    }
    */
  }

  //	if (sys.Private_L2 && sys.number_of_cores!=sys.number_of_L2s)
  //	{
  //		cout<<"Private L2: Number of L2s must equal to Number of
  // Cores"<<endl; 		exit(0);
  //	}

  itmp = xNode2.nChildNode("stat");
  for (i = 0; i < itmp; i++) {
    if (strcmp(xNode2.getChildNode("stat", i).getAttribute("name"),
               "total_cycles") == 0) {
      sys.total_cycles =  // [한국어] sys.total_cycles 에 실수형 XML 파라미터/속성 값 저장
          atof(xNode2.getChildNode("stat", i).getAttribute("value"));
      continue;
    }
    if (strcmp(xNode2.getChildNode("stat", i).getAttribute("name"),
               "num_idle_cores") == 0) {
      sys.num_idle_cores =  // [한국어] sys.num_idle_cores 에 정수형 XML 파라미터/속성 값 저장
          atoi(xNode2.getChildNode("stat", i).getAttribute("value"));
      continue;
    }
  }

  // get the number of components within the second layer
  unsigned int NumofCom_3 = xNode2.nChildNode("component");
  XMLNode xNode3, xNode4;  // define the third-layer(system.core0) and
                           // fourth-layer(system.core0.predictor) xnodes

  unsigned int OrderofComponents_3layer = 0;
  if (NumofCom_3 > OrderofComponents_3layer) {
    //___________________________get all
    // system.core0-n________________________________________________
    if (sys.homogeneous_cores == 1)
      OrderofComponents_3layer = 0;
    else
      OrderofComponents_3layer = sys.number_of_cores - 1;
    for (i = 0; i <= OrderofComponents_3layer; i++) {
      xNode3 = xNode2.getChildNode("component", i);
      if (xNode3.isEmpty() == 1) {
        printf(
            "The value of homogeneous_cores or number_of_cores is not "
            "correct!");
        exit(0);
      } else {
        if (strstr(xNode3.getAttribute("name"), "core") != NULL) {
          {  // For cpu0-cpui
            // Get all params with system.core?
            itmp = xNode3.nChildNode("param");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "clock_rate") == 0) {
                sys.core[i].clock_rate =  // [한국어] sys.core[i].clock_rate 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "opt_local") == 0) {
                sys.core[i].opt_local = (bool)atoi(  // [한국어] sys.core[i].opt_local 에 XML 노드의 속성/자식 값을 읽어 저장
                    xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "x86") == 0) {
                sys.core[i].x86 = (bool)atoi(  // [한국어] sys.core[i].x86 에 XML 노드의 속성/자식 값을 읽어 저장
                    xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "machine_bits") == 0) {
                sys.core[i].machine_bits =  // [한국어] sys.core[i].machine_bits 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "virtual_address_width") == 0) {
                sys.core[i].virtual_address_width =  // [한국어] sys.core[i].virtual_address_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "physical_address_width") == 0) {
                sys.core[i].physical_address_width =  // [한국어] sys.core[i].physical_address_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "instruction_length") == 0) {
                sys.core[i].instruction_length =  // [한국어] sys.core[i].instruction_length 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "opcode_width") == 0) {
                sys.core[i].opcode_width =  // [한국어] sys.core[i].opcode_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "micro_opcode_width") == 0) {
                sys.core[i].micro_opcode_width =  // [한국어] sys.core[i].micro_opcode_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "machine_type") == 0) {
                sys.core[i].machine_type =  // [한국어] sys.core[i].machine_type 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "internal_datapath_width") == 0) {
                sys.core[i].internal_datapath_width =  // [한국어] sys.core[i].internal_datapath_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "number_hardware_threads") == 0) {
                sys.core[i].number_hardware_threads =  // [한국어] sys.core[i].number_hardware_threads 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "fetch_width") == 0) {
                sys.core[i].fetch_width =  // [한국어] sys.core[i].fetch_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "number_instruction_fetch_ports") == 0) {
                sys.core[i].number_instruction_fetch_ports =  // [한국어] sys.core[i].number_instruction_fetch_ports 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "decode_width") == 0) {
                sys.core[i].decode_width =  // [한국어] sys.core[i].decode_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "issue_width") == 0) {
                sys.core[i].issue_width =  // [한국어] sys.core[i].issue_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "peak_issue_width") == 0) {
                sys.core[i].peak_issue_width =  // [한국어] sys.core[i].peak_issue_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "commit_width") == 0) {
                sys.core[i].commit_width =  // [한국어] sys.core[i].commit_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "fp_issue_width") == 0) {
                sys.core[i].fp_issue_width =  // [한국어] sys.core[i].fp_issue_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "prediction_width") == 0) {
                sys.core[i].prediction_width =  // [한국어] sys.core[i].prediction_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }

              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "pipelines_per_core") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.core[i].pipelines_per_core[m] = atoi(chtmp1);  // [한국어] sys.core[i].pipelines_per_core[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.core[i].pipelines_per_core[m] = atoi(chtmp1);  // [한국어] sys.core[i].pipelines_per_core[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "pipeline_depth") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.core[i].pipeline_depth[m] = atoi(chtmp1);  // [한국어] sys.core[i].pipeline_depth[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.core[i].pipeline_depth[m] = atoi(chtmp1);  // [한국어] sys.core[i].pipeline_depth[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }

              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "FPU") == 0) {
                strcpy(sys.core[i].FPU,
                       xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "divider_multiplier") == 0) {
                strcpy(sys.core[i].divider_multiplier,
                       xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "ALU_per_core") == 0) {
                sys.core[i].ALU_per_core =  // [한국어] sys.core[i].ALU_per_core 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "FPU_per_core") == 0) {
                sys.core[i].FPU_per_core =  // [한국어] sys.core[i].FPU_per_core 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "MUL_per_core") == 0) {
                sys.core[i].MUL_per_core =  // [한국어] sys.core[i].MUL_per_core 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "instruction_buffer_size") == 0) {
                sys.core[i].instruction_buffer_size =  // [한국어] sys.core[i].instruction_buffer_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "decoded_stream_buffer_size") == 0) {
                sys.core[i].decoded_stream_buffer_size =  // [한국어] sys.core[i].decoded_stream_buffer_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "instruction_window_scheme") == 0) {
                sys.core[i].instruction_window_scheme =  // [한국어] sys.core[i].instruction_window_scheme 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "instruction_window_size") == 0) {
                sys.core[i].instruction_window_size =  // [한국어] sys.core[i].instruction_window_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "fp_instruction_window_size") == 0) {
                sys.core[i].fp_instruction_window_size =  // [한국어] sys.core[i].fp_instruction_window_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "ROB_size") == 0) {
                sys.core[i].ROB_size =  // [한국어] sys.core[i].ROB_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "rf_banks") == 0) {
                sys.core[i].rf_banks =  // [한국어] sys.core[i].rf_banks 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "simd_width") == 0) {
                sys.core[i].simd_width =  // [한국어] sys.core[i].simd_width 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "collector_units") == 0) {
                sys.core[i].collector_units =  // [한국어] sys.core[i].collector_units 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "core_clock_ratio") == 0) {
                sys.core[i].core_clock_ratio =  // [한국어] sys.core[i].core_clock_ratio 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "warp_size") == 0) {
                sys.core[i].warp_size =  // [한국어] sys.core[i].warp_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }

              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "archi_Regs_IRF_size") == 0) {
                sys.core[i].archi_Regs_IRF_size =  // [한국어] sys.core[i].archi_Regs_IRF_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "archi_Regs_FRF_size") == 0) {
                sys.core[i].archi_Regs_FRF_size =  // [한국어] sys.core[i].archi_Regs_FRF_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "phy_Regs_IRF_size") == 0) {
                sys.core[i].phy_Regs_IRF_size =  // [한국어] sys.core[i].phy_Regs_IRF_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "phy_Regs_FRF_size") == 0) {
                sys.core[i].phy_Regs_FRF_size =  // [한국어] sys.core[i].phy_Regs_FRF_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "rename_scheme") == 0) {
                sys.core[i].rename_scheme =  // [한국어] sys.core[i].rename_scheme 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "register_windows_size") == 0) {
                sys.core[i].register_windows_size =  // [한국어] sys.core[i].register_windows_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "LSU_order") == 0) {
                strcpy(sys.core[i].LSU_order,
                       xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "store_buffer_size") == 0) {
                sys.core[i].store_buffer_size =  // [한국어] sys.core[i].store_buffer_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "load_buffer_size") == 0) {
                sys.core[i].load_buffer_size =  // [한국어] sys.core[i].load_buffer_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "memory_ports") == 0) {
                sys.core[i].memory_ports =  // [한국어] sys.core[i].memory_ports 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "Dcache_dual_pump") == 0) {
                strcpy(sys.core[i].Dcache_dual_pump,
                       xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "RAS_size") == 0) {
                sys.core[i].RAS_size =  // [한국어] sys.core[i].RAS_size 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
            }
            // Get all stats with system.core?
            itmp = xNode3.nChildNode("stat");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_instructions") == 0) {
                sys.core[i].total_instructions =  // [한국어] sys.core[i].total_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "int_instructions") == 0) {
                sys.core[i].int_instructions =  // [한국어] sys.core[i].int_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fp_instructions") == 0) {
                sys.core[i].fp_instructions =  // [한국어] sys.core[i].fp_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "branch_instructions") == 0) {
                sys.core[i].branch_instructions =  // [한국어] sys.core[i].branch_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "branch_mispredictions") == 0) {
                sys.core[i].branch_mispredictions =  // [한국어] sys.core[i].branch_mispredictions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "committed_instructions") == 0) {
                sys.core[i].committed_instructions =  // [한국어] sys.core[i].committed_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "committed_int_instructions") == 0) {
                sys.core[i].committed_int_instructions =  // [한국어] sys.core[i].committed_int_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "committed_fp_instructions") == 0) {
                sys.core[i].committed_fp_instructions =  // [한국어] sys.core[i].committed_fp_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "load_instructions") == 0) {
                sys.core[i].load_instructions =  // [한국어] sys.core[i].load_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "store_instructions") == 0) {
                sys.core[i].store_instructions =  // [한국어] sys.core[i].store_instructions 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_cycles") == 0) {
                sys.core[i].total_cycles =  // [한국어] sys.core[i].total_cycles 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "idle_cycles") == 0) {
                sys.core[i].idle_cycles =  // [한국어] sys.core[i].idle_cycles 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "busy_cycles") == 0) {
                sys.core[i].busy_cycles =  // [한국어] sys.core[i].busy_cycles 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "instruction_buffer_reads") == 0) {
                sys.core[i].instruction_buffer_reads =  // [한국어] sys.core[i].instruction_buffer_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "instruction_buffer_write") == 0) {
                sys.core[i].instruction_buffer_write =  // [한국어] sys.core[i].instruction_buffer_write 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "ROB_reads") == 0) {
                sys.core[i].ROB_reads =  // [한국어] sys.core[i].ROB_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "ROB_writes") == 0) {
                sys.core[i].ROB_writes =  // [한국어] sys.core[i].ROB_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "rename_reads") == 0) {
                sys.core[i].rename_reads =  // [한국어] sys.core[i].rename_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "rename_writes") == 0) {
                sys.core[i].rename_writes =  // [한국어] sys.core[i].rename_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fp_rename_reads") == 0) {
                sys.core[i].fp_rename_reads =  // [한국어] sys.core[i].fp_rename_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fp_rename_writes") == 0) {
                sys.core[i].fp_rename_writes =  // [한국어] sys.core[i].fp_rename_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "inst_window_reads") == 0) {
                sys.core[i].inst_window_reads =  // [한국어] sys.core[i].inst_window_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "inst_window_writes") == 0) {
                sys.core[i].inst_window_writes =  // [한국어] sys.core[i].inst_window_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "inst_window_wakeup_accesses") == 0) {
                sys.core[i].inst_window_wakeup_accesses =  // [한국어] sys.core[i].inst_window_wakeup_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "inst_window_selections") == 0) {
                sys.core[i].inst_window_selections =  // [한국어] sys.core[i].inst_window_selections 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fp_inst_window_reads") == 0) {
                sys.core[i].fp_inst_window_reads =  // [한국어] sys.core[i].fp_inst_window_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fp_inst_window_writes") == 0) {
                sys.core[i].fp_inst_window_writes =  // [한국어] sys.core[i].fp_inst_window_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fp_inst_window_wakeup_accesses") == 0) {
                sys.core[i].fp_inst_window_wakeup_accesses =  // [한국어] sys.core[i].fp_inst_window_wakeup_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "archi_int_regfile_reads") == 0) {
                sys.core[i].archi_int_regfile_reads =  // [한국어] sys.core[i].archi_int_regfile_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "archi_float_regfile_reads") == 0) {
                sys.core[i].archi_float_regfile_reads =  // [한국어] sys.core[i].archi_float_regfile_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "phy_int_regfile_reads") == 0) {
                sys.core[i].phy_int_regfile_reads =  // [한국어] sys.core[i].phy_int_regfile_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "phy_float_regfile_reads") == 0) {
                sys.core[i].phy_float_regfile_reads =  // [한국어] sys.core[i].phy_float_regfile_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "phy_int_regfile_writes") == 0) {
                sys.core[i].archi_int_regfile_writes =  // [한국어] sys.core[i].archi_int_regfile_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "phy_float_regfile_writes") == 0) {
                sys.core[i].archi_float_regfile_writes =  // [한국어] sys.core[i].archi_float_regfile_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "archi_int_regfile_writes") == 0) {
                sys.core[i].phy_int_regfile_writes =  // [한국어] sys.core[i].phy_int_regfile_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "archi_float_regfile_writes") == 0) {
                sys.core[i].phy_float_regfile_writes =  // [한국어] sys.core[i].phy_float_regfile_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "int_regfile_reads") == 0) {
                sys.core[i].int_regfile_reads =  // [한국어] sys.core[i].int_regfile_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "float_regfile_reads") == 0) {
                sys.core[i].float_regfile_reads =  // [한국어] sys.core[i].float_regfile_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "int_regfile_writes") == 0) {
                sys.core[i].int_regfile_writes =  // [한국어] sys.core[i].int_regfile_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "float_regfile_writes") == 0) {
                sys.core[i].float_regfile_writes =  // [한국어] sys.core[i].float_regfile_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "non_rf_operands") == 0) {
                sys.core[i].non_rf_operands =  // [한국어] sys.core[i].non_rf_operands 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }

              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "windowed_reg_accesses") == 0) {
                sys.core[i].windowed_reg_accesses =  // [한국어] sys.core[i].windowed_reg_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "windowed_reg_transports") == 0) {
                sys.core[i].windowed_reg_transports =  // [한국어] sys.core[i].windowed_reg_transports 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "function_calls") == 0) {
                sys.core[i].function_calls =  // [한국어] sys.core[i].function_calls 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "context_switches") == 0) {
                sys.core[i].context_switches =  // [한국어] sys.core[i].context_switches 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "ialu_accesses") == 0) {
                sys.core[i].ialu_accesses =  // [한국어] sys.core[i].ialu_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fpu_accesses") == 0) {
                sys.core[i].fpu_accesses =  // [한국어] sys.core[i].fpu_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "mul_accesses") == 0) {
                sys.core[i].mul_accesses =  // [한국어] sys.core[i].mul_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "cdb_alu_accesses") == 0) {
                sys.core[i].cdb_alu_accesses =  // [한국어] sys.core[i].cdb_alu_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "cdb_mul_accesses") == 0) {
                sys.core[i].cdb_mul_accesses =  // [한국어] sys.core[i].cdb_mul_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "cdb_fpu_accesses") == 0) {
                sys.core[i].cdb_fpu_accesses =  // [한국어] sys.core[i].cdb_fpu_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "load_buffer_reads") == 0) {
                sys.core[i].load_buffer_reads =  // [한국어] sys.core[i].load_buffer_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "load_buffer_writes") == 0) {
                sys.core[i].load_buffer_writes =  // [한국어] sys.core[i].load_buffer_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "load_buffer_cams") == 0) {
                sys.core[i].load_buffer_cams =  // [한국어] sys.core[i].load_buffer_cams 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "store_buffer_reads") == 0) {
                sys.core[i].store_buffer_reads =  // [한국어] sys.core[i].store_buffer_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "store_buffer_writes") == 0) {
                sys.core[i].store_buffer_writes =  // [한국어] sys.core[i].store_buffer_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "store_buffer_cams") == 0) {
                sys.core[i].store_buffer_cams =  // [한국어] sys.core[i].store_buffer_cams 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "store_buffer_forwards") == 0) {
                sys.core[i].store_buffer_forwards =  // [한국어] sys.core[i].store_buffer_forwards 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "main_memory_access") == 0) {
                sys.core[i].main_memory_access =  // [한국어] sys.core[i].main_memory_access 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "main_memory_read") == 0) {
                sys.core[i].main_memory_read =  // [한국어] sys.core[i].main_memory_read 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "main_memory_write") == 0) {
                sys.core[i].main_memory_write =  // [한국어] sys.core[i].main_memory_write 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "pipeline_duty_cycle") == 0) {
                sys.core[i].pipeline_duty_cycle =  // [한국어] sys.core[i].pipeline_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }

              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "IFU_duty_cycle") == 0) {
                sys.core[i].IFU_duty_cycle =  // [한국어] sys.core[i].IFU_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "BR_duty_cycle") == 0) {
                sys.core[i].BR_duty_cycle =  // [한국어] sys.core[i].BR_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "LSU_duty_cycle") == 0) {
                sys.core[i].LSU_duty_cycle =  // [한국어] sys.core[i].LSU_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "MemManU_I_duty_cycle") == 0) {
                sys.core[i].MemManU_I_duty_cycle =  // [한국어] sys.core[i].MemManU_I_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "MemManU_D_duty_cycle") == 0) {
                sys.core[i].MemManU_D_duty_cycle =  // [한국어] sys.core[i].MemManU_D_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "ALU_duty_cycle") == 0) {
                sys.core[i].ALU_duty_cycle =  // [한국어] sys.core[i].ALU_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "MUL_duty_cycle") == 0) {
                sys.core[i].MUL_duty_cycle =  // [한국어] sys.core[i].MUL_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "FPU_duty_cycle") == 0) {
                sys.core[i].FPU_duty_cycle =  // [한국어] sys.core[i].FPU_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "ALU_cdb_duty_cycle") == 0) {
                sys.core[i].ALU_cdb_duty_cycle =  // [한국어] sys.core[i].ALU_cdb_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "MUL_cdb_duty_cycle") == 0) {
                sys.core[i].MUL_cdb_duty_cycle =  // [한국어] sys.core[i].MUL_cdb_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "FPU_cdb_duty_cycle") == 0) {
                sys.core[i].FPU_cdb_duty_cycle =  // [한국어] sys.core[i].FPU_cdb_duty_cycle 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
            }
          }

          NumofCom_4 =
              xNode3.nChildNode("component");  // get the number of components
                                               // within the third layer
          for (j = 0; j < NumofCom_4; j++) {
            xNode4 = xNode3.getChildNode("component", j);
            if (strcmp(xNode4.getAttribute("name"), "PBT") == 0) {  // find PBT
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp; k++) {  // get all items of param in
                                            // system.core0.predictor--PBT
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "prediction_width") == 0) {
                  sys.core[i].predictor.prediction_width = atoi(  // [한국어] sys.core[i].predictor.prediction_width 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "prediction_scheme") == 0) {
                  strcpy(sys.core[i].predictor.prediction_scheme,
                         xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "predictor_size") == 0) {
                  sys.core[i].predictor.predictor_size = atoi(  // [한국어] sys.core[i].predictor.predictor_size 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "predictor_entries") == 0) {
                  sys.core[i].predictor.predictor_entries = atoi(  // [한국어] sys.core[i].predictor.predictor_entries 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "local_predictor_size") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].predictor.local_predictor_size[m] =  // [한국어] sys.core[i].predictor.local_predictor_size[m] 에 정수형 XML 파라미터/속성 값 저장
                          atoi(chtmp1);
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].predictor.local_predictor_size[m] = atoi(chtmp1);  // [한국어] sys.core[i].predictor.local_predictor_size[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "local_predictor_entries") == 0) {
                  sys.core[i].predictor.local_predictor_entries = atoi(  // [한국어] sys.core[i].predictor.local_predictor_entries 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "global_predictor_entries") == 0) {
                  sys.core[i].predictor.global_predictor_entries = atoi(  // [한국어] sys.core[i].predictor.global_predictor_entries 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "global_predictor_bits") == 0) {
                  sys.core[i].predictor.global_predictor_bits = atoi(  // [한국어] sys.core[i].predictor.global_predictor_bits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "chooser_predictor_entries") == 0) {
                  sys.core[i].predictor.chooser_predictor_entries = atoi(  // [한국어] sys.core[i].predictor.chooser_predictor_entries 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "chooser_predictor_bits") == 0) {
                  sys.core[i].predictor.chooser_predictor_bits = atoi(  // [한국어] sys.core[i].predictor.chooser_predictor_bits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  continue;
                }
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {  // get all items of stat in
                                            // system.core0.predictor--PBT
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "predictor_accesses") == 0)
                  sys.core[i].predictor.predictor_accesses = atof(  // [한국어] sys.core[i].predictor.predictor_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
              }
            }
            if (strcmp(xNode4.getAttribute("name"), "itlb") ==
                0) {  // find system.core0.itlb
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp;
                   k++) {  // get all items of param in system.core0.itlb--itlb
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "number_entries") == 0)
                  sys.core[i].itlb.number_entries = atoi(  // [한국어] sys.core[i].itlb.number_entries 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {  // get all items of stat in itlb
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].itlb.total_hits = atof(  // [한국어] sys.core[i].itlb.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].itlb.total_accesses = atof(  // [한국어] sys.core[i].itlb.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].itlb.total_misses = atof(  // [한국어] sys.core[i].itlb.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "conflicts") == 0) {
                  sys.core[i].itlb.conflicts = atof(  // [한국어] sys.core[i].itlb.conflicts 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }
            if (strcmp(xNode4.getAttribute("name"), "icache") ==
                0) {  // find system.core0.icache
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp; k++) {  // get all items of param in
                                            // system.core0.icache--icache
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "icache_config") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].icache.icache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].icache.icache_config[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].icache.icache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].icache.icache_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "buffer_sizes") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].icache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].icache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].icache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].icache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].icache.total_accesses = atof(  // [한국어] sys.core[i].icache.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_accesses") == 0) {
                  sys.core[i].icache.read_accesses = atof(  // [한국어] sys.core[i].icache.read_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_misses") == 0) {
                  sys.core[i].icache.read_misses = atof(  // [한국어] sys.core[i].icache.read_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "replacements") == 0) {
                  sys.core[i].icache.replacements = atof(  // [한국어] sys.core[i].icache.replacements 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_hits") == 0) {
                  sys.core[i].icache.read_hits = atof(  // [한국어] sys.core[i].icache.read_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].icache.total_hits = atof(  // [한국어] sys.core[i].icache.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].icache.total_misses = atof(  // [한국어] sys.core[i].icache.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "miss_buffer_access") == 0) {
                  sys.core[i].icache.miss_buffer_access = atof(  // [한국어] sys.core[i].icache.miss_buffer_access 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "fill_buffer_accesses") == 0) {
                  sys.core[i].icache.fill_buffer_accesses = atof(  // [한국어] sys.core[i].icache.fill_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_accesses") == 0) {
                  sys.core[i].icache.prefetch_buffer_accesses = atof(  // [한국어] sys.core[i].icache.prefetch_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_writes") == 0) {
                  sys.core[i].icache.prefetch_buffer_writes = atof(  // [한국어] sys.core[i].icache.prefetch_buffer_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_reads") == 0) {
                  sys.core[i].icache.prefetch_buffer_reads = atof(  // [한국어] sys.core[i].icache.prefetch_buffer_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_hits") == 0) {
                  sys.core[i].icache.prefetch_buffer_hits = atof(  // [한국어] sys.core[i].icache.prefetch_buffer_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "conflicts") == 0) {
                  sys.core[i].icache.conflicts = atof(  // [한국어] sys.core[i].icache.conflicts 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }
            if (strcmp(xNode4.getAttribute("name"), "dtlb") ==
                0) {  // find system.core0.dtlb
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp;
                   k++) {  // get all items of param in system.core0.dtlb--dtlb
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "number_entries") == 0)
                  sys.core[i].dtlb.number_entries = atoi(  // [한국어] sys.core[i].dtlb.number_entries 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("param", k).getAttribute("value"));
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {  // get all items of stat in dtlb
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].dtlb.total_accesses = atof(  // [한국어] sys.core[i].dtlb.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_accesses") == 0) {
                  sys.core[i].dtlb.read_accesses = atof(  // [한국어] sys.core[i].dtlb.read_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_accesses") == 0) {
                  sys.core[i].dtlb.write_accesses = atof(  // [한국어] sys.core[i].dtlb.write_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_hits") == 0) {
                  sys.core[i].dtlb.read_hits = atof(  // [한국어] sys.core[i].dtlb.read_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_hits") == 0) {
                  sys.core[i].dtlb.write_hits = atof(  // [한국어] sys.core[i].dtlb.write_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_misses") == 0) {
                  sys.core[i].dtlb.read_misses = atof(  // [한국어] sys.core[i].dtlb.read_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_misses") == 0) {
                  sys.core[i].dtlb.write_misses = atof(  // [한국어] sys.core[i].dtlb.write_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].dtlb.total_hits = atof(  // [한국어] sys.core[i].dtlb.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].dtlb.total_misses = atof(  // [한국어] sys.core[i].dtlb.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "conflicts") == 0) {
                  sys.core[i].dtlb.conflicts = atof(  // [한국어] sys.core[i].dtlb.conflicts 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }

            // Added by Jingwen
            if (strcmp(xNode4.getAttribute("name"), "ccache") ==
                0) {  // find system.core0.ccache
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp; k++) {  // get all items of param in
                                            // system.core0.ccache--ccache
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "ccache_config") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].ccache.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].ccache.dcache_config[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].ccache.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].ccache.dcache_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "buffer_sizes") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].ccache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].ccache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].ccache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].ccache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {  // get all items of stat in ccache
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].ccache.total_accesses = atof(  // [한국어] sys.core[i].ccache.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_accesses") == 0) {
                  sys.core[i].ccache.read_accesses = atof(  // [한국어] sys.core[i].ccache.read_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_accesses") == 0) {
                  sys.core[i].ccache.write_accesses = atof(  // [한국어] sys.core[i].ccache.write_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].ccache.total_hits = atof(  // [한국어] sys.core[i].ccache.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].ccache.total_misses = atof(  // [한국어] sys.core[i].ccache.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_hits") == 0) {
                  sys.core[i].ccache.read_hits = atof(  // [한국어] sys.core[i].ccache.read_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_hits") == 0) {
                  sys.core[i].ccache.write_hits = atof(  // [한국어] sys.core[i].ccache.write_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_misses") == 0) {
                  sys.core[i].ccache.read_misses = atof(  // [한국어] sys.core[i].ccache.read_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_misses") == 0) {
                  sys.core[i].ccache.write_misses = atof(  // [한국어] sys.core[i].ccache.write_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "replacements") == 0) {
                  sys.core[i].ccache.replacements = atof(  // [한국어] sys.core[i].ccache.replacements 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_backs") == 0) {
                  sys.core[i].ccache.write_backs = atof(  // [한국어] sys.core[i].ccache.write_backs 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "miss_buffer_access") == 0) {
                  sys.core[i].ccache.miss_buffer_access = atof(  // [한국어] sys.core[i].ccache.miss_buffer_access 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "fill_buffer_accesses") == 0) {
                  sys.core[i].ccache.fill_buffer_accesses = atof(  // [한국어] sys.core[i].ccache.fill_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_accesses") == 0) {
                  sys.core[i].ccache.prefetch_buffer_accesses = atof(  // [한국어] sys.core[i].ccache.prefetch_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_writes") == 0) {
                  sys.core[i].ccache.prefetch_buffer_writes = atof(  // [한국어] sys.core[i].ccache.prefetch_buffer_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_reads") == 0) {
                  sys.core[i].ccache.prefetch_buffer_reads = atof(  // [한국어] sys.core[i].ccache.prefetch_buffer_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_hits") == 0) {
                  sys.core[i].ccache.prefetch_buffer_hits = atof(  // [한국어] sys.core[i].ccache.prefetch_buffer_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_writes") == 0) {
                  sys.core[i].ccache.wbb_writes = atof(  // [한국어] sys.core[i].ccache.wbb_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_reads") == 0) {
                  sys.core[i].ccache.wbb_reads = atof(  // [한국어] sys.core[i].ccache.wbb_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "conflicts") == 0) {
                  sys.core[i].ccache.conflicts = atof(  // [한국어] sys.core[i].ccache.conflicts 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }

            // tcache
            if (strcmp(xNode4.getAttribute("name"), "tcache") ==
                0) {  // find system.core0.tcache
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp; k++) {  // get all items of param in
                                            // system.core0.tcache--tcache
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "tcache_config") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].tcache.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].tcache.dcache_config[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].tcache.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].tcache.dcache_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "buffer_sizes") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].tcache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].tcache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].tcache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].tcache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {  // get all items of stat in tcache
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].tcache.total_accesses = atof(  // [한국어] sys.core[i].tcache.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_accesses") == 0) {
                  sys.core[i].tcache.read_accesses = atof(  // [한국어] sys.core[i].tcache.read_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_accesses") == 0) {
                  sys.core[i].tcache.write_accesses = atof(  // [한국어] sys.core[i].tcache.write_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].tcache.total_hits = atof(  // [한국어] sys.core[i].tcache.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].tcache.total_misses = atof(  // [한국어] sys.core[i].tcache.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_hits") == 0) {
                  sys.core[i].tcache.read_hits = atof(  // [한국어] sys.core[i].tcache.read_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_hits") == 0) {
                  sys.core[i].tcache.write_hits = atof(  // [한국어] sys.core[i].tcache.write_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_misses") == 0) {
                  sys.core[i].tcache.read_misses = atof(  // [한국어] sys.core[i].tcache.read_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_misses") == 0) {
                  sys.core[i].tcache.write_misses = atof(  // [한국어] sys.core[i].tcache.write_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "replacements") == 0) {
                  sys.core[i].tcache.replacements = atof(  // [한국어] sys.core[i].tcache.replacements 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_backs") == 0) {
                  sys.core[i].tcache.write_backs = atof(  // [한국어] sys.core[i].tcache.write_backs 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "miss_buffer_access") == 0) {
                  sys.core[i].tcache.miss_buffer_access = atof(  // [한국어] sys.core[i].tcache.miss_buffer_access 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "fill_buffer_accesses") == 0) {
                  sys.core[i].tcache.fill_buffer_accesses = atof(  // [한국어] sys.core[i].tcache.fill_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_accesses") == 0) {
                  sys.core[i].tcache.prefetch_buffer_accesses = atof(  // [한국어] sys.core[i].tcache.prefetch_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_writes") == 0) {
                  sys.core[i].tcache.prefetch_buffer_writes = atof(  // [한국어] sys.core[i].tcache.prefetch_buffer_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_reads") == 0) {
                  sys.core[i].tcache.prefetch_buffer_reads = atof(  // [한국어] sys.core[i].tcache.prefetch_buffer_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_hits") == 0) {
                  sys.core[i].tcache.prefetch_buffer_hits = atof(  // [한국어] sys.core[i].tcache.prefetch_buffer_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_writes") == 0) {
                  sys.core[i].tcache.wbb_writes = atof(  // [한국어] sys.core[i].tcache.wbb_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_reads") == 0) {
                  sys.core[i].tcache.wbb_reads = atof(  // [한국어] sys.core[i].tcache.wbb_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "conflicts") == 0) {
                  sys.core[i].tcache.conflicts = atof(  // [한국어] sys.core[i].tcache.conflicts 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }

            if (strcmp(xNode4.getAttribute("name"), "sharedmemory") ==
                0) {  // find system.core0.sharedmemory
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp;
                   k++) {  // get all items of param in
                           // system.core0.sharedmemory--sharedmemory
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "sharedmemory_config") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].sharedmemory.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].sharedmemory.dcache_config[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].sharedmemory.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].sharedmemory.dcache_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "buffer_sizes") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].sharedmemory.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].sharedmemory.buffer_sizes[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].sharedmemory.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].sharedmemory.buffer_sizes[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp;
                   k++) {  // get all items of stat in sharedmemory
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].sharedmemory.total_accesses = atof(  // [한국어] sys.core[i].sharedmemory.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_accesses") == 0) {
                  sys.core[i].sharedmemory.read_accesses = atof(  // [한국어] sys.core[i].sharedmemory.read_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_accesses") == 0) {
                  sys.core[i].sharedmemory.write_accesses = atof(  // [한국어] sys.core[i].sharedmemory.write_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].sharedmemory.total_hits = atof(  // [한국어] sys.core[i].sharedmemory.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].sharedmemory.total_misses = atof(  // [한국어] sys.core[i].sharedmemory.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_hits") == 0) {
                  sys.core[i].sharedmemory.read_hits = atof(  // [한국어] sys.core[i].sharedmemory.read_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_hits") == 0) {
                  sys.core[i].sharedmemory.write_hits = atof(  // [한국어] sys.core[i].sharedmemory.write_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_misses") == 0) {
                  sys.core[i].sharedmemory.read_misses = atof(  // [한국어] sys.core[i].sharedmemory.read_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_misses") == 0) {
                  sys.core[i].sharedmemory.write_misses = atof(  // [한국어] sys.core[i].sharedmemory.write_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "replacements") == 0) {
                  sys.core[i].sharedmemory.replacements = atof(  // [한국어] sys.core[i].sharedmemory.replacements 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_backs") == 0) {
                  sys.core[i].sharedmemory.write_backs = atof(  // [한국어] sys.core[i].sharedmemory.write_backs 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "miss_buffer_access") == 0) {
                  sys.core[i].sharedmemory.miss_buffer_access = atof(  // [한국어] sys.core[i].sharedmemory.miss_buffer_access 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "fill_buffer_accesses") == 0) {
                  sys.core[i].sharedmemory.fill_buffer_accesses = atof(  // [한국어] sys.core[i].sharedmemory.fill_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_accesses") == 0) {
                  sys.core[i].sharedmemory.prefetch_buffer_accesses = atof(  // [한국어] sys.core[i].sharedmemory.prefetch_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_writes") == 0) {
                  sys.core[i].sharedmemory.prefetch_buffer_writes = atof(  // [한국어] sys.core[i].sharedmemory.prefetch_buffer_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_reads") == 0) {
                  sys.core[i].sharedmemory.prefetch_buffer_reads = atof(  // [한국어] sys.core[i].sharedmemory.prefetch_buffer_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_hits") == 0) {
                  sys.core[i].sharedmemory.prefetch_buffer_hits = atof(  // [한국어] sys.core[i].sharedmemory.prefetch_buffer_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_writes") == 0) {
                  sys.core[i].sharedmemory.wbb_writes = atof(  // [한국어] sys.core[i].sharedmemory.wbb_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_reads") == 0) {
                  sys.core[i].sharedmemory.wbb_reads = atof(  // [한국어] sys.core[i].sharedmemory.wbb_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "conflicts") == 0) {
                  sys.core[i].sharedmemory.conflicts = atof(  // [한국어] sys.core[i].sharedmemory.conflicts 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }

            if (strcmp(xNode4.getAttribute("name"), "dcache") ==
                0) {  // find system.core0.dcache
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp; k++) {  // get all items of param in
                                            // system.core0.dcache--dcache
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "dcache_config") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].dcache.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].dcache.dcache_config[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].dcache.dcache_config[m] = atof(chtmp1);  // [한국어] sys.core[i].dcache.dcache_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                  continue;
                }
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "buffer_sizes") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].dcache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].dcache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].dcache.buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.core[i].dcache.buffer_sizes[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {  // get all items of stat in dcache
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].dcache.total_accesses = atof(  // [한국어] sys.core[i].dcache.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_accesses") == 0) {
                  sys.core[i].dcache.read_accesses = atof(  // [한국어] sys.core[i].dcache.read_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_accesses") == 0) {
                  sys.core[i].dcache.write_accesses = atof(  // [한국어] sys.core[i].dcache.write_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].dcache.total_hits = atof(  // [한국어] sys.core[i].dcache.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].dcache.total_misses = atof(  // [한국어] sys.core[i].dcache.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_hits") == 0) {
                  sys.core[i].dcache.read_hits = atof(  // [한국어] sys.core[i].dcache.read_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_hits") == 0) {
                  sys.core[i].dcache.write_hits = atof(  // [한국어] sys.core[i].dcache.write_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_misses") == 0) {
                  sys.core[i].dcache.read_misses = atof(  // [한국어] sys.core[i].dcache.read_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_misses") == 0) {
                  sys.core[i].dcache.write_misses = atof(  // [한국어] sys.core[i].dcache.write_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "replacements") == 0) {
                  sys.core[i].dcache.replacements = atof(  // [한국어] sys.core[i].dcache.replacements 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_backs") == 0) {
                  sys.core[i].dcache.write_backs = atof(  // [한국어] sys.core[i].dcache.write_backs 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "miss_buffer_access") == 0) {
                  sys.core[i].dcache.miss_buffer_access = atof(  // [한국어] sys.core[i].dcache.miss_buffer_access 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "fill_buffer_accesses") == 0) {
                  sys.core[i].dcache.fill_buffer_accesses = atof(  // [한국어] sys.core[i].dcache.fill_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_accesses") == 0) {
                  sys.core[i].dcache.prefetch_buffer_accesses = atof(  // [한국어] sys.core[i].dcache.prefetch_buffer_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_writes") == 0) {
                  sys.core[i].dcache.prefetch_buffer_writes = atof(  // [한국어] sys.core[i].dcache.prefetch_buffer_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_reads") == 0) {
                  sys.core[i].dcache.prefetch_buffer_reads = atof(  // [한국어] sys.core[i].dcache.prefetch_buffer_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "prefetch_buffer_hits") == 0) {
                  sys.core[i].dcache.prefetch_buffer_hits = atof(  // [한국어] sys.core[i].dcache.prefetch_buffer_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_writes") == 0) {
                  sys.core[i].dcache.wbb_writes = atof(  // [한국어] sys.core[i].dcache.wbb_writes 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "wbb_reads") == 0) {
                  sys.core[i].dcache.wbb_reads = atof(  // [한국어] sys.core[i].dcache.wbb_reads 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "conflicts") == 0) {
                  sys.core[i].dcache.conflicts = atof(  // [한국어] sys.core[i].dcache.conflicts 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }

            if (strcmp(xNode4.getAttribute("name"), "BTB") ==
                0) {  // find system.core0.BTB
              itmp = xNode4.nChildNode("param");
              for (k = 0; k < itmp;
                   k++) {  // get all items of param in system.core0.BTB--BTB
                if (strcmp(xNode4.getChildNode("param", k).getAttribute("name"),
                           "BTB_config") == 0) {
                  strtmp.assign(
                      xNode4.getChildNode("param", k).getAttribute("value"));
                  m = 0;
                  for (n = 0; n < strtmp.length(); n++) {
                    if (strtmp[n] != ',') {
                      sprintf(chtmp, "%c", strtmp[n]);
                      strcat(chtmp1, chtmp);
                    } else {
                      sys.core[i].BTB.BTB_config[m] = atoi(chtmp1);  // [한국어] sys.core[i].BTB.BTB_config[m] 에 XML에서 읽은 값 저장
                      m++;
                      chtmp1[0] = '\0';
                    }
                  }
                  sys.core[i].BTB.BTB_config[m] = atoi(chtmp1);  // [한국어] sys.core[i].BTB.BTB_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              itmp = xNode4.nChildNode("stat");
              for (k = 0; k < itmp; k++) {  // get all items of stat in BTB
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_accesses") == 0) {
                  sys.core[i].BTB.total_accesses = atof(  // [한국어] sys.core[i].BTB.total_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_accesses") == 0) {
                  sys.core[i].BTB.read_accesses = atof(  // [한국어] sys.core[i].BTB.read_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_accesses") == 0) {
                  sys.core[i].BTB.write_accesses = atof(  // [한국어] sys.core[i].BTB.write_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_hits") == 0) {
                  sys.core[i].BTB.total_hits = atof(  // [한국어] sys.core[i].BTB.total_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "total_misses") == 0) {
                  sys.core[i].BTB.total_misses = atof(  // [한국어] sys.core[i].BTB.total_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_hits") == 0) {
                  sys.core[i].BTB.read_hits = atof(  // [한국어] sys.core[i].BTB.read_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_hits") == 0) {
                  sys.core[i].BTB.write_hits = atof(  // [한국어] sys.core[i].BTB.write_hits 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "read_misses") == 0) {
                  sys.core[i].BTB.read_misses = atof(  // [한국어] sys.core[i].BTB.read_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "write_misses") == 0) {
                  sys.core[i].BTB.write_misses = atof(  // [한국어] sys.core[i].BTB.write_misses 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
                if (strcmp(xNode4.getChildNode("stat", k).getAttribute("name"),
                           "replacements") == 0) {
                  sys.core[i].BTB.replacements = atof(  // [한국어] sys.core[i].BTB.replacements 에 XML 노드의 속성/자식 값을 읽어 저장
                      xNode4.getChildNode("stat", k).getAttribute("value"));
                  continue;
                }
              }
            }
          }
        } else {
          printf(
              "The value of homogeneous_cores or number_of_cores is not "
              "correct!");
          exit(0);
        }
      }
    }

    //__________________________________________Get
    // system.L1Directory0-n____________________________________________
    int w, tmpOrderofComponents_3layer;
    w = OrderofComponents_3layer + 1;
    tmpOrderofComponents_3layer = OrderofComponents_3layer;
    if (sys.homogeneous_L1Directories == 1)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    else
      OrderofComponents_3layer =
          OrderofComponents_3layer + sys.number_of_L1Directories;

    for (i = 0; i < (OrderofComponents_3layer - tmpOrderofComponents_3layer);
         i++) {
      xNode3 = xNode2.getChildNode("component", w);
      if (xNode3.isEmpty() == 1) {
        printf(
            "The value of homogeneous_L1Directories or number_of_L1Directories "
            "is not correct!");
        exit(0);
      } else {
        if (strstr(xNode3.getAttribute("id"), "L1Directory") != NULL) {
          itmp = xNode3.nChildNode("param");
          for (k = 0; k < itmp;
               k++) {  // get all items of param in system.L1Directory
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "Dir_config") == 0) {
              strtmp.assign(
                  xNode3.getChildNode("param", k).getAttribute("value"));
              m = 0;
              for (n = 0; n < strtmp.length(); n++) {
                if (strtmp[n] != ',') {
                  sprintf(chtmp, "%c", strtmp[n]);
                  strcat(chtmp1, chtmp);
                } else {
                  sys.L1Directory[i].Dir_config[m] = atof(chtmp1);  // [한국어] sys.L1Directory[i].Dir_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              sys.L1Directory[i].Dir_config[m] = atof(chtmp1);  // [한국어] sys.L1Directory[i].Dir_config[m] 에 XML에서 읽은 값 저장
              m++;
              chtmp1[0] = '\0';
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "buffer_sizes") == 0) {
              strtmp.assign(
                  xNode3.getChildNode("param", k).getAttribute("value"));
              m = 0;
              for (n = 0; n < strtmp.length(); n++) {
                if (strtmp[n] != ',') {
                  sprintf(chtmp, "%c", strtmp[n]);
                  strcat(chtmp1, chtmp);
                } else {
                  sys.L1Directory[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L1Directory[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              sys.L1Directory[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L1Directory[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
              m++;
              chtmp1[0] = '\0';
              continue;
            }

            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "clockrate") == 0) {
              sys.L1Directory[i].clockrate =  // [한국어] sys.L1Directory[i].clockrate 에 정수형 XML 파라미터/속성 값 저장
                  atoi(xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "ports") == 0) {
              strtmp.assign(
                  xNode3.getChildNode("param", k).getAttribute("value"));
              m = 0;
              for (n = 0; n < strtmp.length(); n++) {
                if (strtmp[n] != ',') {
                  sprintf(chtmp, "%c", strtmp[n]);
                  strcat(chtmp1, chtmp);
                } else {
                  sys.L1Directory[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L1Directory[i].ports[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              sys.L1Directory[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L1Directory[i].ports[m] 에 XML에서 읽은 값 저장
              m++;
              chtmp1[0] = '\0';
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "device_type") == 0) {
              sys.L1Directory[i].device_type =  // [한국어] sys.L1Directory[i].device_type 에 정수형 XML 파라미터/속성 값 저장
                  atoi(xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "Directory_type") == 0) {
              sys.L1Directory[i].Directory_type =  // [한국어] sys.L1Directory[i].Directory_type 에 정수형 XML 파라미터/속성 값 저장
                  atoi(xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "3D_stack") == 0) {
              strcpy(sys.L1Directory[i].threeD_stack,
                     xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
          }
          itmp = xNode3.nChildNode("stat");
          for (k = 0; k < itmp;
               k++) {  // get all items of stat in system.L2directorydirectory
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "total_accesses") == 0) {
              sys.L1Directory[i].total_accesses =  // [한국어] sys.L1Directory[i].total_accesses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "read_accesses") == 0) {
              sys.L1Directory[i].read_accesses =  // [한국어] sys.L1Directory[i].read_accesses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "write_accesses") == 0) {
              sys.L1Directory[i].write_accesses =  // [한국어] sys.L1Directory[i].write_accesses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "read_misses") == 0) {
              sys.L1Directory[i].read_misses =  // [한국어] sys.L1Directory[i].read_misses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "write_misses") == 0) {
              sys.L1Directory[i].write_misses =  // [한국어] sys.L1Directory[i].write_misses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "conflicts") == 0) {
              sys.L1Directory[i].conflicts =  // [한국어] sys.L1Directory[i].conflicts 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "duty_cycle") == 0) {
              sys.L1Directory[i].duty_cycle =  // [한국어] sys.L1Directory[i].duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
          }
          w = w + 1;
        } else {
          printf(
              "The value of homogeneous_L1Directories or "
              "number_of_L1Directories is not correct!");
          exit(0);
        }
      }
    }

    //__________________________________________Get
    // system.L2Directory0-n____________________________________________
    w = OrderofComponents_3layer + 1;
    tmpOrderofComponents_3layer = OrderofComponents_3layer;
    if (sys.homogeneous_L2Directories == 1)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    else
      OrderofComponents_3layer =
          OrderofComponents_3layer + sys.number_of_L2Directories;

    for (i = 0; i < (OrderofComponents_3layer - tmpOrderofComponents_3layer);
         i++) {
      xNode3 = xNode2.getChildNode("component", w);
      if (xNode3.isEmpty() == 1) {
        printf(
            "The value of homogeneous_L2Directories or number_of_L2Directories "
            "is not correct!");
        exit(0);
      } else {
        if (strstr(xNode3.getAttribute("id"), "L2Directory") != NULL) {
          itmp = xNode3.nChildNode("param");
          for (k = 0; k < itmp;
               k++) {  // get all items of param in system.L2Directory
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "Dir_config") == 0) {
              strtmp.assign(
                  xNode3.getChildNode("param", k).getAttribute("value"));
              m = 0;
              for (n = 0; n < strtmp.length(); n++) {
                if (strtmp[n] != ',') {
                  sprintf(chtmp, "%c", strtmp[n]);
                  strcat(chtmp1, chtmp);
                } else {
                  sys.L2Directory[i].Dir_config[m] = atof(chtmp1);  // [한국어] sys.L2Directory[i].Dir_config[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              sys.L2Directory[i].Dir_config[m] = atof(chtmp1);  // [한국어] sys.L2Directory[i].Dir_config[m] 에 XML에서 읽은 값 저장
              m++;
              chtmp1[0] = '\0';
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "buffer_sizes") == 0) {
              strtmp.assign(
                  xNode3.getChildNode("param", k).getAttribute("value"));
              m = 0;
              for (n = 0; n < strtmp.length(); n++) {
                if (strtmp[n] != ',') {
                  sprintf(chtmp, "%c", strtmp[n]);
                  strcat(chtmp1, chtmp);
                } else {
                  sys.L2Directory[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L2Directory[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              sys.L2Directory[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L2Directory[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
              m++;
              chtmp1[0] = '\0';
              continue;
            }

            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "clockrate") == 0) {
              sys.L2Directory[i].clockrate =  // [한국어] sys.L2Directory[i].clockrate 에 정수형 XML 파라미터/속성 값 저장
                  atoi(xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "Directory_type") == 0) {
              sys.L2Directory[i].Directory_type =  // [한국어] sys.L2Directory[i].Directory_type 에 정수형 XML 파라미터/속성 값 저장
                  atoi(xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "ports") == 0) {
              strtmp.assign(
                  xNode3.getChildNode("param", k).getAttribute("value"));
              m = 0;
              for (n = 0; n < strtmp.length(); n++) {
                if (strtmp[n] != ',') {
                  sprintf(chtmp, "%c", strtmp[n]);
                  strcat(chtmp1, chtmp);
                } else {
                  sys.L2Directory[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L2Directory[i].ports[m] 에 XML에서 읽은 값 저장
                  m++;
                  chtmp1[0] = '\0';
                }
              }
              sys.L2Directory[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L2Directory[i].ports[m] 에 XML에서 읽은 값 저장
              m++;
              chtmp1[0] = '\0';
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "device_type") == 0) {
              sys.L2Directory[i].device_type =  // [한국어] sys.L2Directory[i].device_type 에 정수형 XML 파라미터/속성 값 저장
                  atoi(xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                       "3D_stack") == 0) {
              strcpy(sys.L2Directory[i].threeD_stack,
                     xNode3.getChildNode("param", k).getAttribute("value"));
              continue;
            }
          }
          itmp = xNode3.nChildNode("stat");
          for (k = 0; k < itmp;
               k++) {  // get all items of stat in system.L2directorydirectory
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "total_accesses") == 0) {
              sys.L2Directory[i].total_accesses =  // [한국어] sys.L2Directory[i].total_accesses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "read_accesses") == 0) {
              sys.L2Directory[i].read_accesses =  // [한국어] sys.L2Directory[i].read_accesses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "write_accesses") == 0) {
              sys.L2Directory[i].write_accesses =  // [한국어] sys.L2Directory[i].write_accesses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "read_misses") == 0) {
              sys.L2Directory[i].read_misses =  // [한국어] sys.L2Directory[i].read_misses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "write_misses") == 0) {
              sys.L2Directory[i].write_misses =  // [한국어] sys.L2Directory[i].write_misses 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "conflicts") == 0) {
              sys.L2Directory[i].conflicts =  // [한국어] sys.L2Directory[i].conflicts 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
            if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                       "duty_cycle") == 0) {
              sys.L2Directory[i].duty_cycle =  // [한국어] sys.L2Directory[i].duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                  atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              continue;
            }
          }
          w = w + 1;
        } else {
          printf(
              "The value of homogeneous_L2Directories or "
              "number_of_L2Directories is not correct!");
          exit(0);
        }
      }
    }

    //__________________________________________Get
    // system.L2[0..n]____________________________________________
    w = OrderofComponents_3layer + 1;
    tmpOrderofComponents_3layer = OrderofComponents_3layer;
    if (sys.homogeneous_L2s == 1)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    else
      OrderofComponents_3layer = OrderofComponents_3layer + sys.number_of_L2s;

    for (i = 0; i < (OrderofComponents_3layer - tmpOrderofComponents_3layer);
         i++) {
      xNode3 = xNode2.getChildNode("component", w);
      if (xNode3.isEmpty() == 1) {
        printf("The value of homogeneous_L2s or number_of_L2s is not correct!");
        exit(0);
      } else {
        if (strstr(xNode3.getAttribute("name"), "L2") != NULL) {
          {  // For L20-L2i
            // Get all params with system.L2?
            itmp = xNode3.nChildNode("param");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "L2_config") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.L2[i].L2_config[m] = atof(chtmp1);  // [한국어] sys.L2[i].L2_config[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.L2[i].L2_config[m] = atof(chtmp1);  // [한국어] sys.L2[i].L2_config[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "clockrate") == 0) {
                sys.L2[i].clockrate =  // [한국어] sys.L2[i].clockrate 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "merged_dir") == 0) {
                sys.L2[i].merged_dir = (bool)atoi(  // [한국어] sys.L2[i].merged_dir 에 XML 노드의 속성/자식 값을 읽어 저장
                    xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "ports") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.L2[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L2[i].ports[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.L2[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L2[i].ports[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "device_type") == 0) {
                sys.L2[i].device_type =  // [한국어] sys.L2[i].device_type 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "threeD_stack") == 0) {
                strcpy(sys.L2[i].threeD_stack,
                       (xNode3.getChildNode("param", k).getAttribute("value")));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "buffer_sizes") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.L2[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L2[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.L2[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L2[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
            }
            // Get all stats with system.L2?
            itmp = xNode3.nChildNode("stat");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_accesses") == 0) {
                sys.L2[i].total_accesses =  // [한국어] sys.L2[i].total_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "read_accesses") == 0) {
                sys.L2[i].read_accesses =  // [한국어] sys.L2[i].read_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_accesses") == 0) {
                sys.L2[i].write_accesses =  // [한국어] sys.L2[i].write_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_hits") == 0) {
                sys.L2[i].total_hits =  // [한국어] sys.L2[i].total_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_misses") == 0) {
                sys.L2[i].total_misses =  // [한국어] sys.L2[i].total_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "read_hits") == 0) {
                sys.L2[i].read_hits =  // [한국어] sys.L2[i].read_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_hits") == 0) {
                sys.L2[i].write_hits =  // [한국어] sys.L2[i].write_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "read_misses") == 0) {
                sys.L2[i].read_misses =  // [한국어] sys.L2[i].read_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_misses") == 0) {
                sys.L2[i].write_misses =  // [한국어] sys.L2[i].write_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "replacements") == 0) {
                sys.L2[i].replacements =  // [한국어] sys.L2[i].replacements 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_backs") == 0) {
                sys.L2[i].write_backs =  // [한국어] sys.L2[i].write_backs 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "miss_buffer_accesses") == 0) {
                sys.L2[i].miss_buffer_accesses =  // [한국어] sys.L2[i].miss_buffer_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fill_buffer_accesses") == 0) {
                sys.L2[i].fill_buffer_accesses =  // [한국어] sys.L2[i].fill_buffer_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_accesses") == 0) {
                sys.L2[i].prefetch_buffer_accesses =  // [한국어] sys.L2[i].prefetch_buffer_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_writes") == 0) {
                sys.L2[i].prefetch_buffer_writes =  // [한국어] sys.L2[i].prefetch_buffer_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_reads") == 0) {
                sys.L2[i].prefetch_buffer_reads =  // [한국어] sys.L2[i].prefetch_buffer_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_hits") == 0) {
                sys.L2[i].prefetch_buffer_hits =  // [한국어] sys.L2[i].prefetch_buffer_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "wbb_writes") == 0) {
                sys.L2[i].wbb_writes =  // [한국어] sys.L2[i].wbb_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "wbb_reads") == 0) {
                sys.L2[i].wbb_reads =  // [한국어] sys.L2[i].wbb_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "conflicts") == 0) {
                sys.L2[i].conflicts =  // [한국어] sys.L2[i].conflicts 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "duty_cycle") == 0) {
                sys.L2[i].duty_cycle =  // [한국어] sys.L2[i].duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }

              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_accesses") == 0) {
                sys.L2[i].homenode_read_accesses =  // [한국어] sys.L2[i].homenode_read_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_accesses") == 0) {
                sys.L2[i].homenode_read_accesses =  // [한국어] sys.L2[i].homenode_read_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_hits") == 0) {
                sys.L2[i].homenode_read_hits =  // [한국어] sys.L2[i].homenode_read_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_write_hits") == 0) {
                sys.L2[i].homenode_write_hits =  // [한국어] sys.L2[i].homenode_write_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_misses") == 0) {
                sys.L2[i].homenode_read_misses =  // [한국어] sys.L2[i].homenode_read_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_write_misses") == 0) {
                sys.L2[i].homenode_write_misses =  // [한국어] sys.L2[i].homenode_write_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "dir_duty_cycle") == 0) {
                sys.L2[i].dir_duty_cycle =  // [한국어] sys.L2[i].dir_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
            }
          }
          w = w + 1;
        } else {
          printf(
              "The value of homogeneous_L2s or number_of_L2s is not correct!");
          exit(0);
        }
      }
    }
    //__________________________________________Get
    // system.L3[0..n]____________________________________________
    w = OrderofComponents_3layer + 1;
    tmpOrderofComponents_3layer = OrderofComponents_3layer;
    if (sys.homogeneous_L3s == 1)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    else
      OrderofComponents_3layer = OrderofComponents_3layer + sys.number_of_L3s;

    for (i = 0; i < (OrderofComponents_3layer - tmpOrderofComponents_3layer);
         i++) {
      xNode3 = xNode2.getChildNode("component", w);
      if (xNode3.isEmpty() == 1) {
        printf("The value of homogeneous_L3s or number_of_L3s is not correct!");
        exit(0);
      } else {
        if (strstr(xNode3.getAttribute("name"), "L3") != NULL) {
          {  // For L30-L3i
            // Get all params with system.L3?
            itmp = xNode3.nChildNode("param");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "L3_config") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.L3[i].L3_config[m] = atof(chtmp1);  // [한국어] sys.L3[i].L3_config[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.L3[i].L3_config[m] = atof(chtmp1);  // [한국어] sys.L3[i].L3_config[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "clockrate") == 0) {
                sys.L3[i].clockrate =  // [한국어] sys.L3[i].clockrate 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "merged_dir") == 0) {
                sys.L3[i].merged_dir = (bool)atoi(  // [한국어] sys.L3[i].merged_dir 에 XML 노드의 속성/자식 값을 읽어 저장
                    xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "ports") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.L3[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L3[i].ports[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.L3[i].ports[m] = atoi(chtmp1);  // [한국어] sys.L3[i].ports[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "device_type") == 0) {
                sys.L3[i].device_type =  // [한국어] sys.L3[i].device_type 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "threeD_stack") == 0) {
                strcpy(sys.L3[i].threeD_stack,
                       (xNode3.getChildNode("param", k).getAttribute("value")));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "buffer_sizes") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.L3[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L3[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.L3[i].buffer_sizes[m] = atoi(chtmp1);  // [한국어] sys.L3[i].buffer_sizes[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
            }
            // Get all stats with system.L3?
            itmp = xNode3.nChildNode("stat");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_accesses") == 0) {
                sys.L3[i].total_accesses =  // [한국어] sys.L3[i].total_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "read_accesses") == 0) {
                sys.L3[i].read_accesses =  // [한국어] sys.L3[i].read_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_accesses") == 0) {
                sys.L3[i].write_accesses =  // [한국어] sys.L3[i].write_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_hits") == 0) {
                sys.L3[i].total_hits =  // [한국어] sys.L3[i].total_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_misses") == 0) {
                sys.L3[i].total_misses =  // [한국어] sys.L3[i].total_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "read_hits") == 0) {
                sys.L3[i].read_hits =  // [한국어] sys.L3[i].read_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_hits") == 0) {
                sys.L3[i].write_hits =  // [한국어] sys.L3[i].write_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "read_misses") == 0) {
                sys.L3[i].read_misses =  // [한국어] sys.L3[i].read_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_misses") == 0) {
                sys.L3[i].write_misses =  // [한국어] sys.L3[i].write_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "replacements") == 0) {
                sys.L3[i].replacements =  // [한국어] sys.L3[i].replacements 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "write_backs") == 0) {
                sys.L3[i].write_backs =  // [한국어] sys.L3[i].write_backs 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "miss_buffer_accesses") == 0) {
                sys.L3[i].miss_buffer_accesses =  // [한국어] sys.L3[i].miss_buffer_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "fill_buffer_accesses") == 0) {
                sys.L3[i].fill_buffer_accesses =  // [한국어] sys.L3[i].fill_buffer_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_accesses") == 0) {
                sys.L3[i].prefetch_buffer_accesses =  // [한국어] sys.L3[i].prefetch_buffer_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_writes") == 0) {
                sys.L3[i].prefetch_buffer_writes =  // [한국어] sys.L3[i].prefetch_buffer_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_reads") == 0) {
                sys.L3[i].prefetch_buffer_reads =  // [한국어] sys.L3[i].prefetch_buffer_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "prefetch_buffer_hits") == 0) {
                sys.L3[i].prefetch_buffer_hits =  // [한국어] sys.L3[i].prefetch_buffer_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "wbb_writes") == 0) {
                sys.L3[i].wbb_writes =  // [한국어] sys.L3[i].wbb_writes 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "wbb_reads") == 0) {
                sys.L3[i].wbb_reads =  // [한국어] sys.L3[i].wbb_reads 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "conflicts") == 0) {
                sys.L3[i].conflicts =  // [한국어] sys.L3[i].conflicts 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "duty_cycle") == 0) {
                sys.L3[i].duty_cycle =  // [한국어] sys.L3[i].duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }

              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_accesses") == 0) {
                sys.L3[i].homenode_read_accesses =  // [한국어] sys.L3[i].homenode_read_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_accesses") == 0) {
                sys.L3[i].homenode_read_accesses =  // [한국어] sys.L3[i].homenode_read_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_hits") == 0) {
                sys.L3[i].homenode_read_hits =  // [한국어] sys.L3[i].homenode_read_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_write_hits") == 0) {
                sys.L3[i].homenode_write_hits =  // [한국어] sys.L3[i].homenode_write_hits 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_read_misses") == 0) {
                sys.L3[i].homenode_read_misses =  // [한국어] sys.L3[i].homenode_read_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "homenode_write_misses") == 0) {
                sys.L3[i].homenode_write_misses =  // [한국어] sys.L3[i].homenode_write_misses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "dir_duty_cycle") == 0) {
                sys.L3[i].dir_duty_cycle =  // [한국어] sys.L3[i].dir_duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
                continue;
              }
            }
          }
          w = w + 1;
        } else {
          printf(
              "The value of homogeneous_L3s or number_of_L3s is not correct!");
          exit(0);
        }
      }
    }
    //__________________________________________Get
    // system.NoC[0..n]____________________________________________
    w = OrderofComponents_3layer + 1;
    tmpOrderofComponents_3layer = OrderofComponents_3layer;
    if (sys.homogeneous_NoCs == 1)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    else
      OrderofComponents_3layer = OrderofComponents_3layer + sys.number_of_NoCs;

    for (i = 0; i < (OrderofComponents_3layer - tmpOrderofComponents_3layer);
         i++) {
      xNode3 = xNode2.getChildNode("component", w);
      if (xNode3.isEmpty() == 1) {
        printf(
            "The value of homogeneous_NoCs or number_of_NoCs is not correct!");
        exit(0);
      } else {
        if (strstr(xNode3.getAttribute("name"), "noc") != NULL) {
          {  // For NoC0-NoCi
            // Get all params with system.NoC?
            itmp = xNode3.nChildNode("param");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "clockrate") == 0) {
                sys.NoC[i].clockrate =  // [한국어] sys.NoC[i].clockrate 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "type") == 0) {
                sys.NoC[i].type = (bool)atoi(  // [한국어] sys.NoC[i].type 에 XML 노드의 속성/자식 값을 읽어 저장
                    xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "topology") == 0) {
                strcpy(sys.NoC[i].topology,
                       (xNode3.getChildNode("param", k).getAttribute("value")));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "horizontal_nodes") == 0) {
                sys.NoC[i].horizontal_nodes =  // [한국어] sys.NoC[i].horizontal_nodes 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "vertical_nodes") == 0) {
                sys.NoC[i].vertical_nodes =  // [한국어] sys.NoC[i].vertical_nodes 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "has_global_link") == 0) {
                sys.NoC[i].has_global_link = (bool)atoi(  // [한국어] sys.NoC[i].has_global_link 에 XML 노드의 속성/자식 값을 읽어 저장
                    xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "link_throughput") == 0) {
                sys.NoC[i].link_throughput =  // [한국어] sys.NoC[i].link_throughput 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "link_latency") == 0) {
                sys.NoC[i].link_latency =  // [한국어] sys.NoC[i].link_latency 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "input_ports") == 0) {
                sys.NoC[i].input_ports =  // [한국어] sys.NoC[i].input_ports 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "output_ports") == 0) {
                sys.NoC[i].output_ports =  // [한국어] sys.NoC[i].output_ports 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "virtual_channel_per_port") == 0) {
                sys.NoC[i].virtual_channel_per_port =  // [한국어] sys.NoC[i].virtual_channel_per_port 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "flit_bits") == 0) {
                sys.NoC[i].flit_bits =  // [한국어] sys.NoC[i].flit_bits 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "input_buffer_entries_per_vc") == 0) {
                sys.NoC[i].input_buffer_entries_per_vc =  // [한국어] sys.NoC[i].input_buffer_entries_per_vc 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "dual_pump") == 0) {
                sys.NoC[i].dual_pump =  // [한국어] sys.NoC[i].dual_pump 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "chip_coverage") == 0) {
                sys.NoC[i].chip_coverage =  // [한국어] sys.NoC[i].chip_coverage 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "link_routing_over_percentage") == 0) {
                sys.NoC[i].route_over_perc =  // [한국어] sys.NoC[i].route_over_perc 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "ports_of_input_buffer") == 0) {
                strtmp.assign(
                    xNode3.getChildNode("param", k).getAttribute("value"));
                m = 0;
                for (n = 0; n < strtmp.length(); n++) {
                  if (strtmp[n] != ',') {
                    sprintf(chtmp, "%c", strtmp[n]);
                    strcat(chtmp1, chtmp);
                  } else {
                    sys.NoC[i].ports_of_input_buffer[m] = atoi(chtmp1);  // [한국어] sys.NoC[i].ports_of_input_buffer[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                sys.NoC[i].ports_of_input_buffer[m] = atoi(chtmp1);  // [한국어] sys.NoC[i].ports_of_input_buffer[m] 에 XML에서 읽은 값 저장
                m++;
                chtmp1[0] = '\0';
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "number_of_crossbars") == 0) {
                sys.NoC[i].number_of_crossbars =  // [한국어] sys.NoC[i].number_of_crossbars 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "crossbar_type") == 0) {
                strcpy(sys.NoC[i].crossbar_type,
                       (xNode3.getChildNode("param", k).getAttribute("value")));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "crosspoint_type") == 0) {
                strcpy(sys.NoC[i].crosspoint_type,
                       (xNode3.getChildNode("param", k).getAttribute("value")));
                continue;
              }
              if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                         "arbiter_type") == 0) {
                sys.NoC[i].arbiter_type =  // [한국어] sys.NoC[i].arbiter_type 에 정수형 XML 파라미터/속성 값 저장
                    atoi(xNode3.getChildNode("param", k).getAttribute("value"));
                continue;
              }
            }
            NumofCom_4 =
                xNode3.nChildNode("component");  // get the number of components
                                                 // within the third layer
            for (j = 0; j < NumofCom_4; j++) {
              xNode4 = xNode3.getChildNode("component", j);
              if (strcmp(xNode4.getAttribute("name"), "xbar0") ==
                  0) {  // find PBT
                itmp = xNode4.nChildNode("param");
                for (k = 0; k < itmp; k++) {  // get all items of param in
                                              // system.XoC0.xbar0--xbar0
                  if (strcmp(
                          xNode4.getChildNode("param", k).getAttribute("name"),
                          "number_of_inputs_of_crossbars") == 0) {
                    sys.NoC[i].xbar0.number_of_inputs_of_crossbars = atoi(  // [한국어] sys.NoC[i].xbar0.number_of_inputs_of_crossbars 에 XML 노드의 속성/자식 값을 읽어 저장
                        xNode4.getChildNode("param", k).getAttribute("value"));
                    continue;
                  }
                  if (strcmp(
                          xNode4.getChildNode("param", k).getAttribute("name"),
                          "number_of_outputs_of_crossbars") == 0) {
                    sys.NoC[i].xbar0.number_of_outputs_of_crossbars = atoi(  // [한국어] sys.NoC[i].xbar0.number_of_outputs_of_crossbars 에 XML 노드의 속성/자식 값을 읽어 저장
                        xNode4.getChildNode("param", k).getAttribute("value"));
                    continue;
                  }
                  if (strcmp(
                          xNode4.getChildNode("param", k).getAttribute("name"),
                          "flit_bits") == 0) {
                    sys.NoC[i].xbar0.flit_bits = atoi(  // [한국어] sys.NoC[i].xbar0.flit_bits 에 XML 노드의 속성/자식 값을 읽어 저장
                        xNode4.getChildNode("param", k).getAttribute("value"));
                    continue;
                  }
                  if (strcmp(
                          xNode4.getChildNode("param", k).getAttribute("name"),
                          "input_buffer_entries_per_port") == 0) {
                    sys.NoC[i].xbar0.input_buffer_entries_per_port = atoi(  // [한국어] sys.NoC[i].xbar0.input_buffer_entries_per_port 에 XML 노드의 속성/자식 값을 읽어 저장
                        xNode4.getChildNode("param", k).getAttribute("value"));
                    continue;
                  }
                  if (strcmp(
                          xNode4.getChildNode("param", k).getAttribute("name"),
                          "ports_of_input_buffer") == 0) {
                    strtmp.assign(
                        xNode4.getChildNode("param", k).getAttribute("value"));
                    m = 0;
                    for (n = 0; n < strtmp.length(); n++) {
                      if (strtmp[n] != ',') {
                        sprintf(chtmp, "%c", strtmp[n]);
                        strcat(chtmp1, chtmp);
                      } else {
                        sys.NoC[i].xbar0.ports_of_input_buffer[m] =  // [한국어] sys.NoC[i].xbar0.ports_of_input_buffer[m] 에 정수형 XML 파라미터/속성 값 저장
                            atoi(chtmp1);
                        m++;
                        chtmp1[0] = '\0';
                      }
                    }
                    sys.NoC[i].xbar0.ports_of_input_buffer[m] = atoi(chtmp1);  // [한국어] sys.NoC[i].xbar0.ports_of_input_buffer[m] 에 XML에서 읽은 값 저장
                    m++;
                    chtmp1[0] = '\0';
                  }
                }
                itmp = xNode4.nChildNode("stat");
                for (k = 0; k < itmp; k++) {  // get all items of stat in
                                              // system.core0.predictor--PBT
                  if (strcmp(
                          xNode4.getChildNode("stat", k).getAttribute("name"),
                          "predictor_accesses") == 0)
                    sys.core[i].predictor.predictor_accesses = atof(  // [한국어] sys.core[i].predictor.predictor_accesses 에 XML 노드의 속성/자식 값을 읽어 저장
                        xNode4.getChildNode("stat", k).getAttribute("value"));
                }
              }
            }
            // Get all stats with system.NoC?
            itmp = xNode3.nChildNode("stat");
            for (k = 0; k < itmp; k++) {
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "total_accesses") == 0)
                sys.NoC[i].total_accesses =  // [한국어] sys.NoC[i].total_accesses 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
              if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                         "duty_cycle") == 0)
                sys.NoC[i].duty_cycle =  // [한국어] sys.NoC[i].duty_cycle 에 실수형 XML 파라미터/속성 값 저장
                    atof(xNode3.getChildNode("stat", k).getAttribute("value"));
            }
          }
          w = w + 1;
        }
      }
    }
    //__________________________________________Get
    // system.mem____________________________________________
    if (OrderofComponents_3layer > 0)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    xNode3 = xNode2.getChildNode("component", OrderofComponents_3layer);
    if (xNode3.isEmpty() == 1) {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    if (strstr(xNode3.getAttribute("id"), "system.mem") != NULL) {
      itmp = xNode3.nChildNode("param");
      for (k = 0; k < itmp; k++) {  // get all items of param in system.mem
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "mem_tech_node") == 0) {
          sys.mem.mem_tech_node =  // [한국어] sys.mem.mem_tech_node 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "device_clock") == 0) {
          sys.mem.device_clock =  // [한국어] sys.mem.device_clock 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "peak_transfer_rate") == 0) {
          sys.mem.peak_transfer_rate =  // [한국어] sys.mem.peak_transfer_rate 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "capacity_per_channel") == 0) {
          sys.mem.capacity_per_channel =  // [한국어] sys.mem.capacity_per_channel 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "number_ranks") == 0) {
          sys.mem.number_ranks =  // [한국어] sys.mem.number_ranks 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "num_banks_of_DRAM_chip") == 0) {
          sys.mem.num_banks_of_DRAM_chip =  // [한국어] sys.mem.num_banks_of_DRAM_chip 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "Block_width_of_DRAM_chip") == 0) {
          sys.mem.Block_width_of_DRAM_chip =  // [한국어] sys.mem.Block_width_of_DRAM_chip 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "output_width_of_DRAM_chip") == 0) {
          sys.mem.output_width_of_DRAM_chip =  // [한국어] sys.mem.output_width_of_DRAM_chip 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "page_size_of_DRAM_chip") == 0) {
          sys.mem.page_size_of_DRAM_chip =  // [한국어] sys.mem.page_size_of_DRAM_chip 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "burstlength_of_DRAM_chip") == 0) {
          sys.mem.burstlength_of_DRAM_chip =  // [한국어] sys.mem.burstlength_of_DRAM_chip 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "internal_prefetch_of_DRAM_chip") == 0) {
          sys.mem.internal_prefetch_of_DRAM_chip =  // [한국어] sys.mem.internal_prefetch_of_DRAM_chip 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
      }
      itmp = xNode3.nChildNode("stat");
      for (k = 0; k < itmp; k++) {  // get all items of stat in system.mem
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "memory_accesses") == 0) {
          sys.mem.memory_accesses =  // [한국어] sys.mem.memory_accesses 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "memory_reads") == 0) {
          sys.mem.memory_reads =  // [한국어] sys.mem.memory_reads 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "memory_writes") == 0) {
          sys.mem.memory_writes =  // [한국어] sys.mem.memory_writes 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "dram_pre") == 0) {
          sys.mem.dram_pre =  // [한국어] sys.mem.dram_pre 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
      }
    } else {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    //__________________________________________Get
    // system.mc____________________________________________
    if (OrderofComponents_3layer > 0)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    xNode3 = xNode2.getChildNode("component", OrderofComponents_3layer);
    if (xNode3.isEmpty() == 1) {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    if (strstr(xNode3.getAttribute("id"), "system.mc") != NULL) {
      itmp = xNode3.nChildNode("param");
      for (k = 0; k < itmp; k++) {  // get all items of param in system.mem
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "mc_clock") == 0) {
          sys.mc.mc_clock =  // [한국어] sys.mc.mc_clock 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "block_size") == 0) {
          sys.mc.llc_line_length =  // [한국어] sys.mc.llc_line_length 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "number_mcs") == 0) {
          sys.mc.number_mcs =  // [한국어] sys.mc.number_mcs 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "memory_channels_per_mc") == 0) {
          sys.mc.memory_channels_per_mc =  // [한국어] sys.mc.memory_channels_per_mc 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "req_window_size_per_channel") == 0) {
          sys.mc.req_window_size_per_channel =  // [한국어] sys.mc.req_window_size_per_channel 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "IO_buffer_size_per_channel") == 0) {
          sys.mc.IO_buffer_size_per_channel =  // [한국어] sys.mc.IO_buffer_size_per_channel 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "databus_width") == 0) {
          sys.mc.databus_width =  // [한국어] sys.mc.databus_width 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "addressbus_width") == 0) {
          sys.mc.addressbus_width =  // [한국어] sys.mc.addressbus_width 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "PRT_entries") == 0) {
          sys.mc.PRT_entries =  // [한국어] sys.mc.PRT_entries 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "peak_transfer_rate") == 0) {
          sys.mc.peak_transfer_rate =  // [한국어] sys.mc.peak_transfer_rate 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "number_ranks") == 0) {
          sys.mc.number_ranks =  // [한국어] sys.mc.number_ranks 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "LVDS") == 0) {
          sys.mc.LVDS =  // [한국어] sys.mc.LVDS 에 정수형 XML 파라미터/속성 값 저장
              (bool)atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "type") == 0) {
          sys.mc.type =  // [한국어] sys.mc.type 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "withPHY") == 0) {
          sys.mc.withPHY =  // [한국어] sys.mc.withPHY 에 정수형 XML 파라미터/속성 값 저장
              (bool)atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }

        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_cmd_coeff") == 0) {
          sys.mc.dram_cmd_coeff =  // [한국어] sys.mc.dram_cmd_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_act_coeff") == 0) {
          sys.mc.dram_act_coeff =  // [한국어] sys.mc.dram_act_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_nop_coeff") == 0) {
          sys.mc.dram_nop_coeff =  // [한국어] sys.mc.dram_nop_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_activity_coeff") == 0) {
          sys.mc.dram_activity_coeff =  // [한국어] sys.mc.dram_activity_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_pre_coeff") == 0) {
          sys.mc.dram_pre_coeff =  // [한국어] sys.mc.dram_pre_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_rd_coeff") == 0) {
          sys.mc.dram_rd_coeff =  // [한국어] sys.mc.dram_rd_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_wr_coeff") == 0) {
          sys.mc.dram_wr_coeff =  // [한국어] sys.mc.dram_wr_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_req_coeff") == 0) {
          sys.mc.dram_req_coeff =  // [한국어] sys.mc.dram_req_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "dram_const_coeff") == 0) {
          sys.mc.dram_const_coeff =  // [한국어] sys.mc.dram_const_coeff 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
      }
      itmp = xNode3.nChildNode("stat");
      for (k = 0; k < itmp;
           k++) {  // get all items of stat in system.mendirectory
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "memory_accesses") == 0) {
          sys.mc.memory_accesses =  // [한국어] sys.mc.memory_accesses 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "memory_reads") == 0) {
          sys.mc.memory_reads =  // [한국어] sys.mc.memory_reads 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "memory_writes") == 0) {
          sys.mc.memory_writes =  // [한국어] sys.mc.memory_writes 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "dram_pre") == 0) {
          sys.mc.dram_pre =  // [한국어] sys.mc.dram_pre 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
      }
    } else {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    //__________________________________________Get
    // system.niu____________________________________________
    if (OrderofComponents_3layer > 0)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    xNode3 = xNode2.getChildNode("component", OrderofComponents_3layer);
    if (xNode3.isEmpty() == 1) {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    if (strstr(xNode3.getAttribute("id"), "system.niu") != NULL) {
      itmp = xNode3.nChildNode("param");
      for (k = 0; k < itmp; k++) {  // get all items of param in system.mem
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "clockrate") == 0) {
          sys.niu.clockrate =  // [한국어] sys.niu.clockrate 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "number_units") == 0) {
          sys.niu.number_units =  // [한국어] sys.niu.number_units 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "type") == 0) {
          sys.niu.type =  // [한국어] sys.niu.type 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
      }
      itmp = xNode3.nChildNode("stat");
      for (k = 0; k < itmp;
           k++) {  // get all items of stat in system.mendirectory
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "duty_cycle") == 0) {
          sys.niu.duty_cycle =  // [한국어] sys.niu.duty_cycle 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "total_load_perc") == 0) {
          sys.niu.total_load_perc =  // [한국어] sys.niu.total_load_perc 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
      }
    } else {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }

    //__________________________________________Get
    // system.pcie____________________________________________
    if (OrderofComponents_3layer > 0)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    xNode3 = xNode2.getChildNode("component", OrderofComponents_3layer);
    if (xNode3.isEmpty() == 1) {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    if (strstr(xNode3.getAttribute("id"), "system.pcie") != NULL) {
      itmp = xNode3.nChildNode("param");
      for (k = 0; k < itmp; k++) {  // get all items of param in system.mem
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "clockrate") == 0) {
          sys.pcie.clockrate =  // [한국어] sys.pcie.clockrate 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "number_units") == 0) {
          sys.pcie.number_units =  // [한국어] sys.pcie.number_units 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "num_channels") == 0) {
          sys.pcie.num_channels =  // [한국어] sys.pcie.num_channels 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "type") == 0) {
          sys.pcie.type =  // [한국어] sys.pcie.type 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "withPHY") == 0) {
          sys.pcie.withPHY =  // [한국어] sys.pcie.withPHY 에 정수형 XML 파라미터/속성 값 저장
              (bool)atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
      }
      itmp = xNode3.nChildNode("stat");
      for (k = 0; k < itmp;
           k++) {  // get all items of stat in system.mendirectory
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "duty_cycle") == 0) {
          sys.pcie.duty_cycle =  // [한국어] sys.pcie.duty_cycle 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "total_load_perc") == 0) {
          sys.pcie.total_load_perc =  // [한국어] sys.pcie.total_load_perc 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
      }
    } else {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    //__________________________________________Get
    // system.flashcontroller____________________________________________
    if (OrderofComponents_3layer > 0)
      OrderofComponents_3layer = OrderofComponents_3layer + 1;
    xNode3 = xNode2.getChildNode("component", OrderofComponents_3layer);
    if (xNode3.isEmpty() == 1) {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
    if (strstr(xNode3.getAttribute("id"), "system.flashc") != NULL) {
      itmp = xNode3.nChildNode("param");
      for (k = 0; k < itmp; k++) {  // get all items of param in system.mem
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"flashc_clock")==0)
        //{sys.flashc.mc_clock=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"block_size")==0)
        //{sys.flashc.llc_line_length=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "number_flashcs") == 0) {
          sys.flashc.number_mcs =  // [한국어] sys.flashc.number_mcs 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"memory_channels_per_flashc")==0)
        //{sys.flashc.memory_channels_per_mc=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"req_window_size_per_channel")==0)
        //{sys.flashc.req_window_size_per_channel=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"IO_buffer_size_per_channel")==0)
        //{sys.flashc.IO_buffer_size_per_channel=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"databus_width")==0)
        //{sys.flashc.databus_width=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"addressbus_width")==0)
        //{sys.flashc.addressbus_width=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "peak_transfer_rate") == 0) {
          sys.flashc.peak_transfer_rate =  // [한국어] sys.flashc.peak_transfer_rate 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"number_ranks")==0)
        //{sys.flashc.number_ranks=atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("param",k).getAttribute("name"),"LVDS")==0)
        //{sys.flashc.LVDS=(bool)atoi(xNode3.getChildNode("param",k).getAttribute("value"));continue;}
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "type") == 0) {
          sys.flashc.type =  // [한국어] sys.flashc.type 에 정수형 XML 파라미터/속성 값 저장
              atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("param", k).getAttribute("name"),
                   "withPHY") == 0) {
          sys.flashc.withPHY =  // [한국어] sys.flashc.withPHY 에 정수형 XML 파라미터/속성 값 저장
              (bool)atoi(xNode3.getChildNode("param", k).getAttribute("value"));
          continue;
        }
      }
      itmp = xNode3.nChildNode("stat");
      for (k = 0; k < itmp;
           k++) {  // get all items of stat in system.mendirectory
        //				if
        //(strcmp(xNode3.getChildNode("stat",k).getAttribute("name"),"memory_accesses")==0)
        //{sys.flashc.memory_accesses=atof(xNode3.getChildNode("stat",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("stat",k).getAttribute("name"),"memory_reads")==0)
        //{sys.flashc.memory_reads=atof(xNode3.getChildNode("stat",k).getAttribute("value"));continue;}
        //				if
        //(strcmp(xNode3.getChildNode("stat",k).getAttribute("name"),"memory_writes")==0)
        //{sys.flashc.memory_writes=atof(xNode3.getChildNode("stat",k).getAttribute("value"));continue;}
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "duty_cycle") == 0) {
          sys.flashc.duty_cycle =  // [한국어] sys.flashc.duty_cycle 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
        if (strcmp(xNode3.getChildNode("stat", k).getAttribute("name"),
                   "total_load_perc") == 0) {
          sys.flashc.total_load_perc =  // [한국어] sys.flashc.total_load_perc 에 실수형 XML 파라미터/속성 값 저장
              atof(xNode3.getChildNode("stat", k).getAttribute("value"));
          continue;
        }
      }
    } else {
      printf(
          "some value(s) of "
          "number_of_cores/number_of_L2s/number_of_L3s/number_of_NoCs is/are "
          "not correct!");
      exit(0);
    }
  }
}
/*
 * [한국어] ParseXML::initialize - sys 구조체 0 초기화
 * @return 없음
 *
 * 호출 체인:
 *   ParseXML::parse() → ParseXML::initialize()
 */
void ParseXML::initialize()  // Initialize all
{
  // All number_of_* at the level of 'system' 03/21/2009
  sys.number_of_cores = 1;  // [한국어] sys.number_of_cores 에 XML에서 읽은 값 저장
  sys.architecture = 1;  // 1 - fermi  // [한국어] sys.architecture 에 XML에서 읽은 값 저장
  sys.number_of_L1Directories = 1;  // [한국어] sys.number_of_L1Directories 에 XML에서 읽은 값 저장
  sys.number_of_L2Directories = 1;  // [한국어] sys.number_of_L2Directories 에 XML에서 읽은 값 저장
  sys.number_of_L2s = 1;  // [한국어] sys.number_of_L2s 에 XML에서 읽은 값 저장
  sys.Private_L2 = false;  // [한국어] sys.Private_L2 에 XML에서 읽은 값 저장
  sys.number_of_L3s = 1;  // [한국어] sys.number_of_L3s 에 XML에서 읽은 값 저장
  sys.number_of_NoCs = 1;  // [한국어] sys.number_of_NoCs 에 XML에서 읽은 값 저장
  // All params at the level of 'system'
  // strcpy(sys.homogeneous_cores,"default");
  sys.core_tech_node = 1;  // [한국어] sys.core_tech_node 에 XML에서 읽은 값 저장
  sys.target_core_clockrate = 1;  // [한국어] sys.target_core_clockrate 에 XML에서 읽은 값 저장
  sys.modeled_chip_voltage_ref = 1;  // [한국어] sys.modeled_chip_voltage_ref 에 XML에서 읽은 값 저장
  sys.target_chip_area = 1;  // [한국어] sys.target_chip_area 에 XML에서 읽은 값 저장
  sys.temperature = 340;  // [한국어] sys.temperature 에 XML에서 읽은 값 저장
  sys.number_cache_levels = 1;  // [한국어] sys.number_cache_levels 에 XML에서 읽은 값 저장
  sys.homogeneous_cores = 1;  // [한국어] sys.homogeneous_cores 에 XML에서 읽은 값 저장
  sys.homogeneous_L1Directories = 1;  // [한국어] sys.homogeneous_L1Directories 에 XML에서 읽은 값 저장
  sys.homogeneous_L2Directories = 1;  // [한국어] sys.homogeneous_L2Directories 에 XML에서 읽은 값 저장
  sys.homogeneous_L2s = 1;  // [한국어] sys.homogeneous_L2s 에 XML에서 읽은 값 저장
  sys.homogeneous_L3s = 1;  // [한국어] sys.homogeneous_L3s 에 XML에서 읽은 값 저장
  sys.homogeneous_NoCs = 1;  // [한국어] sys.homogeneous_NoCs 에 XML에서 읽은 값 저장
  sys.homogeneous_ccs = 1;  // [한국어] sys.homogeneous_ccs 에 XML에서 읽은 값 저장

  sys.static_cat1_flane = 0;  // [한국어] sys.static_cat1_flane 에 XML에서 읽은 값 저장
  sys.static_cat2_flane = 0;  // [한국어] sys.static_cat2_flane 에 XML에서 읽은 값 저장
  sys.static_cat3_flane = 0;  // [한국어] sys.static_cat3_flane 에 XML에서 읽은 값 저장
  sys.static_cat4_flane = 0;  // [한국어] sys.static_cat4_flane 에 XML에서 읽은 값 저장
  sys.static_cat5_flane = 0;  // [한국어] sys.static_cat5_flane 에 XML에서 읽은 값 저장
  sys.static_cat6_flane = 0;  // [한국어] sys.static_cat6_flane 에 XML에서 읽은 값 저장
  sys.static_shared_flane = 0;  // [한국어] sys.static_shared_flane 에 XML에서 읽은 값 저장
  sys.static_l1_flane = 0;  // [한국어] sys.static_l1_flane 에 XML에서 읽은 값 저장
  sys.static_l2_flane = 0;  // [한국어] sys.static_l2_flane 에 XML에서 읽은 값 저장
  sys.static_light_flane = 0;  // [한국어] sys.static_light_flane 에 XML에서 읽은 값 저장
  sys.static_intadd_flane = 0;  // [한국어] sys.static_intadd_flane 에 XML에서 읽은 값 저장
  sys.static_intmul_flane = 0;  // [한국어] sys.static_intmul_flane 에 XML에서 읽은 값 저장
  sys.static_geomean_flane = 0;  // [한국어] sys.static_geomean_flane 에 XML에서 읽은 값 저장

  sys.static_cat1_addlane = 0;  // [한국어] sys.static_cat1_addlane 에 XML에서 읽은 값 저장
  sys.static_cat2_addlane = 0;  // [한국어] sys.static_cat2_addlane 에 XML에서 읽은 값 저장
  sys.static_cat3_addlane = 0;  // [한국어] sys.static_cat3_addlane 에 XML에서 읽은 값 저장
  sys.static_cat4_addlane = 0;  // [한국어] sys.static_cat4_addlane 에 XML에서 읽은 값 저장
  sys.static_cat5_addlane = 0;  // [한국어] sys.static_cat5_addlane 에 XML에서 읽은 값 저장
  sys.static_cat6_addlane = 0;  // [한국어] sys.static_cat6_addlane 에 XML에서 읽은 값 저장
  sys.static_shared_addlane = 0;  // [한국어] sys.static_shared_addlane 에 XML에서 읽은 값 저장
  sys.static_l1_addlane = 0;  // [한국어] sys.static_l1_addlane 에 XML에서 읽은 값 저장
  sys.static_l2_addlane = 0;  // [한국어] sys.static_l2_addlane 에 XML에서 읽은 값 저장
  sys.static_light_addlane = 0;  // [한국어] sys.static_light_addlane 에 XML에서 읽은 값 저장
  sys.static_intadd_addlane = 0;  // [한국어] sys.static_intadd_addlane 에 XML에서 읽은 값 저장
  sys.static_intmul_addlane = 0;  // [한국어] sys.static_intmul_addlane 에 XML에서 읽은 값 저장
  sys.static_geomean_addlane = 0;  // [한국어] sys.static_geomean_addlane 에 XML에서 읽은 값 저장

  sys.Max_area_deviation = 1;  // [한국어] sys.Max_area_deviation 에 XML에서 읽은 값 저장
  sys.Max_power_deviation = 1;  // [한국어] sys.Max_power_deviation 에 XML에서 읽은 값 저장
  sys.device_type = 1;  // [한국어] sys.device_type 에 XML에서 읽은 값 저장
  sys.longer_channel_device = true;  // [한국어] sys.longer_channel_device 에 XML에서 읽은 값 저장
  sys.Embedded = false;  // [한국어] sys.Embedded 에 XML에서 읽은 값 저장
  sys.opt_dynamic_power = false;  // [한국어] sys.opt_dynamic_power 에 XML에서 읽은 값 저장
  sys.opt_lakage_power = false;  // [한국어] sys.opt_lakage_power 에 XML에서 읽은 값 저장
  sys.opt_clockrate = true;  // [한국어] sys.opt_clockrate 에 XML에서 읽은 값 저장
  sys.opt_area = false;  // [한국어] sys.opt_area 에 XML에서 읽은 값 저장
  sys.interconnect_projection_type = 1;  // [한국어] sys.interconnect_projection_type 에 XML에서 읽은 값 저장
  sys.idle_core_power = 0;  // [한국어] sys.idle_core_power 에 XML에서 읽은 값 저장
  int i, j;
  for (i = 0; i <= 63; i++) {
    sys.scaling_coefficients[i] = 1;  // [한국어] sys.scaling_coefficients[i] 에 XML에서 읽은 값 저장
    sys.core[i].clock_rate = 1;  // [한국어] sys.core[i].clock_rate 에 XML에서 읽은 값 저장
    sys.core[i].opt_local = true;  // [한국어] sys.core[i].opt_local 에 XML에서 읽은 값 저장
    sys.core[i].x86 = false;  // [한국어] sys.core[i].x86 에 XML에서 읽은 값 저장
    sys.core[i].machine_bits = 1;  // [한국어] sys.core[i].machine_bits 에 XML에서 읽은 값 저장
    sys.core[i].virtual_address_width = 1;  // [한국어] sys.core[i].virtual_address_width 에 XML에서 읽은 값 저장
    sys.core[i].physical_address_width = 1;  // [한국어] sys.core[i].physical_address_width 에 XML에서 읽은 값 저장
    sys.core[i].opcode_width = 1;  // [한국어] sys.core[i].opcode_width 에 XML에서 읽은 값 저장
    sys.core[i].micro_opcode_width = 1;  // [한국어] sys.core[i].micro_opcode_width 에 XML에서 읽은 값 저장
    // strcpy(sys.core[i].machine_type,"default");
    sys.core[i].internal_datapath_width = 1;  // [한국어] sys.core[i].internal_datapath_width 에 XML에서 읽은 값 저장
    sys.core[i].number_hardware_threads = 1;  // [한국어] sys.core[i].number_hardware_threads 에 XML에서 읽은 값 저장
    sys.core[i].fetch_width = 1;  // [한국어] sys.core[i].fetch_width 에 XML에서 읽은 값 저장
    sys.core[i].number_instruction_fetch_ports = 1;  // [한국어] sys.core[i].number_instruction_fetch_ports 에 XML에서 읽은 값 저장
    sys.core[i].decode_width = 1;  // [한국어] sys.core[i].decode_width 에 XML에서 읽은 값 저장
    sys.core[i].issue_width = 1;  // [한국어] sys.core[i].issue_width 에 XML에서 읽은 값 저장
    sys.core[i].peak_issue_width = 1;  // [한국어] sys.core[i].peak_issue_width 에 XML에서 읽은 값 저장
    sys.core[i].commit_width = 1;  // [한국어] sys.core[i].commit_width 에 XML에서 읽은 값 저장
    for (j = 0; j < 20; j++) sys.core[i].pipelines_per_core[j] = 1;
    for (j = 0; j < 20; j++) sys.core[i].pipeline_depth[j] = 1;
    strcpy(sys.core[i].FPU, "default");
    strcpy(sys.core[i].divider_multiplier, "default");
    sys.core[i].ALU_per_core = 1;  // [한국어] sys.core[i].ALU_per_core 에 XML에서 읽은 값 저장
    sys.core[i].FPU_per_core = 1.0;  // [한국어] sys.core[i].FPU_per_core 에 XML에서 읽은 값 저장
    sys.core[i].MUL_per_core = 1;  // [한국어] sys.core[i].MUL_per_core 에 XML에서 읽은 값 저장
    sys.core[i].instruction_buffer_size = 1;  // [한국어] sys.core[i].instruction_buffer_size 에 XML에서 읽은 값 저장
    sys.core[i].decoded_stream_buffer_size = 1;  // [한국어] sys.core[i].decoded_stream_buffer_size 에 XML에서 읽은 값 저장
    // strcpy(sys.core[i].instruction_window_scheme,"default");
    sys.core[i].instruction_window_size = 1;  // [한국어] sys.core[i].instruction_window_size 에 XML에서 읽은 값 저장
    sys.core[i].ROB_size = 1;  // [한국어] sys.core[i].ROB_size 에 XML에서 읽은 값 저장
    sys.core[i].archi_Regs_IRF_size = 1;  // [한국어] sys.core[i].archi_Regs_IRF_size 에 XML에서 읽은 값 저장
    sys.core[i].archi_Regs_FRF_size = 1;  // [한국어] sys.core[i].archi_Regs_FRF_size 에 XML에서 읽은 값 저장
    sys.core[i].phy_Regs_IRF_size = 1;  // [한국어] sys.core[i].phy_Regs_IRF_size 에 XML에서 읽은 값 저장
    sys.core[i].phy_Regs_FRF_size = 1;  // [한국어] sys.core[i].phy_Regs_FRF_size 에 XML에서 읽은 값 저장
    // strcpy(sys.core[i].rename_scheme,"default");
    sys.core[i].register_windows_size = 1;  // [한국어] sys.core[i].register_windows_size 에 XML에서 읽은 값 저장
    strcpy(sys.core[i].LSU_order, "default");
    sys.core[i].store_buffer_size = 1;  // [한국어] sys.core[i].store_buffer_size 에 XML에서 읽은 값 저장
    sys.core[i].load_buffer_size = 1;  // [한국어] sys.core[i].load_buffer_size 에 XML에서 읽은 값 저장
    sys.core[i].memory_ports = 1;  // [한국어] sys.core[i].memory_ports 에 XML에서 읽은 값 저장
    strcpy(sys.core[i].Dcache_dual_pump, "default");
    sys.core[i].RAS_size = 1;  // [한국어] sys.core[i].RAS_size 에 XML에서 읽은 값 저장
    // all stats at the level of system.core(0-n)
    sys.core[i].total_instructions = 1;  // [한국어] sys.core[i].total_instructions 에 XML에서 읽은 값 저장
    sys.core[i].int_instructions = 1;  // [한국어] sys.core[i].int_instructions 에 XML에서 읽은 값 저장
    sys.core[i].fp_instructions = 1;  // [한국어] sys.core[i].fp_instructions 에 XML에서 읽은 값 저장
    sys.core[i].branch_instructions = 1;  // [한국어] sys.core[i].branch_instructions 에 XML에서 읽은 값 저장
    sys.core[i].branch_mispredictions = 1;  // [한국어] sys.core[i].branch_mispredictions 에 XML에서 읽은 값 저장
    sys.core[i].committed_instructions = 1;  // [한국어] sys.core[i].committed_instructions 에 XML에서 읽은 값 저장
    sys.core[i].load_instructions = 1;  // [한국어] sys.core[i].load_instructions 에 XML에서 읽은 값 저장
    sys.core[i].store_instructions = 1;  // [한국어] sys.core[i].store_instructions 에 XML에서 읽은 값 저장
    sys.core[i].total_cycles = 1;  // [한국어] sys.core[i].total_cycles 에 XML에서 읽은 값 저장
    sys.core[i].idle_cycles = 1;  // [한국어] sys.core[i].idle_cycles 에 XML에서 읽은 값 저장
    sys.core[i].busy_cycles = 1;  // [한국어] sys.core[i].busy_cycles 에 XML에서 읽은 값 저장
    sys.core[i].instruction_buffer_reads = 1;  // [한국어] sys.core[i].instruction_buffer_reads 에 XML에서 읽은 값 저장
    sys.core[i].instruction_buffer_write = 1;  // [한국어] sys.core[i].instruction_buffer_write 에 XML에서 읽은 값 저장
    sys.core[i].ROB_reads = 1;  // [한국어] sys.core[i].ROB_reads 에 XML에서 읽은 값 저장
    sys.core[i].ROB_writes = 1;  // [한국어] sys.core[i].ROB_writes 에 XML에서 읽은 값 저장
    sys.core[i].rename_accesses = 1;  // [한국어] sys.core[i].rename_accesses 에 XML에서 읽은 값 저장
    sys.core[i].inst_window_reads = 1;  // [한국어] sys.core[i].inst_window_reads 에 XML에서 읽은 값 저장
    sys.core[i].inst_window_writes = 1;  // [한국어] sys.core[i].inst_window_writes 에 XML에서 읽은 값 저장
    sys.core[i].inst_window_wakeup_accesses = 1;  // [한국어] sys.core[i].inst_window_wakeup_accesses 에 XML에서 읽은 값 저장
    sys.core[i].inst_window_selections = 1;  // [한국어] sys.core[i].inst_window_selections 에 XML에서 읽은 값 저장
    sys.core[i].archi_int_regfile_reads = 1;  // [한국어] sys.core[i].archi_int_regfile_reads 에 XML에서 읽은 값 저장
    sys.core[i].archi_float_regfile_reads = 1;  // [한국어] sys.core[i].archi_float_regfile_reads 에 XML에서 읽은 값 저장
    sys.core[i].phy_int_regfile_reads = 1;  // [한국어] sys.core[i].phy_int_regfile_reads 에 XML에서 읽은 값 저장
    sys.core[i].phy_float_regfile_reads = 1;  // [한국어] sys.core[i].phy_float_regfile_reads 에 XML에서 읽은 값 저장
    sys.core[i].windowed_reg_accesses = 1;  // [한국어] sys.core[i].windowed_reg_accesses 에 XML에서 읽은 값 저장
    sys.core[i].windowed_reg_transports = 1;  // [한국어] sys.core[i].windowed_reg_transports 에 XML에서 읽은 값 저장
    sys.core[i].function_calls = 1;  // [한국어] sys.core[i].function_calls 에 XML에서 읽은 값 저장
    sys.core[i].ialu_accesses = 1;  // [한국어] sys.core[i].ialu_accesses 에 XML에서 읽은 값 저장
    sys.core[i].fpu_accesses = 1;  // [한국어] sys.core[i].fpu_accesses 에 XML에서 읽은 값 저장
    sys.core[i].mul_accesses = 1;  // [한국어] sys.core[i].mul_accesses 에 XML에서 읽은 값 저장
    sys.core[i].cdb_alu_accesses = 1;  // [한국어] sys.core[i].cdb_alu_accesses 에 XML에서 읽은 값 저장
    sys.core[i].cdb_mul_accesses = 1;  // [한국어] sys.core[i].cdb_mul_accesses 에 XML에서 읽은 값 저장
    sys.core[i].cdb_fpu_accesses = 1;  // [한국어] sys.core[i].cdb_fpu_accesses 에 XML에서 읽은 값 저장
    sys.core[i].load_buffer_reads = 1;  // [한국어] sys.core[i].load_buffer_reads 에 XML에서 읽은 값 저장
    sys.core[i].load_buffer_writes = 1;  // [한국어] sys.core[i].load_buffer_writes 에 XML에서 읽은 값 저장
    sys.core[i].load_buffer_cams = 1;  // [한국어] sys.core[i].load_buffer_cams 에 XML에서 읽은 값 저장
    sys.core[i].store_buffer_reads = 1;  // [한국어] sys.core[i].store_buffer_reads 에 XML에서 읽은 값 저장
    sys.core[i].store_buffer_writes = 1;  // [한국어] sys.core[i].store_buffer_writes 에 XML에서 읽은 값 저장
    sys.core[i].store_buffer_cams = 1;  // [한국어] sys.core[i].store_buffer_cams 에 XML에서 읽은 값 저장
    sys.core[i].store_buffer_forwards = 1;  // [한국어] sys.core[i].store_buffer_forwards 에 XML에서 읽은 값 저장
    sys.core[i].main_memory_access = 1;  // [한국어] sys.core[i].main_memory_access 에 XML에서 읽은 값 저장
    sys.core[i].main_memory_read = 1;  // [한국어] sys.core[i].main_memory_read 에 XML에서 읽은 값 저장
    sys.core[i].main_memory_write = 1;  // [한국어] sys.core[i].main_memory_write 에 XML에서 읽은 값 저장
    sys.core[i].IFU_duty_cycle = 1;  // [한국어] sys.core[i].IFU_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].BR_duty_cycle = 1;  // [한국어] sys.core[i].BR_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].LSU_duty_cycle = 1;  // [한국어] sys.core[i].LSU_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].MemManU_I_duty_cycle = 1;  // [한국어] sys.core[i].MemManU_I_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].MemManU_D_duty_cycle = 1;  // [한국어] sys.core[i].MemManU_D_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].ALU_duty_cycle = 1;  // [한국어] sys.core[i].ALU_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].MUL_duty_cycle = 1;  // [한국어] sys.core[i].MUL_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].FPU_duty_cycle = 1;  // [한국어] sys.core[i].FPU_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].ALU_cdb_duty_cycle = 1;  // [한국어] sys.core[i].ALU_cdb_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].MUL_cdb_duty_cycle = 1;  // [한국어] sys.core[i].MUL_cdb_duty_cycle 에 XML에서 읽은 값 저장
    sys.core[i].FPU_cdb_duty_cycle = 1;  // [한국어] sys.core[i].FPU_cdb_duty_cycle 에 XML에서 읽은 값 저장
    // system.core?.predictor
    sys.core[i].predictor.prediction_width = 1;  // [한국어] sys.core[i].predictor.prediction_width 에 XML에서 읽은 값 저장
    strcpy(sys.core[i].predictor.prediction_scheme, "default");
    sys.core[i].predictor.predictor_size = 1;  // [한국어] sys.core[i].predictor.predictor_size 에 XML에서 읽은 값 저장
    sys.core[i].predictor.predictor_entries = 1;  // [한국어] sys.core[i].predictor.predictor_entries 에 XML에서 읽은 값 저장
    sys.core[i].predictor.local_predictor_entries = 1;  // [한국어] sys.core[i].predictor.local_predictor_entries 에 XML에서 읽은 값 저장
    for (j = 0; j < 20; j++) sys.core[i].predictor.local_predictor_size[j] = 1;
    sys.core[i].predictor.global_predictor_entries = 1;  // [한국어] sys.core[i].predictor.global_predictor_entries 에 XML에서 읽은 값 저장
    sys.core[i].predictor.global_predictor_bits = 1;  // [한국어] sys.core[i].predictor.global_predictor_bits 에 XML에서 읽은 값 저장
    sys.core[i].predictor.chooser_predictor_entries = 1;  // [한국어] sys.core[i].predictor.chooser_predictor_entries 에 XML에서 읽은 값 저장
    sys.core[i].predictor.chooser_predictor_bits = 1;  // [한국어] sys.core[i].predictor.chooser_predictor_bits 에 XML에서 읽은 값 저장
    sys.core[i].predictor.predictor_accesses = 1;  // [한국어] sys.core[i].predictor.predictor_accesses 에 XML에서 읽은 값 저장
    // system.core?.itlb
    sys.core[i].itlb.number_entries = 1;  // [한국어] sys.core[i].itlb.number_entries 에 XML에서 읽은 값 저장
    sys.core[i].itlb.total_hits = 1;  // [한국어] sys.core[i].itlb.total_hits 에 XML에서 읽은 값 저장
    sys.core[i].itlb.total_accesses = 1;  // [한국어] sys.core[i].itlb.total_accesses 에 XML에서 읽은 값 저장
    sys.core[i].itlb.total_misses = 1;  // [한국어] sys.core[i].itlb.total_misses 에 XML에서 읽은 값 저장
    // system.core?.icache
    for (j = 0; j < 20; j++) sys.core[i].icache.icache_config[j] = 1;
    // strcpy(sys.core[i].icache.buffer_sizes,"default");
    sys.core[i].icache.total_accesses = 1;  // [한국어] sys.core[i].icache.total_accesses 에 XML에서 읽은 값 저장
    sys.core[i].icache.read_accesses = 1;  // [한국어] sys.core[i].icache.read_accesses 에 XML에서 읽은 값 저장
    sys.core[i].icache.read_misses = 1;  // [한국어] sys.core[i].icache.read_misses 에 XML에서 읽은 값 저장
    sys.core[i].icache.replacements = 1;  // [한국어] sys.core[i].icache.replacements 에 XML에서 읽은 값 저장
    sys.core[i].icache.read_hits = 1;  // [한국어] sys.core[i].icache.read_hits 에 XML에서 읽은 값 저장
    sys.core[i].icache.total_hits = 1;  // [한국어] sys.core[i].icache.total_hits 에 XML에서 읽은 값 저장
    sys.core[i].icache.total_misses = 1;  // [한국어] sys.core[i].icache.total_misses 에 XML에서 읽은 값 저장
    sys.core[i].icache.miss_buffer_access = 1;  // [한국어] sys.core[i].icache.miss_buffer_access 에 XML에서 읽은 값 저장
    sys.core[i].icache.fill_buffer_accesses = 1;  // [한국어] sys.core[i].icache.fill_buffer_accesses 에 XML에서 읽은 값 저장
    sys.core[i].icache.prefetch_buffer_accesses = 1;  // [한국어] sys.core[i].icache.prefetch_buffer_accesses 에 XML에서 읽은 값 저장
    sys.core[i].icache.prefetch_buffer_writes = 1;  // [한국어] sys.core[i].icache.prefetch_buffer_writes 에 XML에서 읽은 값 저장
    sys.core[i].icache.prefetch_buffer_reads = 1;  // [한국어] sys.core[i].icache.prefetch_buffer_reads 에 XML에서 읽은 값 저장
    sys.core[i].icache.prefetch_buffer_hits = 1;  // [한국어] sys.core[i].icache.prefetch_buffer_hits 에 XML에서 읽은 값 저장
    // system.core?.dtlb
    sys.core[i].dtlb.number_entries = 1;  // [한국어] sys.core[i].dtlb.number_entries 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.total_accesses = 1;  // [한국어] sys.core[i].dtlb.total_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.read_accesses = 1;  // [한국어] sys.core[i].dtlb.read_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.write_accesses = 1;  // [한국어] sys.core[i].dtlb.write_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.write_hits = 1;  // [한국어] sys.core[i].dtlb.write_hits 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.read_hits = 1;  // [한국어] sys.core[i].dtlb.read_hits 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.read_misses = 1;  // [한국어] sys.core[i].dtlb.read_misses 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.write_misses = 1;  // [한국어] sys.core[i].dtlb.write_misses 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.total_hits = 1;  // [한국어] sys.core[i].dtlb.total_hits 에 XML에서 읽은 값 저장
    sys.core[i].dtlb.total_misses = 1;  // [한국어] sys.core[i].dtlb.total_misses 에 XML에서 읽은 값 저장
    // system.core?.dcache
    for (j = 0; j < 20; j++) sys.core[i].dcache.dcache_config[j] = 1;
    // strcpy(sys.core[i].dcache.buffer_sizes,"default");
    sys.core[i].dcache.total_accesses = 1;  // [한국어] sys.core[i].dcache.total_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.read_accesses = 1;  // [한국어] sys.core[i].dcache.read_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.write_accesses = 1;  // [한국어] sys.core[i].dcache.write_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.total_hits = 1;  // [한국어] sys.core[i].dcache.total_hits 에 XML에서 읽은 값 저장
    sys.core[i].dcache.total_misses = 1;  // [한국어] sys.core[i].dcache.total_misses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.read_hits = 1;  // [한국어] sys.core[i].dcache.read_hits 에 XML에서 읽은 값 저장
    sys.core[i].dcache.write_hits = 1;  // [한국어] sys.core[i].dcache.write_hits 에 XML에서 읽은 값 저장
    sys.core[i].dcache.read_misses = 1;  // [한국어] sys.core[i].dcache.read_misses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.write_misses = 1;  // [한국어] sys.core[i].dcache.write_misses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.replacements = 1;  // [한국어] sys.core[i].dcache.replacements 에 XML에서 읽은 값 저장
    sys.core[i].dcache.write_backs = 1;  // [한국어] sys.core[i].dcache.write_backs 에 XML에서 읽은 값 저장
    sys.core[i].dcache.miss_buffer_access = 1;  // [한국어] sys.core[i].dcache.miss_buffer_access 에 XML에서 읽은 값 저장
    sys.core[i].dcache.fill_buffer_accesses = 1;  // [한국어] sys.core[i].dcache.fill_buffer_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.prefetch_buffer_accesses = 1;  // [한국어] sys.core[i].dcache.prefetch_buffer_accesses 에 XML에서 읽은 값 저장
    sys.core[i].dcache.prefetch_buffer_writes = 1;  // [한국어] sys.core[i].dcache.prefetch_buffer_writes 에 XML에서 읽은 값 저장
    sys.core[i].dcache.prefetch_buffer_reads = 1;  // [한국어] sys.core[i].dcache.prefetch_buffer_reads 에 XML에서 읽은 값 저장
    sys.core[i].dcache.prefetch_buffer_hits = 1;  // [한국어] sys.core[i].dcache.prefetch_buffer_hits 에 XML에서 읽은 값 저장
    sys.core[i].dcache.wbb_writes = 1;  // [한국어] sys.core[i].dcache.wbb_writes 에 XML에서 읽은 값 저장
    sys.core[i].dcache.wbb_reads = 1;  // [한국어] sys.core[i].dcache.wbb_reads 에 XML에서 읽은 값 저장
    // system.core?.BTB
    for (j = 0; j < 20; j++) sys.core[i].BTB.BTB_config[j] = 1;
    sys.core[i].BTB.total_accesses = 1;  // [한국어] sys.core[i].BTB.total_accesses 에 XML에서 읽은 값 저장
    sys.core[i].BTB.read_accesses = 1;  // [한국어] sys.core[i].BTB.read_accesses 에 XML에서 읽은 값 저장
    sys.core[i].BTB.write_accesses = 1;  // [한국어] sys.core[i].BTB.write_accesses 에 XML에서 읽은 값 저장
    sys.core[i].BTB.total_hits = 1;  // [한국어] sys.core[i].BTB.total_hits 에 XML에서 읽은 값 저장
    sys.core[i].BTB.total_misses = 1;  // [한국어] sys.core[i].BTB.total_misses 에 XML에서 읽은 값 저장
    sys.core[i].BTB.read_hits = 1;  // [한국어] sys.core[i].BTB.read_hits 에 XML에서 읽은 값 저장
    sys.core[i].BTB.write_hits = 1;  // [한국어] sys.core[i].BTB.write_hits 에 XML에서 읽은 값 저장
    sys.core[i].BTB.read_misses = 1;  // [한국어] sys.core[i].BTB.read_misses 에 XML에서 읽은 값 저장
    sys.core[i].BTB.write_misses = 1;  // [한국어] sys.core[i].BTB.write_misses 에 XML에서 읽은 값 저장
    sys.core[i].BTB.replacements = 1;  // [한국어] sys.core[i].BTB.replacements 에 XML에서 읽은 값 저장
  }

  // system_L1directory
  for (i = 0; i <= 63; i++) {
    for (j = 0; j < 20; j++) sys.L1Directory[i].Dir_config[j] = 1;
    for (j = 0; j < 20; j++) sys.L1Directory[i].buffer_sizes[j] = 1;
    sys.L1Directory[i].clockrate = 1;  // [한국어] sys.L1Directory[i].clockrate 에 XML에서 읽은 값 저장
    sys.L1Directory[i].ports[20] = 1;  // [한국어] sys.L1Directory[i].ports[20] 에 XML에서 읽은 값 저장
    sys.L1Directory[i].device_type = 1;  // [한국어] sys.L1Directory[i].device_type 에 XML에서 읽은 값 저장
    strcpy(sys.L1Directory[i].threeD_stack, "default");
    sys.L1Directory[i].total_accesses = 1;  // [한국어] sys.L1Directory[i].total_accesses 에 XML에서 읽은 값 저장
    sys.L1Directory[i].read_accesses = 1;  // [한국어] sys.L1Directory[i].read_accesses 에 XML에서 읽은 값 저장
    sys.L1Directory[i].write_accesses = 1;  // [한국어] sys.L1Directory[i].write_accesses 에 XML에서 읽은 값 저장
    sys.L1Directory[i].duty_cycle = 1;  // [한국어] sys.L1Directory[i].duty_cycle 에 XML에서 읽은 값 저장
  }
  // system_L2directory
  for (i = 0; i <= 63; i++) {
    for (j = 0; j < 20; j++) sys.L2Directory[i].Dir_config[j] = 1;
    for (j = 0; j < 20; j++) sys.L2Directory[i].buffer_sizes[j] = 1;
    sys.L2Directory[i].clockrate = 1;  // [한국어] sys.L2Directory[i].clockrate 에 XML에서 읽은 값 저장
    sys.L2Directory[i].ports[20] = 1;  // [한국어] sys.L2Directory[i].ports[20] 에 XML에서 읽은 값 저장
    sys.L2Directory[i].device_type = 1;  // [한국어] sys.L2Directory[i].device_type 에 XML에서 읽은 값 저장
    strcpy(sys.L2Directory[i].threeD_stack, "default");
    sys.L2Directory[i].total_accesses = 1;  // [한국어] sys.L2Directory[i].total_accesses 에 XML에서 읽은 값 저장
    sys.L2Directory[i].read_accesses = 1;  // [한국어] sys.L2Directory[i].read_accesses 에 XML에서 읽은 값 저장
    sys.L2Directory[i].write_accesses = 1;  // [한국어] sys.L2Directory[i].write_accesses 에 XML에서 읽은 값 저장
    sys.L2Directory[i].duty_cycle = 1;  // [한국어] sys.L2Directory[i].duty_cycle 에 XML에서 읽은 값 저장
  }
  for (i = 0; i <= 63; i++) {
    // system_L2
    for (j = 0; j < 20; j++) sys.L2[i].L2_config[j] = 1;
    sys.L2[i].clockrate = 1;  // [한국어] sys.L2[i].clockrate 에 XML에서 읽은 값 저장
    for (j = 0; j < 20; j++) sys.L2[i].ports[j] = 1;
    sys.L2[i].device_type = 1;  // [한국어] sys.L2[i].device_type 에 XML에서 읽은 값 저장
    strcpy(sys.L2[i].threeD_stack, "default");
    for (j = 0; j < 20; j++) sys.L2[i].buffer_sizes[j] = 1;
    sys.L2[i].total_accesses = 1;  // [한국어] sys.L2[i].total_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].read_accesses = 1;  // [한국어] sys.L2[i].read_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].write_accesses = 1;  // [한국어] sys.L2[i].write_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].total_hits = 1;  // [한국어] sys.L2[i].total_hits 에 XML에서 읽은 값 저장
    sys.L2[i].total_misses = 1;  // [한국어] sys.L2[i].total_misses 에 XML에서 읽은 값 저장
    sys.L2[i].read_hits = 1;  // [한국어] sys.L2[i].read_hits 에 XML에서 읽은 값 저장
    sys.L2[i].write_hits = 1;  // [한국어] sys.L2[i].write_hits 에 XML에서 읽은 값 저장
    sys.L2[i].read_misses = 1;  // [한국어] sys.L2[i].read_misses 에 XML에서 읽은 값 저장
    sys.L2[i].write_misses = 1;  // [한국어] sys.L2[i].write_misses 에 XML에서 읽은 값 저장
    sys.L2[i].replacements = 1;  // [한국어] sys.L2[i].replacements 에 XML에서 읽은 값 저장
    sys.L2[i].write_backs = 1;  // [한국어] sys.L2[i].write_backs 에 XML에서 읽은 값 저장
    sys.L2[i].miss_buffer_accesses = 1;  // [한국어] sys.L2[i].miss_buffer_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].fill_buffer_accesses = 1;  // [한국어] sys.L2[i].fill_buffer_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].prefetch_buffer_accesses = 1;  // [한국어] sys.L2[i].prefetch_buffer_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].prefetch_buffer_writes = 1;  // [한국어] sys.L2[i].prefetch_buffer_writes 에 XML에서 읽은 값 저장
    sys.L2[i].prefetch_buffer_reads = 1;  // [한국어] sys.L2[i].prefetch_buffer_reads 에 XML에서 읽은 값 저장
    sys.L2[i].prefetch_buffer_hits = 1;  // [한국어] sys.L2[i].prefetch_buffer_hits 에 XML에서 읽은 값 저장
    sys.L2[i].wbb_writes = 1;  // [한국어] sys.L2[i].wbb_writes 에 XML에서 읽은 값 저장
    sys.L2[i].wbb_reads = 1;  // [한국어] sys.L2[i].wbb_reads 에 XML에서 읽은 값 저장
    sys.L2[i].duty_cycle = 1;  // [한국어] sys.L2[i].duty_cycle 에 XML에서 읽은 값 저장
    sys.L2[i].merged_dir = false;  // [한국어] sys.L2[i].merged_dir 에 XML에서 읽은 값 저장
    sys.L2[i].homenode_read_accesses = 1;  // [한국어] sys.L2[i].homenode_read_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].homenode_write_accesses = 1;  // [한국어] sys.L2[i].homenode_write_accesses 에 XML에서 읽은 값 저장
    sys.L2[i].homenode_read_hits = 1;  // [한국어] sys.L2[i].homenode_read_hits 에 XML에서 읽은 값 저장
    sys.L2[i].homenode_write_hits = 1;  // [한국어] sys.L2[i].homenode_write_hits 에 XML에서 읽은 값 저장
    sys.L2[i].homenode_read_misses = 1;  // [한국어] sys.L2[i].homenode_read_misses 에 XML에서 읽은 값 저장
    sys.L2[i].homenode_write_misses = 1;  // [한국어] sys.L2[i].homenode_write_misses 에 XML에서 읽은 값 저장
    sys.L2[i].dir_duty_cycle = 1;  // [한국어] sys.L2[i].dir_duty_cycle 에 XML에서 읽은 값 저장
  }
  for (i = 0; i <= 63; i++) {
    // system_L3
    for (j = 0; j < 20; j++) sys.L3[i].L3_config[j] = 1;
    sys.L3[i].clockrate = 1;  // [한국어] sys.L3[i].clockrate 에 XML에서 읽은 값 저장
    for (j = 0; j < 20; j++) sys.L3[i].ports[j] = 1;
    sys.L3[i].device_type = 1;  // [한국어] sys.L3[i].device_type 에 XML에서 읽은 값 저장
    strcpy(sys.L3[i].threeD_stack, "default");
    for (j = 0; j < 20; j++) sys.L3[i].buffer_sizes[j] = 1;
    sys.L3[i].total_accesses = 1;  // [한국어] sys.L3[i].total_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].read_accesses = 1;  // [한국어] sys.L3[i].read_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].write_accesses = 1;  // [한국어] sys.L3[i].write_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].total_hits = 1;  // [한국어] sys.L3[i].total_hits 에 XML에서 읽은 값 저장
    sys.L3[i].total_misses = 1;  // [한국어] sys.L3[i].total_misses 에 XML에서 읽은 값 저장
    sys.L3[i].read_hits = 1;  // [한국어] sys.L3[i].read_hits 에 XML에서 읽은 값 저장
    sys.L3[i].write_hits = 1;  // [한국어] sys.L3[i].write_hits 에 XML에서 읽은 값 저장
    sys.L3[i].read_misses = 1;  // [한국어] sys.L3[i].read_misses 에 XML에서 읽은 값 저장
    sys.L3[i].write_misses = 1;  // [한국어] sys.L3[i].write_misses 에 XML에서 읽은 값 저장
    sys.L3[i].replacements = 1;  // [한국어] sys.L3[i].replacements 에 XML에서 읽은 값 저장
    sys.L3[i].write_backs = 1;  // [한국어] sys.L3[i].write_backs 에 XML에서 읽은 값 저장
    sys.L3[i].miss_buffer_accesses = 1;  // [한국어] sys.L3[i].miss_buffer_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].fill_buffer_accesses = 1;  // [한국어] sys.L3[i].fill_buffer_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].prefetch_buffer_accesses = 1;  // [한국어] sys.L3[i].prefetch_buffer_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].prefetch_buffer_writes = 1;  // [한국어] sys.L3[i].prefetch_buffer_writes 에 XML에서 읽은 값 저장
    sys.L3[i].prefetch_buffer_reads = 1;  // [한국어] sys.L3[i].prefetch_buffer_reads 에 XML에서 읽은 값 저장
    sys.L3[i].prefetch_buffer_hits = 1;  // [한국어] sys.L3[i].prefetch_buffer_hits 에 XML에서 읽은 값 저장
    sys.L3[i].wbb_writes = 1;  // [한국어] sys.L3[i].wbb_writes 에 XML에서 읽은 값 저장
    sys.L3[i].wbb_reads = 1;  // [한국어] sys.L3[i].wbb_reads 에 XML에서 읽은 값 저장
    sys.L3[i].duty_cycle = 1;  // [한국어] sys.L3[i].duty_cycle 에 XML에서 읽은 값 저장
    sys.L3[i].merged_dir = false;  // [한국어] sys.L3[i].merged_dir 에 XML에서 읽은 값 저장
    sys.L3[i].homenode_read_accesses = 1;  // [한국어] sys.L3[i].homenode_read_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].homenode_write_accesses = 1;  // [한국어] sys.L3[i].homenode_write_accesses 에 XML에서 읽은 값 저장
    sys.L3[i].homenode_read_hits = 1;  // [한국어] sys.L3[i].homenode_read_hits 에 XML에서 읽은 값 저장
    sys.L3[i].homenode_write_hits = 1;  // [한국어] sys.L3[i].homenode_write_hits 에 XML에서 읽은 값 저장
    sys.L3[i].homenode_read_misses = 1;  // [한국어] sys.L3[i].homenode_read_misses 에 XML에서 읽은 값 저장
    sys.L3[i].homenode_write_misses = 1;  // [한국어] sys.L3[i].homenode_write_misses 에 XML에서 읽은 값 저장
    sys.L3[i].dir_duty_cycle = 1;  // [한국어] sys.L3[i].dir_duty_cycle 에 XML에서 읽은 값 저장
  }
  // system_NoC
  for (i = 0; i <= 63; i++) {
    sys.NoC[i].clockrate = 1;  // [한국어] sys.NoC[i].clockrate 에 XML에서 읽은 값 저장
    sys.NoC[i].type = true;  // [한국어] sys.NoC[i].type 에 XML에서 읽은 값 저장
    sys.NoC[i].chip_coverage = 1;  // [한국어] sys.NoC[i].chip_coverage 에 XML에서 읽은 값 저장
    sys.NoC[i].has_global_link = true;  // [한국어] sys.NoC[i].has_global_link 에 XML에서 읽은 값 저장
    strcpy(sys.NoC[i].topology, "default");
    sys.NoC[i].horizontal_nodes = 1;  // [한국어] sys.NoC[i].horizontal_nodes 에 XML에서 읽은 값 저장
    sys.NoC[i].vertical_nodes = 1;  // [한국어] sys.NoC[i].vertical_nodes 에 XML에서 읽은 값 저장
    sys.NoC[i].input_ports = 1;  // [한국어] sys.NoC[i].input_ports 에 XML에서 읽은 값 저장
    sys.NoC[i].output_ports = 1;  // [한국어] sys.NoC[i].output_ports 에 XML에서 읽은 값 저장
    sys.NoC[i].virtual_channel_per_port = 1;  // [한국어] sys.NoC[i].virtual_channel_per_port 에 XML에서 읽은 값 저장
    sys.NoC[i].flit_bits = 1;  // [한국어] sys.NoC[i].flit_bits 에 XML에서 읽은 값 저장
    sys.NoC[i].input_buffer_entries_per_vc = 1;  // [한국어] sys.NoC[i].input_buffer_entries_per_vc 에 XML에서 읽은 값 저장
    sys.NoC[i].total_accesses = 1;  // [한국어] sys.NoC[i].total_accesses 에 XML에서 읽은 값 저장
    sys.NoC[i].duty_cycle = 1;  // [한국어] sys.NoC[i].duty_cycle 에 XML에서 읽은 값 저장
    sys.NoC[i].route_over_perc = 0.5;  // [한국어] sys.NoC[i].route_over_perc 에 XML에서 읽은 값 저장
    for (j = 0; j < 20; j++) sys.NoC[i].ports_of_input_buffer[j] = 1;
    sys.NoC[i].number_of_crossbars = 1;  // [한국어] sys.NoC[i].number_of_crossbars 에 XML에서 읽은 값 저장
    strcpy(sys.NoC[i].crossbar_type, "default");
    strcpy(sys.NoC[i].crosspoint_type, "default");
    // system.NoC?.xbar0;
    sys.NoC[i].xbar0.number_of_inputs_of_crossbars = 1;  // [한국어] sys.NoC[i].xbar0.number_of_inputs_of_crossbars 에 XML에서 읽은 값 저장
    sys.NoC[i].xbar0.number_of_outputs_of_crossbars = 1;  // [한국어] sys.NoC[i].xbar0.number_of_outputs_of_crossbars 에 XML에서 읽은 값 저장
    sys.NoC[i].xbar0.flit_bits = 1;  // [한국어] sys.NoC[i].xbar0.flit_bits 에 XML에서 읽은 값 저장
    sys.NoC[i].xbar0.input_buffer_entries_per_port = 1;  // [한국어] sys.NoC[i].xbar0.input_buffer_entries_per_port 에 XML에서 읽은 값 저장
    sys.NoC[i].xbar0.ports_of_input_buffer[20] = 1;  // [한국어] sys.NoC[i].xbar0.ports_of_input_buffer[20] 에 XML에서 읽은 값 저장
    sys.NoC[i].xbar0.crossbar_accesses = 1;  // [한국어] sys.NoC[i].xbar0.crossbar_accesses 에 XML에서 읽은 값 저장
  }
  // system_mem
  sys.mem.mem_tech_node = 1;  // [한국어] sys.mem.mem_tech_node 에 XML에서 읽은 값 저장
  sys.mem.device_clock = 1;  // [한국어] sys.mem.device_clock 에 XML에서 읽은 값 저장
  sys.mem.capacity_per_channel = 1;  // [한국어] sys.mem.capacity_per_channel 에 XML에서 읽은 값 저장
  sys.mem.number_ranks = 1;  // [한국어] sys.mem.number_ranks 에 XML에서 읽은 값 저장
  sys.mem.peak_transfer_rate = 1;  // [한국어] sys.mem.peak_transfer_rate 에 XML에서 읽은 값 저장
  sys.mem.num_banks_of_DRAM_chip = 1;  // [한국어] sys.mem.num_banks_of_DRAM_chip 에 XML에서 읽은 값 저장
  sys.mem.Block_width_of_DRAM_chip = 1;  // [한국어] sys.mem.Block_width_of_DRAM_chip 에 XML에서 읽은 값 저장
  sys.mem.output_width_of_DRAM_chip = 1;  // [한국어] sys.mem.output_width_of_DRAM_chip 에 XML에서 읽은 값 저장
  sys.mem.page_size_of_DRAM_chip = 1;  // [한국어] sys.mem.page_size_of_DRAM_chip 에 XML에서 읽은 값 저장
  sys.mem.burstlength_of_DRAM_chip = 1;  // [한국어] sys.mem.burstlength_of_DRAM_chip 에 XML에서 읽은 값 저장
  sys.mem.internal_prefetch_of_DRAM_chip = 1;  // [한국어] sys.mem.internal_prefetch_of_DRAM_chip 에 XML에서 읽은 값 저장
  sys.mem.memory_accesses = 1;  // [한국어] sys.mem.memory_accesses 에 XML에서 읽은 값 저장
  sys.mem.memory_reads = 1;  // [한국어] sys.mem.memory_reads 에 XML에서 읽은 값 저장
  sys.mem.memory_writes = 1;  // [한국어] sys.mem.memory_writes 에 XML에서 읽은 값 저장

  // system_mc
  sys.mc.mc_clock = 1;  // [한국어] sys.mc.mc_clock 에 XML에서 읽은 값 저장
  sys.mc.number_mcs = 1;  // [한국어] sys.mc.number_mcs 에 XML에서 읽은 값 저장
  sys.mc.peak_transfer_rate = 1;  // [한국어] sys.mc.peak_transfer_rate 에 XML에서 읽은 값 저장
  sys.mc.memory_channels_per_mc = 1;  // [한국어] sys.mc.memory_channels_per_mc 에 XML에서 읽은 값 저장
  sys.mc.number_ranks = 1;  // [한국어] sys.mc.number_ranks 에 XML에서 읽은 값 저장
  sys.mc.req_window_size_per_channel = 1;  // [한국어] sys.mc.req_window_size_per_channel 에 XML에서 읽은 값 저장
  sys.mc.IO_buffer_size_per_channel = 1;  // [한국어] sys.mc.IO_buffer_size_per_channel 에 XML에서 읽은 값 저장
  sys.mc.databus_width = 1;  // [한국어] sys.mc.databus_width 에 XML에서 읽은 값 저장
  sys.mc.addressbus_width = 1;  // [한국어] sys.mc.addressbus_width 에 XML에서 읽은 값 저장
  sys.mc.memory_accesses = 1;  // [한국어] sys.mc.memory_accesses 에 XML에서 읽은 값 저장
  sys.mc.memory_reads = 1;  // [한국어] sys.mc.memory_reads 에 XML에서 읽은 값 저장
  sys.mc.memory_writes = 1;  // [한국어] sys.mc.memory_writes 에 XML에서 읽은 값 저장
  sys.mc.LVDS = true;  // [한국어] sys.mc.LVDS 에 XML에서 읽은 값 저장
  sys.mc.type = 1;  // [한국어] sys.mc.type 에 XML에서 읽은 값 저장

  // system_niu
  sys.niu.clockrate = 1;  // [한국어] sys.niu.clockrate 에 XML에서 읽은 값 저장
  sys.niu.number_units = 1;  // [한국어] sys.niu.number_units 에 XML에서 읽은 값 저장
  sys.niu.type = 1;  // [한국어] sys.niu.type 에 XML에서 읽은 값 저장
  sys.niu.duty_cycle = 1;  // [한국어] sys.niu.duty_cycle 에 XML에서 읽은 값 저장
  sys.niu.total_load_perc = 1;  // [한국어] sys.niu.total_load_perc 에 XML에서 읽은 값 저장
  // system_pcie
  sys.pcie.clockrate = 1;  // [한국어] sys.pcie.clockrate 에 XML에서 읽은 값 저장
  sys.pcie.number_units = 1;  // [한국어] sys.pcie.number_units 에 XML에서 읽은 값 저장
  sys.pcie.num_channels = 1;  // [한국어] sys.pcie.num_channels 에 XML에서 읽은 값 저장
  sys.pcie.type = 1;  // [한국어] sys.pcie.type 에 XML에서 읽은 값 저장
  sys.pcie.withPHY = false;  // [한국어] sys.pcie.withPHY 에 XML에서 읽은 값 저장
  sys.pcie.duty_cycle = 1;  // [한국어] sys.pcie.duty_cycle 에 XML에서 읽은 값 저장
  sys.pcie.total_load_perc = 1;  // [한국어] sys.pcie.total_load_perc 에 XML에서 읽은 값 저장
  // system_flash_controller
  sys.flashc.mc_clock = 1;  // [한국어] sys.flashc.mc_clock 에 XML에서 읽은 값 저장
  sys.flashc.number_mcs = 1;  // [한국어] sys.flashc.number_mcs 에 XML에서 읽은 값 저장
  sys.flashc.peak_transfer_rate = 1;  // [한국어] sys.flashc.peak_transfer_rate 에 XML에서 읽은 값 저장
  sys.flashc.memory_channels_per_mc = 1;  // [한국어] sys.flashc.memory_channels_per_mc 에 XML에서 읽은 값 저장
  sys.flashc.number_ranks = 1;  // [한국어] sys.flashc.number_ranks 에 XML에서 읽은 값 저장
  sys.flashc.req_window_size_per_channel = 1;  // [한국어] sys.flashc.req_window_size_per_channel 에 XML에서 읽은 값 저장
  sys.flashc.IO_buffer_size_per_channel = 1;  // [한국어] sys.flashc.IO_buffer_size_per_channel 에 XML에서 읽은 값 저장
  sys.flashc.databus_width = 1;  // [한국어] sys.flashc.databus_width 에 XML에서 읽은 값 저장
  sys.flashc.addressbus_width = 1;  // [한국어] sys.flashc.addressbus_width 에 XML에서 읽은 값 저장
  sys.flashc.memory_accesses = 1;  // [한국어] sys.flashc.memory_accesses 에 XML에서 읽은 값 저장
  sys.flashc.memory_reads = 1;  // [한국어] sys.flashc.memory_reads 에 XML에서 읽은 값 저장
  sys.flashc.memory_writes = 1;  // [한국어] sys.flashc.memory_writes 에 XML에서 읽은 값 저장
  sys.flashc.LVDS = true;  // [한국어] sys.flashc.LVDS 에 XML에서 읽은 값 저장
  sys.flashc.withPHY = false;  // [한국어] sys.flashc.withPHY 에 XML에서 읽은 값 저장
  sys.flashc.type = 1;  // [한국어] sys.flashc.type 에 XML에서 읽은 값 저장
  sys.flashc.duty_cycle = 1;  // [한국어] sys.flashc.duty_cycle 에 XML에서 읽은 값 저장
  sys.flashc.total_load_perc = 1;  // [한국어] sys.flashc.total_load_perc 에 XML에서 읽은 값 저장
}
