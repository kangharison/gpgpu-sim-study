/*****************************************************************************
 *                                McPAT/CACTI
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

/*
 * [한국어 설명] CACTI 입출력 및 최상위 인터페이스 구현 (io.cc)
 *
 * === 파일의 역할 ===
 * CACTI(Cache Access and Cycle Time Information)의 입출력 계층과 외부 인터페이스를
 * 구현한다. 주요 책임은 (1) cache.cfg 형식 설정 파일을 파싱하여 InputParameter를
 * 채우는 parse_cfg(), (2) 입력 파라미터를 사람이 읽을 수 있도록 출력하는 display_ip(),
 * (3) 설정 파일/위치 인수/InputParameter 포인터 세 가지 경로로 CACTI 해(solve)를
 * 호출하는 cacti_interface() 오버로드, (4) 최종 결과를 텍스트/CSV로 출력하는
 * output_UCA()/output_data_csv()이다. AccelWattch는 이 파일의
 * cacti_interface(InputParameter*)를 통해 GPU 캐시(L2/공유 메모리/레지스터 파일 등)의
 * 면적·전력·지연을 추정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → CACTI 진입점 계층의 최상위에 해당한다.
 * 호출 체인:
 *   AccelWattch XML 파싱 (XML_Parse.h / ParseXML)
 *     → InputParameter 구성 (g_ip)
 *     → cacti_interface(InputParameter*) [이 파일]
 *         → g_ip->error_checking()
 *         → init_tech_params()
 *         → solve(&fin_res)
 *         → (독립 실행 시) output_UCA(), output_data_csv()
 *   독립 실행형 CACTI의 경우 main.cc가 cacti_interface(string) 또는
 *   cacti_interface(52/54개 인수)를 호출하여 이 파일로 진입한다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 단일 스레드. GPU 디바이스 코드와 무관.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - io.h: cacti_interface() 오버로드 선언, uca_org_t 결과 구조체.
 *   - cacti_interface.h: InputParameter, uca_org_t, g_ip 전역 포인터 정의.
 *   - parameter.h: g_tp 기술 파라미터, init_tech_params() 선언.
 *   - Ucache.h, nuca.h: solve(), update() 선언 (UCA/NUCA 최적 탐색 엔진).
 *   - area.h, basic_circuit.h: powerComponents/powerDef 연산자 오버로드에 사용.
 *   - crossbar.h, arbiter.h: NoC 라우터 계산에 간접 사용.
 * 이 파일에 의존하는 모듈:
 *   - main.cc (CACTI 독립 실행형): 커맨드라인 인수를 받아 cacti_interface() 호출.
 *   - AccelWattch (processor.h, sharedcache.h 등): InputParameter* 기반 인터페이스로
 *     GPU 캐시/NoC 전력 계수 산출.
 *   - cacti_interface.cc 등: 결과 구조체 uca_org_t의 cleanup()로 메모리 해제.
 * 데이터 흐름:
 *   설정 파일/인수 → InputParameter → init_tech_params → solve → uca_org_t
 *   → output_UCA/output_data_csv.
 *
 * === 주요 함수/구조체 요약 ===
 * InputParameter::parse_cfg(): cache.cfg 파일을 한 줄씩 읽어 g_ip 필드(용량,
 *   연관도, 기술 노드, 포트 수, 와이어 타입, 최적화 가중치 등)를 설정.
 * InputParameter::display_ip(): 파싱된 파라미터를 stdout에 출력 (디버그/검증용).
 * powerComponents/powerDef operator+/operator*: 전력 성분의 덧셈 및 스케일링을
 *   지원하는 연산자 오버로드. CACTI 전반의 전력 집계에 사용.
 * cacti_interface(string): -infile로 지정된 cache.cfg 파일 기반 분석.
 * cacti_interface(52/54 ints): CACTI 6.5/McPAT 레거시 위치 인수 기반 분석.
 * cacti_interface(InputParameter*): AccelWattch가 사용하는 포인터 기반 인터페이스.
 * init_interface(): cacti_interface()와 유사하나 solve()를 호출하지 않고 기술
 *   파라미터만 초기화(일부 재구성 시나리오용).
 * reconfigure(): 이미 할당된 InputParameter를 기반으로 solve()를 다시 수행.
 * InputParameter::error_checking(): 입력값의 물리적/수학적 타당성 검증.
 * output_data_csv(): uca_org_t 결과를 out.csv에 추가/생성.
 * output_UCA(): uca_org_t 결과를 사람이 읽기 좋은 텍스트로 stdout에 출력.
 *
 * === AccelWattch XML / gpgpusim.config 연동 ===
 * 이 파일의 parse_cfg()는 CACTI 전용 cache.cfg 파일을 직접 파싱한다. AccelWattch
 * 통합 시에는 대부분의 파라미터가 AccelWattch XML(gpgpusim.config의
 * --power_config_name 옵션으로 지정)에서 McPAT 파서를 거쳐 InputParameter로
 * 변환된 후 cacti_interface(InputParameter*)로 전달된다. 따라서 GPU 캐시 관련
 * XML 옵션(sys.L2[0].L2_config, sys.dram_config, sys.core[].icache/dcache 등)이
 * 최종적으로 이 코드의 g_ip 필드에 반영된다. 독립 실행 시에는 -infile 인수로
 * 전달된 cache.cfg 파일의 항목(-size, -associativity, -technology, -block size,
 * -Cache model, -Wire inside/outside mat, -Interconnect projection 등)이 직접
 * 영향을 준다.
 */

#include <fstream>
#include <iostream>
#include <sstream>


#include "io.h"
#include "area.h"
#include "basic_circuit.h"
#include "parameter.h"
#include "Ucache.h"
#include "nuca.h"
#include "crossbar.h"
#include "arbiter.h"
//#include "highradix.h"

using namespace std;


/*
 * [한국어]
 * InputParameter::parse_cfg — CACTI cache.cfg 설정 파일을 파싱한다.
 *
 * @in_file: 파싱할 cache.cfg 파일 경로. 파일이 없으면 오류 메시지 후 exit(-1).
 * @return: 없음 (void). 파싱 결과는 g_ip/this의 각 멤버 필드에 직접 저장.
 *
 * cache.cfg 파일은 "-옵션 값" 형식의 텍스트 파일이다. 주요 옵션:
 *   -size, -block size, -associativity, -read-write port, -exclusive read/write,
 *   -single ended, -search, -UCA bank, -technology, -operating temperature,
 *   -cache type, -Data/Tag array cell/peripheral type, -design, -deviate,
 *   -Optimize, -NUCAdesign, -NUCAdeviate, -Cache model, -NUCA bank,
 *   -Wire inside/outside mat, -Interconnect projection, -Wire signalling,
 *   -Core, -Cache level, -Print level, -Add ECC, -Print input parameters,
 *   -Force cache config, -Ndbl, -Ndwl, -Nspd, -Ndsam1, -Ndsam2, -Ndcm.
 *
 * 파일의 각 라인을 fscanf로 읽고 strncmp로 옵션 이름을 매칭한 뒤 sscanf로
 * 값을 추출한다. 매칭되지 않는 라인은 무시되며, 매칭된 경우 continue로
 * 다음 라인으로 걸너뛴다. 파일 마지막에는 rpters_in_htree = true로 강제 설정.
 *
 * 호출 체인:
 *   cacti_interface(string) → [parse_cfg()] → error_checking() → init_tech_params → solve
 */
/* Parses "cache.cfg" file */
  void
InputParameter::parse_cfg(const string & in_file)
{
  FILE *fp = fopen(in_file.c_str(), "r"); // [한국어] cache.cfg 파일을 읽기 모드로 오픈
  char line[5000];   // [한국어] 파일에서 읽어온 한 줄의 원본 버퍼
  char jk[5000];     // [한국어] sscanf 포맷 문자열에서 "junk"(무시) 부분을 임시 저장
  char temp_var[5000]; // [한국어] 문자열 값(예: cache type, wire type)을 임시 저장

  if(!fp) {
    cout << in_file << " is missing!\n"; // [한국어] 설정 파일이 없으면 오류 출력
    exit(-1); // [한국어] 파일 누락 시 비정상 종료
  }

  // [한국어] EOF(파일 끝)에 도달할 때까지 한 줄씩 읽어온다
  while(fscanf(fp, "%[^\n]\n", line) != EOF) {

    // [한국어] 캐시 전체 용량(바이트) 파싱 — 예: "-size (bytes) 65536"
    if (!strncmp("-size", line, strlen("-size"))) {
      sscanf(line, "-size %[(:-~)*]%u", jk, &(cache_sz));
      continue;
    }

    // [한국어] 페이지 크기(비트) 파싱 — DRAM 메인 메모리 모델링 시 사용
    if (!strncmp("-page size", line, strlen("-page size"))) {
      sscanf(line, "-page size %[(:-~)*]%u", jk, &(page_sz_bits));
      continue;
    }

    // [한국어] DRAM 버스트 길이 파싱 — 메인 메모리 모드에서의 버스트 전송 단위
    if (!strncmp("-burst length", line, strlen("-burst length"))) {
      sscanf(line, "-burst %[(:-~)*]%u", jk, &(burst_len));
      continue;
    }

    // [한국어] 내장형 프리페치 폭 파싱 — DRAM prefetch width
    if (!strncmp("-internal prefetch width", line, strlen("-internal prefetch width"))) {
      sscanf(line, "-internal prefetch %[(:-~)*]%u", jk, &(int_prefetch_w));
      continue;
    }

    // [한국어] 캐시 라인(블록) 크기(바이트) 파싱 — 예: "-block size (bytes) 64"
    if (!strncmp("-block", line, strlen("-block"))) {
      sscanf(line, "-block size (bytes) %d", &(line_sz));
      continue;
    }

    // [한국어] 연관도(associativity) 파싱 — 0: 완전 연관, 1: 직접 매핑, N: N-way
    if (!strncmp("-associativity", line, strlen("-associativity"))) {
      sscanf(line, "-associativity %d", &(assoc));
      continue;
    }

    // [한국어] 읽기/쓰기 겸용 포트 수 파싱
    if (!strncmp("-read-write", line, strlen("-read-write"))) {
      sscanf(line, "-read-write port %d", &(num_rw_ports));
      continue;
    }

    // [한국어] 전용 읽기 포트 수 파싱
    if (!strncmp("-exclusive read", line, strlen("exclusive read"))) {
      sscanf(line, "-exclusive read port %d", &(num_rd_ports));
      continue;
    }

    // [한국어] 전용 쓰기 포트 수 파싱
    if(!strncmp("-exclusive write", line, strlen("-exclusive write"))) {
      sscanf(line, "-exclusive write port %d", &(num_wr_ports));
      continue;
    }

    // [한국어] 단일 종단(single-ended) 읽기 포트 수 파싱 — 저전력 CAM/SRAM에서 사용
    if (!strncmp("-single ended", line, strlen("-single ended"))) {
      sscanf(line, "-single %[(:-~)*]%d", jk,
          &(num_se_rd_ports));
      continue;
    }

    // [한국어] CAM/완전 연관 캐시용 검색 포트 수 파싱
    if (!strncmp("-search", line, strlen("-search"))) {
      sscanf(line, "-search port %d", &(num_search_ports));
      continue;
    }

    // [한국어] UCA(Uniform Cache Access) 뱅크 수 파싱
    if (!strncmp("-UCA bank", line, strlen("-UCA bank"))) {
      sscanf(line, "-UCA bank%[((:-~)| )*]%d", jk, &(nbanks));
      continue;
    }

    // [한국어] 공정 기술 노드(µm) 파싱 — 예: 0.032(32nm). nm 단위로도 변환 저장
    if (!strncmp("-technology", line, strlen("-technology"))) {
      sscanf(line, "-technology (u) %lf", &(F_sz_um));
      F_sz_nm = F_sz_um*1000;
      continue;
    }

    // [한국어] 출력/입력 버스 폭(비트) 파싱 — 캐시에서 외부로 전달하는 데이터 폭
    if (!strncmp("-output/input", line, strlen("-output/input"))) {
      sscanf(line, "-output/input bus %[(:-~)*]%d", jk, &(out_w));
      continue;
    }

    // [한국어] 동작 온도(K) 파싱 — 누설 전력 계산에 직접 영향
    if (!strncmp("-operating temperature", line, strlen("-operating temperature"))) {
      sscanf(line, "-operating temperature %[(:-~)*]%d", jk, &(temp));
      continue;
    }

    // [한국어] 캐시 유형 파싱 — "cache", "main memory", "cam", "ram" 중 하나
    // is_cache / is_main_mem / pure_cam / pure_ram 플래그를 설정한다
    if (!strncmp("-cache type", line, strlen("-cache type"))) {
      sscanf(line, "-cache type%[^\"]\"%[^\"]\"", jk, temp_var);

      if (!strncmp("cache", temp_var, sizeof("cache"))) {
        is_cache = true;
      }
      else
      {
        is_cache = false;
      }

      if (!strncmp("main memory", temp_var, sizeof("main memory"))) {
        is_main_mem = true;
      }
      else {
        is_main_mem = false;
      }

      if (!strncmp("cam", temp_var, sizeof("cam"))) {
        pure_cam = true;
      }
      else {
        pure_cam = false;
      }

      if (!strncmp("ram", temp_var, sizeof("ram"))) {
        pure_ram = true;
      }
      else {
    	  // [한국어] 메인 메모리가 아닌 경우 기본적으로 pure_ram=false,
    	  // 메인 메모리인 경우 pure_ram=true로 간주
    	  if (!is_main_mem)
    		  pure_ram = false;
    	  else
    		  pure_ram = true;
      }

      continue;
    }


    // [한국어] 태그 비트 수 파싱 — "default"이면 CACTI가 cache_sz/banks/assoc에 따라 자동 계산
    if (!strncmp("-tag size", line, strlen("-tag size"))) {
      sscanf(line, "-tag size%[^\"]\"%[^\"]\"", jk, temp_var);
      if (!strncmp("default", temp_var, sizeof("default"))) {
        specific_tag = false;
        tag_w = 42; /* the acutal value is calculated
                     * later based on the cache size, bank count, and associativity
                     */
      }
      else {
        specific_tag = true;
        sscanf(line, "-tag size (b) %d", &(tag_w));
      }
      continue;
    }

    // [한국어] 접근 모드 파싱 — fast(2)/sequential(1)/normal(0)
    // 태그/데이터 배열 접근 순서와 파이프라인 지연 모델에 영향
    if (!strncmp("-access mode", line, strlen("-access mode"))) {
      sscanf(line, "-access %[^\"]\"%[^\"]\"", jk, temp_var);
      if (!strncmp("fast", temp_var, strlen("fast"))) {
        access_mode = 2;
      }
      else if (!strncmp("sequential", temp_var, strlen("sequential"))) {
        access_mode = 1;
      }
      else if(!strncmp("normal", temp_var, strlen("normal"))) {
        access_mode = 0;
      }
      else {
        cout << "ERROR: Invalid access mode!\n";
        exit(0);
      }
      continue;
    }

    // [한국어] 데이터 어레이 메모리 셀 공정 타입 파싱
    // 0=itrs-hp, 1=itrs-lstp, 2=itrs-lop, 3=lp-dram, 4=comm-dram
    if (!strncmp("-Data array cell type", line, strlen("-Data array cell type"))) {
      sscanf(line, "-Data array cell type %[^\"]\"%[^\"]\"", jk, temp_var);

      if(!strncmp("itrs-hp", temp_var, strlen("itrs-hp"))) {
        data_arr_ram_cell_tech_type = 0;
      }
      else if(!strncmp("itrs-lstp", temp_var, strlen("itrs-lstp"))) {
        data_arr_ram_cell_tech_type = 1;
      }
      else if(!strncmp("itrs-lop", temp_var, strlen("itrs-lop"))) {
        data_arr_ram_cell_tech_type = 2;
      }
      else if(!strncmp("lp-dram", temp_var, strlen("lp-dram"))) {
        data_arr_ram_cell_tech_type = 3;
      }
      else if(!strncmp("comm-dram", temp_var, strlen("comm-dram"))) {
        data_arr_ram_cell_tech_type = 4;
      }
      else {
        cout << "ERROR: Invalid type!\n";
        exit(0);
      }
      continue;
    }

    // [한국어] 데이터 어레이 주변 회로(peripheral) 공정 타입 파싱
    // 0=itrs-hp, 1=itrs-lstp, 2=itrs-lop (DRAM 옵션 없음)
    if (!strncmp("-Data array peripheral type", line, strlen("-Data array peripheral type"))) {
      sscanf(line, "-Data array peripheral type %[^\"]\"%[^\"]\"", jk, temp_var);

      if(!strncmp("itrs-hp", temp_var, strlen("itrs-hp"))) {
        data_arr_peri_global_tech_type = 0;
      }
      else if(!strncmp("itrs-lstp", temp_var, strlen("itrs-lstp"))) {
        data_arr_peri_global_tech_type = 1;
      }
      else if(!strncmp("itrs-lop", temp_var, strlen("itrs-lop"))) {
        data_arr_peri_global_tech_type = 2;
      }
      else {
        cout << "ERROR: Invalid type!\n";
        exit(0);
      }
      continue;
    }

    // [한국어] 태그 어레이 메모리 셀 공정 타입 파싱 (데이터 어레이와 동일한 인코딩)
    if (!strncmp("-Tag array cell type", line, strlen("-Tag array cell type"))) {
      sscanf(line, "-Tag array cell type %[^\"]\"%[^\"]\"", jk, temp_var);

      if(!strncmp("itrs-hp", temp_var, strlen("itrs-hp"))) {
        tag_arr_ram_cell_tech_type = 0;
      }
      else if(!strncmp("itrs-lstp", temp_var, strlen("itrs-lstp"))) {
        tag_arr_ram_cell_tech_type = 1;
      }
      else if(!strncmp("itrs-lop", temp_var, strlen("itrs-lop"))) {
        tag_arr_ram_cell_tech_type = 2;
      }
      else if(!strncmp("lp-dram", temp_var, strlen("lp-dram"))) {
        tag_arr_ram_cell_tech_type = 3;
      }
      else if(!strncmp("comm-dram", temp_var, strlen("comm-dram"))) {
        tag_arr_ram_cell_tech_type = 4;
      }
      else {
        cout << "ERROR: Invalid type!\n";
        exit(0);
      }
      continue;
    }

    // [한국어] 태그 어레이 주변 회로 공정 타입 파싱
    if (!strncmp("-Tag array peripheral type", line, strlen("-Tag array peripheral type"))) {
      sscanf(line, "-Tag array peripheral type %[^\"]\"%[^\"]\"", jk, temp_var);

      if(!strncmp("itrs-hp", temp_var, strlen("itrs-hp"))) {
        tag_arr_peri_global_tech_type = 0;
      }
      else if(!strncmp("itrs-lstp", temp_var, strlen("itrs-lstp"))) {
        tag_arr_peri_global_tech_type = 1;
      }
      else if(!strncmp("itrs-lop", temp_var, strlen("itrs-lop"))) {
        tag_arr_peri_global_tech_type = 2;
      }
      else {
        cout << "ERROR: Invalid type!\n";
        exit(0);
      }
      continue;
    }
    // [한국어] UCA 설계 목표 가중치 파싱 — delay:dynamic_power:leakage_power:cycle_time:area
    if(!strncmp("-design", line, strlen("-design"))) {
      sscanf(line, "-%[((:-~)| |,)*]%d:%d:%d:%d:%d", jk,
          &(delay_wt), &(dynamic_power_wt),
          &(leakage_power_wt),
          &(cycle_time_wt), &(area_wt));
      continue;
    }

    // [한국어] UCA 설계 목표 허용 편차 파싱 — 동일 5가지 항목의 deviate 값
    if(!strncmp("-deviate", line, strlen("-deviate"))) {
      sscanf(line, "-%[((:-~)| |,)*]%d:%d:%d:%d:%d", jk,
          &(delay_dev), &(dynamic_power_dev),
          &(leakage_power_dev),
          &(cycle_time_dev), &(area_dev));
      continue;
    }

    // [한국어] 최적화 목표 파싱 — ED^2(에너지×지연²), ED(에너지×지연), 또는 가중치 기반
    if(!strncmp("-Optimize", line, strlen("-Optimize"))) {
      sscanf(line, "-Optimize  %[^\"]\"%[^\"]\"", jk, temp_var);

      if(!strncmp("ED^2", temp_var, strlen("ED^2"))) {
        ed = 2;
      }
      else if(!strncmp("ED", temp_var, strlen("ED"))) {
        ed = 1;
      }
      else {
        ed = 0;
      }
    }

    // [한국어] NUCA 설계 목표 가중치 파싱
    if(!strncmp("-NUCAdesign", line, strlen("-NUCAdesign"))) {
      sscanf(line, "-%[((:-~)| |,)*]%d:%d:%d:%d:%d", jk,
          &(delay_wt_nuca), &(dynamic_power_wt_nuca),
          &(leakage_power_wt_nuca),
          &(cycle_time_wt_nuca), &(area_wt_nuca));
      continue;
    }

    // [한국어] NUCA 설계 목표 허용 편차 파싱
    if(!strncmp("-NUCAdeviate", line, strlen("-NUCAdeviate"))) {
      sscanf(line, "-%[((:-~)| |,)*]%d:%d:%d:%d:%d", jk,
          &(delay_dev_nuca), &(dynamic_power_dev_nuca),
          &(leakage_power_dev_nuca),
          &(cycle_time_dev_nuca), &(area_dev_nuca));
      continue;
    }

    // [한국어] 캐시 모델 파싱 — UCA(0) 또는 NUCA(1)
    if(!strncmp("-Cache model", line, strlen("-cache model"))) {
      sscanf(line, "-Cache model %[^\"]\"%[^\"]\"", jk, temp_var);

      if (!strncmp("UCA", temp_var, strlen("UCA"))) {
        nuca = 0;
      }
      else {
        nuca = 1;
      }
      continue;
    }

    // [한국어] NUCA 뱅크 수 파싱 — 0이 아니면 force_nuca_bank=1로 강제
    if(!strncmp("-NUCA bank", line, strlen("-NUCA bank"))) {
      sscanf(line, "-NUCA bank count %d", &(nuca_bank_count));

      if (nuca_bank_count != 0) {
        force_nuca_bank = 1;
      }
      continue;
    }

    // [한국어] 매트 낶의 배선 타입 파싱 — global(2)/local(0)/semi-global(1)
    if(!strncmp("-Wire inside mat", line, strlen("-Wire inside mat"))) {
      sscanf(line, "-Wire%[^\"]\"%[^\"]\"", jk, temp_var);

      if (!strncmp("global", temp_var, strlen("global"))) {
        wire_is_mat_type = 2;
        continue;
      }
      else if (!strncmp("local", temp_var, strlen("local"))) {
        wire_is_mat_type = 0;
        continue;
      }
      else {
        wire_is_mat_type = 1;
        continue;
      }
    }

    // [한국어] 매트 외부 배선 타입 파싱 — global(2)/semi-global(1)
    if(!strncmp("-Wire outside mat", line, strlen("-Wire outside mat"))) {
      sscanf(line, "-Wire%[^\"]\"%[^\"]\"", jk, temp_var);

      if (!strncmp("global", temp_var, strlen("global"))) {
        wire_os_mat_type = 2;
      }
      else {
        wire_os_mat_type = 1;
      }
      continue;
    }

    // [한국어] 인터커넥트 예측 타입 파싱 — aggressive(0)/conservative(1)
    if(!strncmp("-Interconnect projection", line, strlen("-Interconnect projection"))) {
      sscanf(line, "-Interconnect projection%[^\"]\"%[^\"]\"", jk, temp_var);

      if (!strncmp("aggressive", temp_var, strlen("aggressive"))) {
        ic_proj_type = 0;
      }
      else {
        ic_proj_type = 1;
      }
      continue;
    }

    // [한국어] 배선 신호 방식 파싱 — default/Global/Global_5/10/20/30/Low_swing
    // force_wiretype가 1이면 wt를 강제 지정, 0이면 CACTI가 자동 탐색
    if(!strncmp("-Wire signalling", line, strlen("-wire signalling"))) {
      sscanf(line, "-Wire%[^\"]\"%[^\"]\"", jk, temp_var);

      if (!strncmp("default", temp_var, strlen("default"))) {
        force_wiretype = 0;
        wt = Global;
      }
      else if (!(strncmp("Global_10", temp_var, strlen("Global_10")))) {
        force_wiretype = 1;
        wt = Global_10;
      }
      else if (!(strncmp("Global_20", temp_var, strlen("Global_20")))) {
        force_wiretype = 1;
        wt = Global_20;
      }
      else if (!(strncmp("Global_30", temp_var, strlen("Global_30")))) {
        force_wiretype = 1;
        wt = Global_30;
      }
      else if (!(strncmp("Global_5", temp_var, strlen("Global_5")))) {
        force_wiretype = 1;
        wt = Global_5;
      }
      else if (!(strncmp("Global", temp_var, strlen("Global")))) {
        force_wiretype = 1;
        wt = Global;
      }
      else {
        wt = Low_swing;
        force_wiretype = 1;
      }
      continue;
    }



    // [한국어] 코어 수 파싱 — NUCA 모델에서 사용, 16개 초과 시 경고
    if(!strncmp("-Core", line, strlen("-Core"))) {
      sscanf(line, "-Core count %d\n", &(cores));
      if (cores > 16) {
        printf("No. of cores should be less than 16!\n");
      }
      continue;
    }

    // [한국어] 캐시 레벨 파싱 — L2(0) 또는 L3(1)
    if(!strncmp("-Cache level", line, strlen("-Cache level"))) {
      sscanf(line, "-Cache l%[^\"]\"%[^\"]\"", jk, temp_var);
      if (!strncmp("L2", temp_var, strlen("L2"))) {
        cache_level = 0;
      }
      else {
        cache_level = 1;
      }
    }

    // [한국어] 출력 상세도 파싱 — DETAILED(1)이면 상세 지연/전력 분해 출력
    if(!strncmp("-Print level", line, strlen("-Print level"))) {
      sscanf(line, "-Print l%[^\"]\"%[^\"]\"", jk, temp_var);
      if (!strncmp("DETAILED", temp_var, strlen("DETAILED"))) {
        print_detail = 1;
      }
      else {
        print_detail = 0;
      }

    }
    // [한국어] ECC 비트 추가 여부 파싱 — true이면 add_ecc_b_=true
    if(!strncmp("-Add ECC", line, strlen("-Add ECC"))) {
      sscanf(line, "-Add ECC %[^\"]\"%[^\"]\"", jk, temp_var);
      if (!strncmp("true", temp_var, strlen("true"))) {
        add_ecc_b_ = true;
      }
      else {
        add_ecc_b_ = false;
      }
    }

    // [한국어] 입력 파라미터 출력 여부 파싱 — true이면 display_ip()가 stdout에 출력
    if(!strncmp("-Print input parameters", line, strlen("-Print input parameters"))) {
      sscanf(line, "-Print input %[^\"]\"%[^\"]\"", jk, temp_var);
      if (!strncmp("true", temp_var, strlen("true"))) {
        print_input_args = true;
      }
      else {
        print_input_args = false;
      }
    }

    // [한국어] 강제 캐시 구성 여부 파싱 — true이면 Ndbl/Ndwl/Nspd/Ndcm/Ndsam1/Ndsam2를
    // CACTI 자동 탐색 대신 사용자 지정값으로 사용
    if(!strncmp("-Force cache config", line, strlen("-Force cache config"))) {
      sscanf(line, "-Force cache %[^\"]\"%[^\"]\"", jk, temp_var);
      if (!strncmp("true", temp_var, strlen("true"))) {
        force_cache_config = true;
      }
      else {
        force_cache_config = false;
      }
    }

    // [한국어] 데이터 서브어레이 수직 분할 수(Ndbl) 강제 지정
    if(!strncmp("-Ndbl", line, strlen("-Ndbl"))) {
      sscanf(line, "-Ndbl %d\n", &(ndbl));
      continue;
    }
    // [한국어] 데이터 서브어레이 수평 분할 수(Ndwl) 강제 지정
    if(!strncmp("-Ndwl", line, strlen("-Ndwl"))) {
      sscanf(line, "-Ndwl %d\n", &(ndwl));
      continue;
    }
    // [한국어] 데이터 서브어레이 열 다중화 비율(Nspd) 강제 지정
    if(!strncmp("-Nspd", line, strlen("-Nspd"))) {
      sscanf(line, "-Nspd %d\n", &(nspd));
      continue;
    }
    // [한국어] 1단계 센스앰프 다중화 수(Ndsam1) 강제 지정
    if(!strncmp("-Ndsam1", line, strlen("-Ndsam1"))) {
      sscanf(line, "-Ndsam1 %d\n", &(ndsam1));
      continue;
    }
    // [한국어] 2단계 센스앰프 다중화 수(Ndsam2) 강제 지정
    if(!strncmp("-Ndsam2", line, strlen("-Ndsam2"))) {
      sscanf(line, "-Ndsam2 %d\n", &(ndsam2));
      continue;
    }
    // [한국어] 열 디코더 다중화 계수(Ndcm) 강제 지정
   if(!strncmp("-Ndcm", line, strlen("-Ndcm"))) {
      sscanf(line, "-Ndcm %d\n", &(ndcm));
      continue;
    }

  }
  // [한국어] H-tree 세그먼트에 리피터 사용을 기본 true로 설정
  rpters_in_htree = true;
  fclose(fp); // [한국어] 설정 파일 닫기
}

  /*
   * [한국어]
   * InputParameter::display_ip — 파싱된 입력 파라미터를 stdout에 출력한다.
   *
   * @return: 없음 (void).
   *
   * cache_sz, line_sz, assoc, 포트 수, 뱅크 수, 기술 노드, 온도, 태그 크기,
   * 캐시 유형, 접근 모드, 셀/주변 회로 기술, 최적화 가중치, NUCA 관련 파라미터,
   * 와이어 타입, 강제 캐시 구성 값 등을 출력한다. print_input_args 플래그가
   * true일 때 cacti_interface() 시작 부분에서 호출되어 입력값을 검증하는 데 사용.
   *
   * 호출 체인:
   *   cacti_interface() → [display_ip()] → stdout 출력
   */
  void
InputParameter::display_ip()
{
  cout << "Cache size                    : " << cache_sz << endl;
  cout << "Block size                    : " << line_sz << endl;
  cout << "Associativity                 : " << assoc << endl;
  cout << "Read only ports               : " << num_rd_ports << endl;
  cout << "Write only ports              : " << num_wr_ports << endl;
  cout << "Read write ports              : " << num_rw_ports << endl;
  cout << "Single ended read ports       : " << num_se_rd_ports << endl;
  if (fully_assoc||pure_cam)
  {
	  cout << "Search ports                  : " << num_search_ports << endl;
  }
  cout << "Cache banks (UCA)             : " << nbanks << endl;
  cout << "Technology                    : " << F_sz_um << endl;
  cout << "Temperature                   : " << temp << endl;
  cout << "Tag size                      : " << tag_w << endl;
  if (is_cache) {
    cout << "array type                    : " << "Cache" << endl;
  }
  if (pure_ram) {
    cout << "array type                    : " << "Scratch RAM" << endl;
  }
  if (pure_cam)
  {
      cout << "array type                    : " << "CAM" << endl;
  }
  cout << "Model as memory               : " << is_main_mem << endl;
  cout << "Access mode                   : " << access_mode << endl;
  cout << "Data array cell type          : " << data_arr_ram_cell_tech_type << endl;
  cout << "Data array peripheral type    : " << data_arr_peri_global_tech_type << endl;
  cout << "Tag array cell type           : " << tag_arr_ram_cell_tech_type << endl;
  cout << "Tag array peripheral type     : " << tag_arr_peri_global_tech_type << endl;
  cout << "Optimization target           : " << ed << endl;
  cout << "Design objective (UCA wt)     : " << delay_wt << " "
                                                << dynamic_power_wt << " " << leakage_power_wt << " " << cycle_time_wt
                                                << " " << area_wt << endl;
  cout << "Design objective (UCA dev)    : " << delay_dev << " "
                                                << dynamic_power_dev << " " << leakage_power_dev << " " << cycle_time_dev
                                                << " " << area_dev << endl;
  if (nuca)
    {
    cout << "Cores                         : " << cores << endl;


    cout << "Design objective (NUCA wt)    : " << delay_wt_nuca << " "
                                                << dynamic_power_wt_nuca << " " << leakage_power_wt_nuca << " " << cycle_time_wt_nuca
                                                << " " << area_wt_nuca << endl;
    cout << "Design objective (NUCA dev)   : " << delay_dev_nuca << " "
                                                << dynamic_power_dev_nuca << " " << leakage_power_dev_nuca << " " << cycle_time_dev_nuca
                                       << " " << area_dev_nuca << endl;
    }
  cout << "Cache model                   : " << nuca << endl;
  cout << "Nuca bank                     : " << nuca_bank_count << endl;
  cout << "Wire inside mat               : " << wire_is_mat_type << endl;
  cout << "Wire outside mat              : " << wire_os_mat_type << endl;
  cout << "Interconnect projection       : " << ic_proj_type << endl;
  cout << "Wire signalling               : " << force_wiretype << endl;
  cout << "Print level                   : " << print_detail << endl;
  cout << "ECC overhead                  : " << add_ecc_b_ << endl;
  cout << "Page size                     : " << page_sz_bits << endl;
  cout << "Burst length                  : " << burst_len << endl;
  cout << "Internal prefetch width       : " << int_prefetch_w << endl;
  cout << "Force cache config            : " << g_ip->force_cache_config << endl;
  if (g_ip->force_cache_config) {
    cout << "Ndwl                          : " << g_ip->ndwl << endl;
    cout << "Ndbl                          : " << g_ip->ndbl << endl;
    cout << "Nspd                          : " << g_ip->nspd << endl;
    cout << "Ndcm                          : " << g_ip->ndcm << endl;
    cout << "Ndsam1                        : " << g_ip->ndsam1 << endl;
    cout << "Ndsam2                        : " << g_ip->ndsam2 << endl;
  }
}



/*
 * [한국어]
 * operator+ (powerComponents) — 두 powerComponents 구조체를 합산한다.
 *
 * @x, @y: 더할 전력 성분 구조체 (dynamic, leakage, gate_leakage, short_circuit,
 *         longer_channel_leakage 필드 포함).
 * @return: 각 필드별 합산 결과를 담은 새 powerComponents.
 *
 * CACTI 전반에서 서브컴포넌트(예: H-tree, 디코더, 비트라인)별 전력을
 * 누적할 때 사용된다. pppm_* 마스크 배열과는 별개로 단순 덧셈 연산자.
 *
 * 호출 체인: Component/powerDef 나이부 합산 → [operator+] (powerComponents)
 */
powerComponents operator+(const powerComponents & x, const powerComponents & y)
{
  powerComponents z;

  z.dynamic = x.dynamic + y.dynamic;
  z.leakage = x.leakage + y.leakage;
  z.gate_leakage  = x.gate_leakage  + y.gate_leakage;
  z.short_circuit = x.short_circuit + y.short_circuit;
  z.longer_channel_leakage = x.longer_channel_leakage + y.longer_channel_leakage;

  return z;
}

/*
 * [한국어]
 * operator* (powerComponents) — powerComponents에 마스크 배열 y를 곱한다.
 *
 * @x: 스케일링할 전력 성분 구조체.
 * @y: 4원소 double 배열 포인터. [dynamic, leakage, gate_leakage, short_circuit]
 *     순서로 각 성분에 곱할 계수를 지정. longer_channel_leakage는 leakage와
 *     동일한 인덱스(1)를 사용한다.
 * @return: 마스크가 적용된 새 powerComponents.
 *
 * pppm[4], pppm_lkg[4], pppm_dyn[4] 등 const.h에 정의된 마스크와 함께 사용되어
 * total_power, 누설 전력, 동적 전력 등을 선택적으로 추출한다.
 *
 * 호출 체인: Component::power * pppm_* → [operator*] (powerComponents)
 */
powerComponents operator*(const powerComponents & x, double const * const y)
{
  powerComponents z;

  z.dynamic = x.dynamic*y[0];
  z.leakage = x.leakage*y[1];
  z.gate_leakage  = x.gate_leakage*y[2];
  z.short_circuit = x.short_circuit*y[3];
  z.longer_channel_leakage = x.longer_channel_leakage*y[1];//longer channel leakage has the same behavior as normal leakage

  return z;
}


/*
 * [한국어]
 * operator+ (powerDef) — 두 powerDef(readOp/writeOp/searchOp)를 합산한다.
 *
 * @x, @y: 더할 powerDef 구조체. 각각 readOp, writeOp, searchOp powerComponents를 포함.
 * @return: readOp/writeOp/searchOp별로 합산된 새 powerDef.
 *
 * 읽기/쓰기/검색 동작별 전력을 합산하여 총 전력을 계산할 때 사용된다.
 *
 * 호출 체인: Component power 합산 → [operator+] (powerDef)
 */
powerDef operator+(const powerDef & x, const powerDef & y)
{
  powerDef z;

  z.readOp   = x.readOp  + y.readOp;
  z.writeOp  = x.writeOp + y.writeOp;
  z.searchOp = x.searchOp + y.searchOp;
  return z;
}

/*
 * [한국어]
 * operator* (powerDef) — powerDef에 마스크 배열 y를 곱한다.
 *
 * @x: 스케일링할 powerDef 구조체.
 * @y: 4원소 double 배열 포인터. readOp, writeOp, searchOp 각각에
 *     동일한 마스크를 적용한다.
 * @return: 마스크가 적용된 새 powerDef.
 *
 * pppm_* 마스크와 함께 사용되어 readOp/writeOp/searchOp 중 특정 전력 성분만
 * 추출하거나 합산할 때 사용된다.
 *
 * 호출 체인: Component::power * pppm_* → [operator*] (powerDef)
 */
powerDef operator*(const powerDef & x, double const * const y)
{
  powerDef z;

  z.readOp   = x.readOp*y;
  z.writeOp  = x.writeOp*y;
  z.searchOp = x.searchOp*y;
  return z;
}

/*
 * [한국어]
 * cacti_interface(string) — cache.cfg 파일 경로를 받아 CACTI 분석을 수행한다.
 *
 * @infile_name: 입력 설정 파일(cache.cfg) 경로.
 * @return: uca_org_t 결과 구조체. access_time, cycle_time, power, area,
 *          data_array2/tag_array2 포인터 등을 포함.
 *
 * 동작 과정:
 *   1) 새 InputParameter 할당.
 *   2) parse_cfg(infile_name)로 설정 파일 파싱.
 *   3) error_checking()으로 입력값 검증.
 *   4) init_tech_params()로 공정 기술 파라미터 초기화.
 *   5) Wire winit로 전역 배선 모델 초기화.
 *   6) nuca==1이면 Nuca 시뮬레이션 수행.
 *   7) solve(&fin_res)로 UCA 최적 설계 탐색.
 *   8) output_UCA(), output_data_csv()로 결과 출력(독립 실행 시).
 *   9) g_ip 해제.
 *
 * 호출 체인:
 *   main.cc (-infile 모드) → [cacti_interface(string)] → parse_cfg → solve
 */
uca_org_t cacti_interface(const string & infile_name)
{

  uca_org_t fin_res;
  //uca_org_t result;
  fin_res.valid = false; // [한국어] 아직 solve()를 수행하지 않았으므로 유효성 false로 초기화

  g_ip = new InputParameter(); // [한국어] 전역 입력 파라미터 할당
  g_ip->parse_cfg(infile_name); // [한국어] cache.cfg 파일 파싱
  if(!g_ip->error_checking()) // [한국어] 입력값 타당성 검증 실패 시 종료
	  exit(0);
  if (g_ip->print_input_args) // [한국어] print_input_args=true이면 파라미터 출력
    g_ip->display_ip();

  init_tech_params(g_ip->F_sz_um, false); // [한국어] 공정 노드 크기에 따른 기술 파라미터 초기화
  Wire winit; // Do not delete this line. It initializes wires. // [한국어] 전역 Wire 정적 초기화(삭제 금지)


//  For HighRadix Only
//  ////  Wire wirea(g_ip->wt, 1000);
//  ////  wirea.print_wire();
//  ////  cout << "Wire Area " << wirea.area.get_area() << " sq. u" << endl;
//  //  winit.print_wire();
//  //
//    HighRadix *hr;
//      hr = new HighRadix();
//      hr->compute_power();
//      hr->print_router();
//    exit(0);
//
//    double sub_switch_sz = 2;
//    double rows = 32;
//    for (int i=0; i<6; i++) {
//      sub_switch_sz = pow(2, i);
//      rows = 64/sub_switch_sz;
//      hr = new HighRadix(sub_switch_sz, rows, .8/* freq */, 64, 2, 64, 0.7);
//      hr->compute_power();
//      hr->print_router();
//      delete hr;
//    }
//  //  HighRadix yarc;
//  //  yarc.compute_power();
//  //  yarc.print_router();
//    winit.print_wire();
//    exit(0);
//  For HighRadix Only End

  // [한국어] NUCA(Non-Uniform Cache Access) 모델이면 Nuca 시뮬레이션 수행
  if (g_ip->nuca == 1)
  {
    Nuca n(&g_tp.peri_global);
    n.sim_nuca();
  }
  g_ip->display_ip(); // [한국어] 최종 입력 파라미터 출력
  solve(&fin_res); // [한국어] UCA 최적 설계 탐색 — 결과를 fin_res에 기록

  output_UCA(&fin_res); // [한국어] 사람이 읽기 좋은 결과를 stdout에 출력
  output_data_csv(fin_res); // [한국어] CSV 결과를 out.csv에 추가

  delete (g_ip); // [한국어] 전역 InputParameter 해제
  return fin_res; // [한국어] 분석 결과 반환 (data_array2/tag_array2는 호출자가 cleanup)
}

/*
 * [한국어]
 * cacti_interface(52개 정수 인수) — CACTI 6.5 레거시 위치 인수 인터페이스.
 *
 * @cache_size ~ @dev_func_cycle_time: 캐시 용량, 라인 크기, 연관도, 포트 수,
 *   뱅크 수, 기술 노드, 출력 폭, 접근 모드, 캐시/메인메모리 플래그, 최적화
 *   가중치/편차, 온도, 와이어 타입, 셀/주변 회로 기술, 인터커넥트 프로젝션,
 *   NUCA 파라미터 등 총 52개.
 * @return: uca_org_t 분석 결과.
 *
 * main.cc에서 argc==53(프로그램명 포함)일 때 호출된다. 각 인수를 g_ip 필드에
 * 매핑한 후 init_tech_params → solve → output_UCA 순으로 처리한다.
 *
 * 호출 체인:
 *   main.cc (argc==53) → [cacti_interface(52 ints)] → solve
 */
//cacti6.5's plain interface, please keep !!!
uca_org_t cacti_interface(
    int cache_size,
    int line_size,
    int associativity,
    int rw_ports,
    int excl_read_ports,
    int excl_write_ports,
    int single_ended_read_ports,
    int banks,
    double tech_node, // in nm
    int page_sz,
    int burst_length,
    int pre_width,
    int output_width,
    int specific_tag,
    int tag_width,
    int access_mode, //0 normal, 1 seq, 2 fast
    int cache, //scratch ram or cache
    int main_mem,
    int obj_func_delay,
    int obj_func_dynamic_power,
    int obj_func_leakage_power,
    int obj_func_area,
    int obj_func_cycle_time,
    int dev_func_delay,
    int dev_func_dynamic_power,
    int dev_func_leakage_power,
    int dev_func_area,
    int dev_func_cycle_time,
    int ed_ed2_none, // 0 - ED, 1 - ED^2, 2 - use weight and deviate
    int temp,
    int wt, //0 - default(search across everything), 1 - global, 2 - 5% delay penalty, 3 - 10%, 4 - 20 %, 5 - 30%, 6 - low-swing
    int data_arr_ram_cell_tech_flavor_in, // 0-4
    int data_arr_peri_global_tech_flavor_in,
    int tag_arr_ram_cell_tech_flavor_in,
    int tag_arr_peri_global_tech_flavor_in,
    int interconnect_projection_type_in, // 0 - aggressive, 1 - normal
    int wire_inside_mat_type_in,
    int wire_outside_mat_type_in,
    int is_nuca, // 0 - UCA, 1 - NUCA
    int core_count,
    int cache_level, // 0 - L2, 1 - L3
    int nuca_bank_count,
    int nuca_obj_func_delay,
    int nuca_obj_func_dynamic_power,
    int nuca_obj_func_leakage_power,
    int nuca_obj_func_area,
    int nuca_obj_func_cycle_time,
    int nuca_dev_func_delay,
    int nuca_dev_func_dynamic_power,
    int nuca_dev_func_leakage_power,
    int nuca_dev_func_area,
    int nuca_dev_func_cycle_time,
    int REPEATERS_IN_HTREE_SEGMENTS_in,//TODO for now only wires with repeaters are supported
    int p_input)
{
  g_ip = new InputParameter(); // [한국어] 전역 입력 파라미터 동적 할당
  g_ip->add_ecc_b_ = true;     // [한국어] ECC 비트 기본 추가

  // [한국어] 데이터/태그 어레이의 메모리 셀 및 주변 회로 기술 타입 설정 (0~4 인덱스)
  g_ip->data_arr_ram_cell_tech_type    = data_arr_ram_cell_tech_flavor_in;
  g_ip->data_arr_peri_global_tech_type = data_arr_peri_global_tech_flavor_in;
  g_ip->tag_arr_ram_cell_tech_type     = tag_arr_ram_cell_tech_flavor_in;
  g_ip->tag_arr_peri_global_tech_type  = tag_arr_peri_global_tech_flavor_in;

  // [한국어] 인터커넥트 투영 타입, 매트 내외부 배선 타입, DRAM 버스트/프리페치/페이지 설정
  g_ip->ic_proj_type     = interconnect_projection_type_in;
  g_ip->wire_is_mat_type = wire_inside_mat_type_in;
  g_ip->wire_os_mat_type = wire_outside_mat_type_in;
  g_ip->burst_len        = burst_length;
  g_ip->int_prefetch_w   = pre_width;
  g_ip->page_sz_bits     = page_sz;

  // [한국어] 캐시 기하 파라미터(용량, 라인 크기, 연관도, 뱅크 수, 출력 폭, 태그 폭)
  g_ip->cache_sz            = cache_size;
  g_ip->line_sz             = line_size;
  g_ip->assoc               = associativity;
  g_ip->nbanks              = banks;
  g_ip->out_w               = output_width;
  g_ip->specific_tag        = specific_tag;

  // [한국어] 태그 폭이 0이면 CACTI가 자동 계산하도록 42(placeholder)로 설정
  if (tag_width == 0) {
    g_ip->tag_w = 42;
  }
  else {
    g_ip->tag_w               = tag_width;
  }

  // [한국어] 접근 모드(0/1/2) 및 UCA 설계 목표 가중치/허용 편차 설정
  g_ip->access_mode         = access_mode;
  g_ip->delay_wt = obj_func_delay;
  g_ip->dynamic_power_wt = obj_func_dynamic_power;
  g_ip->leakage_power_wt = obj_func_leakage_power;
  g_ip->area_wt = obj_func_area;
  g_ip->cycle_time_wt    = obj_func_cycle_time;
  g_ip->delay_dev = dev_func_delay;
  g_ip->dynamic_power_dev = dev_func_dynamic_power;
  g_ip->leakage_power_dev = dev_func_leakage_power;
  g_ip->area_dev = dev_func_area;
  g_ip->cycle_time_dev    = dev_func_cycle_time;
  g_ip->ed = ed_ed2_none;

  // [한국어] wt 값(0~6)에 따라 강제 배선 타입(force_wiretype)과 Wire enum 설정
  switch(wt) {
    case (0):
      g_ip->force_wiretype = 0;
      g_ip->wt = Global;
      break;
    case (1):
      g_ip->force_wiretype = 1;
      g_ip->wt = Global;
      break;
    case (2):
      g_ip->force_wiretype = 1;
      g_ip->wt = Global_5;
      break;
    case (3):
      g_ip->force_wiretype = 1;
      g_ip->wt = Global_10;
      break;
    case (4):
      g_ip->force_wiretype = 1;
      g_ip->wt = Global_20;
      break;
    case (5):
      g_ip->force_wiretype = 1;
      g_ip->wt = Global_30;
      break;
    case (6):
      g_ip->force_wiretype = 1;
      g_ip->wt = Low_swing;
      break;
    default:
      cout << "Unknown wire type!\n";
      exit(0);
  }

  // [한국어] NUCA 설계 목표 가중치/편차 설정
  g_ip->delay_wt_nuca = nuca_obj_func_delay;
  g_ip->dynamic_power_wt_nuca = nuca_obj_func_dynamic_power;
  g_ip->leakage_power_wt_nuca = nuca_obj_func_leakage_power;
  g_ip->area_wt_nuca = nuca_obj_func_area;
  g_ip->cycle_time_wt_nuca    = nuca_obj_func_cycle_time;
  g_ip->delay_dev_nuca = dev_func_delay;
  g_ip->dynamic_power_dev_nuca = nuca_dev_func_dynamic_power;
  g_ip->leakage_power_dev_nuca = nuca_dev_func_leakage_power;
  g_ip->area_dev_nuca = nuca_dev_func_area;
  g_ip->cycle_time_dev_nuca    = nuca_dev_func_cycle_time;
  g_ip->nuca = is_nuca;
  g_ip->nuca_bank_count = nuca_bank_count;
  if(nuca_bank_count > 0) {
    g_ip->force_nuca_bank = 1; // [한국어] NUCA 뱅크 수가 지정되면 강제 사용
  }
  g_ip->cores = core_count;
  g_ip->cache_level = cache_level; // [한국어] 0=L2, 1=L3

  // [한국어] 동작 온도, 공정 노드(nm→µm 변환), 메모리/캐시 플래그, H-tree 리피터 설정
  g_ip->temp = temp;
  g_ip->F_sz_nm         = tech_node;
  g_ip->F_sz_um         = tech_node / 1000;
  g_ip->is_main_mem     = (main_mem != 0) ? true : false;
  g_ip->is_cache        = (cache != 0) ? true : false;
  g_ip->rpters_in_htree = (REPEATERS_IN_HTREE_SEGMENTS_in != 0) ? true : false;

  // [한국어] 포트 수 및 출력/설정 플래그 (레거시 52인수 인터페이스는 강제 캐시 구성 비활성)
  g_ip->num_rw_ports    = rw_ports;
  g_ip->num_rd_ports    = excl_read_ports;
  g_ip->num_wr_ports    = excl_write_ports;
  g_ip->num_se_rd_ports = single_ended_read_ports;
  g_ip->print_detail = 1;
  g_ip->nuca = 0;

  g_ip->wt = Global_5;
  g_ip->force_cache_config = false;
  g_ip->force_wiretype = false;
  g_ip->print_input_args = p_input;


  uca_org_t fin_res;
  fin_res.valid = false; // [한국어] solve() 전까지 결과 유효성 false

  // [한국어] 입력값 검증 후 기술 파라미터 초기화 및 solve 수행
  if (g_ip->error_checking() == false) exit(0);
  if (g_ip->print_input_args)
    g_ip->display_ip();
  init_tech_params(g_ip->F_sz_um, false);
  Wire winit; // Do not delete this line. It initializes wires. // [한국어] 전역 Wire 정적 초기화(삭제 금지)

  // [한국어] NUCA 모델이면 NoC 라우터 시뮬레이션 수행
  if (g_ip->nuca == 1)
  {
    Nuca n(&g_tp.peri_global);
    n.sim_nuca();
  }
  solve(&fin_res); // [한국어] UCA/NUCA 최적 설계 탐색

  output_UCA(&fin_res); // [한국어] 텍스트 결과 출력

  delete (g_ip); // [한국어] 동적 할당한 InputParameter 해제
  return fin_res; // [한국어] 분석 결과 반환
}

/*
 * [한국어]
 * cacti_interface(54개 정수 인수) — McPAT 형식 위치 인수 인터페이스.
 *
 * CACTI 6.5 형식(52개)과 거의 동일하지만 8번째 위치에 search_ports 파라미터가
 * 추가되어 총 54개 인수를 받는다. 나머지 인수들은 6.5 형식에서 한 칸씩 밀린다.
 *
 * @cache_size ~ @ecc: 캐시 설계 파라미터 54개. 자세한 의미는 main.cc 주석 참조.
 * @return: uca_org_t 분석 결과.
 *
 * 호출 체인:
 *   main.cc (argc==55) → [cacti_interface(54 ints)] → solve
 */
//McPAT's plain interface, please keep !!!
uca_org_t cacti_interface(
    int cache_size,
    int line_size,
    int associativity,
    int rw_ports,
    int excl_read_ports,// para5
    int excl_write_ports,
    int single_ended_read_ports,
    int search_ports,
    int banks,
    double tech_node,//para10
    int output_width,
    int specific_tag,
    int tag_width,
    int access_mode,
    int cache,      //para15
    int main_mem,
    int obj_func_delay,
    int obj_func_dynamic_power,
    int obj_func_leakage_power,
    int obj_func_cycle_time, //para20
    int obj_func_area,
    int dev_func_delay,
    int dev_func_dynamic_power,
    int dev_func_leakage_power,
    int dev_func_area, //para25
    int dev_func_cycle_time,
    int ed_ed2_none, // 0 - ED, 1 - ED^2, 2 - use weight and deviate
    int temp,
    int wt, //0 - default(search across everything), 1 - global, 2 - 5% delay penalty, 3 - 10%, 4 - 20 %, 5 - 30%, 6 - low-swing
    int data_arr_ram_cell_tech_flavor_in,//para30
    int data_arr_peri_global_tech_flavor_in,
    int tag_arr_ram_cell_tech_flavor_in,
    int tag_arr_peri_global_tech_flavor_in,
    int interconnect_projection_type_in,
    int wire_inside_mat_type_in,//para35
    int wire_outside_mat_type_in,
    int REPEATERS_IN_HTREE_SEGMENTS_in,
    int VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in,
    int BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in,
    int PAGE_SIZE_BITS_in,//para40
    int BURST_LENGTH_in,
    int INTERNAL_PREFETCH_WIDTH_in,
    int force_wiretype,
    int wiretype,
    int force_config,//para45
    int ndwl,
    int ndbl,
    int nspd,
    int ndcm,
    int ndsam1,//para50
    int ndsam2,
    int ecc)
{
  g_ip = new InputParameter(); // [한국어] 전역 입력 파라미터 동적 할당

  uca_org_t fin_res;
  fin_res.valid = false; // [한국어] solve() 전까지 결과 유효성 false

  // [한국어] 데이터/태그 어레이의 메모리 셀 및 주변 회로 기술 타입 설정
  g_ip->data_arr_ram_cell_tech_type    = data_arr_ram_cell_tech_flavor_in;
  g_ip->data_arr_peri_global_tech_type = data_arr_peri_global_tech_flavor_in;
  g_ip->tag_arr_ram_cell_tech_type     = tag_arr_ram_cell_tech_flavor_in;
  g_ip->tag_arr_peri_global_tech_type  = tag_arr_peri_global_tech_flavor_in;

  // [한국어] 인터커넥트/배선 타입과 DRAM 버스트/프리페치/페이지 설정
  g_ip->ic_proj_type     = interconnect_projection_type_in;
  g_ip->wire_is_mat_type = wire_inside_mat_type_in;
  g_ip->wire_os_mat_type = wire_outside_mat_type_in;
  g_ip->burst_len        = BURST_LENGTH_in;
  g_ip->int_prefetch_w   = INTERNAL_PREFETCH_WIDTH_in;
  g_ip->page_sz_bits     = PAGE_SIZE_BITS_in;

  // [한국어] 캐시 기하 파라미터(용량, 라인, 연관도, 뱅크, 출력 폭, 태그 폭)
  g_ip->cache_sz            = cache_size;
  g_ip->line_sz             = line_size;
  g_ip->assoc               = associativity;
  g_ip->nbanks              = banks;
  g_ip->out_w               = output_width;
  g_ip->specific_tag        = specific_tag;
  if (specific_tag == 0) {
    g_ip->tag_w = 42; // [한국어] 자동 태그 폭 계산 placeholder
  }
  else {
    g_ip->tag_w               = tag_width;
  }

  // [한국어] 접근 모드, UCA 설계 목표 가중치/편차, 온도, 최적화 목표 설정
  g_ip->access_mode         = access_mode;
  g_ip->delay_wt = obj_func_delay;
  g_ip->dynamic_power_wt = obj_func_dynamic_power;
  g_ip->leakage_power_wt = obj_func_leakage_power;
  g_ip->area_wt = obj_func_area;
  g_ip->cycle_time_wt    = obj_func_cycle_time;
  g_ip->delay_dev = dev_func_delay;
  g_ip->dynamic_power_dev = dev_func_dynamic_power;
  g_ip->leakage_power_dev = dev_func_leakage_power;
  g_ip->area_dev = dev_func_area;
  g_ip->cycle_time_dev    = dev_func_cycle_time;
  g_ip->temp = temp;
  g_ip->ed = ed_ed2_none;

  // [한국어] 공정 노드, 캐시/메모리/ CAM/RAM 플래그, H-tree 관련 플래그 설정
  g_ip->F_sz_nm         = tech_node;
  g_ip->F_sz_um         = tech_node / 1000;
  g_ip->is_main_mem     = (main_mem != 0) ? true : false;
  g_ip->is_cache        = (cache ==1) ? true : false;
  g_ip->pure_ram        = (cache ==0) ? true : false;
  g_ip->pure_cam        = (cache ==2) ? true : false;
  g_ip->rpters_in_htree = (REPEATERS_IN_HTREE_SEGMENTS_in != 0) ? true : false;
  g_ip->ver_htree_wires_over_array = VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in;
  g_ip->broadcast_addr_din_over_ver_htrees = BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in;

  // [한국어] 포트 수 (search_ports 포함) 및 출력 상세도/NUCA 플래그 설정
  g_ip->num_rw_ports    = rw_ports;
  g_ip->num_rd_ports    = excl_read_ports;
  g_ip->num_wr_ports    = excl_write_ports;
  g_ip->num_se_rd_ports = single_ended_read_ports;
  g_ip->num_search_ports = search_ports;

  g_ip->print_detail = 1;
  g_ip->nuca = 0;

  // [한국어] 강제 배선 타입 및 Wire enum 설정 (0=자동, 5/10/20/30=Global_X, 0+force=Low_swing)
  if (force_wiretype == 0)
  {
	  g_ip->wt = Global;
      g_ip->force_wiretype = false;
  }
  else
  {   g_ip->force_wiretype = true;
	  if (wiretype==10) {
		  g_ip->wt = Global_10;
	        }
	  if (wiretype==20) {
		  g_ip->wt = Global_20;
	        }
	  if (wiretype==30) {
		  g_ip->wt = Global_30;
	        }
	  if (wiretype==5) {
	      g_ip->wt = Global_5;
	        }
	  if (wiretype==0) {
		  g_ip->wt = Low_swing;
	  }
  }
  // [한국어] 강제 캐시 구성 여부 — 활성화 시 Ndbl/Ndwl/Nspd/Ndcm/Ndsam 값을 사용자 지정값으로 사용
  if (force_config == 0)
    {
  	  g_ip->force_cache_config = false;
    }
    else
    {
    	g_ip->force_cache_config = true;
    	g_ip->ndbl=ndbl;
    	g_ip->ndwl=ndwl;
    	g_ip->nspd=nspd;
    	g_ip->ndcm=ndcm;
    	g_ip->ndsam1=ndsam1;
    	g_ip->ndsam2=ndsam2;


    }

  // [한국어] ECC 비트 추가 여부 설정
  if (ecc==0){
	  g_ip->add_ecc_b_=false;
  }
  else
  {
	  g_ip->add_ecc_b_=true;
  }

  // [한국어] 입력값 검증 후 기술 파라미터 초기화 및 solve 수행
  if(!g_ip->error_checking())
	  exit(0);

  init_tech_params(g_ip->F_sz_um, false);
  Wire winit; // Do not delete this line. It initializes wires. // [한국어] 전역 Wire 정적 초기화(삭제 금지)

  g_ip->display_ip(); // [한국어] 입력 파라미터 출력
  solve(&fin_res);    // [한국어] UCA/NUCA 최적 설계 탐색
  output_UCA(&fin_res);    // [한국어] 텍스트 결과 출력
  output_data_csv(fin_res); // [한국어] CSV 결과 추가
  delete (g_ip); // [한국어] 동적 할당 해제

  return fin_res; // [한국어] 분석 결과 반환
}


/*
 * [한국어]
 * InputParameter::InputParameter — CACTI 입력 파라미터 구조체 기본 생성자.
 *
 * @return: 없음 (생성자).
 *
 * 모든 정수/실수 필드를 0, 모든 bool 플래그를 false, wire type을 Invalid_wtype으로
 * 초기화한다. 이후 parse_cfg() 또는 AccelWattch XML 파서가 각 필드를 채운다.
 *
 * 호출 체인:
 *   cacti_interface() → new InputParameter() → [InputParameter()]
 */
InputParameter::InputParameter()
{
    // [한국어] 캐시 기하 및 접근 모드 관련 필드 초기화
    cache_sz=0;  // in bytes
    line_sz=0;
    assoc=0;
    nbanks=0;
    out_w=0;// == nr_bits_out
    specific_tag=false;
    tag_w=0;
    access_mode=0;
    obj_func_dyn_energy=0;
    obj_func_dyn_power=0;
    obj_func_leak_power=0;
    obj_func_cycle_t=0;

    // [한국어] 공정 노드, 포트 수, 메모리 유형 플래그, H-tree 및 온도 필드 초기화
    F_sz_nm=0;          // feature size in nm
    F_sz_um=0;          // feature size in um
    num_rw_ports=0;
    num_rd_ports=0;
    num_wr_ports=0;
    num_se_rd_ports=0;  // number of single ended read ports
    num_search_ports=0;  // Sheng: number of search ports for CAM
    is_main_mem=false;
    is_cache=false;
    pure_ram=false;
    pure_cam=false;
    rpters_in_htree=false;  // if there are repeaters in htree segment
    ver_htree_wires_over_array=0;
    broadcast_addr_din_over_ver_htrees=0;
    temp=0;

    // [한국어] 메모리 셀/주변 회로 기술 타입 및 DRAM 버스트/프리페치/페이지 초기화
    ram_cell_tech_type=0;
    peri_global_tech_type=0;
    data_arr_ram_cell_tech_type=0;
    data_arr_peri_global_tech_type=0;
    tag_arr_ram_cell_tech_type=0;
    tag_arr_peri_global_tech_type=0;

    burst_len=0;
    int_prefetch_w=0;
    page_sz_bits=0;

    // [한국어] 인터커넥트/배선 타입, 강제 캐시 구성, NUCA 관련 필드 초기화
    ic_proj_type=0;      // interconnect_projection_type
    wire_is_mat_type=0;  // wire_inside_mat_type
    wire_os_mat_type=0; // wire_outside_mat_type
    wt=Invalid_wtype;
    force_wiretype=0;
    print_input_args=false;
    nuca_cache_sz=0; // TODO
    ndbl=0;
    ndwl=0;
    nspd=0;
    ndsam1=0;
    ndsam2=0;
    ndcm=0;
    force_cache_config=false;

    cache_level=0;
    cores=0;
    nuca_bank_count=0;
    force_nuca_bank=0;

    // [한국어] UCA/NUCA 설계 목표 가중치 및 허용 편차 초기화
    delay_wt=0;
    dynamic_power_wt=0;
    leakage_power_wt=0;
    cycle_time_wt=0;
    area_wt=0;
    delay_wt_nuca=0;
    dynamic_power_wt_nuca=0;
    leakage_power_wt_nuca=0;
    cycle_time_wt_nuca=0;
    area_wt_nuca=0;

    delay_dev=0;
    dynamic_power_dev=0;
    leakage_power_dev=0;
    cycle_time_dev=0;
    area_dev=0;
    delay_dev_nuca=0;
    dynamic_power_dev_nuca=0;
    leakage_power_dev_nuca=0;
    cycle_time_dev_nuca=0;
    area_dev_nuca=0;
    ed=0; //ED or ED2 optimization
    nuca=0;

    // [한국어] 접근 모드, 연관도, 세트 수, 출력 상세도, ECC 및 파이프라인 필드 초기화
    fast_access=false;
    block_sz=0;  // bytes
    tag_assoc=0;
    data_assoc=0;
    is_seq_acc=false;
    fully_assoc=false;
    nsets=0;  // == number_of_sets
    print_detail=0;


    add_ecc_b_=false;
    //parameters for design constraint
    throughput=0;
    latency=0;
    pipelinable=false;
    pipeline_stages=0;
    per_stage_vector=0;
    with_clock_grid=false;
}
/*
 * [한국어]
 * InputParameter::error_checking — 파싱된 입력 파라미터의 물리적/수학적 타당성을 검증.
 *
 * @return: true면 입력값이 유효, false면 오류 메시지를 stderr에 출력하고 false 반환
 *          (호출자가 exit(0)으로 종료).
 *
 * 검증 항목:
 *   - access_mode 값(0/1/2)에 따른 seq_access/fast_access 플래그 설정.
 *   - is_main_mem=true일 때 ic_proj_type은 반드시 1(conservative)이어야 함.
 *   - line_sz >= 1, line_sz*8 >= out_w.
 *   - F_sz_um > 0 && <= 0.091 (90nm 이하 공정만 지원).
 *   - 최소 1개 이상의 포트(RWP+ERP+EWP >= 1).
 *   - nbanks가 2의 거듭제곱이고 cache_sz/nbanks >= 64.
 *   - 연관도(assoc)가 2의 거듭제곱, pure_CAM/완전 연관 캐시 규칙.
 *   - 동작 온도 300~400K, 10K 단위.
 *
 * 호출 체인:
 *   parse_cfg → cacti_interface → [error_checking()] → init_tech_params → solve
 */
bool InputParameter::error_checking()
{
  int  A;
  bool seq_access  = false;
  fast_access = true;

  // [한국어] access_mode(0=normal, 1=sequential, 2=fast)에 따라 접근 플래그 설정
  switch (access_mode)
  {
    case 0:
      seq_access  = false;
      fast_access = false;
      break;
    case 1:
      seq_access  = true;
      fast_access = false;
      break;
    case 2:
      seq_access  = false;
      fast_access = true;
      break;
  }

  // [한국어] 메인 메모리(DRAM) 모델은 보수적 인터커넥트 투영만 지원
  if(is_main_mem)
  {
    if(ic_proj_type == 0)
    {
      cerr << "DRAM model supports only conservative interconnect projection!\n\n";
      return false;
    }
  }


  // [한국어] 블록 크기(B)는 1바이트 이상이어야 하고 출력 폭을 수용해야 함
  uint32_t B = line_sz;

  if (B < 1)
  {
    cerr << "Block size must >= 1" << endl;
    return false;
  }
  else if (B*8 < out_w)
  {
    cerr << "Block size must be at least " << out_w/8 << endl;
    return false;
  }

  // [한국어] 공정 피처 사이즈는 0 < F_sz_um <= 0.091(90nm 이하) 범위만 허용
  if (F_sz_um <= 0)
  {
    cerr << "Feature size must be > 0" << endl;
    return false;
  }
  else if (F_sz_um > 0.091)
  {
    cerr << "Feature size must be <= 90 nm" << endl;
    return false;
  }


  // [한국어] 로컬 별명으로 포트 수 저장 (RWP=읽기쓰기, ERP=읽기전용, EWP=쓰기전용, SCHP=검색)
  uint32_t RWP  = num_rw_ports;
  uint32_t ERP  = num_rd_ports;
  uint32_t EWP  = num_wr_ports;
  uint32_t NSER = num_se_rd_ports;
  uint32_t SCHP = num_search_ports;

//TODO: revisit this. This is an important feature. Sheng thought this should be used
//  // If multiple banks and multiple ports are specified, then if number of ports is less than or equal to
//  // the number of banks, we assume that the multiple ports are implemented via the multiple banks.
//  // In such a case we assume that each bank has 1 RWP port.
//  if ((RWP + ERP + EWP) <= nbanks && nbanks>1)
//  {
//    RWP  = 1;
//    ERP  = 0;
//    EWP  = 0;
//    NSER = 0;
//  }
//  else if ((RWP < 0) || (EWP < 0) || (ERP < 0))
//  {
//    cerr << "Ports must >=0" << endl;
//    return false;
//  }
//  else if (RWP > 2)
//  {
//    cerr << "Maximum of 2 read/write ports" << endl;
//    return false;
//  }
//  else if ((RWP+ERP+EWP) < 1)
  // Changed to new implementation:
  // The number of ports specified at input is per bank
  // [한국어] 뱅크당 최소 1개의 읽기/쓰기 포트 필요
  if ((RWP+ERP+EWP) < 1)
  {
    cerr << "Must have at least one port" << endl;
    return false;
  }

  // [한국어] 뱅크 수는 2의 거듭제곱이어야 함
  if (is_pow2(nbanks) == false)
  {
    cerr << "Number of subbanks should be greater than or equal to 1 and should be a power of 2" << endl;
    return false;
  }

  // [한국어] 뱅크당 캐시 용량(C)은 64바이트 이상이어야 함
  int C = cache_sz/nbanks;
  if (C < 64)
  {
    cerr << "Cache size must >=64" << endl;
    return false;
  }

//TODO: revisit this
//   if (pure_ram==true && assoc!=1)
//    {
//  	  cerr << "Pure RAM must have assoc as 1" << endl;
//  	  return false;
//    }

    // [한국어] 완전 연관 및 CAM 관련 규칙 검사
    if (is_cache && assoc==0)
    	fully_assoc =true;
    else
    	fully_assoc = false;

    if (pure_cam==true && assoc!=0)
    {
  	  cerr << "Pure CAM must have associativity as 0" << endl;
  	  return false;
    }

    if (assoc==0 && (pure_cam==false && is_cache ==false))
    {
  	  cerr << "Only CAM or Fully associative cache can have associativity as 0" << endl;
  	  return false;
    }

    // [한국어] 완전 연관/CAM은 데이터/태그 어레이의 셀 및 주변 회로 기술이 동일해야 함
    if ((fully_assoc==true || pure_cam==true)
  		  &&  (data_arr_ram_cell_tech_type!= tag_arr_ram_cell_tech_type
  				 || data_arr_peri_global_tech_type != tag_arr_peri_global_tech_type  ))
    {
  	  cerr << "CAM and fully associative cache must have same device type for both data and tag array" << endl;
  	  return false;
    }

    // [한국어] DRAM 기반 CAM/완전 연관 캐시는 미지원
    if ((fully_assoc==true || pure_cam==true)
  		  &&  (data_arr_ram_cell_tech_type== lp_dram || data_arr_ram_cell_tech_type== comm_dram))
    {
  	  cerr << "DRAM based CAM and fully associative cache are not supported" << endl;
  	  return false;
    }

    // [한국어] CAM/완전 연관 캐시는 메인 메모리 모델로 사용 불가
    if ((fully_assoc==true || pure_cam==true)
  		  &&  (is_main_mem==true))
    {
  	  cerr << "CAM and fully associative cache cannot be as main memory" << endl;
  	  return false;
    }

    // [한국어] CAM/완전 연관 캐시는 최소 1개의 검색 포트 필요
    if ((fully_assoc || pure_cam) && SCHP<1)
    {
	  cerr << "CAM and fully associative must have at least 1 search port" << endl;
	  return false;
    }

    // [한국어] RWP/ERP가 없는 CAM/완전 연관의 경우 검색 포트를 읽기 포트로 간주
   if (RWP==0 && ERP==0 && SCHP>0 && ((fully_assoc || pure_cam)))
    {
  	  ERP=SCHP;
    }

//    if ((!(fully_assoc || pure_cam)) && SCHP>=1)
//    {
//	  cerr << "None CAM and fully associative cannot have search ports" << endl;
//	  return false;
//    }

  // [한국어] 연관도(A) 계산: 0이면 완전 연관(세트 수=C/B), 1이면 직접 매핑, 그 외 2의 거듭제곱
  if (assoc == 0)
  {
    A = C/B;
    //fully_assoc = true;
  }
  else
  {
    if (assoc == 1)
    {
      A = 1;
      //fully_assoc = false;
    }
    else
    {
      //fully_assoc = false;
      A = assoc;
      if (is_pow2(A) == false)
      {
        cerr << "Associativity must be a power of 2" << endl;
        return false;
      }
    }
  }

  // [한국어] 세트 수가 1 이하이면 완전 연관을 사용하거나 파라미터를 조정해야 함
  if (C/(B*A) <= 1 && assoc!=0)
  {
    cerr << "Number of sets is too small: " << endl;
    cerr << " Need to either increase cache size, or decrease associativity or block size" << endl;
    cerr << " (or use fully associative cache)" << endl;
    return false;
  }

  block_sz = B;

  /*dt: testing sequential access mode*/
  // [한국어] 순차 접근 모드이면 데이터 연관도를 1로, 아니면 A로 설정
  if(seq_access)
  {
    tag_assoc  = A;
    data_assoc = 1;
    is_seq_acc = true;
  }
  else
  {
    tag_assoc  = A;
    data_assoc = A;
    is_seq_acc = false;
  }

  if (assoc==0)
  {
    data_assoc = 1; // [한국어] 완전 연관의 경우 데이터 연관도는 의미상 1
  }
  // [한국어] 검증을 거친 포트 수 및 세트 수를 멤버에 반영
  num_rw_ports     = RWP;
  num_rd_ports     = ERP;
  num_wr_ports     = EWP;
  num_se_rd_ports  = NSER;
  if (!(fully_assoc || pure_cam))
    num_search_ports = 0;
  nsets            = C/(B*A);

  // [한국어] 동작 온도는 300~400K 범위이며 10K 단위여야 함
  if (temp < 300 || temp > 400 || temp%10 != 0)
  {
    cerr << temp << " Temperature must be between 300 and 400 Kelvin and multiple of 10." << endl;
    return false;
  }

  if (nsets < 1)
  {
    cerr << "Less than one set..." << endl;
    return false;
  }

  return true;
}



/*
 * [한국어]
 * output_data_csv — CACTI 분석 결과를 out.csv 파일에 추가한다.
 *
 * @fin_res: 출력할 UCA 결과 구조체 (const 참조).
 * @return: 없음 (void).
 *
 * out.csv가 없으면 헤더 행을 먼저 출력한 뒤, 결과 행을 추가한다.
 * 출력 항목: 기술 노드, 용량, 뱅크 수, 연관도, 출력 폭, 접근 시간, 사이클 시간,
 * 동적 검색/읽기/쓰기 에너지, 대기 누설 전력, 면적, 데이터/태그 어레이의
 * Ndwl/Ndbl/Nspd/Ndcm/Ndsam 레벨 및 면적 효율 등.
 *
 * 호출 체인:
 *   cacti_interface() → [output_data_csv()] → out.csv 파일 쓰기
 */
void output_data_csv(const uca_org_t & fin_res)
{
  //TODO: the csv output should remain
  // [한국어] out.csv가 이미 존재하는지 확인하여 헤더 출력 여부 결정
  fstream file("out.csv", ios::in);
  bool    print_index = file.fail();
  file.close();

  // [한국어] out.csv를 append 모드로 열어 결과 행 추가
  file.open("out.csv", ios::out|ios::app);
  if (file.fail() == true)
  {
    cerr << "File out.csv could not be opened successfully" << endl;
  }
  else
  {
    // [한국어] 파일이 없어 새로 생성된 경우에만 컬럼 헤더 출력
    if (print_index == true)
    {
      file << "Tech node (nm), ";
      file << "Capacity (bytes), ";
      file << "Number of banks, ";
      file << "Associativity, ";
      file << "Output width (bits), ";
      file << "Access time (ns), ";
      file << "Random cycle time (ns), ";
//      file << "Multisubbank interleave cycle time (ns), ";

//      file << "Delay request network (ns), ";
//      file << "Delay inside mat (ns), ";
//      file << "Delay reply network (ns), ";
//      file << "Tag array access time (ns), ";
//      file << "Data array access time (ns), ";
//      file << "Refresh period (microsec), ";
//      file << "DRAM array availability (%), ";
      file << "Dynamic search energy (nJ), ";
      file << "Dynamic read energy (nJ), ";
      file << "Dynamic write energy (nJ), ";
//      file << "Tag Dynamic read energy (nJ), ";
//      file << "Data Dynamic read energy (nJ), ";
//      file << "Dynamic read power (mW), ";
      file << "Standby leakage per bank(mW), ";
//      file << "Leakage per bank with leak power management (mW), ";
//      file << "Leakage per bank with leak power management (mW), ";
//      file << "Refresh power as percentage of standby leakage, ";
      file << "Area (mm2), ";
      file << "Ndwl, ";
      file << "Ndbl, ";
      file << "Nspd, ";
      file << "Ndcm, ";
      file << "Ndsam_level_1, ";
      file << "Ndsam_level_2, ";
      file << "Data arrary area efficiency %, ";
      file << "Ntwl, ";
      file << "Ntbl, ";
      file << "Ntspd, ";
      file << "Ntcm, ";
      file << "Ntsam_level_1, ";
      file << "Ntsam_level_2, ";
      file << "Tag arrary area efficiency %, ";

//      file << "Resistance per unit micron (ohm-micron), ";
//      file << "Capacitance per unit micron (fF per micron), ";
//      file << "Unit-length wire delay (ps), ";
//      file << "FO4 delay (ps), ";
//      file << "delay route to bank (including crossb delay) (ps), ";
//      file << "Crossbar delay (ps), ";
//      file << "Dyn read energy per access from closed page (nJ), ";
//      file << "Dyn read energy per access from open page (nJ), ";
//      file << "Leak power of an subbank with page closed (mW), ";
//      file << "Leak power of a subbank with page  open (mW), ";
//      file << "Leak power of request and reply networks (mW), ";
//      file << "Number of subbanks, ";
//      file << "Page size in bits, ";
//      file << "Activate power, ";
//      file << "Read power, ";
//      file << "Write power, ";
//      file << "Precharge power, ";
//      file << "tRCD, ";
//      file << "CAS latency, ";
//      file << "Precharge delay, ";
//      file << "Perc dyn energy bitlines, ";
//      file << "perc dyn energy wordlines, ";
//      file << "perc dyn energy outside mat, ";
//      file << "Area opt (perc), ";
//      file << "Delay opt (perc), ";
//      file << "Repeater opt (perc), ";
//      file << "Aspect ratio";
      file << endl;
    }
    // [한국어] 기술 노드, 용량, 뱅크 수, 연관도, 출력 폭, 접근/사이클 시간 출력
    file << g_ip->F_sz_nm << ", ";
    file << g_ip->cache_sz << ", ";
    file << g_ip->nbanks << ", ";
    file << g_ip->tag_assoc << ", ";
    file << g_ip->out_w << ", ";
    file << fin_res.access_time*1e+9 << ", ";
    file << fin_res.cycle_time*1e+9 << ", ";
//    file << fin_res.data_array2->multisubbank_interleave_cycle_time*1e+9 << ", ";
//    file << fin_res.data_array2->delay_request_network*1e+9 << ", ";
//    file << fin_res.data_array2->delay_inside_mat*1e+9 <<  ", ";
//    file << fin_res.data_array2.delay_reply_network*1e+9 << ", ";

//    if (!(g_ip->fully_assoc || g_ip->pure_cam || g_ip->pure_ram))
//        {
//    	  file << fin_res.tag_array2->access_time*1e+9 << ", ";
//        }
//    else
//    {
//    	file << 0 << ", ";
//    }
//    file << fin_res.data_array2->access_time*1e+9 << ", ";
//    file << fin_res.data_array2->dram_refresh_period*1e+6 << ", ";
//    file << fin_res.data_array2->dram_array_availability <<  ", ";
    // [한국어] CAM/완전 연관인 경우 검색 에너지를, 아니면 N/A 출력
    if (g_ip->fully_assoc || g_ip->pure_cam)
    {
    	file << fin_res.power.searchOp.dynamic*1e+9 << ", ";
    }
    	else
    {
    		file << "N/A" << ", ";
    }
    // [한국어] 읽기/쓰기 동적 에너지(nJ) 출력
    file << fin_res.power.readOp.dynamic*1e+9 << ", ";
    file << fin_res.power.writeOp.dynamic*1e+9 << ", ";
//    if (!(g_ip->fully_assoc || g_ip->pure_cam || g_ip->pure_ram))
//        {
//        	file << fin_res.tag_array2->power.readOp.dynamic*1e+9 << ", ";
//        }
//        	else
//        {
//        		file << "NA" << ", ";
//        }
//    file << fin_res.data_array2->power.readOp.dynamic*1e+9 << ", ";
//    if (g_ip->fully_assoc || g_ip->pure_cam)
//        {
//    	    file << fin_res.power.searchOp.dynamic*1000/fin_res.cycle_time << ", ";
//        }
//        	else
//        {
//        	file << fin_res.power.readOp.dynamic*1000/fin_res.cycle_time << ", ";
//        }

    // [한국어] 누설+게이트 누설 전력(mW) 및 캐시 면적(mm²) 출력
    file <<( fin_res.power.readOp.leakage + fin_res.power.readOp.gate_leakage )*1000 << ", ";
//    file << fin_res.leak_power_with_sleep_transistors_in_mats*1000 << ", ";
//    file << fin_res.data_array.refresh_power / fin_res.data_array.total_power.readOp.leakage << ", ";
    file << fin_res.area*1e-6 << ", ";

    // [한국어] 데이터 어레이의 최적 서브어레이 구성(Ndwl/Ndbl/Nspd/Ndcm/Ndsam) 및 면적 효율 출력
    file << fin_res.data_array2->Ndwl << ", ";
    file << fin_res.data_array2->Ndbl << ", ";
    file << fin_res.data_array2->Nspd << ", ";
    file << fin_res.data_array2->deg_bl_muxing << ", ";
    file << fin_res.data_array2->Ndsam_lev_1 << ", ";
    file << fin_res.data_array2->Ndsam_lev_2 << ", ";
    file << fin_res.data_array2->area_efficiency << ", ";
    // [한국어] 태그 어레이는 RAM/CAM/완전 연관이 아닌 일반 캐시에서만有意義
    if (!(g_ip->fully_assoc || g_ip->pure_cam || g_ip->pure_ram))
    {
    file << fin_res.tag_array2->Ndwl << ", ";
    file << fin_res.tag_array2->Ndbl << ", ";
    file << fin_res.tag_array2->Nspd << ", ";
    file << fin_res.tag_array2->deg_bl_muxing << ", ";
    file << fin_res.tag_array2->Ndsam_lev_1 << ", ";
    file << fin_res.tag_array2->Ndsam_lev_2 << ", ";
    file << fin_res.tag_array2->area_efficiency << ", ";
    }
    else
    {
    	file << "N/A" << ", ";
    	file << "N/A"<< ", ";
    	file << "N/A" << ", ";
    	file << "N/A" << ", ";
    	file << "N/A" << ", ";
    	file << "N/A" << ", ";
    	file << "N/A" << ", ";
    }

//    file << g_tp.wire_inside_mat.R_per_um << ", ";
//    file << g_tp.wire_inside_mat.C_per_um / 1e-15 << ", ";
//    file << g_tp.unit_len_wire_del / 1e-12 << ", ";
//    file << g_tp.FO4 / 1e-12 << ", ";
//    file << fin_res.data_array.delay_route_to_bank / 1e-9 << ", ";
//    file << fin_res.data_array.delay_crossbar / 1e-9 << ", ";
//    file << fin_res.data_array.dyn_read_energy_from_closed_page / 1e-9 << ", ";
//    file << fin_res.data_array.dyn_read_energy_from_open_page / 1e-9 << ", ";
//    file << fin_res.data_array.leak_power_subbank_closed_page / 1e-3 << ", ";
//    file << fin_res.data_array.leak_power_subbank_open_page / 1e-3 << ", ";
//    file << fin_res.data_array.leak_power_request_and_reply_networks / 1e-3 << ", ";
//    file << fin_res.data_array.number_subbanks << ", " ;
//    file << fin_res.data_array.page_size_in_bits << ", " ;
//    file << fin_res.data_array.activate_energy * 1e9 << ", " ;
//    file << fin_res.data_array.read_energy * 1e9 << ", " ;
//    file << fin_res.data_array.write_energy * 1e9 << ", " ;
//    file << fin_res.data_array.precharge_energy * 1e9 << ", " ;
//    file << fin_res.data_array.trcd * 1e9 << ", " ;
//    file << fin_res.data_array.cas_latency * 1e9 << ", " ;
//    file << fin_res.data_array.precharge_delay * 1e9 << ", " ;
//    file << fin_res.data_array.all_banks_height / fin_res.data_array.all_banks_width;
    file<<endl;
  }
  file.close();
}



/*
 * [한국어]
 * output_UCA — CACTI 분석 결과를 사람이 읽기 좋은 텍스트 형태로 stdout에 출력한다.
 *
 * @fr: 출력할 UCA 결과 구조체 포인터.
 * @return: 없음 (void).
 *
 * 출력 항목:
 *   - 캐시 파라미터(용량, 뱅크 수, 연관도, 블록 크기, 포트 수, 기술 노드).
 *   - 접근 시간, 사이클 시간, DRAM 모드일 경우 precharge/activate/read/write 에너지.
 *   - SRAM/캐시 모드일 경우 동적 읽기/쓰기/검색 에너지, 누설 전력.
 *   - 캐시 면적(height×width), 최적 서브어레이 구성(Ndwl/Ndbl/Nspd/Ndcm/Ndsam).
 *   - 상세 모드(print_detail)에서 지연/전력/면적 분해 항목.
 *
 * 호출 체인:
 *   cacti_interface() → [output_UCA()] → stdout 출력
 */
void output_UCA(uca_org_t *fr)
{
  //    if (NUCA)
  // [한국어] 모델 유형에 따라 헤더 출력 (LP-DRAM/Commodity-DRAM/SRAM)
  if (0) {
    cout << "\n\n Detailed Bank Stats:\n";
    cout << "    Bank Size (bytes): %d\n" <<
                                     (int) (g_ip->cache_sz);
  }
  else {
    if (g_ip->data_arr_ram_cell_tech_type == 3) {
      cout << "\n---------- CACTI version 6.5, Uniform Cache Access " <<
        "Logic Process Based DRAM Model ----------\n";
    }
    else if (g_ip->data_arr_ram_cell_tech_type == 4) {
      cout << "\n---------- CACTI version 6.5, Uniform" <<
        "Cache Access Commodity DRAM Model ----------\n";
    }
    else {
      cout << "\n---------- CACTI version 6.5, Uniform Cache Access "
        "SRAM Model ----------\n";
    }
    cout << "\nCache Parameters:\n";
    cout << "    Total cache size (bytes): " <<
      (int) (g_ip->cache_sz) << endl;
  }

  // [한국어] 뱅크 수, 연관도, 블록 크기, 포트 수, 공정 노드 출력
  cout << "    Number of banks: " << (int) g_ip->nbanks << endl;
  if (g_ip->fully_assoc|| g_ip->pure_cam)
    cout << "    Associativity: fully associative\n";
  else {
    if (g_ip->tag_assoc == 1)
      cout << "    Associativity: direct mapped\n";
    else
      cout << "    Associativity: " <<
        g_ip->tag_assoc << endl;
  }


  cout << "    Block size (bytes): " << g_ip->line_sz << endl;
  cout << "    Read/write Ports: " <<
    g_ip->num_rw_ports << endl;
  cout << "    Read ports: " <<
    g_ip->num_rd_ports << endl;
  cout << "    Write ports: " <<
    g_ip->num_wr_ports << endl;
  if (g_ip->fully_assoc|| g_ip->pure_cam)
	  cout << "    search ports: " <<
	      g_ip->num_search_ports << endl;
  cout << "    Technology size (nm): " <<
    g_ip->F_sz_nm << endl << endl;

  // [한국어] 접근 시간 및 사이클 시간 출력
  cout << "    Access time (ns): " << fr->access_time*1e9 << endl;
  cout << "    Cycle time (ns):  " << fr->cycle_time*1e9 << endl;
  // [한국어] Commodity DRAM 모드일 경우 DRAM 특화 지연/에너지/누설 출력
  if (g_ip->data_arr_ram_cell_tech_type >= 4) {
    cout << "    Precharge Delay (ns): " << fr->data_array2->precharge_delay*1e9 << endl;
    cout << "    Activate Energy (nJ): " << fr->data_array2->activate_energy*1e9 << endl;
    cout << "    Read Energy (nJ): " << fr->data_array2->read_energy*1e9 << endl;
    cout << "    Write Energy (nJ): " << fr->data_array2->write_energy*1e9 << endl;
    cout << "    Precharge Energy (nJ): " << fr->data_array2->precharge_energy*1e9 << endl;
    cout << "    Leakage Power Closed Page (mW): " << fr->data_array2->leak_power_subbank_closed_page*1e3 << endl;
    cout << "    Leakage Power Open Page (mW): " << fr->data_array2->leak_power_subbank_open_page*1e3 << endl;
    cout << "    Leakage Power I/O (mW): " << fr->data_array2->leak_power_request_and_reply_networks*1e3 << endl;
    cout << "    Refresh power (mW): " <<
      fr->data_array2->refresh_power*1e3 << endl;
  }
  else {
	  // [한국어] SRAM/캐시 모드: CAM/완전 연관이면 검색 에너지를, 일반 캐시는 읽기/쓰기 에너지 출력
	  if ((g_ip->fully_assoc|| g_ip->pure_cam))
	  {
		  cout << "    Total dynamic associative search energy per access (nJ): " <<
		  fr->power.searchOp.dynamic*1e9 << endl;
//		  cout << "    Total dynamic read energy per access (nJ): " <<
//		  fr->power.readOp.dynamic*1e9 << endl;
//		  cout << "    Total dynamic write energy per access (nJ): " <<
//		  fr->power.writeOp.dynamic*1e9 << endl;
	  }
//	  else
//	  {
		  cout << "    Total dynamic read energy per access (nJ): " <<
		  fr->power.readOp.dynamic*1e9 << endl;
		  cout << "    Total dynamic write energy per access (nJ): " <<
		  fr->power.writeOp.dynamic*1e9 << endl;
//	  }
	  cout << "    Total leakage power of a bank"
	  " (mW): " << fr->power.readOp.leakage*1e3 << endl;
	  cout << "    Total gate leakage power of a bank"
	  " (mW): " << fr->power.readOp.gate_leakage*1e3 << endl;
  }

  if (g_ip->data_arr_ram_cell_tech_type ==3 || g_ip->data_arr_ram_cell_tech_type ==4)
  {
  }
  // [한국어] 캐시 전체 높이×폭(mm) 출력
  cout <<  "    Cache height x width (mm): " <<
    fr->cache_ht*1e-3 << " x " << fr->cache_len*1e-3 << endl << endl;

  // [한국어] 데이터 어레이 최적 서브어레이 구성 출력
  cout << "    Best Ndwl : " << fr->data_array2->Ndwl << endl;
  cout << "    Best Ndbl : " << fr->data_array2->Ndbl << endl;
  cout << "    Best Nspd : " << fr->data_array2->Nspd << endl;
  cout << "    Best Ndcm : " << fr->data_array2->deg_bl_muxing << endl;
  cout << "    Best Ndsam L1 : " << fr->data_array2->Ndsam_lev_1 << endl;
  cout << "    Best Ndsam L2 : " << fr->data_array2->Ndsam_lev_2 << endl << endl;

  // [한국어] 일반 캐시(태그+데이터)인 경우 태그 어레이 최적 구성도 출력
  if ((!(g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)) && !g_ip->is_main_mem)
  {
    cout << "    Best Ntwl : " << fr->tag_array2->Ndwl << endl;
    cout << "    Best Ntbl : " << fr->tag_array2->Ndbl << endl;
    cout << "    Best Ntspd : " << fr->tag_array2->Nspd << endl;
    cout << "    Best Ntcm : " << fr->tag_array2->deg_bl_muxing << endl;
    cout << "    Best Ntsam L1 : " << fr->tag_array2->Ndsam_lev_1 << endl;
    cout << "    Best Ntsam L2 : " << fr->tag_array2->Ndsam_lev_2 << endl;
  }

  // [한국어] 데이터 어레이 H-tree 배선 타입 출력
  switch (fr->data_array2->wt) {
    case (0):
      cout <<  "    Data array, H-tree wire type: Delay optimized global wires\n";
      break;
    case (1):
      cout <<  "    Data array, H-tree wire type: Global wires with 5\% delay penalty\n";
      break;
    case (2):
      cout <<  "    Data array, H-tree wire type: Global wires with 10\% delay penalty\n";
      break;
    case (3):
      cout <<  "    Data array, H-tree wire type: Global wires with 20\% delay penalty\n";
      break;
    case (4):
      cout <<  "    Data array, H-tree wire type: Global wires with 30\% delay penalty\n";
      break;
    case (5):
      cout <<  "    Data array, wire type: Low swing wires\n";
      break;
    default:
      cout << "ERROR - Unknown wire type " << (int) fr->data_array2->wt <<endl;
      exit(0);
  }

  // [한국어] 태그 어레이 H-tree 배선 타입 출력 (일반 캐시의 경우)
  if (!(g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)) {
    switch (fr->tag_array2->wt) {
      case (0):
        cout <<  "    Tag array, H-tree wire type: Delay optimized global wires\n";
        break;
      case (1):
        cout <<  "    Tag array, H-tree wire type: Global wires with 5\% delay penalty\n";
        break;
      case (2):
        cout <<  "    Tag array, H-tree wire type: Global wires with 10\% delay penalty\n";
        break;
      case (3):
        cout <<  "    Tag array, H-tree wire type: Global wires with 20\% delay penalty\n";
        break;
      case (4):
        cout <<  "    Tag array, H-tree wire type: Global wires with 30\% delay penalty\n";
        break;
      case (5):
        cout <<  "    Tag array, wire type: Low swing wires\n";
        break;
      default:
        cout << "ERROR - Unknown wire type " << (int) fr->tag_array2->wt <<endl;
        exit(-1);
    }
  }

  // [한국어] 상세 출력 모드일 때 지연/전력/면적 분해 항목 출력
  if (g_ip->print_detail)
  {
    //if(g_ip->fully_assoc) return;

    /* Delay stats */
    /* data array stats */
    cout << endl << "Time Components:" << endl << endl;

    // [한국어] 데이터 어레이 지연 분해: 전체 접근 시간 → H-tree 입력/디코더/비트라인/센스앰프/H-tree 출력
    cout << "  Data side (with Output driver) (ns): " <<
      fr->data_array2->access_time/1e-9 << endl;

    cout <<  "\tH-tree input delay (ns): " <<
      fr->data_array2->delay_route_to_bank * 1e9 +
      fr->data_array2->delay_input_htree * 1e9 << endl;

    if (!(g_ip->pure_cam || g_ip->fully_assoc))
    {
      cout <<  "\tDecoder + wordline delay (ns): " <<
        fr->data_array2->delay_row_predecode_driver_and_block * 1e9 +
        fr->data_array2->delay_row_decoder * 1e9 << endl;
    }
    else
    {
        cout <<  "\tCAM search delay (ns): " <<
          fr->data_array2->delay_matchlines * 1e9 << endl;
    }

    cout <<  "\tBitline delay (ns): " <<
      fr->data_array2->delay_bitlines/1e-9 << endl;

    cout <<  "\tSense Amplifier delay (ns): " <<
      fr->data_array2->delay_sense_amp * 1e9 << endl;


    cout <<  "\tH-tree output delay (ns): " <<
      fr->data_array2->delay_subarray_output_driver * 1e9 +
      fr->data_array2->delay_dout_htree * 1e9 << endl;

    // [한국어] 일반 캐시인 경우 태그 어레이 지연 분핏도 출력
    if ((!(g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)) && !g_ip->is_main_mem)
    {
      /* tag array stats */
      cout << endl << "  Tag side (with Output driver) (ns): " <<
        fr->tag_array2->access_time/1e-9 << endl;

      cout <<  "\tH-tree input delay (ns): " <<
        fr->tag_array2->delay_route_to_bank * 1e9 +
        fr->tag_array2->delay_input_htree * 1e9 << endl;

      cout <<  "\tDecoder + wordline delay (ns): " <<
        fr->tag_array2->delay_row_predecode_driver_and_block * 1e9 +
        fr->tag_array2->delay_row_decoder * 1e9 << endl;

      cout <<  "\tBitline delay (ns): " <<
        fr->tag_array2->delay_bitlines/1e-9 << endl;

      cout <<  "\tSense Amplifier delay (ns): " <<
        fr->tag_array2->delay_sense_amp * 1e9 << endl;

      cout <<  "\tComparator delay (ns): " <<
        fr->tag_array2->delay_comparator * 1e9 << endl;

      cout <<  "\tH-tree output delay (ns): " <<
        fr->tag_array2->delay_subarray_output_driver * 1e9 +
        fr->tag_array2->delay_dout_htree * 1e9 << endl;
    }



    /* Energy/Power stats */
    // [한국어] 전력(에너지) 분해: 데이터 어레이, CAM, 완전 연관 캐시별로 출력
    cout << endl << endl << "Power Components:" << endl << endl;

    if (!(g_ip->pure_cam || g_ip->fully_assoc))
    {
    	cout << "  Data array: Total dynamic read energy/access  (nJ): " <<
    	      fr->data_array2->power.readOp.dynamic * 1e9 << endl;
    	cout << "\tTotal leakage read/write power of a bank (mW): " <<
    	        fr->data_array2->power.readOp.leakage * 1e3 << endl;

    	cout << "\tTotal energy in H-tree (that includes both "
    	      "address and data transfer) (nJ): " <<
    	        (fr->data_array2->power_addr_input_htree.readOp.dynamic +
    	         fr->data_array2->power_data_output_htree.readOp.dynamic +
    	         fr->data_array2->power_routing_to_bank.readOp.dynamic) * 1e9 << endl;

    	cout << "\tTotal leakage power in H-tree (that includes both "
    	      "address and data network) ((mW)): " <<
    	        (fr->data_array2->power_addr_input_htree.readOp.leakage +
    	         fr->data_array2->power_data_output_htree.readOp.leakage +
    	         fr->data_array2->power_routing_to_bank.readOp.leakage) * 1e3 << endl;

    	cout << "\tTotal gate leakage power in H-tree (that includes both "
    	      "address and data network) ((mW)): " <<
    	        (fr->data_array2->power_addr_input_htree.readOp.gate_leakage +
    	         fr->data_array2->power_data_output_htree.readOp.gate_leakage +
    	         fr->data_array2->power_routing_to_bank.readOp.gate_leakage) * 1e3 << endl;

    	cout << "\tOutput Htree inside bank Energy (nJ): " <<
    	   fr->data_array2->power_data_output_htree.readOp.dynamic * 1e9 << endl;
    	cout <<  "\tDecoder (nJ): " <<
    	   fr->data_array2->power_row_predecoder_drivers.readOp.dynamic * 1e9 +
    	   fr->data_array2->power_row_predecoder_blocks.readOp.dynamic * 1e9 << endl;
    	cout <<  "\tWordline (nJ): " <<
    	   fr->data_array2->power_row_decoders.readOp.dynamic * 1e9 << endl;
    	cout <<  "\tBitline mux & associated drivers (nJ): " <<
    	   fr->data_array2->power_bit_mux_predecoder_drivers.readOp.dynamic * 1e9 +
    	   fr->data_array2->power_bit_mux_predecoder_blocks.readOp.dynamic * 1e9 +
    	   fr->data_array2->power_bit_mux_decoders.readOp.dynamic * 1e9 << endl;
    	cout <<  "\tSense amp mux & associated drivers (nJ): " <<
    	   fr->data_array2->power_senseamp_mux_lev_1_predecoder_drivers.readOp.dynamic * 1e9 +
    	   fr->data_array2->power_senseamp_mux_lev_1_predecoder_blocks.readOp.dynamic * 1e9 +
    	   fr->data_array2->power_senseamp_mux_lev_1_decoders.readOp.dynamic * 1e9  +
    	   fr->data_array2->power_senseamp_mux_lev_2_predecoder_drivers.readOp.dynamic * 1e9 +
    	   fr->data_array2->power_senseamp_mux_lev_2_predecoder_blocks.readOp.dynamic * 1e9 +
    	   fr->data_array2->power_senseamp_mux_lev_2_decoders.readOp.dynamic * 1e9 << endl;

    	cout <<  "\tBitlines precharge and equalization circuit (nJ): " <<
    	    	   fr->data_array2->power_prechg_eq_drivers.readOp.dynamic * 1e9 << endl;
    	cout <<  "\tBitlines (nJ): " <<
    	   fr->data_array2->power_bitlines.readOp.dynamic * 1e9 << endl;
    	cout <<  "\tSense amplifier energy (nJ): " <<
    	   fr->data_array2->power_sense_amps.readOp.dynamic * 1e9 << endl;
    	cout <<  "\tSub-array output driver (nJ): " <<
    	   fr->data_array2->power_output_drivers_at_subarray.readOp.dynamic * 1e9 << endl;
    }

        else if (g_ip->pure_cam)
        {

           	cout << "  CAM array:"<<endl;
            	cout << "  Total dynamic associative search energy/access  (nJ): " <<
                      fr->data_array2->power.searchOp.dynamic * 1e9 << endl;
    	        cout << "\tTotal energy in H-tree (that includes both "
    	            	      "match key and data transfer) (nJ): " <<
    	              (fr->data_array2->power_htree_in_search.searchOp.dynamic +
    	               fr->data_array2->power_htree_out_search.searchOp.dynamic +
    	               fr->data_array2->power_routing_to_bank.searchOp.dynamic) * 1e9 << endl;
    	        cout << "\tKeyword input and result output Htrees inside bank Energy (nJ): " <<
    	              (fr->data_array2->power_htree_in_search.searchOp.dynamic +
    	       	               fr->data_array2->power_htree_out_search.searchOp.dynamic) * 1e9 << endl;
    	        cout <<  "\tSearchlines (nJ): " <<
    	          	   fr->data_array2->power_searchline.searchOp.dynamic * 1e9 +
    	          	   fr->data_array2->power_searchline_precharge.searchOp.dynamic * 1e9 << endl;
    	        cout <<  "\tMatchlines  (nJ): " <<
    	               fr->data_array2->power_matchlines.searchOp.dynamic * 1e9 +
    	        	   fr->data_array2->power_matchline_precharge.searchOp.dynamic * 1e9 << endl;
    	        cout <<  "\tSub-array output driver (nJ): " <<
    	          	   fr->data_array2->power_output_drivers_at_subarray.searchOp.dynamic * 1e9 << endl;


            	cout <<endl<< "  Total dynamic read energy/access  (nJ): " <<
            	      fr->data_array2->power.readOp.dynamic * 1e9 << endl;
    	        cout << "\tTotal energy in H-tree (that includes both "
    	            	      "address and data transfer) (nJ): " <<
    	              (fr->data_array2->power_addr_input_htree.readOp.dynamic +
    	               fr->data_array2->power_data_output_htree.readOp.dynamic +
    	               fr->data_array2->power_routing_to_bank.readOp.dynamic) * 1e9 << endl;
    	        cout << "\tOutput Htree inside bank Energy (nJ): " <<
    	          	   fr->data_array2->power_data_output_htree.readOp.dynamic * 1e9 << endl;
    	        cout <<  "\tDecoder (nJ): " <<
    	          	   fr->data_array2->power_row_predecoder_drivers.readOp.dynamic * 1e9 +
    	          	   fr->data_array2->power_row_predecoder_blocks.readOp.dynamic * 1e9 << endl;
    	        cout <<  "\tWordline (nJ): " <<
    	          	   fr->data_array2->power_row_decoders.readOp.dynamic * 1e9 << endl;
    	        cout <<  "\tBitline mux & associated drivers (nJ): " <<
    	          	   fr->data_array2->power_bit_mux_predecoder_drivers.readOp.dynamic * 1e9 +
    	          	   fr->data_array2->power_bit_mux_predecoder_blocks.readOp.dynamic * 1e9 +
    	           	   fr->data_array2->power_bit_mux_decoders.readOp.dynamic * 1e9 << endl;
    	        cout <<  "\tSense amp mux & associated drivers (nJ): " <<
    	         	   fr->data_array2->power_senseamp_mux_lev_1_predecoder_drivers.readOp.dynamic * 1e9 +
    	          	   fr->data_array2->power_senseamp_mux_lev_1_predecoder_blocks.readOp.dynamic * 1e9 +
    	          	   fr->data_array2->power_senseamp_mux_lev_1_decoders.readOp.dynamic * 1e9  +
    	           	   fr->data_array2->power_senseamp_mux_lev_2_predecoder_drivers.readOp.dynamic * 1e9 +
    	           	   fr->data_array2->power_senseamp_mux_lev_2_predecoder_blocks.readOp.dynamic * 1e9 +
    	          	   fr->data_array2->power_senseamp_mux_lev_2_decoders.readOp.dynamic * 1e9 << endl;
    	        cout <<  "\tBitlines (nJ): " <<
    	          	   fr->data_array2->power_bitlines.readOp.dynamic * 1e9 +
    	          	   fr->data_array2->power_prechg_eq_drivers.readOp.dynamic * 1e9<< endl;
    	        cout <<  "\tSense amplifier energy (nJ): " <<
    	          	   fr->data_array2->power_sense_amps.readOp.dynamic * 1e9 << endl;
    	        cout <<  "\tSub-array output driver (nJ): " <<
    	          	   fr->data_array2->power_output_drivers_at_subarray.readOp.dynamic * 1e9 << endl;

            	cout << endl <<"  Total leakage power of a bank (mW): " <<
                      fr->data_array2->power.readOp.leakage * 1e3 << endl;
        }
        else
        {
        	cout << "  Fully associative array:"<<endl;
        	cout << "  Total dynamic associative search energy/access  (nJ): " <<
                  fr->data_array2->power.searchOp.dynamic * 1e9 << endl;
	        cout << "\tTotal energy in H-tree (that includes both "
	            	      "match key and data transfer) (nJ): " <<
	              (fr->data_array2->power_htree_in_search.searchOp.dynamic +
	               fr->data_array2->power_htree_out_search.searchOp.dynamic +
	               fr->data_array2->power_routing_to_bank.searchOp.dynamic) * 1e9 << endl;
	        cout << "\tKeyword input and result output Htrees inside bank Energy (nJ): " <<
	              (fr->data_array2->power_htree_in_search.searchOp.dynamic +
	       	               fr->data_array2->power_htree_out_search.searchOp.dynamic) * 1e9 << endl;
	        cout <<  "\tSearchlines (nJ): " <<
	          	   fr->data_array2->power_searchline.searchOp.dynamic * 1e9 +
	          	   fr->data_array2->power_searchline_precharge.searchOp.dynamic * 1e9 << endl;
	        cout <<  "\tMatchlines  (nJ): " <<
	               fr->data_array2->power_matchlines.searchOp.dynamic * 1e9 +
	        	   fr->data_array2->power_matchline_precharge.searchOp.dynamic * 1e9 << endl;
	        cout <<  "\tData portion wordline (nJ): " <<
	          	   fr->data_array2->power_matchline_to_wordline_drv.searchOp.dynamic * 1e9 << endl;
	        cout <<  "\tData Bitlines (nJ): " <<
	          	   fr->data_array2->power_bitlines.searchOp.dynamic * 1e9 +
	          	   fr->data_array2->power_prechg_eq_drivers.searchOp.dynamic * 1e9 << endl;
	        cout <<  "\tSense amplifier energy (nJ): " <<
	          	   fr->data_array2->power_sense_amps.searchOp.dynamic * 1e9 << endl;
	        cout <<  "\tSub-array output driver (nJ): " <<
	          	   fr->data_array2->power_output_drivers_at_subarray.searchOp.dynamic * 1e9 << endl;


        	cout <<endl<< "  Total dynamic read energy/access  (nJ): " <<
        	      fr->data_array2->power.readOp.dynamic * 1e9 << endl;
	        cout << "\tTotal energy in H-tree (that includes both "
	            	      "address and data transfer) (nJ): " <<
	              (fr->data_array2->power_addr_input_htree.readOp.dynamic +
	               fr->data_array2->power_data_output_htree.readOp.dynamic +
	               fr->data_array2->power_routing_to_bank.readOp.dynamic) * 1e9 << endl;
	        cout << "\tOutput Htree inside bank Energy (nJ): " <<
	          	   fr->data_array2->power_data_output_htree.readOp.dynamic * 1e9 << endl;
	        cout <<  "\tDecoder (nJ): " <<
	          	   fr->data_array2->power_row_predecoder_drivers.readOp.dynamic * 1e9 +
	          	   fr->data_array2->power_row_predecoder_blocks.readOp.dynamic * 1e9 << endl;
	        cout <<  "\tWordline (nJ): " <<
	          	   fr->data_array2->power_row_decoders.readOp.dynamic * 1e9 << endl;
	        cout <<  "\tBitline mux & associated drivers (nJ): " <<
	          	   fr->data_array2->power_bit_mux_predecoder_drivers.readOp.dynamic * 1e9 +
	          	   fr->data_array2->power_bit_mux_predecoder_blocks.readOp.dynamic * 1e9 +
	           	   fr->data_array2->power_bit_mux_decoders.readOp.dynamic * 1e9 << endl;
	        cout <<  "\tSense amp mux & associated drivers (nJ): " <<
	         	   fr->data_array2->power_senseamp_mux_lev_1_predecoder_drivers.readOp.dynamic * 1e9 +
	          	   fr->data_array2->power_senseamp_mux_lev_1_predecoder_blocks.readOp.dynamic * 1e9 +
	          	   fr->data_array2->power_senseamp_mux_lev_1_decoders.readOp.dynamic * 1e9  +
	           	   fr->data_array2->power_senseamp_mux_lev_2_predecoder_drivers.readOp.dynamic * 1e9 +
	           	   fr->data_array2->power_senseamp_mux_lev_2_predecoder_blocks.readOp.dynamic * 1e9 +
	          	   fr->data_array2->power_senseamp_mux_lev_2_decoders.readOp.dynamic * 1e9 << endl;
	        cout <<  "\tBitlines (nJ): " <<
	          	   fr->data_array2->power_bitlines.readOp.dynamic * 1e9 +
	          	   fr->data_array2->power_prechg_eq_drivers.readOp.dynamic * 1e9<< endl;
	        cout <<  "\tSense amplifier energy (nJ): " <<
	          	   fr->data_array2->power_sense_amps.readOp.dynamic * 1e9 << endl;
	        cout <<  "\tSub-array output driver (nJ): " <<
	          	   fr->data_array2->power_output_drivers_at_subarray.readOp.dynamic * 1e9 << endl;

        	cout << endl <<"  Total leakage power of a bank (mW): " <<
                  fr->data_array2->power.readOp.leakage * 1e3 << endl;
      }


    // [한국어] 일반 캐시의 태그 어레이 전력(읽기 동적 에너지, 누설, H-tree, 서브컴포넌트) 출력
    if ((!(g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)) && !g_ip->is_main_mem)
    {
      cout << endl << "  Tag array:  Total dynamic read energy/access (nJ): " <<
        fr->tag_array2->power.readOp.dynamic * 1e9 << endl;
      cout << "\tTotal leakage read/write power of a bank (mW): " <<
          fr->tag_array2->power.readOp.leakage * 1e3 << endl;
      cout << "\tTotal energy in H-tree (that includes both "
        "address and data transfer) (nJ): " <<
          (fr->tag_array2->power_addr_input_htree.readOp.dynamic +
           fr->tag_array2->power_data_output_htree.readOp.dynamic +
           fr->tag_array2->power_routing_to_bank.readOp.dynamic) * 1e9 << endl;

      cout << "\tTotal leakage power in H-tree (that includes both "
  	      "address and data network) ((mW)): " <<
  	        (fr->tag_array2->power_addr_input_htree.readOp.leakage +
  	         fr->tag_array2->power_data_output_htree.readOp.leakage +
  	         fr->tag_array2->power_routing_to_bank.readOp.leakage) * 1e3 << endl;

  	  cout << "\tTotal gate leakage power in H-tree (that includes both "
  	      "address and data network) ((mW)): " <<
  	        (fr->tag_array2->power_addr_input_htree.readOp.gate_leakage +
  	         fr->tag_array2->power_data_output_htree.readOp.gate_leakage +
  	         fr->tag_array2->power_routing_to_bank.readOp.gate_leakage) * 1e3 << endl;

      cout << "\tOutput Htree inside a bank Energy (nJ): " <<
        fr->tag_array2->power_data_output_htree.readOp.dynamic * 1e9 << endl;
      cout <<  "\tDecoder (nJ): " <<
        fr->tag_array2->power_row_predecoder_drivers.readOp.dynamic * 1e9 +
        fr->tag_array2->power_row_predecoder_blocks.readOp.dynamic * 1e9 << endl;
      cout <<  "\tWordline (nJ): " <<
        fr->tag_array2->power_row_decoders.readOp.dynamic * 1e9 << endl;
      cout <<  "\tBitline mux & associated drivers (nJ): " <<
        fr->tag_array2->power_bit_mux_predecoder_drivers.readOp.dynamic * 1e9 +
        fr->tag_array2->power_bit_mux_predecoder_blocks.readOp.dynamic * 1e9 +
        fr->tag_array2->power_bit_mux_decoders.readOp.dynamic * 1e9 << endl;
      cout <<  "\tSense amp mux & associated drivers (nJ): " <<
        fr->tag_array2->power_senseamp_mux_lev_1_predecoder_drivers.readOp.dynamic * 1e9 +
        fr->tag_array2->power_senseamp_mux_lev_1_predecoder_blocks.readOp.dynamic * 1e9 +
        fr->tag_array2->power_senseamp_mux_lev_1_decoders.readOp.dynamic * 1e9  +
        fr->tag_array2->power_senseamp_mux_lev_2_predecoder_drivers.readOp.dynamic * 1e9 +
        fr->tag_array2->power_senseamp_mux_lev_2_predecoder_blocks.readOp.dynamic * 1e9 +
        fr->tag_array2->power_senseamp_mux_lev_2_decoders.readOp.dynamic * 1e9 << endl;
      cout <<  "\tBitlines precharge and equalization circuit (nJ): " <<
        fr->tag_array2->power_prechg_eq_drivers.readOp.dynamic * 1e9 << endl;
      cout <<  "\tBitlines (nJ): " <<
        fr->tag_array2->power_bitlines.readOp.dynamic * 1e9 << endl;
      cout <<  "\tSense amplifier energy (nJ): " <<
        fr->tag_array2->power_sense_amps.readOp.dynamic * 1e9 << endl;
      cout <<  "\tSub-array output driver (nJ): " <<
        fr->tag_array2->power_output_drivers_at_subarray.readOp.dynamic * 1e9 << endl;
    }

    // [한국어] 면적 분해: 데이터/CAM/완전 연관 어레이 면적, 높이, 폭, 면적 효율 출력
    cout << endl << endl <<  "Area Components:" << endl << endl;
    /* Data array area stats */
    if (!(g_ip->pure_cam || g_ip->fully_assoc))
    	cout <<  "  Data array: Area (mm2): " << fr->data_array2->area * 1e-6 << endl;
    else if (g_ip->pure_cam)
    	cout <<  "  CAM array: Area (mm2): " << fr->data_array2->area * 1e-6 << endl;
    else
    	cout <<  "  Fully associative cache array: Area (mm2): " << fr->data_array2->area * 1e-6 << endl;
    cout <<  "\tHeight (mm): " <<
      fr->data_array2->all_banks_height*1e-3 << endl;
    cout <<  "\tWidth (mm): " <<
      fr->data_array2->all_banks_width*1e-3 << endl;
    if (g_ip->print_detail) {
      cout <<  "\tArea efficiency (Memory cell area/Total area) - " <<
        fr->data_array2->area_efficiency << " %" << endl;
      cout << "\t\tMAT Height (mm): " <<
        fr->data_array2->mat_height*1e-3 << endl;
      cout << "\t\tMAT Length (mm): " <<
        fr->data_array2->mat_length*1e-3 << endl;
      cout << "\t\tSubarray Height (mm): " <<
        fr->data_array2->subarray_height*1e-3 << endl;
      cout << "\t\tSubarray Length (mm): " <<
        fr->data_array2->subarray_length*1e-3 << endl;
    }

    // [한국어] 일반 캐시의 태그 어레이 면적 및 하위 구조(MAT/Subarray) 출력
    /* Tag array area stats */
    if ((!(g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)) && !g_ip->is_main_mem)
    {
      cout << endl << "  Tag array: Area (mm2): " << fr->tag_array2->area * 1e-6 << endl;
      cout <<  "\tHeight (mm): " <<
        fr->tag_array2->all_banks_height*1e-3 << endl;
      cout <<  "\tWidth (mm): " <<
        fr->tag_array2->all_banks_width*1e-3 << endl;
      if (g_ip->print_detail)
      {
        cout <<  "\tArea efficiency (Memory cell area/Total area) - " <<
          fr->tag_array2->area_efficiency << " %" << endl;
      cout << "\t\tMAT Height (mm): " <<
        fr->tag_array2->mat_height*1e-3 << endl;
      cout << "\t\tMAT Length (mm): " <<
        fr->tag_array2->mat_length*1e-3 << endl;
      cout << "\t\tSubarray Height (mm): " <<
        fr->tag_array2->subarray_height*1e-3 << endl;
      cout << "\t\tSubarray Length (mm): " <<
        fr->tag_array2->subarray_length*1e-3 << endl;
      }
    }
    // [한국어] 선택된 배선 모델의 전기적 특성 출력
    Wire wpr;
    wpr.print_wire();

    //cout << "FO4 = " << g_tp.FO4 << endl;
  }
}

/*
 * [한국어]
 * cacti_interface(InputParameter*) — AccelWattch가 사용하는 포인터 기반 인터페이스.
 *
 * @local_interface: 이미 외부(AccelWattch XML 파서 등)에서 채워진 InputParameter
 *   포인터. 이 함수는 g_ip를 local_interface로 설정하고 별도 할당/해제를 하지 않는다.
 * @return: uca_org_t 분석 결과.
 *
 * 이 인터페이스는 CACTI의 메모리 할당을 호출자(AccelWattch)가 관리할 수 있도록
 * 설계되었다. g_ip->error_checking()과 init_tech_params()만 수행한 뒤
 * solve(&fin_res)를 호출한다. 출력 함수(output_UCA/output_data_csv)는 호출하지
 * 않으므로 결과는 AccelWattch 나이부에서 소비된다.
 *
 * 호출 체인:
 *   AccelWattch (processor.h/sharedcache.h) → [cacti_interface(InputParameter*)]
 *       → solve(&fin_res)
 */
//McPAT's plain interface, please keep !!!
uca_org_t cacti_interface(InputParameter  * const local_interface)
{
//  g_ip = new InputParameter(); // [한국어] 포인터 인터페이스에서는 새 할당 안 함
  //g_ip->add_ecc_b_ = true;

  uca_org_t fin_res;
  fin_res.valid = false; // [한국어] solve() 전까지 유효성 false

  g_ip = local_interface; // [한국어] 호출자가 제공한 InputParameter를 전역 g_ip로 사용


//  g_ip->data_arr_ram_cell_tech_type    = data_arr_ram_cell_tech_flavor_in;
//  g_ip->data_arr_peri_global_tech_type = data_arr_peri_global_tech_flavor_in;
//  g_ip->tag_arr_ram_cell_tech_type     = tag_arr_ram_cell_tech_flavor_in;
//  g_ip->tag_arr_peri_global_tech_type  = tag_arr_peri_global_tech_flavor_in;
//
//  g_ip->ic_proj_type     = interconnect_projection_type_in;
//  g_ip->wire_is_mat_type = wire_inside_mat_type_in;
//  g_ip->wire_os_mat_type = wire_outside_mat_type_in;
//  g_ip->burst_len        = BURST_LENGTH_in;
//  g_ip->int_prefetch_w   = INTERNAL_PREFETCH_WIDTH_in;
//  g_ip->page_sz_bits     = PAGE_SIZE_BITS_in;
//
//  g_ip->cache_sz            = cache_size;
//  g_ip->line_sz             = line_size;
//  g_ip->assoc               = associativity;
//  g_ip->nbanks              = banks;
//  g_ip->out_w               = output_width;
//  g_ip->specific_tag        = specific_tag;
//  if (tag_width == 0) {
//    g_ip->tag_w = 42;
//  }
//  else {
//    g_ip->tag_w               = tag_width;
//  }
//
//  g_ip->access_mode         = access_mode;
//  g_ip->delay_wt = obj_func_delay;
//  g_ip->dynamic_power_wt = obj_func_dynamic_power;
//  g_ip->leakage_power_wt = obj_func_leakage_power;
//  g_ip->area_wt = obj_func_area;
//  g_ip->cycle_time_wt    = obj_func_cycle_time;
//  g_ip->delay_dev = dev_func_delay;
//  g_ip->dynamic_power_dev = dev_func_dynamic_power;
//  g_ip->leakage_power_dev = dev_func_leakage_power;
//  g_ip->area_dev = dev_func_area;
//  g_ip->cycle_time_dev    = dev_func_cycle_time;
//  g_ip->temp = temp;
//
//  g_ip->F_sz_nm         = tech_node;
//  g_ip->F_sz_um         = tech_node / 1000;
//  g_ip->is_main_mem     = (main_mem != 0) ? true : false;
//  g_ip->is_cache        = (cache ==1) ? true : false;
//  g_ip->pure_ram        = (cache ==0) ? true : false;
//  g_ip->pure_cam        = (cache ==2) ? true : false;
//  g_ip->rpters_in_htree = (REPEATERS_IN_HTREE_SEGMENTS_in != 0) ? true : false;
//  g_ip->ver_htree_wires_over_array = VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in;
//  g_ip->broadcast_addr_din_over_ver_htrees = BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in;
//
//  g_ip->num_rw_ports    = rw_ports;
//  g_ip->num_rd_ports    = excl_read_ports;
//  g_ip->num_wr_ports    = excl_write_ports;
//  g_ip->num_se_rd_ports = single_ended_read_ports;
//  g_ip->num_search_ports = search_ports;
//
//  g_ip->print_detail = 1;
//    g_ip->nuca = 0;
//    g_ip->is_cache=true;
//
//  if (force_wiretype == 0)
//  {
//	  g_ip->wt = Global;
//      g_ip->force_wiretype = false;
//  }
//  else
//  {   g_ip->force_wiretype = true;
//	  if (wiretype==10) {
//		  g_ip->wt = Global_10;
//	        }
//	  if (wiretype==20) {
//		  g_ip->wt = Global_20;
//	        }
//	  if (wiretype==30) {
//		  g_ip->wt = Global_30;
//	        }
//	  if (wiretype==5) {
//	      g_ip->wt = Global_5;
//	        }
//	  if (wiretype==0) {
//		  g_ip->wt = Low_swing;
//	  }
//  }
//  //g_ip->wt = Global_5;
//  if (force_config == 0)
//    {
//  	  g_ip->force_cache_config = false;
//    }
//    else
//    {
//    	g_ip->force_cache_config = true;
//    	g_ip->ndbl=ndbl;
//    	g_ip->ndwl=ndwl;
//    	g_ip->nspd=nspd;
//    	g_ip->ndcm=ndcm;
//    	g_ip->ndsam1=ndsam1;
//    	g_ip->ndsam2=ndsam2;
//
//
//    }
//
//  if (ecc==0){
//	  g_ip->add_ecc_b_=false;
//  }
//  else
//  {
//	  g_ip->add_ecc_b_=true;
//  }


  g_ip->error_checking(); // [한국어] 호출자가 채운 입력값 타당성 검증


  init_tech_params(g_ip->F_sz_um, false); // [한국어] 공정 기술 파라미터 초기화
  Wire winit; // Do not delete this line. It initializes wires. // [한국어] 전역 Wire 정적 초기화(삭제 금지)

  solve(&fin_res); // [한국어] UCA/NUCA 최적 설계 탐색

//  g_ip->display_ip(); // [한국어] AccelWattch 인터페이스에서는 stdout 출력 생략
//  output_UCA(&fin_res); // [한국어] AccelWattch 인터페이스에서는 텍스트 출력 생략
//  output_data_csv(fin_res); // [한국어] AccelWattch 인터페이스에서는 CSV 출력 생략

 // delete (g_ip); // [한국어] 호출자가 local_interface 메모리를 관리하므로 해제 안 함

  return fin_res; // [한국어] 분석 결과 반환
}

/*
 * [한국어]
 * init_interface — cacti_interface(InputParameter*)와 유사하나 solve()를 호출하지 않는다.
 *
 * @local_interface: 이미 채워진 InputParameter 포인터.
 * @return: uca_org_t 결과 구조체(fin_res.valid=false, solve() 미호출).
 *
 * 이 함수는 기술 파라미터 초기화만 수행하고 실제 캐시 설계 탐색은 수행하지 않는다.
 * AccelWattch의 일부 재구성/증분 시나리오에서 사용될 수 있으나, 현재 코드에서는
 * solve()가 주석 처리되어 있어 결과가 비어 있다.
 *
 * 호출 체인:
 *   AccelWattch 또는 외부 재구성 코드 → [init_interface()] → (기술 파라미터 초기화)
 */
//McPAT's plain interface, please keep !!!
uca_org_t init_interface(InputParameter* const local_interface)
{
 // g_ip = new InputParameter();
  //g_ip->add_ecc_b_ = true;

  uca_org_t fin_res;
  fin_res.valid = false; // [한국어] solve() 미호출이므로 결과는 유효하지 않음

   g_ip = local_interface; // [한국어] 호출자가 제공한 파라미터를 전역 g_ip로 사용


//  g_ip->data_arr_ram_cell_tech_type    = data_arr_ram_cell_tech_flavor_in;
//  g_ip->data_arr_peri_global_tech_type = data_arr_peri_global_tech_flavor_in;
//  g_ip->tag_arr_ram_cell_tech_type     = tag_arr_ram_cell_tech_flavor_in;
//  g_ip->tag_arr_peri_global_tech_type  = tag_arr_peri_global_tech_flavor_in;
//
//  g_ip->ic_proj_type     = interconnect_projection_type_in;
//  g_ip->wire_is_mat_type = wire_inside_mat_type_in;
//  g_ip->wire_os_mat_type = wire_outside_mat_type_in;
//  g_ip->burst_len        = BURST_LENGTH_in;
//  g_ip->int_prefetch_w   = INTERNAL_PREFETCH_WIDTH_in;
//  g_ip->page_sz_bits     = PAGE_SIZE_BITS_in;
//
//  g_ip->cache_sz            = cache_size;
//  g_ip->line_sz             = line_size;
//  g_ip->assoc               = associativity;
//  g_ip->nbanks              = banks;
//  g_ip->out_w               = output_width;
//  g_ip->specific_tag        = specific_tag;
//  if (tag_width == 0) {
//    g_ip->tag_w = 42;
//  }
//  else {
//    g_ip->tag_w               = tag_width;
//  }
//
//  g_ip->access_mode         = access_mode;
//  g_ip->delay_wt = obj_func_delay;
//  g_ip->dynamic_power_wt = obj_func_dynamic_power;
//  g_ip->leakage_power_wt = obj_func_leakage_power;
//  g_ip->area_wt = obj_func_area;
//  g_ip->cycle_time_wt    = obj_func_cycle_time;
//  g_ip->delay_dev = dev_func_delay;
//  g_ip->dynamic_power_dev = dev_func_dynamic_power;
//  g_ip->leakage_power_dev = dev_func_leakage_power;
//  g_ip->area_dev = dev_func_area;
//  g_ip->cycle_time_dev    = dev_func_cycle_time;
//  g_ip->temp = temp;
//
//  g_ip->F_sz_nm         = tech_node;
//  g_ip->F_sz_um         = tech_node / 1000;
//  g_ip->is_main_mem     = (main_mem != 0) ? true : false;
//  g_ip->is_cache        = (cache ==1) ? true : false;
//  g_ip->pure_ram        = (cache ==0) ? true : false;
//  g_ip->pure_cam        = (cache ==2) ? true : false;
//  g_ip->rpters_in_htree = (REPEATERS_IN_HTREE_SEGMENTS_in != 0) ? true : false;
//  g_ip->ver_htree_wires_over_array = VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in;
//  g_ip->broadcast_addr_din_over_ver_htrees = BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in;
//
//  g_ip->num_rw_ports    = rw_ports;
//  g_ip->num_rd_ports    = excl_read_ports;
//  g_ip->num_wr_ports    = excl_write_ports;
//  g_ip->num_se_rd_ports = single_ended_read_ports;
//  g_ip->num_search_ports = search_ports;
//
//  g_ip->print_detail = 1;
//  g_ip->nuca = 0;
//
//  if (force_wiretype == 0)
//  {
//	  g_ip->wt = Global;
//      g_ip->force_wiretype = false;
//  }
//  else
//  {   g_ip->force_wiretype = true;
//	  if (wiretype==10) {
//		  g_ip->wt = Global_10;
//	        }
//	  if (wiretype==20) {
//		  g_ip->wt = Global_20;
//	        }
//	  if (wiretype==30) {
//		  g_ip->wt = Global_30;
//	        }
//	  if (wiretype==5) {
//	      g_ip->wt = Global_5;
//	        }
//	  if (wiretype==0) {
//		  g_ip->wt = Low_swing;
//	  }
//  }
//  //g_ip->wt = Global_5;
//  if (force_config == 0)
//    {
//  	  g_ip->force_cache_config = false;
//    }
//    else
//    {
//    	g_ip->force_cache_config = true;
//    	g_ip->ndbl=ndbl;
//    	g_ip->ndwl=ndwl;
//    	g_ip->nspd=nspd;
//    	g_ip->ndcm=ndcm;
//    	g_ip->ndsam1=ndsam1;
//    	g_ip->ndsam2=ndsam2;
//
//
//    }
//
//  if (ecc==0){
//	  g_ip->add_ecc_b_=false;
//  }
//  else
//  {
//	  g_ip->add_ecc_b_=true;
//  }


  // [한국어] 입력값 검증 및 기술 파라미터 초기화 (solve는 주석 처리됨)
  g_ip->error_checking();

  init_tech_params(g_ip->F_sz_um, false);
  Wire winit; // Do not delete this line. It initializes wires. // [한국어] 전역 Wire 정적 초기화(삭제 금지)
  //solve(&fin_res);
  //g_ip->display_ip();

  //solve(&fin_res);
  //output_UCA(&fin_res);
  //output_data_csv(fin_res);
 // delete (g_ip);

  return fin_res; // [한국어] 비어 있는 결과 반환
}

/*
 * [한국어]
 * reconfigure — 이미 채워진 InputParameter를 기반으로 캐시 설계를 재계산한다.
 *
 * @local_interface: 새 입력 파라미터를 담은 InputParameter 포인터.
 * @fin_res: 기존 uca_org_t 결과 구조체 포인터. update()가 이 구조체를 갱신.
 * @return: 없음 (void).
 *
 * 동작 과정:
 *   1) g_ip = local_interface (전역 포인터 갱신).
 *   2) error_checking()으로 입력값 검증.
 *   3) init_tech_params()로 기술 파라미터 재초기화.
 *   4) Wire winit로 배선 모델 초기화.
 *   5) update(fin_res)로 기존 결과 구조체를 새 파라미터에 맞게 갱신.
 *
 * 호출 체인:
 *   외부 재구성 코드 → [reconfigure()] → update(fin_res)
 */
void reconfigure(InputParameter *local_interface, uca_org_t *fin_res)
{
  // Copy the InputParameter to global interface (g_ip) and do error checking.
  g_ip = local_interface; // [한국어] 전역 g_ip를 호출자가 제공한 파라미터로 교체
  g_ip->error_checking(); // [한국어] 새 입력값 타당성 검증

  // Initialize technology parameters
  init_tech_params(g_ip->F_sz_um,false); // [한국어] 공정 기술 파라미터 재초기화

  Wire winit; // Do not delete this line. It initializes wires. // [한국어] 전역 Wire 정적 초기화(삭제 금지)

  // This corresponds to solve() in the initialization process.
  update(fin_res); // [한국어] 기존 결과 구조체를 새 파라미터로 갱신
}
