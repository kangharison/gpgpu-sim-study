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
 * [한국어 설명] CACTI 공개 인터페이스 및 핵심 자료구조 선언 (cacti_interface.h)
 *
 * === 파일의 역할 ===
 * CACTI(Cache Access and Cycle Time Information) 시뮬레이터의 최상위 공개 인터페이스를 정의한다.
 * GPU 캐시의 에너지(동적/누설/게이트 누설), 타이밍(접근 시간/사이클 시간), 면적을 추정하기 위한
 * 핵심 자료구조(powerComponents, powerDef, InputParameter, mem_array, uca_org_t, results_mem_array)와
 * cacti_interface() 진입 함수 오버로드들을 선언한다.
 * AccelWattch(GPU 전력 모델)가 GPU L1/L2 캐시 에너지를 추정할 때 이 파일의 인터페이스를 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch(GPU 전력 모델) → cacti_interface.h(CACTI 진입점) → CACTI 내부 모델
 *   (Ucache, mem_array, decoder, crossbar, wire, mat, dram 등).
 * 호출 체인: AccelWattch 캐시 에너지 계산 → cacti_interface(InputParameter*) → [CACTI 내부]
 *   → uca_org_t(결과 반환) → AccelWattch 전력 누적.
 * 실행 컨텍스트: 호스트 유저스페이스 — GPU 시뮬레이션 시작 전 캐시 설계점 탐색 단계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: const.h (공통 상수), 표준 STL (map, string, vector, list, iostream).
 * 이 파일에 의존: AccelWattch 전력 모델, Ucache.cc, cacti_interface.cc, basic_circuit.*,
 *   decoder.*, crossbar.*, wire.*, mat.* 등 CACTI 전체.
 * 핵심 데이터 흐름: InputParameter → cacti_interface() → mem_array 탐색 → uca_org_t 결과.
 *
 * === 주요 함수/구조체 요약 ===
 * powerComponents   : 동적/누설/게이트누설/단락/긴채널 누설 에너지 성분 컨테이너
 * powerDef          : 읽기(readOp)/쓰기(writeOp)/검색(searchOp) 연산별 powerComponents 묶음
 * InputParameter    : 캐시 설계 입력 파라미터 (크기, 결합도, 포트 수, 공정 노드 등)
 * results_mem_array : 메모리 어레이 상세 타이밍/전력/면적 분해 결과 flat 구조체
 * uca_org_t         : UCA(Uniform Cache Access) 최종 결과 (태그/데이터 어레이 + 총합)
 * mem_array         : 단일 메모리 어레이 후보 설계점의 모든 타이밍/전력/면적 정보
 * cacti_interface() : CACTI 진입 함수 (다양한 파라미터 오버로드)
 */

#ifndef __CACTI_INTERFACE_H__
#define __CACTI_INTERFACE_H__

#include <map>      // [한국어] 설정 파일 파싱 시 키-값 맵 저장용
#include <string>   // [한국어] 파일명, 설정 문자열 처리
#include <vector>   // [한국어] 설계 후보 목록 관리
#include <list>     // [한국어] 메모리 어레이 후보 리스트
#include <iostream> // [한국어] 결과 출력
#include "const.h"  // [한국어] CACTI 공통 상수 (MAX_NUMBER_GATES_STAGE 등)

using namespace std;


class min_values_t; // [한국어] 전방 선언: 목적 함수별 최솟값 추적 (Ucache.h에서 정의)
class mem_array;    // [한국어] 전방 선언: 단일 메모리 어레이 설계 후보
class uca_org_t;    // [한국어] 전방 선언: UCA(Uniform Cache Access) 최종 결과


/*
 * [한국어] powerComponents — 전력 에너지 성분을 분리 보관하는 컨테이너 클래스
 * CACTI의 모든 서브회로(decoder, crossbar, wire, mat 등)가 이 클래스로 에너지를 누적.
 * 성분별로 분리 관리하여 각 기여분을 독립적으로 분석할 수 있다.
 */
class powerComponents
{
  public:
    double dynamic;
    /* 동적 에너지 (J/access)
     * 설정자: 각 서브회로의 compute_delays()에서 C*Vdd² 항으로 누적.
     * 읽는 자: uca_org_t::find_energy()가 최종 접근당 동적 에너지를 보고.
     * 값 범위: 0 이상 (캐시 크기/공정/주파수에 따라 pJ~nJ 수준).
     * 동기화: 단일 스레드 초기화 후 읽기 전용으로 사용. */

    double leakage;
    /* 서브임계 누설 전력 (W) — 항상 소비됨
     * 설정자: 각 서브회로의 compute_area()에서 cmos_Isub_leakage()*Vdd로 계산.
     * 읽는 자: uca_org_t::find_energy(), AccelWattch 전력 누적.
     * 값 범위: 0 이상 (비례적으로 캐시 크기에 비례).
     * 동기화: 단일 스레드 사용. */

    double gate_leakage;
    /* 게이트 터널링 누설 전력 (W) — 극소 공정 노드에서 중요해짐
     * 설정자: compute_area()에서 cmos_Ig_leakage()*Vdd로 계산.
     * 읽는 자: uca_org_t::find_energy(), AccelWattch.
     * 값 범위: 0 이상, 서브임계 누설보다 보통 작음.
     * 동기화: 단일 스레드 사용. */

    double short_circuit;
    /* 단락전류 에너지 (J/access) — CMOS 전이 중 풀업/풀다운 동시 ON 구간
     * 설정자: shortcircuit_simple()로 계산 후 dynamic에 합산되기도 함.
     * 읽는 자: 전력 분석 보고 시.
     * 값 범위: 0 이상, dynamic보다 보통 훨씬 작음.
     * 동기화: 단일 스레드 사용. */

    double longer_channel_leakage;
    /* 긴 채널 길이 효과 보정 누설 전력 (W)
     * 설정자: 긴 채널 트랜지스터(예: I/O 트랜지스터)가 있을 때 별도 계산.
     * 읽는 자: 특수 누설 분석 시.
     * 값 범위: 0 이상.
     * 동기화: 단일 스레드 사용. */

    powerComponents() : dynamic(0), leakage(0), gate_leakage(0), short_circuit(0), longer_channel_leakage(0)  { }
    // [한국어] 기본 생성자: 모든 에너지 성분을 0으로 초기화
    powerComponents(const powerComponents & obj) { *this = obj; }
    // [한국어] 복사 생성자: 대입 연산자 위임
    powerComponents & operator=(const powerComponents & rhs)
    {
      dynamic = rhs.dynamic;                              // [한국어] 동적 에너지 복사
      leakage = rhs.leakage;                              // [한국어] 누설 전력 복사
      gate_leakage  = rhs.gate_leakage;                  // [한국어] 게이트 터널링 복사
      short_circuit = rhs.short_circuit;                  // [한국어] 단락전류 에너지 복사
      longer_channel_leakage = rhs.longer_channel_leakage;// [한국어] 긴채널 누설 복사
      return *this;
    }
    void reset() { dynamic = 0; leakage = 0; gate_leakage = 0; short_circuit = 0;longer_channel_leakage = 0;}
    // [한국어] 모든 에너지 성분을 0으로 초기화 — 누적 계산 재시작 시 사용

    friend powerComponents operator+(const powerComponents & x, const powerComponents & y);
    // [한국어] 성분별 합산 연산자 — 두 서브회로의 에너지를 합칠 때 사용
    friend powerComponents operator*(const powerComponents & x, double const * const y);
    // [한국어] 스칼라 곱 연산자 — 파워 포인트 프로덕트 마스크(pppm) 적용 시 사용
};



/*
 * [한국어] powerDef — 읽기/쓰기/검색 연산별 전력 성분을 묶는 클래스
 * 캐시 접근 유형에 따라 다른 에너지 소비를 분리하여 추적한다.
 * readOp: 일반 읽기 접근, writeOp: 쓰기 접근, searchOp: CAM 검색 접근.
 */
class powerDef
{
  public:
    powerComponents readOp;
    /* 읽기 연산 전력 성분
     * 설정자: compute_delays/compute_area에서 읽기 경로 에너지 누적.
     * 읽는 자: uca_org_t::find_energy(), AccelWattch 캐시 읽기 에너지 추정.
     * 동기화: 단일 스레드 초기화. */

    powerComponents writeOp;
    /* 쓰기 연산 전력 성분
     * 설정자: 일부 서브회로에서 쓰기 경로 별도 계산.
     * 읽는 자: uca_org_t::find_energy(), AccelWattch 캐시 쓰기 에너지 추정.
     * 동기화: 단일 스레드 초기화. */

    powerComponents searchOp;//Sheng: for CAM and FA
    /* CAM(Content Addressable Memory)/완전 결합(FA) 검색 연산 전력 성분
     * 설정자: CAM 매치라인/서치라인 에너지 계산 시.
     * 읽는 자: CAM 기반 캐시 분석 시.
     * 동기화: 단일 스레드 초기화. */

    powerDef() : readOp(), writeOp(), searchOp() { }
    // [한국어] 기본 생성자: 모든 연산 유형의 에너지 성분을 0으로 초기화
    void reset() { readOp.reset(); writeOp.reset(); searchOp.reset();}
    // [한국어] 모든 연산 유형의 에너지를 0으로 초기화

    friend powerDef operator+(const powerDef & x, const powerDef & y);
    // [한국어] powerDef 합산: 두 서브회로의 연산별 에너지를 합침
    friend powerDef operator*(const powerDef & x, double const * const y);
    // [한국어] powerDef 스칼라 곱: pppm 마스크 적용
};

/* [한국어] Wire_type — 와이어 종류별 RC 파라미터를 결정하는 enum.
 * CACTI는 와이어 유형에 따라 반복기(repeater) 삽입 여부와 지연 페널티를 다르게 적용한다. */
enum Wire_type
{
    Global /* gloabl wires with repeaters */,
    // [한국어] 전역 와이어 — 반복기 삽입, 기준 지연 (뱅크 간 H-tree 등)
    Global_5 /* 5% delay penalty */,
    // [한국어] 전역 와이어 + 5% 지연 페널티
    Global_10 /* 10% delay penalty */,
    // [한국어] 전역 와이어 + 10% 지연 페널티
    Global_20 /* 20% delay penalty */,
    // [한국어] 전역 와이어 + 20% 지연 페널티
    Global_30 /* 30% delay penalty */,
    // [한국어] 전역 와이어 + 30% 지연 페널티
    Low_swing /* differential low power wires with high area overhead */,
    // [한국어] 저진폭 차동 와이어 — 저전력이나 면적 오버헤드 큼
    Semi_global /* mid-level wires with repeaters*/,
    // [한국어] 중간 레벨 와이어 — 반복기 포함, 전역과 로컬의 중간
    Transmission /* tranmission lines with high area overhead */,
    // [한국어] 전송 선로 — 높은 면적 오버헤드
    Optical /* optical wires */,
    // [한국어] 광 인터커넥트 — 대역폭 높지만 변환 오버헤드 존재
    Invalid_wtype
    // [한국어] 유효하지 않은 와이어 유형 (초기화 또는 오류 표시용)
};



/*
 * [한국어] InputParameter — CACTI에 전달하는 캐시 설계 입력 파라미터 클래스
 * AccelWattch가 GPU L1/L2 캐시 에너지를 추정할 때 이 클래스의 인스턴스를 채워
 * cacti_interface(InputParameter*) 함수에 전달한다.
 * gpgpusim.config의 캐시 관련 설정(크기, 결합도, 포트 수, 공정 노드 등)이
 * 이 클래스의 필드로 변환되어 CACTI에 입력된다.
 */
class InputParameter
{
  public:
	InputParameter();                   // [한국어] 기본 생성자 — 기본값으로 초기화
    void parse_cfg(const string & infile); // [한국어] CACTI 설정 파일(cfg) 파싱하여 필드 채움

    bool error_checking();  // return false if the input parameters are problematic
    // [한국어] 입력 파라미터 유효성 검사 — 비정상 조합(예: assoc > nsets) 탐지
    void display_ip(); // [한국어] 현재 입력 파라미터를 표준 출력으로 덤프 (디버깅용)

    unsigned int cache_sz;  // in bytes
    /* 캐시 총 크기 (바이트)
     * 설정자: AccelWattch가 GPU 캐시 크기 설정으로부터 계산하여 설정.
     * 읽는 자: CACTI 내부 어레이 파티셔닝 및 설계점 탐색.
     * 값 범위: 1B ~ 수십 MB (2의 거듭제곱이어야 함). */

    unsigned int line_sz;
    /* 캐시 라인 크기 (바이트)
     * 설정자: GPU 캐시 라인 크기 설정 (보통 128B).
     * 읽는 자: 비트라인 길이, 데이터 어레이 폭 계산에 사용.
     * 값 범위: 8B ~ 256B (2의 거듭제곱). */

    unsigned int assoc;
    /* 결합도(associativity) — 집합 연관 캐시의 웨이(way) 수
     * 설정자: AccelWattch GPU 캐시 설정.
     * 읽는 자: 태그 어레이 설계 (비교기 수 = assoc), 디코더 구성.
     * 값 범위: 1(직접 사상) ~ 16+ (고결합도). */

    unsigned int nbanks;
    /* 캐시 뱅크 수 — 병렬 접근 가능한 서브뱅크
     * 설정자: AccelWattch 또는 CACTI 탐색기.
     * 읽는 자: 면적/전력 계산 시 뱅크별 분할.
     * 값 범위: 1 ~ 수십 개. */

    unsigned int out_w;// == nr_bits_out
    /* 출력 폭(비트) — 캐시 단일 접근에서 반환되는 데이터 비트 수
     * 설정자: GPU 캐시 인터페이스 폭으로부터.
     * 읽는 자: 비트 먹스 설계, 출력 드라이버 크기 결정.
     * 값 범위: 8 ~ 512+ 비트. */

    bool     specific_tag;
    /* 태그 폭을 명시적으로 지정할지 여부
     * 설정자: 태그 폭 수동 지정 시 true.
     * 읽는 자: tag_w 필드를 그대로 쓸지 내부 계산값을 쓸지 결정.
     * 값 범위: true(tag_w 사용) / false(내부 계산). */

    unsigned int tag_w;
    /* 태그 폭(비트) — specific_tag=true일 때 사용
     * 설정자: specific_tag=true인 경우 사용자 지정.
     * 읽는 자: 태그 어레이 폭 계산.
     * 값 범위: 보통 물리 주소 비트 수에서 index/offset 비트를 뺀 값. */

    unsigned int access_mode;
    /* 캐시 접근 모드 — 0: 일반(normal), 1: 순차(sequential), 2: 고속(fast)
     * 설정자: AccelWattch 또는 사용자 설정.
     * 읽는 자: uca_org_t::find_delay()에서 접근 시간 계산 방식 결정.
     * 값 범위: 0(태그+데이터 병렬) / 1(태그 먼저 후 데이터) / 2(전체 집합 전송). */

    unsigned int obj_func_dyn_energy;
    /* 목적 함수 가중치 — 동적 에너지 최적화 비중
     * 설정자: CACTI 최적화 목표 설정.
     * 읽는 자: 설계 공간 탐색 시 우선순위 결정.
     * 값 범위: 0(무시) ~ 100(최고 우선순위). */

    unsigned int obj_func_dyn_power;  // [한국어] 목적 함수 가중치 — 동적 전력 최적화 비중
    unsigned int obj_func_leak_power; // [한국어] 목적 함수 가중치 — 누설 전력 최적화 비중
    unsigned int obj_func_cycle_t;    // [한국어] 목적 함수 가중치 — 사이클 시간 최적화 비중

    double   F_sz_nm;          // feature size in nm
    /* 공정 피처 크기 (나노미터) — 예: 45nm, 22nm
     * 설정자: AccelWattch GPU 공정 노드 설정.
     * 읽는 자: TechnologyParameter 초기화 — 모든 물리 파라미터의 기준.
     * 값 범위: 65nm(구형) ~ 7nm(최신). */

    double   F_sz_um;          // feature size in um
    /* 공정 피처 크기 (마이크로미터) — F_sz_nm / 1000
     * 설정자: F_sz_nm으로부터 변환하여 설정.
     * 읽는 자: 내부 um 단위 계산.
     * 값 범위: F_sz_nm/1000. */

    unsigned int num_rw_ports;   // [한국어] 읽기/쓰기 겸용(read-write) 포트 수
    unsigned int num_rd_ports;   // [한국어] 읽기 전용(read-only) 포트 수
    unsigned int num_wr_ports;   // [한국어] 쓰기 전용(write-only) 포트 수
    unsigned int num_se_rd_ports;  // number of single ended read ports
    // [한국어] 단일 종단(single-ended) 읽기 포트 수 — 차동 읽기보다 면적 절감
    unsigned int num_search_ports;  // Sheng: number of search ports for CAM
    // [한국어] CAM 검색 포트 수 — CAM 캐시에서 병렬 검색 지원

    bool     is_main_mem; // [한국어] 메인 메모리(DRAM) 모델링 여부 — true이면 DRAM 특화 계산
    bool     is_cache;    // [한국어] 캐시(SRAM) 모델링 여부
    bool     pure_ram;    // [한국어] 순수 RAM (태그 없는 스크래치패드) 여부
    bool     pure_cam;    // [한국어] 순수 CAM (Content Addressable Memory) 여부

    bool     rpters_in_htree;  // if there are repeaters in htree segment
    /* H-tree 세그먼트 내 반복기(repeater) 삽입 여부
     * 설정자: 와이어 타입/길이 기반으로 결정.
     * 읽는 자: H-tree 지연/전력 계산 시 반복기 효과 포함 여부.
     * 값 범위: true(반복기 삽입, 기본) / false. */

    unsigned int ver_htree_wires_over_array; // [한국어] 수직 H-tree 배선이 어레이 위를 지나가는지 여부
    unsigned int broadcast_addr_din_over_ver_htrees; // [한국어] 수직 H-tree로 주소/데이터 브로드캐스트 여부
    unsigned int temp; // [한국어] 동작 온도 (켈빈) — 누설전류는 온도에 강하게 의존

    unsigned int ram_cell_tech_type;         // [한국어] RAM 셀 기술 유형 (0=HP, 1=LSTP, 2=LOP 등)
    unsigned int peri_global_tech_type;      // [한국어] 주변 회로 전역 기술 유형
    unsigned int data_arr_ram_cell_tech_type;     // [한국어] 데이터 어레이 RAM 셀 기술 유형
    unsigned int data_arr_peri_global_tech_type;  // [한국어] 데이터 어레이 주변 회로 기술 유형
    unsigned int tag_arr_ram_cell_tech_type;      // [한국어] 태그 어레이 RAM 셀 기술 유형
    unsigned int tag_arr_peri_global_tech_type;   // [한국어] 태그 어레이 주변 회로 기술 유형

    unsigned int burst_len;       // [한국어] DRAM 버스트 길이 (DRAM 모드에서 사용)
    unsigned int int_prefetch_w;  // [한국어] 내부 프리패치 폭 (DRAM 페이지 버퍼 크기 관련)
    unsigned int page_sz_bits;    // [한국어] DRAM 페이지 크기 (비트)

    unsigned int ic_proj_type;      // interconnect_projection_type
    /* 인터커넥트 예측 유형 — 0: 공격적(aggressive), 1: 보수적(conservative)
     * 설정자: AccelWattch 또는 사용자 설정.
     * 읽는 자: 와이어 RC 파라미터 선택.
     * 값 범위: 0 또는 1. */

    unsigned int wire_is_mat_type;  // wire_inside_mat_type
    /* MAT 내부 와이어 유형 (비트라인, 셀 내부 배선)
     * 값 범위: 0=로컬, 1=세미글로벌, 2=글로벌 등. */

    unsigned int wire_os_mat_type; // wire_outside_mat_type
    /* MAT 외부 와이어 유형 (H-tree, 서브뱅크 간)
     * 값 범위: 0=로컬, 1=세미글로벌, 2=글로벌 등. */

    enum Wire_type wt;        // [한국어] 글로벌 와이어 유형 (Wire_type enum)
    int force_wiretype;       // [한국어] 와이어 유형 강제 지정 여부 (0=탐색, 1=강제)
    bool print_input_args;    // [한국어] 입력 파라미터 출력 여부 (디버깅)
    unsigned int nuca_cache_sz; // TODO
    // [한국어] NUCA(Non-Uniform Cache Access) 캐시 크기 (미사용, 추후 개발)

    int ndbl, ndwl, nspd, ndsam1, ndsam2, ndcm;
    /* CACTI 설계 파라미터:
     * ndbl: 비트라인 방향 서브뱅크 수 분할 (Ndbl)
     * ndwl: 워드라인 방향 서브뱅크 수 분할 (Ndwl)
     * nspd: 비트라인당 셀 수 비율 (Nspd)
     * ndsam1/2: 센스앰프 먹스 레벨 1/2 분할
     * ndcm: 데이터 열 먹스 분할
     * 설정자: force_cache_config=true일 때 사용자 지정, 아니면 CACTI가 탐색.
     * 읽는 자: mem_array 생성 파라미터로 사용. */

    bool force_cache_config;  // [한국어] ndbl/ndwl/nspd 등을 강제 지정할지 탐색할지 여부

    int cache_level;          // [한국어] 캐시 레벨 (0=L2, 1=L3) — NUCA 최적화 기준
    int cores;                // [한국어] 코어 수 — NUCA 뱅크 배치 최적화에 사용
    int nuca_bank_count;      // [한국어] NUCA 뱅크 수
    int force_nuca_bank;      // [한국어] NUCA 뱅크 수 강제 지정 여부

    int delay_wt, dynamic_power_wt, leakage_power_wt,
        cycle_time_wt, area_wt;
    /* 목적 함수 가중치 (UCA) — 설계 공간 탐색 시 각 성능 지표의 상대적 중요도
     * delay_wt: 접근 시간 가중치
     * dynamic_power_wt: 동적 전력 가중치
     * leakage_power_wt: 누설 전력 가중치
     * cycle_time_wt: 사이클 시간 가중치
     * area_wt: 면적 가중치 */

    int delay_wt_nuca, dynamic_power_wt_nuca, leakage_power_wt_nuca,
        cycle_time_wt_nuca, area_wt_nuca;
    // [한국어] NUCA 최적화 목적 함수 가중치 (UCA와 별도)

    int delay_dev, dynamic_power_dev, leakage_power_dev,
        cycle_time_dev, area_dev;
    /* 최적 설계점에서의 허용 편차(%) — 최적값에서 이 비율 이내의 후보만 허용
     * delay_dev: 지연 허용 편차 (%)
     * dynamic_power_dev: 동적 전력 허용 편차 (%)
     * leakage_power_dev / cycle_time_dev / area_dev: 각 지표 편차 */

    int delay_dev_nuca, dynamic_power_dev_nuca, leakage_power_dev_nuca,
        cycle_time_dev_nuca, area_dev_nuca;
    // [한국어] NUCA 최적화 허용 편차

    int ed; //ED or ED2 optimization
    /* 에너지-지연 최적화 방법 선택
     * 0: ED(Energy×Delay) 곱 최소화
     * 1: ED²(Energy×Delay²) 곱 최소화
     * 2: 가중치 및 편차 방법 사용 */

    int nuca; // [한국어] NUCA 모드 활성화 여부 (0=UCA, 1=NUCA)

    bool     fast_access;   // [한국어] 고속 접근 모드 — 태그+데이터 동시 접근 (access_mode=2)
    unsigned int block_sz;  // bytes
    /* 블록(라인) 크기 (바이트) — line_sz와 동일하거나 연관
     * 설정자: 내부 파라미터 변환 시.
     * 읽는 자: 서브어레이 폭 계산. */

    unsigned int tag_assoc;   // [한국어] 태그 어레이 결합도 (assoc과 동일하거나 별도)
    unsigned int data_assoc;  // [한국어] 데이터 어레이 결합도
    bool     is_seq_acc;      // [한국어] 순차 접근(sequential access) 모드 — 태그 먼저 후 데이터
    bool     fully_assoc;     // [한국어] 완전 결합(fully-associative) 캐시 여부
    unsigned int nsets;  // == number_of_sets
    /* 집합(set) 수 — cache_sz / (line_sz * assoc)
     * 설정자: 내부 파라미터 검증 시 자동 계산.
     * 읽는 자: 행 디코더 비트 수, 워드라인 수 결정.
     * 값 범위: 1 ~ 수천. */

    int print_detail; // [한국어] 상세 출력 레벨 (0=최소, 1=표준, 5=최대)

    bool     add_ecc_b_; // [한국어] ECC(Error Correction Code) 비트 추가 여부

  //parameters for design constraint
  double throughput; // [한국어] 처리율(throughput) 제약 (초당 접근 횟수)
  double latency;    // [한국어] 지연(latency) 제약 (초)
  bool pipelinable;  // [한국어] 파이프라인 가능 여부
  int pipeline_stages; // [한국어] 파이프라인 단계 수
  int per_stage_vector; // [한국어] 파이프라인 단계당 데이터 벡터 크기
  bool with_clock_grid; // [한국어] 클럭 그리드 포함 면적 계산 여부
};


/*
 * [한국어]
 * results_mem_array - 단일 메모리 어레이 설계점의 평탄화된(flat) 결과 구조체
 *
 * CACTI 탐색이 끝난 후 최적 태그/데이터 어레이의 상세 타이밍/전력/면적을
 * 외부(McPAT/AccelWattch)로 전달하기 위해 mem_array에서 복사된 필드 모음.
 * 필드는 크게 5그룹으로 구성된다:
 *   1) 파티션 파라미터: Ndwl, Ndbl, Nspd, deg_bl_muxing, Ndsam_lev_1/2,
 *      number_activated_mats_horizontal_direction, number_subbanks, page_size_in_bits
 *   2) 경로별 지연: delay_* (뱅크 라우팅, H-tree, 디코더, 비트라인, SA, 출력 드라이버 등)
 *   3) 접근/사이클 시간: access_time, cycle_time, multisubbank_interleave_cycle_time,
 *      delay_request_network, delay_inside_mat, delay_reply_network,
 *      trcd, cas_latency, precharge_delay (DRAM)
 *   4) 전력 항목: power_* (routing, H-tree, decoder, bitline, SA, precharge,
 *      comparator, crossbar, total)
 *   5) 면적/치수 및 DRAM 에너지: area, all_banks_*, bank_*, subarray_*, mat_*,
 *      routing_area_*, area_efficiency, refresh_power, dram_refresh_period,
 *      dram_array_availability, dyn_read_energy_from_*, leak_power_subbank_*,
 *      leak_power_request_and_reply_networks, activate/read/write/precharge_energy
 */
typedef struct{
  int Ndwl;   // [한국어] 워드라인 방향 서브어레이 분할 수
  int Ndbl;   // [한국어] 비트라인 방향 서브어레이 분할 수
  double Nspd; // [한국어] 서브어레이당 병렬 데이터선 비율
  int deg_bl_muxing; // [한국어] 비트라인 멀티플렉싱 차수
  int Ndsam_lev_1;   // [한국어] 센스앰프 MUX 레벨 1 크기
  int Ndsam_lev_2;   // [한국어] 센스앰프 MUX 레벨 2 크기
  int number_activated_mats_horizontal_direction; // [한국어] 수평 방향 동시 활성 mat 수
  int number_subbanks; // [한국어] 뱅크 내 서브뱅크 수
  int page_size_in_bits; // [한국어] 페이지 모드/주 메모리의 페이지 크기 [비트]
  double delay_route_to_bank;
  double delay_crossbar;
  double delay_addr_din_horizontal_htree;
  double delay_addr_din_vertical_htree;
  double delay_row_predecode_driver_and_block;
  double delay_row_decoder;
  double delay_bitlines;
  double delay_sense_amp;
  double delay_subarray_output_driver;
  double delay_bit_mux_predecode_driver_and_block;
  double delay_bit_mux_decoder;
  double delay_senseamp_mux_lev_1_predecode_driver_and_block;
  double delay_senseamp_mux_lev_1_decoder;
  double delay_senseamp_mux_lev_2_predecode_driver_and_block;
  double delay_senseamp_mux_lev_2_decoder;
  double delay_input_htree;
  double delay_output_htree;
  double delay_dout_vertical_htree;
  double delay_dout_horizontal_htree;
  double delay_comparator;
  double access_time;
  double cycle_time;
  double multisubbank_interleave_cycle_time;
  double delay_request_network;
  double delay_inside_mat;
  double delay_reply_network;
  double trcd;
  double cas_latency;
  double precharge_delay;
  powerDef power_routing_to_bank;
  powerDef power_addr_input_htree;
  powerDef power_data_input_htree;
  powerDef power_data_output_htree;
  powerDef power_addr_horizontal_htree;
  powerDef power_datain_horizontal_htree;
  powerDef power_dataout_horizontal_htree;
  powerDef power_addr_vertical_htree;
  powerDef power_datain_vertical_htree;
  powerDef power_row_predecoder_drivers;
  powerDef power_row_predecoder_blocks;
  powerDef power_row_decoders;
  powerDef power_bit_mux_predecoder_drivers;
  powerDef power_bit_mux_predecoder_blocks;
  powerDef power_bit_mux_decoders;
  powerDef power_senseamp_mux_lev_1_predecoder_drivers;
  powerDef power_senseamp_mux_lev_1_predecoder_blocks;
  powerDef power_senseamp_mux_lev_1_decoders;
  powerDef power_senseamp_mux_lev_2_predecoder_drivers;
  powerDef power_senseamp_mux_lev_2_predecoder_blocks;
  powerDef power_senseamp_mux_lev_2_decoders;
  powerDef power_bitlines;
  powerDef power_sense_amps;
  powerDef power_prechg_eq_drivers;
  powerDef power_output_drivers_at_subarray;
  powerDef power_dataout_vertical_htree;
  powerDef power_comparators;
  powerDef power_crossbar;
  powerDef total_power;
  double area;
  double all_banks_height;
  double all_banks_width;
  double bank_height;
  double bank_width;
  double subarray_memory_cell_area_height;
  double subarray_memory_cell_area_width;
  double mat_height;
  double mat_width;
  double routing_area_height_within_bank;
  double routing_area_width_within_bank;
  double area_efficiency;
//  double perc_power_dyn_routing_to_bank;
//  double perc_power_dyn_addr_horizontal_htree;
//  double perc_power_dyn_datain_horizontal_htree;
//  double perc_power_dyn_dataout_horizontal_htree;
//  double perc_power_dyn_addr_vertical_htree;
//  double perc_power_dyn_datain_vertical_htree;
//  double perc_power_dyn_row_predecoder_drivers;
//  double perc_power_dyn_row_predecoder_blocks;
//  double perc_power_dyn_row_decoders;
//  double perc_power_dyn_bit_mux_predecoder_drivers;
//  double perc_power_dyn_bit_mux_predecoder_blocks;
//  double perc_power_dyn_bit_mux_decoders;
//  double perc_power_dyn_senseamp_mux_lev_1_predecoder_drivers;
//  double perc_power_dyn_senseamp_mux_lev_1_predecoder_blocks;
//  double perc_power_dyn_senseamp_mux_lev_1_decoders;
//  double perc_power_dyn_senseamp_mux_lev_2_predecoder_drivers;
//  double perc_power_dyn_senseamp_mux_lev_2_predecoder_blocks;
//  double perc_power_dyn_senseamp_mux_lev_2_decoders;
//  double perc_power_dyn_bitlines;
//  double perc_power_dyn_sense_amps;
//  double perc_power_dyn_prechg_eq_drivers;
//  double perc_power_dyn_subarray_output_drivers;
//  double perc_power_dyn_dataout_vertical_htree;
//  double perc_power_dyn_comparators;
//  double perc_power_dyn_crossbar;
//  double perc_power_dyn_spent_outside_mats;
//  double perc_power_leak_routing_to_bank;
//  double perc_power_leak_addr_horizontal_htree;
//  double perc_power_leak_datain_horizontal_htree;
//  double perc_power_leak_dataout_horizontal_htree;
//  double perc_power_leak_addr_vertical_htree;
//  double perc_power_leak_datain_vertical_htree;
//  double perc_power_leak_row_predecoder_drivers;
//  double perc_power_leak_row_predecoder_blocks;
//  double perc_power_leak_row_decoders;
//  double perc_power_leak_bit_mux_predecoder_drivers;
//  double perc_power_leak_bit_mux_predecoder_blocks;
//  double perc_power_leak_bit_mux_decoders;
//  double perc_power_leak_senseamp_mux_lev_1_predecoder_drivers;
//  double perc_power_leak_senseamp_mux_lev_1_predecoder_blocks;
//  double perc_power_leak_senseamp_mux_lev_1_decoders;
//  double perc_power_leak_senseamp_mux_lev_2_predecoder_drivers;
//  double perc_power_leak_senseamp_mux_lev_2_predecoder_blocks;
//  double perc_power_leak_senseamp_mux_lev_2_decoders;
//  double perc_power_leak_bitlines;
//  double perc_power_leak_sense_amps;
//  double perc_power_leak_prechg_eq_drivers;
//  double perc_power_leak_subarray_output_drivers;
//  double perc_power_leak_dataout_vertical_htree;
//  double perc_power_leak_comparators;
//  double perc_power_leak_crossbar;
//  double perc_leak_mats;
//  double perc_active_mats;
  double refresh_power;
  double dram_refresh_period;
  double dram_array_availability;
  double dyn_read_energy_from_closed_page;
  double dyn_read_energy_from_open_page;
  double leak_power_subbank_closed_page;
  double leak_power_subbank_open_page;
  double leak_power_request_and_reply_networks;
  double activate_energy;
  double read_energy;
  double write_energy;
  double precharge_energy;
} results_mem_array;


/*
 * [한국어] uca_org_t — UCA(Uniform Cache Access) 캐시 분석 최종 결과를 담는 클래스
 * CACTI의 설계 공간 탐색이 완료된 후 최적 설계점의 타이밍/전력/면적 결과를 저장.
 * AccelWattch는 이 클래스의 access_time, power 필드를 읽어 GPU 전력을 추정한다.
 */
class uca_org_t
{
  public:
    mem_array * tag_array2;
    /* 최적 태그 어레이 설계 후보 포인터
     * 설정자: Ucache가 탐색을 완료한 후 최적 설계점 선택.
     * 읽는 자: find_delay/find_energy/find_area가 태그 어레이 결과를 읽음.
     * 값 범위: 유효한 mem_array 포인터 또는 NULL(순수 RAM일 때).
     * 동기화: 단일 스레드 초기화 후 읽기 전용. */

    mem_array * data_array2;
    /* 최적 데이터 어레이 설계 후보 포인터
     * 설정자: Ucache 탐색 완료 후 선택.
     * 읽는 자: find_delay/find_energy/find_area, AccelWattch.
     * 값 범위: 유효한 mem_array 포인터.
     * 동기화: 단일 스레드 초기화 후 읽기 전용. */

    double access_time;
    /* 캐시 접근 시간 (초) — 태그+데이터 어레이 경로의 최대 지연
     * 설정자: find_delay()에서 access_mode에 따라 계산.
     * 읽는 자: AccelWattch가 GPU 캐시 접근 지연 추정에 사용.
     * 값 범위: 수십~수백 ps (공정/크기에 따라). */

    double cycle_time;
    /* 최소 사이클 시간 (초) — 연속 접근 가능한 최소 간격
     * 설정자: find_cyc()에서 계산.
     * 읽는 자: 최대 캐시 주파수 추정.
     * 값 범위: access_time 이상. */

    double area;
    /* 캐시 총 면적 (m²) — 태그+데이터 어레이+라우팅 포함
     * 설정자: find_area()에서 계산.
     * 읽는 자: 면적 제약 검사, AccelWattch 면적 보고.
     * 값 범위: 수십~수백 mm² (캐시 크기에 따라). */

    double area_efficiency; // [한국어] 면적 효율 (%) — 실제 셀 면적 / 총 면적

    powerDef power;
    /* 캐시 총 전력 — 읽기/쓰기/검색 연산별 동적+누설+게이트누설 합계
     * 설정자: find_energy()에서 태그+데이터 어레이 전력을 합산.
     * 읽는 자: AccelWattch GPU 전력 누적의 핵심 입력.
     * 동기화: 단일 스레드 초기화 후 읽기 전용. */

    double leak_power_with_sleep_transistors_in_mats;
    /* 슬립 트랜지스터(sleep transistor) 적용 시 누설 전력 절감량
     * 설정자: 슬립 트랜지스터 옵션 활성화 시 계산.
     * 읽는 자: 저전력 모드 전력 분석. */

    double cache_ht; // [한국어] 캐시 높이 (m) — 물리 레이아웃 치수
    double cache_len; // [한국어] 캐시 길이/폭 (m) — 물리 레이아웃 치수

    char file_n[100]; // [한국어] CACTI 결과 저장 파일명 (100자 이내)
    double vdd_periph_global; // [한국어] 주변 회로 전원 전압 (V)

    bool valid;
    /* 이 결과가 유효한지 여부
     * 설정자: 탐색 완료 후 유효한 설계점이 발견되면 true.
     * 읽는 자: 상위 호출자가 결과 유효성 검사.
     * 값 범위: true(유효) / false(설계점 없음). */

    results_mem_array tag_array;   // [한국어] 태그 어레이 상세 타이밍/전력/면적 분해 결과 (flat 구조체)
    results_mem_array data_array;  // [한국어] 데이터 어레이 상세 타이밍/전력/면적 분해 결과

    uca_org_t(); // [한국어] 기본 생성자 — 포인터를 NULL, 숫자 필드를 0으로 초기화

    /* [한국어] find_delay - 접근 시간 계산 (access_mode에 따라 다른 방식)
     * 호출 체인: Ucache → [find_delay] → tag_array2/data_array2 접근 시간 참조 */
    void find_delay();

    /* [한국어] find_energy - 총 전력 계산 (태그+데이터 어레이 전력 합산)
     * 호출 체인: Ucache → [find_energy] → power 필드 설정 */
    void find_energy();

    /* [한국어] find_area - 총 면적 계산 (태그+데이터+라우팅 면적 합산)
     * 호출 체인: Ucache → [find_area] → area 필드 설정 */
    void find_area();

    /* [한국어] find_cyc - 사이클 시간 계산 (피드백 경로의 최대 지연)
     * 호출 체인: Ucache → [find_cyc] → cycle_time 설정 */
    void find_cyc();

    void adjust_area();//for McPAT only to adjust routing overhead
    // [한국어] McPAT 전용: 라우팅 오버헤드를 반영하여 면적 조정

    /* [한국어] cleanup - 동적 할당된 mem_array 객체 해제
     * 호출 체인: 결과 사용 후 → [cleanup] → delete tag_array2, delete data_array2 */
    void cleanup();

    ~uca_org_t(){}; // [한국어] 소멸자 — cleanup()을 명시적으로 호출해야 메모리 해제됨
};

/* [한국어]
 * reconfigure - 기존 결과를 새 InputParameter로 재구성
 * @local_interface : 새 입력 파라미터
 * @fin_res         : 재구성될 결과 객체
 * McPAT에서 여러 캐시 설정을 순차적으로 분석할 때 사용.
 * 호출 체인: McPAT → [reconfigure] → cacti_interface */
void reconfigure(InputParameter *local_interface, uca_org_t *fin_res);

/* [한국어]
 * cacti_interface - 설정 파일로부터 CACTI 실행하는 최상위 진입 함수
 * @infile_name : CACTI cfg 파일 경로
 * @return      : UCA 분석 결과 (uca_org_t)
 * 호출 체인: 커맨드라인 CACTI 실행 → [cacti_interface(string)] */
uca_org_t cacti_interface(const string & infile_name);
/* [한국어]
 * cacti_interface - InputParameter 구조체를 직접 받는 McPAT 인터페이스
 * @local_interface : 미리 채워진 InputParameter 포인터
 * @return          : UCA 분석 결과
 * AccelWattch/McPAT가 주로 사용하는 인터페이스 — 설정 파일 없이 파라미터를 직접 전달.
 * 호출 체인: AccelWattch → [cacti_interface(InputParameter*)] → Ucache::solve() → uca_org_t */
//McPAT's plain interface, please keep !!!
uca_org_t cacti_interface(InputParameter * const local_interface);

/* [한국어]
 * init_interface - CACTI 인터페이스 초기화 (파라미터 검증 및 전처리)
 * @local_interface : 입력 파라미터 포인터
 * @return          : 초기화된 UCA 결과 (실제 탐색 전 구조 설정)
 * 호출 체인: AccelWattch → [init_interface] → 내부 파라미터 정규화 */
//McPAT's plain interface, please keep !!!
uca_org_t init_interface(InputParameter * const local_interface);
//McPAT's plain interface, please keep !!!
uca_org_t cacti_interface(
	    int cache_size,
	    int line_size,
	    int associativity,
	    int rw_ports,
	    int excl_read_ports,
	    int excl_write_ports,
	    int single_ended_read_ports,
	    int search_ports,
	    int banks,
	    double tech_node,
	    int output_width,
	    int specific_tag,
	    int tag_width,
	    int access_mode,
	    int cache,
	    int main_mem,
	    int obj_func_delay,
	    int obj_func_dynamic_power,
	    int obj_func_leakage_power,
	    int obj_func_cycle_time,
	    int obj_func_area,
	    int dev_func_delay,
	    int dev_func_dynamic_power,
	    int dev_func_leakage_power,
	    int dev_func_area,
	    int dev_func_cycle_time,
	    int ed_ed2_none, // 0 - ED, 1 - ED^2, 2 - use weight and deviate
	    int temp,
	    int wt, //0 - default(search across everything), 1 - global, 2 - 5% delay penalty, 3 - 10%, 4 - 20 %, 5 - 30%, 6 - low-swing
	    int data_arr_ram_cell_tech_flavor_in,
	    int data_arr_peri_global_tech_flavor_in,
	    int tag_arr_ram_cell_tech_flavor_in,
	    int tag_arr_peri_global_tech_flavor_in,
	    int interconnect_projection_type_in,
	    int wire_inside_mat_type_in,
	    int wire_outside_mat_type_in,
	    int REPEATERS_IN_HTREE_SEGMENTS_in,
	    int VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in,
	    int BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in,
	    int PAGE_SIZE_BITS_in,
	    int BURST_LENGTH_in,
	    int INTERNAL_PREFETCH_WIDTH_in,
	    int force_wiretype,
	    int wiretype,
	    int force_config,
	    int ndwl,
	    int ndbl,
	    int nspd,
	    int ndcm,
	    int ndsam1,
	    int ndsam2,
	    int ecc);
//    int cache_size,
//    int line_size,
//    int associativity,
//    int rw_ports,
//    int excl_read_ports,
//    int excl_write_ports,
//    int single_ended_read_ports,
//    int banks,
//    double tech_node,
//    int output_width,
//    int specific_tag,
//    int tag_width,
//    int access_mode,
//    int cache,
//    int main_mem,
//    int obj_func_delay,
//    int obj_func_dynamic_power,
//    int obj_func_leakage_power,
//    int obj_func_area,
//    int obj_func_cycle_time,
//    int dev_func_delay,
//    int dev_func_dynamic_power,
//    int dev_func_leakage_power,
//    int dev_func_area,
//    int dev_func_cycle_time,
//    int temp,
//    int data_arr_ram_cell_tech_flavor_in,
//    int data_arr_peri_global_tech_flavor_in,
//    int tag_arr_ram_cell_tech_flavor_in,
//    int tag_arr_peri_global_tech_flavor_in,
//    int interconnect_projection_type_in,
//    int wire_inside_mat_type_in,
//    int wire_outside_mat_type_in,
//    int REPEATERS_IN_HTREE_SEGMENTS_in,
//    int VERTICAL_HTREE_WIRES_OVER_THE_ARRAY_in,
//    int BROADCAST_ADDR_DATAIN_OVER_VERTICAL_HTREES_in,
////    double MAXAREACONSTRAINT_PERC_in,
////    double MAXACCTIMECONSTRAINT_PERC_in,
////    double MAX_PERC_DIFF_IN_DELAY_FROM_BEST_DELAY_REPEATER_SOLUTION_in,
//    int PAGE_SIZE_BITS_in,
//    int BURST_LENGTH_in,
//    int INTERNAL_PREFETCH_WIDTH_in);

//Naveen's interface
uca_org_t cacti_interface(
    int cache_size,
    int line_size,
    int associativity,
    int rw_ports,
    int excl_read_ports,
    int excl_write_ports,
    int single_ended_read_ports,
    int banks,
    double tech_node,
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
    int data_arr_ram_cell_tech_flavor_in,
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
    int p_input);

/*
 * [한국어] mem_array — 단일 메모리 어레이 설계 후보의 완전한 타이밍/전력/면적 정보
 * CACTI의 설계 공간 탐색에서 Ndwl/Ndbl/Nspd 등의 파라미터 조합마다 하나의 mem_array가 생성.
 * uca_org_t는 tag_array2와 data_array2로 각각 최적 mem_array를 선택하여 보관.
 * Ucache::solve()가 모든 후보 중 목적 함수를 최소화하는 mem_array를 선택한다.
 */
class mem_array
{
  public:
  int    Ndcm;
  /* 데이터 열 먹스 분할 인수 (Data Column Mux)
   * 설정자: 탐색 파라미터 생성 시.
   * 읽는 자: 어레이 크기, 비트 먹스 설계.
   * 값 범위: 1 이상 정수. */

  int    Ndwl;
  /* 워드라인 방향 서브어레이 분할 수 (Number of Divisions in WordLine direction)
   * 설정자: 설계 공간 탐색 파라미터.
   * 읽는 자: 행 디코더 설계, 어레이 면적 계산.
   * 값 범위: 1, 2, 4, 8, ... (2의 거듭제곱). */

  int    Ndbl;
  /* 비트라인 방향 서브어레이 분할 수 (Number of Divisions in BitLine direction)
   * 설정자: 설계 공간 탐색 파라미터.
   * 읽는 자: 열 디코더, 비트라인 길이 계산.
   * 값 범위: 1, 2, 4, 8, ... (2의 거듭제곱). */

  double Nspd;
  /* 비트라인당 셀 수 비율 (Number of Subarray Pages Divisor)
   * 설정자: 설계 공간 탐색 파라미터 — 실수형으로 비트라인 공유 비율을 표현.
   * 읽는 자: 워드라인 길이, 비트라인 용량 계산.
   * 값 범위: 0.25, 0.5, 1, 2, ... */

  int    deg_bl_muxing; // [한국어] 비트라인 먹스 차수 — 센스앰프 하나당 비트라인 수
  int    Ndsam_lev_1;   // [한국어] 센스앰프 먹스 레벨 1 분할 (1차 출력 먹스)
  int    Ndsam_lev_2;   // [한국어] 센스앰프 먹스 레벨 2 분할 (2차 출력 먹스)

  double access_time;
  /* 접근 시간 (초) — 이 어레이 설계점의 총 접근 지연
   * 설정자: uca_org_t::find_delay()에서 경로 지연 합산.
   * 읽는 자: 최적 설계점 선택 기준, uca_org_t.access_time 설정.
   * 값 범위: 수십~수백 ps. */

  double cycle_time;
  /* 사이클 시간 (초) — 이 어레이의 최소 연속 접근 간격
   * 설정자: uca_org_t::find_cyc()에서 계산.
   * 읽는 자: 최대 동작 주파수 추정.
   * 값 범위: access_time 이상. */

  double multisubbank_interleave_cycle_time;
  /* 다중 서브뱅크 인터리빙 사이클 시간 — 병렬 서브뱅크 접근 시 유효 사이클 시간
   * 설정자: 다중 서브뱅크 있을 때 계산.
   * 읽는 자: 인터리빙 모드 성능 추정. */

  double area_ram_cells; // [한국어] RAM 셀 면적만의 합 (m²) — 오버헤드 제외
  double area;
  /* 총 어레이 면적 (m²) — RAM 셀 + 디코더 + 감지증폭기 + 라우팅 포함
   * 설정자: Component::compute_area()들의 합산.
   * 읽는 자: 목적 함수(면적 최적화), uca_org_t.area 설정.
   * 값 범위: 어레이 크기에 비례, mm² 수준. */

  powerDef power;
  /* 전력 합계 — 이 어레이의 읽기/쓰기/검색 연산별 전력 성분
   * 설정자: 각 서브회로 compute_area/compute_delays 결과 누적.
   * 읽는 자: uca_org_t.power 계산, AccelWattch.
   * 동기화: 단일 스레드 초기화. */

  double delay_senseamp_mux_decoder;
  /* 감지증폭기 먹스 + 디코더까지의 지연 (초)
   * 설정자: 순차 접근 모드에서 태그 확인 이후 데이터 접근 경로 분석.
   * 읽는 자: uca_org_t::find_delay()의 순차 접근 시간 계산. */

  double delay_before_subarray_output_driver;
  /* 서브어레이 출력 드라이버 이전까지의 지연 (초)
   * 설정자: MAT 내부 경로 지연 합산.
   * 읽는 자: uca_org_t::find_delay() — 일반 접근 모드에서 사용. */

  double delay_from_subarray_output_driver_to_output;
  /* 서브어레이 출력 드라이버에서 최종 출력까지의 지연 (초)
   * 설정자: H-tree 출력 경로 지연.
   * 읽는 자: uca_org_t::find_delay() — 접근 시간 계산의 마지막 단계. */

  double height; // [한국어] 어레이 높이 (m) — 물리 레이아웃 치수
  double width;  // [한국어] 어레이 폭 (m) — 물리 레이아웃 치수

  double mat_height;     // [한국어] MAT(Memory Array Tile) 높이 (m)
  double mat_length;     // [한국어] MAT 길이/폭 (m)
  double subarray_length;// [한국어] 서브어레이 길이 (m) — MAT 내 분할 단위
  double subarray_height;// [한국어] 서브어레이 높이 (m)

  /* [한국어] 접근 경로별 지연 (초) — 뱅크 라우팅부터 출력 H-tree까지 각 단계별 기여분 */
  double delay_route_to_bank,         // [한국어] 뱅크 라우팅 지연
         delay_input_htree,             // [한국어] 주소/데이터 입력 H-tree 지연
         delay_row_predecode_driver_and_block, // [한국어] 행 예비디코더 드라이버+블록 지연
         delay_row_decoder,             // [한국어] 행 디코더 지연
         delay_bitlines,                // [한국어] 비트라인 충전/방전 지연
         delay_sense_amp,               // [한국어] 감지증폭기 지연
         delay_subarray_output_driver,  // [한국어] 서브어레이 출력 드라이버 지연
         delay_dout_htree,              // [한국어] 데이터 출력 H-tree 지연
         delay_comparator,              // [한국어] 태그 비교기 지연
         delay_matchlines;              // [한국어] CAM/FA 매치라인 지연

  double all_banks_height,  // [한국어] 전체 뱅크 조직 높이 (m)
         all_banks_width,   // [한국어] 전체 뱅크 조직 폭 (m)
         area_efficiency;   // [한국어] 면적 효율 (%) — 실제 셀 면적 / 총 면적

  /* [한국어] 일반 SRAM/캐시 경로의 동적+누설 전력 (powerDef: read/write/search Op) */
  powerDef power_routing_to_bank;     // [한국어] 뱅크 라우팅 전력
  powerDef power_addr_input_htree;    // [한국어] 주소 입력 H-tree 전력
  powerDef power_data_input_htree;    // [한국어] 데이터 입력 H-tree 전력
  powerDef power_data_output_htree;   // [한국어] 데이터 출력 H-tree 전력
  powerDef power_htree_in_search;     // [한국어] 검색 입력 H-tree 전력 (CAM/FA)
  powerDef power_htree_out_search;    // [한국어] 검색 출력 H-tree 전력 (CAM/FA)
  powerDef power_row_predecoder_drivers; // [한국어] 행 예비디코더 드라이버 전력
  powerDef power_row_predecoder_blocks;  // [한국어] 행 예비디코더 블록 전력
  powerDef power_row_decoders;           // [한국어] 행 디코더 전력
  powerDef power_bit_mux_predecoder_drivers; // [한국어] 비트 MUX 예비디코더 드라이버 전력
  powerDef power_bit_mux_predecoder_blocks;  // [한국어] 비트 MUX 예비디코더 블록 전력
  powerDef power_bit_mux_decoders;           // [한국어] 비트 MUX 디코더 전력
  powerDef power_senseamp_mux_lev_1_predecoder_drivers; // [한국어] SA MUX L1 예비디코더 드라이버 전력
  powerDef power_senseamp_mux_lev_1_predecoder_blocks;  // [한국어] SA MUX L1 예비디코더 블록 전력
  powerDef power_senseamp_mux_lev_1_decoders;           // [한국어] SA MUX L1 디코더 전력
  powerDef power_senseamp_mux_lev_2_predecoder_drivers; // [한국어] SA MUX L2 예비디코더 드라이버 전력
  powerDef power_senseamp_mux_lev_2_predecoder_blocks;  // [한국어] SA MUX L2 예비디코더 블록 전력
  powerDef power_senseamp_mux_lev_2_decoders;           // [한국어] SA MUX L2 디코더 전력
  powerDef power_bitlines;          // [한국어] 비트라인 전력
  powerDef power_sense_amps;        // [한국어] 감지증폭기 전력
  powerDef power_prechg_eq_drivers; // [한국어] 프리차지/이퀄라이즈 드라이버 전력
  powerDef power_output_drivers_at_subarray; // [한국어] 서브어레이 출력 드라이버 전력
  powerDef power_dataout_vertical_htree;     // [한국어] 수직 데이터 출력 H-tree 전력
  powerDef power_comparators;       // [한국어] 태그 비교기 전력

  /* [한국어] CAM/FA(Fully Associative) 전용 검색 경로 전력 */
  powerDef power_cam_bitline_precharge_eq_drv; // [한국어] CAM 비트라인 프리차지/이퀄라이즈 드라이버 전력
  powerDef power_searchline;        // [한국어] 검색 라인(searchline) 전력
  powerDef power_searchline_precharge; // [한국어] 검색 라인 프리차지 전력
  powerDef power_matchlines;        // [한국어] 매치라인 전력
  powerDef power_matchline_precharge; // [한국어] 매치라인 프리차지 전력
  powerDef power_matchline_to_wordline_drv; // [한국어] 매치라인→워드라인 드라이버 전력

  min_values_t *arr_min; // [한국어] 이 후보가 속한 어레이 클래스의 최솟값 추적기
  enum Wire_type wt;     // [한국어] 이 후보에 사용된 와이어 타입

  /* [한국어] DRAM 주 메모리 통계 — 활성화/읽기/쓰기/프리차지 에너지 및 누설 전력 */
  double activate_energy, read_energy, write_energy, precharge_energy, // [한국어] 각 동작 에너지 (J)
  refresh_power, leak_power_subbank_closed_page, leak_power_subbank_open_page, // [한국어] 리프레시/누설 전력 (W)
  leak_power_request_and_reply_networks; // [한국어] 요청/응답 네트워크 누설 전력 (W)

  double precharge_delay; // [한국어] DRAM 프리차지 지연 (초)

  static bool lt(const mem_array * m1, const mem_array * m2); // [한국어] mem_array 후보 정렬용 6-key 사전순 비교자
};


#endif
