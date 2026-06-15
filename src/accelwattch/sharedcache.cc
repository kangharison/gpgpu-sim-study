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

/*
 * [한국어 설명] AccelWattch 공유 캐시(L2/L3/Directory) 전력 모델 구현 (sharedcache.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT/AccelWattch에서 GPU/CPU 공유 캐시(SharedCache)의 전력과 면적을
 * 모델링한다. SharedCache는 L2 SRAM(데이터/태그 배열)뿐 아니라 미스 버퍼(MissB),
 * 채우기 버퍼(FillB), 프리페치 버퍼(PrefetchB), 쓰기 버퍼(WBB) 등의
 * 보조 SRAM 구조물을 ArrayST(DataCache)로 모델링한다. computeEnergy()는
 * TDP(peak) 모드와 런타임 모드를 모두 지원하며, XML에서 수집된 실제 L2/L3/
 * Directory 접근 횟수를 기반으로 동적 에너지(dynamic energy)를 산출하고,
 * CACTI가 계산한 local_result.power 누설 전력과 결합해 최종 power/rt_power를
 * 완성한다. GPU 모델에서는 통합 L2 캐시(all SM 공유) 전력 추정에 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층:
 *   gpgpu_sim::cycle() → AccelWattch 카운터 수집
 *     → gpgpu_sim_wrapper::compute() [gpgpu_sim_wrapper.cc]
 *         → Processor::Processor() / compute() [processor.cc]
 *             → [이 파일] SharedCache 객체 생성 및 computeEnergy(false)
 *                 → SharedCache::displayEnergy() (결과 출력)
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 종료 후 전력 보고서 생성 단계.
 * 타이밍 시뮬레이션과는 별개로, McPAT이 사후적으로 에너지를 추정한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - sharedcache.h           : SharedCache/CCdir 클래스 선언, CacheDynParam,
 *                               statsDef, DataCache 필드 정의
 *   - XML_Parse.h / ParseXML  : XML 설정 트리(sys.L2[], sys.L3[], sys.l2 등)
 *   - array.h / DataCache     : SRAM 배열 모델(ArrayST)과 캐시 보조 버퍼 집합
 *   - cacti/parameter.h       : CACTI InputParameter (용량, 연상도, 라인 크기,
 *                               포트 수, 클럭 등)
 *   - logic.h / CacheDynParam : XML에서 파싱된 캐시 동적 파라미터(duty_cycle,
 *                               throughput, latency, buffer 크기 등)
 * 이 파일에 의존하는 모듈:
 *   - processor.cc            : Processor가 SharedCache 객체를 생성·소유
 *   - processor.h             : SharedCache 전방 선언/사용
 *
 * === 주요 함수 요약 ===
 * SharedCache::SharedCache()   : XML 파라미터를 읽어 캐시/버퍼 ArrayST를 생성하고
 *                                면적을 누적한다.
 * SharedCache::computeEnergy() : TDP 또는 런타임 통계를 이용해 동적 에너지와
 *                                누설 전력을 계산하여 power/rt_power에 반영한다.
 * SharedCache::displayEnergy() : 계산된 면적/Peak Dynamic/누설/Runtime Dynamic을
 *                                출력한다.
 * SharedCache::set_cache_param() : cacheL(L2/L3/L1Directory/L2Directory)에 따라
 *                                  cachep 필드를 XML에서 채운다.
 */
#include "sharedcache.h"          // [한국어] SharedCache 클래스 선언
#include <assert.h>              // [한국어] assert
#include <string.h>              // [한국어] string 함수
#include <algorithm>             // [한국어] std::min/max 등
#include <cmath>                 // [한국어] ceil/log2 등
#include <iostream>              // [한국어] cout/cerr
#include "XML_Parse.h"            // [한국어] ParseXML 및 시스템 파라미터
#include "array.h"              // [한국어] ArrayST/DataCache
#include "cacti/arbiter.h"        // [한국어] CACTI arbiter
#include "cacti/basic_circuit.h" // [한국어] cmos 누설 함수
#include "cacti/parameter.h"      // [한국어] CACTI InputParameter
#include "const.h"               // [한국어] McPAT 상수
#include "io.h"                  // [한국어] CACTI 입출력
#include "logic.h"               // [한국어] CacheDynParam 등

/*
 * [한국어]
 * SharedCache::SharedCache - 공유 캐시 전력 모델 생성자
 *
 * @XML_interface: ParseXML 객체 포인터 (시스템 설정 및 시뮬레이션 통계).
 * @ithCache_:    XML sys.L2[]/sys.L3[] 등에서 이 인스턴스의 인덱스.
 * @interface_ip_: CACTI InputParameter 초기값 포인터.
 * @cacheL_:      모델링할 캐시 계층 (L2/L3/L1Directory/L2Directory).
 *
 * 동작 순서:
 *   1. Private L2 여부에 따라 장치 유형(Core_device vs LLC_device)과 코어 유형 설정.
 *   2. Embedded/GPU 환경에 따른 와이어 타입(Global_30 vs Global) 선택.
 *   3. set_cache_param()으로 cachep(용량/연상도/라인/버퍼 크기 등) 초기화.
 *   4. 주 캐시 SRAM 배열(unicache.caches)과 missb/ifb/prefetchb/wbb를 생성.
 *   5. 각 ArrayST의 local_result.area를 unicache.area 및 Component::area에 누적.
 */
SharedCache::SharedCache(ParseXML* XML_interface, int ithCache_,
                         InputParameter* interface_ip_,
                         enum cache_level cacheL_)
    : XML(XML_interface),
      ithCache(ithCache_),
      interface_ip(*interface_ip_),
      cacheL(cacheL_),
      dir_overhead(0) {
  int idx;
  int tag, data;
  bool debug;
  enum Device_ty device_t;
  enum Core_type core_t;
  double size, line, assoc, banks;
  // [한국어] Private L2일 때는 코어 장치/유형으로 모델링
  if (cacheL == L2 && XML->sys.Private_L2) {
    device_t = Core_device;
    core_t = (enum Core_type)XML->sys.core[ithCache].machine_type;
  // [한국어] 공유 LLC(L3) 또는 Directory는 LLC 장치, Inorder 코어 가정
  } else {
    device_t = LLC_device;
    core_t = Inorder;
  }

  debug = false;
  // [한국어] 임베디드/GPU 환경: 저전력 글로벌 와이어 모델 사용
  if (XML->sys.Embedded) {
    interface_ip.wt = Global_30;
    interface_ip.wire_is_mat_type = 0;
    interface_ip.wire_os_mat_type = 1;
  } else {
    interface_ip.wt = Global;
    interface_ip.wire_is_mat_type = 2;
    interface_ip.wire_os_mat_type = 2;
  }
  // [한국어] XML에서 L2/L3/Directory 파라미터를 cachep로 로드
  set_cache_param();

  // All lower level cache are physically indexed and tagged.
  // [한국어] CACTI용 캐시 용량/라인/연상도/뱅크 수를 cachep에서 복사
  size = cachep.capacity;
  line = cachep.blockW;
  assoc = cachep.assoc;
  banks = cachep.nbanks;
  // [한국어] ST(포인터+태그) Directory: CAM 구조로 모델링(assoc=0, search port=1)
  if ((cachep.dir_ty == ST && cacheL == L1Directory) ||
      (cachep.dir_ty == ST && cacheL == L2Directory)) {
    assoc = 0;
    tag = XML->sys.physical_address_width + EXTRA_TAG_BITS;
    interface_ip.num_search_ports = 1;
  } else {
    // [한국어] 일반 캐시: 인덱스 비트 = log2(용량/라인/연상도)
    idx = debug ? 9 : int(ceil(log2(size / line / assoc)));
    tag = debug ? 51
                : XML->sys.physical_address_width - idx -
                      int(ceil(log2(line))) + EXTRA_TAG_BITS;
    interface_ip.num_search_ports = 0;
    // [한국어] SBT(흩어진 디렉토리 비트)는 데이터 블록에 디렉토리 비트를 병합
    if (cachep.dir_ty == SBT) {
      dir_overhead =
          ceil(XML->sys.number_of_cores / 8.0) * 8 / (cachep.blockW * 8);
      line = cachep.blockW * (1 + dir_overhead);
      size = cachep.capacity * (1 + dir_overhead);
    }
  }
  //  if (XML->sys.first_level_dir==2)
  //	  tag += int(XML->sys.domain_size + 5);
  // [한국어] CACTI에 태그 폭을 명시적으로 지정
  interface_ip.specific_tag = 1;
  interface_ip.tag_w = tag;
  interface_ip.cache_sz = (int)size;
  interface_ip.line_sz = (int)line;
  interface_ip.assoc = (int)assoc;
  interface_ip.nbanks = (int)banks;
  interface_ip.out_w = interface_ip.line_sz * 8 / 2;
  interface_ip.access_mode = 1;
  interface_ip.throughput = cachep.throughput;
  interface_ip.latency = cachep.latency;
  interface_ip.is_cache = true;
  interface_ip.pure_ram = false;
  interface_ip.pure_cam = false;
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t = 1;
  interface_ip.num_rw_ports = 1;  // lower level cache usually has one port.
  interface_ip.num_rd_ports = 0;
  interface_ip.num_wr_ports = 0;
  interface_ip.num_se_rd_ports = 0;
  //  interface_ip.force_cache_config  =true;
  //  interface_ip.ndwl = 4;
  //  interface_ip.ndbl = 8;
  //  interface_ip.nspd = 1;
  //  interface_ip.ndcm =1 ;
  //  interface_ip.ndsam1 =1;
  //  interface_ip.ndsam2 =1;
  // [한국어] 주 캐시 SRAM 배열(ArrayST) 생성: 태그+데이터 또는 Directory
  unicache.caches =
      new ArrayST(&interface_ip, cachep.name + "cache", device_t, true, core_t);
  // [한국어] unicache 총면적에 주 캐시 면적 누적
  unicache.area.set_area(unicache.area.get_area() +
                         unicache.caches->local_result.area);
  // [한국어] SharedCache Component 전체 면적에 주 캐시 면적 추가
  area.set_area(area.get_area() + unicache.caches->local_result.area);
  // [한국어] 이후 보조 버퍼들은 CACTI가 자동 최적화하도록 강제 설정 해제
  interface_ip.force_cache_config = false;

  // [한국어] 일반 캐시/Directory가 아닌 경우에만 missb/ifb/prefetchb/wbb 생성
  if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||
        (cachep.dir_ty == ST && cacheL == L2Directory))) {
    tag = XML->sys.physical_address_width + EXTRA_TAG_BITS;
    data = (XML->sys.physical_address_width) + int(ceil(log2(size / line))) +
           unicache.caches->l_ip.line_sz;
    interface_ip.specific_tag = 1;
    interface_ip.tag_w = tag;
    interface_ip.line_sz =
        int(ceil(data / 8.0));  // int(ceil(pow(2.0,ceil(log2(data)))/8.0));
    interface_ip.cache_sz = cachep.missb_size * interface_ip.line_sz;
    interface_ip.assoc = 0;
    interface_ip.is_cache = true;
    interface_ip.pure_ram = false;
    interface_ip.pure_cam = false;
    interface_ip.nbanks = 1;
    interface_ip.out_w = interface_ip.line_sz * 8 / 2;
    interface_ip.access_mode = 0;
    interface_ip.throughput = cachep.throughput;  // means cycle time
    interface_ip.latency = cachep.latency;        // means access time
    interface_ip.obj_func_dyn_energy = 0;
    interface_ip.obj_func_dyn_power = 0;
    interface_ip.obj_func_leak_power = 0;
    interface_ip.obj_func_cycle_t = 1;
    interface_ip.num_rw_ports = 1;
    interface_ip.num_rd_ports = 0;
    interface_ip.num_wr_ports = 0;
    interface_ip.num_se_rd_ports = 0;
    interface_ip.num_search_ports = 1;
    // [한국어] Miss Buffer(MSHR): 태그+데이터 상태를 CAM+RAM으로 모델링
    unicache.missb = new ArrayST(&interface_ip, cachep.name + "MissB", device_t,
                                 true, core_t);
    unicache.area.set_area(unicache.area.get_area() +
                           unicache.missb->local_result.area);
    area.set_area(area.get_area() + unicache.missb->local_result.area);
    // fill buffer
    tag = XML->sys.physical_address_width + EXTRA_TAG_BITS;
    data = unicache.caches->l_ip.line_sz;
    interface_ip.specific_tag = 1;
    interface_ip.tag_w = tag;
    interface_ip.line_sz = data;  // int(pow(2.0,ceil(log2(data))));
    interface_ip.cache_sz = data * cachep.fu_size;
    interface_ip.assoc = 0;
    interface_ip.nbanks = 1;
    interface_ip.out_w = interface_ip.line_sz * 8 / 2;
    interface_ip.access_mode = 0;
    interface_ip.throughput = cachep.throughput;
    interface_ip.latency = cachep.latency;
    interface_ip.obj_func_dyn_energy = 0;
    interface_ip.obj_func_dyn_power = 0;
    interface_ip.obj_func_leak_power = 0;
    interface_ip.obj_func_cycle_t = 1;
    interface_ip.num_rw_ports = 1;
    interface_ip.num_rd_ports = 0;
    interface_ip.num_wr_ports = 0;
    interface_ip.num_se_rd_ports = 0;
    // [한국어] Fill Buffer: 하위 메모리에서 읽어온 라인을 임시 저장
    unicache.ifb = new ArrayST(&interface_ip, cachep.name + "FillB", device_t,
                               true, core_t);
    unicache.area.set_area(unicache.area.get_area() +
                           unicache.ifb->local_result.area);
    area.set_area(area.get_area() + unicache.ifb->local_result.area);
    // prefetch buffer
    tag = XML->sys.physical_address_width +
          EXTRA_TAG_BITS;  // check with previous entries to decide wthether to
                           // merge.
    data = unicache.caches->l_ip
               .line_sz;  // separate queue to prevent from cache polution.
    interface_ip.specific_tag = 1;
    interface_ip.tag_w = tag;
    interface_ip.line_sz = data;  // int(pow(2.0,ceil(log2(data))));
    interface_ip.cache_sz = cachep.prefetchb_size * interface_ip.line_sz;
    interface_ip.assoc = 0;
    interface_ip.nbanks = 1;
    interface_ip.out_w = interface_ip.line_sz * 8 / 2;
    interface_ip.access_mode = 0;
    interface_ip.throughput = cachep.throughput;
    interface_ip.latency = cachep.latency;
    interface_ip.obj_func_dyn_energy = 0;
    interface_ip.obj_func_dyn_power = 0;
    interface_ip.obj_func_leak_power = 0;
    interface_ip.obj_func_cycle_t = 1;
    interface_ip.num_rw_ports = 1;
    interface_ip.num_rd_ports = 0;
    interface_ip.num_wr_ports = 0;
    interface_ip.num_se_rd_ports = 0;
    // [한국어] Prefetch Buffer: 미리 가져온 캐시 라인 큐
    unicache.prefetchb = new ArrayST(&interface_ip, cachep.name + "PrefetchB",
                                     device_t, true, core_t);
    unicache.area.set_area(unicache.area.get_area() +
                           unicache.prefetchb->local_result.area);
    area.set_area(area.get_area() + unicache.prefetchb->local_result.area);
    // WBB
    tag = XML->sys.physical_address_width + EXTRA_TAG_BITS;
    data = unicache.caches->l_ip.line_sz;
    interface_ip.specific_tag = 1;
    interface_ip.tag_w = tag;
    interface_ip.line_sz = data;
    interface_ip.cache_sz = cachep.wbb_size * interface_ip.line_sz;
    interface_ip.assoc = 0;
    interface_ip.nbanks = 1;
    interface_ip.out_w = interface_ip.line_sz * 8 / 2;
    interface_ip.access_mode = 0;
    interface_ip.throughput = cachep.throughput;
    interface_ip.latency = cachep.latency;
    interface_ip.obj_func_dyn_energy = 0;
    interface_ip.obj_func_dyn_power = 0;
    interface_ip.obj_func_leak_power = 0;
    interface_ip.obj_func_cycle_t = 1;
    interface_ip.num_rw_ports = 1;
    interface_ip.num_rd_ports = 0;
    interface_ip.num_wr_ports = 0;
    interface_ip.num_se_rd_ports = 0;
    // [한국어] Write Back Buffer: 더티 라인을 하위 계층에 쓰기 전 임시 보관
    unicache.wbb =
        new ArrayST(&interface_ip, cachep.name + "WBB", device_t, true, core_t);
    unicache.area.set_area(unicache.area.get_area() +
                           unicache.wbb->local_result.area);
    area.set_area(area.get_area() + unicache.wbb->local_result.area);
  }
  //  //pipeline
  //  interface_ip.pipeline_stages =
  //  int(ceil(llCache.caches.local_result.access_time/llCache.caches.local_result.cycle_time));
  //  interface_ip.per_stage_vector = llCache.caches.l_ip.out_w +
  //  llCache.caches.l_ip.tag_w ; pipeLogicCache.init_pipeline(is_default,
  //  &interface_ip); pipeLogicCache.compute_pipeline();

  /*
  if (!((XML->sys.number_of_dir_levels==1 && XML->sys.first_level_dir ==1)
                  ||(XML->sys.number_of_dir_levels==1 &&
  XML->sys.first_level_dir ==2)))//not single level IC and DIC
  {
  //directory Now assuming one directory per bank, TODO:should change it later
  size                             = XML->sys.L2directory.L2Dir_config[0];
  line                             = XML->sys.L2directory.L2Dir_config[1];
  assoc                            = XML->sys.L2directory.L2Dir_config[2];
  banks                            = XML->sys.L2directory.L2Dir_config[3];
  tag							   =
  debug?51:XML->sys.physical_address_width + EXTRA_TAG_BITS;//TODO: a little bit
  over estimate interface_ip.specific_tag        = 0; interface_ip.tag_w = tag;
  interface_ip.cache_sz            = XML->sys.L2directory.L2Dir_config[0];
  interface_ip.line_sz             = XML->sys.L2directory.L2Dir_config[1];
  interface_ip.assoc               = XML->sys.L2directory.L2Dir_config[2];
  interface_ip.nbanks              = XML->sys.L2directory.L2Dir_config[3];
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         =
  0;//debug?0:XML->sys.core[ithCore].icache.icache_config[5];
  interface_ip.throughput          =
  XML->sys.L2directory.L2Dir_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2directory.L2Dir_config[5]/clockRate; interface_ip.is_cache
  = true; interface_ip.obj_func_dyn_energy = 0; interface_ip.obj_func_dyn_power
  = 0; interface_ip.obj_func_leak_power = 0; interface_ip.obj_func_cycle_t    =
  1; interface_ip.num_rw_ports    = 1;//lower level cache usually has one port.
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;

  strcpy(directory.caches.name,"L2 Directory");
  directory.caches.init_cache(&interface_ip);
  directory.caches.optimize_array();
  directory.area += directory.caches.local_result.area;
  //output_data_csv(directory.caches.local_result);
  ///cout<<"area="<<area<<endl;

  //miss buffer Each MSHR contains enough state to handle one or more accesses
  of any type to a single memory line.
  //Due to the generality of the MSHR mechanism, the amount of state involved is
  non-trivial,
  //including the address, pointers to the cache entry and destination register,
  written data, and various other pieces of state. tag
  = XML->sys.physical_address_width + EXTRA_TAG_BITS;
  data							   =
  (XML->sys.physical_address_width) + int(ceil(log2(size/line))) +
  directory.caches.l_ip.line_sz; interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.line_sz             =
  int(ceil(data/8.0));//int(ceil(pow(2.0,ceil(log2(data)))/8.0));
  interface_ip.cache_sz            =
  XML->sys.L2[ithCache].buffer_sizes[0]*interface_ip.line_sz; interface_ip.assoc
  = 0; interface_ip.nbanks              = 1; interface_ip.out_w               =
  interface_ip.line_sz*8; interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate;//means cycle time
  interface_ip.latency             =
  XML->sys.L2[ithCache].L2_config[5]/clockRate;//means access time
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory.missb.name,"directoryMissB");
  directory.missb.init_cache(&interface_ip);
  directory.missb.optimize_array();
  directory.area += directory.missb.local_result.area;
  //output_data_csv(directory.missb.local_result);
  ///cout<<"area="<<area<<endl;

  //fill buffer
  tag							   =
  XML->sys.physical_address_width + EXTRA_TAG_BITS; data
  = directory.caches.l_ip.line_sz; interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.line_sz             = data;//int(pow(2.0,ceil(log2(data))));
  interface_ip.cache_sz            = data*XML->sys.L2[ithCache].buffer_sizes[1];
  interface_ip.assoc               = 0;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2[ithCache].L2_config[5]/clockRate; interface_ip.obj_func_dyn_energy
  = 0; interface_ip.obj_func_dyn_power  = 0; interface_ip.obj_func_leak_power =
  0; interface_ip.obj_func_cycle_t    = 1; interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory.ifb.name,"directoryFillB");
  directory.ifb.init_cache(&interface_ip);
  directory.ifb.optimize_array();
  directory.area += directory.ifb.local_result.area;
  //output_data_csv(directory.ifb.local_result);
  ///cout<<"area="<<area<<endl;

  //prefetch buffer
  tag							   =
  XML->sys.physical_address_width + EXTRA_TAG_BITS;//check with previous entries
  to decide wthether to merge.
  data							   =
  directory.caches.l_ip.line_sz;//separate queue to prevent from cache polution.
  interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.line_sz             = data;//int(pow(2.0,ceil(log2(data))));
  interface_ip.cache_sz            =
  XML->sys.L2[ithCache].buffer_sizes[2]*interface_ip.line_sz; interface_ip.assoc
  = 0; interface_ip.nbanks              = 1; interface_ip.out_w               =
  interface_ip.line_sz*8; interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2[ithCache].L2_config[5]/clockRate; interface_ip.obj_func_dyn_energy
  = 0; interface_ip.obj_func_dyn_power  = 0; interface_ip.obj_func_leak_power =
  0; interface_ip.obj_func_cycle_t    = 1; interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory.prefetchb.name,"directoryPrefetchB");
  directory.prefetchb.init_cache(&interface_ip);
  directory.prefetchb.optimize_array();
  directory.area += directory.prefetchb.local_result.area;
  //output_data_csv(directory.prefetchb.local_result);
  ///cout<<"area="<<area<<endl;

  //WBB
  tag							   =
  XML->sys.physical_address_width + EXTRA_TAG_BITS; data
  = directory.caches.l_ip.line_sz; interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.line_sz             = data;
  interface_ip.cache_sz            =
  XML->sys.L2[ithCache].buffer_sizes[3]*interface_ip.line_sz; interface_ip.assoc
  = 0; interface_ip.nbanks              = 1; interface_ip.out_w               =
  interface_ip.line_sz*8; interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.obj_func_dyn_energy
  = 0; interface_ip.obj_func_dyn_power  = 0; interface_ip.obj_func_leak_power =
  0; interface_ip.obj_func_cycle_t    = 1; interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory.wbb.name,"directoryWBB");
  directory.wbb.init_cache(&interface_ip);
  directory.wbb.optimize_array();
  directory.area += directory.wbb.local_result.area;
  }

  if (XML->sys.number_of_dir_levels ==2 && XML->sys.first_level_dir==0)
  {
  //first level directory
  size                             =
  XML->sys.L2directory.L2Dir_config[0]*XML->sys.domain_size/128; line =
  int(ceil(XML->sys.domain_size/8.0)); assoc                            =
  XML->sys.L2directory.L2Dir_config[2]; banks                            =
  XML->sys.L2directory.L2Dir_config[3]; tag
  = debug?51:XML->sys.physical_address_width + EXTRA_TAG_BITS;//TODO: a little
  bit over estimate interface_ip.specific_tag        = 1; interface_ip.tag_w =
  tag; interface_ip.cache_sz            = XML->sys.L2directory.L2Dir_config[0];
  interface_ip.line_sz             = XML->sys.L2directory.L2Dir_config[1];
  interface_ip.assoc               = XML->sys.L2directory.L2Dir_config[2];
  interface_ip.nbanks              = XML->sys.L2directory.L2Dir_config[3];
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         =
  0;//debug?0:XML->sys.core[ithCore].icache.icache_config[5];
  interface_ip.throughput          =
  XML->sys.L2directory.L2Dir_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2directory.L2Dir_config[5]/clockRate; interface_ip.is_cache
  = true; interface_ip.obj_func_dyn_energy = 0; interface_ip.obj_func_dyn_power
  = 0; interface_ip.obj_func_leak_power = 0; interface_ip.obj_func_cycle_t    =
  1; interface_ip.num_rw_ports    = 1;//lower level cache usually has one port.
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;

  strcpy(directory1.caches.name,"first level Directory");
  directory1.caches.init_cache(&interface_ip);
  directory1.caches.optimize_array();
  directory1.area += directory1.caches.local_result.area;
  //output_data_csv(directory.caches.local_result);
  ///cout<<"area="<<area<<endl;

  //miss buffer Each MSHR contains enough state to handle one or more accesses
  of any type to a single memory line.
  //Due to the generality of the MSHR mechanism, the amount of state involved is
  non-trivial,
  //including the address, pointers to the cache entry and destination register,
  written data, and various other pieces of state. tag
  = XML->sys.physical_address_width + EXTRA_TAG_BITS;
  data							   =
  (XML->sys.physical_address_width) + int(ceil(log2(size/line))) +
  directory1.caches.l_ip.line_sz; interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.line_sz             =
  int(ceil(data/8.0));//int(ceil(pow(2.0,ceil(log2(data)))/8.0));
  interface_ip.cache_sz            =
  XML->sys.L2[ithCache].buffer_sizes[0]*interface_ip.line_sz; interface_ip.assoc
  = 0; interface_ip.nbanks              = 1; interface_ip.out_w               =
  interface_ip.line_sz*8; interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate;//means cycle time
  interface_ip.latency             =
  XML->sys.L2[ithCache].L2_config[5]/clockRate;//means access time
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory1.missb.name,"directory1MissB");
  directory1.missb.init_cache(&interface_ip);
  directory1.missb.optimize_array();
  directory1.area += directory1.missb.local_result.area;
  //output_data_csv(directory.missb.local_result);
  ///cout<<"area="<<area<<endl;

  //fill buffer
  tag							   =
  XML->sys.physical_address_width + EXTRA_TAG_BITS; data
  = directory1.caches.l_ip.line_sz; interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.line_sz             = data;//int(pow(2.0,ceil(log2(data))));
  interface_ip.cache_sz            = data*XML->sys.L2[ithCache].buffer_sizes[1];
  interface_ip.assoc               = 0;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2[ithCache].L2_config[5]/clockRate; interface_ip.obj_func_dyn_energy
  = 0; interface_ip.obj_func_dyn_power  = 0; interface_ip.obj_func_leak_power =
  0; interface_ip.obj_func_cycle_t    = 1; interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory1.ifb.name,"directory1FillB");
  directory1.ifb.init_cache(&interface_ip);
  directory1.ifb.optimize_array();
  directory1.area += directory1.ifb.local_result.area;
  //output_data_csv(directory.ifb.local_result);
  ///cout<<"area="<<area<<endl;

  //prefetch buffer
  tag							   =
  XML->sys.physical_address_width + EXTRA_TAG_BITS;//check with previous entries
  to decide wthether to merge.
  data							   =
  directory1.caches.l_ip.line_sz;//separate queue to prevent from cache
  polution. interface_ip.specific_tag        = 1; interface_ip.tag_w = tag;
  interface_ip.line_sz             = data;//int(pow(2.0,ceil(log2(data))));
  interface_ip.cache_sz            =
  XML->sys.L2[ithCache].buffer_sizes[2]*interface_ip.line_sz; interface_ip.assoc
  = 0; interface_ip.nbanks              = 1; interface_ip.out_w               =
  interface_ip.line_sz*8; interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2[ithCache].L2_config[5]/clockRate; interface_ip.obj_func_dyn_energy
  = 0; interface_ip.obj_func_dyn_power  = 0; interface_ip.obj_func_leak_power =
  0; interface_ip.obj_func_cycle_t    = 1; interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory1.prefetchb.name,"directory1PrefetchB");
  directory1.prefetchb.init_cache(&interface_ip);
  directory1.prefetchb.optimize_array();
  directory1.area += directory1.prefetchb.local_result.area;
  //output_data_csv(directory.prefetchb.local_result);
  ///cout<<"area="<<area<<endl;

  //WBB
  tag							   =
  XML->sys.physical_address_width + EXTRA_TAG_BITS; data
  = directory1.caches.l_ip.line_sz; interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.line_sz             = data;
  interface_ip.cache_sz            =
  XML->sys.L2[ithCache].buffer_sizes[3]*interface_ip.line_sz; interface_ip.assoc
  = 0; interface_ip.nbanks              = 1; interface_ip.out_w               =
  interface_ip.line_sz*8; interface_ip.access_mode         = 0;
  interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2[ithCache].L2_config[5]/clockRate; interface_ip.obj_func_dyn_energy
  = 0; interface_ip.obj_func_dyn_power  = 0; interface_ip.obj_func_leak_power =
  0; interface_ip.obj_func_cycle_t    = 1; interface_ip.num_rw_ports    = 1;
  interface_ip.num_rd_ports    = 0;
  interface_ip.num_wr_ports    = 0;
  interface_ip.num_se_rd_ports = 0;
  strcpy(directory1.wbb.name,"directoryWBB");
  directory1.wbb.init_cache(&interface_ip);
  directory1.wbb.optimize_array();
  directory1.area += directory1.wbb.local_result.area;
  }

  if (XML->sys.first_level_dir==1)//IC
  {
          tag							   =
  XML->sys.physical_address_width + EXTRA_TAG_BITS; data
  = int(ceil(XML->sys.domain_size/8.0)); interface_ip.specific_tag        = 1;
          interface_ip.tag_w               = tag;
          interface_ip.line_sz             = data;
          interface_ip.cache_sz            =
  XML->sys.domain_size*data*XML->sys.L2[ithCache].L2_config[0]/XML->sys.L2[ithCache].L2_config[1];
          interface_ip.assoc               = 0;
          interface_ip.nbanks              = 1024;
          interface_ip.out_w               = interface_ip.line_sz*8;
          interface_ip.access_mode         = 0;
          interface_ip.throughput          =
  XML->sys.L2[ithCache].L2_config[4]/clockRate; interface_ip.latency =
  XML->sys.L2[ithCache].L2_config[5]/clockRate; interface_ip.obj_func_dyn_energy
  = 0; interface_ip.obj_func_dyn_power  = 0; interface_ip.obj_func_leak_power =
  0; interface_ip.obj_func_cycle_t    = 1; interface_ip.num_rw_ports    = 1;
          interface_ip.num_rd_ports    = 0;
          interface_ip.num_wr_ports    = 0;
          interface_ip.num_se_rd_ports = 0;
          strcpy(inv_dir.caches.name,"inv_dir");
          inv_dir.caches.init_cache(&interface_ip);
          inv_dir.caches.optimize_array();
          inv_dir.area = inv_dir.caches.local_result.area;

  }
*/
  //  //pipeline
  //  interface_ip.pipeline_stages =
  //  int(ceil(directory.caches.local_result.access_time/directory.caches.local_result.cycle_time));
  //  interface_ip.per_stage_vector = directory.caches.l_ip.out_w +
  //  directory.caches.l_ip.tag_w ; pipeLogicDirectory.init_pipeline(is_default,
  //  &interface_ip); pipeLogicDirectory.compute_pipeline();
  //
  //  //clock power
  //  clockNetwork.init_wire_external(is_default, &interface_ip);
  //  clockNetwork.clk_area           =area*1.1;//10% of placement overhead.
  //  rule of thumb clockNetwork.end_wiring_level   =5;//toplevel metal
  //  clockNetwork.start_wiring_level =5;//toplevel metal
  //  clockNetwork.num_regs           = pipeLogicCache.tot_stage_vector +
  //  pipeLogicDirectory.tot_stage_vector; clockNetwork.optimize_wire();
}

/*
 * [한국어]
 * SharedCache::computeEnergy - 공유 캐시 동적 에너지 및 누설 전력 계산
 *
 * @is_tdp: true이면 TDP(Peak) 모드, false이면 런타임 통계 기반 모드.
 *
 * 동작 순서:
 *   1. TDP 모드: cachep.duty_cycle과 포트 수로부터 read/write 접근 횟수를 추정.
 *      Directory(ST)인 경우 search port 기반 접근만 계산.
 *   2. 런타임 모드: XML->sys.L2[], sys.L3[], sys.L1Directory[], sys.L2Directory[]의
 *      실제 read_accesses/write_accesses/misses/hits를 stats_t에 복사.
 *   3. homenode_data_access 비율로 SBT(디렉토리 병합) 캐시의 홈 노드 부하를 조정.
 *   4. 주 캐시 + missb/ifb/prefetchb/wbb의 동적 에너지를 local_result.power에
 *      stats_t를 곱해 누적(unicache.power_t).
 *   5. 누설 전력(local_result.power * pppm_lkg)을 더해 power(TDP) 또는
 *      rt_power(런타임)에 반영.
 */
void SharedCache::computeEnergy(bool is_tdp) {
  // [한국어] SBT 디렉토리 병합 캐시: 데이터 접근의 90%가 로컬, 10%가 홈 노드
  double homenode_data_access = (cachep.dir_ty == SBT) ? 0.9 : 1.0;
  // [한국어] TDP 모드: Peak 부하 가정(읽기 67%, 쓰기 33%)으로 접근 수 추정
  if (is_tdp) {
    if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||
          (cachep.dir_ty == ST && cacheL == L2Directory))) {
      // init stats for Peak
      // [한국어] TDP 읽기 접근 = 0.67 * rw_port * duty_cycle * homenode_data_factor
      unicache.caches->stats_t.readAc.access =
          .67 * unicache.caches->l_ip.num_rw_ports * cachep.duty_cycle *
          homenode_data_access;
      unicache.caches->stats_t.readAc.miss = 0;
      unicache.caches->stats_t.readAc.hit =
          unicache.caches->stats_t.readAc.access -
          unicache.caches->stats_t.readAc.miss;
      // [한국어] TDP 쓰기 접근 = 0.33 * rw_port * duty_cycle * homenode_data_factor
      unicache.caches->stats_t.writeAc.access =
          .33 * unicache.caches->l_ip.num_rw_ports * cachep.duty_cycle *
          homenode_data_access;
      unicache.caches->stats_t.writeAc.miss = 0;
      unicache.caches->stats_t.writeAc.hit =
          unicache.caches->stats_t.writeAc.access -
          unicache.caches->stats_t.writeAc.miss;
      // [한국어] TDP 통계 복사본 저장
      unicache.caches->tdp_stats = unicache.caches->stats_t;

      // [한국어] SBT 캐시: 나머지 10% 디렉토리 홈 노드 접근에 대한 TDP 통계
    if (cachep.dir_ty == SBT) {
        homenode_stats_t.readAc.access =
            .67 * unicache.caches->l_ip.num_rw_ports * cachep.dir_duty_cycle *
            (1 - homenode_data_access);
        homenode_stats_t.readAc.miss = 0;
        homenode_stats_t.readAc.hit =
            homenode_stats_t.readAc.access - homenode_stats_t.readAc.miss;
        homenode_stats_t.writeAc.access =
            .67 * unicache.caches->l_ip.num_rw_ports * cachep.dir_duty_cycle *
            (1 - homenode_data_access);
        homenode_stats_t.writeAc.miss = 0;
        homenode_stats_t.writeAc.hit =
            homenode_stats_t.writeAc.access - homenode_stats_t.writeAc.miss;
        homenode_tdp_stats = homenode_stats_t;
      }

      // [한국어] Miss Buffer: search port당 한 번의 CAM 검색과 한 번의 RAM 쓰기
      unicache.missb->stats_t.readAc.access =
          unicache.missb->l_ip.num_search_ports;
      unicache.missb->stats_t.writeAc.access =
          unicache.missb->l_ip.num_search_ports;
      unicache.missb->tdp_stats = unicache.missb->stats_t;

      // [한국어] Fill Buffer: search port 기반 읽기/쓰기 접근
      unicache.ifb->stats_t.readAc.access = unicache.ifb->l_ip.num_search_ports;
      unicache.ifb->stats_t.writeAc.access =
          unicache.ifb->l_ip.num_search_ports;
      unicache.ifb->tdp_stats = unicache.ifb->stats_t;

      // [한국어] Prefetch Buffer: search port 기반 접근
      unicache.prefetchb->stats_t.readAc.access =
          unicache.prefetchb->l_ip.num_search_ports;
      unicache.prefetchb->stats_t.writeAc.access =
          unicache.ifb->l_ip.num_search_ports;
      unicache.prefetchb->tdp_stats = unicache.prefetchb->stats_t;

      // [한국어] Write Back Buffer: search port 기반 접근
      unicache.wbb->stats_t.readAc.access = unicache.wbb->l_ip.num_search_ports;
      unicache.wbb->stats_t.writeAc.access =
          unicache.wbb->l_ip.num_search_ports;
      unicache.wbb->tdp_stats = unicache.wbb->stats_t;
    } else {
      // [한국어] Directory(ST) TDP: search port와 duty cycle로 접근 추정
      unicache.caches->stats_t.readAc.access =
          unicache.caches->l_ip.num_search_ports * cachep.duty_cycle;
      unicache.caches->stats_t.readAc.miss = 0;
      unicache.caches->stats_t.readAc.hit =
          unicache.caches->stats_t.readAc.access -
          unicache.caches->stats_t.readAc.miss;
      unicache.caches->stats_t.writeAc.access = 0;
      unicache.caches->stats_t.writeAc.miss = 0;
      unicache.caches->stats_t.writeAc.hit =
          unicache.caches->stats_t.writeAc.access -
          unicache.caches->stats_t.writeAc.miss;
      unicache.caches->tdp_stats = unicache.caches->stats_t;
    }

  } else {
  // [한국어] 런타임 모드: GPGPU-Sim/AccelWattch가 채운 실제 접근 통계 사용
    // init stats for runtime power (RTP)
    if (cacheL == L2) {
      // Copy stats from l1 to L1[0]
      // [한국어] GPU 통합 L2 통계를 XML->sys.L2[0]로 복사(Processor가 읽음)
      XML->sys.L2[ithCache].total_accesses = XML->sys.l2.total_accesses;
      XML->sys.L2[ithCache].read_accesses = XML->sys.l2.read_accesses;
      XML->sys.L2[ithCache].write_accesses = XML->sys.l2.write_accesses;
      XML->sys.L2[ithCache].read_hits = XML->sys.l2.read_hits;
      XML->sys.L2[ithCache].read_misses = XML->sys.l2.read_misses;
      XML->sys.L2[ithCache].write_hits = XML->sys.l2.write_hits;
      XML->sys.L2[ithCache].write_misses = XML->sys.l2.write_misses;

      unicache.caches->stats_t.readAc.access =
          XML->sys.L2[ithCache].read_accesses;
      unicache.caches->stats_t.readAc.miss = XML->sys.L2[ithCache].read_misses;
      unicache.caches->stats_t.readAc.hit =
          unicache.caches->stats_t.readAc.access -
          unicache.caches->stats_t.readAc.miss;
      unicache.caches->stats_t.writeAc.access =
          XML->sys.L2[ithCache].write_accesses;
      unicache.caches->stats_t.writeAc.miss =
          XML->sys.L2[ithCache].write_misses;
      unicache.caches->stats_t.writeAc.hit =
          unicache.caches->stats_t.writeAc.access -
          unicache.caches->stats_t.writeAc.miss;
      unicache.caches->rtp_stats = unicache.caches->stats_t;

      // [한국어] SBT 홈 노드 write miss도 버퍼 접근에 추가
      if (cachep.dir_ty == SBT) {
        homenode_rtp_stats.readAc.access =
            XML->sys.L2[ithCache].homenode_read_accesses;
        homenode_rtp_stats.readAc.miss =
            XML->sys.L2[ithCache].homenode_read_misses;
        homenode_rtp_stats.readAc.hit =
            homenode_rtp_stats.readAc.access - homenode_rtp_stats.readAc.miss;
        homenode_rtp_stats.writeAc.access =
            XML->sys.L2[ithCache].homenode_write_accesses;
        homenode_rtp_stats.writeAc.miss =
            XML->sys.L2[ithCache].homenode_write_misses;
        homenode_rtp_stats.writeAc.hit =
            homenode_rtp_stats.writeAc.access - homenode_rtp_stats.writeAc.miss;
      }
    // [한국어] CPU/GPU L3 캐시: sys.L3[] 통계를 직접 사용
  } else if (cacheL == L3) {
      unicache.caches->stats_t.readAc.access =
          XML->sys.L3[ithCache].read_accesses;
      unicache.caches->stats_t.readAc.miss = XML->sys.L3[ithCache].read_misses;
      unicache.caches->stats_t.readAc.hit =
          unicache.caches->stats_t.readAc.access -
          unicache.caches->stats_t.readAc.miss;
      unicache.caches->stats_t.writeAc.access =
          XML->sys.L3[ithCache].write_accesses;
      unicache.caches->stats_t.writeAc.miss =
          XML->sys.L3[ithCache].write_misses;
      unicache.caches->stats_t.writeAc.hit =
          unicache.caches->stats_t.writeAc.access -
          unicache.caches->stats_t.writeAc.miss;
      unicache.caches->rtp_stats = unicache.caches->stats_t;

      if (cachep.dir_ty == SBT) {
        homenode_rtp_stats.readAc.access =
            XML->sys.L3[ithCache].homenode_read_accesses;
        homenode_rtp_stats.readAc.miss =
            XML->sys.L3[ithCache].homenode_read_misses;
        homenode_rtp_stats.readAc.hit =
            homenode_rtp_stats.readAc.access - homenode_rtp_stats.readAc.miss;
        homenode_rtp_stats.writeAc.access =
            XML->sys.L3[ithCache].homenode_write_accesses;
        homenode_rtp_stats.writeAc.miss =
            XML->sys.L3[ithCache].homenode_write_misses;
        homenode_rtp_stats.writeAc.hit =
            homenode_rtp_stats.writeAc.access - homenode_rtp_stats.writeAc.miss;
      }
    // [한국어] L1 Directory: sys.L1Directory[] 통계 사용
  } else if (cacheL == L1Directory) {
      unicache.caches->stats_t.readAc.access =
          XML->sys.L1Directory[ithCache].read_accesses;
      unicache.caches->stats_t.readAc.miss =
          XML->sys.L1Directory[ithCache].read_misses;
      unicache.caches->stats_t.readAc.hit =
          unicache.caches->stats_t.readAc.access -
          unicache.caches->stats_t.readAc.miss;
      unicache.caches->stats_t.writeAc.access =
          XML->sys.L1Directory[ithCache].write_accesses;
      unicache.caches->stats_t.writeAc.miss =
          XML->sys.L1Directory[ithCache].write_misses;
      unicache.caches->stats_t.writeAc.hit =
          unicache.caches->stats_t.writeAc.access -
          unicache.caches->stats_t.writeAc.miss;
      unicache.caches->rtp_stats = unicache.caches->stats_t;
    // [한국어] L2 Directory: sys.L2Directory[] 통계 사용
  } else if (cacheL == L2Directory) {  // cout<<"L2 directory"<<endl;
      // Copy stats from l1 to L1[0]
      // XML->sys.L2[ithCache].total_accesses=XML->sys.l2.total_accesses;
      // XML->sys.L2[ithCache].read_accesses=XML->sys.l2.read_accesses;
      // XML->sys.L2[ithCache].write_accesses=XML->sys.l2.write_accesses;
      // XML->sys.L2[ithCache].read_hits=XML->sys.l2.read_hits;
      // XML->sys.L2[ithCache].read_misses=XML->sys.l2.read_misses;
      // XML->sys.L2[ithCache].write_hits=XML->sys.l2.write_hits;
      // XML->sys.L2[ithCache].write_misses=XML->sys.l2.write_misses;
      unicache.caches->stats_t.readAc.access =
          XML->sys.L2Directory[ithCache].read_accesses;
      unicache.caches->stats_t.readAc.miss =
          XML->sys.L2Directory[ithCache].read_misses;
      unicache.caches->stats_t.readAc.hit =
          unicache.caches->stats_t.readAc.access -
          unicache.caches->stats_t.readAc.miss;
      unicache.caches->stats_t.writeAc.access =
          XML->sys.L2Directory[ithCache].write_accesses;
      unicache.caches->stats_t.writeAc.miss =
          XML->sys.L2Directory[ithCache].write_misses;
      unicache.caches->stats_t.writeAc.hit =
          unicache.caches->stats_t.writeAc.access -
          unicache.caches->stats_t.writeAc.miss;
      unicache.caches->rtp_stats = unicache.caches->stats_t;
    }
    // [한국어] 일반 캐시/Directory가 아닐 때만 missb/ifb/prefetchb/wbb 에너지 포함
    if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||
          (cachep.dir_ty == ST &&
           cacheL ==
               L2Directory))) {  // Assuming write back and write-allocate cache

      // [한국어] W-back/W-allocate 가정: write miss 수만큼 missb/ifb/wbb 접근
      unicache.missb->stats_t.readAc.access =
          unicache.caches->stats_t.writeAc.miss;
      unicache.missb->stats_t.writeAc.access =
          unicache.caches->stats_t.writeAc.miss;
      unicache.missb->rtp_stats = unicache.missb->stats_t;

      unicache.ifb->stats_t.readAc.access =
          unicache.caches->stats_t.writeAc.miss;
      unicache.ifb->stats_t.writeAc.access =
          unicache.caches->stats_t.writeAc.miss;
      unicache.ifb->rtp_stats = unicache.ifb->stats_t;

      unicache.prefetchb->stats_t.readAc.access =
          unicache.caches->stats_t.writeAc.miss;
      unicache.prefetchb->stats_t.writeAc.access =
          unicache.caches->stats_t.writeAc.miss;
      unicache.prefetchb->rtp_stats = unicache.prefetchb->stats_t;

      unicache.wbb->stats_t.readAc.access =
          unicache.caches->stats_t.writeAc.miss;
      unicache.wbb->stats_t.writeAc.access =
          unicache.caches->stats_t.writeAc.miss;
      if (cachep.dir_ty == SBT) {
        unicache.missb->stats_t.readAc.access +=
            homenode_rtp_stats.writeAc.miss;
        unicache.missb->stats_t.writeAc.access +=
            homenode_rtp_stats.writeAc.miss;
        unicache.missb->rtp_stats = unicache.missb->stats_t;

        unicache.missb->stats_t.readAc.access +=
            homenode_rtp_stats.writeAc.miss;
        unicache.missb->stats_t.writeAc.access +=
            homenode_rtp_stats.writeAc.miss;
        unicache.missb->rtp_stats = unicache.missb->stats_t;

        unicache.ifb->stats_t.readAc.access += homenode_rtp_stats.writeAc.miss;
        unicache.ifb->stats_t.writeAc.access += homenode_rtp_stats.writeAc.miss;
        unicache.ifb->rtp_stats = unicache.ifb->stats_t;

        unicache.prefetchb->stats_t.readAc.access +=
            homenode_rtp_stats.writeAc.miss;
        unicache.prefetchb->stats_t.writeAc.access +=
            homenode_rtp_stats.writeAc.miss;
        unicache.prefetchb->rtp_stats = unicache.prefetchb->stats_t;

        unicache.wbb->stats_t.readAc.access += homenode_rtp_stats.writeAc.miss;
        unicache.wbb->stats_t.writeAc.access += homenode_rtp_stats.writeAc.miss;
      }
      unicache.wbb->rtp_stats = unicache.wbb->stats_t;
    }
  }

  // [한국어] 동적 에너지 누적값 초기화
  unicache.power_t.reset();
  // [한국어] 런타임 전력 누적값 초기화
  unicache.rt_power.reset();
  if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||
        (cachep.dir_ty == ST && cacheL == L2Directory))) {
    // [한국어] 주 캐시 동적 에너지: read hit + read miss(tag) + write miss(tag) + write access
    unicache.power_t.readOp.dynamic +=
        (unicache.caches->stats_t.readAc.hit *
             unicache.caches->local_result.power.readOp.dynamic +
         unicache.caches->stats_t.readAc.miss *
             unicache.caches->local_result.tag_array2->power.readOp.dynamic +
         unicache.caches->stats_t.writeAc.miss *
             unicache.caches->local_result.tag_array2->power.writeOp.dynamic +
         unicache.caches->stats_t.writeAc.access *
             unicache.caches->local_result.power.writeOp
                 .dynamic);  // write miss will also generate a write later

    // [한국어] SBT 홈 노드 접근에 대한 추가 동적 에너지(dir_overhead 반영)
    if (cachep.dir_ty == SBT) {
      unicache.power_t.readOp.dynamic +=
          homenode_stats_t.readAc.hit *
              (unicache.caches->local_result.data_array2->power.readOp.dynamic *
                   dir_overhead +
               unicache.caches->local_result.tag_array2->power.readOp.dynamic) +
          homenode_stats_t.readAc.miss *
              unicache.caches->local_result.tag_array2->power.readOp.dynamic +
          homenode_stats_t.writeAc.miss *
              unicache.caches->local_result.tag_array2->power.readOp.dynamic +
          homenode_stats_t.writeAc.hit *
              (unicache.caches->local_result.data_array2->power.writeOp
                       .dynamic *
                   dir_overhead +
               unicache.caches->local_result.tag_array2->power.readOp.dynamic +
               homenode_stats_t.writeAc.miss *
                   unicache.caches->local_result.power.writeOp
                       .dynamic);  // write miss on dynamic home node will
                                   // generate a replacement write on whole
                                   // cache block
    }

    // [한국어] Miss Buffer 동적 에너지: searchOp(전체 CAM 검색) + writeOp(상태 갱신)
    unicache.power_t.readOp.dynamic +=
        unicache.missb->stats_t.readAc.access *
            unicache.missb->local_result.power.searchOp.dynamic +
        unicache.missb->stats_t.writeAc.access *
            unicache.missb->local_result.power.writeOp
                .dynamic;  // each access to missb involves a CAM and a write
    // [한국어] Fill Buffer 동적 에너지
    unicache.power_t.readOp.dynamic +=
        unicache.ifb->stats_t.readAc.access *
            unicache.ifb->local_result.power.searchOp.dynamic +
        unicache.ifb->stats_t.writeAc.access *
            unicache.ifb->local_result.power.writeOp.dynamic;
    // [한국어] Prefetch Buffer 동적 에너지
    unicache.power_t.readOp.dynamic +=
        unicache.prefetchb->stats_t.readAc.access *
            unicache.prefetchb->local_result.power.searchOp.dynamic +
        unicache.prefetchb->stats_t.writeAc.access *
            unicache.prefetchb->local_result.power.writeOp.dynamic;
    // [한국어] Write Back Buffer 동적 에너지
    unicache.power_t.readOp.dynamic +=
        unicache.wbb->stats_t.readAc.access *
            unicache.wbb->local_result.power.searchOp.dynamic +
        unicache.wbb->stats_t.writeAc.access *
            unicache.wbb->local_result.power.writeOp.dynamic;
  } else {
  // [한국어] Directory(ST) 구조: searchOp + writeOp 기반 동적 에너지
    unicache.power_t.readOp.dynamic +=
        (unicache.caches->stats_t.readAc.access *
             unicache.caches->local_result.power.searchOp.dynamic +
         unicache.caches->stats_t.writeAc.access *
             unicache.caches->local_result.power.writeOp.dynamic);
  }

  // [한국어] TDP: 동적 에너지 + 주 캐시 누설 전력을 power에 합산
  if (is_tdp) {
    // [한국어] power_t(TDP 동적) + caches 누설(pppm_lkg 마스크 적용)
    unicache.power =
        unicache.power_t + (unicache.caches->local_result.power) * pppm_lkg;
    if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||
          (cachep.dir_ty == ST && cacheL == L2Directory))) {
      // [한국어] 보조 버퍼들의 누설 전력도 추가
      unicache.power =
          unicache.power + (unicache.missb->local_result.power +
                            unicache.ifb->local_result.power +
                            unicache.prefetchb->local_result.power +
                            unicache.wbb->local_result.power) *
                               pppm_lkg;
    }
    // [한국어] SharedCache의 TDP 전력을 Processor 전체 power에 누적
    power = power + unicache.power;
    //		cout<<"unicache.caches->local_result.power.readOp.dynamic"<<unicache.caches->local_result.power.readOp.dynamic<<endl;
    //		cout<<"unicache.caches->local_result.power.writeOp.dynamic"<<unicache.caches->local_result.power.writeOp.dynamic<<endl;
  } else {
  // [한국어] 런타임: 동적 에너지 + 주 캐시 누설 전력을 rt_power에 합산
    unicache.rt_power =
        unicache.power_t + (unicache.caches->local_result.power) * pppm_lkg;
    if (!((cachep.dir_ty == ST && cacheL == L1Directory) ||
          (cachep.dir_ty == ST && cacheL == L2Directory))) {
      // [한국어] 보조 버퍼들의 누설 전력도 추가
      unicache.rt_power =
          unicache.rt_power + (unicache.missb->local_result.power +
                               unicache.ifb->local_result.power +
                               unicache.prefetchb->local_result.power +
                               unicache.wbb->local_result.power) *
                                  pppm_lkg;
    }

    // [한국어] SharedCache의 런타임 전력을 Processor 전체 rt_power에 누적
    rt_power = rt_power + unicache.rt_power;
  }
}

/*
 * [한국어]
 * SharedCache::displayEnergy - 공유 캐시 전력/면적 결과 출력
 *
 * @indent: 출력 들여쓰기 칸 수.
 * @is_tdp: true이면 TDP 모드 결과 출력(현재 TDP 블록만 구현되어 있음).
 *
 * 출력 항목:
 *   - 캐시 이름(Private L2 여부에 따라 indent 차이)
 *   - Area (um^2 → mm^2)
 *   - Peak Dynamic (W)
 *   - Subthreshold Leakage (W, longer_channel_device 선택)
 *   - Gate Leakage (W)
 *   - Runtime Dynamic (W)
 */
void SharedCache::displayEnergy(uint32_t indent, bool is_tdp) {
  string indent_str(indent, ' ');
  string indent_str_next(indent + 2, ' ');
  // [한국어] 장채널 소자 사용 여부에 따라 누설 전력 출력 분기
  bool long_channel = XML->sys.longer_channel_device;

  if (is_tdp) {
    // [한국어] Private L2가 아니면 상위 Processor에서 이미 인덴트를 출력
    cout << (XML->sys.Private_L2 ? indent_str : "") << cachep.name << endl;
    cout << indent_str << "Area = " << area.get_area() * 1e-6 << " mm^2"
         << endl;
    cout << indent_str
    // [한국어] Peak 동적 전력 = 동적 에너지 × 클럭주파수
         << "Peak Dynamic = " << power.readOp.dynamic * cachep.clockRate << " W"
         << endl;
    cout << indent_str << "Subthreshold Leakage = "
         << (long_channel ? power.readOp.longer_channel_leakage
                          : power.readOp.leakage)
         << " W" << endl;
    // cout << indent_str << "Subthreshold Leakage = " <<
    // power.readOp.longer_channel_leakage <<" W" << endl;
    cout << indent_str << "Gate Leakage = " << power.readOp.gate_leakage << " W"
         << endl;
    // [한국어] 런타임 동적 전력 = 런타임 에너지 / 실행 시간
    cout << indent_str << "Runtime Dynamic = "
         << rt_power.readOp.dynamic / cachep.executionTime << " W" << endl;
    cout << endl;
  } else {
  }
}

// void SharedCache::computeMaxPower()
//{
//  //Compute maximum power and runtime power.
//  //When computing runtime power, McPAT gets or reasons out the statistics
//  based on XML input. maxPower		= 0.0;
//  //llCache,itlb
//  llCache.maxPower   = 0.0;
//  llCache.maxPower	+=
//  (llCache.caches.l_ip.num_rw_ports*(0.67*llCache.caches.local_result.power.readOp.dynamic+0.33*llCache.caches.local_result.power.writeOp.dynamic)
//                        +llCache.caches.l_ip.num_rd_ports*llCache.caches.local_result.power.readOp.dynamic+llCache.caches.l_ip.num_wr_ports*llCache.caches.local_result.power.writeOp.dynamic
//                        +llCache.caches.l_ip.num_se_rd_ports*llCache.caches.local_result.power.readOp.dynamic)*clockRate;
//  ///cout<<"llCache.maxPower=" <<llCache.maxPower<<endl;
//
//  llCache.maxPower	+=
//  llCache.missb.l_ip.num_search_ports*llCache.missb.local_result.power.searchOp.dynamic*clockRate;
//  ///cout<<"llCache.maxPower=" <<llCache.maxPower<<endl;
//
//  llCache.maxPower	+=
//  llCache.ifb.l_ip.num_search_ports*llCache.ifb.local_result.power.searchOp.dynamic*clockRate;
//  ///cout<<"llCache.maxPower=" <<llCache.maxPower<<endl;
//
//  llCache.maxPower	+=
//  llCache.prefetchb.l_ip.num_search_ports*llCache.prefetchb.local_result.power.searchOp.dynamic*clockRate;
//  ///cout<<"llCache.maxPower=" <<llCache.maxPower<<endl;
//
//  llCache.maxPower	+=
//  llCache.wbb.l_ip.num_search_ports*llCache.wbb.local_result.power.searchOp.dynamic*clockRate;
//  //llCache.maxPower *=  scktRatio; //TODO: this calculation should be
//  self-contained
//  ///cout<<"llCache.maxPower=" <<llCache.maxPower<<endl;
//
////  directory_power =
///(directory.caches.l_ip.num_rw_ports*(0.67*directory.caches.local_result.power.readOp.dynamic+0.33*directory.caches.local_result.power.writeOp.dynamic)
////
///+directory.caches.l_ip.num_rd_ports*directory.caches.local_result.power.readOp.dynamic+directory.caches.l_ip.num_wr_ports*directory.caches.local_result.power.writeOp.dynamic
////
///+directory.caches.l_ip.num_se_rd_ports*directory.caches.local_result.power.readOp.dynamic)*clockRate;
//
//  L2Tot.power.readOp.dynamic = llCache.maxPower;
//  L2Tot.power.readOp.leakage =
//  llCache.caches.local_result.power.readOp.leakage +
//                               llCache.missb.local_result.power.readOp.leakage
//                               + llCache.ifb.local_result.power.readOp.leakage
//                               +
//                               llCache.prefetchb.local_result.power.readOp.leakage
//                               +
//                               llCache.wbb.local_result.power.readOp.leakage;
//
//  L2Tot.area.set_area(llCache.area*1.1*1e-6);//placement and routing overhead
//
//  if (XML->sys.number_of_dir_levels==1)
//  {
//	  if (XML->sys.first_level_dir==0)
//	  {
//		  directory.maxPower   = 0.0;
//		  directory.maxPower	+=
//(directory.caches.l_ip.num_rw_ports*(0.67*directory.caches.local_result.power.readOp.dynamic+0.33*directory.caches.local_result.power.writeOp.dynamic)
//		                        +directory.caches.l_ip.num_rd_ports*directory.caches.local_result.power.readOp.dynamic+directory.caches.l_ip.num_wr_ports*directory.caches.local_result.power.writeOp.dynamic
//		                        +directory.caches.l_ip.num_se_rd_ports*directory.caches.local_result.power.readOp.dynamic)*clockRate;
//		  ///cout<<"directory.maxPower=" <<directory.maxPower<<endl;
//
//		  directory.maxPower	+=
// directory.missb.l_ip.num_search_ports*directory.missb.local_result.power.searchOp.dynamic*clockRate;
//		  ///cout<<"directory.maxPower=" <<directory.maxPower<<endl;
//
//		  directory.maxPower	+=
// directory.ifb.l_ip.num_search_ports*directory.ifb.local_result.power.searchOp.dynamic*clockRate;
//		  ///cout<<"directory.maxPower=" <<directory.maxPower<<endl;
//
//		  directory.maxPower	+=
// directory.prefetchb.l_ip.num_search_ports*directory.prefetchb.local_result.power.searchOp.dynamic*clockRate;
//		  ///cout<<"directory.maxPower=" <<directory.maxPower<<endl;
//
//		  directory.maxPower	+=
// directory.wbb.l_ip.num_search_ports*directory.wbb.local_result.power.searchOp.dynamic*clockRate;
//
//		  cc.power.readOp.dynamic = directory.maxPower*scktRatio*8;//8
// is the memory controller counts 		  cc.power.readOp.leakage =
// directory.caches.local_result.power.readOp.leakage +
//                                     directory.missb.local_result.power.readOp.leakage
//                                     +
//                                     directory.ifb.local_result.power.readOp.leakage
//                                     +
//                                     directory.prefetchb.local_result.power.readOp.leakage
//                                     +
//                                     directory.wbb.local_result.power.readOp.leakage;
//
//		  cc.power.readOp.leakage *=8;
//
//		  cc.area.set_area(directory.area*8);
//		  cout<<"CC area="<<cc.area.get_area()*1e-6<<endl;
//		  cout<<"CC Power="<<cc.power.readOp.dynamic<<endl;
//		  ccTot.area.set_area(cc.area.get_area()*1e-6);
//		  ccTot.power = cc.power;
//		  cout<<"DC energy per access" <<
// cc.power.readOp.dynamic/clockRate/8;
//	  }
//	  else if (XML->sys.first_level_dir==1)
//	  {
//		  inv_dir.maxPower =
// inv_dir.caches.local_result.power.searchOp.dynamic*clockRate*XML->sys.domain_size;
//		  cc.power.readOp.dynamic  =
// inv_dir.maxPower*scktRatio*64/XML->sys.domain_size;
// cc.power.readOp.leakage  =
// inv_dir.caches.local_result.power.readOp.leakage*inv_dir.caches.l_ip.nbanks*64/XML->sys.domain_size;
//
//		  cc.area.set_area(inv_dir.area*64/XML->sys.domain_size);
//		  cout<<"CC area="<<cc.area.get_area()*1e-6<<endl;
//		  cout<<"CC Power="<<cc.power.readOp.dynamic<<endl;
//		  ccTot.area.set_area(cc.area.get_area()*1e-6);
//		  cout<<"DC energy per access" <<
// cc.power.readOp.dynamic/clockRate/8; 		  ccTot.power =
// cc.power;
//	  }
//  }
//
//  else if (XML->sys.number_of_dir_levels==2)
//  {
//
//	  		  directory.maxPower   = 0.0;
//	  		  directory.maxPower	+=
//(directory.caches.l_ip.num_rw_ports*(0.67*directory.caches.local_result.power.readOp.dynamic+0.33*directory.caches.local_result.power.writeOp.dynamic)
//	  		                        +directory.caches.l_ip.num_rd_ports*directory.caches.local_result.power.readOp.dynamic+directory.caches.l_ip.num_wr_ports*directory.caches.local_result.power.writeOp.dynamic
//	  		                        +directory.caches.l_ip.num_se_rd_ports*directory.caches.local_result.power.readOp.dynamic)*clockRate;
//	  		  ///cout<<"directory.maxPower="
//<<directory.maxPower<<endl;
//
//	  		  directory.maxPower	+=
// directory.missb.l_ip.num_search_ports*directory.missb.local_result.power.searchOp.dynamic*clockRate;
//	  		  ///cout<<"directory.maxPower="
//<<directory.maxPower<<endl;
//
//	  		  directory.maxPower	+=
// directory.ifb.l_ip.num_search_ports*directory.ifb.local_result.power.searchOp.dynamic*clockRate;
//	  		  ///cout<<"directory.maxPower="
//<<directory.maxPower<<endl;
//
//	  		  directory.maxPower	+=
// directory.prefetchb.l_ip.num_search_ports*directory.prefetchb.local_result.power.searchOp.dynamic*clockRate;
//	  		  ///cout<<"directory.maxPower="
//<<directory.maxPower<<endl;
//
//	  		  directory.maxPower	+=
// directory.wbb.l_ip.num_search_ports*directory.wbb.local_result.power.searchOp.dynamic*clockRate;
//
//	  		  cc.power.readOp.dynamic =
// directory.maxPower*scktRatio*8;//8 is the memory controller counts
//			  cc.power.readOp.leakage =
// directory.caches.local_result.power.readOp.leakage +
//	                                     directory.missb.local_result.power.readOp.leakage
//+ directory.ifb.local_result.power.readOp.leakage +
//	                                     directory.prefetchb.local_result.power.readOp.leakage
//+ directory.wbb.local_result.power.readOp.leakage;
// cc.power.readOp.leakage
//*=8; 	  		  cc.area.set_area(directory.area*8);
//
//	  		if (XML->sys.first_level_dir==0)
//	  		{
//	  		  directory1.maxPower   = 0.0;
//	  		  directory1.maxPower	+=
//(directory1.caches.l_ip.num_rw_ports*(0.67*directory1.caches.local_result.power.readOp.dynamic+0.33*directory1.caches.local_result.power.writeOp.dynamic)
//	  				  +directory1.caches.l_ip.num_rd_ports*directory1.caches.local_result.power.readOp.dynamic+directory1.caches.l_ip.num_wr_ports*directory1.caches.local_result.power.writeOp.dynamic
//	  				  +directory1.caches.l_ip.num_se_rd_ports*directory1.caches.local_result.power.readOp.dynamic)*clockRate;
//	  		  ///cout<<"directory1.maxPower="
//<<directory1.maxPower<<endl;
//
//	  		  directory1.maxPower	+=
// directory1.missb.l_ip.num_search_ports*directory1.missb.local_result.power.searchOp.dynamic*clockRate;
//	  		  ///cout<<"directory1.maxPower="
//<<directory1.maxPower<<endl;
//
//	  		  directory1.maxPower	+=
// directory1.ifb.l_ip.num_search_ports*directory1.ifb.local_result.power.searchOp.dynamic*clockRate;
//	  		  ///cout<<"directory1.maxPower="
//<<directory1.maxPower<<endl;
//
//	  		  directory1.maxPower	+=
// directory1.prefetchb.l_ip.num_search_ports*directory1.prefetchb.local_result.power.searchOp.dynamic*clockRate;
//	  		  ///cout<<"directory1.maxPower="
//<<directory1.maxPower<<endl;
//
//	  		  directory1.maxPower	+=
// directory1.wbb.l_ip.num_search_ports*directory1.wbb.local_result.power.searchOp.dynamic*clockRate;
//
//	  		  cc1.power.readOp.dynamic =
// directory1.maxPower*scktRatio*64/XML->sys.domain_size;
//			  cc1.power.readOp.leakage =
// directory1.caches.local_result.power.readOp.leakage +
//	                                     directory1.missb.local_result.power.readOp.leakage
//+ directory1.ifb.local_result.power.readOp.leakage +
//	                                     directory1.prefetchb.local_result.power.readOp.leakage
//+ directory1.wbb.local_result.power.readOp.leakage;
// cc1.power.readOp.leakage
//*= 64/XML->sys.domain_size;
//	  		  cc1.area.set_area(directory1.area*64/XML->sys.domain_size);
//
//	  		  cout<<"CC
// area="<<(cc.area.get_area()+cc1.area.get_area())*1e-6<<endl;
// cout<<"CC Power="<<cc.power.readOp.dynamic + cc1.power.readOp.dynamic <<endl;
//			  ccTot.area.set_area((cc.area.get_area()+cc1.area.get_area())*1e-6);
//			  ccTot.power = cc.power + cc1.power;
//	  	  }
//	  	  else if (XML->sys.first_level_dir==1)
//	  	  {
//	  		  inv_dir.maxPower =
// inv_dir.caches.local_result.power.searchOp.dynamic*clockRate*XML->sys.domain_size;
//	  		  cc1.power.readOp.dynamic =
// inv_dir.maxPower*scktRatio*(64/XML->sys.domain_size);
// cc1.power.readOp.leakage
//=
// inv_dir.caches.local_result.power.readOp.leakage*inv_dir.caches.l_ip.nbanks*XML->sys.domain_size;
//
//	  		  cc1.area.set_area(inv_dir.area*64/XML->sys.domain_size);
//			  cout<<"CC
// area="<<(cc.area.get_area()+cc1.area.get_area())*1e-6<<endl;
// cout<<"CC Power="<<cc.power.readOp.dynamic + cc1.power.readOp.dynamic <<endl;
//			  ccTot.area.set_area((cc.area.get_area()+cc1.area.get_area())*1e-6);
//			  ccTot.power = cc.power + cc1.power;
//
//	  	  }
//	  	  else if (XML->sys.first_level_dir==2)
//	  	  {
//			  cout<<"CC area="<<cc.area.get_area()*1e-6<<endl;
//			  cout<<"CC Power="<<cc.power.readOp.dynamic<<endl;
//			  ccTot.area.set_area(cc.area.get_area()*1e-6);
//			  ccTot.power = cc.power;
//	  	  }
//  }
//
// cout<<"L2cache size="<<L2Tot.area.get_area()*1e-6<<endl;
// cout<<"L2cache dynamic power="<<L2Tot.power.readOp.dynamic<<endl;
// cout<<"L2cache laeakge power="<<L2Tot.power.readOp.leakage<<endl;
//
//  ///cout<<"llCache.maxPower=" <<llCache.maxPower<<endl;
//
//
//  maxPower          +=  llCache.maxPower;
//  ///cout<<"maxpower=" <<maxPower<<endl;
//
////  maxPower	  +=  pipeLogicCache.power.readOp.dynamic*clockRate;
////
//////cout<<"pipeLogic.power="<<pipeLogicCache.power.readOp.dynamic*clockRate<<endl;
////  ///cout<<"maxpower=" <<maxPower<<endl;
////
////  maxPower	  +=  pipeLogicDirectory.power.readOp.dynamic*clockRate;
////
//////cout<<"pipeLogic.power="<<pipeLogicDirectory.power.readOp.dynamic*clockRate<<endl;
////  ///cout<<"maxpower=" <<maxPower<<endl;
////
////  //clock power
////  maxPower += clockNetwork.total_power.readOp.dynamic*clockRate;
////
//////cout<<"clockNetwork.total_power="<<clockNetwork.total_power.readOp.dynamic*clockRate<<endl;
////  ///cout<<"maxpower=" <<maxPower<<endl;
//
//}

/*
 * [한국어]
 * SharedCache::set_cache_param - XML에서 캐시 파라미터를 cachep로 복사
 *
 * cacheL 값에 따라 L2, L3, L1Directory, L2Directory 중 하나를 선택하고,
 * 각 계층의 clockRate, executionTime, device_type, capacity, blockW, assoc,
 * nbanks, throughput, latency, buffer_sizes, duty_cycle, 디렉토리 병합 여부 등을
 * XML->sys.* 에서 읽어 cachep 및 interface_ip 관련 필드를 채운다.
 */
void SharedCache::set_cache_param() {
  // [한국어] GPU 통합 L2 파라미터 로드
  if (cacheL == L2) {
    cachep.name = "L2";
    cachep.clockRate = XML->sys.L2[ithCache].clockrate;
    // [한국어] MHz → Hz 변환
    cachep.clockRate *= 1e6;
    // [한국어] 전체 실행 시간(초) = 사이클 수 / 목표 코어 클럭
    cachep.executionTime =
        XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);
    interface_ip.data_arr_ram_cell_tech_type =
        XML->sys.L2[ithCache].device_type;  // long channel device LSTP
    interface_ip.data_arr_peri_global_tech_type =
        XML->sys.L2[ithCache].device_type;
    interface_ip.tag_arr_ram_cell_tech_type = XML->sys.L2[ithCache].device_type;
    interface_ip.tag_arr_peri_global_tech_type =
        XML->sys.L2[ithCache].device_type;
    cachep.capacity = XML->sys.L2[ithCache].L2_config[0];
    cachep.blockW = XML->sys.L2[ithCache].L2_config[1];
    cachep.assoc = XML->sys.L2[ithCache].L2_config[2];
    cachep.nbanks = XML->sys.L2[ithCache].L2_config[3];
    cachep.throughput = XML->sys.L2[ithCache].L2_config[4] / cachep.clockRate;
    cachep.latency = XML->sys.L2[ithCache].L2_config[5] / cachep.clockRate;
    cachep.missb_size = XML->sys.L2[ithCache].buffer_sizes[0];
    cachep.fu_size = XML->sys.L2[ithCache].buffer_sizes[1];
    cachep.prefetchb_size = XML->sys.L2[ithCache].buffer_sizes[2];
    cachep.wbb_size = XML->sys.L2[ithCache].buffer_sizes[3];
    cachep.duty_cycle = XML->sys.L2[ithCache].duty_cycle;
    // [한국어] merged_dir=false면 일반 캐시(NonDir)
    if (!XML->sys.L2[ithCache].merged_dir) {
      cachep.dir_ty = NonDir;
    } else {
      // [한국어] merged_dir=true면 SBT(디렉토리 비트 병합) 캐시
      cachep.dir_ty = SBT;
      cachep.dir_duty_cycle = XML->sys.L2[ithCache].dir_duty_cycle;
    }
  // [한국어] L3(Last-Level Cache) 파라미터 로드
  } else if (cacheL == L3) {
    cachep.name = "L3";
    cachep.clockRate = XML->sys.L3[ithCache].clockrate;
    cachep.clockRate *= 1e6;
    cachep.executionTime =
        XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);
    interface_ip.data_arr_ram_cell_tech_type =
        XML->sys.L3[ithCache].device_type;  // long channel device LSTP
    interface_ip.data_arr_peri_global_tech_type =
        XML->sys.L3[ithCache].device_type;
    interface_ip.tag_arr_ram_cell_tech_type = XML->sys.L3[ithCache].device_type;
    interface_ip.tag_arr_peri_global_tech_type =
        XML->sys.L3[ithCache].device_type;
    cachep.capacity = XML->sys.L3[ithCache].L3_config[0];
    cachep.blockW = XML->sys.L3[ithCache].L3_config[1];
    cachep.assoc = XML->sys.L3[ithCache].L3_config[2];
    cachep.nbanks = XML->sys.L3[ithCache].L3_config[3];
    cachep.throughput = XML->sys.L3[ithCache].L3_config[4] / cachep.clockRate;
    cachep.latency = XML->sys.L3[ithCache].L3_config[5] / cachep.clockRate;
    cachep.missb_size = XML->sys.L3[ithCache].buffer_sizes[0];
    cachep.fu_size = XML->sys.L3[ithCache].buffer_sizes[1];
    cachep.prefetchb_size = XML->sys.L3[ithCache].buffer_sizes[2];
    cachep.wbb_size = XML->sys.L3[ithCache].buffer_sizes[3];
    cachep.duty_cycle = XML->sys.L3[ithCache].duty_cycle;
    if (!XML->sys.L2[ithCache].merged_dir) {
      cachep.dir_ty = NonDir;
    } else {
      cachep.dir_ty = SBT;
      cachep.dir_duty_cycle = XML->sys.L2[ithCache].dir_duty_cycle;
    }
  // [한국어] L1 Directory 파라미터 로드
  } else if (cacheL == L1Directory) {
    cachep.name = "First Level Directory";
    cachep.dir_ty =
        (enum Dir_type)XML->sys.L1Directory[ithCache].Directory_type;
    cachep.clockRate = XML->sys.L1Directory[ithCache].clockrate;
    cachep.clockRate *= 1e6;
    cachep.executionTime =
        XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);
    interface_ip.data_arr_ram_cell_tech_type =
        XML->sys.L1Directory[ithCache].device_type;  // long channel device LSTP
    interface_ip.data_arr_peri_global_tech_type =
        XML->sys.L1Directory[ithCache].device_type;
    interface_ip.tag_arr_ram_cell_tech_type =
        XML->sys.L1Directory[ithCache].device_type;
    interface_ip.tag_arr_peri_global_tech_type =
        XML->sys.L1Directory[ithCache].device_type;
    cachep.capacity = XML->sys.L1Directory[ithCache].Dir_config[0];
    cachep.blockW = XML->sys.L1Directory[ithCache].Dir_config[1];
    cachep.assoc = XML->sys.L1Directory[ithCache].Dir_config[2];
    cachep.nbanks = XML->sys.L1Directory[ithCache].Dir_config[3];
    cachep.throughput =
        XML->sys.L1Directory[ithCache].Dir_config[4] / cachep.clockRate;
    cachep.latency =
        XML->sys.L1Directory[ithCache].Dir_config[5] / cachep.clockRate;
    cachep.missb_size = XML->sys.L1Directory[ithCache].buffer_sizes[0];
    cachep.fu_size = XML->sys.L1Directory[ithCache].buffer_sizes[1];
    cachep.prefetchb_size = XML->sys.L1Directory[ithCache].buffer_sizes[2];
    cachep.wbb_size = XML->sys.L1Directory[ithCache].buffer_sizes[3];
    cachep.duty_cycle = XML->sys.L1Directory[ithCache].duty_cycle;
  // [한국어] L2 Directory 파라미터 로드
  } else if (cacheL == L2Directory) {
    cachep.name = "Second Level Directory";
    cachep.dir_ty =
        (enum Dir_type)XML->sys.L2Directory[ithCache].Directory_type;
    cachep.clockRate = XML->sys.L2Directory[ithCache].clockrate;
    cachep.clockRate *= 1e6;
    cachep.executionTime =
        XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);
    interface_ip.data_arr_ram_cell_tech_type =
        XML->sys.L2Directory[ithCache].device_type;  // long channel device LSTP
    interface_ip.data_arr_peri_global_tech_type =
        XML->sys.L2Directory[ithCache].device_type;
    interface_ip.tag_arr_ram_cell_tech_type =
        XML->sys.L2Directory[ithCache].device_type;
    interface_ip.tag_arr_peri_global_tech_type =
        XML->sys.L2Directory[ithCache].device_type;
    cachep.capacity = XML->sys.L2Directory[ithCache].Dir_config[0];
    cachep.blockW = XML->sys.L2Directory[ithCache].Dir_config[1];
    cachep.assoc = XML->sys.L2Directory[ithCache].Dir_config[2];
    cachep.nbanks = XML->sys.L2Directory[ithCache].Dir_config[3];
    cachep.throughput =
        XML->sys.L2Directory[ithCache].Dir_config[4] / cachep.clockRate;
    cachep.latency =
        XML->sys.L2Directory[ithCache].Dir_config[5] / cachep.clockRate;
    cachep.missb_size = XML->sys.L2Directory[ithCache].buffer_sizes[0];
    cachep.fu_size = XML->sys.L2Directory[ithCache].buffer_sizes[1];
    cachep.prefetchb_size = XML->sys.L2Directory[ithCache].buffer_sizes[2];
    cachep.wbb_size = XML->sys.L2Directory[ithCache].buffer_sizes[3];
    cachep.duty_cycle = XML->sys.L2Directory[ithCache].duty_cycle;
  }
  // cachep.cache_duty_cycle=cachep.dir_duty_cycle = 0.35;
}
