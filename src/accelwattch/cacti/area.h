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
 * [한국어 설명] CACTI 면적 표현 헤더 (area.h)
 *
 * === 파일의 역할 ===
 * CACTI 내 모든 하드웨어 컴포넌트의 물리적 면적(가로×세로, 또는 직접 지정 면적)을
 * 표현하는 Area 클래스를 정의한다. Area는 단순히 w(폭), h(높이), area(직접 지정 값)를
 * 보관하고 get_area()를 통해 w×h 또는 area 중 적절한 값을 반환한다. 이 클래스는
 * Component, Bank, Mat, Subarray 등 CACTI의 모든 컴포넌트에 포함(포함 관계)되어
 * 면적 집계의 기본 단위가 된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI 면적 모델의 최하위 기본 타입이다. 모든 CACTI 컴포넌트(Component, Bank, Mat,
 * Subarray, Wire, Decoder 등)가 Area 인스턴스를 멤버로 보유하며, 상위 컴포넌트는
 * 하위 컴포넌트의 area를 읽어 자신의 총 면적을 계산한다. AccelWattch가 GPU 캐시
 * 면적을 추정할 때 최종적으로 Bank::area 혹은 Component::area가 참조된다.
 *
 * === 타 모듈과의 연결 ===
 * - cacti_interface.h: g_ip, g_tp 등 전역 파라미터 (Area 자체는 직접 참조하지 않음).
 * - basic_circuit.h: 기초 회로 함수 헤더 (Area 자체에서는 미사용, 포함만).
 * - component.h: Component 클래스가 Area area 멤버를 선언. 모든 컴포넌트가 의존.
 * - bank.h, mat.h, htree2.h: 각 컴포넌트가 Area area 멤버로 자신의 면적을 보유.
 *
 * === 주요 함수/구조체 요약 ===
 * - Area(): 기본 생성자. w=0, h=0, area=0으로 초기화.
 * - get_area(): w, h 중 하나라도 0이면 직접 지정된 area 반환, 아니면 w×h 반환.
 * - set_w()/set_h()/set_area(): 폭, 높이, 면적을 직접 설정하는 세터.
 * - get_w()/get_h(): 현재 폭, 높이 반환하는 게터.
 *
 * === AccelWattch XML / gpgpusim.config 연동 ===
 * Area 클래스 자체는 설정을 직접 읽지 않지만, 상위 Bank/Mat/Subarray가 계산하는
 * 면적은 AccelWattch XML의 캐시 구성 옵션(용량, 연관도, 블록 크기, 뱅크 수,
 * 기술 노드, 셀 타입 등)과 gpgpusim.config의 -gpgpu_l2_rop_latency,
 * -gpgpu_n_mem 등 간접적으로 영향을 받는다. 특히 기술 노드(F_sz_um)와 캐시
 * 크기(cache_sz)는 면적 계산의 핵심 입력이며, 이 값들은 XML의
 * sys.L2[0].L2_config, sys.dcache_config, sys.core[].icache_config 등에서
 * 파생된다.
 */

#ifndef __AREA_H__
#define __AREA_H__

#include "cacti_interface.h" // [한국어] g_ip, g_tp 전역 파라미터 포함 (Area 자체에서 직접 사용 안 하지만 의존 헤더 체인에 포함됨)
#include "basic_circuit.h"   // [한국어] 기초 CMOS 회로 계산 함수 (Area 자체에서는 미사용, 포함 헤더)

using namespace std; // [한국어] std 네임스페이스 전역 using (CACTI 코드베이스 전체 관례)

/*
 * [한국어]
 * Area - CACTI 컴포넌트의 물리적 면적을 표현하는 기본 클래스
 *
 * 가로(w) × 세로(h)로 직사각형 면적을 표현하거나, 직접 지정된 area 값을 보유한다.
 * get_area()는 w와 h 중 하나라도 0이면 set_area()로 직접 지정한 값을 반환하고,
 * 그 외에는 w×h를 반환한다. CACTI의 모든 하드웨어 컴포넌트 (Component, Bank, Mat,
 * Subarray, Wire, Decoder 등)가 이 클래스의 인스턴스를 멤버로 보유한다.
 */
class Area
{
 public:
  double w;
  /* w: 컴포넌트의 가로 폭 (µm).
   * 설정자: set_w() 또는 Bank/Mat 등 상위 모델에서 직접 대입 (e.g., area.w = htree->area.w).
   * 읽는 자: 상위 컴포넌트가 면적 집계 시 get_w() 또는 직접 접근.
   * 값 범위: 0 이상 (0이면 area 필드를 면적으로 사용).
   * 동기화: 단일 스레드 내 CACTI 계산에서만 사용; 별도 락 불필요. */
  double h;
  /* h: 컴포넌트의 세로 높이 (µm).
   * 설정자: set_h() 또는 상위 모델에서 직접 대입.
   * 읽는 자: 상위 컴포넌트의 면적 집계 시 get_h() 또는 직접 접근.
   * 값 범위: 0 이상 (0이면 area 필드를 면적으로 사용).
   * 동기화: 단일 스레드; 별도 락 불필요. */

  /*
   * [한국어]
   * Area - 기본 생성자
   *
   * w, h, area 모두 0으로 초기화. 이후 set_w/set_h 또는 set_area로 값을 지정한다.
   */
  Area():w(0), h(0), area(0) { }

  /*
   * [한국어]
   * get_w - 현재 컴포넌트 가로 폭 반환
   *
   * @return: w (µm)
   */
  double get_w() const { return w; }

  /*
   * [한국어]
   * get_h - 현재 컴포넌트 세로 높이 반환
   *
   * @return: h (µm)
   */
  double get_h() const { return h; }

  /*
   * [한국어]
   * get_area - 컴포넌트 면적 반환
   *
   * @return: w==0 && h==0이면 set_area()로 직접 지정한 area 반환,
   *          그 외에는 w×h (µm²) 반환.
   *
   * w와 h 양쪽이 모두 설정된 경우 직사각형 면적(w×h)을 반환하고,
   * 레이아웃 폭/높이가 아닌 전체 면적만 알고 있을 때는 area를 직접 반환한다.
   * 이 이중 경로를 통해 컴포넌트 초기화 순서에 무관하게 면적을 질의할 수 있다.
   *
   * 호출 체인: 상위 컴포넌트(Bank/Mat 등) 면적 집계 → [get_area()]
   */
  double get_area() const
  {
    if (w == 0 && h == 0) // [한국어] 폭·높이 미설정 → 직접 지정된 area 반환
    {
      return area; // [한국어] set_area()로 지정한 절대 면적 반환
    }
    else
    {
      return w*h; // [한국어] 직사각형 면적 = 폭 × 높이 (µm²)
    }
  }

  /*
   * [한국어]
   * set_w - 컴포넌트 가로 폭 설정
   *
   * @w_: 설정할 폭 (µm)
   */
  void set_w(double w_) { w = w_; } // [한국어] w 멤버에 직접 대입

  /*
   * [한국어]
   * set_h - 컴포넌트 세로 높이 설정
   *
   * @h_: 설정할 높이 (µm)
   */
  void set_h(double h_) { h = h_; } // [한국어] h 멤버에 직접 대입

  /*
   * [한국어]
   * set_area - 면적 직접 지정 (w, h 미사용 시)
   *
   * @a_: 설정할 면적 (µm²). w==h==0인 상태에서 get_area()가 이 값을 반환한다.
   *
   * 컴포넌트의 폭·높이 분해 없이 전체 면적만 알고 있을 때 사용한다.
   */
  void set_area(double a_) { area = a_; } // [한국어] private area 멤버에 직접 대입

 private:
  double area;
  /* area: 직접 지정 면적 (µm²). w==0 && h==0일 때 get_area()가 반환하는 값.
   * 설정자: set_area().
   * 읽는 자: get_area() — w, h가 모두 0인 경우에만 반환.
   * 값 범위: 0 이상 (일반적으로 수 µm² ~ 수십만 µm²).
   * 동기화: 단일 스레드 CACTI 계산; 별도 락 불필요. */
};

#endif

