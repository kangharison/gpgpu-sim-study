// $Id: routefunc.cpp 5507 2013-05-23 14:54:38Z qtedq $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*routefunc.cpp
 *
 *This is where most of the routing functions reside. Some of the topologies
 *has their own "register routing functions" which must be called to access
 *those routing functions.
 *
 *After writing a routing function, don't forget to register it. The reg
 *format is rfname_topologyname.
 *
 */

/*
 * [한국어 설명] NoC 라우팅 알고리즘 구현 (routefunc.cpp)
 *
 * === 파일의 역할 ===
 * BookSim2에서 지원하는 다양한 NoC 라우팅 알고리즘을 구현하고 gRoutingFunctionMap에 등록한다.
 * 각 라우팅 함수는 현재 라우터 위치와 Flit의 목적지를 입력으로 받아,
 * 사용 가능한 출구 포트와 VC 범위를 OutputSet에 추가한다.
 * GPGPU-Sim에서는 주로 "dor_mesh" (Dimension-Order Routing)나
 * "valiant_mesh" (Valiant 부하 분산 라우팅)을 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 초기화: CreateInterconnect() → InitializeRoutingMap() → gRoutingFunctionMap 채움
 * 사용:   VC::Route(rf, router, flit, in_ch) → rf() (매 라우팅 단계마다 호출)
 *         GPUTrafficManager::_Step() → _rf() (주입 단계 VC 결정)
 *
 * === 타 모듈과의 연결 ===
 * - vc.cpp: Route()에서 함수 포인터로 호출
 * - gputrafficmanager.cpp: _rf를 주입 단계 VC 결정에 사용
 * - flit.hpp: Flit::type, dest, ph, intm 필드 읽기/쓰기
 * - router.hpp: Router::GetID(), GetUsedCredit() 참조
 *
 * === 주요 함수/구조체 요약 ===
 * - InitializeRoutingMap(): VC 범위 설정 + 모든 알고리즘 등록
 * - dor_next_mesh(): 메시에서 DOR 다음 출구 포트 계산 (보조 함수)
 * - dim_order_mesh(): DOR 라우팅 (결정론적, 데드락 없음)
 * - valiant_mesh(): Valiant 라우팅 (2단계: 랜덤 중간→목적지)
 * - min_adapt_mesh(): 최소 경로 적응형 + DOR 이탈 채널
 * - fattree_nca/anca(): Fat-tree NCA 라우팅 (상향/하향)
 */

#include <map>      // [한국어] gRoutingFunctionMap의 map<string, tRoutingFunction> 타입
#include <cstdlib>  // [한국어] rand(), abs() 등 표준 라이브러리
#include <cassert>  // [한국어] assert() 매크로

#include "booksim.hpp"       // [한국어] BookSim 공통 정의 (gN, gK, gNodes 등 글로벌 변수)
#include "routefunc.hpp"     // [한국어] tRoutingFunction, gRoutingFunctionMap 선언
#include "kncube.hpp"        // [한국어] k-ary n-cube 토폴로지 라우팅 등록용
#include "random_utils.hpp"  // [한국어] RandomInt(): 랜덤 중간 노드 선택에 사용
#include "misc_utils.hpp"    // [한국어] powi(): 지수 계산 (fat-tree 라우팅에서 사용)
#include "fattree.hpp"       // [한국어] Fat-tree 토폴로지 라우팅 등록용
#include "tree4.hpp"         // [한국어] Tree4 토폴로지 라우팅 등록용
#include "qtree.hpp"         // [한국어] QTree 토폴로지 라우팅 등록용
#include "cmesh.hpp"         // [한국어] Concentrated mesh 토폴로지 라우팅 등록용



map<string, tRoutingFunction> gRoutingFunctionMap;
/* [한국어] 라우팅 알고리즘 이름 → 함수 포인터 맵.
 * InitializeRoutingMap()에서 채워지며, Router 생성자가 설정 파일의
 * "routing_function" 문자열로 함수를 조회한다.
 * 예: gRoutingFunctionMap["dor_mesh"] = &dim_order_mesh */

/* Global information used by routing functions */

int gNumVCs;
/* [한국어] 라우터당 전체 VC 수. 라우팅 함수에서 VC 범위 계산의 상한.
 * InitializeRoutingMap()에서 설정 파일의 num_vcs로 초기화. */

/* Add more functions here
 *
 */

// ============================================================
//  Balfour-Schultz
// [한국어] Balfour-Schultz 방식의 타입별 VC 분리 변수.
// 요청(0..gNumVCs/2-1)과 응답(gNumVCs/2..gNumVCs-1)으로 VC를 분리하여
// 요청-응답 순환 의존에 의한 데드락을 방지한다.
int gReadReqBeginVC, gReadReqEndVC;    // [한국어] 읽기 요청 VC 범위 [Begin, End]
int gWriteReqBeginVC, gWriteReqEndVC;  // [한국어] 쓰기 요청 VC 범위 [Begin, End]
int gReadReplyBeginVC, gReadReplyEndVC;   // [한국어] 읽기 응답 VC 범위 [Begin, End]
int gWriteReplyBeginVC, gWriteReplyEndVC; // [한국어] 쓰기 응답 VC 범위 [Begin, End]

// ============================================================
//  QTree: Nearest Common Ancestor
// ===
/*
 * [한국어]
 * qtree_nca() - QTree 토폴로지에서 NCA(Nearest Common Ancestor) 라우팅
 *
 * @r: 현재 라우터 (위치 = 트리 높이와 위치로 표현)
 * @f: 라우팅 대상 Flit
 * @in_channel: 입구 채널 번호
 * @outputs: 라우팅 결과 출구 포트 집합 (AddRange로 채움)
 * @inject: true이면 주입 단계 (out_port = -1)
 *
 * QTree의 NCA 라우팅: 현재 라우터가 목적지의 NCA이면 아래로 라우팅,
 * 아니면 부모 방향(위)으로 라우팅한다.
 * VC 범위는 패킷 타입(요청/응답)에 따라 분리된다.
 *
 * 호출 체인: gRoutingFunctionMap["nca_qtree"] → [이 함수]
 */
void qtree_nca( const Router *r, const Flit *f,
		int in_channel, OutputSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int height = QTree::HeightFromID( r->GetID() );
    int pos    = QTree::PosFromID( r->GetID() );
    
    int dest   = f->dest;
    
    for (int i = height+1; i < gN; i++) 
      dest /= gK;
    if ( pos == dest / gK ) 
      // Route down to child
      out_port = dest % gK ; 
    else
      // Route up to parent
      out_port = gK;        

  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

// ============================================================
//  Tree4: Nearest Common Ancestor w/ Adaptive Routing Up
// ===
void tree4_anca( const Router *r, const Flit *f,
		 int in_channel, OutputSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int range = 1;
  
  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int dest = f->dest;
    
    const int NPOS = 16;
    
    int rH = r->GetID( ) / NPOS;
    int rP = r->GetID( ) % NPOS;
    
    if ( rH == 0 ) {
      dest /= 16;
      out_port = 2 * dest + RandomInt(1);
    } else if ( rH == 1 ) {
      dest /= 4;
      if ( dest / 4 == rP / 2 )
	out_port = dest % 4;
      else {
	out_port = gK;
	range = gK;
      }
    } else {
      if ( dest/4 == rP )
	out_port = dest % 4;
      else {
	out_port = gK;
	range = 2;
      }
    }
    
    //  cout << "Router("<<rH<<","<<rP<<"): id= " << f->id << " dest= " << f->dest << " out_port = "
    //       << out_port << endl;

  }

  outputs->Clear( );

  for (int i = 0; i < range; ++i) 
    outputs->AddRange( out_port + i, vcBegin, vcEnd );
}

// ============================================================
//  Tree4: Nearest Common Ancestor w/ Random Routing Up
// ===
void tree4_nca( const Router *r, const Flit *f,
		int in_channel, OutputSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int dest = f->dest;
    
    const int NPOS = 16;
    
    int rH = r->GetID( ) / NPOS;
    int rP = r->GetID( ) % NPOS;
    
    if ( rH == 0 ) {
      dest /= 16;
      out_port = 2 * dest + RandomInt(1);
    } else if ( rH == 1 ) {
      dest /= 4;
      if ( dest / 4 == rP / 2 )
	out_port = dest % 4;
      else
	out_port = gK + RandomInt(gK-1);
    } else {
      if ( dest/4 == rP )
	out_port = dest % 4;
      else
	out_port = gK + RandomInt(1);
    }
    
    //  cout << "Router("<<rH<<","<<rP<<"): id= " << f->id << " dest= " << f->dest << " out_port = "
    //       << out_port << endl;

  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

// ============================================================
//  FATTREE: Nearest Common Ancestor w/ Random  Routing Up
// ===
void fattree_nca( const Router *r, const Flit *f,
               int in_channel, OutputSet* outputs, bool inject)
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {
    
    int dest = f->dest;
    int router_id = r->GetID(); //routers are numbered with smallest at the top level
    int routers_per_level = powi(gK, gN-1);
    int pos = router_id%routers_per_level;
    int router_depth  = router_id/ routers_per_level; //which level
    int routers_per_neighborhood = powi(gK,gN-router_depth-1);
    int router_neighborhood = pos/routers_per_neighborhood; //coverage of this tree
    int router_coverage = powi(gK, gN-router_depth);  //span of the tree from this router
    

    //NCA reached going down
    if(dest <(router_neighborhood+1)* router_coverage && 
       dest >=router_neighborhood* router_coverage){
      //down ports are numbered first

      //ejection
      if(router_depth == gN-1){
	out_port = dest%gK;
      } else {	
	//find the down port for the destination
	int router_branch_coverage = powi(gK, gN-(router_depth+1)); 
	out_port = (dest-router_neighborhood* router_coverage)/router_branch_coverage;
      }
    } else {
      //up ports are numbered last
      assert(in_channel<gK);//came from a up channel
      out_port = gK+RandomInt(gK-1);
    }
  }  
  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

// ============================================================
//  FATTREE: Nearest Common Ancestor w/ Adaptive Routing Up
// ===
void fattree_anca( const Router *r, const Flit *f,
                int in_channel, OutputSet* outputs, bool inject)
{

  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));


  int out_port;

  if(inject) {

    out_port = -1;

  } else {


    int dest = f->dest;
    int router_id = r->GetID(); //routers are numbered with smallest at the top level
    int routers_per_level = powi(gK, gN-1);
    int pos = router_id%routers_per_level;
    int router_depth  = router_id/ routers_per_level; //which level
    int routers_per_neighborhood = powi(gK,gN-router_depth-1);
    int router_neighborhood = pos/routers_per_neighborhood; //coverage of this tree
    int router_coverage = powi(gK, gN-router_depth);  //span of the tree from this router
    

    //NCA reached going down
    if(dest <(router_neighborhood+1)* router_coverage && 
       dest >=router_neighborhood* router_coverage){
      //down ports are numbered first

      //ejection
      if(router_depth == gN-1){
	out_port = dest%gK;
      } else {	
	//find the down port for the destination
	int router_branch_coverage = powi(gK, gN-(router_depth+1)); 
	out_port = (dest-router_neighborhood* router_coverage)/router_branch_coverage;
      }
    } else {
      //up ports are numbered last
      assert(in_channel<gK);//came from a up channel
      out_port = gK;
      int random1 = RandomInt(gK-1); // Chose two ports out of the possible at random, compare loads, choose one.
      int random2 = RandomInt(gK-1);
      if (r->GetUsedCredit(out_port + random1) > r->GetUsedCredit(out_port + random2)){
	out_port = out_port + random2;
      }else{
	out_port =  out_port + random1;
      }
    }
  }  
  outputs->Clear( );
  
  outputs->AddRange( out_port, vcBegin, vcEnd );
}




// ============================================================
//  Mesh - adatpive XY,YX Routing 
//         pick xy or yx min routing adaptively at the source router
// ===

int dor_next_mesh( int cur, int dest, bool descending = false );

void adaptive_xy_yx_mesh( const Router *r, const Flit *f, 
		 int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else if(r->GetID() == f->dest) {

    // at destination router, we don't need to separate VCs by dim order
    out_port = 2*gN;

  } else {

    //each class must have at least 2 vcs assigned or else xy_yx will deadlock
    int const available_vcs = (vcEnd - vcBegin + 1) / 2;
    assert(available_vcs > 0);
    
    int out_port_xy = dor_next_mesh( r->GetID(), f->dest, false );
    int out_port_yx = dor_next_mesh( r->GetID(), f->dest, true );

    // Route order (XY or YX) determined when packet is injected
    //  into the network, adaptively
    bool x_then_y;
    if(in_channel < 2*gN){
      x_then_y =  (f->vc < (vcBegin + available_vcs));
    } else {
      int credit_xy = r->GetUsedCredit(out_port_xy);
      int credit_yx = r->GetUsedCredit(out_port_yx);
      if(credit_xy > credit_yx) {
	x_then_y = false;
      } else if(credit_xy < credit_yx) {
	x_then_y = true;
      } else {
	x_then_y = (RandomInt(1) > 0);
      }
    }
    
    if(x_then_y) {
      out_port = out_port_xy;
      vcEnd -= available_vcs;
    } else {
      out_port = out_port_yx;
      vcBegin += available_vcs;
    }

  }

  outputs->Clear();

  outputs->AddRange( out_port , vcBegin, vcEnd );
  
}

void xy_yx_mesh( const Router *r, const Flit *f, 
		 int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else if(r->GetID() == f->dest) {

    // at destination router, we don't need to separate VCs by dim order
    out_port = 2*gN;

  } else {

    //each class must have at least 2 vcs assigned or else xy_yx will deadlock
    int const available_vcs = (vcEnd - vcBegin + 1) / 2;
    assert(available_vcs > 0);

    // Route order (XY or YX) determined when packet is injected
    //  into the network
    bool x_then_y = ((in_channel < 2*gN) ?
		     (f->vc < (vcBegin + available_vcs)) :
		     (RandomInt(1) > 0));

    if(x_then_y) {
      out_port = dor_next_mesh( r->GetID(), f->dest, false );
      vcEnd -= available_vcs;
    } else {
      out_port = dor_next_mesh( r->GetID(), f->dest, true );
      vcBegin += available_vcs;
    }

  }

  outputs->Clear();

  outputs->AddRange( out_port , vcBegin, vcEnd );
  
}

//
// End Balfour-Schultz
//=============================================================

//=============================================================

/*
 * [한국어]
 * dor_next_mesh() - 메시에서 DOR 방식의 다음 출구 포트를 계산하는 보조 함수
 *
 * @cur: 현재 라우터의 선형화 ID (좌표: x + y*gK + z*gK^2 + ...)
 * @dest: 목적지 라우터의 선형화 ID
 * @descending: false=XY 방향(낮은 차원 우선), true=YX 방향(높은 차원 우선)
 * @return: 다음 홉 출구 포트 (짝수=+방향, 홀수=-방향, 2*gN=이젝션)
 *
 * 동작:
 *   1. cur==dest이면 이젝션 포트(2*gN) 반환
 *   2. descending=false: 낮은 차원(x)부터 불일치 차원 탐색
 *      descending=true:  높은 차원(y,z...)부터 불일치 차원 탐색
 *   3. 불일치 차원 dim_left에서:
 *      cur%gK < dest%gK → 오른쪽(+방향): 2*dim_left 반환
 *      cur%gK > dest%gK → 왼쪽(-방향): 2*dim_left+1 반환
 *
 * 메시 포트 번호 규칙:
 *   dim 0: 포트 0(+x), 포트 1(-x)
 *   dim 1: 포트 2(+y), 포트 3(-y)
 *   이젝션: 포트 2*gN
 *
 * 호출 체인: dim_order_mesh(), xy_yx_mesh(), valiant_mesh() 등 → [이 함수]
 */
int dor_next_mesh( int cur, int dest, bool descending )
{
  if ( cur == dest ) {
    return 2*gN;  // Eject
  }

  int dim_left;

  if(descending) {
    for ( dim_left = ( gN - 1 ); dim_left > 0; --dim_left ) {
      if ( ( cur * gK / gNodes ) != ( dest * gK / gNodes ) ) { break; }
      cur = (cur * gK) % gNodes; dest = (dest * gK) % gNodes;
    }
    cur = (cur * gK) / gNodes;
    dest = (dest * gK) / gNodes;
  } else {
    for ( dim_left = 0; dim_left < ( gN - 1 ); ++dim_left ) {
      if ( ( cur % gK ) != ( dest % gK ) ) { break; }
      cur /= gK; dest /= gK;
    }
    cur %= gK;
    dest %= gK;
  }

  if ( cur < dest ) {
    return 2*dim_left;     // Right
  } else {
    return 2*dim_left + 1; // Left
  }
}

//=============================================================

void dor_next_torus( int cur, int dest, int in_port,
		     int *out_port, int *partition,
		     bool balance = false )
{
  int dim_left;
  int dir;
  int dist2;

  for ( dim_left = 0; dim_left < gN; ++dim_left ) {
    if ( ( cur % gK ) != ( dest % gK ) ) { break; }
    cur /= gK; dest /= gK;
  }
  
  if ( dim_left < gN ) {

    if ( (in_port/2) != dim_left ) {
      // Turning into a new dimension

      cur %= gK; dest %= gK;
      dist2 = gK - 2 * ( ( dest - cur + gK ) % gK );
      
      if ( ( dist2 > 0 ) || 
	   ( ( dist2 == 0 ) && ( RandomInt( 1 ) ) ) ) {
	*out_port = 2*dim_left;     // Right
	dir = 0;
      } else {
	*out_port = 2*dim_left + 1; // Left
	dir = 1;
      }
      
      if ( partition ) {
	if ( balance ) {
	  // Cray's "Partition" allocation
	  // Two datelines: one between k-1 and 0 which forces VC 1
	  //                another between ((k-1)/2) and ((k-1)/2 + 1) which 
	  //                forces VC 0 otherwise any VC can be used
	  
	  if ( ( ( dir == 0 ) && ( cur > dest ) ) ||
	       ( ( dir == 1 ) && ( cur < dest ) ) ) {
	    *partition = 1;
	  } else if ( ( ( dir == 0 ) && ( cur <= (gK-1)/2 ) && ( dest >  (gK-1)/2 ) ) ||
		      ( ( dir == 1 ) && ( cur >  (gK-1)/2 ) && ( dest <= (gK-1)/2 ) ) ) {
	    *partition = 0;
	  } else {
	    *partition = RandomInt( 1 ); // use either VC set
	  }
	} else {
	  // Deterministic, fixed dateline between nodes k-1 and 0
	  
	  if ( ( ( dir == 0 ) && ( cur > dest ) ) ||
	       ( ( dir == 1 ) && ( dest < cur ) ) ) {
	    *partition = 1;
	  } else {
	    *partition = 0;
	  }
	}
      }
    } else {
      // Inverting the least significant bit keeps
      // the packet moving in the same direction
      *out_port = in_port ^ 0x1;
    }    

  } else {
    *out_port = 2*gN;  // Eject
  }
}

//=============================================================

/*
 * [한국어]
 * dim_order_mesh() - 메시 토폴로지의 Dimension-Order Routing (DOR)
 *
 * @r: 현재 라우터
 * @f: 라우팅 대상 Flit
 * @in_channel: 입구 채널 (DOR에서는 사용하지 않음)
 * @outputs: 라우팅 결과 출구 포트 집합
 * @inject: true이면 주입 단계 (out_port = -1)
 *
 * 결정론적 DOR: 항상 X→Y→Z 순서로 목적지 방향으로 이동.
 * 교착 없음(Deadlock-free): 채널 의존 그래프에 사이클이 없음.
 * 패킷 타입에 따른 VC 분리: 요청(하위 VC)과 응답(상위 VC) 분리.
 *
 * GPGPU-Sim에서 가장 많이 사용되는 기본 라우팅 알고리즘.
 *
 * 호출 체인: gRoutingFunctionMap["dor_mesh"/"dim_order_mesh"] → [이 함수]
 */
void dim_order_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int out_port = inject ? -1 : dor_next_mesh( r->GetID( ), f->dest );
  
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  if ( !inject && f->watch ) {
    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
	       << "Adding VC range [" 
	       << vcBegin << "," 
	       << vcEnd << "]"
	       << " at output port " << out_port
	       << " for flit " << f->id
	       << " (input port " << in_channel
	       << ", destination " << f->dest << ")"
	       << "." << endl;
  }
  
  outputs->Clear();

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_ni_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int out_port = inject ? -1 : dor_next_mesh( r->GetID( ), f->dest );
  
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  // at the destination router, we don't need to separate VCs by destination
  if(inject || (r->GetID() != f->dest)) {

    int const vcs_per_dest = (vcEnd - vcBegin + 1) / gNodes;
    assert(vcs_per_dest > 0);

    vcBegin += f->dest * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;

  }
  
  if( !inject && f->watch ) {
    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
	       << "Adding VC range [" 
	       << vcBegin << "," 
	       << vcEnd << "]"
	       << " at output port " << out_port
	       << " for flit " << f->id
	       << " (input port " << in_channel
	       << ", destination " << f->dest << ")"
	       << "." << endl;
  }
  
  outputs->Clear( );
  
  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_pni_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int out_port = inject ? -1 : dor_next_mesh( r->GetID(), f->dest );
  
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  if(inject || (r->GetID() != f->dest)) {
    int next_coord = f->dest;
    if(!inject) {
      int out_dim = out_port / 2;
      for(int d = 0; d < out_dim; ++d) {
	next_coord /= gK;
      }
    }
    next_coord %= gK;
    assert(next_coord >= 0 && next_coord < gK);
    int vcs_per_dest = (vcEnd - vcBegin + 1) / gK;
    assert(vcs_per_dest > 0);
    vcBegin += next_coord * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;
  }

  if( !inject && f->watch ) {
    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
	       << "Adding VC range [" 
	       << vcBegin << "," 
	       << vcEnd << "]"
	       << " at output port " << out_port
	       << " for flit " << f->id
	       << " (input port " << in_channel
	       << ", destination " << f->dest << ")"
	       << "." << endl;
  }
  
  outputs->Clear( );
  
  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

// Random intermediate in the minimal quadrant defined
// by the source and destination
int rand_min_intr_mesh( int src, int dest )
{
  int dist;

  int intm = 0;
  int offset = 1;

  for ( int n = 0; n < gN; ++n ) {
    dist = ( dest % gK ) - ( src % gK );

    if ( dist > 0 ) {
      intm += offset * ( ( src % gK ) + RandomInt( dist ) );
    } else {
      intm += offset * ( ( dest % gK ) + RandomInt( -dist ) );
    }

    offset *= gK;
    dest /= gK; src /= gK;
  }

  return intm;
}

//=============================================================

void romm_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    if ( in_channel == 2*gN ) {
      f->ph   = 0;  // Phase 0
      f->intm = rand_min_intr_mesh( f->src, f->dest );
    } 

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
    }

    out_port = dor_next_mesh( r->GetID( ), (f->ph == 0) ? f->intm : f->dest );

    // at the destination router, we don't need to separate VCs by phase
    if(r->GetID() != f->dest) {

      //each class must have at least 2 vcs assigned or else valiant valiant will deadlock
      int available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if(f->ph == 0) {
	vcEnd -= available_vcs;
      } else {
	assert(f->ph == 1);
	vcBegin += available_vcs;
      }
    }

  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void romm_ni_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  // at the destination router, we don't need to separate VCs by destination
  if(inject || (r->GetID() != f->dest)) {

    int const vcs_per_dest = (vcEnd - vcBegin + 1) / gNodes;
    assert(vcs_per_dest > 0);

    vcBegin += f->dest * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;

  }

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    if ( in_channel == 2*gN ) {
      f->ph   = 0;  // Phase 0
      f->intm = rand_min_intr_mesh( f->src, f->dest );
    } 

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
    }

    out_port = dor_next_mesh( r->GetID( ), (f->ph == 0) ? f->intm : f->dest );

  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void min_adapt_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  outputs->Clear( );
  
  if(inject) {
    // injection can use all VCs
    outputs->AddRange(-1, vcBegin, vcEnd);
    return;
  } else if(r->GetID() == f->dest) {
    // ejection can also use all VCs
    outputs->AddRange(2*gN, vcBegin, vcEnd);
    return;
  }

  int in_vc;

  if ( in_channel == 2*gN ) {
    in_vc = vcEnd; // ignore the injection VC
  } else {
    in_vc = f->vc;
  }
  
  // DOR for the escape channel (VC 0), low priority 
  int out_port = dor_next_mesh( r->GetID( ), f->dest );    
  outputs->AddRange( out_port, 0, vcBegin, vcBegin );
  
  if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		  << "Adding VC range [" 
		  << vcBegin << "," 
		  << vcBegin << "]"
		  << " at output port " << out_port
		  << " for flit " << f->id
		  << " (input port " << in_channel
		  << ", destination " << f->dest << ")"
		  << "." << endl;
   }
  
  if ( in_vc != vcBegin ) { // If not in the escape VC
    // Minimal adaptive for all other channels
    int cur = r->GetID( );
    int dest = f->dest;
    
    for ( int n = 0; n < gN; ++n ) {
      if ( ( cur % gK ) != ( dest % gK ) ) { 
	// Add minimal direction in dimension 'n'
	if ( ( cur % gK ) < ( dest % gK ) ) { // Right
	  if ( f->watch ) {
	    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
			<< "Adding VC range [" 
		       << (vcBegin+1) << "," 
			<< vcEnd << "]"
			<< " at output port " << 2*n
			<< " with priority " << 1
			<< " for flit " << f->id
			<< " (input port " << in_channel
			<< ", destination " << f->dest << ")"
			<< "." << endl;
	  }
	  outputs->AddRange( 2*n, vcBegin+1, vcEnd, 1 ); 
	} else { // Left
	  if ( f->watch ) {
	    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
			<< "Adding VC range [" 
		       << (vcBegin+1) << "," 
			<< vcEnd << "]"
			<< " at output port " << 2*n+1
			<< " with priority " << 1
			<< " for flit " << f->id
			<< " (input port " << in_channel
			<< ", destination " << f->dest << ")"
			<< "." << endl;
	  }
	  outputs->AddRange( 2*n + 1, vcBegin+1, vcEnd, 1 ); 
	}
      }
      cur  /= gK;
      dest /= gK;
    }
  } 
}

//=============================================================

void planar_adapt_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  outputs->Clear( );
  
  if(inject) {
    // injection can use all VCs
    outputs->AddRange(-1, vcBegin, vcEnd);
    return;
  }

  int cur     = r->GetID( ); 
  int dest    = f->dest;

  if ( cur != dest ) {
   
    int in_vc   = f->vc;
    int vc_mult = (vcEnd - vcBegin + 1) / 3;

    // Find the first unmatched dimension -- except
    // for when we're in the first dimension because
    // of misrouting in the last adaptive plane.
    // In this case, go to the last dimension instead.

    int n;
    for ( n = 0; n < gN; ++n ) {
      if ( ( ( cur % gK ) != ( dest % gK ) ) &&
	   !( ( in_channel/2 == 0 ) &&
	      ( n == 0 ) &&
	      ( in_vc < vcBegin+2*vc_mult ) ) ) {
	break;
      }

      cur  /= gK;
      dest /= gK;
    }

    assert( n < gN );

    if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		  << "PLANAR ADAPTIVE: flit " << f->id 
		  << " in adaptive plane " << n << "." << endl;
    }

    // We're in adaptive plane n

    // Can route productively in d_{i,2}
    bool increase;
    bool fault;
    if ( ( cur % gK ) < ( dest % gK ) ) { // Increasing
      increase = true;
      if ( !r->IsFaultyOutput( 2*n ) ) {
	outputs->AddRange( 2*n, vcBegin+2*vc_mult, vcEnd );
	fault = false;

	if ( f->watch ) {
	  *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		      << "PLANAR ADAPTIVE: increasing in dimension " << n
		      << "." << endl;
	}
      } else {
	fault = true;
      }
    } else { // Decreasing
      increase = false;
      if ( !r->IsFaultyOutput( 2*n + 1 ) ) {
	outputs->AddRange( 2*n + 1, vcBegin+2*vc_mult, vcEnd ); 
	fault = false;

	if ( f->watch ) {
	  *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		      << "PLANAR ADAPTIVE: decreasing in dimension " << n
		      << "." << endl;
	}
      } else {
	fault = true;
      }
    }
      
    n = ( n + 1 ) % gN;
    cur  /= gK;
    dest /= gK;
      
    if ( !increase ) {
      vcBegin += vc_mult;
    }
    vcEnd = vcBegin + vc_mult - 1;
      
    int d1_min_c;
    if ( ( cur % gK ) < ( dest % gK ) ) { // Increasing in d_{i+1}
      d1_min_c = 2*n;
    } else if ( ( cur % gK ) != ( dest % gK ) ) {  // Decreasing in d_{i+1}
      d1_min_c = 2*n + 1;
    } else {
      d1_min_c = -1;
    }
      
    // do we want to 180?  if so, the last
    // route was a misroute in this dimension,
    // if there is no fault in d_i, just ignore
    // this dimension, otherwise continue to misroute
    if ( d1_min_c == in_channel ) { 
      if ( fault ) {
	d1_min_c = in_channel ^ 1;
      } else {
	d1_min_c = -1;
      }

      if ( f->watch ) {
	*gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		    << "PLANAR ADAPTIVE: avoiding 180 in dimension " << n
		    << "." << endl;
      }
    }
      
    if ( d1_min_c != -1 ) {
      if ( !r->IsFaultyOutput( d1_min_c ) ) {
	outputs->AddRange( d1_min_c, vcBegin, vcEnd );
      } else if ( fault ) {
	// major problem ... fault in d_i and d_{i+1}
	r->Error( "There seem to be faults in d_i and d_{i+1}" );
      }
    } else if ( fault ) { // need to misroute!
      bool atedge;
      if ( cur % gK == 0 ) {
	d1_min_c = 2*n;
	atedge = true;
      } else if ( cur % gK == gK - 1 ) {
	d1_min_c = 2*n + 1;
	atedge = true;
      } else {
	d1_min_c = 2*n + RandomInt( 1 ); // random misroute

	if ( d1_min_c  == in_channel ) { // don't 180
	  d1_min_c = in_channel ^ 1;
	}
	atedge = false;
      }
      
      if ( !r->IsFaultyOutput( d1_min_c ) ) {
	outputs->AddRange( d1_min_c, vcBegin, vcEnd );
      } else if ( !atedge && !r->IsFaultyOutput( d1_min_c ^ 1 ) ) {
	outputs->AddRange( d1_min_c ^ 1, vcBegin, vcEnd );
      } else {
	// major problem ... fault in d_i and d_{i+1}
	r->Error( "There seem to be faults in d_i and d_{i+1}" );
      }
    }
  } else {
    outputs->AddRange( 2*gN, vcBegin, vcEnd ); 
  }
}

//=============================================================
/*
  FIXME: This is broken (note that f->dr is never actually modified).
  Even if it were, this should really use f->ph instead of introducing a single-
  use field.

void limited_adapt_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  outputs->Clear( );

  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  if ( inject ) {
    outputs->AddRange( -1, vcBegin, vcEnd - 1 );
    f->dr = 0; // zero dimension reversals
    return;
  }

  int cur = r->GetID( );
  int dest = f->dest;
  
  if ( cur != dest ) {
    if ( ( f->vc != vcEnd ) && 
	 ( f->dr != vcEnd - 1 ) ) {
      
      for ( int n = 0; n < gN; ++n ) {
	if ( ( cur % gK ) != ( dest % gK ) ) { 
	  int min_port;
	  if ( ( cur % gK ) < ( dest % gK ) ) { 
	    min_port = 2*n; // Right
	  } else {
	    min_port = 2*n + 1; // Left
	  }
	  
	  // Go in a productive direction with high priority
	  outputs->AddRange( min_port, vcBegin, vcEnd - 1, 2 );
	  
	  // Go in the non-productive direction with low priority
	  outputs->AddRange( min_port ^ 0x1, vcBegin, vcEnd - 1, 1 );
	} else {
	  // Both directions are non-productive
	  outputs->AddRange( 2*n, vcBegin, vcEnd - 1, 1 );
	  outputs->AddRange( 2*n+1, vcBegin, vcEnd - 1, 1 );
	}
	
	cur  /= gK;
	dest /= gK;
      }
      
    } else {
      outputs->AddRange( dor_next_mesh( cur, dest ),
			 vcEnd, vcEnd, 0 );
    }
    
  } else { // at destination
    outputs->AddRange( 2*gN, vcBegin, vcEnd ); 
  }
}
*/
//=============================================================

/*
 * [한국어]
 * valiant_mesh() - Valiant 부하 분산 라우팅 (메시 토폴로지)
 *
 * @r: 현재 라우터
 * @f: 라우팅 대상 Flit (ph, intm 필드 사용)
 * @in_channel: 입구 채널
 * @outputs: 라우팅 결과 출구 포트 집합
 * @inject: true이면 주입 단계
 *
 * 2단계 라우팅:
 *   Phase 0 (f->ph == 0): 랜덤 중간 노드(intm)로 DOR 이동
 *   Phase 1 (f->ph == 1): 실제 목적지(dest)로 DOR 이동
 *
 * 장점: 트래픽 부하를 랜덤 중간 노드를 통해 전체 메시에 균등 분산.
 * 단점: 경로가 최소 경로의 최대 2배. VC를 phase별로 분리하여 데드락 방지.
 *   phase 0 → 하위 VC (vcBegin .. vcBegin+available_vcs-1)
 *   phase 1 → 상위 VC (vcBegin+available_vcs .. vcEnd)
 *
 * mutable 필드: f->ph, f->intm 가 라우팅 중 변경됨 (const Flit* 이지만 mutable).
 *
 * 호출 체인: gRoutingFunctionMap["valiant_mesh"] → [이 함수]
 */
void valiant_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    if ( in_channel == 2*gN ) {
      f->ph   = 0;  // Phase 0
      f->intm = RandomInt( gNodes - 1 );
    }

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
    }

    out_port = dor_next_mesh( r->GetID( ), (f->ph == 0) ? f->intm : f->dest );

    // at the destination router, we don't need to separate VCs by phase
    if(r->GetID() != f->dest) {

      //each class must have at least 2 vcs assigned or else valiant valiant will deadlock
      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if(f->ph == 0) {
	vcEnd -= available_vcs;
      } else {
	assert(f->ph == 1);
	vcBegin += available_vcs;
      }
    }

  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void valiant_torus( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int phase;
    if ( in_channel == 2*gN ) {
      phase   = 0;  // Phase 0
      f->intm = RandomInt( gNodes - 1 );
    } else {
      phase = f->ph / 2;
    }

    if ( ( phase == 0 ) && ( r->GetID( ) == f->intm ) ) {
      phase = 1; // Go to phase 1
      in_channel = 2*gN; // ensures correct vc selection at the beginning of phase 2
    }
  
    int ring_part;
    dor_next_torus( r->GetID( ), (phase == 0) ? f->intm : f->dest, in_channel,
		    &out_port, &ring_part, false );

    f->ph = 2 * phase + ring_part;

    // at the destination router, we don't need to separate VCs by phase, etc.
    if(r->GetID() != f->dest) {

      int const ring_available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(ring_available_vcs > 0);

      if(ring_part == 0) {
	vcEnd -= ring_available_vcs;
      } else {
	assert(ring_part == 1);
	vcBegin += ring_available_vcs;
      }

      int const ph_available_vcs = ring_available_vcs / 2;
      assert(ph_available_vcs > 0);

      if(phase == 0) {
	vcEnd -= ph_available_vcs;
      } else {
	assert(phase == 1);
	vcBegin += ph_available_vcs;
      }
    }

  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void valiant_ni_torus( const Router *r, const Flit *f, int in_channel, 
		       OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  // at the destination router, we don't need to separate VCs by destination
  if(inject || (r->GetID() != f->dest)) {

    int const vcs_per_dest = (vcEnd - vcBegin + 1) / gNodes;
    assert(vcs_per_dest > 0);

    vcBegin += f->dest * vcs_per_dest;
    vcEnd = vcBegin + vcs_per_dest - 1;

  }

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int phase;
    if ( in_channel == 2*gN ) {
      phase   = 0;  // Phase 0
      f->intm = RandomInt( gNodes - 1 );
    } else {
      phase = f->ph / 2;
    }

    if ( ( f->ph == 0 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 1; // Go to phase 1
      in_channel = 2*gN; // ensures correct vc selection at the beginning of phase 2
    }
  
    int ring_part;
    dor_next_torus( r->GetID( ), (f->ph == 0) ? f->intm : f->dest, in_channel,
		    &out_port, &ring_part, false );

    f->ph = 2 * phase + ring_part;

    // at the destination router, we don't need to separate VCs by phase, etc.
    if(r->GetID() != f->dest) {

      int const ring_available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(ring_available_vcs > 0);

      if(ring_part == 0) {
	vcEnd -= ring_available_vcs;
      } else {
	assert(ring_part == 1);
	vcBegin += ring_available_vcs;
      }

      int const ph_available_vcs = ring_available_vcs / 2;
      assert(ph_available_vcs > 0);

      if(phase == 0) {
	vcEnd -= ph_available_vcs;
      } else {
	assert(phase == 1);
	vcBegin += ph_available_vcs;
      }
    }

    if (f->watch) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dest << ")"
		 << "." << endl;
    }

  }
  
  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_torus( const Router *r, const Flit *f, int in_channel, 
		      OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {
    
    int cur  = r->GetID( );
    int dest = f->dest;

    dor_next_torus( cur, dest, in_channel,
		    &out_port, &f->ph, false );


    // at the destination router, we don't need to separate VCs by ring partition
    if(cur != dest) {

      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if ( f->ph == 0 ) {
	vcEnd -= available_vcs;
      } else {
	vcBegin += available_vcs;
      } 
    }

    if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dest << ")"
		 << "." << endl;
    }

  }
 
  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_ni_torus( const Router *r, const Flit *f, int in_channel, 
			 OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {
    
    int cur  = r->GetID( );
    int dest = f->dest;

    dor_next_torus( cur, dest, in_channel,
		    &out_port, NULL, false );

    // at the destination router, we don't need to separate VCs by destination
    if(cur != dest) {

      int const vcs_per_dest = (vcEnd - vcBegin + 1) / gNodes;
      assert(vcs_per_dest);

      vcBegin += f->dest * vcs_per_dest;
      vcEnd = vcBegin + vcs_per_dest - 1;

    }

    if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dest << ")"
		 << "." << endl;
    }

  }
  
  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void dim_order_bal_torus( const Router *r, const Flit *f, int in_channel, 
			  OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int cur  = r->GetID( );
    int dest = f->dest;

    dor_next_torus( cur, dest, in_channel,
		    &out_port, &f->ph, true );

    // at the destination router, we don't need to separate VCs by ring partition
    if(cur != dest) {

      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if ( f->ph == 0 ) {
	vcEnd -= available_vcs;
      } else {
	assert(f->ph == 1);
	vcBegin += available_vcs;
      } 
    }

    if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
		 << "Adding VC range [" 
		 << vcBegin << "," 
		 << vcEnd << "]"
		 << " at output port " << out_port
		 << " for flit " << f->id
		 << " (input port " << in_channel
		 << ", destination " << f->dest << ")"
		 << "." << endl;
    }

  }
  
  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}

//=============================================================

void min_adapt_torus( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  outputs->Clear( );

  if(inject) {
    // injection can use all VCs
    outputs->AddRange(-1, vcBegin, vcEnd);
    return;
  } else if(r->GetID() == f->dest) {
    // ejection can also use all VCs
    outputs->AddRange(2*gN, vcBegin, vcEnd);
  }

  int in_vc;
  if ( in_channel == 2*gN ) {
    in_vc = vcEnd; // ignore the injection VC
  } else {
    in_vc = f->vc;
  }
  
  int cur = r->GetID( );
  int dest = f->dest;

  int out_port;

  if ( in_vc > ( vcBegin + 1 ) ) { // If not in the escape VCs
    // Minimal adaptive for all other channels
    
    for ( int n = 0; n < gN; ++n ) {
      if ( ( cur % gK ) != ( dest % gK ) ) {
	int dist2 = gK - 2 * ( ( ( dest % gK ) - ( cur % gK ) + gK ) % gK );
	
	if ( dist2 > 0 ) { /*) || 
			     ( ( dist2 == 0 ) && ( RandomInt( 1 ) ) ) ) {*/
	  outputs->AddRange( 2*n, vcBegin+3, vcBegin+3, 1 ); // Right
	} else {
	  outputs->AddRange( 2*n + 1, vcBegin+3, vcBegin+3, 1 ); // Left
	}
      }

      cur  /= gK;
      dest /= gK;
    }
    
    // DOR for the escape channel (VCs 0-1), low priority --- 
    // trick the algorithm with the in channel.  want VC assignment
    // as if we had injected at this node
    dor_next_torus( r->GetID( ), f->dest, 2*gN,
		    &out_port, &f->ph, false );
  } else {
    // DOR for the escape channel (VCs 0-1), low priority 
    dor_next_torus( cur, dest, in_channel,
		    &out_port, &f->ph, false );
  }

  if ( f->ph == 0 ) {
    outputs->AddRange( out_port, vcBegin, vcBegin, 0 );
  } else  {
    outputs->AddRange( out_port, vcBegin+1, vcBegin+1, 0 );
  } 
}

//=============================================================

void dest_tag_fly( const Router *r, const Flit *f, int in_channel, 
		   OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    int stage = ( r->GetID( ) * gK ) / gNodes;
    int dest  = f->dest;

    while( stage < ( gN - 1 ) ) {
      dest /= gK;
      ++stage;
    }

    out_port = dest % gK;
  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}



//=============================================================

void chaos_torus( const Router *r, const Flit *f, 
		  int in_channel, OutputSet *outputs, bool inject )
{
  outputs->Clear( );

  if(inject) {
    outputs->AddRange(-1, 0, 0);
    return;
  }

  int cur = r->GetID( );
  int dest = f->dest;
  
  if ( cur != dest ) {
    for ( int n = 0; n < gN; ++n ) {

      if ( ( cur % gK ) != ( dest % gK ) ) { 
	int dist2 = gK - 2 * ( ( ( dest % gK ) - ( cur % gK ) + gK ) % gK );
      
	if ( dist2 >= 0 ) {
	  outputs->AddRange( 2*n, 0, 0 ); // Right
	} 
	
	if ( dist2 <= 0 ) {
	  outputs->AddRange( 2*n + 1, 0, 0 ); // Left
	}
      }

      cur  /= gK;
      dest /= gK;
    }
  } else {
    outputs->AddRange( 2*gN, 0, 0 ); 
  }
}


//=============================================================

void chaos_mesh( const Router *r, const Flit *f, 
		  int in_channel, OutputSet *outputs, bool inject )
{
  outputs->Clear( );

  if(inject) {
    outputs->AddRange(-1, 0, 0);
    return;
  }

  int cur = r->GetID( );
  int dest = f->dest;
  
  if ( cur != dest ) {
    for ( int n = 0; n < gN; ++n ) {
      if ( ( cur % gK ) != ( dest % gK ) ) { 
	// Add minimal direction in dimension 'n'
	if ( ( cur % gK ) < ( dest % gK ) ) { // Right
	  outputs->AddRange( 2*n, 0, 0 ); 
	} else { // Left
	  outputs->AddRange( 2*n + 1, 0, 0 ); 
	}
      }
      cur  /= gK;
      dest /= gK;
    }
  } else {
    outputs->AddRange( 2*gN, 0, 0 ); 
  }
}

//=============================================================

/*
 * [한국어]
 * InitializeRoutingMap() - VC 범위 설정 및 모든 라우팅 함수를 등록한다
 *
 * @config: BookSim 설정 파일 파싱 결과
 *
 * 동작 순서:
 *   1. gNumVCs = num_vcs 설정값 읽기
 *   2. 타입별 VC 범위 설정 (읽기 요청/쓰기 요청/읽기 응답/쓰기 응답)
 *      - 음수(-1)이면 기본값 사용:
 *        요청: 0 .. gNumVCs/2-1
 *        응답: gNumVCs/2 .. gNumVCs-1
 *   3. gRoutingFunctionMap에 모든 알고리즘 등록
 *
 * VC 분리의 중요성:
 *   요청 패킷과 응답 패킷이 같은 VC를 사용하면 데드락 가능.
 *   (응답이 요청 버퍼를 점유 → 요청 진행 불가 → 순환 의존)
 *   VC를 분리하면 응답이 별도 VC에서 독립적으로 진행됨.
 *
 * 등록되는 주요 알고리즘:
 *   "dor_mesh" / "dim_order_mesh": 결정론적 DOR (GPGPU-Sim 기본)
 *   "valiant_mesh": Valiant 부하 분산
 *   "xy_yx_mesh": 두 방향 DOR 중 하나 선택 (oblivious)
 *   "adaptive_xy_yx_mesh": 부하 기반 적응형 XY/YX 선택
 *   "min_adapt_mesh": 최소 경로 적응형 + DOR 이탈 채널
 *   "dim_order_torus": 토러스 DOR (데이트라인 VC 분리)
 *   "nca_fattree", "anca_fattree": Fat-tree NCA 라우팅
 *
 * 호출 체인: InterconnectInterface::CreateInterconnect() → [이 함수]
 */
void InitializeRoutingMap( const Configuration & config )
{

  gNumVCs = config.GetInt( "num_vcs" ); // [한국어] 총 VC 수 설정

  //
  // traffic class partitions
  // [한국어] 패킷 타입별 VC 범위 설정 (요청/응답 분리로 데드락 방지)
  //
  gReadReqBeginVC    = config.GetInt("read_request_begin_vc"); // [한국어] 읽기 요청 VC 시작
  if(gReadReqBeginVC < 0) { // [한국어] 설정 없으면(-1) 기본값 0 사용
    gReadReqBeginVC = 0;
  }
  gReadReqEndVC      = config.GetInt("read_request_end_vc"); // [한국어] 읽기 요청 VC 끝
  if(gReadReqEndVC < 0) { // [한국어] 기본값: 전체 VC의 하위 절반 끝 (0..gNumVCs/2-1)
    gReadReqEndVC = gNumVCs / 2 - 1;
  }
  gWriteReqBeginVC   = config.GetInt("write_request_begin_vc"); // [한국어] 쓰기 요청 VC 시작
  if(gWriteReqBeginVC < 0) { // [한국어] 기본값 0 (읽기 요청과 VC 공유)
    gWriteReqBeginVC = 0;
  }
  gWriteReqEndVC     = config.GetInt("write_request_end_vc"); // [한국어] 쓰기 요청 VC 끝
  if(gWriteReqEndVC < 0) { // [한국어] 기본값: gNumVCs/2-1 (읽기 요청과 동일 VC 풀)
    gWriteReqEndVC = gNumVCs / 2 - 1;
  }
  gReadReplyBeginVC  = config.GetInt("read_reply_begin_vc"); // [한국어] 읽기 응답 VC 시작
  if(gReadReplyBeginVC < 0) { // [한국어] 기본값: 전체 VC의 상위 절반 시작
    gReadReplyBeginVC = gNumVCs / 2;
  }
  gReadReplyEndVC    = config.GetInt("read_reply_end_vc"); // [한국어] 읽기 응답 VC 끝
  if(gReadReplyEndVC < 0) { // [한국어] 기본값: 마지막 VC
    gReadReplyEndVC = gNumVCs - 1;
  }
  gWriteReplyBeginVC = config.GetInt("write_reply_begin_vc"); // [한국어] 쓰기 응답 VC 시작
  if(gWriteReplyBeginVC < 0) { // [한국어] 기본값: gNumVCs/2 (읽기 응답과 동일)
    gWriteReplyBeginVC = gNumVCs / 2;
  }
  gWriteReplyEndVC   = config.GetInt("write_reply_end_vc"); // [한국어] 쓰기 응답 VC 끝
  if(gWriteReplyEndVC < 0) { // [한국어] 기본값: gNumVCs-1
    gWriteReplyEndVC = gNumVCs - 1;
  }

  /* Register routing functions here */
  // [한국어] 모든 라우팅 알고리즘을 이름 → 함수 포인터 맵에 등록

  // ===================================================
  // Balfour-Schultz
  // [한국어] Balfour-Schultz 방식 라우팅 함수 등록 (VC 타입 분리 지원)
  gRoutingFunctionMap["nca_fattree"]         = &fattree_nca;         // [한국어] Fat-tree NCA 랜덤 라우팅
  gRoutingFunctionMap["anca_fattree"]        = &fattree_anca;        // [한국어] Fat-tree NCA 적응형 라우팅
  gRoutingFunctionMap["nca_qtree"]           = &qtree_nca;           // [한국어] QTree NCA 라우팅
  gRoutingFunctionMap["nca_tree4"]           = &tree4_nca;           // [한국어] Tree4 NCA 랜덤 라우팅
  gRoutingFunctionMap["anca_tree4"]          = &tree4_anca;          // [한국어] Tree4 NCA 적응형 라우팅
  gRoutingFunctionMap["dor_mesh"]            = &dim_order_mesh;      // [한국어] 메시 DOR (GPGPU-Sim 기본값)
  gRoutingFunctionMap["xy_yx_mesh"]          = &xy_yx_mesh;          // [한국어] oblivious XY/YX 혼합
  gRoutingFunctionMap["adaptive_xy_yx_mesh"] = &adaptive_xy_yx_mesh; // [한국어] 부하 기반 적응형 XY/YX
  // End Balfour-Schultz
  // ===================================================

  gRoutingFunctionMap["dim_order_mesh"]      = &dim_order_mesh;      // [한국어] 메시 DOR (dor_mesh와 동일)
  gRoutingFunctionMap["dim_order_ni_mesh"]   = &dim_order_ni_mesh;   // [한국어] 메시 DOR + 목적지별 VC 분리
  gRoutingFunctionMap["dim_order_pni_mesh"]  = &dim_order_pni_mesh;  // [한국어] 메시 DOR + 다음 좌표 기반 VC
  gRoutingFunctionMap["dim_order_torus"]     = &dim_order_torus;     // [한국어] 토러스 DOR (데이트라인 VC 분리)
  gRoutingFunctionMap["dim_order_ni_torus"]  = &dim_order_ni_torus;  // [한국어] 토러스 DOR + 목적지 VC
  gRoutingFunctionMap["dim_order_bal_torus"] = &dim_order_bal_torus; // [한국어] 토러스 DOR + 균형 파티션

  gRoutingFunctionMap["romm_mesh"]    = &romm_mesh;    // [한국어] ROMM: 랜덤 최소 경로 메시 (Valiant 변형)
  gRoutingFunctionMap["romm_ni_mesh"] = &romm_ni_mesh; // [한국어] ROMM + 목적지 VC 분리

  gRoutingFunctionMap["min_adapt_mesh"]  = &min_adapt_mesh;  // [한국어] 최소 경로 적응형 + DOR 이탈
  gRoutingFunctionMap["min_adapt_torus"] = &min_adapt_torus; // [한국어] 최소 경로 적응형 토러스

  gRoutingFunctionMap["planar_adapt_mesh"] = &planar_adapt_mesh; // [한국어] 평면 적응형 메시 라우팅

  // FIXME: This is broken.
  //  gRoutingFunctionMap["limited_adapt_mesh"] = &limited_adapt_mesh;

  gRoutingFunctionMap["valiant_mesh"]     = &valiant_mesh;     // [한국어] Valiant 2단계 랜덤 라우팅 (메시)
  gRoutingFunctionMap["valiant_torus"]    = &valiant_torus;    // [한국어] Valiant 2단계 랜덤 라우팅 (토러스)
  gRoutingFunctionMap["valiant_ni_torus"] = &valiant_ni_torus; // [한국어] Valiant 토러스 + 목적지 VC

  gRoutingFunctionMap["dest_tag_fly"] = &dest_tag_fly; // [한국어] Butterfly 네트워크용 목적지 태그 라우팅

  gRoutingFunctionMap["chaos_mesh"]  = &chaos_mesh;  // [한국어] Chaos 라우터용 메시 라우팅 (연구용)
  gRoutingFunctionMap["chaos_torus"] = &chaos_torus; // [한국어] Chaos 라우터용 토러스 라우팅
}
