/*
 * [한국어 설명] ICNT 네트워크 트래픽 분류 및 통계 구현 (traffic_breakdown.cc)
 *
 * === 파일의 역할 ===
 * traffic_breakdown 클래스의 세 메서드(print, record_traffic, classify_memfetch)를
 * 구현한다. GPU SM과 메모리 파티션 사이의 ICNT(intersim2) 네트워크를 통과하는
 * 메모리 요청을 유형별로 분류하고, 전송된 바이트와 패킷 수를 집계한다.
 * AerialVision 리포트에서 네트워크 병목 유형을 식별하는 데 활용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: SM의 메모리 요청 → ICNT 주입 시 record_traffic() 호출
 *   → classify_memfetch()로 유형 분류 → m_stats에 카운터 누적
 *   → 시뮬레이션 종료 시 print()로 파일 출력
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내.
 * 사이클-레벨 함수가 아님 — ICNT 주입/추출 이벤트 단위 호출.
 *
 * === 타 모듈과의 연결 ===
 * 의존: traffic_breakdown.h (클래스 선언), mem_fetch.h (mem_access_type 열거형,
 *       mem_access_type_str(), get_access_type(), isatomic())
 * 사용처: ICNT 인터페이스 코드 (record_traffic 호출),
 *         gpgpu_sim::print_stats() (print 호출)
 * 공유 자료구조: mem_fetch::mem_access_type (GLOBAL_ACC_R, CONST_ACC_R 등)
 *
 * === 주요 함수/구조체 요약 ===
 * print()             - m_stats 전체 순회, 유형별 총 바이트 + 크기 분포 출력
 * record_traffic()    - classify_memfetch()로 유형 결정, m_stats[type][size]++ 누적
 * classify_memfetch() - mem_access_type switch → 트래픽 유형 문자열 변환
 *                       (GLOBAL_ACC_R+isatomic() → "GLOBAL_ATOMIC" 특수 처리)
 */

#include "traffic_breakdown.h" /* [한국어] traffic_breakdown 클래스, 타입 별칭 선언 */
#include "mem_fetch.h"         /* [한국어] mem_access_type 열거형, mem_access_type_str(), isatomic() */

/*
 * [한국어]
 * traffic_breakdown::print() - 트래픽 통계를 FILE*에 출력
 *
 * @fout: 출력 대상 FILE 포인터 (stdout 또는 stats 파일)
 *
 * m_stats(유형 → 크기 → 카운터 이중 맵)를 외부 루프(유형)와 내부 루프(크기)로 순회.
 * 각 유형에 대해:
 *   1. 총 전송 바이트 = Σ(크기 × 패킷 수) 계산
 *   2. "traffic_breakdown_<네트워크>[<유형>] = <총바이트> {<크기>:<수>,...}" 형식 출력
 * 실행 컨텍스트: 시뮬레이션 종료 후 단일 스레드.
 *
 * 호출 체인:
 *   gpgpu_sim::print_stats() → [print(fout)]
 */
void traffic_breakdown::print(FILE* fout) {
  for (traffic_stat_t::const_iterator i_stat = m_stats.begin();
       i_stat != m_stats.end(); i_stat++) {
    /* [한국어] 외부 루프: 트래픽 유형 문자열(예: "GLOBAL_ACC_R") 순회 */
    unsigned int byte_transferred = 0; /* [한국어] 이 유형의 총 전송 바이트 누산자 */
    for (traffic_class_t::const_iterator i_class = i_stat->second.begin();
         i_class != i_stat->second.end(); i_class++) {
      /* [한국어] 내부 루프: 이 유형 내 패킷 크기별 순회 */
      byte_transferred +=
          i_class->first * i_class->second;  // byte/packet x #packets
      /* [한국어] 총 바이트 = 패킷 크기(first) × 해당 크기 패킷 수(second) 누적 */
    }
    fprintf(fout, "traffic_breakdown_%s[%s] = %u {", m_network_name.c_str(),
            i_stat->first.c_str(), byte_transferred);
    /* [한국어] 형식: "traffic_breakdown_<네트워크명>[<유형>] = <총바이트> {"
     * 예: "traffic_breakdown_Interconnect0[GLOBAL_ACC_R] = 32768 {" */
    for (traffic_class_t::const_iterator i_class = i_stat->second.begin();
         i_class != i_stat->second.end(); i_class++) {
      /* [한국어] 크기 분포 목록 출력 */
      fprintf(fout, "%u:%u,", i_class->first, i_class->second);
      /* [한국어] 형식: "<패킷크기>:<패킷수>," 예: "32:1024,128:256," */
    }
    fprintf(fout, "}\n"); /* [한국어] 크기 분포 목록 닫기 */
  }
}

/*
 * [한국어]
 * traffic_breakdown::record_traffic() - mem_fetch 1건 ICNT 주입 시 통계 기록
 *
 * @mf:   ICNT에 주입되는 메모리 요청 객체 (유형 분류에 사용)
 * @size: 실제 전송 패킷 크기 (바이트, ICNT 헤더 포함 실제 전송 단위)
 *
 * classify_memfetch(mf)로 유형 문자열 결정 후
 * m_stats[type][size] += 1로 카운터 1 증가.
 * std::map의 operator[]는 키 부재 시 기본값(0)으로 자동 삽입 — 초기화 불필요.
 * ICNT push/inject 코드에서 mem_fetch가 네트워크에 주입될 때마다 호출.
 *
 * 호출 체인:
 *   icnt_push() 또는 shader.cc ICNT 주입 코드 → [record_traffic(mf, size)]
 *     → classify_memfetch(mf) → m_stats[type][size]++
 */
void traffic_breakdown::record_traffic(class mem_fetch* mf, unsigned int size) {
  m_stats[classify_memfetch(mf)][size] += 1;
  /* [한국어] 이중 맵 접근: 유형 문자열 키 → 크기 키 → 카운터 1 증가.
   * std::map operator[]는 키 없을 때 기본값(0)으로 자동 생성. */
}

/*
 * [한국어]
 * traffic_breakdown::classify_memfetch() - mem_fetch → 트래픽 유형 문자열 변환
 *
 * @mf:     분류할 메모리 요청 객체
 * @return: 트래픽 유형 문자열 (mem_access_type_str() 반환값 또는 "GLOBAL_ATOMIC")
 *
 * mf->get_access_type()으로 mem_access_type 열거형 값을 가져와 switch로 분류.
 * 대부분 유형: mem_access_type_str(access_type)으로 표준 문자열 반환.
 * GLOBAL_ACC_R 특수 처리: mf->isatomic() 여부로 "GLOBAL_ATOMIC" vs "GLOBAL_ACC_R" 구분.
 *   → Global atomic 연산(atomicAdd 등)은 읽기+쓰기 혼합 특성이 있어 별도 분류.
 * 알 수 없는 유형: assert(0)으로 컴파일 메시지와 함께 종료 (프로그래밍 오류 감지).
 *
 * 호출 체인:
 *   record_traffic(mf, size) → [classify_memfetch(mf)]
 */
std::string traffic_breakdown::classify_memfetch(class mem_fetch* mf) {
  std::string traffic_name; /* [한국어] 반환할 트래픽 유형 문자열 */

  enum mem_access_type access_type = mf->get_access_type();
  /* [한국어] mem_fetch에서 메모리 접근 유형 추출 (GLOBAL_ACC_R, CONST_ACC_R 등) */

  switch (access_type) {
    case CONST_ACC_R:       /* [한국어] 상수 메모리 읽기 */
    case TEXTURE_ACC_R:     /* [한국어] 텍스처 메모리 읽기 */
    case GLOBAL_ACC_W:      /* [한국어] 글로벌 메모리 쓰기 */
    case LOCAL_ACC_R:       /* [한국어] 로컬(스택) 메모리 읽기 */
    case LOCAL_ACC_W:       /* [한국어] 로컬 메모리 쓰기 */
    case INST_ACC_R:        /* [한국어] 명령어 캐시 읽기 (I-cache fetch) */
    case L1_WRBK_ACC:       /* [한국어] L1 캐시 라인 write-back */
    case L2_WRBK_ACC:       /* [한국어] L2 캐시 라인 write-back */
    case L1_WR_ALLOC_R:     /* [한국어] L1 write-allocate 읽기 */
    case L2_WR_ALLOC_R:     /* [한국어] L2 write-allocate 읽기 */
      traffic_name = mem_access_type_str(access_type);
      /* [한국어] 표준 유형: mem_access_type_str()이 반환하는 열거형 이름 문자열 그대로 사용 */
      break;
    case GLOBAL_ACC_R:
      // check for global atomic operation
      /* [한국어] 글로벌 읽기는 atomic 여부에 따라 분리 분류:
       *   isatomic() == true  → "GLOBAL_ATOMIC" (atomicAdd, atomicCAS 등)
       *   isatomic() == false → "GLOBAL_ACC_R" (일반 글로벌 로드) */
      traffic_name = (mf->isatomic()) ? "GLOBAL_ATOMIC"
                                      : mem_access_type_str(GLOBAL_ACC_R);
      break;
    default:
      assert(0 && "Unknown traffic type");
      /* [한국어] 처리되지 않은 mem_access_type → 프로그래밍 오류, assert로 종료 */
  }
  return traffic_name; /* [한국어] 결정된 트래픽 유형 문자열 반환 */
}
