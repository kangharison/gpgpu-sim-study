// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

/*
 * [한국어 설명] 메모리 뱅크/셋 인덱싱 해시 함수 구현 (hashing.cc)
 *
 * === 파일의 역할 ===
 * GPU DRAM 뱅크 및 L2 캐시 셋 인덱스를 비선형 방식으로 계산하는 해시 함수 3종을
 * 구현한다. GPU 프로그램에서는 coalesced 접근 이후에도 stride 패턴이 남아 있어
 * 특정 뱅크로 요청이 집중되는 뱅크 충돌(bank conflict)이 발생하기 쉽다.
 * 이 파일은 수학적으로 설계된 해시 함수로 주소를 뱅크에 균등하게 분산시켜
 * DRAM 대역폭 활용률을 높이고 메모리 레이턴시를 줄이는 역할을 한다.
 * 사용할 함수는 gpgpusim.config의 memory_partition_indexing 옵션으로 선택된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * addrdec.cc의 linear_to_raw_address_translation()이 물리 주소를 채널·뱅크·행·열로
 * 분해하는 과정에서, 뱅크 인덱스를 확정하기 직전 단계에서 이 파일의 함수를 호출한다.
 * 반환된 해시값은 뱅크 번호로 DRAM 컨트롤러(dram.cc)에 전달되며, L2 캐시 셋을
 * 결정하는 데에도 동일한 방식이 적용된다.
 * 실행 컨텍스트: 호스트 유저스페이스 — 매 L2 미스 및 DRAM 접근마다 사이클 루프 내에서 호출.
 * 재진입 가능(전역 상태 없음), 단일 스레드 가정.
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: abstract_hardware_model.h (new_addr_type 정의),
 *   gpu-cache.h (캐시 파라미터 타입), math.h/string.h (표준 라이브러리)
 * - 이 파일에 의존하는 모듈: addrdec.cc — linear_to_raw_address_translation()에서 독점 호출
 * - 데이터 흐름: 물리 주소에서 추출한 higher_bits + 초기 index → 해시 함수 →
 *   재배치된 뱅크/셋 인덱스 → addrdec.cc → dram.cc / gpu-cache.cc
 * - 설정 연동: memory_partition_indexing = 1(bitwise), 2(ipoly), 3(PAE) 중 택 1
 *
 * === 주요 함수/구조체 요약 ===
 * - ipoly_hash_function: GF(2) 기약 다항식 기반 해시, 16/32/64 뱅크 지원.
 *   std::bitset을 사용하여 각 출력 비트를 입력 비트들의 XOR로 계산한다.
 * - bitwise_hash_function: index XOR (higher_bits & mask) 단순 XOR 해시, 1줄 구현.
 * - PAE_hash_function: 페이지 비트와 뱅크 비트를 혼합하는 엔트로피 기반 해시,
 *   32 뱅크 전용.
 */

/* [한국어] math.h — 수학 함수 포함 (현재 이 파일에서 직접 사용하지 않으나 포함됨) */
#include <math.h>
/* [한국어] string.h — 문자열 함수 포함 (현재 이 파일에서 직접 사용하지 않으나 포함됨) */
#include <string.h>
/* [한국어] abstract_hardware_model.h — new_addr_type (GPU 64비트 주소 타입) 정의 포함 */
#include "../abstract_hardware_model.h"
/* [한국어] gpu-cache.h — 캐시 파라미터 관련 타입 및 상수 포함 */
#include "gpu-cache.h"

/*
 * [한국어]
 * ipoly_hash_function - IPOLY GF(2) 기약 다항식 기반 비선형 뱅크/셋 인덱스 해시
 *
 * @higher_bits: 물리 주소에서 뱅크 인덱스 계산에 사용하는 상위 비트들 (64비트 전체).
 *               addrdec.cc에서 캐시 라인 오프셋 비트를 제거한 후 넘겨진 값이다.
 * @index: 단순 modulo 방식으로 계산된 초기 뱅크 인덱스.
 *         이 값의 각 비트를 higher_bits의 특정 비트들과 XOR하여 새 인덱스를 만든다.
 * @bank_set_num: 전체 뱅크(또는 L2 셋) 수. 16, 32, 64 중 하나여야 한다.
 *                그 외 값이면 assert로 시뮬레이터가 즉시 종료된다.
 * @return: GF(2) 다항식 해시로 재배치된 뱅크/셋 인덱스 (0 이상 bank_set_num 미만).
 *
 * IPOLY 해시는 GF(2)(이진 유한체) 위의 기약 다항식 모듈러 연산을 H-행렬 형태로
 * 사전 계산한 결과이다. 각 출력 비트 new_index[i]는 입력 주소 비트 a[j]들과 기존
 * 인덱스 비트 b[k]의 XOR 선형 결합으로 표현된다. 이 조합은 모든 2^n 스트라이드
 * 패턴에서 뱅크 충돌이 발생하지 않음을 수학적으로 보장한다.
 * std::bitset을 사용하여 64비트 주소 각 비트에 쉽게 접근하고, 결과를 to_ulong()으로
 * 변환하여 반환한다.
 *
 * 호출 체인: addrdec.cc::linear_to_raw_address_translation() → [ipoly_hash_function]
 */
unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
                             unsigned bank_set_num) {
  /*
   * Set Indexing function from "Pseudo-randomly interleaved memory."
   * Rau, B. R et al.
   * ISCA 1991
   * http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
   *
   * equations are corresponding to IPOLY(37) and are adopted from:
   * "Sacat: streaming-aware conflict-avoiding thrashing-resistant gpgpu
   * cache management scheme." Khairy et al. IEEE TPDS 2017.
   *
   * equations for 16 banks are corresponding to IPOLY(5)
   * equations for 32 banks are corresponding to IPOLY(37)
   * equations for 64 banks are corresponding to IPOLY(67)
   * To see all the IPOLY equations for all the degrees, see
   * http://wireless-systems.ece.gatech.edu/6604/handouts/Peterson's%20Table.pdf
   *
   * We generate these equations using GF(2) arithmetic:
   * http://www.ee.unb.ca/cgi-bin/tervo/calc.pl?num=&den=&f=d&e=1&m=1
   *
   * We go through all the strides 128 (10000000), 256 (100000000),...  and
   * do modular arithmetic in GF(2) Then, we create the H-matrix and group
   * each bit together, for more info read the ISCA 1991 paper
   *
   * IPOLY hashing guarantees conflict-free for all 2^n strides which widely
   * exit in GPGPU applications and also show good performance for other
   * strides.
   */
  /* [한국어] 16 뱅크 케이스: IPOLY(5) 다항식 기반 4비트 인덱스 재계산 */
  if (bank_set_num == 16) {
    std::bitset<64> a(higher_bits); /* [한국어] 64비트 물리 주소 higher_bits를 bitset으로 변환 — 각 비트에 a[i]로 접근 가능 */
    std::bitset<4> b(index);        /* [한국어] 4비트 초기 인덱스를 bitset으로 변환 — 출력 비트 계산 시 b[k]로 XOR 조합 */
    std::bitset<4> new_index(index); /* [한국어] 출력 인덱스 초기화 — 아래 4줄에서 각 비트를 IPOLY(5) H-행렬로 덮어씀 */

    /* [한국어] new_index[0]: IPOLY(5) H-행렬 0행 — a의 지정 비트들과 b[0]의 XOR 선형 결합 */
    new_index[0] =
        a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[4] ^ a[3] ^ a[0] ^ b[0];
    /* [한국어] new_index[1]: IPOLY(5) H-행렬 1행 */
    new_index[1] =
        a[12] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ a[1] ^ a[0] ^ b[1];
    /* [한국어] new_index[2]: IPOLY(5) H-행렬 2행 */
    new_index[2] = a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[2] ^ a[1] ^ b[2];
    /* [한국어] new_index[3]: IPOLY(5) H-행렬 3행 */
    new_index[3] = a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[3] ^ a[2] ^ b[3];

    return new_index.to_ulong(); /* [한국어] 4비트 bitset을 unsigned long으로 변환하여 뱅크 인덱스(0~15)로 반환 */

  /* [한국어] 32 뱅크 케이스: IPOLY(37) 다항식 기반 5비트 인덱스 재계산 */
  } else if (bank_set_num == 32) {
    std::bitset<64> a(higher_bits); /* [한국어] 64비트 주소 bitset — 비트 0~63에 접근 */
    std::bitset<5> b(index);        /* [한국어] 5비트 초기 인덱스 */
    std::bitset<5> new_index(index); /* [한국어] 출력 인덱스 (5비트 → 32 뱅크 구분) */

    /* [한국어] new_index[0]: IPOLY(37) H-행렬 0행 — 비트 13,12,11,10,9,6,5,3,0과 b[0]의 XOR */
    new_index[0] =
        a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[5] ^ a[3] ^ a[0] ^ b[0];
    /* [한국어] new_index[1]: IPOLY(37) H-행렬 1행 */
    new_index[1] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[6] ^ a[4] ^
                   a[1] ^ b[1];
    /* [한국어] new_index[2]: IPOLY(37) H-행렬 2행 */
    new_index[2] =
        a[14] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[2] ^ a[0] ^ b[2];
    /* [한국어] new_index[3]: IPOLY(37) H-행렬 3행 */
    new_index[3] =
        a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[3] ^ a[1] ^ b[3];
    /* [한국어] new_index[4]: IPOLY(37) H-행렬 4행 */
    new_index[4] =
        a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ a[2] ^ b[4];
    return new_index.to_ulong(); /* [한국어] 5비트 bitset을 unsigned long으로 변환하여 뱅크 인덱스(0~31)로 반환 */

  /* [한국어] 64 뱅크 케이스: IPOLY(67) 다항식 기반 6비트 인덱스 재계산 */
  } else if (bank_set_num == 64) {
    std::bitset<64> a(higher_bits); /* [한국어] 64비트 주소 bitset */
    std::bitset<6> b(index);        /* [한국어] 6비트 초기 인덱스 */
    std::bitset<6> new_index(index); /* [한국어] 출력 인덱스 (6비트 → 64 뱅크 구분) */

    /* [한국어] new_index[0]: IPOLY(67) H-행렬 0행 */
    new_index[0] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[12] ^ a[10] ^ a[6] ^ a[5] ^
                   a[0] ^ b[0];
    /* [한국어] new_index[1]: IPOLY(67) H-행렬 1행 */
    new_index[1] = a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[5] ^ a[1] ^
                   a[0] ^ b[1];
    /* [한국어] new_index[2]: IPOLY(67) H-행렬 2행 */
    new_index[2] = a[16] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[6] ^ a[2] ^
                   a[1] ^ b[2];
    /* [한국어] new_index[3]: IPOLY(67) H-행렬 3행 */
    new_index[3] = a[17] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[7] ^ a[3] ^
                   a[2] ^ b[3];
    /* [한국어] new_index[4]: IPOLY(67) H-행렬 4행 */
    new_index[4] = a[18] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[10] ^ a[8] ^ a[4] ^
                   a[3] ^ b[4];
    /* [한국어] new_index[5]: IPOLY(67) H-행렬 5행 */
    new_index[5] =
        a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[11] ^ a[9] ^ a[5] ^ a[4] ^ b[5];
    return new_index.to_ulong(); /* [한국어] 6비트 bitset을 unsigned long으로 변환하여 뱅크 인덱스(0~63)로 반환 */
  } else { /* Else incorrect number of channels for the hashing function */
    /* [한국어] 지원하지 않는 뱅크 수가 입력된 경우 — assert로 즉시 시뮬레이터 종료.
     * IPOLY 방정식은 16, 32, 64 뱅크에 대해서만 사전 계산되어 있다.
     * 다른 뱅크 수가 필요하면 GF(2) 산술로 H-행렬을 직접 생성해야 한다. */
    assert(
        "\nmemory_partition_indexing error: The number of "
        "channels should be "
        "16, 32 or 64 for the hashing IPOLY index function. other banks "
        "numbers are not supported. Generate it by yourself! \n" &&
        0);

    return 0; /* [한국어] assert가 발동하므로 실제로 도달하지 않음 — 컴파일러 경고 억제용 */
  }
}

/*
 * [한국어]
 * bitwise_hash_function - XOR 기반 단순 비트와이즈 뱅크/셋 인덱스 해시
 *
 * @higher_bits: 물리 주소 상위 비트들. 하위 log2(bank_set_num) 비트를 AND 마스크로 추출.
 * @index: 초기 순차 뱅크/셋 인덱스 (0 이상 bank_set_num 미만).
 * @bank_set_num: 뱅크(또는 셋) 총 개수. 반드시 2의 거듭제곱이어야 마스크가 올바르다.
 * @return: index XOR (higher_bits & (bank_set_num - 1)).
 *          higher_bits의 하위 log2(bank_set_num) 비트를 index에 XOR한 값.
 *
 * 가장 오버헤드가 작은 해시 방식이다. bank_set_num - 1을 비트 마스크로 사용하여
 * higher_bits의 하위 비트를 추출하고, 이를 index에 XOR하여 뱅크를 분산시킨다.
 * IPOLY보다 충돌 회피 능력이 낮지만, 단순하고 빠르다는 장점이 있다.
 *
 * 호출 체인: addrdec.cc::linear_to_raw_address_translation() → [bitwise_hash_function]
 */
unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
                               unsigned bank_set_num) {
  /* [한국어] index XOR (higher_bits & (bank_set_num - 1)):
   * - bank_set_num - 1: 뱅크 수가 2의 거듭제곱이므로 이 값은 하위 비트 마스크가 된다.
   *   예) bank_set_num=32 → 마스크=0x1F (하위 5비트)
   * - higher_bits & mask: 주소 상위 비트에서 뱅크 수에 해당하는 비트폭만 추출
   * - index ^ ...: 추출한 비트를 초기 인덱스에 XOR하여 뱅크를 분산시킴 */
  return (index) ^ (higher_bits & (bank_set_num - 1));
}

/*
 * [한국어]
 * PAE_hash_function - 페이지 주소 엔트로피(Page Address Entropy) 기반 해시
 *
 * @higher_bits: 물리 주소 상위 비트들 — 페이지 내 비트와 페이지 간 비트를 모두 포함.
 * @index: 초기 순차 뱅크/셋 인덱스.
 * @bank_set_num: 뱅크(또는 셋) 총 개수. 현재 구현은 32만 지원한다.
 * @return: 페이지 주소 비트와 기존 인덱스 비트를 XOR 혼합한 새 인덱스 (0~31).
 *          지원하지 않는 bank_set_num이면 assert(0)으로 종료.
 *
 * PAE 방식은 페이지 번호 비트를 뱅크 인덱스 비트와 함께 혼합하여, DRAM 행 버퍼
 * 지역성(row buffer locality)을 최대한 보존하면서도 뱅크 충돌을 줄이는 것을 목표로 한다.
 * 동일 페이지 내 연속 접근은 같은 DRAM 행에 머물러 행 버퍼 히트를 유지하고,
 * 다른 페이지로의 접근은 다른 뱅크로 분산시켜 뱅크 충돌을 줄인다.
 * 참고: Liu et al. "Get Out of the Valley: Power-Efficient Address Mapping for GPUs"
 *
 * 호출 체인: addrdec.cc::linear_to_raw_address_translation() → [PAE_hash_function]
 */
unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
                           unsigned bank_set_num) {
  // Page Address Entropy
  // random selected bits from the page and bank bits
  // similar to
  // Liu, Yuxi, et al. "Get Out of the Valley: Power-Efficient Address
  /* [한국어] 32 뱅크만 지원하는 케이스 — PAE 방정식은 32 뱅크 구성에서만 도출됨 */
  if (bank_set_num == 32) {
    std::bitset<64> a(higher_bits); /* [한국어] 64비트 주소를 bitset으로 변환 — 페이지 번호 비트와 뱅크 비트에 접근 */
    std::bitset<5> b(index);        /* [한국어] 5비트 초기 인덱스 bitset */
    std::bitset<5> new_index(index); /* [한국어] 출력 인덱스 — 아래 5줄에서 각 비트를 PAE 방정식으로 덮어씀 */

    /* [한국어] new_index[0]: 페이지 비트(a[13],a[10],a[9],a[5],a[0])와 뱅크 비트(b[3],b[0]) XOR 결합
     * b[0]이 두 번 XOR되어 있어 사실상 상쇄됨 (b[0]^b[0]=0) — 원본 코드 그대로 유지 */
    new_index[0] = a[13] ^ a[10] ^ a[9] ^ a[5] ^ a[0] ^ b[3] ^ b[0] ^ b[0];
    /* [한국어] new_index[1]: 페이지 비트와 b[3],b[2] XOR 결합. b[1]도 두 번 등장하여 상쇄됨 */
    new_index[1] = a[12] ^ a[11] ^ a[6] ^ a[1] ^ b[3] ^ b[2] ^ b[1] ^ b[1];
    /* [한국어] new_index[2]: 페이지 비트(a[14],a[9],a[8],a[7],a[2])와 b[1],b[2] XOR 결합 */
    new_index[2] = a[14] ^ a[9] ^ a[8] ^ a[7] ^ a[2] ^ b[1] ^ b[2];
    /* [한국어] new_index[3]: 페이지 비트와 b[2],b[3] XOR 결합. b[3]이 두 번 등장하여 상쇄됨 */
    new_index[3] = a[11] ^ a[10] ^ a[8] ^ a[3] ^ b[2] ^ b[3] ^ b[3];
    /* [한국어] new_index[4]: 페이지 비트(a[12],a[9],a[8],a[5],a[4])와 b[1],b[0],b[4] XOR 결합 */
    new_index[4] = a[12] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ b[1] ^ b[0] ^ b[4];

    return new_index.to_ulong(); /* [한국어] 5비트 bitset을 unsigned long으로 변환하여 뱅크 인덱스(0~31)로 반환 */
  } else {
    /* [한국어] 32 이외의 뱅크 수는 미구현 — assert(0)으로 즉시 시뮬레이터 종료.
     * 다른 뱅크 수 지원이 필요하면 PAE 방정식을 새로 설계해야 한다. */
    assert(0);
    return 0; /* [한국어] assert 발동으로 실제 도달 불가 — 컴파일러 반환값 경고 억제용 */
  }
}
