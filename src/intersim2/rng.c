/*    This program by D E Knuth is in the public domain and freely copyable.
 *    It is explained in Seminumerical Algorithms, 3rd edition, Section 3.6
 *    (or in the errata to the 2nd edition --- see
 *        http://www-cs-faculty.stanford.edu/~knuth/taocp.html
 *    in the changes to Volume 2 on pages 171 and following).              */

/*    N.B. The MODIFICATIONS introduced in the 9th printing (2002) are
      included here; there's no backwards compatibility with the original. */

/*    This version also adopts Brendan McKay's suggestion to
      accommodate naive users who forget to call ran_start(seed).          */

/*    If you find any bugs, please report them immediately to
 *                 taocp@cs.stanford.edu
 *    (and you will be rewarded if the bug is genuine). Thanks!            */

/*
 * [한국어 설명] Knuth RANARRAY 정수 난수 생성기 (rng.c)
 *
 * === 파일의 역할 ===
 * Donald E. Knuth의 RANARRAY 알고리즘을 구현한 정수 난수 생성기이다.
 * BookSim/intersim2 시뮬레이터의 재현성(reproducibility)을 위해 사용되며,
 * rng_wrapper.cpp에서 #include하여 C++ 환경에 통합한다.
 * 긴 지연(long lag)과 짧은 지연(short lag)을 활용한 고품질 의사난수를 생성한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * random_utils.hpp::RandomInt() / RandomIntLong() → ran_next() [rng_wrapper.cpp]
 *                                                   → ran_arr_next() [이 파일]
 *                                                   → ran_array()/ran_x[]
 *
 * === 타 모듈과의 연결 ===
 * - rng_wrapper.cpp : 이 파일을 #include하고 main을 rng_main으로 rename하여
 *                     C++ 링크와 main 충돌을 회피.
 * - random_utils.hpp : RandomSeed(seed) → ran_start(seed), RandomInt() → ran_next().
 *
 * === 주요 함수/매크로 요약 ===
 * - KK/LL/MM       : 긴 지연(100), 짧은 지연(37), 모듈러(2^30).
 * - ran_x[KK]      : 정수 RNG 상태 배열.
 * - ran_array()    : 상태를 기반으로 n개의 새 난수를 aa[]에 생성.
 * - ran_start()    : seed로부터 상태 배열 초기화(warm-up 포함).
 * - ran_arr_next() : 다음 정수 난수 반환(버퍼가 비면 ran_arr_cycle()로 재생성).
 * - ran_arr_cycle(): 버퍼를 채우고 첫 번째 값 반환.
 */

/************ see the book for explanations and caveats! *******************/
/************ in particular, you need two's complement arithmetic **********/

#define KK 100                     /* [한국어] 긴 지연(long lag) — 상태 배열 크기 */ /* the long lag */
#define LL  37                     /* [한국어] 짧은 지연(short lag) */ /* the short lag */
#define MM (1L<<30)                 /* [한국어] 모듈러 값 2^30 */ /* the modulus */
#define mod_diff(x,y) (((x)-(y))&(MM-1)) /* [한국어] 모듈러 MM에서의 뺄셈: 결과를 MM-1로 마스크 */

long ran_x[KK];                    /* [한국어] 정수 RNG 상태 배열 (the generator state) */

#ifdef __STDC__
void ran_array(long aa[],int n)
#else
void ran_array(aa,n)    /* put n new random numbers in aa */
  long *aa;   /* destination */
  int n;      /* array length (must be at least KK) */
#endif
{
  register int i,j;
  for (j=0;j<KK;j++) aa[j]=ran_x[j]; /* [한국어] 현재 상태 ran_x를 목적지 버퍼에 복사 */
  for (;j<n;j++) aa[j]=mod_diff(aa[j-KK],aa[j-LL]); /* [한국어] 긴 지연과 짧은 지연 값을 mod_diff로 조합하며 새 난수 생성 */
  for (i=0;i<LL;i++,j++) ran_x[i]=mod_diff(aa[j-KK],aa[j-LL]); /* [한국어] 다음 상태의 앞부분(LL개) 갱신 */
  for (;i<KK;i++,j++) ran_x[i]=mod_diff(aa[j-KK],ran_x[i-LL]); /* [한국어] 다음 상태의 나머지 부분(KK-LL개) 갱신 */
}

/* the following routines are from exercise 3.6--15 */
/* after calling ran_start, get new randoms by, e.g., "x=ran_arr_next()" */

#define QUALITY 1009 /* [한국어] 고해상도 사용을 위한 권장 품질 수준(버퍼 크기) */ /* recommended quality level for high-res use */
long ran_arr_buf[QUALITY]; /* [한국어] 사전 생성된 난수 버퍼 */
long ran_arr_dummy=-1, ran_arr_started=-1; /* [한국어] 초기화 여부 표시용 더미 값들 */
long *ran_arr_ptr=&ran_arr_dummy; /* [한국어] 다음 난수 위치 포인터, 초기화 안 된 경우 &ran_arr_dummy */

#define TT  70   /* [한국어] 스트림 간 보장된 분리 거리 */ /* guaranteed separation between streams */
#define is_odd(x)  ((x)&1)          /* [한국어] x의 최하위 비트(홀수 여부) */

#ifdef __STDC__
void ran_start(long seed)
#else
void ran_start(seed)    /* do this before using ran_array */
  long seed;            /* selector for different streams */
#endif
{
  register int t,j;
  long x[KK+KK-1];              /* [한국어] 초기화를 위한 준비 버퍼 */
  register long ss=(seed+2)&(MM-2); /* [한국어] seed를 짝수로 변환하여 초기 상태 bootstrap */
  for (j=0;j<KK;j++) {
    x[j]=ss;                      /* [한국어] 준비 버퍼에 seed 기반 값 채우기 */
    ss<<=1; if (ss>=MM) ss-=MM-2; /* [한국어] 29비트 순환 시프트로 다음 초기값 생성 */
  }
  x[1]++;              /* [한국어] x[1]만 홀수로 만들어 상태 다양성 확보 */
  for (ss=seed&(MM-1),t=TT-1; t; ) {       
    for (j=KK-1;j>0;j--) x[j+j]=x[j], x[j+j-1]=0; /* [한국어] 버퍼 "제곱" 연산: 각 값을 두 칸에 복사하고 중간은 0 */
    for (j=KK+KK-2;j>=KK;j--)
      x[j-(KK-LL)]=mod_diff(x[j-(KK-LL)],x[j]),
      x[j-KK]=mod_diff(x[j-KK],x[j]); /* [한국어] 지연 위치 조합으로 다음 상태 후보 생성 */
    if (is_odd(ss)) {              /* [한국어] seed의 현재 비트가 1이면 "z 곱셈" 수행 */
      for (j=KK;j>0;j--)  x[j]=x[j-1];
      x[0]=x[KK];            /* [한국어] 버퍼를 순환 시프트 */
      x[LL]=mod_diff(x[LL],x[KK]);
    }
    if (ss) ss>>=1; else t--; /* [한국어] seed 비트를 소비하거나, 0이면 반복 카운트 감소 */
  }
  for (j=0;j<LL;j++) ran_x[j+KK-LL]=x[j]; /* [한국어] 준비 버퍼의 앞부분을 ran_x 뒷부분에 복사 */
  for (;j<KK;j++) ran_x[j-LL]=x[j];       /* [한국어] 준비 버퍼의 나머지를 ran_x 앞부분에 복사 */
  for (j=0;j<10;j++) ran_array(x,KK+KK-1); /* [한국어] 10회 웜업으로 초기 상태 안정화 */
  ran_arr_ptr=&ran_arr_started; /* [한국어] 난수 생성기 초기화 완료 표시 (ran_arr_next가 실제 생성기 사용) */
}

#define ran_arr_next() (*ran_arr_ptr>=0? *ran_arr_ptr++: ran_arr_cycle()) /* [한국어] 버퍼에 남은 값이 있으면 소비, 없으면 재생성 */
long ran_arr_cycle()
{
  if (ran_arr_ptr==&ran_arr_dummy)
    ran_start(314159L); /* [한국어] ran_start()를 잊은 경우 기본 시드로 자동 초기화 */
  ran_array(ran_arr_buf,QUALITY); /* [한국어] QUALITY 개수만큼 새 난수 생성하여 버퍼 채움 */
  ran_arr_buf[KK]=-1; /* [한국어] 버퍼 끝 표시 마커(실제로는 ran_arr_ptr이 관리) */
  ran_arr_ptr=ran_arr_buf+1; /* [한국어] 두 번째 요소부터 소비 시작 */
  return ran_arr_buf[0]; /* [한국어] 첫 번째 난수 반환 */
}

#include <stdio.h>
int main()
{
  register int m; long a[2009]; 
  ran_start(310952L); /* [한국어] 알고리즘 검증용 고정 시드로 초기화 */
  for (m=0;m<=2009;m++) ran_array(a,1009); /* [한국어] 1009개씩 2010번 난수 배열 생성 */
  printf("%ld\n", a[0]);             /* 995235265 */
  ran_start(310952L); /* [한국어] 동일 시드로 재초기화 — 재현성 검증 */
  for (m=0;m<=1009;m++) ran_array(a,2009); /* [한국어] 2009개씩 1010번 생성 */
  printf("%ld\n", a[0]);             /* 995235265 */
  return 0;
}
