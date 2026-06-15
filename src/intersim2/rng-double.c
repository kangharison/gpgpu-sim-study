/*    This program by D E Knuth is in the public domain and freely copyable.
 *    It is explained in Seminumerical Algorithms, 3rd edition, Section 3.6
 *    (or in the errata to the 2nd edition --- see
 *        http://www-cs-faculty.stanford.edu/~knuth/taocp.html
 *    in the changes to Volume 2 on pages 171 and following).              */

/*    N.B. The MODIFICATIONS introduced in the 9th printing (2002) are
      included here; there's no backwards compatibility with the original. */

/*    This version also adopts Brendan McKay's suggestion to
      accommodate naive users who forget to call ranf_start(seed).         */

/*    If you find any bugs, please report them immediately to
 *                 taocp@cs.stanford.edu
 *    (and you will be rewarded if the bug is genuine). Thanks!            */

/*
 * [한국어 설명] Knuth RANARRAY 부동소수점 난수 생성기 (rng-double.c)
 *
 * === 파일의 역할 ===
 * Donald E. Knuth의 RANARRAY 알고리즘을 double(부동소수점)로 구현한
 * 난수 생성기이다. rng.c의 정수 버전과 대칭적으로 설계되어 있으며,
 * rng_double_wrapper.cpp에서 #include하여 C++ 환경에 통합한다.
 * BookSim/intersim2에서 RandomFloat()를 통해 [0.0, 1.0) 범위 난수를 공급.
 *
 * === 전체 아키텍처에서의 위치 ===
 * random_utils.hpp::RandomFloat() → ranf_next() [rng_double_wrapper.cpp]
 *                                   → ranf_arr_next() [이 파일]
 *                                   → ranf_array()/ran_u[]
 *
 * === 타 모듈과의 연결 ===
 * - rng_double_wrapper.cpp : 이 파일을 #include하고 main을 rng_double_main으로
 *                            rename하여 C++ 링크/충돌 회피.
 * - random_utils.hpp       : RandomSeed(seed) → ranf_start(seed),
 *                            RandomFloat() → ranf_next().
 *
 * === 주요 함수/매크로 요약 ===
 * - KK/LL           : 긴 지연(100), 짧은 지연(37).
 * - ran_u[KK]       : 실수 RNG 상태 배열.
 * - mod_sum()       : 1.0 modulo 덧셈 (실수 버전 mod_diff).
 * - ranf_array()    : 상태로부터 n개의 새 실수 난수 생성.
 * - ranf_start()    : seed로부터 실수 상태 초기화(warm-up 포함).
 * - ranf_arr_next() : 다음 실수 난수 반환(버퍼 비면 ranf_arr_cycle()).
 * - ranf_arr_cycle(): 버퍼 채우고 첫 번째 값 반환.
 */

/************ see the book for explanations and caveats! *******************/
/************ in particular, you need two's complement arithmetic **********/

#define KK 100                     /* [한국어] 긴 지연(long lag) */ /* the long lag */
#define LL  37                     /* [한국어] 짧은 지연(short lag) */ /* the short lag */
#define mod_sum(x,y) (((x)+(y))-(int)((x)+(y)))   /* [한국어] (x+y) mod 1.0 — 실수 modulo 덧셈 */

double ran_u[KK];           /* [한국어] 부동소수점 RNG 상태 배열 (the generator state) */

#ifdef __STDC__
void ranf_array(double aa[], int n)
#else
void ranf_array(aa,n)    /* put n new random fractions in aa */
  double *aa;   /* destination */
  int n;      /* array length (must be at least KK) */
#endif
{
  register int i,j;
  for (j=0;j<KK;j++) aa[j]=ran_u[j]; /* [한국어] 현재 실수 상태 ran_u를 목적지 버퍼에 복사 */
  for (;j<n;j++) aa[j]=mod_sum(aa[j-KK],aa[j-LL]); /* [한국어] 긴/짧은 지연 값을 mod_sum으로 조합하며 새 난수 생성 */
  for (i=0;i<LL;i++,j++) ran_u[i]=mod_sum(aa[j-KK],aa[j-LL]); /* [한국어] 다음 상태의 앞부분(LL개) 갱신 */
  for (;i<KK;i++,j++) ran_u[i]=mod_sum(aa[j-KK],ran_u[i-LL]); /* [한국어] 다음 상태의 나머지 부분(KK-LL개) 갱신 */
}

/* the following routines are adapted from exercise 3.6--15 */
/* after calling ranf_start, get new randoms by, e.g., "x=ranf_arr_next()" */

#define QUALITY 1009 /* [한국어] 고해상도 사용을 위한 권장 품질 수준(버퍼 크기) */ /* recommended quality level for high-res use */
double ranf_arr_buf[QUALITY]; /* [한국어] 사전 생성된 실수 난수 버퍼 */
double ranf_arr_dummy=-1.0, ranf_arr_started=-1.0; /* [한국어] 초기화 여부 표시용 더미 값들 */
double *ranf_arr_ptr=&ranf_arr_dummy; /* [한국어] 다음 실수 난수 위치 포인터 */

#define TT  70   /* [한국어] 스트림 간 보장된 분리 거리 */ /* guaranteed separation between streams */
#define is_odd(s) ((s)&1) /* [한국어] 최하위 비트로 홀수 여부 판별 */

#ifdef __STDC__
void ranf_start(long seed)
#else
void ranf_start(seed)    /* do this before using ranf_array */
  long seed;            /* selector for different streams */
#endif
{
  register int t,s,j;
  double u[KK+KK-1]; /* [한국어] 초기화를 위한 실수 준비 버퍼 */
  double ulp=(1.0/(1L<<30))/(1L<<22);               /* [한국어] 2^-52, double 최하위 비트 단위 */ /* 2 to the -52 */
  double ss=2.0*ulp*((seed&0x3fffffff)+2); /* [한국어] seed를 기반으로 한 초기 실수 값 */

  for (j=0;j<KK;j++) {
    u[j]=ss;                                /* [한국어] 준비 버퍼에 초기 값 채우기 */
    ss+=ss; if (ss>=1.0) ss-=1.0-2*ulp;  /* [한국어] 51비트 순환 시프트로 다음 초기값 생성 */
  }
  u[1]+=ulp;                     /* [한국어] u[1]만 "홀수"로 만들어 상태 다양성 확보 */
  for (s=seed&0x3fffffff,t=TT-1; t; ) {
    for (j=KK-1;j>0;j--)
      u[j+j]=u[j],u[j+j-1]=0.0;                         /* [한국어] 실수 버퍼 "제곱" 연산 */
    for (j=KK+KK-2;j>=KK;j--) {
      u[j-(KK-LL)]=mod_sum(u[j-(KK-LL)],u[j]);
      u[j-KK]=mod_sum(u[j-KK],u[j]); /* [한국어] 지연 위치 조합으로 다음 상태 후보 생성 */
    }
    if (is_odd(s)) {                             /* [한국어] seed의 현재 비트가 1이면 "z 곱셈" 수행 */
      for (j=KK;j>0;j--) u[j]=u[j-1];
      u[0]=u[KK];                    /* [한국어] 버퍼를 순환 시프트 */
      u[LL]=mod_sum(u[LL],u[KK]);
    }
    if (s) s>>=1; else t--; /* [한국어] seed 비트를 소비하거나 반복 카운트 감소 */
  }
  for (j=0;j<LL;j++) ran_u[j+KK-LL]=u[j]; /* [한국어] 준비 버퍼 앞부분을 ran_u 뒷부분에 복사 */
  for (;j<KK;j++) ran_u[j-LL]=u[j];       /* [한국어] 준비 버퍼 나머지를 ran_u 앞부분에 복사 */
  for (j=0;j<10;j++) ranf_array(u,KK+KK-1);  /* [한국어] 10회 웜업으로 초기 상태 안정화 */
  ranf_arr_ptr=&ranf_arr_started; /* [한국어] 실수 난수 생성기 초기화 완료 표시 */
}

#define ranf_arr_next() (*ranf_arr_ptr>=0? *ranf_arr_ptr++: ranf_arr_cycle()) /* [한국어] 버퍼에 남은 값이 있으면 소비, 없으면 재생성 */
double ranf_arr_cycle()
{
  if (ranf_arr_ptr==&ranf_arr_dummy)
    ranf_start(314159L); /* [한국어] ranf_start()를 잊은 경우 기본 시드로 자동 초기화 */
  ranf_array(ranf_arr_buf,QUALITY); /* [한국어] QUALITY 개수만큼 새 실수 난수 생성 */
  ranf_arr_buf[KK]=-1; /* [한국어] 버퍼 끝 마커 */
  ranf_arr_ptr=ranf_arr_buf+1; /* [한국어] 두 번째 요소부터 소비 시작 */
  return ranf_arr_buf[0]; /* [한국어] 첫 번째 실수 난수 반환 */
}

#include <stdio.h>
int main()
{
  register int m; double a[2009]; /* [한국어] 알고리즘 검증용 배열 */
  ranf_start(310952); /* [한국어] 고정 시드로 실수 RNG 초기화 */
  for (m=0;m<2009;m++) ranf_array(a,1009); /* [한국어] 1009개씩 2009번 생성 */
  printf("%.20f\n", ran_u[0]);            /* 0.36410514377569680455 */
     /* beware of buggy printf routines that do not give full accuracy here! */
  ranf_start(310952); /* [한국어] 동일 시드로 재초기화 — 재현성 검증 */
  for (m=0;m<1009;m++) ranf_array(a,2009); /* [한국어] 2009개씩 1009번 생성 */
  printf("%.20f\n", ran_u[0]);            /* 0.36410514377569680455 */
  return 0;
}
