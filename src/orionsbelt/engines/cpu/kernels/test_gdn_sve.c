#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
void gdn_cumdecay_f32(const float*,float*,size_t,size_t);
void gdn_gated_scan_f32(const float*,const float*,float*,float*,size_t,size_t);
void gdn_causal_dwconv1d_f32(const float*,const float*,float*,float*,size_t,size_t);

/* fp16/bf16 state variants (ob-8qt.4) */
void gdn_gated_scan_f16(const float*,const float*,float*,__fp16*,size_t,size_t);
void gdn_cumdecay_f16(const float*,__fp16*,size_t,size_t);
void gdn_gated_scan_bf16(const float*,const float*,float*,uint16_t*,size_t,size_t);
void gdn_cumdecay_bf16(const float*,uint16_t*,size_t,size_t);

/* precision-MATCHED scalar reference (float accumulators, like the kernel) */
static void refF_scan(const float*g,const float*x,float*s,float*st,size_t T,size_t C){
  for(size_t c=0;c<C;c++){float a=st[c];for(size_t t=0;t<T;t++){a=x[t*C+c]+a*g[t*C+c];s[t*C+c]=a;}st[c]=a;}}
static void refF_conv(const float*in,const float*w,float*o,float*h,size_t T,size_t C){
  for(size_t c=0;c<C;c++){float H[3]={h[0*C+c],h[1*C+c],h[2*C+c]};
    for(size_t t=0;t<T;t++){float cur=in[t*C+c];
      o[t*C+c]=H[0]*w[0*C+c]+H[1]*w[1*C+c]+H[2]*w[2*C+c]+cur*w[3*C+c];
      H[0]=H[1];H[1]=H[2];H[2]=cur;}
    h[0*C+c]=H[0];h[1*C+c]=H[1];h[2*C+c]=H[2];}}
/* double reference, for the honest numerical-quality view */
static void refD_scan(const float*g,const float*x,float*s,float*st,size_t T,size_t C){
  for(size_t c=0;c<C;c++){double a=st[c];for(size_t t=0;t<T;t++){a=x[t*C+c]+a*g[t*C+c];s[t*C+c]=(float)a;}st[c]=(float)a;}}

static void report(const char*n,const float*a,const float*b,size_t N){
  double mabs=0,mrel=0; size_t big=0;
  for(size_t i=0;i<N;i++){double d=fabs((double)a[i]-b[i]); if(d>mabs)mabs=d;
    if(fabs((double)b[i])>1e-2){big++; double r=d/fabs((double)b[i]); if(r>mrel)mrel=r;}}
  printf("  %-28s max_abs=%.3e  max_rel(|ref|>1e-2, n=%zu)=%.3e\n",n,mabs,big,mrel);}

int main(void){
  size_t T=64, C=2051, N=T*C;   /* 2051 exercises the predicated tail */
  float *g=malloc(N*4),*x=malloc(N*4),*w=malloc(4*C*4);
  float *s1=malloc(N*4),*s2=malloc(N*4),*s3=malloc(N*4);
  float *o1=malloc(N*4),*o2=malloc(N*4);
  float *stA=malloc(C*4),*stB=malloc(C*4),*stC=malloc(C*4);
  float *hA=malloc(3*C*4),*hB=malloc(3*C*4);
  srand(7);
  for(size_t i=0;i<N;i++){g[i]=0.5f+0.4f*(rand()/(float)RAND_MAX);x[i]=(rand()/(float)RAND_MAX)-0.5f;}
  for(size_t i=0;i<4*C;i++)w[i]=(rand()/(float)RAND_MAX)-0.5f;
  for(size_t i=0;i<C;i++){float v=(rand()/(float)RAND_MAX)-0.5f;stA[i]=stB[i]=stC[i]=v;}
  for(size_t i=0;i<3*C;i++){float v=(rand()/(float)RAND_MAX)-0.5f;hA[i]=hB[i]=v;}

  gdn_gated_scan_f32(g,x,s1,stA,T,C);
  refF_scan(g,x,s2,stB,T,C);
  refD_scan(g,x,s3,stC,T,C);
  gdn_causal_dwconv1d_f32(x,w,o1,hA,T,C);
  refF_conv(x,w,o2,hB,T,C);

  printf("SVE kernel vs PRECISION-MATCHED float reference (expect ~0):\n");
  report("gated_scan",s1,s2,N); report("gated_scan carried state",stA,stB,C);
  report("causal_dwconv1d",o1,o2,N); report("conv history",hA,hB,3*C);
  printf("SVE kernel vs DOUBLE reference (fp32 accumulation quality):\n");
  report("gated_scan",s1,s3,N);

  double mabs=0; for(size_t i=0;i<N;i++){double d=fabs((double)s1[i]-s2[i]);if(d>mabs)mabs=d;}
  int exact = (mabs==0.0);
  printf("\ngated_scan bit-identical to matched reference: %s\n", exact?"YES":"no");

  /* ====================================================================
   * Mixed-precision state variants (ob-8qt.4)
   *
   * fp16/bf16 state should approximate fp32 closely — the state carries a
   * running value in [−1, 1] (bounded by gating), so narrowing it loses a few
   * ULPs per step but does not blow up over 64 tokens.
   *
   * The test: run fp32, fp16, and bf16 variants on the SAME inputs and initial
   * state, then compare the narrowed variants against fp32.  Also compare
   * against the double reference to see how much total error each adds.
   * ==================================================================== */
  printf("\nMixed-precision state variants vs fp32 (ob-8qt.4):\n");

  /* --- gated_scan: fp16 state --- */
  float *sF16=malloc(N*4),*sBF16=malloc(N*4);
  float *stF32=malloc(C*4),*stBF32=malloc(C*4);
  __fp16 *stH=malloc(C*sizeof(__fp16));
  uint16_t *stBraw=malloc(C*sizeof(uint16_t));
  /* same initial state for all variants */
  for(size_t i=0;i<C;i++){
    float v=(rand()/(float)RAND_MAX)-0.5f;
    stF32[i]=stBF32[i]=v;
    stH[i]=(__fp16)v;     /* fp16 init */
  }
  for(size_t i=0;i<C;i++){
    /* simulate bf16 init: convert to bf16 and back */
    uint32_t u; float fv=stF32[i]; memcpy(&u,&fv,4);
    uint32_t lsb=(u>>16)&1, bias=0x7FFF+lsb;
    uint16_t bf=(uint16_t)((u+bias)>>16);
    uint32_t u2=(uint32_t)bf<<16; float fb; memcpy(&fb,&u2,4);
    stBF32[i]=fb;
    stBraw[i]=bf;
  }

  /* fp32 baseline (re-run with matching initial state) */
  float *sRef=malloc(N*4);
  gdn_gated_scan_f32(g,x,sRef,stF32,T,C);

  /* fp16 state */
  gdn_gated_scan_f16(g,x,sF16,stH,T,C);
  printf("  fp16-state gated_scan:\n");
  report("    vs fp32",sF16,sRef,N);

  /* bf16 state */
  gdn_gated_scan_bf16(g,x,sBF16,stBraw,T,C);
  printf("  bf16-state gated_scan:\n");
  report("    vs fp32",sBF16,sRef,N);

  /* --- cumulative decay: fp16 and bf16 output --- */
  __fp16 *dH=malloc(N*sizeof(__fp16));
  uint16_t *dB=malloc(N*sizeof(uint16_t));
  float *dF=malloc(N*4),*dFfromH=malloc(N*4),*dFfromB=malloc(N*4);
  /* decay input: values in (0.90, 0.99) so the cumulative product stays in fp32 range */
  float *aDecay=malloc(N*sizeof(float));
  for(size_t i=0;i<N;i++) aDecay[i]=0.90f+0.09f*(float)((i*2654435761u)%1000)/1000.0f;
  gdn_cumdecay_f32(aDecay,dF,T,C);
  gdn_cumdecay_f16(aDecay,dH,T,C);
  gdn_cumdecay_bf16(aDecay,dB,T,C);
  for(size_t i=0;i<N;i++){dFfromH[i]=(float)dH[i];}
  for(size_t i=0;i<N;i++){
    uint32_t u=(uint32_t)dB[i]<<16; memcpy(&dFfromB[i],&u,4);
  }
  printf("  fp16-output cumdecay:\n");
  report("    vs fp32",dFfromH,dF,N);
  printf("  bf16-output cumdecay:\n");
  report("    vs fp32",dFfromB,dF,N);

  /* --- drift over repeated chunks (simulate multi-chunk decode) --- */
  /* Reset state and run 8 chunks back-to-back to see if narrowing error
   * compounds catastrophically. Each chunk reuses the same g,x patterns. */
  printf("\nMulti-chunk drift (8 chunks, same data per chunk):\n");
  for(size_t i=0;i<C;i++){
    float v=(rand()/(float)RAND_MAX)-0.5f;
    stF32[i]=v; stH[i]=(__fp16)v;
  }
  float drift_scan=0;
  for(int chunk=0;chunk<8;chunk++){
    gdn_gated_scan_f32(g,x,sRef,stF32,T,C);
    gdn_gated_scan_f16(g,x,sF16,stH,T,C);
    double m=0; for(size_t i=0;i<N;i++){double d=fabs((double)sRef[i]-sF16[i]);if(d>m)m=d;}
    printf("  chunk %d: fp16 vs fp32 max_abs=%.3e\n",chunk+1,m);
    if(m>drift_scan)drift_scan=m;
  }
  printf("  worst over 8 chunks: %.3e %s\n",(double)drift_scan,
         drift_scan<1e-3?"(acceptable)":"(CHECK)");

  free(sF16);free(sBF16);free(stF32);free(stBF32);free(stH);free(stBraw);
  free(sRef);free(dH);free(dB);free(dF);free(dFfromH);free(dFfromB);free(aDecay);
  return 0;
}
