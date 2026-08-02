#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
void gdn_cumdecay_f32(const float*,float*,size_t,size_t);
void gdn_gated_scan_f32(const float*,const float*,float*,float*,size_t,size_t);
void gdn_causal_dwconv1d_f32(const float*,const float*,float*,float*,size_t,size_t);

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

  printf("SVE2 kernel vs PRECISION-MATCHED float reference (expect ~0):\n");
  report("gated_scan",s1,s2,N); report("gated_scan carried state",stA,stB,C);
  report("causal_dwconv1d",o1,o2,N); report("conv history",hA,hB,3*C);
  printf("SVE2 kernel vs DOUBLE reference (fp32 accumulation quality):\n");
  report("gated_scan",s1,s3,N);

  double mabs=0; for(size_t i=0;i<N;i++){double d=fabs((double)s1[i]-s2[i]);if(d>mabs)mabs=d;}
  int exact = (mabs==0.0);
  printf("\ngated_scan bit-identical to matched reference: %s\n", exact?"YES":"no");
  return 0;
}
