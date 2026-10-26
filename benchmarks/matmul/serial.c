/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
int main(int argc,char **argv) {
    Config cfg=config(argc,argv,1,1024); int n=cfg.n; size_t nn=(size_t)n*n;
    double *a=allocate(nn,sizeof(double)), *b=allocate(nn,sizeof(double)), *c=allocate(nn,sizeof(double));
    for(size_t i=0;i<nn;i++) { a[i]=initial((uint32_t)(2*i),cfg.seed); b[i]=initial((uint32_t)(2*i+1),cfg.seed); }
    double t=now();
    for(int i=0;i<n;i++) for(int j=0;j<n;j++)
        for(int k=0;k<n;k++) c[(size_t)i*n+j]+=a[(size_t)i*n+k]*b[(size_t)k*n+j];
    double elapsed=now()-t; emit(c,nn,cfg.full,elapsed,0); free(a); free(b); free(c); return 0;
}
