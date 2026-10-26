/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <omp.h>
#include <stdatomic.h>
int main(int argc,char **argv) {
    Config cfg=config(argc,argv,3,1024); int n=cfg.n; size_t nn=(size_t)n*n;
    double *a=allocate(nn,sizeof(double)), *b=allocate(nn,sizeof(double));
    for(int i=1;i<n-1;i++) for(int j=1;j<n-1;j++) a[(size_t)i*n+j]=initial((uint32_t)(i*n+j),cfg.seed);
    double t=now();
    for(int step=0;step<cfg.steps;step++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=1;i<n-1;i++) for(int j=1;j<n-1;j++) {
            size_t q=(size_t)i*n+j; b[q]=0.25*(a[q-n]+a[q+n]+a[q-1]+a[q+1]);
        }
        double *swap=a; a=b; b=swap;
    }
    double elapsed=now()-t; emit(a,nn,cfg.full,elapsed,0); free(a); free(b); return 0;
}
