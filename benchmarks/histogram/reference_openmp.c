/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <omp.h>
#include <stdatomic.h>
int main(int argc,char **argv) {
    
    Config cfg=config(argc,argv,1,100000000); int n=cfg.n; unsigned long long bins[256]={0};
    double t=now(),comm=0;
    #pragma omp parallel
    {
        unsigned long long local[256]={0};
        #pragma omp for schedule(static)
        for(int i=0;i<n;i++) local[mix32(cfg.seed+(uint32_t)i)%256]++;
        for(int j=0;j<256;j++) {
            #pragma omp atomic update
            bins[j]+=local[j];
        }
    }
    double elapsed=now()-t, values[256]; for(int j=0;j<256;j++) values[j]=(double)bins[j];
    emit(values,256,cfg.full,elapsed,comm);
     return 0;
}
