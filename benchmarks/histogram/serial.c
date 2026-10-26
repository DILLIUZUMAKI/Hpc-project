/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
int main(int argc,char **argv) {
    
    Config cfg=config(argc,argv,1,100000000); int n=cfg.n; unsigned long long bins[256]={0};
    double t=now(),comm=0;
    for(int i=0;i<n;i++) bins[mix32(cfg.seed+(uint32_t)i)%256]++;
    double elapsed=now()-t, values[256]; for(int j=0;j<256;j++) values[j]=(double)bins[j];
    emit(values,256,cfg.full,elapsed,comm);
     return 0;
}
