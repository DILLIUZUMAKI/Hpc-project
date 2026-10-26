/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <omp.h>
#include <stdatomic.h>
int main(int argc,char **argv) {
    
    Config cfg=config(argc,argv,1,100000000); int n=cfg.n; unsigned long long hits=0;
    double t=now(), comm=0;
    #pragma omp parallel for reduction(+:hits) schedule(static)
    for(int i=0;i<n;i++) {
        double x=mix32(cfg.seed+2u*(uint32_t)i)/4294967296.0;
        double y=mix32(cfg.seed+2u*(uint32_t)i+1u)/4294967296.0;
        if(x*x+y*y<=1.0) hits++;
    }
    
    double elapsed=now()-t, values[2]={(double)hits,4.0*hits/n};
    emit(values,2,cfg.full,elapsed,comm);
     return 0;
}
