/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <mpi.h>
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv); int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);
    Config cfg=config(argc,argv,1,100000000); int n=cfg.n; unsigned long long hits=0;
    double t=now(), comm=0;
    
    for(int i=rank;i<n;i+=p) {
        double x=mix32(cfg.seed+2u*(uint32_t)i)/4294967296.0;
        double y=mix32(cfg.seed+2u*(uint32_t)i+1u)/4294967296.0;
        if(x*x+y*y<=1.0) hits++;
    }
    double ct=now(); unsigned long long total=0;
    MPI_Reduce(&hits,&total,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD); hits=total; comm=now()-ct;
    double elapsed=now()-t, values[2]={(double)hits,4.0*hits/n};
    if(rank==0) emit(values,2,cfg.full,elapsed,comm);
    MPI_Finalize(); return 0;
}
