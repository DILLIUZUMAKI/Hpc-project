/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <mpi.h>
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv); int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);
    Config cfg=config(argc,argv,1,100000000); int n=cfg.n; unsigned long long bins[256]={0};
    double t=now(),comm=0;
    for(int i=rank;i<n;i+=p) bins[mix32(cfg.seed+(uint32_t)i)%256]++;
    double ct=now(); unsigned long long total[256]={0};
    MPI_Reduce(bins,total,256,MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    memcpy(bins,total,sizeof(bins)); comm=now()-ct;
    double elapsed=now()-t, values[256]; for(int j=0;j<256;j++) values[j]=(double)bins[j];
    if(rank==0) emit(values,256,cfg.full,elapsed,comm);
    MPI_Finalize(); return 0;
}
