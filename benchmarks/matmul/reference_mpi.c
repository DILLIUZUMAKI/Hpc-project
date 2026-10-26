/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <mpi.h>
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv); int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);
    Config cfg=config(argc,argv,1,1024); int n=cfg.n, start=n*rank/p, end=n*(rank+1)/p;
    size_t nn=(size_t)n*n;
    double *a=allocate(nn,sizeof(double)), *b=allocate(nn,sizeof(double));
    double *local=allocate((size_t)(end-start)*n,sizeof(double)), *all=allocate(rank==0?nn:1,sizeof(double));
    int *counts=allocate(p,sizeof(int)), *displs=allocate(p,sizeof(int));
    for(int r=0;r<p;r++) { counts[r]=(n*(r+1)/p-n*r/p)*n; displs[r]=(n*r/p)*n; }
    for(size_t i=0;i<nn;i++) { a[i]=initial((uint32_t)(2*i),cfg.seed); b[i]=initial((uint32_t)(2*i+1),cfg.seed); }
    MPI_Barrier(MPI_COMM_WORLD); double t=MPI_Wtime();
    for(int i=start;i<end;i++) for(int k=0;k<n;k++) {
        double v=a[(size_t)i*n+k]; for(int j=0;j<n;j++) local[(size_t)(i-start)*n+j]+=v*b[(size_t)k*n+j];
    }
    double ct=MPI_Wtime();
    MPI_Gatherv(local,(end-start)*n,MPI_DOUBLE,all,counts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    double comm=MPI_Wtime()-ct, elapsed=MPI_Wtime()-t, max_elapsed,max_comm;
    MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&comm,&max_comm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) emit(all,nn,cfg.full,max_elapsed,max_comm);
    free(a); free(b); free(local); free(all); free(counts); free(displs); MPI_Finalize(); return 0;
}
