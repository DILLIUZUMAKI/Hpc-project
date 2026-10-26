/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <mpi.h>
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv); int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);
    Config cfg=config(argc,argv,3,1024); int n=cfg.n;
    int rows=n/p+(rank<n%p), start=rank*(n/p)+(rank<n%p?rank:n%p), active=n<p?n:p;
    double *a=allocate((size_t)(rows+2)*n,sizeof(double)), *b=allocate((size_t)(rows+2)*n,sizeof(double));
    double *all=allocate(rank==0?(size_t)n*n:1,sizeof(double));
    int *counts=allocate(p,sizeof(int)), *displs=allocate(p,sizeof(int));
    for(int r=0;r<p;r++) { counts[r]=(n/p+(r<n%p))*n; displs[r]=(r*(n/p)+(r<n%p?r:n%p))*n; }
    for(int i=1;i<=rows;i++) for(int j=1;j<n-1;j++) {
        int global=start+i-1; if(global>0 && global<n-1) a[(size_t)i*n+j]=initial((uint32_t)(global*n+j),cfg.seed);
    }
    MPI_Barrier(MPI_COMM_WORLD); double t=MPI_Wtime(), comm=0;
    for(int step=0;step<cfg.steps;step++) {
        if(rows) {
            int prev=rank?rank-1:MPI_PROC_NULL, next=rank+1<active?rank+1:MPI_PROC_NULL;
            double ct=MPI_Wtime();
            MPI_Sendrecv(a+n,n,MPI_DOUBLE,prev,0,a+(size_t)(rows+1)*n,n,MPI_DOUBLE,next,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Sendrecv(a+(size_t)rows*n,n,MPI_DOUBLE,next,1,a,n,MPI_DOUBLE,prev,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            comm+=MPI_Wtime()-ct;
            for(int i=1;i<=rows;i++) for(int j=1;j<n-1;j++) {
                int global=start+i-1; size_t q=(size_t)i*n+j;
                if(global>0 && global<n-1) b[q]=0.25*(a[q-n]+a[q+n]+a[q-1]+a[q+1]);
            }
            double *swap=a; a=b; b=swap;
        }
    }
    double ct=MPI_Wtime();
    MPI_Gatherv(a+n,rows*n,MPI_DOUBLE,all,counts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD); comm+=MPI_Wtime()-ct;
    double elapsed=MPI_Wtime()-t,max_elapsed,max_comm;
    MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&comm,&max_comm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) emit(all,(size_t)n*n,cfg.full,max_elapsed,max_comm);
    free(a); free(b); free(all); free(counts); free(displs); MPI_Finalize(); return 0;
}
