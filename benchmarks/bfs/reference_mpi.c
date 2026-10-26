/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <mpi.h>
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv); int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);
    Config cfg=config(argc,argv,1,200000); int n=cfg.n;
    int *dist=allocate(n,sizeof(int)), *next=allocate(n,sizeof(int));
    for(int i=0;i<n;i++) dist[i]=INT_MAX;
    dist[0]=0; int level=0,changed=1; double t=now(),comm=0;
    while(changed) {
        memcpy(next,dist,(size_t)n*sizeof(int));
        for(int u=rank;u<n;u+=p) if(dist[u]==level) for(int k=0;k<4;k++) {
            int v=neighbor(u,k,n,cfg.seed); if(next[v]>level+1) next[v]=level+1;
        }
        double ct=now(); MPI_Allreduce(MPI_IN_PLACE,next,n,MPI_INT,MPI_MIN,MPI_COMM_WORLD); comm+=now()-ct;
        changed=0; for(int i=0;i<n;i++) if(next[i]<dist[i]) { dist[i]=next[i]; changed=1; }
        level++;
    }
    double elapsed=now()-t, *values=allocate(n,sizeof(double));
    for(int i=0;i<n;i++) values[i]=dist[i]==INT_MAX?-1:dist[i];
    if(rank==0) emit(values,n,cfg.full,elapsed,comm);
    free(dist); free(next); free(values); MPI_Finalize(); return 0;
}
