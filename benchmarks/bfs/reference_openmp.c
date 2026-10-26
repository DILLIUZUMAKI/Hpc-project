/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
#include <omp.h>
#include <stdatomic.h>
int main(int argc,char **argv) {
    Config cfg=config(argc,argv,1,200000); int n=cfg.n;
    atomic_int *dist=allocate(n,sizeof(atomic_int));
    for(int i=0;i<n;i++) atomic_init(&dist[i],-1);
    atomic_store(&dist[0],0); int level=0,changed=1; double t=now();
    while(changed) {
        changed=0;
        #pragma omp parallel for reduction(|:changed) schedule(static)
        for(int u=0;u<n;u++) if(atomic_load(&dist[u])==level) {
            for(int k=0;k<4;k++) {
                int v=neighbor(u,k,n,cfg.seed), expected=-1;
                if(atomic_compare_exchange_strong(&dist[v],&expected,level+1)) changed=1;
            }
        }
        level++;
    }
    double elapsed=now()-t, *values=allocate(n,sizeof(double));
    for(int i=0;i<n;i++) values[i]=atomic_load(&dist[i]);
    emit(values,n,cfg.full,elapsed,0); free(dist); free(values); return 0;
}
