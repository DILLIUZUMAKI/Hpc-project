/* Curated project reference; assistant-authored, not a human-study baseline. */
#include "common.h"
int main(int argc,char **argv) {
    Config cfg=config(argc,argv,1,200000); int n=cfg.n;
    int *dist=allocate(n,sizeof(int)), *queue=allocate(n,sizeof(int));
    for(int i=0;i<n;i++) dist[i]=-1;
    dist[0]=0; int head=0,tail=0; queue[tail++]=0; double t=now();
    while(head<tail) {
        int u=queue[head++];
        for(int k=0;k<4;k++) {
            int v=neighbor(u,k,n,cfg.seed);
            if(dist[v]<0) { dist[v]=dist[u]+1; queue[tail++]=v; }
        }
    }
    double elapsed=now()-t, *values=allocate(n,sizeof(double));
    for(int i=0;i<n;i++) values[i]=dist[i];
    emit(values,n,cfg.full,elapsed,0); free(dist); free(queue); free(values); return 0;
}
