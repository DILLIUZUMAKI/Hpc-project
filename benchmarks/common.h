#ifndef HPC_COMMON_H
#define HPC_COMMON_H
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct { int n, steps, full; uint32_t seed; } Config;
static unsigned long long integer(const char *s) {
    char *end; errno=0;
    unsigned long long v=strtoull(s,&end,10);
    if(errno || !*s || *end || *s=='-') { fprintf(stderr,"Invalid integer\n"); exit(2); }
    return v;
}
static Config config(int argc,char **argv,int min_n,int max_n) {
    if(argc!=5) { fprintf(stderr,"Usage: program N STEPS SEED --check|--bench\n"); exit(2); }
    unsigned long long n=integer(argv[1]), t=integer(argv[2]), s=integer(argv[3]);
    if(n<(unsigned)min_n || n>(unsigned)max_n || t<1 || t>1000 || s>UINT32_MAX ||
       (strcmp(argv[4],"--check") && strcmp(argv[4],"--bench"))) {
        fprintf(stderr,"Arguments outside supported range\n"); exit(2);
    }
    return (Config){(int)n,(int)t,!strcmp(argv[4],"--check"),(uint32_t)s};
}
static void *allocate(size_t n,size_t size) {
    if(n && size>SIZE_MAX/n) { fprintf(stderr,"Allocation overflow\n"); exit(3); }
    void *p=calloc(n?n:1,size);
    if(!p) { fprintf(stderr,"Allocation failed\n"); exit(3); }
    return p;
}
static uint32_t mix32(uint32_t x) {
    x^=x>>16; x*=UINT32_C(0x7feb352d); x^=x>>15; x*=UINT32_C(0x846ca68b); x^=x>>16;
    return x;
}
static double initial(uint32_t i,uint32_t seed) { return (mix32(seed+i)%100)/100.0; }
static int neighbor(int u,int k,int n,uint32_t seed) {
    return k==0 ? (u+1)%n : (int)(mix32(seed+3u*(uint32_t)u+(uint32_t)k)%(uint32_t)n);
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9;
}
static void emit(const double *a,size_t count,int full,double seconds,double comm) {
    double sum=0,squares=0,weighted=0;
    for(size_t i=0;i<count;i++) { sum+=a[i]; squares+=a[i]*a[i]; weighted+=a[i]*(i%97+1); }
    printf("{\"length\":%zu,\"sum\":%.17g,\"sum_squares\":%.17g,\"weighted_sum\":%.17g,\"kernel_seconds\":%.17g,\"communication_seconds\":%.17g",
           count,sum,squares,weighted,seconds,comm);
    if(full) { printf(",\"values\":["); for(size_t i=0;i<count;i++) printf("%s%.17g",i?",":"",a[i]); printf("]"); }
    printf("}\n");
}
#endif
