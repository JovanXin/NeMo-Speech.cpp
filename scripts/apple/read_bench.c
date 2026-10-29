#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
// rd FILE MODE CHUNK_KB THREADS : read 2 GB of random CHUNK-sized slices, nocache
static int fd; static char* map; static size_t fsz, chunk; static int mode; static long per;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void* work(void* a){ unsigned s=(unsigned)(size_t)a*7919+1; char* buf=malloc(chunk); volatile long sum=0;
  for(long i=0;i<per;i++){ size_t off=((size_t)rand_r(&s)*4096*13)%(fsz-chunk); off&=~(size_t)16383;
    if(mode==0){ pread(fd,buf,chunk,off);} else { if(mode==2) madvise(map+off,chunk,MADV_WILLNEED); for(size_t j=0;j<chunk;j+=16384) sum+=map[off+j]; } }
  return 0;}
int main(int c,char**v){ fd=open(v[1],O_RDONLY); fcntl(fd,F_NOCACHE,1); fsz=lseek(fd,0,SEEK_END); mode=atoi(v[2]); chunk=atol(v[3])*1024; int th=atoi(v[4]);
  map=mmap(0,fsz,PROT_READ,MAP_SHARED,fd,0); size_t total=(size_t)2<<30; per=total/chunk/th; pthread_t t[64]; double t0=now();
  for(int i=0;i<th;i++) pthread_create(&t[i],0,work,(void*)(size_t)i); for(int i=0;i<th;i++) pthread_join(t[i],0);
  double dt=now()-t0; printf("mode=%s chunk=%sKB threads=%d  %.0f MB/s\n", mode==0?"pread":mode==1?"mmap-fault":"mmap+willneed", v[3], th, per*th*chunk/dt/1e6); }
