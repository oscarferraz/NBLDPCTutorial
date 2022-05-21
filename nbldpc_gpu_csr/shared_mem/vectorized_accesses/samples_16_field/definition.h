#ifndef DEF_H
#define DEF_H

#include <cuda_runtime.h> 

typedef struct __align__(16) {
    unsigned short s0, s1, s2, s3, s4, s5, s6, s7;
}ushort8;

typedef struct __align__(16) {
    unsigned int s0, s1, s2, s3, s4, s5, s6, s7;
}uint8;

typedef struct __align__(16) {
    signed int s0, s1, s2, s3, s4, s5, s6, s7;
}int8;

typedef struct __align__(16) {
    unsigned char s0, s1, s2, s3, s4, s5, s6, s7;
}uchar8;

typedef struct __align__(16) {
    unsigned char s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
}uchar16;

#endif