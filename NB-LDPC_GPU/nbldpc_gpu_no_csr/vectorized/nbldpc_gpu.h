/* #ifndef DEF_H
#define DEF_H


typedef struct __align__(16) {
    signed char s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
}char16;


#endif */

#include <cuda_runtime.h> 

#if Q==32
    __global__ void GPU_VN( unsigned char * d_beta,  unsigned char * d_alpha,  unsigned char * d_alpha_t, unsigned char * d_alpha_t2,unsigned char * d_gamma, int iter);

    __global__ void GPU_CN( unsigned char * d_beta,  unsigned char * d_F,  unsigned char * d_B, int iter, unsigned char * d_h_nb);

    __global__ void GPU_FB_metrics(unsigned char * d_alpha, unsigned char * d_F, unsigned char * d_B, int iter, unsigned char * d_h_nb);

#else
    __global__ void GPU_VN( unsigned char * d_beta,  unsigned char * d_alpha,  unsigned char * d_alpha_t, unsigned char * d_alpha_t2,unsigned char * d_gamma, int iter);

    __global__ void GPU_CN( uchar4 * d_beta,  uchar4 * d_F,  uchar4 * d_B, int iter);

    __global__ void GPU_FB_metrics( uchar4 * d_alpha,  uchar4 * d_F,  uchar4 * d_B, int iter);
#endif

extern "C" int cuda_minmax(unsigned char* h_beta, unsigned char* ALPHAmn_, unsigned char* GAMMAn_, unsigned char* h_F,unsigned char* h_B,int* iteration, int* decoded_bit);
