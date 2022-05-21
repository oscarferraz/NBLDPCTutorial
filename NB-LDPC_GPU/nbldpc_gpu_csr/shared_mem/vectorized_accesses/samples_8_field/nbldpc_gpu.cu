
#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include <time.h>
#include <string.h>
//#include <ap_fixed.h>
#include "nbldpc.h"
#include <cuda_runtime.h>      // CUDA Runtime Functions
#include <helper_cuda.h>

extern "C" {
    #include "nbldpc_gpu.h"
}


#ifdef VS
#define _CRT_SECURE_NO_DEPRECATE
#endif


__constant__ unsigned char c_add[Q][Q];
__constant__ unsigned char c_mult[Q][Q];
__constant__ unsigned char c_inv[Q];


__constant__ unsigned short	c_row_ptr[257];
__constant__ unsigned char	c_val[768];
__constant__ unsigned short	c_ptr_to_val[768];
__constant__ unsigned short	c_col_ptr[385];
__constant__ unsigned short	c_row_ind[768];
__constant__ unsigned short	c_col_ind[768];





__constant__ unsigned short c_M;
__constant__ unsigned short c_N;
__constant__ unsigned char c_w_row;
__constant__ unsigned char c_w_col;

extern unsigned char *h_F;
extern unsigned char *h_B;



    //===================================
	// CUDA kernel
	//===================================
	__global__ void GPU_FB_metrics( uchar8 * d_alpha_8,  uchar8 * d_F_8,  uchar8 * d_B_8, int iter){
		
		unsigned short x=threadIdx.x+blockIdx.x*blockDim.x;
		uchar8 min_F[1];
		uchar8 max_F;
		uchar8 min_B[1];
		uchar8 max_B;
		uchar8 a_F;
		uchar8 a_B;

		#if Q==8
		__shared__ uchar8 s_alpha[128*3];
		__shared__ uchar8 s_F[128*3];
		__shared__ uchar8 s_B[128*3];
		#elif Q==16
		__shared__ uchar16 s_alpha[8*3*16];
		__shared__ uchar16 s_F[8*16*3];
		__shared__ uchar16 s_B[8*16*3];
		#elif Q==32
		__shared__ uchar16 s_alpha[8*3*32];
		__shared__ uchar16 s_F[8*32*3];
		__shared__ uchar16 s_B[8*32*3];
		#endif

		//if(x<c_M){

			for(unsigned char index=0; index<c_w_row; index++){
				reinterpret_cast<uchar8*>(s_alpha)[(threadIdx.x*c_w_row)+index]=reinterpret_cast<uchar8*>(d_alpha_8)[(x*c_w_row)+index];
				/* if(x==0){
					printf("s_alpha[%d][%d][0]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0, d_alpha_8[(x*c_w_row)+index].s0);  
					printf("s_alpha[%d][%d][1]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s1, d_alpha_8[(x*c_w_row)+index].s1);  
					printf("s_alpha[%d][%d][2]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s2, d_alpha_8[(x*c_w_row)+index].s2);  
					printf("s_alpha[%d][%d][3]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s3, d_alpha_8[(x*c_w_row)+index].s3);
					printf("s_alpha[%d][%d][4]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s4, d_alpha_8[(x*c_w_row)+index].s4);  
					printf("s_alpha[%d][%d][5]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s5, d_alpha_8[(x*c_w_row)+index].s5);  
					printf("s_alpha[%d][%d][6]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s6, d_alpha_8[(x*c_w_row)+index].s6);  
					printf("s_alpha[%d][%d][7]=%d col=%d\n", threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s7, d_alpha_8[(x*c_w_row)+index].s7);  
				}  */
			}
			__syncthreads();

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==0)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==1)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==2)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==3)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==4)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==5)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==6)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][0]==7)
				s_F[threadIdx.x*c_w_row].s0=s_alpha[threadIdx.x*c_w_row].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==0)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==1)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==2)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==3)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==4)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==5)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==6)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][1]==7)
				s_F[threadIdx.x*c_w_row].s1=s_alpha[threadIdx.x*c_w_row].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==0)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==1)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==2)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==3)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==4)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==5)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==6)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][2]==7)
				s_F[threadIdx.x*c_w_row].s2=s_alpha[threadIdx.x*c_w_row].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==0)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==1)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==2)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==3)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==4)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==5)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==6)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][3]==7)
				s_F[threadIdx.x*c_w_row].s3=s_alpha[threadIdx.x*c_w_row].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==0)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==1)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==2)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==3)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==4)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==5)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==6)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][4]==7)
				s_F[threadIdx.x*c_w_row].s4=s_alpha[threadIdx.x*c_w_row].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==0)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==1)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==2)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==3)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==4)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==5)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==6)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][5]==7)
				s_F[threadIdx.x*c_w_row].s5=s_alpha[threadIdx.x*c_w_row].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==0)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==1)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==2)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==3)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==4)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==5)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==6)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][6]==7)
				s_F[threadIdx.x*c_w_row].s6=s_alpha[threadIdx.x*c_w_row].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==0)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==1)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==2)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==3)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==4)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==5)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==6)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x]]]][7]==7)
				s_F[threadIdx.x*c_w_row].s7=s_alpha[threadIdx.x*c_w_row].s7;

			/* if(x==0){
				printf("s_alpha[%d][0]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s0, c_mult[c_inv[c_val[c_row_ptr[x]]]][0]);  
				printf("s_alpha[%d][1]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s1, c_mult[c_inv[c_val[c_row_ptr[x]]]][1]);  
				printf("s_alpha[%d][2]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s2, c_mult[c_inv[c_val[c_row_ptr[x]]]][2]);  
				printf("s_alpha[%d][3]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s3, c_mult[c_inv[c_val[c_row_ptr[x]]]][3]); 
				printf("s_alpha[%d][4]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s4, c_mult[c_inv[c_val[c_row_ptr[x]]]][4]);  
				printf("s_alpha[%d][5]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s5, c_mult[c_inv[c_val[c_row_ptr[x]]]][5]);  
				printf("s_alpha[%d][6]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s6, c_mult[c_inv[c_val[c_row_ptr[x]]]][6]);  
				printf("s_alpha[%d][7]=%d mult=%d\n", threadIdx.x, s_F[threadIdx.x*c_w_row].s7, c_mult[c_inv[c_val[c_row_ptr[x]]]][7]);   
			} */


			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][0]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;


			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][1]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][2]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][3]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][4]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s4=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][5]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s5=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][6]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s6=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;

			if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==0)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s0;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==1)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s1;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==2)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s2;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==3)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s3;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==4)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s4;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==5)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s5;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==6)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s6;
			else if(c_mult[c_inv[c_val[c_row_ptr[x+1]-1]]][7]==7)
				s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s7=s_alpha[(threadIdx.x*c_w_row)+(c_w_row-1)].s7;
			/* if(x==6){
				printf("s_B[%d][0][%d]=%d \n", (threadIdx.x*c_w_row), (c_w_row-1), s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s0);  
				printf("s_B[%d][1][%d]=%d \n", (threadIdx.x*c_w_row), (c_w_row-1), s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s1);  
				printf("s_B[%d][2][%d]=%d \n", (threadIdx.x*c_w_row), (c_w_row-1), s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s2);  
				printf("s_B[%d][3][%d]=%d \n", (threadIdx.x*c_w_row), (c_w_row-1), s_B[(threadIdx.x*c_w_row)+(c_w_row-1)].s3);  
			} */

			reinterpret_cast<uchar8*>(d_F_8)[(x*c_w_row)]=reinterpret_cast<uchar8*>(s_F)[(threadIdx.x*c_w_row)];                   
			/* if(x==0){
				printf("s_F[%d][0]=%d \n", x, d_F_8[(x*c_w_row)].s0);  
				printf("s_F[%d][1]=%d \n", x, d_F_8[(x*c_w_row)].s1);  
				printf("s_F[%d][2]=%d \n", x, d_F_8[(x*c_w_row)].s2);  
				printf("s_F[%d][3]=%d \n", x, d_F_8[(x*c_w_row)].s3);  
				printf("s_F[%d][4]=%d \n", x, d_F_8[(x*c_w_row)].s4);  
				printf("s_F[%d][5]=%d \n", x, d_F_8[(x*c_w_row)].s5);  
				printf("s_F[%d][6]=%d \n", x, d_F_8[(x*c_w_row)].s6);  
				printf("s_F[%d][7]=%d \n", x, d_F_8[(x*c_w_row)].s7);  
			} */

			reinterpret_cast<uchar8*>(d_B_8)[(x*c_w_row)+(c_w_row-1)]=reinterpret_cast<uchar8*>(s_B)[(threadIdx.x*c_w_row)+(c_w_row-1)];                   
			__syncthreads();

			for(unsigned char index=1; index<c_w_row; index++){
				
				min_F[0].s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
				min_F[0].s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
				min_F[0].s2 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
				min_F[0].s3 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
				min_F[0].s4 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
				min_F[0].s5 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
				min_F[0].s6 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
				min_F[0].s7 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
				/* if(x==0){
					printf("min_F=%d, s_F[%d][0][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s0, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s0,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );  
					printf("min_F=%d, s_F[%d][1][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s1, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s1,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );  
					printf("min_F=%d, s_F[%d][2][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s2, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s2,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );  
					printf("min_F=%d, s_F[%d][3][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s3, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s3,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );
					printf("min_F=%d, s_F[%d][4][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s4, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s4,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );  
					printf("min_F=%d, s_F[%d][5][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s5, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s5,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );  
					printf("min_F=%d, s_F[%d][6][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s6, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s6,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );  
					printf("min_F=%d, s_F[%d][7][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_F[0].s7, threadIdx.x, index-1,s_F[(threadIdx.x*c_w_row)+index-1].s7,threadIdx.x, index, s_alpha[(threadIdx.x*c_w_row)+index].s0 );  
				} */

				min_B[0].s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
				min_B[0].s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
				min_B[0].s2 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
				min_B[0].s3 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
				min_B[0].s4 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
				min_B[0].s5 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
				min_B[0].s6 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
				min_B[0].s7 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
				/* if(x==0){
					printf("min_B=%d, s_B[%d][0][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_B[0].s0, threadIdx.x, c_w_row-index, s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0, threadIdx.x, c_w_row-index-1, s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0 );  
					printf("min_B=%d, s_B[%d][1][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_B[0].s1, threadIdx.x, c_w_row-index, s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1, threadIdx.x, c_w_row-index-1, s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0 );  
					printf("min_B=%d, s_B[%d][2][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_B[0].s2, threadIdx.x, c_w_row-index, s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2, threadIdx.x, c_w_row-index-1, s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0 );  
					printf("min_B=%d, s_B[%d][3][%d]=%d, s_alpha[%d][%d][0]=%d\n", min_B[0].s3, threadIdx.x, c_w_row-index, s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3, threadIdx.x, c_w_row-index-1, s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0 );  
				} */

				for(unsigned char b=1; b<Q; b++){
					a_F.s0 = c_add[0][c_mult[c_val[c_row_ptr[x]+index]][b]];
					a_F.s1 = c_add[1][c_mult[c_val[c_row_ptr[x]+index]][b]];
					a_F.s2 = c_add[2][c_mult[c_val[c_row_ptr[x]+index]][b]];
					a_F.s3 = c_add[3][c_mult[c_val[c_row_ptr[x]+index]][b]];
					a_F.s4 = c_add[4][c_mult[c_val[c_row_ptr[x]+index]][b]];
					a_F.s5 = c_add[5][c_mult[c_val[c_row_ptr[x]+index]][b]];
					a_F.s6 = c_add[6][c_mult[c_val[c_row_ptr[x]+index]][b]];
					a_F.s7 = c_add[7][c_mult[c_val[c_row_ptr[x]+index]][b]];
					/* if(x==0){
						printf("a_F=%d, z=0, index=%d, b=%d\n", a_F.s0, index, b );  
						printf("a_F=%d, z=1, index=%d, b=%d\n", a_F.s1, index, b );  
						printf("a_F=%d, z=2, index=%d, b=%d\n", a_F.s2, index, b );  
						printf("a_F=%d, z=3, index=%d, b=%d\n", a_F.s3, index, b );  
						printf("a_F=%d, z=4, index=%d, b=%d\n", a_F.s4, index, b );  
						printf("a_F=%d, z=5, index=%d, b=%d\n", a_F.s5, index, b );  
						printf("a_F=%d, z=6, index=%d, b=%d\n", a_F.s6, index, b );  
						printf("a_F=%d, z=7, index=%d, b=%d\n", a_F.s7, index, b );  
					} */

					a_B.s0 = c_add[0][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					a_B.s1 = c_add[1][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					a_B.s2 = c_add[2][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					a_B.s3 = c_add[3][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					a_B.s4 = c_add[4][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					a_B.s5 = c_add[5][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					a_B.s6 = c_add[6][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					a_B.s7 = c_add[7][c_mult[c_val[c_row_ptr[x]+c_row_ptr[x+1]-(c_row_ptr[x]+index)-1]][b]];
					/* if(x==0){
						printf("a_B=%d, z=0, index=%d, b=%d\n", a_B.x, index, b );  
						printf("a_B=%d, z=1, index=%d, b=%d\n", a_B.y, index, b );  
						printf("a_B=%d, z=2, index=%d, b=%d\n", a_B.z, index, b );  
						printf("a_B=%d, z=3, index=%d, b=%d\n", a_B.w, index, b );  
					} */
					
					// FORWARD
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S0
						if(a_F.s0==0){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s0==1){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s0==2){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s0==3){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s0==4){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s0==5){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s0==6){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s0==7){
							if(b==0)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s0 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S1
						if(a_F.s1==0){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s1==1){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s1==2){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s1==3){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s1==4){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s1==5){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s1==6){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s1==7){
							if(b==0)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s1 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S2
						if(a_F.s2==0){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s2==1){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s2==2){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s2==3){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s2==4){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s2==5){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s2==6){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s2==7){
							if(b==0)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s2= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s2 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S3
						if(a_F.s3==0){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s3==1){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s3==2){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s3==3){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s3==4){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s3==5){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s3==6){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s3==7){
							if(b==0)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s3= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s3 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S4
						if(a_F.s4==0){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s4==1){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s4==2){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s4==3){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s4==4){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s4==5){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s4==6){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s4==7){
							if(b==0)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s4= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S5
						if(a_F.s5==0){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s5==1){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s5==2){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s5==3){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s5==4){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s5==5){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s5==6){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s5==7){
							if(b==0)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s5= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s5 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S6
						if(a_F.s6==0){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s6==1){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s6==2){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s6==3){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s6==4){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s6==5){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s6==6){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s6==7){
							if(b==0)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s6= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S7
						if(a_F.s7==0){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else  if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s0;
							else if(b==7)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s0 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s0;
						}
						else if(a_F.s7==1){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s1;
							else if(b==7)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s1 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s1;
						}
						else if(a_F.s7==2){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s2;
							else if(b==7)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s2 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s2;
						}
						else if(a_F.s7==3){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s3;
							else if(b==7)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s3 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s3;
						}

						else if(a_F.s7==4){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else  if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s4;
							else if(b==7)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s4 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s4;
						}
						else if(a_F.s7==5){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s5;
							else if(b==7)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s5 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s5;
						}
						else if(a_F.s7==6){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s6;
							else if(b==7)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s6 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s6;
						}
						else if(a_F.s7==7){
							if(b==0)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s0) ? s_alpha[(threadIdx.x*c_w_row)+index].s0: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==1)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s1) ? s_alpha[(threadIdx.x*c_w_row)+index].s1: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==2)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s2) ? s_alpha[(threadIdx.x*c_w_row)+index].s2: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==3)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s3) ? s_alpha[(threadIdx.x*c_w_row)+index].s3: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==4)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s4) ? s_alpha[(threadIdx.x*c_w_row)+index].s4: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==5)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s5) ? s_alpha[(threadIdx.x*c_w_row)+index].s5: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==6)
								max_F.s7= (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s6) ? s_alpha[(threadIdx.x*c_w_row)+index].s6: s_F[(threadIdx.x*c_w_row)+index-1].s7;
							else if(b==7)
								max_F.s7 = (s_F[(threadIdx.x*c_w_row)+index-1].s7 < s_alpha[(threadIdx.x*c_w_row)+index].s7) ? s_alpha[(threadIdx.x*c_w_row)+index].s7: s_F[(threadIdx.x*c_w_row)+index-1].s7;
						}
					/* if(x==0){
						printf("max_F=%d, z=0, index=%d, b=%d\n", max_F.s0, index, b );  
						printf("max_F=%d, z=1, index=%d, b=%d\n", max_F.s1, index, b );  
						printf("max_F=%d, z=2, index=%d, b=%d\n", max_F.s2, index, b );  
						printf("max_F=%d, z=3, index=%d, b=%d\n", max_F.s3, index, b );  
						printf("max_F=%d, z=4, index=%d, b=%d\n", max_F.s4, index, b );  
						printf("max_F=%d, z=5, index=%d, b=%d\n", max_F.s5, index, b );  
						printf("max_F=%d, z=6, index=%d, b=%d\n", max_F.s6, index, b );  
						printf("max_F=%d, z=7, index=%d, b=%d\n", max_F.s7, index, b );  
					} */

					// BACKWARD
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S0
						if(a_B.s0==0){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s0==1){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s0==2){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s0==3){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s0==4){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s0==5){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s0==6){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s0==7){
							if(b==0)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s0 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S1
						if(a_B.s1==0){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s1==1){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s1==2){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s1==3){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s1==4){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s1==5){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s1==6){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s1==7){
							if(b==0)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s1 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S2
						if(a_B.s2==0){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s2==1){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s2==2){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s2==3){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s2==4){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s2==5){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s2==6){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s2==7){
							if(b==0)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s2= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s2 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S3
						if(a_B.s3==0){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s3==1){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s3==2){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s3==3){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s3==4){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s3==5){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s3==6){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s3==7){
							if(b==0)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s3= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s3 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S4
						if(a_B.s4==0){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s4==1){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s4==2){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s4==3){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s4==4){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s4==5){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s4==6){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s4==7){
							if(b==0)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s4= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s4 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S5
						if(a_B.s5==0){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s5==1){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s5==2){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s5==3){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s5==4){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s5==5){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s5==6){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s5==7){
							if(b==0)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s5= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s5 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S6
						if(a_B.s6==0){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s6==1){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s6==2){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s6==3){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s6==4){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s6==5){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s6==6){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s6==7){
							if(b==0)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s6= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s6 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S7
						if(a_B.s7==0){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else  if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
							else if(b==7)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s0;
						}
						else if(a_B.s7==1){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
							else if(b==7)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s1;
						}
						else if(a_B.s7==2){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
							else if(b==7)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s2;
						}
						else if(a_B.s7==3){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
							else if(b==7)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s3;
						}

						else if(a_B.s7==4){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else  if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
							else if(b==7)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s4;
						}
						else if(a_B.s7==5){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
							else if(b==7)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s5;
						}
						else if(a_B.s7==6){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
							else if(b==7)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s6;
						}
						else if(a_B.s7==7){
							if(b==0)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s0: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==1)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s1: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==2)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s2: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==3)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s3: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==4)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s4: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==5)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s5: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==6)
								max_B.s7= (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s6: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
							else if(b==7)
								max_B.s7 = (s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7 < s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7) ? s_alpha[(threadIdx.x*c_w_row)+(c_w_row-index-1)].s7: s_B[(threadIdx.x*c_w_row)+c_w_row-index].s7;
						}

					/* if(x==0){
						printf("max_B=%d, z=0, index=%d, b=%d\n", max_B.x, index, b );  
						printf("max_B=%d, z=1, index=%d, b=%d\n", max_B.y, index, b );  
						printf("max_B=%d, z=2, index=%d, b=%d\n", max_B.z, index, b );  
						printf("max_B=%d, z=3, index=%d, b=%d\n", max_B.w, index, b );  
					} */

					min_F[0].s0 = (min_F[0].s0 < max_F.s0) ? min_F[0].s0: max_F.s0;
					min_F[0].s1 = (min_F[0].s1 < max_F.s1) ? min_F[0].s1: max_F.s1;
					min_F[0].s2 = (min_F[0].s2 < max_F.s2) ? min_F[0].s2: max_F.s2;
					min_F[0].s3 = (min_F[0].s3 < max_F.s3) ? min_F[0].s3: max_F.s3;
					min_F[0].s4 = (min_F[0].s4 < max_F.s4) ? min_F[0].s4: max_F.s4;
					min_F[0].s5 = (min_F[0].s5 < max_F.s5) ? min_F[0].s5: max_F.s5;
					min_F[0].s6 = (min_F[0].s6 < max_F.s6) ? min_F[0].s6: max_F.s6;
					min_F[0].s7 = (min_F[0].s7 < max_F.s7) ? min_F[0].s7: max_F.s7;
					/* if(x==0){
						printf("min_F[0]=%d, max_F=%d, z=0, index=%d, b=%d\n", min_F[0].s0, max_F.s0, index, b );  
						printf("min_F[0]=%d, max_F=%d, z=1, index=%d, b=%d\n", min_F[0].s1, max_F.s1, index, b );  
						printf("min_F[0]=%d, max_F=%d, z=2, index=%d, b=%d\n", min_F[0].s2, max_F.s2, index, b );  
						printf("min_F[0]=%d, max_F=%d, z=3, index=%d, b=%d\n", min_F[0].s3, max_F.s3, index, b ); 
						printf("min_F[0]=%d, max_F=%d, z=4, index=%d, b=%d\n", min_F[0].s4, max_F.s4, index, b );  
						printf("min_F[0]=%d, max_F=%d, z=5, index=%d, b=%d\n", min_F[0].s5, max_F.s5, index, b );  
						printf("min_F[0]=%d, max_F=%d, z=6, index=%d, b=%d\n", min_F[0].s6, max_F.s6, index, b );  
						printf("min_F[0]=%d, max_F=%d, z=7, index=%d, b=%d\n", min_F[0].s7, max_F.s7, index, b );  
					} */

					min_B[0].s0 = (min_B[0].s0 < max_B.s0) ? min_B[0].s0: max_B.s0;
					min_B[0].s1 = (min_B[0].s1 < max_B.s1) ? min_B[0].s1: max_B.s1;
					min_B[0].s2 = (min_B[0].s2 < max_B.s2) ? min_B[0].s2: max_B.s2;
					min_B[0].s3 = (min_B[0].s3 < max_B.s3) ? min_B[0].s3: max_B.s3;
					min_B[0].s4 = (min_B[0].s4 < max_B.s4) ? min_B[0].s4: max_B.s4;
					min_B[0].s5 = (min_B[0].s5 < max_B.s5) ? min_B[0].s5: max_B.s5;
					min_B[0].s6 = (min_B[0].s6 < max_B.s6) ? min_B[0].s6: max_B.s6;
					min_B[0].s7 = (min_B[0].s7 < max_B.s7) ? min_B[0].s7: max_B.s7;
					/* if(x==6){
						printf("min_F[0]=%d, z=0, index=%d, b=%d\n", min_B[0].s0, index, b );  
						printf("min_F[0]=%d, z=1, index=%d, b=%d\n", min_B[0].s1, index, b );  
						printf("min_F[0]=%d, z=2, index=%d, b=%d\n", min_B[0].s2, index, b );  
						printf("min_F[0]=%d, z=3, index=%d, b=%d\n", min_B[0].s3, index, b );  
					} */

				}
				__syncthreads();

				reinterpret_cast<uchar8*>(d_F_8)[(x*c_w_row)+index]=reinterpret_cast<uchar8*>(min_F)[0];                   
				s_F[(threadIdx.x*c_w_row)+index].s0 = min_F[0].s0;
				s_F[(threadIdx.x*c_w_row)+index].s1 = min_F[0].s1;
				s_F[(threadIdx.x*c_w_row)+index].s2 = min_F[0].s2;
				s_F[(threadIdx.x*c_w_row)+index].s3 = min_F[0].s3;
				s_F[(threadIdx.x*c_w_row)+index].s4 = min_F[0].s4;
				s_F[(threadIdx.x*c_w_row)+index].s5 = min_F[0].s5;
				s_F[(threadIdx.x*c_w_row)+index].s6 = min_F[0].s6;
				s_F[(threadIdx.x*c_w_row)+index].s7 = min_F[0].s7;
				/* if(x==0){
					printf("d_F_8[%d][0][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s0);  
					printf("d_F_8[%d][1][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s1);  
					printf("d_F_8[%d][2][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s2);  
					printf("d_F_8[%d][3][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s3); 
					printf("d_F_8[%d][4][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s4);  
					printf("d_F_8[%d][5][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s5);  
					printf("d_F_8[%d][6][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s6);  
					printf("d_F_8[%d][7][%d]=%d\n", x, index, d_F_8[(x*c_w_row)+index].s7);   
				} */

				reinterpret_cast<uchar8*>(d_B_8)[(x*c_w_row)+c_w_row - index - 1]=reinterpret_cast<uchar8*>(min_B)[0];                   
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s0 = min_B[0].s0;
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s1 = min_B[0].s1;
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s2 = min_B[0].s2;
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s3 = min_B[0].s3;
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s4 = min_B[0].s4;
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s5 = min_B[0].s5;
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s6 = min_B[0].s6;
				s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s7 = min_B[0].s7;
				/* if(x==6){
					printf("s_F[%d][0][%d]=%d\n", threadIdx.x, c_w_row - index - 1, s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s0);  
					printf("s_F[%d][1][%d]=%d\n", threadIdx.x, c_w_row - index - 1, s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s1);  
					printf("s_F[%d][2][%d]=%d\n", threadIdx.x, c_w_row - index - 1, s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s2);  
					printf("s_F[%d][3][%d]=%d\n", threadIdx.x, c_w_row - index - 1, s_B[(threadIdx.x*c_w_row)+c_w_row - index - 1].s3);  
				} */
			
			}  
		//}

		

	} 

	__global__ void GPU_CN( uchar8 * d_beta_8,  uchar8 * d_F_8,  uchar8 * d_B_8, int iter){
		unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;		

		#if Q==8
		__shared__ uchar8 s_F[32*3];
		__shared__ uchar8 s_B[32*3];
		#elif Q==16
		__shared__ uchar16 s_F[4*16*3];
		__shared__ uchar16 s_B[4*16*3];
		#elif Q==32
		__shared__ uchar16 s_F[4*32*3];
		__shared__ uchar16 s_B[4*32*3];
		#endif

		//if(x<c_M){

			reinterpret_cast<uchar8*>(s_F)[(threadIdx.x*c_w_row)+y]=reinterpret_cast<uchar8*>(d_F_8)[(x*c_w_row)+y];
			reinterpret_cast<uchar8*>(s_B)[(threadIdx.x*c_w_row)+y]=reinterpret_cast<uchar8*>(d_B_8)[(x*c_w_row)+y];
			__syncthreads();
			/* if(x==1){
				printf("s_F[%d][0][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s0);  
				printf("s_F[%d][1][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s1);  
				printf("s_F[%d][2][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s2);  
				printf("s_F[%d][3][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s3);  
				printf("s_F[%d][4][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s4);  
				printf("s_F[%d][5][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s5);  
				printf("s_F[%d][6][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s6);  
				printf("s_F[%d][7][%d]=%d\n", threadIdx.x, y, s_F[(threadIdx.x*c_w_row)+y].s7); 
			}

			if(x==1){
				printf("s_B[%d][0][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s0);  
				printf("s_B[%d][1][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s1);  
				printf("s_B[%d][2][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s2);  
				printf("s_B[%d][3][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s3);  
				printf("s_B[%d][4][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s4);  
				printf("s_B[%d][5][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s5);  
				printf("s_B[%d][6][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s6);  
				printf("s_B[%d][7][%d]=%d\n", threadIdx.x, y, s_B[(threadIdx.x*c_w_row)+y].s7); 
			} */

			if(y==0){		
				if(c_mult[c_val[c_row_ptr[x]]][0]==0)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][0]==1)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][0]==2)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][0]==3)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][0]==4)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][0]==5)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][0]==6)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][0]==7)
					d_beta_8[(x*c_w_row)+y].s0=s_B[(threadIdx.x*c_w_row)+1].s7;
	
	
				if(c_mult[c_val[c_row_ptr[x]]][1]==0)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][1]==1)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][1]==2)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][1]==3)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][1]==4)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][1]==5)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][1]==6)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][1]==7)
					d_beta_8[(x*c_w_row)+y].s1=s_B[(threadIdx.x*c_w_row)+1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]]][2]==0)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][2]==1)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][2]==2)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][2]==3)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][2]==4)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][2]==5)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][2]==6)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][2]==7)
					d_beta_8[(x*c_w_row)+y].s2=s_B[(threadIdx.x*c_w_row)+1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]]][3]==0)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][3]==1)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][3]==2)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][3]==3)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][3]==4)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][3]==5)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][3]==6)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][3]==7)
					d_beta_8[(x*c_w_row)+y].s3=s_B[(threadIdx.x*c_w_row)+1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]]][4]==0)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][4]==1)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][4]==2)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][4]==3)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][4]==4)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][4]==5)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][4]==6)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][4]==7)
					d_beta_8[(x*c_w_row)+y].s4=s_B[(threadIdx.x*c_w_row)+1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]]][5]==0)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][5]==1)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][5]==2)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][5]==3)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][5]==4)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][5]==5)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][5]==6)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][5]==7)
					d_beta_8[(x*c_w_row)+y].s5=s_B[(threadIdx.x*c_w_row)+1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]]][6]==0)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][6]==1)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][6]==2)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][6]==3)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][6]==4)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][6]==5)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][6]==6)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][6]==7)
					d_beta_8[(x*c_w_row)+y].s6=s_B[(threadIdx.x*c_w_row)+1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]]][7]==0)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s0;
				else if(c_mult[c_val[c_row_ptr[x]]][7]==1)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s1;
				else if(c_mult[c_val[c_row_ptr[x]]][7]==2)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s2;
				else if(c_mult[c_val[c_row_ptr[x]]][7]==3)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s3;
				else if(c_mult[c_val[c_row_ptr[x]]][7]==4)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s4;
				else if(c_mult[c_val[c_row_ptr[x]]][7]==5)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s5;
				else if(c_mult[c_val[c_row_ptr[x]]][7]==6)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s6;
				else if(c_mult[c_val[c_row_ptr[x]]][7]==7)
					d_beta_8[(x*c_w_row)+y].s7=s_B[(threadIdx.x*c_w_row)+1].s7;
				/* if(x==3){
					printf("BETAmn_[%d][%d][0]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].x);  
					printf("BETAmn_[%d][%d][1]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].y);  
					printf("BETAmn_[%d][%d][2]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].z);  
					printf("BETAmn_[%d][%d][3]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].w);  
				} */

			}
			else if(y==c_w_row-1){
				if(c_mult[c_val[c_row_ptr[x]+y]][0]==0)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==1)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==2)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==3)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==4)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==5)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==6)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==7)
					d_beta_8[(x*c_w_row)+y].s0=s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
	
				if(c_mult[c_val[c_row_ptr[x]+y]][1]==0)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==1)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==2)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==3)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==4)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==5)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==6)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==7)
					d_beta_8[(x*c_w_row)+y].s1=s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][2]==0)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==1)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==2)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==3)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==4)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==5)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==6)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==7)
					d_beta_8[(x*c_w_row)+y].s2=s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][3]==0)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==1)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==2)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==3)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==4)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==5)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==6)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==7)
					d_beta_8[(x*c_w_row)+y].s3=s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][4]==0)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==1)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==2)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==3)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==4)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==5)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==6)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==7)
					d_beta_8[(x*c_w_row)+y].s4=s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][5]==0)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==1)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==2)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==3)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==4)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==5)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==6)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==7)
					d_beta_8[(x*c_w_row)+y].s5=s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][6]==0)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==1)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==2)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==3)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==4)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==5)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==6)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==7)
					d_beta_8[(x*c_w_row)+y].s6=s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][7]==0)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==1)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==2)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==3)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==4)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==5)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==6)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==7)
					d_beta_8[(x*c_w_row)+y].s7=s_F[(threadIdx.x*c_w_row)+y-1].s7;
				/* if(x==3){
					printf("BETAmn_[%d][%d][0]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].x);  
					printf("BETAmn_[%d][%d][1]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].y);  
					printf("BETAmn_[%d][%d][2]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].z);  
					printf("BETAmn_[%d][%d][3]=%d\n", x, y, d_beta_8[(x*c_w_row)+y].w);  
				} */

			}
			else{
				uchar8 min[1];
				uchar8 max;
				uchar8 a;

				if(c_mult[c_val[c_row_ptr[x]+y]][0]==0)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==1)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==2)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==3)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==4)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==5)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==6)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][0]==7)
					min[0].s0=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
	
				if(c_mult[c_val[c_row_ptr[x]+y]][1]==0)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==1)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==2)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==3)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==4)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==5)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==6)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][1]==7)
					min[0].s1=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][2]==0)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==1)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==2)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==3)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==4)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==5)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==6)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][2]==7)
					min[0].s2=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][3]==0)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==1)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==2)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==3)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==4)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==5)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==6)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][3]==7)
					min[0].s3=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][4]==0)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==1)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==2)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==3)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==4)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==5)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==6)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][4]==7)
					min[0].s4=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][5]==0)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==1)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==2)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==3)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==4)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==5)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==6)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][5]==7)
					min[0].s5=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][6]==0)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==1)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==2)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==3)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==4)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==5)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==6)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][6]==7)
					min[0].s6=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
	
				if(c_mult[c_val[c_row_ptr[x]+y]][7]==0)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s0;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==1)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s1;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==2)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s2;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==3)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s3;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==4)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s4;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==5)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s5;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==6)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s6;
				else if(c_mult[c_val[c_row_ptr[x]+y]][7]==7)
					min[0].s7=(s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0 : s_F[(threadIdx.x*c_w_row)+y-1].s7;
				/* if(x==1){
					printf("min[0]=%d, s_f=%d , s_b=%d\n", min[0].s0,s_F[(threadIdx.x*c_w_row)+y-1].s2, s_B[(threadIdx.x*c_w_row)+y+1].s2 );  
					printf("min[1]=%d\n", min[0].s1);  
					printf("min[2]=%d\n", min[0].s2);  
					printf("min[3]=%d\n", min[0].s3);  
					printf("min[4]=%d\n", min[0].s4);  
					printf("min[5]=%d\n", min[0].s5);  
					printf("min[6]=%d\n", min[0].s6);  
					printf("min[7]=%d\n", min[0].s7); 
				} */
				
				for(unsigned char b=1; b<Q; b++){
					if(c_mult[c_val[c_row_ptr[x]+y]][0]==0)
						a.s0=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][0]==1)
						a.s0=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][0]==2)
						a.s0=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][0]==3)
						a.s0=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][0]==4)
						a.s0=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][0]==5)
						a.s0=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][0]==6)
						a.s0=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][0]==7)
						a.s0=c_add[b][7];
		
		
					if(c_mult[c_val[c_row_ptr[x]+y]][1]==0)
						a.s1=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][1]==1)
						a.s1=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][1]==2)
						a.s1=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][1]==3)
						a.s1=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][1]==4)
						a.s1=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][1]==5)
						a.s1=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][1]==6)
						a.s1=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][1]==7)
						a.s1=c_add[b][7];
		
					if(c_mult[c_val[c_row_ptr[x]+y]][2]==0)
						a.s2=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][2]==1)
						a.s2=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][2]==2)
						a.s2=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][2]==3)
						a.s2=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][2]==4)
						a.s2=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][2]==5)
						a.s2=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][2]==6)
						a.s2=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][2]==7)
						a.s2=c_add[b][7];
		
					if(c_mult[c_val[c_row_ptr[x]+y]][3]==0)
						a.s3=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][3]==1)
						a.s3=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][3]==2)
						a.s3=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][3]==3)
						a.s3=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][3]==4)
						a.s3=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][3]==5)
						a.s3=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][3]==6)
						a.s3=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][3]==7)
						a.s3=c_add[b][7];
		
					if(c_mult[c_val[c_row_ptr[x]+y]][4]==0)
						a.s4=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][4]==1)
						a.s4=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][4]==2)
						a.s4=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][4]==3)
						a.s4=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][4]==4)
						a.s4=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][4]==5)
						a.s4=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][4]==6)
						a.s4=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][4]==7)
						a.s4=c_add[b][7];
		
					if(c_mult[c_val[c_row_ptr[x]+y]][5]==0)
						a.s5=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][5]==1)
						a.s5=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][5]==2)
						a.s5=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][5]==3)
						a.s5=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][5]==4)
						a.s5=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][5]==5)
						a.s5=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][5]==6)
						a.s5=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][5]==7)
						a.s5=c_add[b][7];
		
					if(c_mult[c_val[c_row_ptr[x]+y]][6]==0)
						a.s6=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][6]==1)
						a.s6=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][6]==2)
						a.s6=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][6]==3)
						a.s6=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][6]==4)
						a.s6=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][6]==5)
						a.s6=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][6]==6)
						a.s6=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][6]==7)
						a.s6=c_add[b][7];
		
					if(c_mult[c_val[c_row_ptr[x]+y]][7]==0)
						a.s7=c_add[b][0];
					else if(c_mult[c_val[c_row_ptr[x]+y]][7]==1)
						a.s7=c_add[b][1];
					else if(c_mult[c_val[c_row_ptr[x]+y]][7]==2)
						a.s7=c_add[b][2];
					else if(c_mult[c_val[c_row_ptr[x]+y]][7]==3)
						a.s7=c_add[b][3];
					else if(c_mult[c_val[c_row_ptr[x]+y]][7]==4)
						a.s7=c_add[b][4];
					else if(c_mult[c_val[c_row_ptr[x]+y]][7]==5)
						a.s7=c_add[b][5];
					else if(c_mult[c_val[c_row_ptr[x]+y]][7]==6)
						a.s7=c_add[b][6];
					else if(c_mult[c_val[c_row_ptr[x]+y]][7]==7)
						a.s7=c_add[b][7];
					/* if(x==3){
						printf("a[%d][0]=%d\n", b, a.x);  
						printf("a[%d][1]=%d\n", b, a.y);  
						printf("a[%d][2]=%d\n", b, a.z);  
						printf("a[%d][3]=%d\n", b, a.w);  
					} */

					

					// FORWARD
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S0
						if(a.s0==0){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s0==1){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s0==2){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s0==3){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s0==4){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s0==5){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s0==6){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s0==7){
							if(b==0)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s0 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S1
						if(a.s1==0){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s1==1){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s1==2){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s1==3){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s1==4){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s1==5){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s1==6){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s1==7){
							if(b==0)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s1 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S2
						if(a.s2==0){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s2==1){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s2==2){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s2==3){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s2==4){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s2==5){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s2==6){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s2==7){
							if(b==0)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s2= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s2 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S3
						if(a.s3==0){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s3==1){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s3==2){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s3==3){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s3==4){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s3==5){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s3==6){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s3==7){
							if(b==0)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s3= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s3 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S4
						if(a.s4==0){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s4==1){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s4==2){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s4==3){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s4==4){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s4==5){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s4==6){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s4==7){
							if(b==0)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s4= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S5
						if(a.s5==0){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s5==1){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s5==2){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s5==3){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s5==4){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s5==5){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s5==6){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s5==7){
							if(b==0)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s5= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s5 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S6
						if(a.s6==0){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s6==1){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s6==2){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s6==3){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s6==4){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s6==5){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s6==6){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s6==7){
							if(b==0)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s6= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77
					// S7
						if(a.s7==0){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else  if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s0;
							else if(b==7)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s0 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s0;
						}
						else if(a.s7==1){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s1;
							else if(b==7)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s1 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s1;
						}
						else if(a.s7==2){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s2;
							else if(b==7)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s2 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s2;
						}
						else if(a.s7==3){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s3;
							else if(b==7)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s3 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s3;
						}

						else if(a.s7==4){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else  if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s4;
							else if(b==7)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s4 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s4;
						}
						else if(a.s7==5){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s5;
							else if(b==7)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s5 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s5;
						}
						else if(a.s7==6){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s6;
							else if(b==7)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s6 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s6;
						}
						else if(a.s7==7){
							if(b==0)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s0) ? s_B[(threadIdx.x*c_w_row)+y+1].s0: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==1)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s1) ? s_B[(threadIdx.x*c_w_row)+y+1].s1: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==2)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s2) ? s_B[(threadIdx.x*c_w_row)+y+1].s2: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==3)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s3) ? s_B[(threadIdx.x*c_w_row)+y+1].s3: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==4)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s4) ? s_B[(threadIdx.x*c_w_row)+y+1].s4: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==5)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s5) ? s_B[(threadIdx.x*c_w_row)+y+1].s5: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==6)
								max.s7= (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s6) ? s_B[(threadIdx.x*c_w_row)+y+1].s6: s_F[(threadIdx.x*c_w_row)+y-1].s7;
							else if(b==7)
								max.s7 = (s_F[(threadIdx.x*c_w_row)+y-1].s7 < s_B[(threadIdx.x*c_w_row)+y+1].s7) ? s_B[(threadIdx.x*c_w_row)+y+1].s7: s_F[(threadIdx.x*c_w_row)+y-1].s7;
						}
					/* if(x==3){
						printf("max[%d][0]=%d\n", b, max.x);  
						printf("max[%d][1]=%d\n", b, max.y);  
						printf("max[%d][2]=%d\n", b, max.z);  
						printf("max[%d][3]=%d\n", b, max.w);  
					} */
				
					min[0].s0 = (min[0].s0 < max.s0) ? min[0].s0: max.s0;
					min[0].s1 = (min[0].s1 < max.s1) ? min[0].s1: max.s1;
					min[0].s2 = (min[0].s2 < max.s2) ? min[0].s2: max.s2;
					min[0].s3 = (min[0].s3 < max.s3) ? min[0].s3: max.s3;
					min[0].s4 = (min[0].s4 < max.s4) ? min[0].s4: max.s4;
					min[0].s5 = (min[0].s5 < max.s5) ? min[0].s5: max.s5;
					min[0].s6 = (min[0].s6 < max.s6) ? min[0].s6: max.s6;
					min[0].s7 = (min[0].s7 < max.s7) ? min[0].s7: max.s7;
				}
				reinterpret_cast<uchar8*>(d_beta_8)[(x*c_w_row)+y]=reinterpret_cast<uchar8*>(min)[0];

			}
		//}

	}

	__global__ void GPU_VN( uchar8 * d_beta_8,  uchar8 * d_alpha_8, uchar8 * d_gamma_8, int iter){
		unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;
		uchar8 temp;

		temp.s0=0;
		temp.s1=0;
		temp.s2=0;
		temp.s3=0;
		temp.s4=0;
		temp.s5=0;
		temp.s6=0;
		temp.s7=0;

		#if Q==8
		__shared__ uchar8 s_beta[2*8];
		__shared__ uchar8 s_alpha_t[2*8];
		__shared__ uchar8 s_alpha_t2[2*8];
		#elif Q==16
		__shared__ unsigned char s_beta[2][32][16];
		__shared__ unsigned char s_alpha_t[2][32][16];
		__shared__ unsigned char s_alpha_t2[2][32][16];
		#elif Q==32
		__shared__ unsigned char s_beta[2][16][32];
		__shared__ unsigned char s_alpha_t[2][16][32];
		__shared__ unsigned char s_alpha_t2[2][16][32];
		#endif

		//#if Q==4
			//if(y<c_N){
		//#endif

			reinterpret_cast<uchar8*>(s_beta)[(x*c_w_row)+threadIdx.y]=reinterpret_cast<uchar8*>(d_beta_8)[(c_ptr_to_val[c_col_ptr[y]+x])];
			__syncthreads();
			/* if(threadIdx.x==1 && y==161 && iter==4){
				printf("s_beta[%d][%d][0]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s0);  
				printf("s_beta[%d][%d][1]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s1);  
				printf("s_beta[%d][%d][2]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s2);  
				printf("s_beta[%d][%d][3]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s3);
				printf("s_beta[%d][%d][4]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s4);  
				printf("s_beta[%d][%d][5]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s5);  
				printf("s_beta[%d][%d][6]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s6);  
				printf("s_beta[%d][%d][7]=%d\n", x, y, s_beta[(x*c_w_row)+threadIdx.y].s7);    
			} */


			for(unsigned char index=0; index<c_w_col; index++){
				/* if(y==297){
					printf("index[0]=%d\n",index);  
					printf("index[1]=%d\n",index);  
					printf("index[2]=%d\n",index);  
					printf("index[3]=%d\n",index);  
				}  */
				if(index!=x){
					/* if(threadIdx.x==0 && y==9){
						printf("index[0]=%d, temp=%d\n",index, temp.s0);  
						printf("index[1]=%d, temp=%d\n",index, temp.s1);  
						printf("index[2]=%d, temp=%d\n",index, temp.s2);  
						printf("index[3]=%d, temp=%d\n",index, temp.s3);
						printf("index[4]=%d, temp=%d\n",index, temp.s4);  
						printf("index[5]=%d, temp=%d\n",index, temp.s5);  
						printf("index[6]=%d, temp=%d\n",index, temp.s6);  
						printf("index[7]=%d, temp=%d\n",index, temp.s7);    
					} */
					temp.s0=temp.s0+s_beta[(index*c_w_row)+threadIdx.y].s0;
					temp.s1=temp.s1+s_beta[(index*c_w_row)+threadIdx.y].s1;
					temp.s2=temp.s2+s_beta[(index*c_w_row)+threadIdx.y].s2;
					temp.s3=temp.s3+s_beta[(index*c_w_row)+threadIdx.y].s3;
					temp.s4=temp.s4+s_beta[(index*c_w_row)+threadIdx.y].s4;
					temp.s5=temp.s5+s_beta[(index*c_w_row)+threadIdx.y].s5;
					temp.s6=temp.s6+s_beta[(index*c_w_row)+threadIdx.y].s6;
					temp.s7=temp.s7+s_beta[(index*c_w_row)+threadIdx.y].s7;
					/* if(threadIdx.x==1 && y==161 && iter==4){
						printf("temp=%d s_beta[%d][%d][0]=%d\n", temp.s0, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s0);  
						printf("temp=%d s_beta[%d][%d][1]=%d\n", temp.s1, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s1);  
						printf("temp=%d s_beta[%d][%d][2]=%d\n", temp.s2, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s2);  
						printf("temp=%d s_beta[%d][%d][3]=%d\n", temp.s3, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s3);
						printf("temp=%d s_beta[%d][%d][4]=%d\n", temp.s4, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s4);  
						printf("temp=%d s_beta[%d][%d][5]=%d\n", temp.s5, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s5);  
						printf("temp=%d s_beta[%d][%d][6]=%d\n", temp.s6, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s6);  
						printf("temp=%d s_beta[%d][%d][7]=%d\n", temp.s7, index, threadIdx.y, s_beta[(index*c_w_row)+threadIdx.y].s7);  
					} */
				}
				__syncthreads();
			}
			/* if(threadIdx.x==1 && y==161 && iter==4){
				printf("temp[%d][%d][0]=%d\n", x, y, temp.s0);  
				printf("temp[%d][%d][1]=%d\n", x, y, temp.s1);  
				printf("temp[%d][%d][2]=%d\n", x, y, temp.s2);  
				printf("temp[%d][%d][3]=%d\n", x, y, temp.s3);  
				printf("temp[%d][%d][4]=%d\n", x, y, temp.s4);  
				printf("temp[%d][%d][5]=%d\n", x, y, temp.s5);  
				printf("temp[%d][%d][6]=%d\n", x, y, temp.s6);  
				printf("temp[%d][%d][7]=%d\n", x, y, temp.s7);  
			}  */

			s_alpha_t[(x*c_w_row)+threadIdx.y].s0= temp.s0+ d_gamma_8[y].s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s1= temp.s1+ d_gamma_8[y].s1;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s2= temp.s2+ d_gamma_8[y].s2;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s3= temp.s3+ d_gamma_8[y].s3;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s4= temp.s4+ d_gamma_8[y].s4;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s5= temp.s5+ d_gamma_8[y].s5;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s6= temp.s6+ d_gamma_8[y].s6;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s7= temp.s7+ d_gamma_8[y].s7;
			/* if(threadIdx.x==1 && y==161 && iter==4){
				printf("s_alpha_t[%d][%d][0]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s0);  
				printf("s_alpha_t[%d][%d][1]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s1);  
				printf("s_alpha_t[%d][%d][2]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s2);  
				printf("s_alpha_t[%d][%d][3]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s3); 
				printf("s_alpha_t[%d][%d][4]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s4);  
				printf("s_alpha_t[%d][%d][5]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s5);  
				printf("s_alpha_t[%d][%d][6]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s6);  
				printf("s_alpha_t[%d][%d][7]=%d\n", x, y, s_alpha_t[(x*c_w_row)+threadIdx.y].s7); 
			} */

			s_alpha_t2[(x*c_w_row)+threadIdx.y].s0=0;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s1=1;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s2=2;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s3=3;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s4=4;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s5=5;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s6=6;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s7=7;
			__syncthreads();


			s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s0 > s_alpha_t[(x*c_w_row)+threadIdx.y].s4) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s4 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s0;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s1= (s_alpha_t[(x*c_w_row)+threadIdx.y].s1 > s_alpha_t[(x*c_w_row)+threadIdx.y].s5) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s5 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s1;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s2= (s_alpha_t[(x*c_w_row)+threadIdx.y].s2 > s_alpha_t[(x*c_w_row)+threadIdx.y].s6) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s6 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s2;
			s_alpha_t2[(x*c_w_row)+threadIdx.y].s3= (s_alpha_t[(x*c_w_row)+threadIdx.y].s3 > s_alpha_t[(x*c_w_row)+threadIdx.y].s7) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s7 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s3;
			/* if(threadIdx.x==1 && y==161 && iter==4){
				printf("s_alpha_t2[%d][%d][0]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s0);
				printf("s_alpha_t2[%d][%d][1]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s1);
				printf("s_alpha_t2[%d][%d][2]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s2);
				printf("s_alpha_t2[%d][%d][3]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s3); 
				printf("s_alpha_t2[%d][%d][4]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s4);
				printf("s_alpha_t2[%d][%d][5]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s5);
				printf("s_alpha_t2[%d][%d][6]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s6);
				printf("s_alpha_t2[%d][%d][7]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s7);  
			} */

			if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==0){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s2==2)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s0 > s_alpha_t[(x*c_w_row)+threadIdx.y].s2) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s2 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s0;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s2==6)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s0 > s_alpha_t[(x*c_w_row)+threadIdx.y].s6) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s6 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s0;
			}
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==4){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s2==2)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s4 > s_alpha_t[(x*c_w_row)+threadIdx.y].s2) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s2 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s4;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s2==6)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s4 > s_alpha_t[(x*c_w_row)+threadIdx.y].s6) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s6 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s4;
			}

			if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==1){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s3==3)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s1= (s_alpha_t[(x*c_w_row)+threadIdx.y].s1 > s_alpha_t[(x*c_w_row)+threadIdx.y].s3) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s3 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s1;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s3==7)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s1= (s_alpha_t[(x*c_w_row)+threadIdx.y].s1 > s_alpha_t[(x*c_w_row)+threadIdx.y].s7) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s7 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s1;
			}
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==5){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s3==3)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s1= (s_alpha_t[(x*c_w_row)+threadIdx.y].s5 > s_alpha_t[(x*c_w_row)+threadIdx.y].s3) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s3 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s5;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s3==7)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s1= (s_alpha_t[(x*c_w_row)+threadIdx.y].s5 > s_alpha_t[(x*c_w_row)+threadIdx.y].s7) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s7 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s5;
			}
			/* if(threadIdx.x==1 && y==161 && iter==4){
				printf("s_alpha_t2[%d][%d][0]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s0);
				printf("s_alpha_t2[%d][%d][1]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s1);
				printf("s_alpha_t2[%d][%d][2]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s2);
				printf("s_alpha_t2[%d][%d][3]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s3); 
				printf("s_alpha_t2[%d][%d][4]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s4);
				printf("s_alpha_t2[%d][%d][5]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s5);
				printf("s_alpha_t2[%d][%d][6]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s6);
				printf("s_alpha_t2[%d][%d][7]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s7);  
			}  */
			


			if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==0){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==1)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s0 > s_alpha_t[(x*c_w_row)+threadIdx.y].s1) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s1 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s0;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==3)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s0 > s_alpha_t[(x*c_w_row)+threadIdx.y].s3) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s3 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s0;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==5)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s0 > s_alpha_t[(x*c_w_row)+threadIdx.y].s5) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s5 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s0;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==7)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s0 > s_alpha_t[(x*c_w_row)+threadIdx.y].s7) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s7 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s0;
			}
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==2){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==1)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s2 > s_alpha_t[(x*c_w_row)+threadIdx.y].s1) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s1 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s2;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==3)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s2 > s_alpha_t[(x*c_w_row)+threadIdx.y].s3) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s3 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s2;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==5)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s2 > s_alpha_t[(x*c_w_row)+threadIdx.y].s5) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s5 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s2;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==7)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s2 > s_alpha_t[(x*c_w_row)+threadIdx.y].s7) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s7 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s2;
			}
			if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==4){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==1)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s4 > s_alpha_t[(x*c_w_row)+threadIdx.y].s1) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s1 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s4;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==3)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s4 > s_alpha_t[(x*c_w_row)+threadIdx.y].s3) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s3 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s4;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==5)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s4 > s_alpha_t[(x*c_w_row)+threadIdx.y].s5) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s5 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s4;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==7)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s4 > s_alpha_t[(x*c_w_row)+threadIdx.y].s7) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s7 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s4;
			}
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==6){
				if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==1)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s6 > s_alpha_t[(x*c_w_row)+threadIdx.y].s1) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s1 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s6;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==3)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s6 > s_alpha_t[(x*c_w_row)+threadIdx.y].s3) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s3 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s6;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==5)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s6 > s_alpha_t[(x*c_w_row)+threadIdx.y].s5) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s5 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s6;
				else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s1==7)
					s_alpha_t2[(x*c_w_row)+threadIdx.y].s0= (s_alpha_t[(x*c_w_row)+threadIdx.y].s6 > s_alpha_t[(x*c_w_row)+threadIdx.y].s7) ? s_alpha_t2[(x*c_w_row)+threadIdx.y].s7 : s_alpha_t2[(x*c_w_row)+threadIdx.y].s6;
			}
			__syncthreads();
			/* if(threadIdx.x==1 && y==161 && iter==4){
				printf("s_alpha_t2[%d][%d][0]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s0);
				printf("s_alpha_t2[%d][%d][1]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s1);
				printf("s_alpha_t2[%d][%d][2]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s2);
				printf("s_alpha_t2[%d][%d][3]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s3); 
				printf("s_alpha_t2[%d][%d][4]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s4);
				printf("s_alpha_t2[%d][%d][5]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s5);
				printf("s_alpha_t2[%d][%d][6]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s6);
				printf("s_alpha_t2[%d][%d][7]=%d\n", x, y, s_alpha_t2[(x*c_w_row)+threadIdx.y].s7);  
			} */
			
			if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==0)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s0;
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==1)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s1;
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==2)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s2;
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==3)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s3;
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==4)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s4;
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==5)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s5;
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==6)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s6;
			else if(s_alpha_t2[(x*c_w_row)+threadIdx.y].s0==7)
				temp.s0=s_alpha_t[(x*c_w_row)+threadIdx.y].s7;

			s_alpha_t[(x*c_w_row)+threadIdx.y].s0 =s_alpha_t[(x*c_w_row)+threadIdx.y].s0 - temp.s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s1 =s_alpha_t[(x*c_w_row)+threadIdx.y].s1 - temp.s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s2 =s_alpha_t[(x*c_w_row)+threadIdx.y].s2 - temp.s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s3 =s_alpha_t[(x*c_w_row)+threadIdx.y].s3 - temp.s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s4 =s_alpha_t[(x*c_w_row)+threadIdx.y].s4 - temp.s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s5 =s_alpha_t[(x*c_w_row)+threadIdx.y].s5 - temp.s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s6 =s_alpha_t[(x*c_w_row)+threadIdx.y].s6 - temp.s0;
			s_alpha_t[(x*c_w_row)+threadIdx.y].s7 =s_alpha_t[(x*c_w_row)+threadIdx.y].s7 - temp.s0;


			reinterpret_cast<uchar8*>(d_alpha_8)[(c_ptr_to_val[c_col_ptr[y]+x])]=reinterpret_cast<uchar8*>(s_alpha_t)[(x*c_w_row)+threadIdx.y];
			/* if(y==161 && iter==4){
				printf("d_alpha_8[%d][0]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s0);
				printf("d_alpha_8[%d][1]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s1);
				printf("d_alpha_8[%d][2]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s2);
				printf("d_alpha_8[%d][3]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s3);  
				printf("d_alpha_8[%d][4]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s4);
				printf("d_alpha_8[%d][5]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s5);
				printf("d_alpha_8[%d][6]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s6);
				printf("d_alpha_8[%d][7]=%d\n", (c_ptr_to_val[c_col_ptr[y]+x]), d_alpha_8[(c_ptr_to_val[c_col_ptr[y]+x])].s7); 
			} */
	}
//#endif



//===================================
// CUDA Wrapper
//===================================

extern "C" int cuda_minmax(unsigned char* h_beta, unsigned char* ALPHAmn_, unsigned char* GAMMAn_, unsigned char* h_F,unsigned char* h_B,int* iteration){
	
	#if Q==4
		const unsigned char add[4][4] =	{	{ 0,1,2,3 },
										{ 1,0,3,2 },
										{ 2,3,0,1 },
										{ 3,2,1,0 }};
		const unsigned char mult[4][4] = {{ 0,0,0,0}, { 0,1,2,3 },{ 0,2,3,1 },{ 0,3,1,2 }};
		const unsigned char inv[4] = { 0,1,3,2};
	#elif Q==8
		const unsigned char add[8][8] = {{ 0,1,2,3,4,5,6,7 }, { 1,0,3,2,5,4,7,6 },{ 2,3,0,1,6,7,4,5 },{ 3,2,1,0,7,6,5,4 },{ 4,5,6,7,0,1,2,3 }, { 5,4,7,6,1,0,3,2 },{ 6,7,4,5,2,3,0,1 },{ 7,6,5,4,3,2,1,0 }};
		const unsigned char mult[8][8] = {{ 0,0,0,0,0,0,0,0}, { 0,1,2,3,4,5,6,7 },{ 0,2,4,6,3,1,7,5 },{ 0,3,6,5,7,4,1,2 },{ 0,4,3,7,6,2,5,1}, { 0,5,1,4,2,7,3,6 },{ 0,6,7,1,5,3,2,4 },{ 0,7,5,2,1,6,4,3 }};
		const unsigned char inv[8] = { 0,1,5,6,7,2,3,4};
	#elif Q==16
		const unsigned char add[16][16] = {	{ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
											{ 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14 },
											{ 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13 },
											{ 3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12 },
											{ 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11 },
											{ 5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10 },
											{ 6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9 },
											{ 7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8 },
											{ 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 },
											{ 9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6 },
											{ 10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5 },
											{ 11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4 },
											{ 12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3 },
											{ 13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2 },
											{ 14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1 },
											{ 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 }};

		const unsigned char mult[16][16] = {{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
											{ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 },
											{ 0,2,4,6,8,10,12,14,3,1,7,5,11,9,15,13 },
											{ 0,3,6,5,12,15,10,9,11,8,13,14,7,4,1,2 },
											{ 0,4,8,12,3,7,11,15,6,2,14,10,5,1,13,9},
											{ 0,5,10,15,7,2,13,8,14,11,4,1,9,12,3,6},
											{ 0,6,12,10,11,13,7,1,5,3,9,15,14,8,2,4},
											{ 0,7,14,9,15,8,1,6,13,10,3,4,2,5,12,11},
											{ 0,8,3,11,6,14,5,13,12,4,15,7,10,2,9,1},
											{ 0,9,1,8,2,11,3,10,4,13,5,12,6,15,7,14},
											{ 0,10,7,13,14,4,9,3,15,5,8,2,1,11,6,12},
											{ 0,11,5,14,10,1,15,4,7,12,2,9,13,6,8,3},
											{ 0,12,11,7,5,9,14,2,10,6,1,13,15,3,4,8},
											{ 0,13,9,4,1,12,8,5,2,15,11,6,3,14,10,7},
											{ 0,14,15,1,13,3,2,12,9,7,6,8,4,10,11,5},
											{ 0,15,13,2,9,6,4,11,1,14,12,3,8,7,5,10}};

		const unsigned char inv[16] = { 0,1,9,14,13,11,7,6,15,2,12,5,10,4,3,8};
	#elif Q==32
		const unsigned char add[32][32] = {	{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31},
											{1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30},
											{2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29},
											{3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16,23,22,21,20,27,26,25,24,31,30,29,28},
											{4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,20,21,22,23,16,17,18,19,28,29,30,31,24,25,26,27},
											{5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10,21,20,23,22,17,16,19,18,29,28,31,30,25,24,27,26},
											{6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9,22,23,20,21,18,19,16,17,30,31,28,29,26,27,24,25},
											{7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24},
											{8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23},
											{9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6,25,24,27,26,29,28,31,30,17,16,19,18,21,20,23,22},
											{10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5,26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21},
											{11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,27,26,25,24,31,30,29,28,19,18,17,16,23,22,21,20},
											{12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19},
											{13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,29,28,31,30,25,24,27,26,21,20,23,22,17,16,19,18},
											{14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,30,31,28,29,26,27,24,25,22,23,20,21,18,19,16,17},
											{15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16},
											{16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
											{17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30,1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
											{18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29,2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
											{19,18,17,16,23,22,21,20,27,26,25,24,31,30,29,28,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12},
											{20,21,22,23,16,17,18,19,28,29,30,31,24,25,26,27,4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11},
											{21,20,23,22,17,16,19,18,29,28,31,30,25,24,27,26,5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10},
											{22,23,20,21,18,19,16,17,30,31,28,29,26,27,24,25,6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9},
											{23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24,7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8},
											{24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7},
											{25,24,27,26,29,28,31,30,17,16,19,18,21,20,23,22,9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6},
											{26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21,10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5},
											{27,26,25,24,31,30,29,28,19,18,17,16,23,22,21,20,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4},
											{28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19,12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3},
											{29,28,31,30,25,24,27,26,21,20,23,22,17,16,19,18,13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2},
											{30,31,28,29,26,27,24,25,22,23,20,21,18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1},
											{31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}};


		const unsigned char mult[32][32] = {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
											{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31},
											{0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,5,7,1,3,13,15,9,11,21,23,17,19,29,31,25,27},
											{0,3,6,5,12,15,10,9,24,27,30,29,20,23,18,17,21,22,19,16,25,26,31,28,13,14,11,8,1,2,7,4},
											{0,4,8,12,16,20,24,28,5,1,13,9,21,17,29,25,10,14,2,6,26,30,18,22,15,11,7,3,31,27,23,19},
											{0,5,10,15,20,17,30,27,13,8,7,2,25,28,19,22,26,31,16,21,14,11,4,1,23,18,29,24,3,6,9,12},
											{0,6,12,10,24,30,20,18,21,19,25,31,13,11,1,7,15,9,3,5,23,17,27,29,26,28,22,16,2,4,14,8},
											{0,7,14,9,28,27,18,21,29,26,19,20,1,6,15,8,31,24,17,22,3,4,13,10,2,5,12,11,30,25,16,23},
											{0,8,16,24,5,13,21,29,10,2,26,18,15,7,31,23,20,28,4,12,17,25,1,9,30,22,14,6,27,19,11,3},
											{0,9,18,27,1,8,19,26,2,11,16,25,3,10,17,24,4,13,22,31,5,12,23,30,6,15,20,29,7,14,21,28},
											{0,10,20,30,13,7,25,19,26,16,14,4,23,29,3,9,17,27,5,15,28,22,8,2,11,1,31,21,6,12,18,24},
											{0,11,22,29,9,2,31,20,18,25,4,15,27,16,13,6,1,10,23,28,8,3,30,21,19,24,5,14,26,17,12,7},
											{0,12,24,20,21,25,13,1,15,3,23,27,26,22,2,14,30,18,6,10,11,7,19,31,17,29,9,5,4,8,28,16},
											{0,13,26,23,17,28,11,6,7,10,29,16,22,27,12,1,14,3,20,25,31,18,5,8,9,4,19,30,24,21,2,15},
											{0,14,28,18,29,19,1,15,31,17,3,13,2,12,30,16,27,21,7,9,6,8,26,20,4,10,24,22,25,23,5,11},
											{0,15,30,17,25,22,7,8,23,24,9,6,14,1,16,31,11,4,21,26,18,29,12,3,28,19,2,13,5,10,27,20},
											{0,16,5,21,10,26,15,31,20,4,17,1,30,14,27,11,13,29,8,24,7,23,2,18,25,9,28,12,19,3,22,6},
											{0,17,7,22,14,31,9,24,28,13,27,10,18,3,21,4,29,12,26,11,19,2,20,5,1,16,6,23,15,30,8,25},
											{0,18,1,19,2,16,3,17,4,22,5,23,6,20,7,21,8,26,9,27,10,24,11,25,12,30,13,31,14,28,15,29},
											{0,19,3,16,6,21,5,22,12,31,15,28,10,25,9,26,24,11,27,8,30,13,29,14,20,7,23,4,18,1,17,2},
											{0,20,13,25,26,14,23,3,17,5,28,8,11,31,6,18,7,19,10,30,29,9,16,4,22,2,27,15,12,24,1,21},
											{0,21,15,26,30,11,17,4,25,12,22,3,7,18,8,29,23,2,24,13,9,28,6,19,14,27,1,20,16,5,31,10},
											{0,22,9,31,18,4,27,13,1,23,8,30,19,5,26,12,2,20,11,29,16,6,25,15,3,21,10,28,17,7,24,14},
											{0,23,11,28,22,1,29,10,9,30,2,21,31,8,20,3,18,5,25,14,4,19,15,24,27,12,16,7,13,26,6,17},
											{0,24,21,13,15,23,26,2,30,6,11,19,17,9,4,28,25,1,12,20,22,14,3,27,7,31,18,10,8,16,29,5},
											{0,25,23,14,11,18,28,5,22,15,1,24,29,4,10,19,9,16,30,7,2,27,21,12,31,6,8,17,20,13,3,26},
											{0,26,17,11,7,29,22,12,14,20,31,5,9,19,24,2,28,6,13,23,27,1,10,16,18,8,3,25,21,15,4,30},
											{0,27,19,8,3,24,16,11,6,29,21,14,5,30,22,13,12,23,31,4,15,20,28,7,10,17,25,2,9,18,26,1},
											{0,28,29,1,31,3,2,30,27,7,6,26,4,24,25,5,19,15,14,18,12,16,17,13,8,20,21,9,23,11,10,22},
											{0,29,31,2,27,6,4,25,19,14,12,17,8,21,23,10,3,30,28,1,24,5,7,26,16,13,15,18,11,22,20,9},
											{0,30,25,7,23,9,14,16,11,21,18,12,28,2,5,27,22,8,15,17,1,31,24,6,29,3,4,26,10,20,19,13},
											{0,31,27,4,19,12,8,23,3,28,24,7,16,15,11,20,6,25,29,2,21,10,14,17,5,26,30,1,22,9,13,18}};
	
		const unsigned char inv[32] = { 0,1,18,28,9,23,14,12,22,4,25,16,7,15,6,13,11,24,2,29,30,26,8,5,17,10,21,31,3,19,20,27};
		
	#elif Q==64
		const unsigned char add[64][64] = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63},
		{1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30,33,32,35,34,37,36,39,38,41,40,43,42,45,44,47,46,49,48,51,50,53,52,55,54,57,56,59,58,61,60,63,62},
		{2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29,34,35,32,33,38,39,36,37,42,43,40,41,46,47,44,45,50,51,48,49,54,55,52,53,58,59,56,57,62,63,60,61},
		{3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16,23,22,21,20,27,26,25,24,31,30,29,28,35,34,33,32,39,38,37,36,43,42,41,40,47,46,45,44,51,50,49,48,55,54,53,52,59,58,57,56,63,62,61,60},
		{4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,20,21,22,23,16,17,18,19,28,29,30,31,24,25,26,27,36,37,38,39,32,33,34,35,44,45,46,47,40,41,42,43,52,53,54,55,48,49,50,51,60,61,62,63,56,57,58,59},
		{5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10,21,20,23,22,17,16,19,18,29,28,31,30,25,24,27,26,37,36,39,38,33,32,35,34,45,44,47,46,41,40,43,42,53,52,55,54,49,48,51,50,61,60,63,62,57,56,59,58},
		{6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9,22,23,20,21,18,19,16,17,30,31,28,29,26,27,24,25,38,39,36,37,34,35,32,33,46,47,44,45,42,43,40,41,54,55,52,53,50,51,48,49,62,63,60,61,58,59,56,57},
		{7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24,39,38,37,36,35,34,33,32,47,46,45,44,43,42,41,40,55,54,53,52,51,50,49,48,63,62,61,60,59,58,57,56},
		{8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,40,41,42,43,44,45,46,47,32,33,34,35,36,37,38,39,56,57,58,59,60,61,62,63,48,49,50,51,52,53,54,55},
		{9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6,25,24,27,26,29,28,31,30,17,16,19,18,21,20,23,22,41,40,43,42,45,44,47,46,33,32,35,34,37,36,39,38,57,56,59,58,61,60,63,62,49,48,51,50,53,52,55,54},
		{10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5,26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21,42,43,40,41,46,47,44,45,34,35,32,33,38,39,36,37,58,59,56,57,62,63,60,61,50,51,48,49,54,55,52,53},
		{11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,27,26,25,24,31,30,29,28,19,18,17,16,23,22,21,20,43,42,41,40,47,46,45,44,35,34,33,32,39,38,37,36,59,58,57,56,63,62,61,60,51,50,49,48,55,54,53,52},
		{12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19,44,45,46,47,40,41,42,43,36,37,38,39,32,33,34,35,60,61,62,63,56,57,58,59,52,53,54,55,48,49,50,51},
		{13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,29,28,31,30,25,24,27,26,21,20,23,22,17,16,19,18,45,44,47,46,41,40,43,42,37,36,39,38,33,32,35,34,61,60,63,62,57,56,59,58,53,52,55,54,49,48,51,50},
		{14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,30,31,28,29,26,27,24,25,22,23,20,21,18,19,16,17,46,47,44,45,42,43,40,41,38,39,36,37,34,35,32,33,62,63,60,61,58,59,56,57,54,55,52,53,50,51,48,49},
		{15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48},
		{16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47},
		{17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30,1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,49,48,51,50,53,52,55,54,57,56,59,58,61,60,63,62,33,32,35,34,37,36,39,38,41,40,43,42,45,44,47,46},
		{18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29,2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,50,51,48,49,54,55,52,53,58,59,56,57,62,63,60,61,34,35,32,33,38,39,36,37,42,43,40,41,46,47,44,45},
		{19,18,17,16,23,22,21,20,27,26,25,24,31,30,29,28,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,51,50,49,48,55,54,53,52,59,58,57,56,63,62,61,60,35,34,33,32,39,38,37,36,43,42,41,40,47,46,45,44},
		{20,21,22,23,16,17,18,19,28,29,30,31,24,25,26,27,4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,52,53,54,55,48,49,50,51,60,61,62,63,56,57,58,59,36,37,38,39,32,33,34,35,44,45,46,47,40,41,42,43},
		{21,20,23,22,17,16,19,18,29,28,31,30,25,24,27,26,5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10,53,52,55,54,49,48,51,50,61,60,63,62,57,56,59,58,37,36,39,38,33,32,35,34,45,44,47,46,41,40,43,42},
		{22,23,20,21,18,19,16,17,30,31,28,29,26,27,24,25,6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9,54,55,52,53,50,51,48,49,62,63,60,61,58,59,56,57,38,39,36,37,34,35,32,33,46,47,44,45,42,43,40,41},
		{23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24,7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,55,54,53,52,51,50,49,48,63,62,61,60,59,58,57,56,39,38,37,36,35,34,33,32,47,46,45,44,43,42,41,40},
		{24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,56,57,58,59,60,61,62,63,48,49,50,51,52,53,54,55,40,41,42,43,44,45,46,47,32,33,34,35,36,37,38,39},
		{25,24,27,26,29,28,31,30,17,16,19,18,21,20,23,22,9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6,57,56,59,58,61,60,63,62,49,48,51,50,53,52,55,54,41,40,43,42,45,44,47,46,33,32,35,34,37,36,39,38},
		{26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21,10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5,58,59,56,57,62,63,60,61,50,51,48,49,54,55,52,53,42,43,40,41,46,47,44,45,34,35,32,33,38,39,36,37},
		{27,26,25,24,31,30,29,28,19,18,17,16,23,22,21,20,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,59,58,57,56,63,62,61,60,51,50,49,48,55,54,53,52,43,42,41,40,47,46,45,44,35,34,33,32,39,38,37,36},
		{28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19,12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,60,61,62,63,56,57,58,59,52,53,54,55,48,49,50,51,44,45,46,47,40,41,42,43,36,37,38,39,32,33,34,35},
		{29,28,31,30,25,24,27,26,21,20,23,22,17,16,19,18,13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,61,60,63,62,57,56,59,58,53,52,55,54,49,48,51,50,45,44,47,46,41,40,43,42,37,36,39,38,33,32,35,34},
		{30,31,28,29,26,27,24,25,22,23,20,21,18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,62,63,60,61,58,59,56,57,54,55,52,53,50,51,48,49,46,47,44,45,42,43,40,41,38,39,36,37,34,35,32,33},
		{31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32},
		{32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31},
		{33,32,35,34,37,36,39,38,41,40,43,42,45,44,47,46,49,48,51,50,53,52,55,54,57,56,59,58,61,60,63,62,1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30},
		{34,35,32,33,38,39,36,37,42,43,40,41,46,47,44,45,50,51,48,49,54,55,52,53,58,59,56,57,62,63,60,61,2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29},
		{35,34,33,32,39,38,37,36,43,42,41,40,47,46,45,44,51,50,49,48,55,54,53,52,59,58,57,56,63,62,61,60,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16,23,22,21,20,27,26,25,24,31,30,29,28},
		{36,37,38,39,32,33,34,35,44,45,46,47,40,41,42,43,52,53,54,55,48,49,50,51,60,61,62,63,56,57,58,59,4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,20,21,22,23,16,17,18,19,28,29,30,31,24,25,26,27},
		{37,36,39,38,33,32,35,34,45,44,47,46,41,40,43,42,53,52,55,54,49,48,51,50,61,60,63,62,57,56,59,58,5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10,21,20,23,22,17,16,19,18,29,28,31,30,25,24,27,26},
		{38,39,36,37,34,35,32,33,46,47,44,45,42,43,40,41,54,55,52,53,50,51,48,49,62,63,60,61,58,59,56,57,6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9,22,23,20,21,18,19,16,17,30,31,28,29,26,27,24,25},
		{39,38,37,36,35,34,33,32,47,46,45,44,43,42,41,40,55,54,53,52,51,50,49,48,63,62,61,60,59,58,57,56,7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24},
		{40,41,42,43,44,45,46,47,32,33,34,35,36,37,38,39,56,57,58,59,60,61,62,63,48,49,50,51,52,53,54,55,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23},
		{41,40,43,42,45,44,47,46,33,32,35,34,37,36,39,38,57,56,59,58,61,60,63,62,49,48,51,50,53,52,55,54,9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6,25,24,27,26,29,28,31,30,17,16,19,18,21,20,23,22},
		{42,43,40,41,46,47,44,45,34,35,32,33,38,39,36,37,58,59,56,57,62,63,60,61,50,51,48,49,54,55,52,53,10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5,26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21},
		{43,42,41,40,47,46,45,44,35,34,33,32,39,38,37,36,59,58,57,56,63,62,61,60,51,50,49,48,55,54,53,52,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,27,26,25,24,31,30,29,28,19,18,17,16,23,22,21,20},
		{44,45,46,47,40,41,42,43,36,37,38,39,32,33,34,35,60,61,62,63,56,57,58,59,52,53,54,55,48,49,50,51,12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19},
		{45,44,47,46,41,40,43,42,37,36,39,38,33,32,35,34,61,60,63,62,57,56,59,58,53,52,55,54,49,48,51,50,13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,29,28,31,30,25,24,27,26,21,20,23,22,17,16,19,18},
		{46,47,44,45,42,43,40,41,38,39,36,37,34,35,32,33,62,63,60,61,58,59,56,57,54,55,52,53,50,51,48,49,14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,30,31,28,29,26,27,24,25,22,23,20,21,18,19,16,17},
		{47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16},
		{48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
		{49,48,51,50,53,52,55,54,57,56,59,58,61,60,63,62,33,32,35,34,37,36,39,38,41,40,43,42,45,44,47,46,17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30,1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
		{50,51,48,49,54,55,52,53,58,59,56,57,62,63,60,61,34,35,32,33,38,39,36,37,42,43,40,41,46,47,44,45,18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29,2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
		{51,50,49,48,55,54,53,52,59,58,57,56,63,62,61,60,35,34,33,32,39,38,37,36,43,42,41,40,47,46,45,44,19,18,17,16,23,22,21,20,27,26,25,24,31,30,29,28,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12},
		{52,53,54,55,48,49,50,51,60,61,62,63,56,57,58,59,36,37,38,39,32,33,34,35,44,45,46,47,40,41,42,43,20,21,22,23,16,17,18,19,28,29,30,31,24,25,26,27,4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11},
		{53,52,55,54,49,48,51,50,61,60,63,62,57,56,59,58,37,36,39,38,33,32,35,34,45,44,47,46,41,40,43,42,21,20,23,22,17,16,19,18,29,28,31,30,25,24,27,26,5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10},
		{54,55,52,53,50,51,48,49,62,63,60,61,58,59,56,57,38,39,36,37,34,35,32,33,46,47,44,45,42,43,40,41,22,23,20,21,18,19,16,17,30,31,28,29,26,27,24,25,6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9},
		{55,54,53,52,51,50,49,48,63,62,61,60,59,58,57,56,39,38,37,36,35,34,33,32,47,46,45,44,43,42,41,40,23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24,7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8},
		{56,57,58,59,60,61,62,63,48,49,50,51,52,53,54,55,40,41,42,43,44,45,46,47,32,33,34,35,36,37,38,39,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7},
		{57,56,59,58,61,60,63,62,49,48,51,50,53,52,55,54,41,40,43,42,45,44,47,46,33,32,35,34,37,36,39,38,25,24,27,26,29,28,31,30,17,16,19,18,21,20,23,22,9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6},
		{58,59,56,57,62,63,60,61,50,51,48,49,54,55,52,53,42,43,40,41,46,47,44,45,34,35,32,33,38,39,36,37,26,27,24,25,30,31,28,29,18,19,16,17,22,23,20,21,10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5},
		{59,58,57,56,63,62,61,60,51,50,49,48,55,54,53,52,43,42,41,40,47,46,45,44,35,34,33,32,39,38,37,36,27,26,25,24,31,30,29,28,19,18,17,16,23,22,21,20,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4},
		{60,61,62,63,56,57,58,59,52,53,54,55,48,49,50,51,44,45,46,47,40,41,42,43,36,37,38,39,32,33,34,35,28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19,12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3},
		{61,60,63,62,57,56,59,58,53,52,55,54,49,48,51,50,45,44,47,46,41,40,43,42,37,36,39,38,33,32,35,34,29,28,31,30,25,24,27,26,21,20,23,22,17,16,19,18,13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2},
		{62,63,60,61,58,59,56,57,54,55,52,53,50,51,48,49,46,47,44,45,42,43,40,41,38,39,36,37,34,35,32,33,30,31,28,29,26,27,24,25,22,23,20,21,18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1},
		{63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}
		};


		const unsigned char mult[64][64] = {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63},
		{0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,3,1,7,5,11,9,15,13,19,17,23,21,27,25,31,29,35,33,39,37,43,41,47,45,51,49,55,53,59,57,63,61},
		{0,3,6,5,12,15,10,9,24,27,30,29,20,23,18,17,48,51,54,53,60,63,58,57,40,43,46,45,36,39,34,33,35,32,37,38,47,44,41,42,59,56,61,62,55,52,49,50,19,16,21,22,31,28,25,26,11,8,13,14,7,4,1,2},
		{0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,6,2,14,10,22,18,30,26,38,34,46,42,54,50,62,58,5,1,13,9,21,17,29,25,37,33,45,41,53,49,61,57},
		{0,5,10,15,20,17,30,27,40,45,34,39,60,57,54,51,19,22,25,28,7,2,13,8,59,62,49,52,47,42,37,32,38,35,44,41,50,55,56,61,14,11,4,1,26,31,16,21,53,48,63,58,33,36,43,46,29,24,23,18,9,12,3,6},
		{0,6,12,10,24,30,20,18,48,54,60,58,40,46,36,34,35,37,47,41,59,61,55,49,19,21,31,25,11,13,7,1,5,3,9,15,29,27,17,23,53,51,57,63,45,43,33,39,38,32,42,44,62,56,50,52,22,16,26,28,14,8,2,4},
		{0,7,14,9,28,27,18,21,56,63,54,49,36,35,42,45,51,52,61,58,47,40,33,38,11,12,5,2,23,16,25,30,37,34,43,44,57,62,55,48,29,26,19,20,1,6,15,8,22,17,24,31,10,13,4,3,46,41,32,39,50,53,60,59},
		{0,8,16,24,32,40,48,56,3,11,19,27,35,43,51,59,6,14,22,30,38,46,54,62,5,13,21,29,37,45,53,61,12,4,28,20,44,36,60,52,15,7,31,23,47,39,63,55,10,2,26,18,42,34,58,50,9,1,25,17,41,33,57,49},
		{0,9,18,27,36,45,54,63,11,2,25,16,47,38,61,52,22,31,4,13,50,59,32,41,29,20,15,6,57,48,43,34,44,37,62,55,8,1,26,19,39,46,53,60,3,10,17,24,58,51,40,33,30,23,12,5,49,56,35,42,21,28,7,14},
		{0,10,20,30,40,34,60,54,19,25,7,13,59,49,47,37,38,44,50,56,14,4,26,16,53,63,33,43,29,23,9,3,15,5,27,17,39,45,51,57,28,22,8,2,52,62,32,42,41,35,61,55,1,11,21,31,58,48,46,36,18,24,6,12},
		{0,11,22,29,44,39,58,49,27,16,13,6,55,60,33,42,54,61,32,43,26,17,12,7,45,38,59,48,1,10,23,28,47,36,57,50,3,8,21,30,52,63,34,41,24,19,14,5,25,18,15,4,53,62,35,40,2,9,20,31,46,37,56,51},
		{0,12,24,20,48,60,40,36,35,47,59,55,19,31,11,7,5,9,29,17,53,57,45,33,38,42,62,50,22,26,14,2,10,6,18,30,58,54,34,46,41,37,49,61,25,21,1,13,15,3,23,27,63,51,39,43,44,32,52,56,28,16,4,8},
		{0,13,26,23,52,57,46,35,43,38,49,60,31,18,5,8,21,24,15,2,33,44,59,54,62,51,36,41,10,7,16,29,42,39,48,61,30,19,4,9,1,12,27,22,53,56,47,34,63,50,37,40,11,6,17,28,20,25,14,3,32,45,58,55},
		{0,14,28,18,56,54,36,42,51,61,47,33,11,5,23,25,37,43,57,55,29,19,1,15,22,24,10,4,46,32,50,60,9,7,21,27,49,63,45,35,58,52,38,40,2,12,30,16,44,34,48,62,20,26,8,6,31,17,3,13,39,41,59,53},
		{0,15,30,17,60,51,34,45,59,52,37,42,7,8,25,22,53,58,43,36,9,6,23,24,14,1,16,31,50,61,44,35,41,38,55,56,21,26,11,4,18,29,12,3,46,33,48,63,28,19,2,13,32,47,62,49,39,40,57,54,27,20,5,10},
		{0,16,32,48,3,19,35,51,6,22,38,54,5,21,37,53,12,28,44,60,15,31,47,63,10,26,42,58,9,25,41,57,24,8,56,40,27,11,59,43,30,14,62,46,29,13,61,45,20,4,52,36,23,7,55,39,18,2,50,34,17,1,49,33},
		{0,17,34,51,7,22,37,52,14,31,44,61,9,24,43,58,28,13,62,47,27,10,57,40,18,3,48,33,21,4,55,38,56,41,26,11,63,46,29,12,54,39,20,5,49,32,19,2,36,53,6,23,35,50,1,16,42,59,8,25,45,60,15,30},
		{0,18,36,54,11,25,47,61,22,4,50,32,29,15,57,43,44,62,8,26,39,53,3,17,58,40,30,12,49,35,21,7,27,9,63,45,16,2,52,38,13,31,41,59,6,20,34,48,55,37,19,1,60,46,24,10,33,51,5,23,42,56,14,28},
		{0,19,38,53,15,28,41,58,30,13,56,43,17,2,55,36,60,47,26,9,51,32,21,6,34,49,4,23,45,62,11,24,59,40,29,14,52,39,18,1,37,54,3,16,42,57,12,31,7,20,33,50,8,27,46,61,25,10,63,44,22,5,48,35},
		{0,20,40,60,19,7,59,47,38,50,14,26,53,33,29,9,15,27,39,51,28,8,52,32,41,61,1,21,58,46,18,6,30,10,54,34,13,25,37,49,56,44,16,4,43,63,3,23,17,5,57,45,2,22,42,62,55,35,31,11,36,48,12,24},
		{0,21,42,63,23,2,61,40,46,59,4,17,57,44,19,6,31,10,53,32,8,29,34,55,49,36,27,14,38,51,12,25,62,43,20,1,41,60,3,22,16,5,58,47,7,18,45,56,33,52,11,30,54,35,28,9,15,26,37,48,24,13,50,39},
		{0,22,44,58,27,13,55,33,54,32,26,12,45,59,1,23,47,57,3,21,52,34,24,14,25,15,53,35,2,20,46,56,29,11,49,39,6,16,42,60,43,61,7,17,48,38,28,10,50,36,30,8,41,63,5,19,4,18,40,62,31,9,51,37},
		{0,23,46,57,31,8,49,38,62,41,16,7,33,54,15,24,63,40,17,6,32,55,14,25,1,22,47,56,30,9,48,39,61,42,19,4,34,53,12,27,3,20,45,58,28,11,50,37,2,21,44,59,29,10,51,36,60,43,18,5,35,52,13,26},
		{0,24,48,40,35,59,19,11,5,29,53,45,38,62,22,14,10,18,58,34,41,49,25,1,15,23,63,39,44,52,28,4,20,12,36,60,55,47,7,31,17,9,33,57,50,42,2,26,30,6,46,54,61,37,13,21,27,3,43,51,56,32,8,16},
		{0,25,50,43,39,62,21,12,13,20,63,38,42,51,24,1,26,3,40,49,61,36,15,22,23,14,37,60,48,41,2,27,52,45,6,31,19,10,33,56,57,32,11,18,30,7,44,53,46,55,28,5,9,16,59,34,35,58,17,8,4,29,54,47},
		{0,26,52,46,43,49,31,5,21,15,33,59,62,36,10,16,42,48,30,4,1,27,53,47,63,37,11,17,20,14,32,58,23,13,35,57,60,38,8,18,2,24,54,44,41,51,29,7,61,39,9,19,22,12,34,56,40,50,28,6,3,25,55,45},
		{0,27,54,45,47,52,25,2,29,6,43,48,50,41,4,31,58,33,12,23,21,14,35,56,39,60,17,10,8,19,62,37,55,44,1,26,24,3,46,53,42,49,28,7,5,30,51,40,13,22,59,32,34,57,20,15,16,11,38,61,63,36,9,18},
		{0,28,56,36,51,47,11,23,37,57,29,1,22,10,46,50,9,21,49,45,58,38,2,30,44,48,20,8,31,3,39,59,18,14,42,54,33,61,25,5,55,43,15,19,4,24,60,32,27,7,35,63,40,52,16,12,62,34,6,26,13,17,53,41},
		{0,29,58,39,55,42,13,16,45,48,23,10,26,7,32,61,25,4,35,62,46,51,20,9,52,41,14,19,3,30,57,36,50,47,8,21,5,24,63,34,31,2,37,56,40,53,18,15,43,54,17,12,28,1,38,59,6,27,60,33,49,44,11,22},
		{0,30,60,34,59,37,7,25,53,43,9,23,14,16,50,44,41,55,21,11,18,12,46,48,28,2,32,62,39,57,27,5,17,15,45,51,42,52,22,8,36,58,24,6,31,1,35,61,56,38,4,26,3,29,63,33,13,19,49,47,54,40,10,20},
		{0,31,62,33,63,32,1,30,61,34,3,28,2,29,60,35,57,38,7,24,6,25,56,39,4,27,58,37,59,36,5,26,49,46,15,16,14,17,48,47,12,19,50,45,51,44,13,18,8,23,54,41,55,40,9,22,53,42,11,20,10,21,52,43},
		{0,32,3,35,6,38,5,37,12,44,15,47,10,42,9,41,24,56,27,59,30,62,29,61,20,52,23,55,18,50,17,49,48,16,51,19,54,22,53,21,60,28,63,31,58,26,57,25,40,8,43,11,46,14,45,13,36,4,39,7,34,2,33,1},
		{0,33,1,32,2,35,3,34,4,37,5,36,6,39,7,38,8,41,9,40,10,43,11,42,12,45,13,44,14,47,15,46,16,49,17,48,18,51,19,50,20,53,21,52,22,55,23,54,24,57,25,56,26,59,27,58,28,61,29,60,30,63,31,62},
		{0,34,7,37,14,44,9,43,28,62,27,57,18,48,21,55,56,26,63,29,54,20,49,19,36,6,35,1,42,8,45,15,51,17,52,22,61,31,58,24,47,13,40,10,33,3,38,4,11,41,12,46,5,39,2,32,23,53,16,50,25,59,30,60},
		{0,35,5,38,10,41,15,44,20,55,17,50,30,61,27,56,40,11,45,14,34,1,39,4,60,31,57,26,54,21,51,16,19,48,22,53,25,58,28,63,7,36,2,33,13,46,8,43,59,24,62,29,49,18,52,23,47,12,42,9,37,6,32,3},
		{0,36,11,47,22,50,29,57,44,8,39,3,58,30,49,21,27,63,16,52,13,41,6,34,55,19,60,24,33,5,42,14,54,18,61,25,32,4,43,15,26,62,17,53,12,40,7,35,45,9,38,2,59,31,48,20,1,37,10,46,23,51,28,56},
		{0,37,9,44,18,55,27,62,36,1,45,8,54,19,63,26,11,46,2,39,25,60,16,53,47,10,38,3,61,24,52,17,22,51,31,58,4,33,13,40,50,23,59,30,32,5,41,12,29,56,20,49,15,42,6,35,57,28,48,21,43,14,34,7},
		{0,38,15,41,30,56,17,55,60,26,51,21,34,4,45,11,59,29,52,18,37,3,42,12,7,33,8,46,25,63,22,48,53,19,58,28,43,13,36,2,9,47,6,32,23,49,24,62,14,40,1,39,16,54,31,57,50,20,61,27,44,10,35,5},
		{0,39,13,42,26,61,23,48,52,19,57,30,46,9,35,4,43,12,38,1,49,22,60,27,31,56,18,53,5,34,8,47,21,50,24,63,15,40,2,37,33,6,44,11,59,28,54,17,62,25,51,20,36,3,41,14,10,45,7,32,16,55,29,58},
		{0,40,19,59,38,14,53,29,15,39,28,52,41,1,58,18,30,54,13,37,56,16,43,3,17,57,2,42,55,31,36,12,60,20,47,7,26,50,9,33,51,27,32,8,21,61,6,46,34,10,49,25,4,44,23,63,45,5,62,22,11,35,24,48},
		{0,41,17,56,34,11,51,26,7,46,22,63,37,12,52,29,14,39,31,54,44,5,61,20,9,32,24,49,43,2,58,19,28,53,13,36,62,23,47,6,27,50,10,35,57,16,40,1,18,59,3,42,48,25,33,8,21,60,4,45,55,30,38,15},
		{0,42,23,61,46,4,57,19,31,53,8,34,49,27,38,12,62,20,41,3,16,58,7,45,33,11,54,28,15,37,24,50,63,21,40,2,17,59,6,44,32,10,55,29,14,36,25,51,1,43,22,60,47,5,56,18,30,52,9,35,48,26,39,13},
		{0,43,21,62,42,1,63,20,23,60,2,41,61,22,40,3,46,5,59,16,4,47,17,58,57,18,44,7,19,56,6,45,31,52,10,33,53,30,32,11,8,35,29,54,34,9,55,28,49,26,36,15,27,48,14,37,38,13,51,24,12,39,25,50},
		{0,44,27,55,54,26,45,1,47,3,52,24,25,53,2,46,29,49,6,42,43,7,48,28,50,30,41,5,4,40,31,51,58,22,33,13,12,32,23,59,21,57,14,34,35,15,56,20,39,11,60,16,17,61,10,38,8,36,19,63,62,18,37,9},
		{0,45,25,52,50,31,43,6,39,10,62,19,21,56,12,33,13,32,20,57,63,18,38,11,42,7,51,30,24,53,1,44,26,55,3,46,40,5,49,28,61,16,36,9,15,34,22,59,23,58,14,35,37,8,60,17,48,29,41,4,2,47,27,54},
		{0,46,31,49,62,16,33,15,63,17,32,14,1,47,30,48,61,19,34,12,3,45,28,50,2,44,29,51,60,18,35,13,57,23,38,8,7,41,24,54,6,40,25,55,56,22,39,9,4,42,27,53,58,20,37,11,59,21,36,10,5,43,26,52},
		{0,47,29,50,58,21,39,8,55,24,42,5,13,34,16,63,45,2,48,31,23,56,10,37,26,53,7,40,32,15,61,18,25,54,4,43,35,12,62,17,46,1,51,28,20,59,9,38,52,27,41,6,14,33,19,60,3,44,30,49,57,22,36,11},
		{0,48,35,19,5,53,38,22,10,58,41,25,15,63,44,28,20,36,55,7,17,33,50,2,30,46,61,13,27,43,56,8,40,24,11,59,45,29,14,62,34,18,1,49,39,23,4,52,60,12,31,47,57,9,26,42,54,6,21,37,51,3,16,32},
		{0,49,33,16,1,48,32,17,2,51,35,18,3,50,34,19,4,53,37,20,5,52,36,21,6,55,39,22,7,54,38,23,8,57,41,24,9,56,40,25,10,59,43,26,11,58,42,27,12,61,45,28,13,60,44,29,14,63,47,30,15,62,46,31},
		{0,50,39,21,13,63,42,24,26,40,61,15,23,37,48,2,52,6,19,33,57,11,30,44,46,28,9,59,35,17,4,54,43,25,12,62,38,20,1,51,49,3,22,36,60,14,27,41,31,45,56,10,18,32,53,7,5,55,34,16,8,58,47,29},
		{0,51,37,22,9,58,44,31,18,33,55,4,27,40,62,13,36,23,1,50,45,30,8,59,54,5,19,32,63,12,26,41,11,56,46,29,2,49,39,20,25,42,60,15,16,35,53,6,47,28,10,57,38,21,3,48,61,14,24,43,52,7,17,34},
		{0,52,43,31,21,33,62,10,42,30,1,53,63,11,20,32,23,35,60,8,2,54,41,29,61,9,22,34,40,28,3,55,46,26,5,49,59,15,16,36,4,48,47,27,17,37,58,14,57,13,18,38,44,24,7,51,19,39,56,12,6,50,45,25},
		{0,53,41,28,17,36,56,13,34,23,11,62,51,6,26,47,7,50,46,27,22,35,63,10,37,16,12,57,52,1,29,40,14,59,39,18,31,42,54,3,44,25,5,48,61,8,20,33,9,60,32,21,24,45,49,4,43,30,2,55,58,15,19,38},
		{0,54,47,25,29,43,50,4,58,12,21,35,39,17,8,62,55,1,24,46,42,28,5,51,13,59,34,20,16,38,63,9,45,27,2,52,48,6,31,41,23,33,56,14,10,60,37,19,26,44,53,3,7,49,40,30,32,22,15,57,61,11,18,36},
		{0,55,45,26,25,46,52,3,50,5,31,40,43,28,6,49,39,16,10,61,62,9,19,36,21,34,56,15,12,59,33,22,13,58,32,23,20,35,57,14,63,8,18,37,38,17,11,60,42,29,7,48,51,4,30,41,24,47,53,2,1,54,44,27},
		{0,56,51,11,37,29,22,46,9,49,58,2,44,20,31,39,18,42,33,25,55,15,4,60,27,35,40,16,62,6,13,53,36,28,23,47,1,57,50,10,45,21,30,38,8,48,59,3,54,14,5,61,19,43,32,24,63,7,12,52,26,34,41,17},
		{0,57,49,8,33,24,16,41,1,56,48,9,32,25,17,40,2,59,51,10,35,26,18,43,3,58,50,11,34,27,19,42,4,61,53,12,37,28,20,45,5,60,52,13,36,29,21,44,6,63,55,14,39,30,22,47,7,62,54,15,38,31,23,46},
		{0,58,55,13,45,23,26,32,25,35,46,20,52,14,3,57,50,8,5,63,31,37,40,18,43,17,28,38,6,60,49,11,39,29,16,42,10,48,61,7,62,4,9,51,19,41,36,30,21,47,34,24,56,2,15,53,12,54,59,1,33,27,22,44},
		{0,59,53,14,41,18,28,39,17,42,36,31,56,3,13,54,34,25,23,44,11,48,62,5,51,8,6,61,26,33,47,20,7,60,50,9,46,21,27,32,22,45,35,24,63,4,10,49,37,30,16,43,12,55,57,2,52,15,1,58,29,38,40,19},
		{0,60,59,7,53,9,14,50,41,21,18,46,28,32,39,27,17,45,42,22,36,24,31,35,56,4,3,63,13,49,54,10,34,30,25,37,23,43,44,16,11,55,48,12,62,2,5,57,51,15,8,52,6,58,61,1,26,38,33,29,47,19,20,40},
		{0,61,57,4,49,12,8,53,33,28,24,37,16,45,41,20,1,60,56,5,48,13,9,52,32,29,25,36,17,44,40,21,2,63,59,6,51,14,10,55,35,30,26,39,18,47,43,22,3,62,58,7,50,15,11,54,34,31,27,38,19,46,42,23},
		{0,62,63,1,61,3,2,60,57,7,6,56,4,58,59,5,49,15,14,48,12,50,51,13,8,54,55,9,53,11,10,52,33,31,30,32,28,34,35,29,24,38,39,25,37,27,26,36,16,46,47,17,45,19,18,44,41,23,22,40,20,42,43,21},
		{0,63,61,2,57,6,4,59,49,14,12,51,8,55,53,10,33,30,28,35,24,39,37,26,16,47,45,18,41,22,20,43,1,62,60,3,56,7,5,58,48,15,13,50,9,54,52,11,32,31,29,34,25,38,36,27,17,46,44,19,40,23,21,42},
		};

		const unsigned char inv[64] = {0,1,33,62,49,43,31,44,57,37,52,28,46,40,22,25,61,54,51,39,26,35,14,24,23,15,20,34,11,53,45,6,63,2,27,21,56,9,50,19,13,47,48,5,7,30,12,41,42,4,38,18,10,29,17,60,36,8,59,58,55,16,3,32};
	#endif
	

	unsigned char w_row=row_weight[0];
	unsigned char w_col=col_weight[0];



	cudaError_t err=cudaSuccess;
    cudaDeviceProp prop;
    
	uchar8 *cuda_ALPHAmn_=NULL;
	uchar8 *cuda_gamma=NULL;
	size_t size=sizeof(uchar8)*M*w_row;
	size_t size_cuda_gamma=sizeof(uchar8)*N;

    //allow pinned memory
    cudaGetDeviceProperties(&prop, 0);
    if (prop.canMapHostMemory==0) 
        cudaSetDeviceFlags(cudaDeviceMapHost);

    //aloccate the image on pinned memory
    err=cudaHostAlloc((void **)&cuda_ALPHAmn_,size, cudaHostAllocDefault);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate cuda_ALPHAmn_ samples(error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
	}
	
	err=cudaHostAlloc((void **)&cuda_gamma,size_cuda_gamma, cudaHostAllocDefault);
    if(err!=cudaSuccess){
        fprintf(stderr, "Failed to allocate cuda_gamma samples(error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

	

	for (int index = 0; index < N; index++) {// initialize variable node message ALPHA with channel info GAMMA
		cuda_gamma[index].s0 = GAMMAn_[(Q*index)+0];
		cuda_gamma[index].s1 = GAMMAn_[(Q*index)+1];
		cuda_gamma[index].s2 = GAMMAn_[(Q*index)+2];
		cuda_gamma[index].s3 = GAMMAn_[(Q*index)+3];
		cuda_gamma[index].s4 = GAMMAn_[(Q*index)+4];
		cuda_gamma[index].s5 = GAMMAn_[(Q*index)+5];
		cuda_gamma[index].s6 = GAMMAn_[(Q*index)+6];
		cuda_gamma[index].s7 = GAMMAn_[(Q*index)+7];
	} 

	

	

	for (int index = 0; index < row_weight[0]*M; index++) {// initialize variable node message ALPHA with channel info GAMMA
		for (int a = 0; a < Q; a++) {
			ALPHAmn_[(index*Q)+a] = GAMMAn_[(Q*col_ind[index])+a];
			//printf("Alpha[%d][%d]=%d, col=%d\n", index,a,ALPHAmn_[(index*Q)+a],col_ind[index]);

		}
		//printf("row=%d, col=%d\n", 0, col_ind[index]);
	}

	for (int row = 0; row < M; row++) {
		for (int index = 0; index < row_weight[0]; index++) {// initialize variable node message ALPHA with channel info GAMMA
			for (int a = 0; a < Q; a++) {
				if(a==0){
					cuda_ALPHAmn_[(row*row_weight[0])+index].s0 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
					//if(col_ind[index]==1)
					//printf("Alpha[%d][%d]=%d, ol_ind[]=%d\n", index,a,cuda_ALPHAmn_[((index%16)*Q)+a].s0, col_ind[index]);
				}
				else if(a==1)
					cuda_ALPHAmn_[(row*row_weight[0])+index].s1 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
				else if(a==2)
					cuda_ALPHAmn_[(row*row_weight[0])+index].s2 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
				else if(a==3)
					cuda_ALPHAmn_[(row*row_weight[0])+index].s3 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
				else if(a==4)
					cuda_ALPHAmn_[(row*row_weight[0])+index].s4 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
				else if(a==5)
					cuda_ALPHAmn_[(row*row_weight[0])+index].s5 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
				else if(a==6)
					cuda_ALPHAmn_[(row*row_weight[0])+index].s6 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
				else if(a==7)
					cuda_ALPHAmn_[(row*row_weight[0])+index].s7 = ALPHAmn_[(((row*row_weight[0])+index)*Q)+a];
			}
		}
	}
	
	printf("w_row=%d\n",M);

	

	err=cudaSuccess; 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	//======================================================================================================================================================================
	//kernel dimensions
		#if Q==8
			dim3 FB_threadsPerBlock(128,1,1);
			dim3 FB_numBlocks(2,1,1);

			dim3 CN_threadsPerBlock(32,w_row,1);
			dim3 CN_numBlocks(8,1,1);

			dim3 VN_threadsPerBlock(w_col,8,1);
			dim3 VN_numBlocks(1,48,1);
		#elif Q==16
			dim3 FB_threadsPerBlock(8,1,Q);
			dim3 FB_numBlocks(2,1,1);

			dim3 CN_threadsPerBlock(4,w_row,Q);
			dim3 CN_numBlocks(4,1,1);

			dim3 VN_threadsPerBlock(w_col,32,Q);
			dim3 VN_numBlocks(1,12,1);
		#elif Q==32
			dim3 FB_threadsPerBlock(8,1,Q);
			dim3 FB_numBlocks(2,1,1);

			dim3 CN_threadsPerBlock(4,w_row,Q);
			dim3 CN_numBlocks(4,1,1);

			dim3 VN_threadsPerBlock(w_col,16,Q);
			dim3 VN_numBlocks(1,24,1);
		#endif
  
	//======================================================================================================================================================================
	//size of variables
		size_t size_gamma_8=sizeof(uchar8)*N;
		size_t size_alpha_8=sizeof(uchar8)*M*w_row;
		size_t size_FB_8=sizeof(uchar8)*M*w_row;
		size_t size_FB=sizeof(unsigned char)*M*w_row*Q;
		size_t size_dimension=sizeof(unsigned short);
		size_t size_GF=sizeof(unsigned char)*Q*Q;
		size_t size_GF_inv=sizeof(unsigned char)*Q;
		size_t size_row_ptr=sizeof(unsigned short)*(M+1);
		size_t size_val=sizeof(unsigned char)*M*w_row;
		size_t size_weight=sizeof(unsigned char);
		size_t size_col_ptr=sizeof(unsigned short)*(N+1);
		size_t size_ptr_val=sizeof(unsigned short)*M*w_row;
	
	//======================================================================================================================================================================
    //variables declaration
		uchar8 *d_gamma_8=NULL;
		uchar8 *d_alpha_8=NULL;;
		uchar8 *d_F_8=NULL;
		uchar8 *d_B_8=NULL;
		uchar8 *d_beta_8=NULL;

		uchar8 *h_alpha_8=NULL;
		uchar8 *h_F_8=NULL;
		uchar8 *h_B_8=NULL;
		uchar8 *h_beta_8=NULL;
		/* #if Q==32
			unsigned char *d_h_nb=NULL;
		#endif */
		
				
	//======================================================================================================================================================================
	//allocate host memory

		h_F=(unsigned char *)malloc(size_FB);
		if(h_F == NULL){
			printf("Failed to allocate host F\n" );
			exit(EXIT_FAILURE);
		}

		h_B=(unsigned char *)malloc(size_FB);
		if(h_B == NULL){
			printf("Failed to allocate host B\n" );
			exit(EXIT_FAILURE);
		}

		err=cudaHostAlloc((void **)&h_F_8, size_FB_8, cudaHostAllocDefault );
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate host h_F_8(error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaHostAlloc((void **)&h_B_8, size_FB_8, cudaHostAllocDefault );
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate host h_B_8(error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaHostAlloc((void **)&h_alpha_8, size_alpha_8, cudaHostAllocDefault );
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate host h_alpha_8(error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaHostAlloc((void **)&h_beta_8, size_alpha_8, cudaHostAllocDefault );
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate host h_beta_8(error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		

	//======================================================================================================================================================================
    //allocate device memory
		err=cudaMalloc((void **)&d_gamma_8, size_gamma_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device gamma (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_alpha_8, size_alpha_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device alpha8 (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_F_8, size_FB_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device F (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_B_8, size_FB_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device B (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_beta_8, size_alpha_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device beta (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}


		/* #if Q==32
			err=cudaMalloc((void **)&d_h_nb, size_h_nb);
			if(err!=cudaSuccess){
				fprintf(stderr, "Failed to allocate device h_nb (error code %d)!\n", cudaGetLastError());
				exit(EXIT_FAILURE);
			}
		#endif */

	//======================================================================================================================================================================
    //device memory initialization
		err=cudaMemset(d_gamma_8, 0, size_gamma_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device gamma (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_alpha_8, 0, size_alpha_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device alpha (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_F_8, 0, size_FB_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device d_F_8 (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_B_8, 0, size_FB_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device B (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_beta_8, 0, size_alpha_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device beta (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

				
	//======================================================================================================================================================================
	//copy data to constant

		err=cudaMemcpyToSymbol(c_add, &add, size_GF,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy add from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_M, &M, size_dimension,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy M from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_N, &N, size_dimension,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy N from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_mult, &mult, size_GF,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy mult from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_inv, &inv, size_GF_inv,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy w_row from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_row_ptr, &row_ptr, size_row_ptr, 0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy row_ptr from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_val, &val, size_val,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy val from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_w_col, &w_col, size_weight,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy w_col from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_w_row, &w_row, size_weight,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy w_row from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_col_ptr, &col_ptr, size_col_ptr, 0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy row_ptr from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_ptr_to_val, &ptr_to_val, size_ptr_val,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy val from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_row_ind, &row_ind, size_ptr_val,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy row_ind from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_col_ind, &col_ind, size_ptr_val,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy row_ind from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}
	//======================================================================================================================================================================
	//copy data to device  

		err=cudaMemcpy(d_gamma_8, cuda_gamma, size_gamma_8, cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy gamma from host to device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(d_alpha_8, cuda_ALPHAmn_, size_alpha_8, cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy alpha_8 from host to device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}
	//======================================================================================================================================================================
	//execute the kernel
		cudaEventRecord(start);
		for (int iter=0; iter < MAX_ITERATION; iter++) {
				GPU_FB_metrics<<<FB_numBlocks, FB_threadsPerBlock>>>( d_alpha_8, d_F_8, d_B_8, iter);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
				}

				GPU_CN<<<CN_numBlocks, CN_threadsPerBlock>>>( d_beta_8, d_F_8, d_B_8, iter);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
				} 

				GPU_VN<<<VN_numBlocks, VN_threadsPerBlock>>>( d_beta_8, d_alpha_8, d_gamma_8, iter);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE); 
				} 
			//#endif
		}
		cudaEventRecord(stop);


		

	//======================================================================================================================================================================
	//copy the data from device to host

		err=cudaMemcpy(h_F_8, d_F_8, size_FB_8, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the F_8 from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(h_B_8, d_B_8, size_FB_8, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the B_8 from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(h_beta_8, d_beta_8, size_alpha_8, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the betta from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(cuda_ALPHAmn_, d_alpha_8, size_alpha_8, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the d_alpha_8 from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		cudaEventSynchronize(stop);

	//======================================================================================================================================================================
	//free the device memory
		err=cudaFree(d_gamma_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the gamma_8 from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_alpha_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the alpha_8 from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_F_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the F_8 from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_B_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the B_8 from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_beta_8);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the d_beta_8 from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

	//======================================================================================================================================================================
	//save data

		for (int row = 0; row < M; row++) {
			for (int index = 0; index < row_weight[0]; index++) {// initialize variable node message ALPHA with channel info GAMMA
				h_F[(row*Q*row_weight[0])+(0*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s0;
				h_F[(row*Q*row_weight[0])+(1*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s1;
				h_F[(row*Q*row_weight[0])+(2*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s2;
				h_F[(row*Q*row_weight[0])+(3*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s3;
				h_F[(row*Q*row_weight[0])+(4*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s4;
				h_F[(row*Q*row_weight[0])+(5*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s5;
				h_F[(row*Q*row_weight[0])+(6*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s6;
				h_F[(row*Q*row_weight[0])+(7*row_weight[0])+index]=h_F_8[(row*row_weight[0])+index].s7;
				
				h_B[(row*Q*row_weight[0])+(0*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s0;
				h_B[(row*Q*row_weight[0])+(1*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s1;
				h_B[(row*Q*row_weight[0])+(2*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s2;
				h_B[(row*Q*row_weight[0])+(3*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s3;
				h_B[(row*Q*row_weight[0])+(4*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s4;
				h_B[(row*Q*row_weight[0])+(5*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s5;
				h_B[(row*Q*row_weight[0])+(6*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s6;
				h_B[(row*Q*row_weight[0])+(7*row_weight[0])+index]=h_B_8[(row*row_weight[0])+index].s7;
			}
		}

		for (int row = 0; row < M; row++) {
			for (int index = 0; index < row_weight[0]; index++) {// initialize variable node message ALPHA with channel info GAMMA
				h_beta[(row*Q*row_weight[0])+(index*Q)+0]=h_beta_8[(row*row_weight[0])+index].s0;
				h_beta[(row*Q*row_weight[0])+(index*Q)+1]=h_beta_8[(row*row_weight[0])+index].s1;
				h_beta[(row*Q*row_weight[0])+(index*Q)+2]=h_beta_8[(row*row_weight[0])+index].s2;
				h_beta[(row*Q*row_weight[0])+(index*Q)+3]=h_beta_8[(row*row_weight[0])+index].s3;
				h_beta[(row*Q*row_weight[0])+(index*Q)+4]=h_beta_8[(row*row_weight[0])+index].s4;
				h_beta[(row*Q*row_weight[0])+(index*Q)+5]=h_beta_8[(row*row_weight[0])+index].s5;
				h_beta[(row*Q*row_weight[0])+(index*Q)+6]=h_beta_8[(row*row_weight[0])+index].s6;
				h_beta[(row*Q*row_weight[0])+(index*Q)+7]=h_beta_8[(row*row_weight[0])+index].s7;
			}
		}
		
		

		FILE *fp_B_val = fopen("./data/B_val.txt", "w");
		FILE *fp_F_val = fopen("./data/F_val.txt", "w");

		
		for (int row = 0; row < M; row++) {
			for (int a = 0; a < Q; a++) {	
				for (int col = 0; col < row_weight[0]; col++) {	//2. update BETA values from B and F values
					
					//if(row==0)
					//printf("F[%d][%d][%d]=%d\n",row, a,col,h_F[(row*Q*w_row)+(a*w_row)+col]);
					//printf("B[%d][%d][%d]=%d\n",row, a,w_row - index_c -1,h_B[(row*Q*w_row)+(a*w_row)+w_row - index_c -1]);
					//printf("Alpha[%d][%d][%d]=%d\n",row, col,a ,h_alpha[(Q*N*row)+(Q*col)+a]);
					fprintf(fp_B_val,"%d ",h_B[(row*Q*w_row)+(col*Q)+a]);
					fprintf(fp_F_val,"%d ", h_F[(row*Q*w_row)+(col*Q)+a]);
				}
				fprintf(fp_B_val,"\n");
				fprintf(fp_F_val,"\n");
			}
			fprintf(fp_B_val,"\n\n");
			fprintf(fp_F_val,"\n\n");
		}

		fclose(fp_B_val);
		fclose(fp_F_val);

	//======================================================================================================================================================================
	//free the host memory
		
	cudaEventElapsedTime(&milliseconds, start, stop);

	/* #if Q==32
		free(h_nb);
	#endif */



	for (int row = 0; row < M; row++) {
		for (int index = 0; index < row_weight[0]; index++) {// initialize variable node message ALPHA with channel info GAMMA
			for (int a = 0; a < Q; a++) {
				if(a==0){
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s0 ;
					//if(col_ind[index]==1)
					//printf(index%16)*Q)+a].s0, col_ind[index])"Alpha[%d][%d]=%d, ol_ind[]=%d\n", index,a,cuda_ALPHAmn_[((;
				}
				else if(a==1)
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s1 ;
				else if(a==2)
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s2 ;
				else if(a==3)
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s3 ;
				else if(a==4)
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s4 ;
				else if(a==5)
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s5 ;
				else if(a==6)
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s6 ;
				else if(a==7)
					ALPHAmn_[(((row*row_weight[0])+index)*Q)+a]=cuda_ALPHAmn_[(row*row_weight[0])+index].s7 ;
			}

		}
	}
    
    err=cudaFreeHost(cuda_ALPHAmn_);
	if(err!=cudaSuccess){
		fprintf(stderr, "Failed to free the cuda_ALPHAmn_ from the host (error code %d)!\n", cudaGetLastError());
		exit(EXIT_FAILURE);
	}

	err=cudaFreeHost(cuda_gamma);
	if(err!=cudaSuccess){
		fprintf(stderr, "Failed to free the cuda_gamma from the host (error code %d)!\n", cudaGetLastError());
		exit(EXIT_FAILURE);
	}

	err=cudaFreeHost(h_B_8);
	if(err!=cudaSuccess){
		fprintf(stderr, "Failed to free the h_B_8 from the host (error code %d)!\n", cudaGetLastError());
		exit(EXIT_FAILURE);
	}

	err=cudaFreeHost(h_F_8);
	if(err!=cudaSuccess){
		fprintf(stderr, "Failed to free the h_F_8 from the host (error code %d)!\n", cudaGetLastError());
		exit(EXIT_FAILURE);
	}

	err=cudaFreeHost(h_beta_8);
	if(err!=cudaSuccess){
		fprintf(stderr, "Failed to free the h_beta_8 from the host (error code %d)!\n", cudaGetLastError());
		exit(EXIT_FAILURE);
	}

	
	

			
	return 0;
}
