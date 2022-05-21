
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

#if Q==4
__constant__ unsigned short c_col_row[3][2];
__constant__ unsigned short c_row_col[1][6];
__constant__ unsigned char c_h_nb[3][6];
#elif Q==8
__constant__ unsigned short c_col_row[14][3];
__constant__ unsigned short c_row_col[1][42];
__constant__ unsigned char c_h_nb[14][42];
#elif Q==16
__constant__ unsigned short c_col_row[150][6];
__constant__ unsigned short c_row_col[3][300];
__constant__ unsigned char c_h_nb[150][300];
#elif Q==32
__constant__ unsigned short c_col_row[310][6];
__constant__ unsigned short c_row_col[3][620];
#endif


__constant__ unsigned short c_M;
__constant__ unsigned short c_N;
__constant__ unsigned char c_add[Q][Q];
__constant__ unsigned char c_mult[Q][Q];
__constant__ unsigned char c_inv[Q];
__constant__ unsigned char c_w_row;
__constant__ unsigned char c_w_col;

extern unsigned char *h_F;
extern unsigned char *h_B;

#if Q==32
	//===================================
	// CUDA kernel
	//===================================
	__global__ void GPU_FB_metrics( unsigned char * d_alpha,  unsigned char * d_F,  unsigned char * d_B, int iter, unsigned char * d_h_nb){
		
		unsigned short x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned short z=threadIdx.z+blockIdx.z*blockDim.z;
		unsigned char min_F, max_F, min_B, max_B, a_F, a_B;

		__shared__ unsigned char s_alpha[32][6][32];
		__shared__ unsigned char s_F[32][32][6];
		__shared__ unsigned char s_B[32][32][6];

		if(x<c_M){


			for(unsigned char index=0; index<c_w_row; index++){
				s_alpha[threadIdx.x][index][z]=d_alpha[(Q*c_N*x)+(Q*c_col_row[x][index])+z];
			}
			__syncthreads();

			s_F[threadIdx.x][z][0]=s_alpha[threadIdx.x][0][c_mult[c_inv[d_h_nb[(x*c_N)+c_col_row[x][0]]]][z]];
			//if(x==0)
			//printf("F[%d][%d][0]=%d\n",x,z,s_F[threadIdx.x][z][0]);

			s_B[threadIdx.x][z][c_w_row-1]=s_alpha[threadIdx.x][c_w_row-1][c_mult[c_inv[d_h_nb[(x*c_N)+c_col_row[x][c_w_row-1]]]][z]];
			//if(x==1)
			//printf(" B[%d][%d][%d]=%d, row=%d, lastcol=%d, mult=%d\n",x ,z,c_w_row-1, s_B[threadIdx.x][z][c_w_row-1], x, c_col_row[x][c_w_row-1],c_mult[c_inv[c_h_nb[x][c_col_row[x][0]]]][z]);

			
			d_F[(x*Q*c_w_row)+z*c_w_row]=s_F[threadIdx.x][z][0];
			//if(x==0)
			//printf("F[%d][%d]=%d\n",x,z,d_F[(x*Q*c_w_row)+z*c_w_row]);

			d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row-1]=s_B[threadIdx.x][z][c_w_row-1];

			__syncthreads();
			


			for(unsigned char index=1; index<c_w_row; index++){
				//s_alpha[threadIdx.x][0][z]=d_alpha[(Q*c_N*x)+(Q*c_col_row[x][index])+z];
				//s_alpha[threadIdx.x][1][z]=d_alpha[(Q*c_N*x)+(Q*c_col_row[x][c_w_row - index -1])+z];
				min_F = (s_F[threadIdx.x][z][index-1] < s_alpha[threadIdx.x][index][0]) ? s_alpha[threadIdx.x][index][0] : s_F[threadIdx.x][z][index-1];
				//if(x==0 && index==1)
				//printf("min_F=%d, F[%d][%d]=%d, ALPHAmn_[%d][%d][0]=%d\n", min_F,z,index, s_F[threadIdx.x][z][index-1],x,c_col_row[x][index], s_alpha[threadIdx.x][index][z]);
				min_B = (s_B[threadIdx.x][z][c_w_row - index] < s_alpha[threadIdx.x][c_w_row - index-1][0]) ? s_alpha[threadIdx.x][c_w_row - index-1][0] : s_B[threadIdx.x][z][c_w_row - index];
				//if(x==1)
				//printf("min_F=%d, B[%d][%d]=%d, ALPHAmn_[%d][%d][0]=%d\n", min_B,z,c_w_row - index, d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row - index],x,c_col_row[x][c_w_row - index -1], d_alpha[(Q*c_N*x)+(Q*c_col_row[x][c_w_row - index -1])]);
				for(unsigned char b=1; b<Q; b++){
					a_F=c_add[z][c_mult[d_h_nb[(x*c_N)+c_col_row[x][index]]][b]];
					a_B=c_add[z][c_mult[d_h_nb[(x*c_N)+c_col_row[x][c_w_row - index -1]]][b]];
					
					
					max_F = (s_F[threadIdx.x][a_F][index-1] < s_alpha[threadIdx.x][index][b]) ? s_alpha[threadIdx.x][index][b] : s_F[threadIdx.x][a_F][index-1];
					max_B = (s_B[threadIdx.x][a_B][c_w_row - index] < s_alpha[threadIdx.x][c_w_row - index-1][b]) ? s_alpha[threadIdx.x][c_w_row - index-1][b] : s_B[threadIdx.x][a_B][c_w_row - index] ;
					//if(x==0 && index==2)
					//printf("min_F=%d, max_F=%d, F[%d][%d][%d]=%d,alpha[%d][%d][%d]=%d \n", min_F,max_F,x,a_F,index-1,d_F[(x*Q*c_w_row)+(a_F*c_w_row)+index-1], x,c_col_row[x][index],b, d_alpha[(Q*c_N*x)+(Q*c_col_row[x][index])+b]);

					min_F = (min_F < max_F) ? min_F : max_F;
					min_B = (min_B < max_B) ? min_B : max_B;

					//if(x==0 && index==2)
					//printf("min_F=%d, max_F=%d, F[%d][%d][%d]=%d,alpha[%d][%d][%d]=%d \n", min_F,max_F,x,a_F,index-1,s_F[threadIdx.x][a_F][index-1], x,c_col_row[x][index],b, s_alpha[threadIdx.x][0][b]);
					
				}
				__syncthreads();

				d_F[(x*Q*c_w_row)+(z*c_w_row)+index]=min_F;
				s_F[threadIdx.x][z][index]=min_F;
				//if(x==0 && index==2)
				//printf("F[%d][%d][%d]=%d\n",x,z,index,d_F[(x*Q*c_w_row)+(z*c_w_row)+index]);
				d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row - index -1]=min_B;
				s_B[threadIdx.x][z][c_w_row - index-1]=min_B;
				//if(x==7)
				//printf("B[%d][%d][%d]=%d\n",x,z,c_w_row - index -1,d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row - index -1]);
				//__syncthreads();
			}
		}

		

	} 

	__global__ void GPU_CN( unsigned char * d_beta,  unsigned char * d_F,  unsigned char * d_B, int iter, unsigned char * d_h_nb){
		unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;
		unsigned int z=threadIdx.z+blockIdx.z*blockDim.z;

		__shared__ unsigned char s_F[5][32][6];
		__shared__ unsigned char s_B[5][32][6];

		s_F[threadIdx.x][z][y]=d_F[(x*Q*c_w_row)+(z*c_w_row)+y];
		s_B[threadIdx.x][z][y]=d_B[(x*Q*c_w_row)+(z*c_w_row)+y];
		__syncthreads();

		if(y==0){
			d_beta[(x*c_N*Q)+(Q*c_col_row[x][0])+z]=s_B[threadIdx.x][c_mult[d_h_nb[(x*c_N)+c_col_row[x][0]]][z]][1];
			//if(x==1)
			//printf("BETAmn_[%d][%d][%d]=%d\n", x, c_col_row[x][0], z, d_beta[(x*c_N*Q)+(Q*c_col_row[x][0])+z]);

		}
		else if(y==c_w_row-1){
			d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z]=s_F[threadIdx.x][c_mult[d_h_nb[(x*c_N)+c_col_row[x][y]]][z]][y-1];
			//if(x==0)
			//printf("BETAmn_[%d][%d][%d]=%d\n", x, c_col_row[x][y], z, d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z]);

		}
		else{
			unsigned char min, max, a;
			min = (s_F[threadIdx.x][c_mult[d_h_nb[(x*c_N)+c_col_row[x][y]]][z]][y-1] < s_B[threadIdx.x][0][y+1]) ? s_B[threadIdx.x][0][y+1] : s_F[threadIdx.x][c_mult[d_h_nb[(x*c_N)+c_col_row[x][y]]][z]][y-1];
			//if(x==0 && z==2) 
			//printf("mmin=%d F[%d][%d][%d]=%d, B[%d][%d][%d]=%d, col=%d\n", min,x, (c_mult[c_h_nb[x][c_col_row[x][y]]][z]), y-1, d_F[(x*Q*c_w_row)+((c_mult[c_h_nb[x][c_col_row[x][y]]][z])*c_w_row)+y-1], x, 0, y+1, d_B[(x*Q*c_w_row)+y+1],c_col_row[x][y]);
			for(unsigned char b=1; b<Q; b++){
				a=c_add[b][c_mult[d_h_nb[(x*c_N)+c_col_row[x][y]]][z]];
				max = (s_F[threadIdx.x][a][y-1] < s_B[threadIdx.x][b][y+1]) ? s_B[threadIdx.x][b][y+1] : s_F[threadIdx.x][a][y-1] ;
				//if(x==0 && z==2)
				//printf("max=%d min=%d z=%d, b=%d, a=%d\n", max, min, z, b, a);

				min = (min < max) ? min : max;
				//if(x==0 && z==2)
				//printf("min=%d\n", min);

			}
			d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z]=min;
			//if(x==0)
			//printf("BETAmn_[%d][%d][%d]=%d y=%d\n", x, c_col_row[x][y], z, d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z], y);

		}

	}

	__global__ void GPU_VN( unsigned char * d_beta,  unsigned char * d_alpha,  unsigned char * d_alpha_t, unsigned char * d_alpha_t2, unsigned char * d_gamma, int iter){
		unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;
		unsigned int z=threadIdx.z+blockIdx.z*blockDim.z;
		unsigned char temp=0;

		__shared__ unsigned char s_beta[3][10][32];
		__shared__ unsigned char s_alpha_t[3][10][32];
		__shared__ unsigned char s_alpha_t2[3][10][32];

		s_beta[x][threadIdx.y][z]=d_beta[(c_row_col[x][y]*c_N*Q)+(y*Q)+z];
			__syncthreads();

			for(unsigned char index=0; index<c_w_col; index++){
				if(index!=x){
					temp=temp+s_beta[index][threadIdx.y][z];
				}
				__syncthreads();
			}

			//if( c_row_col[x][y]==0 && y==8 )
			//printf("temp=%d, GAMMAn_[%d][%d]=%d\n", temp, y,z,d_gamma[(y*Q)+z]);


			//__syncthreads();

			s_alpha_t[x][threadIdx.y][z]= temp+ d_gamma[(y*Q)+z];
			
			s_alpha_t2[x][threadIdx.y][z]=z;
			__syncthreads();

			//if(x==0 && y==0)
			//printf("alpha_t[%d][%d][%d]=%d\n",x,y,z,d_alpha_t[(x*c_N*Q)+(Q*y)+z]);


			
			for(unsigned char stride=blockDim.z/2; stride>0; stride>>=1){
				if(z<stride){
					/* if( c_row_col[x][y]==0 ){
					printf("t=%d min=%d y=%d z=%d\n", d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z]], d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride]],y,z);
					} */
					s_alpha_t2[x][threadIdx.y][z]= (s_alpha_t[x][threadIdx.y][s_alpha_t2[x][threadIdx.y][z]] > s_alpha_t[x][threadIdx.y][s_alpha_t2[x][threadIdx.y][z+stride]]) ? s_alpha_t2[x][threadIdx.y][z+stride] : s_alpha_t2[x][threadIdx.y][z];
					//if( c_row_col[x][y]==0 ){
						//printf("t=%d min=%d y=%d z=%d\n", d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z]], d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride]],y,z);
						//}
					//if(x==0 && y==0 )
					//printf("alpha_t2[%d][%d][%d]=%d\n",x,y,z,d_alpha_t2[(x*c_N*Q)+(Q*y)+z]);

				}
				__syncthreads();
			}

			//__syncthreads();

			
			//if(x==0  && z==0)
			//printf("alpha_t2[%d][%d][%d]=%d\n",x,y,z,d_alpha_t2[(x*c_N*Q)+(Q*y)+z]);

			//if(c_row_col[x][y]==41 )
			//printf("ALPHA_t[%d]=%d ALPHA_t2[%d]=%d\n", z, d_alpha_t[(x*c_N*Q)+(Q*y)+z], d_alpha_t2[(x*c_N*Q)+(Q*y)],  d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)]]);
			
			temp=s_alpha_t[x][threadIdx.y][s_alpha_t2[x][threadIdx.y][0]];
			//__syncthreads();
			//if( c_row_col[x][y]==0 && y==155 && z==1)
			//printf("temp=%d\n", temp);
			d_alpha[(c_row_col[x][y]*c_N*Q)+(y*Q)+z] = s_alpha_t[x][threadIdx.y][z] - temp;

	}

#else
	//===================================
	// CUDA kernel
	//===================================
	__global__ void GPU_FB_metrics( unsigned char * d_alpha,  unsigned char * d_F,  unsigned char * d_B, int iter){
		
		unsigned short x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned short z=threadIdx.z+blockIdx.z*blockDim.z;
		unsigned char min_F, max_F, min_B, max_B, a_F, a_B;
		
		#if Q==4
		__shared__ unsigned char s_alpha[3][2][4];
		__shared__ unsigned char s_F[3][4][2];
		__shared__ unsigned char s_B[3][4][2];
		#elif Q==8
		__shared__ unsigned char s_alpha[14][2][8];
		__shared__ unsigned char s_F[14][8][2];
		__shared__ unsigned char s_B[14][8][2];
		#elif Q==16
		__shared__ unsigned char s_alpha[64][6][16];
		__shared__ unsigned char s_F[64][16][6];
		__shared__ unsigned char s_B[64][16][6];
		#endif

		/* if(x==2)
		printf("alpha[%d][0][%d]=%d\n",x,z,d_alpha[(Q*c_N*x)+(Q*c_col_row[x][0])+z]); */


		if(x<c_M){


			for(unsigned char index=0; index<c_w_row; index++){
				s_alpha[threadIdx.x][index][z]=d_alpha[(Q*c_N*x)+(Q*c_col_row[x][index])+z];
			}
			__syncthreads();

			s_F[threadIdx.x][z][0]=s_alpha[threadIdx.x][0][c_mult[c_inv[c_h_nb[x][c_col_row[x][0]]]][z]];
			/* if(x==2)
			printf("F[%d][%d][0]=%d\n",x,z,s_F[threadIdx.x][z][0]); */

			s_B[threadIdx.x][z][c_w_row-1]=s_alpha[threadIdx.x][c_w_row-1][c_mult[c_inv[c_h_nb[x][c_col_row[x][c_w_row-1]]]][z]];
			//if(x==2)
			//printf(" B[%d][%d][%d]=%d, row=%d, lastcol=%d, mult=%d\n",x ,z,c_w_row-1, s_B[threadIdx.x][z][c_w_row-1], x, c_col_row[x][c_w_row-1],c_mult[c_inv[c_h_nb[x][c_col_row[x][0]]]][z]);

			
			d_F[(x*Q*c_w_row)+z*c_w_row]=s_F[threadIdx.x][z][0];
			//if(x==0)
			//printf("F[%d][%d]=%d\n",x,z,d_F[(x*Q*c_w_row)+z*c_w_row]);

			d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row-1]=s_B[threadIdx.x][z][c_w_row-1];

			__syncthreads();
			


			for(unsigned char index=1; index<c_w_row; index++){
				//s_alpha[threadIdx.x][0][z]=d_alpha[(Q*c_N*x)+(Q*c_col_row[x][index])+z];
				//s_alpha[threadIdx.x][1][z]=d_alpha[(Q*c_N*x)+(Q*c_col_row[x][c_w_row - index -1])+z];
				min_F = (s_F[threadIdx.x][z][index-1] < s_alpha[threadIdx.x][index][0]) ? s_alpha[threadIdx.x][index][0] : s_F[threadIdx.x][z][index-1];
				//if(x==0 && index==1)
				//printf("min_F=%d, F[%d][%d]=%d, ALPHAmn_[%d][%d][0]=%d\n", min_F,z,index, s_F[threadIdx.x][z][index-1],x,c_col_row[x][index], s_alpha[threadIdx.x][index][z]);
				min_B = (s_B[threadIdx.x][z][c_w_row - index] < s_alpha[threadIdx.x][c_w_row - index-1][0]) ? s_alpha[threadIdx.x][c_w_row - index-1][0] : s_B[threadIdx.x][z][c_w_row - index];
				//if(x==1)
				//printf("min_F=%d, B[%d][%d]=%d, ALPHAmn_[%d][%d][0]=%d\n", min_B,z,c_w_row - index, d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row - index],x,c_col_row[x][c_w_row - index -1], d_alpha[(Q*c_N*x)+(Q*c_col_row[x][c_w_row - index -1])]);
				for(unsigned char b=1; b<Q; b++){
					a_F=c_add[z][c_mult[c_h_nb[x][c_col_row[x][index]]][b]];
					a_B=c_add[z][c_mult[c_h_nb[x][c_col_row[x][c_w_row - index -1]]][b]];
					//if(x==0)
					//printf("a[%d][%d][%d][%d]=%d\n",x, index, z, b, a_B);
					
					
					max_F = (s_F[threadIdx.x][a_F][index-1] < s_alpha[threadIdx.x][index][b]) ? s_alpha[threadIdx.x][index][b] : s_F[threadIdx.x][a_F][index-1];
					max_B = (s_B[threadIdx.x][a_B][c_w_row - index] < s_alpha[threadIdx.x][c_w_row - index-1][b]) ? s_alpha[threadIdx.x][c_w_row - index-1][b] : s_B[threadIdx.x][a_B][c_w_row - index] ;
					//if(x==0 && index==2)
					//printf("min_F=%d, max_F=%d, F[%d][%d][%d]=%d,alpha[%d][%d][%d]=%d \n", min_F,max_F,x,a_F,index-1,d_F[(x*Q*c_w_row)+(a_F*c_w_row)+index-1], x,c_col_row[x][index],b, d_alpha[(Q*c_N*x)+(Q*c_col_row[x][index])+b]);

					//if(x==0)
					//printf("max_B[%d][%d][%d]=%d\n",z,x, b, max_B);
					

					min_F = (min_F < max_F) ? min_F : max_F;
					min_B = (min_B < max_B) ? min_B : max_B;

					/* if(x==0)
					printf("min_B[%d][%d][%d]=%d\n",z,x, b, min_B); */

					//if(x==0 && index==2)
					//printf("min_F=%d, max_F=%d, F[%d][%d][%d]=%d,alpha[%d][%d][%d]=%d \n", min_F,max_F,x,a_F,index-1,s_F[threadIdx.x][a_F][index-1], x,c_col_row[x][index],b, s_alpha[threadIdx.x][0][b]);
					
				}
				__syncthreads();

				d_F[(x*Q*c_w_row)+(z*c_w_row)+index]=min_F;
				s_F[threadIdx.x][z][index]=min_F;

				//if(x==0)
				//printf("min_B[%d][%d]=%d\n",z,x, d_F[(x*Q*c_w_row)+(z*c_w_row)+index]);
				//if(x==0 && index==2)
				//printf("F[%d][%d][%d]=%d\n",x,z,index,d_F[(x*Q*c_w_row)+(z*c_w_row)+index]);
				d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row - index -1]=min_B;
				s_B[threadIdx.x][z][c_w_row - index-1]=min_B;

				//if(x==0)
				//printf("min_B[%d][%d]=%d\n",z,x, d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row - index -1]);

				//if(x==7)
				//printf("B[%d][%d][%d]=%d\n",x,z,c_w_row - index -1,d_B[(x*Q*c_w_row)+(z*c_w_row)+c_w_row - index -1]);
				//__syncthreads();
			}
		} 

	} 

	__global__ void GPU_CN( unsigned char * d_beta,  unsigned char * d_F,  unsigned char * d_B, int iter){
		unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;
		unsigned int z=threadIdx.z+blockIdx.z*blockDim.z;
		

		#if Q==4
		__shared__ unsigned char s_F[3][4][2];
		__shared__ unsigned char s_B[3][4][2];
		#elif Q==8
		__shared__ unsigned char s_F[14][8][2];
		__shared__ unsigned char s_B[14][8][2];
		#elif Q==16
		__shared__ unsigned char s_F[10][16][6];
		__shared__ unsigned char s_B[10][16][6];
		#endif

		//if(x<c_M){

			s_F[threadIdx.x][z][y]=d_F[(x*Q*c_w_row)+(z*c_w_row)+y];
			s_B[threadIdx.x][z][y]=d_B[(x*Q*c_w_row)+(z*c_w_row)+y];
			__syncthreads();

			if(y==0){
				d_beta[(x*c_N*Q)+(Q*c_col_row[x][0])+z]=s_B[threadIdx.x][c_mult[c_h_nb[x][c_col_row[x][0]]][z]][1];
				if(x==1)
				printf("BETAmn_[%d][%d][%d]=%d, new_Z=%d\n", x, c_col_row[x][0], z, d_beta[(x*c_N*Q)+(Q*c_col_row[x][0])+z], c_mult[c_h_nb[x][c_col_row[x][0]]][z]);

			}
			else if(y==c_w_row-1){
				d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z]=s_F[threadIdx.x][c_mult[c_h_nb[x][c_col_row[x][y]]][z]][y-1];
				//if(x==0)
				//printf("BETAmn_[%d][%d][%d]=%d\n", x, c_col_row[x][y], z, d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z]);

			}
			else{
				unsigned char min, max, a;
				min = (s_F[threadIdx.x][c_mult[c_h_nb[x][c_col_row[x][y]]][z]][y-1] < s_B[threadIdx.x][0][y+1]) ? s_B[threadIdx.x][0][y+1] : s_F[threadIdx.x][c_mult[c_h_nb[x][c_col_row[x][y]]][z]][y-1];
				//if(x==0 && z==2) 
				//printf("mmin=%d F[%d][%d][%d]=%d, B[%d][%d][%d]=%d, col=%d\n", min,x, (c_mult[c_h_nb[x][c_col_row[x][y]]][z]), y-1, d_F[(x*Q*c_w_row)+((c_mult[c_h_nb[x][c_col_row[x][y]]][z])*c_w_row)+y-1], x, 0, y+1, d_B[(x*Q*c_w_row)+y+1],c_col_row[x][y]);
				for(unsigned char b=1; b<Q; b++){
					a=c_add[b][c_mult[c_h_nb[x][c_col_row[x][y]]][z]];
					max = (s_F[threadIdx.x][a][y-1] < s_B[threadIdx.x][b][y+1]) ? s_B[threadIdx.x][b][y+1] : s_F[threadIdx.x][a][y-1] ;
					//if(x==0 && z==2)
					//printf("max=%d min=%d z=%d, b=%d, a=%d\n", max, min, z, b, a);

					min = (min < max) ? min : max;
					//if(x==0 && z==2)
					//printf("min=%d\n", min);

				}
				d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z]=min;
				//if(x==0)
				//printf("BETAmn_[%d][%d][%d]=%d y=%d\n", x, c_col_row[x][y], z, d_beta[(x*c_N*Q)+(Q*c_col_row[x][y])+z], y);

			}
		//}

	}

	__global__ void GPU_VN( unsigned char * d_beta,  unsigned char * d_alpha,  unsigned char * d_alpha_t, unsigned char * d_alpha_t2, unsigned char * d_gamma, int iter){
		unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
		unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;
		unsigned int z=threadIdx.z+blockIdx.z*blockDim.z;
		unsigned short temp=0;

		#if Q==4
		__shared__ unsigned char s_beta[1][6][4];
		__shared__ unsigned char s_alpha_t[1][6][4];
		__shared__ unsigned char s_alpha_t2[1][6][4];
		#elif Q==8
		__shared__ unsigned char s_beta[1][42][8];
		__shared__ unsigned char s_alpha_t[1][42][8];
		__shared__ unsigned char s_alpha_t2[1][42][8];
		#elif Q==16
		__shared__ unsigned char s_beta[3][21][16];
		__shared__ unsigned char s_alpha_t[3][21][16];
		__shared__ unsigned char s_alpha_t2[3][21][16];
		#endif

		if(y<c_N){

			s_beta[x][threadIdx.y][z]=d_beta[(c_row_col[x][y]*c_N*Q)+(y*Q)+z];
			__syncthreads();

			for(unsigned char index=0; index<c_w_col; index++){
				if(index!=x){
					temp=temp+s_beta[index][threadIdx.y][z];
				}
				__syncthreads();
			}

			//if( c_row_col[x][y]==0 && y==8 )
			//printf("temp=%d, GAMMAn_[%d][%d]=%d\n", temp, y,z,d_gamma[(y*Q)+z]);


			//__syncthreads();

			s_alpha_t[x][threadIdx.y][z]= temp+ d_gamma[(y*Q)+z];
			
			s_alpha_t2[x][threadIdx.y][z]=z;
			__syncthreads();

			//if(x==0 && y==0)
			//printf("alpha_t[%d][%d][%d]=%d\n",x,y,z,d_alpha_t[(x*c_N*Q)+(Q*y)+z]);


			
			for(unsigned char stride=blockDim.z/2; stride>0; stride>>=1){
				if(z<stride){
					/* if( c_row_col[x][y]==0 ){
					printf("t=%d min=%d y=%d z=%d\n", d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z]], d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride]],y,z);
					} */
					s_alpha_t2[x][threadIdx.y][z]= (s_alpha_t[x][threadIdx.y][s_alpha_t2[x][threadIdx.y][z]] > s_alpha_t[x][threadIdx.y][s_alpha_t2[x][threadIdx.y][z+stride]]) ? s_alpha_t2[x][threadIdx.y][z+stride] : s_alpha_t2[x][threadIdx.y][z];
					//if( c_row_col[x][y]==0 ){
						//printf("t=%d min=%d y=%d z=%d\n", d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z]], d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride]],y,z);
						//}
					//if(x==0 && y==0 )
					//printf("alpha_t2[%d][%d][%d]=%d\n",x,y,z,d_alpha_t2[(x*c_N*Q)+(Q*y)+z]);

				}
				__syncthreads();
			}

			//__syncthreads();

			
			//if(x==0  && z==0)
			//printf("alpha_t2[%d][%d][%d]=%d\n",x,y,z,d_alpha_t2[(x*c_N*Q)+(Q*y)+z]);

			//if(c_row_col[x][y]==41 )
			//printf("ALPHA_t[%d]=%d ALPHA_t2[%d]=%d\n", z, d_alpha_t[(x*c_N*Q)+(Q*y)+z], d_alpha_t2[(x*c_N*Q)+(Q*y)],  d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)]]);
			
			temp=s_alpha_t[x][threadIdx.y][s_alpha_t2[x][threadIdx.y][0]];
			//__syncthreads();
			//if( c_row_col[x][y]==0 && y==155 && z==1)
			//printf("temp=%d\n", temp);
			d_alpha[(c_row_col[x][y]*c_N*Q)+(y*Q)+z] = s_alpha_t[x][threadIdx.y][z] - temp;
			
			//__syncthreads();
			
			//if( c_row_col[x][y]==0 && y==155 && z==1)
			//printf("alpha[%d][%d][%d]=%d\n",c_row_col[x][y],y,z,d_alpha[(c_row_col[x][y]*c_N*Q)+(y*Q)+z]);

			
			//if(y==0)
			//printf("F[%d][0]=%d, B[%d][%d]=%d, row=%d, lastcol=%d, mult=%d\n", z, d_F[(x*Q*c_N)+z*c_N],z,c_w_row-1, d_B[(x*Q*c_N)+(z*c_N)+c_w_row-1], x, c_col_row[x][c_w_row-1],c_mult[c_inv[c_h_nb[x][c_col_row[x][0]]]][z]);
		} 


			/* for(unsigned char index=0; index<c_w_col; index++){
				if(index!=x){
					temp=temp+d_beta[(c_row_col[index][y]*c_N*Q)+(y*Q)+z];
				}
				__syncthreads();
				//if( c_row_col[x][y]==0 && y==155 && z==1)
				//printf("temp=%d\n",temp);

			}

			
			
			__syncthreads();

			d_alpha_t[(x*c_N*Q)+(Q*y)+z]= temp+ d_gamma[(y*Q)+z];
			__syncthreads();
			//if( c_row_col[x][y]==0 && y==155 && z==1)
			//printf("temp=%d, GAMMAn_[%d][%d]=%d\n", temp, y,z,d_gamma[(y*Q)+z]);

			d_alpha_t2[(x*c_N*Q)+(Q*y)+z]=z;
			__syncthreads();

			//if(x==0 && y==0)
			//printf("alpha_t[%d][%d][%d]=%d\n",x,y,z,d_alpha_t[(x*c_N*Q)+(Q*y)+z]);


			for(unsigned char stride=1; stride<Q; stride*=2){
				if(z%(2*stride)==0){
					/* if( c_row_col[x][y]==0 ){
					printf("t=%d min=%d y=%d z=%d\n", d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z]], d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride]],y,z);
					} */
					/*d_alpha_t2[(x*c_N*Q)+(Q*y)+z]= (d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z]] > d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride]]) ? d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride] : d_alpha_t2[(x*c_N*Q)+(Q*y)+z];
					//if( c_row_col[x][y]==0 ){
						//printf("t=%d min=%d y=%d z=%d\n", d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z]], d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)+z+stride]],y,z);
						//}
					//if(x==0 && y==0 )
					//printf("alpha_t2[%d][%d][%d]=%d\n",x,y,z,d_alpha_t2[(x*c_N*Q)+(Q*y)+z]);

				}
				__syncthreads();
			}

			__syncthreads();
			

			//if(x==0  && z==0)
			//printf("alpha_t2[%d][%d][%d]=%d\n",x,y,z,d_alpha_t2[(x*c_N*Q)+(Q*y)+z]);

			//if(c_row_col[x][y]==41 )
				//printf("ALPHA_t[%d]=%d ALPHA_t2[%d]=%d\n", z, d_alpha_t[(x*c_N*Q)+(Q*y)+z], d_alpha_t2[(x*c_N*Q)+(Q*y)],  d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)]]);
				
			temp=d_alpha_t[(x*c_N*Q)+(Q*y)+d_alpha_t2[(x*c_N*Q)+(Q*y)]];
			__syncthreads();
			//if( c_row_col[x][y]==0 && y==155 && z==1)
			//printf("temp=%d\n", temp);
			d_alpha[(c_row_col[x][y]*c_N*Q)+(y*Q)+z] = d_alpha_t[(x*c_N*Q)+(Q*y)+z] - temp;
			
			__syncthreads();
			
			//if( c_row_col[x][y]==0 && y==155 && z==1)
			//printf("alpha[%d][%d][%d]=%d\n",c_row_col[x][y],y,z,d_alpha[(c_row_col[x][y]*c_N*Q)+(y*Q)+z]);

			
			//if(y==0)
			//printf("F[%d][0]=%d, B[%d][%d]=%d, row=%d, lastcol=%d, mult=%d\n", z, d_F[(x*Q*c_N)+z*c_N],z,c_w_row-1, d_B[(x*Q*c_N)+(z*c_N)+c_w_row-1], x, c_col_row[x][c_w_row-1],c_mult[c_inv[c_h_nb[x][c_col_row[x][0]]]][z]);
		} */

	}
#endif


//===================================
// CUDA Wrapper
//===================================

extern "C" int cuda_minmax(unsigned char* h_beta, unsigned char* ALPHAmn_, unsigned char* GAMMAn_, unsigned char* h_F,unsigned char* h_B,int* iteration, int* decoded_bit){
	
	#if Q==4
		const unsigned char add[4][4] = {{ 0,1,2,3 }, { 1,0,3,2 },{ 2,3,0,1 },{ 3,2,1,0 }};
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

	#endif
	


	for (int row = 0; row < M; row++){
		int col = col_row[row][0];
		for (int index = 0; index < row_weight[row]; index++){
			col = col_row[row][index];
			for (int a = 0; a < Q; a++) {
				ALPHAmn_[(Q*N*row)+(Q*col)+a] = GAMMAn_[(col*Q)+a];

			}
		}
	}


	/* for (int col = 0; col < N; col++){
		for (int a = 0; a < Q; a++) {
			printf("Gamma[%d][%d]=%d\n",col,a,GAMMAn_[(col*Q)+a]);

		}
	} */

	/* for (int row = 0; row < M; row++){
		for (int col = 0; col < N; col++){
			for (int a = 0; a < Q; a++) {
				printf("alpha[%d][%d][%d]=%d\n",row,col,a,ALPHAmn_[(Q*N*row)+(Q*col)+a]);

			}
		}
	} */
	
	#if Q==32

		unsigned char *h_nb=NULL;

		h_nb=(unsigned char *)malloc(sizeof(unsigned char)*M*N);
		if(h_nb == NULL){
			printf("Failed to allocate host h_nb\n" );
			exit(EXIT_FAILURE);
		}

		for (int row = 0; row < M; row++){
			for (int index = 0; index < N; index++){
				h_nb[(row*N)+index] = H_nb[row][index];
				//printf("H_nb[%d][%d]=%d\n", row, index, h_nb[row][index]);
			}
		}

	#else
		unsigned char h_nb[M][N];

		for (int row = 0; row < M; row++){
			for (int index = 0; index < N; index++){
				h_nb[row][index] = H_nb[row][index];
				//printf("H_nb[%d][%d]=%d\n", row, index, h_nb[row][index]);
			}
		}

	#endif

	free(H_nb);

	unsigned char w_row=row_weight[0];
	unsigned char w_col=col_weight[0];


	

	cudaError_t err=cudaSuccess; 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	//======================================================================================================================================================================
	//kernel dimensions

		#if Q==4
			dim3 FB_threadsPerBlock(M,1,Q);
			dim3 FB_numBlocks(3,1,1);

			dim3 CN_threadsPerBlock(M,w_row,Q);
			dim3 CN_numBlocks(1,1,1);

			dim3 VN_threadsPerBlock(w_col,N,Q);
			dim3 VN_numBlocks(1,1,1);
		#elif Q==8
			dim3 FB_threadsPerBlock(M,1,Q);
			dim3 FB_numBlocks(3,1,1);

			dim3 CN_threadsPerBlock(M,w_row,Q);
			dim3 CN_numBlocks(1,1,1);

			dim3 VN_threadsPerBlock(w_col,N,Q);
			dim3 VN_numBlocks(1,1,1);
		#elif Q==16
			dim3 FB_threadsPerBlock(64,1,Q);
			dim3 FB_numBlocks(3,1,1);

			dim3 CN_threadsPerBlock(10,w_row,Q);
			dim3 CN_numBlocks(15,1,1);

			dim3 VN_threadsPerBlock(w_col,21,Q);
			dim3 VN_numBlocks(1,15,1);
		#elif Q==32
			dim3 FB_threadsPerBlock(32,1,Q);
			dim3 FB_numBlocks(10,1,1);

			dim3 CN_threadsPerBlock(5,w_row,Q);
			dim3 CN_numBlocks(62,1,1);

			dim3 VN_threadsPerBlock(w_col,10,Q);
			dim3 VN_numBlocks(1,62,1);
		#endif
  
	//======================================================================================================================================================================
	//size of variables
		size_t size_dimension=sizeof(unsigned short);
		size_t size_col_row=sizeof(unsigned short)*M*w_row;
		size_t size_row_col=sizeof(unsigned short)*w_col*N;
		size_t size_gamma=sizeof(unsigned char)*Q*N;
		size_t size_alpha=sizeof(unsigned char)*M*N*Q;
		size_t size_FB=sizeof(unsigned char)*M*w_row*Q;
		size_t size_GF=sizeof(unsigned char)*Q*Q;
		size_t size_GF_inv=sizeof(unsigned char)*Q;
		size_t size_h_nb=sizeof(unsigned char)*M*N;
		size_t size_weight=sizeof(unsigned char);
		size_t size_alpha_t=sizeof(unsigned char)*N*w_col*Q;
	
	//======================================================================================================================================================================
    //variables declaration
		unsigned char *d_gamma=NULL;
		unsigned char *d_alpha=NULL;
		unsigned char *d_F=NULL;
		unsigned char *d_B=NULL;
		unsigned char *d_beta=NULL;
		unsigned char *d_alpha_t=NULL;
		unsigned char *d_alpha_t2=NULL;
		#if Q==32
			unsigned char *d_h_nb=NULL;
		#endif
				
	//======================================================================================================================================================================
	//allocate host memory

		/* h_F=(unsigned char *)malloc(size_FB);
		if(h_F == NULL){
			printf("Failed to allocate host F\n" );
			exit(EXIT_FAILURE);
		}

		h_B=(unsigned char *)malloc(size_FB);
		if(h_B == NULL){
			printf("Failed to allocate host B\n" );
			exit(EXIT_FAILURE);
		} */

		

	//======================================================================================================================================================================
    //allocate device memory
		err=cudaMalloc((void **)&d_gamma, size_gamma);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device gamma (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_alpha, size_alpha);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device alpha (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_F, size_FB);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device F (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_B, size_FB);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device B (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_beta, size_alpha);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device beta (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_alpha_t, size_alpha_t);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device alpha_t (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMalloc((void **)&d_alpha_t2, size_alpha_t);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to allocate device alpha_t2 (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		#if Q==32
			err=cudaMalloc((void **)&d_h_nb, size_h_nb);
			if(err!=cudaSuccess){
				fprintf(stderr, "Failed to allocate device h_nb (error code %d)!\n", cudaGetLastError());
				exit(EXIT_FAILURE);
			}
		#endif

	//======================================================================================================================================================================
    //device memory initialization
		err=cudaMemset(d_gamma, 0, size_gamma);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device gamma (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_alpha, 0, size_alpha);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device alpha (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_F, 0, size_FB);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device F (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_B, 0, size_FB);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device B (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemset(d_beta, 0, size_alpha);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to initialize device beta (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

				
	//======================================================================================================================================================================
	//copy data to constant


		err=cudaMemcpyToSymbol(c_col_row, &h_col_row, size_col_row);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy col_row from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_row_col, &h_row_col, size_row_col);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy row_col from host to constant (error code %d)!\n", cudaGetLastError());
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

		err=cudaMemcpyToSymbol(c_add, &add, size_GF,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy add from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpyToSymbol(c_mult, &mult, size_GF,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy mult from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		#if Q!=32
			err=cudaMemcpyToSymbol(c_h_nb, &h_nb, size_h_nb,0,cudaMemcpyHostToDevice);
			if(err!=cudaSuccess){
				fprintf(stderr, "Failed to copy H from host to constant (error code %d)!\n", cudaGetLastError());
				exit(EXIT_FAILURE);
			}
		#endif

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

		err=cudaMemcpyToSymbol(c_inv, &inv, size_GF_inv,0,cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy w_row from host to constant (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

	//======================================================================================================================================================================
	//copy data to device  
	
		err=cudaMemcpy(d_gamma, GAMMAn_, size_gamma, cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy gamma from host to device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(d_alpha, ALPHAmn_, size_alpha, cudaMemcpyHostToDevice);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy alpha from host to device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		#if Q==32
			err=cudaMemcpy(d_h_nb, h_nb, size_h_nb, cudaMemcpyHostToDevice);
			if(err!=cudaSuccess){
				fprintf(stderr, "Failed to copy alpha from host to device (error code %d)!\n", cudaGetLastError());
				exit(EXIT_FAILURE);
			}
		#endif

	//======================================================================================================================================================================
	//execute the kernel
		cudaEventRecord(start);
		for (int iter=0; iter < MAX_ITERATION; iter++) {
			#if Q==32
				GPU_FB_metrics<<<FB_numBlocks, FB_threadsPerBlock>>>( d_alpha, d_F, d_B, iter, d_h_nb);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
				}

				GPU_CN<<<CN_numBlocks, CN_threadsPerBlock>>>( d_beta, d_F, d_B, iter, d_h_nb);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
				} 

				GPU_VN<<<VN_numBlocks, VN_threadsPerBlock>>>( d_beta, d_alpha, d_alpha_t, d_alpha_t2, d_gamma, iter);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
				} 

			#else
				GPU_FB_metrics<<<FB_numBlocks, FB_threadsPerBlock>>>( d_alpha, d_F, d_B, iter);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
				}

				GPU_CN<<<CN_numBlocks, CN_threadsPerBlock>>>( d_beta, d_F, d_B, iter);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
				} 

				GPU_VN<<<VN_numBlocks, VN_threadsPerBlock>>>( d_beta, d_alpha, d_alpha_t, d_alpha_t2, d_gamma, iter);
				if(err!=cudaSuccess){
					fprintf(stderr, "Failed to launch the kernel (error code %d)!\n", cudaGetLastError());
					exit(EXIT_FAILURE);
			} 
			#endif 
		}
		cudaEventRecord(stop);


		

	//======================================================================================================================================================================
	//copy the data from device to host

	 	err=cudaMemcpy(h_F, d_F, size_FB, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the F from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(h_B, d_B, size_FB, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the F from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(h_beta, d_beta, size_alpha, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the betta from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaMemcpy(h_alpha, d_alpha, size_alpha, cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to copy the betta from device to host (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		cudaEventSynchronize(stop);

	//======================================================================================================================================================================
	//free the device memory
		err=cudaFree(d_gamma);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the gamma from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_alpha);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the alpha from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_F);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the F from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_B);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the B from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_beta);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the beta from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_alpha_t);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the alpha_t from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		err=cudaFree(d_alpha_t2);
		if(err!=cudaSuccess){
			fprintf(stderr, "Failed to free the alpha_t2 from the device (error code %d)!\n", cudaGetLastError());
			exit(EXIT_FAILURE);
		}

		#if Q==32
			err=cudaFree(d_h_nb);
			if(err!=cudaSuccess){
				fprintf(stderr, "Failed to free the H-nb from the device (error code %d)!\n", cudaGetLastError());
				exit(EXIT_FAILURE);
			}
		#endif

	//======================================================================================================================================================================
	//save data
		
		

		/* for (int row = 0; row < M; row++) {
			for (int index_c = 0; index_c < w_row; index_c++) {	//2. update BETA values from B and F values
				for (int a = 0; a < Q; a++) {
					//printf("F[%d][%d][%d]=%d\n",row, a,index_c,h_F[(row*Q*w_row)+(a*w_row)+index_c]);
					printf("B[%d][%d][%d]=%d\n",row, a,w_row - index_c -1,h_B[(row*Q*w_row)+(a*w_row)+w_row - index_c -1]);
				}
			}
		} */

	//======================================================================================================================================================================
	//free the host memory
		
	cudaEventElapsedTime(&milliseconds, start, stop);

	#if Q==32
		free(h_nb);
	#endif

			
	return 0;
}
