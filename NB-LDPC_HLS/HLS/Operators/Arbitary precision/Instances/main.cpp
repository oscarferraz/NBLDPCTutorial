
#include <ap_int.h>
#include"header.h"
#include <cmath>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include"header.h"
#include"CSR4.h"
using namespace std;

ap_uint<3> logq[4] = { 0,0,1,2 };
ap_uint<3> expq[3] = { 1,2,3 };
ap_uint<6>B[Q][Y];
ap_uint<6>F[Q][Y];
ap_uint<5> codeword_sym[V];
ap_uint<5>GAMMAn_post[Q][V];
ap_uint<6>ALPHAmn_1[nnnz][Q];
ap_uint<5>Gamalocal[Q][V];
ap_uint<5>BETAmn_1[nnnz][Q];
ap_uint<6>ALPHA_t[Q];
ap_uint<3> mul_gf( ap_uint<3> a, ap_uint<3>b)
{


		if (a == 0 || b == 0) return 0;
		if (a == 1) return b;
		if (b == 1) return a;
		if ((logq[a] + logq[b])>=(Q-1))
				{
				return	expq[(logq[a] + logq[b])- (Q-1)];
				}
				else
				return expq[(logq[a] + logq[b])];

}
ap_uint<3> inv_gf(ap_uint<3> a)
{


	if (a == 0) return 0;
	if (a == 1) return 1;
	return expq[(Q - 1 - logq[a])];

}
ap_uint<5> IN()
{
	int i,row,j,col,error;
	int e;
  ap_uint<3>a;
  ap_uint<3>b;

	for (i = 0; i < Q; i++) {
		#pragma HLS unroll
			for (j = 0; j <Y; j++) {
		#pragma HLS unroll
				F[i][j] = B[i][j] = 0;
			}
		}



		for (i = 0; i < Q; i++) {
				#pragma HLS unroll
			for (j = 0; j < V; j++) {
				#pragma HLS unroll
				GAMMAn_post[i][j] = 0;
			}
		}


		for (row = 0; row < nnnz; row++) {// initialize variable node message ALPHA with channel info GAMMA
				#pragma HLS unroll
			for (a = 0; a < Q; a++) {
				#pragma HLS unroll
				ALPHAmn_1[row][a] = Gamalocal[a][col_ind[row]];
				if(ALPHAmn_1[row][a]!=0)	{
		//	std::cout<<"ALPHAMN"<<"["<<row<<"]"<<"["<<a<<"]" <<"=" <<ALPHAmn_1[row][a]<<"\n";
				}
			}
		}
		for (col = 0,a=0; col < V; col++) {	// tentative decoding
			#pragma HLS pipeline II=3
			//a = 0;
			for (i = 0; i < Q; i++) {
				if (Gamalocal[i][col] < Gamalocal[a][col]) {
					a = i;
				}

			}
			codeword_sym[col] = a;

		}
		for (i = 0; i < (V - U); i++) {
				if (encoded_sym[i + U] != codeword_sym[i + U])
					error++;
		}
return 0;
}

ap_uint<5> CN()
{
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=ALPHAmn_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=BETAmn_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=F complete dim=1

	ap_uint<9>index_r;
	ap_uint<3>index,e,d;
	ap_uint<3>c;
	ap_uint<5>index_p,index_B,min_value,max_value;
	ap_uint<5> min_F,max_F,min_B,max_B;
	ap_uint<3>a;
	ap_uint<3>b;

	for (index_r =0; index_r < U; index_r++) {
	#pragma HLS pipeline
				loop1:for (a= 0; a< 4; a++) {
								F[a][0] = ALPHAmn_1[index_r*3][mul_gf(inv_gf(val[index_r*3]), a)];
								B[a][row_weight[index_r] - 1] = ALPHAmn_1[row_ptr[index_r + 1] - 1][mul_gf(inv_gf(val[row_ptr[index_r + 1] - 1]), a)];

						}
						index_p = ((index_r*3)+1) - (index_r*3) - 1;
							for (c = 0; c < Q; c++) {
								min_F = max(F[c][index_p], ALPHAmn_1[((index_r*3)+1)][0]);
								for (b= 1;b< Q; b++) {
									a = sub_gf(c,mul_gf(val[((index_r*3)+1)], b));
									max_F = max(F[a][index_p], ALPHAmn_1[((index_r*3)+1)][b]);
									min_F = min(min_F, max_F);
								}
								F[c][((index_r*3)+1) - (index_r*3)] = min_F;

							}

							index_B = (index_r*3)+3 - ((index_r*3)+1)-1;

							for (c = 0; c < Q; c++) {
	#pragma HLS loop_flatten off
								min_B = max(ALPHAmn_1[row_ptr[index_r ] + index_B][0], B[c][index_B + 1]);
								for (b= 0; b < Q; b++) {
									a = sub_gf(c, mul_gf(val[row_ptr[index_r] + index_B ], b));
									max_B = max(B[a][index_B + 1], ALPHAmn_1[row_ptr[index_r ] + index_B][b]);
									min_B = min(min_B, max_B);
								}
								B[c][index_B] = min_B;

							}


	vn_2:for (a= 0; a < 4; a++) {

				BETAmn_1[index_r*3][a] = B[mul_gf(val[index_r*3], a)][1];
				BETAmn_1[(index_r*3)+2][a] = F[mul_gf(val[(index_r*3)+2], a)][(index_r*3+2)-row_ptr[index_r]-1];

									}

	for (c = 0; c < Q; c++) {
										min_value = max(F[mul_gf(val[((index_r*3)+1)], c)][((index_r*3)+1)-row_ptr[index_r]-1], B[0][((index_r*3)+1)-row_ptr[index_r]+1]);
											for (b = 0; b < Q; b++) {
														#pragma HLS pipeline
															a = add_gf(b, mul_gf(val[((index_r*3)+1)], c));
															max_value = max(F[a][((index_r*3)+1)-row_ptr[index_r] - 1], B[b][((index_r*3)+1)-row_ptr[index_r] + 1]);
															min_value = min(max_value, min_value);
																			}
															BETAmn_1[(index_r*3)+1][c] = min_value;


								}}

	return 0;

		}

ap_uint<5> VN()
			{
//#pragma HLS ARRAY_PARTITION variable=ALPHA_t complete dim=1
#pragma HLS ARRAY_PARTITION variable=BETAmn_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=ALPHAmn_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=Gamalocal complete dim=1
		ap_uint<9>col_v;
		ap_uint<3>index_v,a;
		ap_uint<4>index_,min_index;
		ap_uint<6>temp;
		for(col_v = 0; col_v < V; col_v ++){
				#pragma HLS pipeline II=3
				for(index_v=0;index_v< 2;index_v++){
					//#pragma HLS unroll
					for(a = 0,min_index=0; a < Q; a++){
						temp = 0;
						//#pragma HLS pipeline
						for(index_=0;index_< 2;index_++){
							if(index_ != index_v){
								temp = temp + BETAmn_1[ptr_to_val[col_ptr[col_v]+index_]][a];
							}
						}

						ALPHA_t[a] = temp + Gamalocal[a][col_v];


					}


					for(a = 1; a < Q; a++){
	#pragma HLS unroll
						if(ALPHA_t[a] < ALPHA_t[min_index]){
							min_index = a;
						}
					}

					minmax_label0:for(a = 0; a < Q; a++){
	#pragma HLS unroll
						ALPHAmn_1[ptr_to_val[col_ptr[col_v]+index_v]][a] = ALPHA_t[a] - ALPHA_t[min_index];

					}


				}
			}
	return 0;
			}









ap_uint<5> minmax(ap_uint<5>GAMMAn[Q][V])
{
#pragma HLS INTERFACE s_axilite depth=11250 port=GAMMAn
#pragma HLS INLINE
int d,error;
ap_uint<9>j;
ap_uint<4>i;
ap_uint<4> iter,e,f;



    		for(int i=0;i<4;i++)
    		{
    			#pragma HLS unroll
    			for (int j=0;j<384;j++)
    			{
				#pragma HLS unroll

    			 Gamalocal[i][j]=GAMMAn[i][j];
    			}
    		}


#ifdef COSIM
IN();
#endif

for(int i=0;i<10;i++)
{
#pragma HLS unroll
CN();
VN();
}


		#ifdef COSIM
		for (int it=0;it<Q;it++)
		{
			std::cout<<"ALphat"<<"["<<it<<"]"<<"="<<ALPHA_t[it]<<"\n";


		}
		#endif


#ifdef COSIM
		for (int i=0;i<nnnz;i++)
		{
			for(int j=0;j<Q;j++)
			{
				if(ALPHAmn_1[i][j]!=0)
				{
		std::cout<<"ALPHAMN"<<"["<<i<<"]"<<"["<<j<<"]"<<"="<<ALPHAmn_1[i][j]<<"\n";

			}
	}}
#endif

	#ifdef COSIM
				for (int i=0;i<nnnz;i++)
								{
									for(int j=0;j<Q;j++)
									{
										if(BETAmn_1[i][j]!=0)
										{
											std::cout<<"BETAMN"<<"["<<i<<"]"<<"["<<j<<"]"<<"="<<BETAmn_1[i][j]<<"\n";

									}
									//std::cout<<"\n";
							}}
			#endif






return error;
}


