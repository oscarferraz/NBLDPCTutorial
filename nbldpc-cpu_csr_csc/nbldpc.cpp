
#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include <time.h>
#include <string.h>
//#include <ap_fixed.h>
#include "nbldpc.h"

#ifdef VS
#define _CRT_SECURE_NO_DEPRECATE
#endif

//===================================
// Allocate 2D array of double
//===================================
double **malloc2Ddouble(int arraySizeX, int arraySizeY) // allocates array[a][b]
{
	double** theArray;
	theArray = (double**)malloc(arraySizeX * sizeof(double*));
	for (int i = 0; i < arraySizeX; i++)
		theArray[i] = (double*)malloc(arraySizeY * sizeof(double));
	return theArray;
}

//===================================
// Allocate 2D array of unsigned char
//===================================
unsigned char **malloc2Dunsigned_char(int arraySizeX, int arraySizeY) // allocates array[a][b]
{
	unsigned char** theArray;
	theArray = (unsigned char**)malloc(arraySizeX * sizeof(unsigned char*));
	for (int i = 0; i < arraySizeX; i++)
		theArray[i] = (unsigned char*)malloc(arraySizeY * sizeof(unsigned char));
	return theArray;
}

//===================================
// Galios Field Multiplication
//===================================
unsigned char mul_gf(unsigned char a, unsigned char b)
{
	if (a == 0 || b == 0) return 0;
	if (a == 1) return b;
	if (b == 1) return a;
	return expq[(logq[a] + logq[b]) % (Q - 1)];	//table look-up multiplication
	//return expq[(logq[a] + logq[b])];	//table look-up multiplication
}

//===================================
// Galios Field Inverse
//===================================
unsigned char inv_gf(unsigned char a)
{
	//if(a == 0) exit (EXIT_FAILURE);
	if (a == 0) return 0;
	if (a == 1) return 1;
	return expq[(Q - 1 - logq[a])];
	//return expq[(Q - 1 - logq[a]) % (Q-1)];

}


//===================================
// Minmax nonbinary decode function
//===================================
// Refer to "Min-max decoding for nonbinary LDPC codes"
unsigned short minmax(unsigned char** BETAmn_, unsigned char** ALPHAmn_, unsigned char** GAMMAn_, int* iteration)
{
	#ifdef DEBUG
		int  row_v, row_, last_col, syndrome;
	#endif
	int row,col, index, index_v, col_v , error;
	int index_ ;
	int index_B;
	int index_p;
	int i, j;
	int a, b, c;
	int iter = 0;
	int min_index;
	unsigned char **B;
	unsigned char **F;
	unsigned char min_F, min_B;
	unsigned char max_F, max_B;
	int temp;
	unsigned char min_value, max_value;
	unsigned char* ALPHA_t;

	int* codeword_sym;



	#ifdef DEBUG
		FILE *fp = fopen("./data/codeword(enc_dec).txt", "w");
		FILE *fp1 = fopen("./data/alpha.txt", "w");
		FILE *fp2 = fopen("./data/gamma_post.txt", "w");
		FILE *fp3 = fopen("./data/gamma.txt", "w");
		FILE *fp4 = fopen("./data/beta.txt", "w");
		FILE *fp5 = fopen("./data/F.txt", "w");
		FILE *fp6 = fopen("./data/B.txt", "w");
		FILE *fp7 = fopen("./data/alpha_t.txt", "w");
		//FILE *fp8 = fopen("./data/beta_t.txt","w");
		FILE *fp_beta_F = fopen("./data/beta_F.txt", "w");
		FILE *fp_beta_B = fopen("./data/beta_B.txt", "w");
		FILE *fp_validation = fopen("./data/validation.txt", "w");

	#endif
	printf("CHECK\n");


	ALPHA_t = (unsigned char *)malloc(sizeof(unsigned char) * Q);
	codeword_sym = (int *)malloc(sizeof(int) * N);
	GAMMAn_post = malloc2Dunsigned_char(Q, N);

	printf("CHECK\n");


	B = (unsigned char **)malloc2Dunsigned_char(Q, dv);
	F = (unsigned char **)malloc2Dunsigned_char(Q, dv);

	printf("CHECK\n");


	#ifdef DEBUG
		for (col = 0; col < N; col++) {
			for (a = 0; a < Q; a++) {
				//printf("GAMMAn_[%d][%d]=%d\n", a, col, GAMMAn_[a][col]);
				fprintf(fp3, "%d ", GAMMAn_[a][col]);
				
			}
			fprintf(fp3, "\n");
		}
	#endif

	
	printf("CHECK\n");

	for (i = 0; i < Q; i++) {
		for (j = 0; j < dv; j++) {
			F[i][j] = B[i][j] = 0;
		}
	}

	printf("CHECK\n");

	for (i = 0; i < Q; i++) {
		for (j = 0; j < N; j++) {
			GAMMAn_post[i][j] = 0;
		}
	}

	printf("CHECK\n");

	/* for(row = 0; row < M; row++)// initialize variable node message ALPHA with channel info GAMMA
	{
		col = col_row[row][0];
		for(index = 0; index < row_weight[row]; index++)
		{
			col = col_row[row][index];
			for(a = 0; a < Q; a++){
				ALPHAmn_[row][a] = GAMMAn_[a][col];

			}
			printf("row=%d, col=%d\n", row, col);
		}
	}  */ 

	//printf("\n\n");
	#ifdef DEBUG
		FILE *fp_alpha_val = fopen("./data/alpha_val.txt", "w");
	#endif


	for (index = 0; index < nnz; index++) {// initialize variable node message ALPHA with channel info GAMMA
		for (a = 0; a < Q; a++) {
			ALPHAmn_[index][a] = GAMMAn_[a][col_ind[index]];
			//printf("Alpha[%d][%d]=%d, col=%d\n", index,a,ALPHAmn_[index][a],col_ind[index]);
			/*#ifdef DEBUG
				fprintf(fp_alpha_val, "ALPHAmn_[%d][%d]=%d\n",index, a,ALPHAmn_[index][a]);
			#endif*/
			if(ALPHAmn_[index][a]!=0)
				fprintf(fp_alpha_val,"ALPHAMN[%d][%d]=%hhu\n",index, a,ALPHAmn_[index][a]);

		}
		//printf("row=%d, col=%d\n", 0, col_ind[index]);
	}

	printf("CHECK\n");

	#ifdef DEBUG
		fclose(fp_alpha_val);
	#endif


	#ifdef PRINT_A_B
		for (row = 0; row < M; row++) {
			for (col = 0; col < N; col++) {
				for (a = 0; a < Q; a++) {
					fprintf(fp1, "%hhu ", ALPHAmn_[row][col][a]);
				}
				fprintf(fp1, "\n");
			}
		}
	#endif

	#ifdef DEBUG
		for (col = 0; col < N; col++) {
			fprintf(fp, "encoded_sym[%3d]=%3d\n", col, encoded_sym[col]);
		}
		fprintf(fp, "\n");
	#endif

	for (col = 0; col < N; col++) {	// tentative decoding
		a = 0;
		for (i = 0; i < Q; i++) {
			#ifdef DEBUG
						//fprintf(fp3, "%hhu ", GAMMAn_[i][col]);
			#endif
			if (GAMMAn_[i][col] < GAMMAn_[a][col]) {
				a = i;
			}

		}
		codeword_sym[col] = a;
		#ifdef DEBUG
				fprintf(fp, "%3d ", codeword_sym[col]);
				//fprintf(fp3, "\n");
		#endif
	}

	printf("CHECK\n");

	error = 0;

	for (i = 0; i < (N - M); i++) {
		if (encoded_sym[i + M] != codeword_sym[i + M])
			error++;
	}

	printf("CHECK\n");

	#ifdef DEBUG
	#else
		if (error == 0) {
			printf("This received codeword has no error.\n");
			for (i = 0; i < Q; i++) {
				free(B[i]);
			}
			free(B);

			for (i = 0; i < Q; i++) {
				free(F[i]);
			}
			free(F);

			free(ALPHA_t);
			free(codeword_sym);

			for (i = 0; i < Q; i++) {
				free(GAMMAn_post[i]);
			}
			free(GAMMAn_post);
			return 0;
		}
	#endif

	printf("CHECK\n");

	#ifdef DEBUG
		printf("error iss %d\n", error);

		/*for(row = 0; row < M; row ++){
			syndrome = 0;
			for(col = 0; col < N; col ++){
				syndrome = add_gf(syndrome,mul_gf(H_nb[row][col],codeword_sym[col]));
			}
			printf("row = %d syndrome = %d\n",row, syndrome);
		}  */
		

		FILE *fp_a_val = fopen("./data/a_val.txt", "w");
		FILE *fp_B_val = fopen("./data/B_val.txt", "w");
		FILE *fp_F_val = fopen("./data/F_val.txt", "w");
		FILE *fp_beta_val = fopen("./data/beta_val.txt", "w");
		FILE *fp_alpha_t_val = fopen("./data/alpha_t_val.txt", "w");
		
	#endif

	FILE *fp_alphal_val = fopen("./data/alphal_val.txt", "w");

	clock_gettime(CLOCK_MONOTONIC, &start);

	while (iter < MAX_ITERATION) {	// iteration starts
		for (int index_r = 0; index_r < M; index_r++) {// check node processing: update check node message BETA
			for (int index = row_ptr[index_r]; index < row_ptr[index_r + 1]; index++) {// check node processing: update check node message BETA
				if (index == row_ptr[index_r]) {
					//printf("row=%d, col=%d, last_col=%d\n", index_r, col_ind[index],col_ind[row_ptr[index_r+1]-1]);
					for (a = 0; a < Q; a++) {
						F[a][0] = ALPHAmn_[index][mul_gf(inv_gf(val[index]), a)];
						//if(index_r==0)
						//printf("F[%d][0]=%d, alpha[%d][%d][%d]=%d\n", a, F[a][0],index_r, index,mul_gf(inv_gf(val[index]), a),ALPHAmn_[index][mul_gf(inv_gf(val[index]), a)]);
						B[a][row_weight[index_r] - 1] = ALPHAmn_[row_ptr[index_r + 1] - 1][mul_gf(inv_gf(val[row_ptr[index_r + 1] - 1]), a)];
						//printf("B[%d][%d]=%d, row=%d, last_col=%d, mult=%d\n",a,row_weight[index_r]-1, B[a][row_weight[index_r]-1],index_r, row_ptr[index_r+1]-1,mul_gf(inv_gf(val[row_ptr[index_r+1]-1]),a));
					}
				}
				else {
					index_p = index - row_ptr[index_r] - 1;

					//printf("index_p=%d\n", index_p);

					for (c = 0; c < Q; c++) {
						min_F = max(F[c][index_p], ALPHAmn_[index][0]);
						//if(index_r==1 )
						//printf("min_F=%d, F[%d][%d]=%d, ,ALPHAmn_[%d][0]=%d\n", min_F,c,index_p,F[c][index_p],index,ALPHAmn_[index][0]);
						for (b = 1; b < Q; b++) {
							a = sub_gf(c, mul_gf(val[index], b));
							//if(index_r==2 )
							//printf("a=%d, add[%d][%d]=%d\n",a, c,mul_gf(val[index], b),sub_gf(c, mul_gf(val[index], b)));	
							max_F = max(F[a][index_p], ALPHAmn_[index][b]);
							//if(index_r==2 )
							//printf("max_F=%d, F[%d][%d][%d]=%d, ,ALPHAmn_[%d][%d]=%d, c=%d\n", max_F,index_r,a,index_p,F[a][index_p],index,b,ALPHAmn_[index][b], c);	
							min_F = min(min_F, max_F);
						}
						F[c][index - row_ptr[index_r]] = min_F;
						//if(index_r==0 && c==6)
						//printf("F[%d][%d]=%d\n", c,index-row_ptr[index_r ],F[c][index-row_ptr[index_r]]);

					}
					//printf("check\n");

					index_B = row_ptr[index_r+1] - index-1;
					//if(index == row_ptr[0]+1)
					//printf("index_B=%d, row_weight[%d]=%d, index=%d, row_ptr=%d\n==============================================================================\n", index_B,index_r,row_weight[index_r-1], index, row_ptr[index_r-1]);
					for (c = 0; c < Q; c++) {
						min_B = max(ALPHAmn_[row_ptr[index_r ] + index_B][0], B[c][index_B + 1]);
						//if(index_r == 1)
						//printf("min_B=%d, B[%d][%d]=%d, alpha[%d][%d][%d]=%d %d+%d-%d-1\n", min_B,index_B + 1, c,B[c][index_B + 1],index_r, row_ptr[index_r ] + index_B, 0, ALPHAmn_[row_ptr[index_r ] + index_B][0], row_ptr[index_r ], row_ptr[index_r +1], index);
						for (b = 0; b < Q; b++) {
							a = sub_gf(c, mul_gf(val[row_ptr[index_r] + index_B ], b));
							//printf("a=%d\n", a);
							max_B = max(B[a][index_B + 1], ALPHAmn_[row_ptr[index_r ] + index_B][b]);
							//fprintf(fp_a_val,"%d ", max_B);
							//printf("max_B=%d\n", max_B);
							//if(index == row_ptr[0]+1)
							//printf("B[%d][%d]=%d, ALPHAmn_[%d][%d]=%d\n", a, index_B + 1,B[a][index_B + 1] , row_ptr[index_r ] + index_B,b,ALPHAmn_[row_ptr[index_r ] + index_B ][b] );
							//printf("a=%d, index_b+1=%d, inde2=%d, b=%d, h=%d\n", a, index_B + 1,row_ptr[index_r] + index_B , b,val[row_ptr[index_r] + index_B ] );
							min_B = min(min_B, max_B);
							//printf("min_B=%d\n", min_B);
						}
						B[c][index_B] = min_B;
						//if(index_r==0 && c==6)
						//printf("last min_B=%d\n", min_B);
						//printf("B[%d][%d]=%d\n", c,index_B,B[c][index_B]);

					}
				}
			}

			//break;

			

			
			

			//printf("check=%d\n",index_r);	
			for (int index = row_ptr[index_r]; index < row_ptr[index_r + 1]; index++) {//2. update BETA values from B and F values				/*if(index_c == 0){
				if (index == row_ptr[index_r]) {
					for (a = 0; a < Q; a++) {
						BETAmn_[index][a] = B[mul_gf(val[index], a)][1];
						//printf("BETAmn_[%d][%d]=%d\n", index, a, BETAmn_[index][a]);
					}
				}
				else if (index == row_ptr[index_r+1]-1) {
					for (a = 0; a < Q; a++) {
						BETAmn_[index][a] = F[mul_gf(val[index], a)][index-row_ptr[index_r]-1];
						//if(index_r==0)
						//printf("mul_gf[%d][%d]=%d\n",val[index], a,mul_gf(val[index], a) );
						//printf("BETAmn_[%d][%d] = F[mul_gf(val[%d], %d)][%d]=%d\n", index, a,index,a, index-row_ptr[index_r]-1,BETAmn_[index][a]);
					}
				}
				else{
					for (c = 0; c < Q; c++) {
						min_value = max(F[mul_gf(val[index], c)][index-row_ptr[index_r]-1], B[0][index-row_ptr[index_r]+1]);
						for (b = 0; b < Q; b++) {
							a = add_gf(b, mul_gf(val[index], c));
							max_value = max(F[a][index-row_ptr[index_r] - 1], B[b][index-row_ptr[index_r] + 1]);
							min_value = min(max_value, min_value);
						}
					BETAmn_[index][c] = min_value;
					//printf("BETAmn_[%d][%d]=%d\n", index, a, BETAmn_[index][c]);
					}
				}
			}
		
			for (a = 0; a < Q; a++) {
				for (index = 0; index < row_weight[0]; index++) {
				 	//printf("B=%d\n", B[a][index]);
					if(index_r==0)
					printf("F[%d][%d]=%d\n", a, index,  F[a][index]);
					fprintf(fp_B_val,"B[%d][%d]=%d ",a,index, B[a][index]);
					fprintf(fp_F_val,"F[%d][%d]=%d ", a,index,F[a][index]);
				}
				fprintf(fp_B_val,"\n");
				fprintf(fp_F_val,"\n");
			}
			fprintf(fp_B_val,"\n\n");
			fprintf(fp_F_val,"\n\n"); 

		}


		int e=0;
		for (index = 0; index < nnz; index++) {
			for (a = 0; a < Q; a++) {
				if(BETAmn_[index][a]!=0)
				fprintf(fp_beta_val,"BETAMN[%d][%d]=%hhu\n ", index, a, BETAmn_[index][a]);
			}
			//fprintf(fp_beta_val,"\n");
			if(row_ptr[e+1]-1==index){
				e++;
				fprintf(fp_beta_val,"\n");
			}
		} 

			
	
		//printf("check node processing finished\n");

		for(col_v = 0; col_v < N; col_v ++){	// variable node processing: update variable node message ALPHA
			index_v = 0;
			//row_v = row_ind[col_ptr[col_v]];
			while(index_v < col_weight[col_v]){
				//row_v = row_ind[col_ptr[col_v]+index_v];
				//printf("row_v = %d\n", row_v);
				for(a = 0; a < Q; a++){
					temp = 0;
					index_ = 0;
					int index2 = 0;
					//row_ = row_ind[col_ptr[col_v]];
					while(index_ < col_weight[col_v]){
						//row_ = row_ind[col_ptr[col_v]+index_];
						//printf("row_ = %d\n", row_);
						if(index_ != index_v){
							temp = temp + BETAmn_[ptr_to_val[col_ptr[col_v]+index_]][a];
							//if(ptr_to_val[col_ptr[col_v]+index_]==42 && col_v==0 &&a==0)
							//printf("BETAmn_[%d][%d]%d = %hhu temp = %hhu\n", ptr_to_val[col_ptr[col_v]+index_], a, index_,BETAmn_[ptr_to_val[col_ptr[col_v]+index_]][a], temp);
							index2++;
						} 
						
						index_++;
					}
					//printf("temp=%d, GAMMAn_[a][col_v]=%d\n", temp, GAMMAn_[a][col_v]);
					ALPHA_t[a] = temp + GAMMAn_[a][col_v];
					//if(ptr_to_val[col_v*col_weight[col_v]+index_v]==2)
					printf("ALPHA_t[%d]=%d, temp=%d, GAMMAn_[%d][%d]=%d, x=%d\n",a, ALPHA_t[a],temp, a,col_v,GAMMAn_[a][col_v], ptr_to_val[col_ptr[col_v]+index_v]); 

				}

				/* #ifdef DEBUG
					for (a = 0; a < Q; a++) {
						fprintf(fp_alpha_t_val,"%d\n", ALPHA_t[a]);
					}
				#endif */

				min_index = 0;
				for(a = 1; a < Q; a++){
					//printf("col=%d, t= %hhu, min=%d\n",col_v, ALPHA_t[a],ALPHA_t[min_index]);
					if(ALPHA_t[a] < ALPHA_t[min_index]){
						min_index = a;
					}
				}	
				//printf("========================================\n" );
				//if(ptr_to_val[col_ptr[col_v]+index_v]==44 &&col_v==141)
				//printf("min= %hhu\n", min_index);

				for(a = 0; a < Q; a++){
					ALPHAmn_[ptr_to_val[col_ptr[col_v]+index_v]][a] = ALPHA_t[a] - ALPHA_t[min_index];
					/*if(ptr_to_val[col_v*col_weight[col_v]+index_v]==2)
					printf("ALPHAmn_[%d][%d]=%d alpha_t[%d]=%d-ALPHA_t[%d]=%d\n", ptr_to_val[col_v*col_weight[col_v]+index_v], a,ALPHAmn_[ptr_to_val[col_v*col_weight[col_v]+index_v]][a],a,ALPHA_t[a],min_index,ALPHA_t[min_index]);*/
 				//fprintf(fp_alphal_val, "%hhu ", ALPHAmn_[ptr_to_val[col_ptr[col_v]+index_v]][a]);
				}

				index_v++;
			}
		} 
		//printf("variable node processing finished\n");
		/* for (int row = 0; row < nnz; row++) {
			for (int a = 0; a < Q; a++) {
				
			}
			fprintf(fp_alphal_val, "%hhu\n"ALPHAmn_[ptr_to_val[col_ptr[col_v]+index_v]][a]);
		} */



		/* for(col = 0; col < N; col ++){	// post processing
			for( a = 0; a < Q; a++){
				temp = 0;
				index_ = 0;
				//row_ = row_ind[col_ptr[col]];
				while(index_ < col_weight[col]){
					//row_ = row_ind[col_ptr[col]+index_];
					temp = temp +BETAmn_[ptr_to_val[col_ptr[col]+index_]][a];
					//printf("BETAmn_[%d][%d]%d = %hhu temp = %hhu\n", ptr_to_val[col_ptr[col]+index_], a, index_,BETAmn_[ptr_to_val[col_ptr[col]+index_]][a], temp);
					index_++;
				}
				GAMMAn_post[a][col] = temp + GAMMAn[a][col];

				
			//printf("GAMMAn[%d][%d] = %d GAMMAn_post[%d][%d] = %d\n", a, col, GAMMAn_post[a][col], a, col, GAMMAn_[a][col]);
				
			}
		}
		//printf("post processing finished\n");
		for(col = 0; col < N; col ++){	// tentative decoding
			a = 0;
			for(i = 0; i < Q; i++){

				#ifdef DEBUG
								fprintf(fp2,"GAMMAn_post[%3d][%3d] = %hhu\n", i, col, GAMMAn_post[i][col] );
								fprintf(fp_validation,"%hhu\n", GAMMAn_post[i][col] );
				#endif

				if(GAMMAn_post[i][col] < GAMMAn_post[a][col]){
					a = i;
				}
			}
			codeword_sym[col] = a;

			#ifdef DEBUG
				fprintf(fp,"codeword_sym[%3d] = %3d\n", col, codeword_sym[col]);
				fprintf(fp2, "\n");
			#endif

		}

		error = 0;

		for(i = 0;i < (N - M);i ++){
			if(encoded_sym[i + M] != codeword_sym[i + M])
				error++;
		}


		#if  (EARLY_STOP == 1)
			if(error == 0){
				for(i = 0; i < Q; i++){
					free(B[i]);
				}
				free(B);

				for(i = 0; i < Q; i++){
					free(F[i]);
				}
				free(F);

				free(ALPHA_t);
				free(codeword_sym);

				for(i = 0; i < Q; i++){
					free(GAMMAn_post[i]);
				}
				free(GAMMAn_post);
				return 0;
			}
		#endif

		#ifdef DEBUG
			fprintf(fp,"error = %d ",error);
			fprintf(fp, "\n");
		#endif

		#ifdef DEBUG
			for(row = 0; row < M; row ++){
				syndrome = 0;
				for(col = 0; col < N; col ++){
					syndrome = add_gf(syndrome,mul_gf(H_nb[row][col],codeword_sym[col]));
				}
				//printf("row = %d syndrome = %d\n",row, syndrome);
			}
		#endif */
		iter++;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	int cnt=0;
	for (int row = 0; row < nnz; row++) {
		for (int a = 0; a < Q; a++) {
			//fprintf(fp_alphal_val, "%hhu ",ALPHAmn_[row][a]);
			if(ALPHAmn_[row][a]!=0)
				fprintf(fp_alphal_val,"ALPHAMN[%d][%d]=%hhu\n",row, a,ALPHAmn_[row][a]);

		}
		/*fprintf(fp_alphal_val, "\n");
		cnt++;
		if(cnt==row_weight[0]){
			cnt=0;
			fprintf(fp_alphal_val, "\n\n");
		} */
	}

	fclose(fp_B_val);
	fclose(fp_F_val);
	fclose(fp_a_val);
	fclose(fp_beta_val);
		fclose(fp_alpha_t_val);


	/* for (int index_r = 0; index_r < M; index_r++) {// check node processing: update check node message BETA
		for (int index = row_ptr[index_r]; index < row_ptr[index_r + 1]; index++) {// check node processing: update check node message BETA
			for (a = 0; a < Q; a++) {
				fprintf(fp_alphal_val, "%d ", ALPHAmn_[ptr_to_val[col_ptr[col_v]+index_v]][a]);
			}
			fprintf(fp_alphal_val, "\n");
		}
		fprintf(fp_alphal_val, "\n\n");
	}  */

	fclose(fp_alphal_val);


#ifdef DEBUG
	fclose(fp);
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
	fclose(fp5);
	fclose(fp6);
	fclose(fp7);
	fclose(fp_beta_B);
	fclose(fp_beta_F);
	fclose(fp_validation);
#endif

	for (i = 0; i < Q; i++) {
		free(B[i]);
	}
	free(B);

	for (i = 0; i < Q; i++) {
		free(F[i]);
	}
	free(F);

	free(ALPHA_t);
	free(codeword_sym);

	for (i = 0; i < Q; i++) {
		free(GAMMAn_post[i]);
	}
	free(GAMMAn_post);


	return error;

}
