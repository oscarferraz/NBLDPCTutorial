
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
unsigned char  mul_gf(unsigned char  a, unsigned char  b)
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
unsigned char  inv_gf(unsigned char  a)
{
	//if(a == 0) exit (EXIT_FAILURE);
	if (a == 0) return 0;
	if (a == 1) return 1;
	return expq[(Q - 1 - logq[a])];
	//return expq[(Q - 1 - logq[a]) % (Q-1)];

}

//===================================
// Galios Field Multiplication
//===================================
unsigned char div_gf(unsigned char a, unsigned char b)
{
	if (a == 0) return 0;
	if (b == 1) return a;
	return expq[(logq[a] - logq[b] + (Q - 1)) % (Q - 1)];	//table look-up multiplication
}

//===================================
// Forward function
//===================================
#ifdef TREE
double forward(int symbol, int row, int col, int index)
{
	int a, b;
	double max_value, min_value;
	int index_;

	index_ = index - 1;
	col = col_row[row][index_];
	max_value = 0.0;
	min_value = 100.0;

	if (index_ == 0) {	// reach the first non-zero element in current row
		return ALPHAmn[row][col][add_gf((H_nb[row][col]), symbol)];
	}
	else {
		for (b = 0; b < Q; b++) {
			a = sub_gf(symbol, add_gf(H_nb[row][col], b));
			max_value = max(forward(a, row, col, index_), ALPHAmn[row][col][b]);
			min_value = min(min_value, max_value);
		}
		return min_value;
	}
}
#else
void forward(int row, int col, int index, double* F)
{
	int index_, col_;
	int a, b, c, i;
	double min, max;
	double* F_;

	F_ = (double*)malloc(sizeof(double) * Q);
	max = min = F[0];

	for (i = 0; i < Q; i++) {
		F_[i] = F[i];
	}

	for (index_ = 1, col_ = col_row[row][0]; index_ <= index; index_++, col_ = col_row[row][index_]) {
		for (c = 0; c < Q; c++) {
			for (b = 0; b < Q; b++) {
				a = sub_gf(c, add_gf(H_nb[row][col], b));
				max = max(F[a], ALPHAmn[row][col_][b]);
				min = min(min, max);
			}
			F[c] = min;
		}
	}
}
#endif

//===================================
// Backward function
//===================================
#ifdef TREE
double backward(int symbol, int row, int col, int index)
{
	int a, b;
	double max_value, min_value;
	int index_;

	index_ = index + 1;
	col = col_row[row][index_];
	max_value = 0.0;
	min_value = 100.0;

	if (index_ == row_weight[row] - 1) {	// reach the last non-zero element in current row
		return ALPHAmn[row][col][add_gf((H_nb[row][col]), symbol)];
	}
	else {
		for (b = 0; b < Q; b++) {
			a = sub_gf(symbol, add_gf(H_nb[row][col], b));
			max_value = max(backward(a, row, col, index_), ALPHAmn[row][col][b]);
			min_value = min(min_value, max_value);
		}
		return min_value;
	}
}
#else
void backward(int row, int col, int index, double* B)
{
	int index_, col_;
	int a, b, c, i;
	double min, max;
	double* B_;

	B_ = (double*)malloc(sizeof(double) * Q);
	max = min = B[0];

	for (i = 0; i < Q; i++) {
		B_[i] = B[i];
	}

	for (index_ = row_weight[row] - 1, col_ = col_row[row][index_]; index_ >= index; index_--, col_ = col_row[row][index_]) {
		for (c = 0; c < Q; c++) {
			for (b = 0; b < Q; b++) {
				a = sub_gf(c, add_gf(H_nb[row][col], b));
				max = max(B[a], ALPHAmn[row][col_][b]);
				min = min(min, max);
			}
			B[c] = min;
		}
	}
}
#endif



//===================================
// Root function
//===================================
#ifdef TREE
double root(int symbol, int row, int col, int index)
{
	int a, b;
	double max_value, min_value;

	max_value = 0.0;
	min_value = 100.0;

	for (a = 0; a < Q; a++)
	{
		b = add_gf(add_gf(H_nb[row][col], symbol), a);
		max_value = max(forward(a, row, col, index), backward(b, row, col, index));
		min_value = min(min_value, max_value);
	}
	return min_value;
}
#else
void root(int row, int col, int index, double* temp)
{
	int a, b, c;
	double min, max;
	int index_f, index_b;
	int col_f, col_b;
	double* F;
	double* B;

	index_f = index - 1;
	index_b = index + 1;

	col_f = col_row[row][index_f];
	col_b = col_row[row][index_b];

	F = (double *)malloc(sizeof(double) * Q);
	B = (double *)malloc(sizeof(double) * Q);

	forward(row, col_f, index, F);
	backward(row, col_b, index, B);

	max = min = F[0];

	for (c = 0; c < Q; c++) {
		for (b = 0; b < Q; b++) {
			a = sub_gf(c, add_gf(H_nb[row][col], b));
			max = max(F[a], B[b]);
			min = min(min, max);
		}
		temp[c] = min;
	}
}
#endif

//===================================
// Minmax nonbinary decode function
//===================================
// Refer to "Min-max decoding for nonbinary LDPC codes"
int minmax(unsigned char*** BETAmn_, unsigned char*** ALPHAmn_, unsigned char** GAMMAn_, int* iteration, int* decoded_bit)
{
	int row, col, index, index_c, index_v, col_v, row_v, error;
	int index_, row_;
	int last_col;
	int index_B, col_B;
	//int index_p;
	int row_weight_;
	int i, j;
	int a, b, c;
	int iter = 0;
	int min_index;
	int syndrome;
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
	//FILE *fp_alpha_t_val = fopen("./data/alpha_t_val.txt", "w");
	FILE *fp_alphal_val = fopen("./data/alphal_val.txt", "w"); 

#endif

	ALPHA_t = (unsigned char *)malloc(sizeof(unsigned char) * Q);
	codeword_sym = (int *)malloc(sizeof(int) * N);
	GAMMAn_post = malloc2Dunsigned_char(Q, N);

	B = (unsigned char **)malloc2Dunsigned_char(Q, dv);
	F = (unsigned char **)malloc2Dunsigned_char(Q, dv);

#ifdef DEBUG
	for (col = 0; col < N; col++) {
		for (a = 0; a < Q; a++) {
			fprintf(fp3, "%d ", GAMMAn_[a][col]);
			//printf("GAMMAn_[%d][%d]=%d\n", a, col, GAMMAn_[a][col]);
		}
		fprintf(fp3, "\n");
	}
#endif



	for (i = 0; i < Q; i++) {
		for (j = 0; j < dv; j++) {
			F[i][j] = B[i][j] = 0;
		}
	}

	for (i = 0; i < Q; i++) {
		for (j = 0; j < N; j++) {
			GAMMAn_post[i][j] = 0;
		}
	}

	FILE *fp_alpha_val = fopen("./data/alpha_val.txt", "w");

	for (row = 0; row < M; row++)// initialize variable node message ALPHA with channel info GAMMA
	{
		col = col_row[row][0];
		for (index = 0; index < row_weight[row]; index++){
			col = col_row[row][index];
			for (a = 0; a < Q; a++) {
				ALPHAmn_[row][col][a] = GAMMAn_[a][col];
				//printf("Alpha[%d][%d][%d]=%d\n", row, col,a,ALPHAmn_[row][col][a]);
				fprintf(fp_alpha_val, "%d\n", ALPHAmn_[row][col][a]);
			}
		}
	}

	
	fclose(fp_alpha_val);


	for (row = 0; row < M; row++) {
		for (col = 0; col < N; col++) {
			for (a = 0; a < Q; a++) {
				fprintf(fp1, "ALPHAmn_[%d][%d][%d]=%hhu\n", row, col, a, ALPHAmn_[row][col][a]);
			}
		}
	} 




#ifdef DEBUG
	for (col = 0; col < N; col++) {
		fprintf(fp, "encoded_sym[%3d]=%3d\n", col, encoded_sym[col]);
	}
	printf("chcek\n");
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

		


	error = 0;

	for (i = 0; i < (N - M); i++) {
		if (encoded_sym[i + M] != codeword_sym[i + M])
			error++;
	}

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

#ifdef DEBUG
	fprintf(fp,"error iss %d\n", error);
	printf("error iss %d\n", error);

	/* for(row = 0; row < M; row ++){
		syndrome = 0;
		for(col = 0; col < N; col ++){
			syndrome = add_gf(syndrome,mul_gf(H_nb[row][col],codeword_sym[col]));
		}
		printf("row = %d syndrome = %d\n",row, syndrome);
	}  */
#endif	

	//FILE *fp_a_val = fopen("./data/a_val.txt", "w");
	FILE *fp_B_val = fopen("./data/B_val.txt", "w");
	FILE *fp_F_val = fopen("./data/F_val.txt", "w");
	FILE *fp_beta_val = fopen("./data/beta_val.txt", "w");

		


	clock_gettime(CLOCK_MONOTONIC, &start);

	while (iter < MAX_ITERATION) {	// iteration starts
		for (row = 0; row < M; row++) {	// check node processing: update check node message BETA
			row_weight_ = row_weight[row];
			for (index_c = 0, col = col_row[row][0]; index_c < row_weight_; index_c++, col = col_row[row][index_c]) {		// 1. Initialize F and B value for each row
				if (index_c == 0) {
					//printf("row=%d, col=%d, last_col=%d\n", row, col ,col_row[row][row_weight_ - 1]);
					for (a = 0; a < Q; a++) {
						F[a][0] = ALPHAmn_[row][col][mul_gf(inv_gf(H_nb[row][col]), a)];
						//if(row==0)
						//printf("F[%d][%d][0]=%d mult=%d \n", row, a, F[a][0],inv_gf(H_nb[row][col]));
						last_col = col_row[row][row_weight_ - 1];
						B[a][row_weight_ - 1] = ALPHAmn_[row][last_col][mul_gf(inv_gf(H_nb[row][last_col]), a)];
						//if(row==45)
						//printf("B[%d][%d][%d]=%d\n", row,  a, row_weight_ - 1, B[a][row_weight_ - 1]);

					}
				}
				else {

					//index_p = index_c - 1;
					//col_p = col_row[row][index_p];
					//printf("index1=%d\n", index_c-1);

					for (c = 0; c < Q; c++) {

						//min_F = max(F[c][index_c - 1],ALPHAmn_[row][col][0]);
						min_F = max(F[c][index_c - 1], ALPHAmn_[row][col][0]);
						//if(row==0)
						//printf("min_F=%d, F[%d][%d]=%d, ,ALPHAmn_[%d][%d][0]=%d\n", min_F,c,index_c,F[c][index_c - 1],row, col,ALPHAmn_[row][col][0]);

						for (b = 1; b < Q; b++) {
							a = sub_gf(c, mul_gf(H_nb[row][col], b));
							//a = mul_gf(sub_gf(c,mul_gf(H_nb[row][col],b)),inv_gf(H_nb[row][col_p]));
							
							//max_F = max(F[mul_gf(H_nb[row][col],b)][index_c - 1], ALPHAmn_[row][col][a]);
							max_F = max(F[a][index_c - 1], ALPHAmn_[row][col][b]);
							//if(row==0 )
							//printf("min_F=%d, max=%d, F[%d][%d][%d]=%d, alpha[%d][%d][%d]=%d\n", min_F,max_F,row, a, index_c-1,F[a][index_c - 1],row, col, b, ALPHAmn_[row][col][b]);

							
							min_F = min(min_F, max_F);

							//if(row==0)
							//printf("min_F=%d, max=%d, F[%d][%d][%d]=%d, alpha[%d][%d][%d]=%d\n", min_F,max_F,row, a, index_c-1,F[a][index_c - 1],row, col, b, ALPHAmn_[row][col][b]);

							
						}
						F[c][index_c] = min_F;
						//if(row==0  )
						//printf("F[%d][%d][%d]=%d\n",row, c,index_c,F[c][index_c]);



					}
					
					index_B = row_weight_ - index_c - 1;
					//if(index_c==1)
					//printf("index_B=%d\n==============================================================================\n", index_B);

					col_B = col_row[row][index_B];
					//printf("col_B=%d\n", col_B);

					//index_p = index_B + 1;
					//col_p = col_row[row][index_p];
					for (c = 0; c < Q; c++) {
						min_B = max(ALPHAmn_[row][col_B][0], B[c][index_B + 1]);
						//printf("min_B=%d, row=%d, col_B=%d, c=%d, index_B + 1=%d, alpha=%d, b=%d\n", min_B, row, col_B, c, index_B + 1,ALPHAmn_[row][col_B][0],B[c][index_B + 1]);

						for (b = 0; b < Q; b++) {
							a = sub_gf(c, mul_gf(H_nb[row][col_B], b));
							//fprintf(fp_a_val,"%d ", a);
							//printf("a=%d\n", a);
							max_B = max(B[a][index_B + 1], ALPHAmn_[row][col_B][b]);
							//fprintf(fp_a_val,"%d ", max_B);
							//printf("max_B=%d\n", max_B);
							//if(index_c==1)
							//printf("B[%d][%d]=%d, ALPHAmn_[%d][%d][%d]=%d\n", a, index_B + 1,B[a][index_B + 1] , row,col_B,b,ALPHAmn_[row][col_B][b] );
							//printf("a=%d, index_b+1=%d, row=%d,colB=%d, b=%d, h=%d\n", a, index_B + 1,row, col_B, b,H_nb[row][col_B] );
							min_B = min(min_B, max_B);
							//printf("min_B=%d\n", min_B);

						}
						B[c][index_B] = min_B;
						//if(index_c==1)
						//printf("last min_B=%d\n", min_B);
						//if(row==5)
						//printf("B[%d][%d][%d]=%d\n",row, c,index_B,B[c][index_B]);




					}
				}
			}

			//break;
			

			

		
			

			for (index_c = 0, col = col_row[row][0]; index_c < row_weight_; index_c++, col = col_row[row][index_c]) {	//2. update BETA values from B and F values
				if (index_c == 0) {
					for (a = 0; a < Q; a++) {
						BETAmn_[row][col][a] = B[mul_gf(H_nb[row][col], a)][index_c + 1];
						//printf("BETAmn_[%d][%d][%d]=%d\n", row, col, a, BETAmn_[row][col][a]);



					}
				}
				else if (index_c == (row_weight[row] - 1)) {
					for (a = 0; a < Q; a++) {
						BETAmn_[row][col][a] = F[mul_gf(H_nb[row][col], a)][index_c - 1];
						//printf("BETAmn_[%d][%d][%d] = F[mul_gf(H_nb[%d][%d], %d)][%d] = %hhu\n", row, col, a, row, col, a,index_c - 1,BETAmn_[row][col][a] );


					}
				}
				else {

					for (c = 0; c < Q; c++) {

						min_value = max(F[mul_gf(H_nb[row][col], c)][index_c - 1], B[0][index_c + 1]);
						//if(row==0 && c==2)
						//printf("mmin=%d F[%d][%d][%d]=%d, B[%d][%d][%d]=%d, col=%d\n", min_value,row, mul_gf(H_nb[row][col], c), index_c - 1, F[mul_gf(H_nb[row][col], c)][index_c - 1],row, 0, index_c+1, B[0][index_c + 1], col);

						for (b = 0; b < Q; b++) {
							a = add_gf(b, mul_gf(H_nb[row][col], c));
							max_value = max(F[a][index_c - 1], B[b][index_c + 1]);
							//if(row==0 && c==2)
							//printf("max=%d min=%d z=%d, b=%d, a=%d\n", max_value, min_value, c, b, a);
							min_value = min(max_value, min_value);
							//if(row==0 && c==2)
							//printf("min_value = %d\n", min_value);
						}
						BETAmn_[row][col][c] = min_value;
						//if(row==0)
						//printf("BETAmn_[%d][%d][%d] = %d\n", row, col, c, BETAmn_[row][col][c]);

					}
				}
			} 

			if(iter == MAX_ITERATION-1){
				for (a = 0; a < Q; a++) {
					for (index = 0; index < row_weight[0]; index++) {
						if(row==0 && iter==1){
							//printf("B=%d\n", B[a][index]);
							printf("F[%d][%d]=%d\n",a,index, F[a][index]);
						}
						fprintf(fp_B_val,"%d ", B[a][index]);
						fprintf(fp_F_val,"%d ", F[a][index]);
					}
					fprintf(fp_B_val,"\n");
					fprintf(fp_F_val,"\n");
				}
				fprintf(fp_B_val,"\n\n");
				fprintf(fp_F_val,"\n\n");
			}
 
		}

		
		if(iter == MAX_ITERATION-1){
			for (int row = 0; row < M; row++) {
				for (index_c = 0, col = col_row[row][0]; index_c < row_weight_; index_c++, col = col_row[row][index_c]) {	//2. update BETA values from B and F values
					for (a = 0; a < Q; a++) {
						fprintf(fp_beta_val,"%d ", BETAmn_[row][col][a]);
						//printf("BETAmn_[%d][%d][%d]=%d\n",row, col, a, BETAmn_[row][col][a]);
					}
					fprintf(fp_beta_val,"\n");
				}
				fprintf(fp_beta_val,"\n\n");
			}
		}

		
		//fclose(fp_a_val);
	   	//printf("check node processing finished\n");


		for (col_v = 0; col_v < N; col_v++) {	// variable node processing: update variable node message ALPHA


			index_v = 0;
			row_v = row_col[0][col_v];
			while (index_v < (col_weight[col_v])) {
				row_v = row_col[index_v][col_v];

				for (a = 0; a < Q; a++) {

					temp = 0;
					index_ = 0;
					row_ = row_col[0][col_v];
					while (index_ < col_weight[col_v]) {
						row_ = row_col[index_][col_v];
						//printf("row_ = %d\n", row_);
						if (index_ != index_v) {


							temp = temp+ BETAmn_[row_][col_v][a];
							//if(col_v==0)

							//printf("BETAmn_[%d][%d][%d] = %hhu temp = %hhu\n", row_, col_v, a, BETAmn_[row_][col_v][a], temp);

						}
						
						index_++;
					}
					//if( row_v==0 && col_v==8 )
					//printf("temp=%d\n",temp);

					ALPHA_t[a] = temp + GAMMAn_[a][col_v];
					//if( row_v==0 && col_v==276 && a==1 )
					//printf("ALPHA_t[%d]=%d GAMMAn_[%d][%d]=%d\n", a,ALPHA_t[a],a,col_v,GAMMAn_[a][col_v] );

				}
				/* for (a = 0; a < Q; a++) {
					fprintf(fp_alpha_t_val,"%d\n", ALPHA_t[a]);
				} */

				min_index = 0;
				for (a = 1; a < Q; a++) {
					//if( row_v==0 && col_v==276 )
					//printf("t[%d]= %hhu, min=%d\n", a,ALPHA_t[a],ALPHA_t[min_index]);
					if (ALPHA_t[a] < ALPHA_t[min_index]) {
						min_index = a;
					}
					//if( row_v==0 && col_v==155 )
				//printf("min= %hhu\n", min_index);

				}
				//printf("========================================\n" );

				//if( row_v==0 && col_v==276)
				//printf("min= %hhu\n", min_index);
				
				for (a = 0; a < Q; a++)
				{
					ALPHAmn_[row_v][col_v][a] = ALPHA_t[a] - ALPHA_t[min_index];
					//printf("ALPHA_t[%d] = %hhu ALPHA_t2[%d] = %hhu\n", a, ALPHA_t[a], min_index,  ALPHA_t[min_index]);
					//fprintf(fp_alphal_val, "%hhu ",ALPHAmn_[row_v][col_v][a]);
					//if(row_v==29 )
				 	//printf("alpha[%d][%d][%d]=%d, ALPHA_t[%d]=%d, temp[%d]=%d\n",row_v,col_v,a,ALPHAmn_[row_v][col_v][a],a,ALPHA_t[a] ,min_index,ALPHA_t[min_index]);

				}


				index_v++;
			}
		}
		
 

				//fclose(fp_alpha_t_val);



		
		//printf("variable node processing finished\n");

		


		/*for (col = 0; col < N; col++) {	// post processing
			for (a = 0; a < Q; a++) {
				temp = 0;
				index_ = 0;
				row_ = row_col[0][col];
				while (index_ < col_weight[col]) {

					row_ = row_col[index_][col];
					temp += BETAmn_[row_][col][a];
					//printf("BETAmn_[%d][%d][%d] = %hhu temp = %hhu\n", row_, col, a, BETAmn_[row_][col][a], temp);
					
					index_++;
				}
				GAMMAn_post[a][col] = temp + GAMMAn[a][col];

				//printf("GAMMAn[%d][%d] = %d GAMMAn_post[%d][%d] = %d\n", a, col, GAMMAn_post[a][col], a, col, GAMMAn_[a][col]);

			}
		}
		//printf("post processing finished\n");
		for (col = 0; col < N; col++) {	// tentative decoding
			a = 0;
			for (i = 0; i < Q; i++) {
#ifdef DEBUG
				fprintf(fp2, "GAMMAn_post[%3d][%3d] = %hhu\n", i, col, GAMMAn_post[i][col]);
				fprintf(fp_validation, "%hhu\n", GAMMAn_post[i][col]);

#endif
				if (GAMMAn_post[i][col] < GAMMAn_post[a][col]) {
					a = i;
				}

			}
			codeword_sym[col] = a;
#ifdef DEBUG
			fprintf(fp, "codeword_sym[%3d] = %3d\n", col, codeword_sym[col]);
			fprintf(fp2, "\n");
#endif
		}

		error = 0;

		for (i = 0; i < (N - M); i++) {
			if (encoded_sym[i + M] != codeword_sym[i + M])
				error++;
		}


#if  (EARLY_STOP == 1)
		if (error == 0) {
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

	printf("error = %d \n", error);

#ifdef DEBUG
		fprintf(fp, "error = %d ", error);
		fprintf(fp, "\n");
#endif		

#ifdef DEBUG
		for (row = 0; row < M; row++) {
			syndrome = 0;
			for (col = 0; col < N; col++) {
				syndrome = add_gf(syndrome, mul_gf(H_nb[row][col], codeword_sym[col]));
			}
			//printf("row = %d syndrome = %d\n",row, syndrome);
		} 
#endif	*/
		iter++;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	for (row = 0; row < M; row++) {
		for (index_c = 0, col = col_row[row][0]; index_c < row_weight[row]; index_c++, col = col_row[row][index_c]) {	//2. update BETA values from B and F values
			for (a = 0; a < Q; a++) {
				fprintf(fp_alphal_val, "%d ", ALPHAmn_[row][col][a]);
				if(row==0)
				printf("Alpha[%d][%d][%d]=%d\n", row, col, a, ALPHAmn_[row][col][a]);

			}
			fprintf(fp_alphal_val, "\n");
		}
		fprintf(fp_alphal_val, "\n\n");
	} 

	fclose(fp_alphal_val);
	fclose(fp_B_val);
	fclose(fp_F_val);
	fclose(fp_beta_val);

	printf("check\n");

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
