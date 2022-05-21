
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
	int row_weight_=0;
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
	FILE *fp_alpha_t_val = fopen("./data/alpha_t_val.txt", "w");
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
		for (index = 0; index < row_weight[row]; index++)
		{
			col = col_row[row][index];
			for (a = 0; a < Q; a++) {
				ALPHAmn_[row][col][a] = GAMMAn_[a][col];
				//printf("Alpha[%d][%d][%d]=%d\n", row, col,a,ALPHAmn_[row][col][a]);
				fprintf(fp_alpha_val, "%d\n", ALPHAmn_[row][col][a]);
			}
		}
	}

	fclose(fp_alpha_val);

/* 
	for (row = 0; row < M; row++) {
		for (col = 0; col < N; col++) {
			for (a = 0; a < Q; a++) {
				fprintf(fp1, "ALPHAmn_[%d][%d][%d]=%hhu\n", row, col, a, ALPHAmn_[row][col][a]);
			}
		}
	} */


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
	/* fprintf(fp,"error iss %d\n", error);

	for(row = 0; row < M; row ++){
		syndrome = 0;
		for(col = 0; col < N; col ++){
			syndrome = add_gf(syndrome,mul_gf(H_nb[row][col],codeword_sym[col]));
		}
		printf("row = %d syndrome = %d\n",row, syndrome);
	} */
#endif	

			FILE *fp_a_val = fopen("./data/a_val.txt", "w");
			FILE *fp_B_val = fopen("./data/B_val.txt", "w");
			FILE *fp_F_val = fopen("./data/F_val.txt", "w");
			FILE *fp_beta_val = fopen("./data/beta_val.txt", "w");


	while (iter < MAX_ITERATION) {	// iteration starts
		for (row = 0; row < M; row++) {	// check node processing: update check node message BETA
#ifdef PRINTMSG			
			printf("row = %d\n", row);
#endif
			row_weight_ = row_weight[row];
			for (index_c = 0, col = col_row[row][0]; index_c < row_weight_; index_c++, col = col_row[row][index_c]) {		// 1. Initialize F and B value for each row
				if (index_c == 0) {
					//printf("row=%d, col=%d, last_col=%d\n", row, col ,col_row[row][row_weight_ - 1]);
					for (a = 0; a < Q; a++) {
						F[a][0] = ALPHAmn_[row][col][mul_gf(inv_gf(H_nb[row][col]), a)];
						//printf("F[%d][0] = %hhu b = %d\n", a, F[a][0], mul_gf(inv_gf(H_nb[row][col]), a));
						last_col = col_row[row][row_weight_ - 1];
						B[a][row_weight_ - 1] = ALPHAmn_[row][last_col][mul_gf(inv_gf(H_nb[row][last_col]), a)];
						//printf("F[%d][0]=%d, B[%d][%d]=%d, row=%d, lastcol=%d, mult=%d\n", a, F[a][0],a,row_weight_ - 1, B[a][row_weight_ - 1], row, last_col, mul_gf(inv_gf(H_nb[row][last_col]),a));

					}
				}
				else {
#ifdef PRINTMSG
					printf("row = %d col = %d index_c = %d\n", row, col, index_c);
#endif

					//index_p = index_c - 1;
					//col_p = col_row[row][index_p];
					//printf("index1=%d\n", index_c-1);

					for (c = 0; c < Q; c++) {
#ifdef PRINTMSG
						//printf("c = %d\n",c);
#endif
						//min_F = mmax(F[c][index_c - 1],ALPHAmn_[row][col][0]);
						min_F = mmax(F[c][index_c - 1], ALPHAmn_[row][col][0]);
						//printf("min_F=%d, F[%d][%d]=%d, ,ALPHAmn_[%d][%d][0]=%d\n", min_F,c,index_c,F[c][index_c - 1],row, col,ALPHAmn_[row][col][0]);

#ifdef PRINTMSG
						printf("min_F = %hhu\n", min_F);
#endif
						for (b = 1; b < Q; b++) {
							a = sub_gf(c, mul_gf(H_nb[row][col], b));
							//a = mul_gf(sub_gf(c,mul_gf(H_nb[row][col],b)),inv_gf(H_nb[row][col_p]));
#ifdef PRINTMSG
							printf("a = %d b = %d\n", a, b);
							printf("F[%d][%d] = %hhu ALPHAmn_[%d][%d][%d] = %hhu\n", a, index_c - 1, F[a][index_c - 1], row, col, b, ALPHAmn_[row][col][b]);
#endif
							//max_F = mmax(F[mul_gf(H_nb[row][col],b)][index_c - 1], ALPHAmn_[row][col][a]);
							max_F = mmax(F[a][index_c - 1], ALPHAmn_[row][col][b]);
							min_F = mmin(min_F, max_F);
#ifdef PRINTMSG
							//printf("max_F = %hhu min_F = %hhu\n", max_F, min_F);
#endif
						}
						F[c][index_c] = min_F;
						//printf("F[%d][%d]=%d\n", c,index_c,F[c][index_c]);

#ifdef DEBUG
						if (row == 58) {
							fprintf(fp5, "F[%3d][%3d] = %hhu\n", c, index_c, F[c][index_c]);
						}
#endif

#ifdef PRINTMSG
						printf("F[%d][%d] = %hhu\n", c, index_c, F[c][index_c]);
#endif
					}
					;
					index_B = row_weight_ - index_c - 1;
					//if(index_c==1)
					//printf("index_B=%d\n==============================================================================\n", index_B);

					col_B = col_row[row][index_B];
					//printf("col_B=%d\n", col_B);
#ifdef PRINTMSG
					printf("index_B = %d col_B = %d\n", index_B, col_B);
#endif
					//index_p = index_B + 1;
					//col_p = col_row[row][index_p];
					for (c = 0; c < Q; c++) {
						min_B = mmax(ALPHAmn_[row][col_B][0], B[c][index_B + 1]);
						//printf("1st min_B=%d, row=%d, col_B=%d, c=%d, index_B + 1=%d, alpha=%d, b=%d\n", min_B, row, col_B, c, index_B + 1,ALPHAmn_[row][col_B][0],B[c][index_B + 1]);

						for (b = 0; b < Q; b++) {
							a = sub_gf(c, mul_gf(H_nb[row][col_B], b));
							fprintf(fp_a_val,"%d ", a);
							//printf("a=%d\n", a);
							max_B = mmax(B[a][index_B + 1], ALPHAmn_[row][col_B][b]);
							fprintf(fp_a_val,"%d ", max_B);
							//printf("max_B=%d\n", max_B);
							//if(index_c==1)
							//printf("B[%d][%d]=%d, ALPHAmn_[%d][%d][%d]=%d\n", a, index_B + 1,B[a][index_B + 1] , row,col_B,b,ALPHAmn_[row][col_B][b] );
							//printf("a=%d, index_b+1=%d, row=%d,colB=%d, b=%d, h=%d\n", a, index_B + 1,row, col_B, b,H_nb[row][col_B] );
							min_B = mmin(min_B, max_B);
							//printf("min_B=%d\n", min_B);

						}
						B[c][index_B] = min_B;
						//if(index_c==1)
						//printf("last min_B=%d\n", min_B);

						//printf("B[%d][%d]=%d\n", c,index_B,B[c][index_B]);

#ifdef DEBUG
						if (row == 58) {
							fprintf(fp6, "B[%3d][%3d] = %hhu\n", c, index_B, B[c][index_B]);
						}
#endif

#ifdef PRINTMSG
						printf("B[%d][%d] = %hhu\n", c, index_B, B[c][index_B]);
#endif
					}
				}
			}

			//break;
			

			

		
			

			for (index_c = 0, col = col_row[row][0]; index_c < row_weight_; index_c++, col = col_row[row][index_c]) {	//2. update BETA values from B and F values
				if (index_c == 0) {
					for (a = 0; a < Q; a++) {
						BETAmn_[row][col][a] = B[mul_gf(H_nb[row][col], a)][index_c + 1];
						//printf("BETAmn_[%d][%d][%d]=%d\n", row, col, a, BETAmn_[row][col][a]);


#ifdef DEBUG
						//fprintf(fp_beta_B, "%hhu\n", row, col, a, a, index_c + 1, BETAmn_[row][col][a]);
#endif
					}
				}
				else if (index_c == (row_weight[row] - 1)) {
					for (a = 0; a < Q; a++) {
						BETAmn_[row][col][a] = F[mul_gf(H_nb[row][col], a)][index_c - 1];
						//printf("BETAmn_[%d][%d][%d] = F[mul_gf(H_nb[%d][%d], %d)][%d] = %hhu\n", row, col, a, row, col, a,index_c - 1,BETAmn_[row][col][a] );

#ifdef DEBUG
						fprintf(fp_beta_F, "BETAmn_[%3d][%3d][%3d] = F[%3d][%3d] = %hhu\n", row, col, a, a, index_c - 1, BETAmn_[row][col][a]);
#endif
					}
				}
				else {

					for (c = 0; c < Q; c++) {
#ifdef PRINTMSG
						printf("c = %d\n", c);
#endif
						min_value = mmax(F[mul_gf(H_nb[row][col], c)][index_c - 1], B[0][index_c + 1]);
#ifdef PRINTMSG
						printf("F[%d][%d]  = %hhu B[%d][%d] = %hhu\n", mul_gf(H_nb[row][col], c), index_c - 1, F[mul_gf(H_nb[row][col], c)][index_c - 1], 0, index_c + 1, B[0][index_c + 1]);
						printf("min_value = %hhu\n", min_value);
#endif
						for (b = 0; b < Q; b++) {
							a = add_gf(b, mul_gf(H_nb[row][col], c));
#ifdef PRINTMSG
							printf("a = %d b = %d\n", a, b);
							printf("F[%d][%d] = %hhu  B[%d][%d] = %hhu\n", a, index_c - 1, F[a][index_c - 1], b, index_c + 1, B[b][index_c + 1]);
#endif
							max_value = mmax(F[a][index_c - 1], B[b][index_c + 1]);
#ifdef PRINTMSG
							printf("max_value = %hhu\n", max_value);
#endif
							min_value = mmin(max_value, min_value);
#ifdef PRINTMSG
							printf("min_value = %hhu\n", min_value);
#endif
						}
						BETAmn_[row][col][c] = min_value;
						//printf("BETAmn_[%d][%d][%d] = B[%d][%d] = %hhu\n", row, col, a,a, index_c + 1, BETAmn_[row][col][c]);
#ifdef PRINTMSG
						printf("BETAmn_[%d][%d][%d] = B[%d][%d] = %hhu\n", row, col, a,
							a, index_c + 1, BETAmn_[row][col][c]);
#endif

#ifdef DEBUG
						fprintf(fp4, "BETAmn_[%3d][%3d][%3d] = %hhu\n", row, col, c, BETAmn_[row][col][c]);
#endif
					}
				}
			} 
		}

		for (a = 0; a < Q; a++) {
			for (index = 0; index < dv; index++) {
				//printf("B=%d\n", B[a][index]);
				//printf("F=%d\n", F[a][index]);
				fprintf(fp_B_val,"%d ", B[a][index]);
				fprintf(fp_F_val,"%d ", F[a][index]);
			}
		}

		for (int row = 0; row < M; row++) {
			for (index_c = 0, col = col_row[row][0]; index_c < row_weight_; index_c++, col = col_row[row][index_c]) {	//2. update BETA values from B and F values
				for (a = 0; a < Q; a++) {
					fprintf(fp_beta_val,"%d ", BETAmn_[row][col][a]);
				}
			}
		}

		fclose(fp_B_val);
		fclose(fp_F_val);
		fclose(fp_a_val);
		//printf("check node processing finished\n");

#ifdef PRINT_A_B
		for (row = 0; row < M; row++) {
			for (col = 0; col < N; col++) {
				for (a = 0; a < Q; a++) {
					fprintf(fp4, "%hhu ", BETAmn_[row][col][a]);
				}
				fprintf(fp4, "\n");
			}
		}
#endif

		for (col_v = 0; col_v < N; col_v++) {	// variable node processing: update variable node message ALPHA

#ifdef PRINTMSG_V
			printf("col_v = %d\n", col_v);
#endif
			index_v = 0;
			row_v = row_col[0][col_v];
			while (index_v < (col_weight[col_v])) {
				row_v = row_col[index_v][col_v];
				//printf("index_v = %d row_v = %d\n", index_v, row_v);
#ifdef PRINTMSG_V
				
#endif
				for (a = 0; a < Q; a++) {
#ifdef PRINTMSG_V
					printf("a = %d\n", a);
#endif
					temp = 0;
					index_ = 0;
					row_ = row_col[0][col_v];
					while (index_ < col_weight[col_v]) {
						row_ = row_col[index_][col_v];
						//printf("row_ = %d\n", row_);
						if (index_ != index_v) {
#ifdef PRINTMSG_V
							//printf("index_ = %d row_ = %d\n", index_, row_);
#endif
#ifdef PRINT_A_B
							fprintf(fp7, "%hhu ", BETAmn_[row_][col_v][a]);
#endif
							temp = temp+ BETAmn_[row_][col_v][a];
							//if(col_v==0)

							//printf("BETAmn_[%d][%d][%d] = %hhu temp = %hhu\n", row_, col_v, a, BETAmn_[row_][col_v][a], temp);
#ifdef PRINTMSG_V
#endif
						}
						index_++;
					}
#ifdef PRINT_A_B
					fprintf(fp7, "  ");
#endif
					ALPHA_t[a] = temp + GAMMAn_[a][col_v];
					//printf("temp=%d, GAMMAn_[a][col_v]=%d\n", temp, GAMMAn_[a][col_v]);
#ifdef PRINTMSG_V
					//printf("temp = %hhu GAMMAn_[%d][%d] = %hhu ALPHA_t[%d] = %hhu\n", temp, a, col_v, GAMMAn_[a][col_v], a, ALPHA_t[a]);
#endif
				}
				for (a = 0; a < Q; a++) {
					fprintf(fp_alpha_t_val,"%d\n", ALPHA_t[a]);
				}
#ifdef PRINT_A_B
				fprintf(fp7, "\n");
#endif
				min_index = 0;
				for (a = 1; a < Q; a++) {
					//printf("t= %hhu, min=%d\n", ALPHA_t[a],ALPHA_t[min_index]);
					if (ALPHA_t[a] < ALPHA_t[min_index]) {
						min_index = a;
					}
				}
				//printf("========================================\n" );
				//printf("min= %hhu\n", min_index);

				for (a = 0; a < Q; a++)
				{
					ALPHAmn_[row_v][col_v][a] = ALPHA_t[a] - ALPHA_t[min_index];
					//printf("ALPHA_t[min] = %hhu ALPHA_t[%d] = %hhu ALPHAmn_[%d][%d] = %hhu\n", ALPHA_t[min_index], a, ALPHA_t[a], row_v, col_v, ALPHAmn_[row_v][col_v][a]);
					fprintf(fp_alphal_val, "%hhu ",ALPHAmn_[row_v][col_v][a]);
#ifdef DEBUG
					//fprintf(fp1, "ALPHAmn_[%3d, %3d, %3d] = %hhu, ALPHA_t[%3d] = %hhu\n", row_v, col_v, a, ALPHAmn_[row_v][col_v][a],a, ALPHA_t[a]);
#endif
				}

#ifdef PRINTMSG_V
#endif
				index_v++;
			}
		}
		


				fclose(fp_alpha_t_val);



		
		//printf("variable node processing finished\n");

		/* for (row = 0; row < M; row++) {
			for (index_c = 0, col = col_row[row][0]; index_c < row_weight[row]; index_c++, col = col_row[row][index_c]) {	//2. update BETA values from B and F values
				for (a = 0; a < Q; a++) {
					fprintf(fp_alphal_val, "%hhu ", ALPHAmn_[row][col][a]);
				}
				fprintf(fp_alphal_val, "\n");
			}
		} */

		fclose(fp_alphal_val);


		for (col = 0; col < N; col++) {	// post processing
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

				printf("GAMMAn[%d][%d] = %d GAMMAn_post[%d][%d] = %d\n", a, col, GAMMAn_post[a][col], a, col, GAMMAn_[a][col]);

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
#endif		
		iter++;
	}

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
