
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

	/* for(index_ = 1, col_ = col_row[row][0]; index_ <= index ; index_++, col_ = col_row[row][index_] ){
		for(c = 0; c < Q; c++){
			for(b = 0; b < Q; b++){
				a = sub_gf(c,add_gf(H_nb[row][col],b));
				max = max(F[a],ALPHAmn[row][col_][b]);
				min = min(min,max);
			}
			F[c] = min;
		}
	} */
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

	/* for(index_ = row_weight[row]-1, col_ = col_row[row][index_]; index_ >= index ; index_--, col_ = col_row[row][index_] ){
		for(c = 0; c < Q; c++){
			for(b = 0; b < Q; b++){
				a = sub_gf(c,add_gf(H_nb[row][col],b));
				max = max(B[a],ALPHAmn[row][col_][b]);
				min = min(min,max);
			}
			B[c] = min;
		}
	} */
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

	//col_f = col_row[row][index_f];
	//col_b = col_row[row][index_b];

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
unsigned short minmax(unsigned char** BETAmn_, unsigned char** ALPHAmn_, unsigned char** GAMMAn_, int* iteration, int* decoded_bit)
{
	int row, col, index, index_c, index_v, col_v, row_v, error;
	int index_, row_;
	int last_col;
	int index_B, col_B;
	int index_p, col_p;
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
	unsigned char temp;
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

	ALPHA_t = (unsigned char *)malloc(sizeof(unsigned char) * Q);
	codeword_sym = (int *)malloc(sizeof(int) * N);
	GAMMAn_post = malloc2Dunsigned_char(Q, N);

	B = (unsigned char **)malloc2Dunsigned_char(Q, dv);
	F = (unsigned char **)malloc2Dunsigned_char(Q, dv);

#ifdef DEBUG
	for (col = 0; col < N; col++) {
		for (a = 0; a < Q; a++) {
			fprintf(fp3, "%hhu ", GAMMAn_[a][col]);
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

	//FILE *fp_alpha_val = fopen("./data/alpha_val.txt", "w");


	for (index = 0; index < nnz; index++) {// initialize variable node message ALPHA with channel info GAMMA
		for (a = 0; a < Q; a++) {
			ALPHAmn_[index][a] = GAMMAn_[a][col_ind[index]];
			//printf("Alpha[%d][%d]=%d, col=%d\n", index,a,ALPHAmn_[index][a],col_ind[index]);

			//fprintf(fp_alpha_val, "%d ",ALPHAmn_[index][a]);

		}
		//printf("row=%d, col=%d\n", 0, col_ind[index]);
	}

	//fclose(fp_alpha_val);


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
			fprintf(fp3, "%hhu ", GAMMAn_[i][col]);
#endif
			if (GAMMAn_[i][col] < GAMMAn_[a][col]) {
				a = i;
			}

		}
		codeword_sym[col] = a;
#ifdef DEBUG
		fprintf(fp, "%3d ", codeword_sym[col]);
		fprintf(fp3, "\n");
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

	while (iter < MAX_ITERATION) {	// iteration starts
		for (int index_r = 0; index_r < M; index_r++) {// check node processing: update check node message BETA
			for (int index = row_ptr[index_r]; index < row_ptr[index_r + 1]; index++) {// check node processing: update check node message BETA
				if (index == row_ptr[index_r]) {
					//printf("row=%d, col=%d, last_col=%d\n", index_r, col_ind[index],col_ind[row_ptr[index_r+1]-1]);
					for (a = 0; a < Q; a++) {
						F[a][0] = ALPHAmn_[index][mul_gf(inv_gf(val[index]), a)];
						B[a][row_weight[index_r] - 1] = ALPHAmn_[row_ptr[index_r + 1] - 1][mul_gf(inv_gf(val[row_ptr[index_r + 1] - 1]), a)];
						//printf("F[%d][0]=%d, B[%d][%d]=%d, row=%d, last_col=%d, mult=%d\n", a, F[a][0],a,row_weight[index_r]-1, B[a][row_weight[index_r]-1],index_r, row_ptr[index_r+1]-1,mul_gf(inv_gf(val[row_ptr[index_r+1]-1]),a));
					}
				}
				else {
					index_p = index - row_ptr[index_r] - 1;

					//printf("index_p=%d\n", index_p);

					for (c = 0; c < Q; c++) {
						min_F = max(F[c][index_p], ALPHAmn_[index][0]);
						//printf("min_F=%d, F[%d][%d]=%d, ,ALPHAmn_[%d][0]=%d\n", min_F,c,index_p,F[c][index_p],index,ALPHAmn_[index][0]);
						for (b = 1; b < Q; b++) {
							a = sub_gf(c, mul_gf(val[index], b));
							max_F = max(F[a][index_p], ALPHAmn_[index][b]);
							min_F = min(min_F, max_F);
						}
						F[c][index - row_ptr[index_r]] = min_F;
						//printf("F[%d][%d]=%d\n", c,index-row_ptr[index_r ],F[c][index-row_ptr[index_r]]);

					}
					//printf("check\n");

					index_B = row_ptr[index_r+1] - index-1;
					//if(index == row_ptr[0]+1)
					//printf("index_B=%d, row_weight[%d]=%d, index=%d, row_ptr=%d\n==============================================================================\n", index_B,index_r,row_weight[index_r-1], index, row_ptr[index_r-1]);
					for (c = 0; c < Q; c++) {
						min_B = max(ALPHAmn_[row_ptr[index_r ] + index_B][0], B[c][index_B + 1]);
						//printf("1st min_B=%d, row_ptr[index_r] + index_B=%d, c=%d, index_B+1=%d, alpha=%d, b=%d \n", min_B, row_ptr[index_r] + index_B, c, index_B +1,ALPHAmn_[row_ptr[index_r ] + index_B][0],B[c][index_B + 1]);
						for (b = 0; b < Q; b++) {
							a = sub_gf(c, mul_gf(val[row_ptr[index_r] + index_B ], b));
							//printf("a=%d\n", a);
							max_B = max(B[a][index_B + 1], ALPHAmn_[row_ptr[index_r ] + index_B][b]);
							fprintf(fp_a_val,"%d ", max_B);
							//printf("max_B=%d\n", max_B);
							//if(index == row_ptr[0]+1)
							//printf("B[%d][%d]=%d, ALPHAmn_[%d][%d]=%d\n", a, index_B + 1,B[a][index_B + 1] , row_ptr[index_r ] + index_B,b,ALPHAmn_[row_ptr[index_r ] + index_B ][b] );
							//printf("a=%d, index_b+1=%d, inde2=%d, b=%d, h=%d\n", a, index_B + 1,row_ptr[index_r] + index_B , b,val[row_ptr[index_r] + index_B ] );
							min_B = min(min_B, max_B);
							//printf("min_B=%d\n", min_B);
						}
						B[c][index_B] = min_B;
						//if(index == row_ptr[0]+1)
						//printf("last min_B=%d\n", min_B);
						//printf("B[%d][%d]=%d\n", c,index_B,B[c][index_B]);

					}
				}
			}

			//break;

			

			for (a = 0; a < Q; a++) {
				for (index = 0; index < dv; index++) {
					//printf("B=%d\n", B[a][index]);
					//printf("F=%d\n", F[a][index]);
					fprintf(fp_B_val,"%d ", B[a][index]);
					fprintf(fp_F_val,"%d ", F[a][index]);
				}
			}
			

			//printf("check=%d\n",index_r);	

			/* for (int index = row_ptr[index_r]; index < row_ptr[index_r + 1]; index++) {//2. update BETA values from B and F values				/*if(index_c == 0){
				if (index == row_ptr[index_r]) {
					for (a = 0; a < Q; a++) {
						BETAmn_[index][a] = B[mul_gf(val[index], a)][1];
						//printf("BETAmn_[%d][%d]=%d\n", index, a, BETAmn_[index][a]);
					}
				}
				if (index == row_ptr[index_r+1]-1) {
					for (a = 0; a < Q; a++) {
						BETAmn_[index][a] = B[mul_gf(val[index], a)][index-row_ptr[index_r]];
						//printf("BETAmn_[%d][%d]=%d\n", index, a, BETAmn_[index][a]);
					}
				}
			} */
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

		/* for(col_v = 0; col_v < N; col_v ++){	// variable node processing: update variable node message ALPHA

			#ifdef PRINTMSG_V
				printf("col_v = %d\n",col_v);
			#endif

			index_v = 0;
			row_v = row_col[0][col_v];
			while(index_v < (col_weight[col_v])){
				row_v = row_col[index_v][col_v];

				#ifdef PRINTMSG_V
					printf("index_v = %d row_v = %d\n",index_v, row_v);
				#endif

				for(a = 0; a < Q; a++){

					#ifdef PRINTMSG_V
						printf("a = %d\n",a);
					#endif

					temp = 0;
					index_ = 0;
					row_ = row_col[0][col_v];
					while(index_ < col_weight[col_v]){
						row_ = row_col[index_][col_v];
						if(index_ != index_v){

							#ifdef PRINTMSG_V
								printf("index_ = %d row_ = %d\n",index_, row_);
							#endif
							#ifdef PRINT_A_B
								fprintf(fp7, "%hhu ", BETAmn_[row_][col_v][a]);
							#endif

							temp += BETAmn_[row_][col_v][a];

							#ifdef PRINTMSG_V
								printf("BETAmn_[%d][%d][%d] = %hhu temp = %hhu\n",row_, col_v, a, BETAmn_[row_][col_v][a], temp);
							#endif

						}
						index_++;
					}

					#ifdef PRINT_A_B
						fprintf(fp7, "  ");
					#endif

					ALPHA_t[a] = temp + GAMMAn_[a][col_v];

					#ifdef PRINTMSG_V
						printf("temp = %hhu GAMMAn_[%d][%d] = %hhu ALPHA_t[%d] = %hhu\n",temp, a, col_v, GAMMAn_[a][col_v] , a, ALPHA_t[a]);
					#endif

				}

				#ifdef PRINT_A_B
					fprintf(fp7, "\n");
				#endif

				min_index = 0;
				for(a = 1; a < Q; a++){
					if(ALPHA_t[a] < ALPHA_t[min_index]){
						min_index = a;
					}
				}
				for(a = 0; a < Q; a++){
					ALPHAmn_[row_v][col_v][a] = ALPHA_t[a] - ALPHA_t[min_index];

					#ifdef DEBUG
						fprintf(fp1, "ALPHAmn_[%3d, %3d, %3d] = %hhu, ALPHA_t[%3d] = %hhu\n",row_v, col_v, a, ALPHAmn_[row_v][col_v][a],a, ALPHA_t[a]);
					#endif

				}

				#ifdef PRINTMSG_V
					printf("ALPHA_t[min] = %hhu ALPHA_t[%d] = %hhu ALPHAmn_[%d][%d] = %hhu\n", ALPHA_t[min_index], a, ALPHA_t[a], row_v, col_v, ALPHAmn_[row_v][col_v][a]);
				#endif

				index_v++;
			}
		} */
		//printf("variable node processing finished\n");

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

		/* for(col = 0; col < N; col ++){	// post processing
			for( a = 0; a < Q; a++){
				temp = 0;
				index_ = 0;
				row_ = row_col[0][col];
				while(index_ < col_weight[col]){
					row_ = row_col[index_][col];

					#ifdef PRINTMSG_P
						printf("index_ = %d row_ = %d\n",index_, row_);
					#endif

					temp += BETAmn_[row_][col][a];

					#ifdef PRINTMSG_P
						printf("BETAmn_[%d][%d][%d] = %hhu temp = %hhu\n", row_, col, a, BETAmn_[row_][col][a], temp);
					#endif

					index_++;
				}
				GAMMAn_post[a][col] = temp + GAMMAn[a][col];

				#ifdef DEBUG
					//printf("GAMMAn[%d][%d] = %3.3f GAMMAn_post[%d][%d] = %3.3f\n", a, col, GAMMAn_post[a][col], a, col, GAMMAn_[a][col]);
				#endif

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
		#endif*/
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
