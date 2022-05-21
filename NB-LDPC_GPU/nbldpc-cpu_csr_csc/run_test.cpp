//***************************************************************************
//	Nonbinary LDPC decoder simulation
//	Minmax soft decoding
//	Hao Shen
//	Rice University
//***************************************************************************
#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "nbldpc.h"

#include <time.h>        // for Windows APIs
clock_t t1, t2;           // ticks


unsigned short	N;
unsigned short	M;
unsigned short	dc, dv;
unsigned char	**H_nb;
int	**G_nb;
unsigned short	*row_weight;
unsigned short	*col_weight;
unsigned char	*val;
unsigned short	*row_ptr;
unsigned short	*col_ind;
unsigned short	*row_ind;
unsigned short	*col_ptr;
unsigned short	*ptr_to_val;
unsigned char	**BETAmn;
unsigned char	**ALPHAmn;
unsigned char	**GAMMAn;
unsigned char	*encoded_sym;
unsigned char	**GAMMAn_post;
unsigned short 	nnz;
struct timespec start, end;


int				**row_col;
int				**col_row;

//===================================
// Allocate 2D array of int
//===================================
int **malloc2Dint(int a, int b) // allocates array[a][b]
{
	int i;
	int **pp = (int **)malloc(sizeof(int*) * a);
	int *p = (int *)malloc(sizeof(int) * a * b);
	if (pp == NULL || p == NULL) exit(-1);
	for (i = 0; i < a; i++) {
		pp[i] = p + b * i;
	}
	return pp;
}



double ***malloc3Ddouble(int a, int b, int c)
{
	int i, j;
	double*** ppp = (double ***)malloc(sizeof(double **) * M);
	if (ppp == NULL) exit(-1);
	for (i = 0; i < a; i++)
	{
		ppp[i] = (double **)malloc(sizeof(double *) * b);
		for (j = 0; j < b; j++)
			ppp[i][j] = (double *)malloc(sizeof(double) * c);
	}
	return ppp;
}

unsigned char ***malloc3Dunsigned_char(int a, int b, int c)
{
	int i, j;
	unsigned char*** ppp = (unsigned char ***)malloc(sizeof(unsigned char **) * M);
	if (ppp == NULL) exit(-1);
	for (i = 0; i < a; i++)
	{
		ppp[i] = (unsigned char **)malloc(sizeof(unsigned char *) * b);
		for (j = 0; j < b; j++)
			ppp[i][j] = (unsigned char *)malloc(sizeof(unsigned char) * c);
	}
	return ppp;
}

//===================================
// Generate Parity Check Matrix
// Non-binary H = (310,620) degree = (3,6)
// Refer to "Construction of Non-Binary Quasi-cyclic LDPC Codes by Arrays and Array Dispersions", Bo Zhou etc.
//===================================
void qc_gen(void)
{
	int i, j, k, x, x_index, m, n;
	int** W;
	int** H;
	int** H_tmp;
	int** Hj;
	int** Hj_up;
	int** Hj_low;
	int** H_disp;
	int* H_disp_loc_orig;
	int* H_disp_loc_current;

	int* H_disp_mask_loc_orig;
	int* H_disp_mask_loc_current;

	int* one_row;
	int** col_ext_matrix;
	unsigned char K, L, S, T;
	int row_w;
	int col_w;
	FILE *fp = fopen("data/qc.txt", "w+");
	FILE *fp1 = fopen("data/hnb.txt", "w+");
	FILE *fp_params = fopen("data/params.txt", "w+");
	FILE *fp_row_weight = fopen("data/row_weight.txt", "w+");
	FILE *fp_col_weight = fopen("data/col_weight.txt", "w+");
	FILE *fp_row_col = fopen("data/row_col.txt", "w+");
	FILE *fp_col_row = fopen("data/col_row.txt", "w+");


	if (Q > 8) {
		K = 2;
		L = 2;
		S = 2;
		T = 5;
#ifdef ERROR_CHECK
		printf("check Q>8\n");
#endif
	}
	else if (Q == 8) {
		/* K = 3;
		L = 1;
		S = 0;
		T = 2; */

		K = 2;
		L = 1;
		S = 0;
		T = 1;
	}
	else {
		K = 2;
		L = 1;
		S = 0;
		T = 1;
#ifdef ERROR_CHECK
		printf("check Q<=8\n");
#endif
	}

	nnz = 0;

	M = (Q - 1) * T * L;
	N = (Q - 1) * T * L * K;

	fprintf(fp_params, "%d\n", M);
	fprintf(fp_params, "%d\n", N);


	#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif


	dc = T - S;
	dv = K * (T - S);

	fprintf(fp_params, "%d\n", dc);
	fprintf(fp_params, "%d\n", dv);

	#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif


	W = malloc2Dint((Q - 1), (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	H_tmp = malloc2Dint((Q - 1) * (Q - 1), (Q - 1) * (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	one_row = (int*)malloc(sizeof(int) * (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	col_ext_matrix = malloc2Dint((Q - 1) * (Q - 1), (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	Hj_up = malloc2Dint(T * (Q - 1), T * (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	Hj_low = malloc2Dint(T * (Q - 1), T * (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	Hj = malloc2Dint(T * (Q - 1), T * (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	H_disp = malloc2Dint(L * T * (Q - 1), L * T * (Q - 1));
		#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif
	H_disp_loc_orig = (int*)malloc(sizeof(int) * L);

	#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif


	H_disp_loc_current = (int*)malloc(sizeof(int) * L);
	H_disp_mask_loc_orig = (int*)malloc(sizeof(int) * L * T);
	H_disp_mask_loc_current = (int*)malloc(sizeof(int) * L * T);
	H = malloc2Dint((Q - 1) * T, K * (Q - 1) * T);

	#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif


	row_weight = (unsigned short*)malloc(sizeof(unsigned short) * M);	//row weight table
	col_weight = (unsigned short*)malloc(sizeof(unsigned short) * N);	//colum weight table
#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif

	
	H_nb = malloc2Dunsigned_char(M, N);
	G_nb = malloc2Dint(N - M, N);
	GAMMAn = malloc2Dunsigned_char(Q, N);

	#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif



	row_col = malloc2Dint(dc, N);
	col_row = malloc2Dint(M, dv);

#ifdef ERROR_CHECK
	printf("Qc_gen() alloc check\n");
#endif

	for (i = 0; i < M; i++) {
		row_weight[i] = 0;
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() row_weight init check\n");
#endif

	for (i = 0; i < N; i++) {
		col_weight[i] = 0;
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() col_weight init check\n");
#endif



#ifdef ERROR_CHECK
	printf("Qc_gen() alpha beta init check\n");
#endif

	for (i = 0; i < Q; i++) {
		for (j = 0; j < N; j++) {
			GAMMAn[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() gamma init check\n");
#endif

	for (i = 0; i < dc; i++) {
		for (j = 0; j < N; j++) {
			row_col[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() row_col init check\n");
#endif

	for (i = 0; i < M; i++) {
		for (j = 0; j < dv; j++) {
			col_row[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() col_row init check\n");
#endif

	for (i = 0; i < (Q - 1) * T; i++) {
		for (j = 0; j < K * (Q - 1) * T; j++) {
			H[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() H init check\n");
#endif

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			H_nb[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() H_nb init check\n");
#endif

	for (i = 0; i < N - M; i++) {
		for (j = 0; j < N; j++) {
			G_nb[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() G_nb init check\n");
#endif

	for (i = 0; i < (Q - 1); i++) {
		for (j = 0; j < (Q - 1); j++) {
			W[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() W init check\n");
#endif

	for (i = 0; i < (Q - 1) * (Q - 1); i++) {
		for (j = 0; j < (Q - 1) * (Q - 1); j++) {
			H_tmp[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() H_tmp init check\n");
#endif

	for (i = 0; i < T * (Q - 1); i++) {
		for (j = 0; j < T * (Q - 1); j++) {
			Hj_up[i][j] = Hj_low[i][j] = Hj[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() Hj init check\n");
#endif

	for (i = 0; i < L * T * (Q - 1); i++) {
		for (j = 0; j < L * T * (Q - 1); j++) {
			H_disp[i][j] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() H_disp init check\n");
#endif

	for (i = 0; i < L; i++) {
		if (i == 0) {
			H_disp_loc_orig[i] = 1;
		}
		else if (i == (L - 1)) {
			H_disp_loc_orig[i] = -1;
		}
		else {
			H_disp_loc_orig[i] = 0;
		}
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() H_disp_loc init check\n");
#endif

	for (i = 0; i < L * T; i++) {
#ifdef ERROR_CHECK
		printf("i=%d, L=%d, T=%d\n", i, L, T);
#endif
		if (i == 0) {
			H_disp_mask_loc_orig[i] = 1;
		}
		else if (i >= (T + S + 1)) {
			H_disp_mask_loc_orig[i] = 1;
		}
		else {
			H_disp_mask_loc_orig[i] = 0;
		}
		fprintf(fp, "%2d ", H_disp_mask_loc_orig[i]);
	}

#ifdef ERROR_CHECK
	printf("Qc_gen() H_disp_mask init check\n");
#endif

#ifdef ERROR_CHECK
	printf("Qc_gen() variable init check\n");
#endif

	fprintf(fp, "\n\n\n\n");

	for (i = 0; i < (Q - 1); i++) {
		one_row[i] = expq[i];
	}

	for (i = 0; i < (Q - 1); i++) {
		for (j = 0; j < (Q - 1); j++) {
			W[i][j] = one_row[((j - i) + (Q - 1)) % (Q - 1)] - 1;
			fprintf(fp, "%2d ", W[i][j]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n\n\n\n");


	for (i = 0; i < (Q - 1) * (Q - 1); i++) {
		x = i;
		x_index = 0;
		while (x >= (Q - 1)) {
			x -= (Q - 1);
			x_index++;
		}
		for (j = 0; j < (Q - 1); j++) {
			col_ext_matrix[i][j] = mul_gf(W[x_index][j], one_row[x]);
			fprintf(fp, "%2d ", col_ext_matrix[i][j]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n");
	fprintf(fp, "\n");
	fprintf(fp, "\n");
	fprintf(fp, "\n");

	fprintf(fp, "hello\n");

	for (i = 0; i < (Q - 1) * (Q - 1); i++) {
		for (j = 0; j < (Q - 1); j++) {
			k = 0;
			while (one_row[k] != col_ext_matrix[i][j]) {
				k++;
			}
			H_tmp[i][j * (Q - 1) + k] = col_ext_matrix[i][j];
		}
	}

	for (i = 0; i < (Q - 1) * (Q - 1); i++) {
		for (j = 0; j < (Q - 1) * (Q - 1); j++) {
			fprintf(fp, "%2d ", H_tmp[i][j]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n\n\n\n");

	for (i = 0; i < (Q - 1) * T; i++) {
		for (j = 0; j < K * (Q - 1) * T; j++) {
			H[i][j] = H_tmp[(Q - 1) * (Q - 1) - (Q - 1) * T + i][j];
			fprintf(fp, "%2d ", H[i][j]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n\n\n\n");

	for (k = 0; k < K; k++) {
		fprintf(fp, "Hj\n");
		for (i = 0; i < T * (Q - 1); i++) {
			for (j = 0; j < T * (Q - 1); j++) {
				Hj[i][j] = H[i][j + (Q - 1) * k * T];
				fprintf(fp, "%2d ", Hj[i][j]);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n\n\n\n");

		for (i = 0; i < T * (Q - 1); i++) {
			for (j = 0; j < T * (Q - 1); j++) {
				if (j > (i + (Q - 1))) {
					Hj_up[i][j] = Hj[i][j];
					Hj_low[i][j] = 0;
				}
				else {
					Hj_low[i][j] = Hj[i][j];
					Hj_up[i][j] = 0;
				}
			}
		}

		fprintf(fp, "Hj_low\n");
		for (i = 0; i < T * (Q - 1); i++) {
			for (j = 0; j < T * (Q - 1); j++) {
				fprintf(fp, "%2d ", Hj_low[i][j]);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n\n\n\n");
		fprintf(fp, "Hj_up\n");
		for (i = 0; i < T * (Q - 1); i++) {
			for (j = 0; j < T * (Q - 1); j++) {
				fprintf(fp, "%2d ", Hj_up[i][j]);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n\n\n\n");


		fprintf(fp, "\n\n\n\n");


		for (m = 0; m < L; m++) {
			for (n = 0; n < L; n++) {
				H_disp_loc_current[n] = H_disp_loc_orig[((n - m) + L) % L];
				fprintf(fp, "%2d ", H_disp_loc_current[n]);
			}
			for (n = 0; n < L; n++) {
				if (H_disp_loc_current[n] == 1) {
					for (i = 0 + m * T * (Q - 1); i < T * (Q - 1) + m * T * (Q - 1); i++) {
						for (j = 0 + n * T * (Q - 1); j < T * (Q - 1) + n * T * (Q - 1); j++) {
							H_disp[i][j] = Hj_low[i - m * T * (Q - 1)][j - n * T * (Q - 1)];
						}
					}
				}
				else if (H_disp_loc_current[n] == -1) {
					for (i = 0 + m * T * (Q - 1); i < T * (Q - 1) + m * T * (Q - 1); i++) {
						for (j = 0 + n * T * (Q - 1); j < T * (Q - 1) + n * T * (Q - 1); j++) {
							H_disp[i][j] = Hj_up[i - m * T * (Q - 1)][j - n * T * (Q - 1)];
						}
					}
				}
				else {
					for (i = 0 + m * T * (Q - 1); i < T * (Q - 1) + m * T * (Q - 1); i++) {
						for (j = 0 + n * T * (Q - 1); j < T * (Q - 1) + n * T * (Q - 1); j++) {
							H_disp[i][j] = 0;
						}
					}
				}
			}
		}

		for (i = 0; i < L * T * (Q - 1); i++) {
			for (j = 0; j < L * T * (Q - 1); j++) {
				fprintf(fp, "%2d ", H_disp[i][j]);
			}
			fprintf(fp, "\n");
		}

		for (m = 0; m < L * T; m++) {
			for (n = 0; n < L * T; n++) {
				H_disp_mask_loc_current[n] = H_disp_mask_loc_orig[((n - m) + L * T) % (L * T)];
				if (H_disp_mask_loc_current[n] == 0) {
					for (i = m * (Q - 1); i < (m + 1) * (Q - 1); i++) {
						for (j = n * (Q - 1); j < (n + 1) * (Q - 1); j++) {
							H_disp[i][j] = 0;
						}
					}
				}
			}
		}

		for (i = 0; i < L * T * (Q - 1); i++) {
			for (j = 0; j < L * T * (Q - 1); j++) {
				H_nb[i][j + k * L * T * (Q - 1)] = H_disp[i][j];
			}
		}
	}

	fprintf(fp, "\n\n\n\n");
	fprintf(fp, "\n\n\n\n");

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			fprintf(fp1, "%2d ", H_nb[i][j]);
		}
		fprintf(fp1, "\n");
	}

	for (i = 0; i < M; i++) {
		row_w = 0;
		for (j = 0; j < N; j++) {
			if (H_nb[i][j] != 0) {
				row_w++;
			}
		}
		fprintf(fp, "row_w[%d] = %d\n", i, row_w);
	}

	for (j = 0; j < N; j++) {
		col_w = 0;
		for (i = 0; i < M; i++) {
			if (H_nb[i][j] != 0) {
				col_w++;
			}
		}
		fprintf(fp, "col_w[%d] = %d\n", j, col_w);
	}

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			if (H_nb[i][j] != 0) {
				row_weight[i]++;
			}
		}
		fprintf(fp_row_weight,"%d\n", row_weight[i]);
	}

	for (j = 0; j < N; j++) {
		for (i = 0; i < M; i++) {
			if (H_nb[i][j] != 0) {
				col_weight[j]++;
			}
		}
		fprintf(fp_col_weight, "%d\n", col_weight[j]);
	}

	fclose(fp_row_weight);
	fclose(fp_col_weight);

	for (i = 0; i < M; i++) {
		for (j = 0; j < row_weight[i]; j++) {
			for (k = 0; k < N; k++) {
				if (H_nb[i][k] != 0) {
					col_row[i][j] = k;
					fprintf(fp_col_row, "%d ", col_row[i][j]);
					j++;
				}
			}
		}
		fprintf(fp, "\n");
	}

	for (j = 0; j < N; j++) {
		for (i = 0; i < col_weight[j]; i++) {
			for (k = 0; k < M; k++) {
				if (H_nb[k][j] != 0) {
					row_col[i][j] = k;
					fprintf(fp_row_col, "%d ", row_col[i][j]);
					i++;
				}
			}
		}
		fprintf(fp, "\n");
	}


	fprintf(fp, "\n\n\n\n");

	fclose(fp);
	fclose(fp1);
	fclose(fp_row_col);
	fclose(fp_col_row);

	/* for(int j = 0; j < M; j++){
		for(int k = 0; k < N;k ++){
			if( H_nb[j][k]!=0)
			printf("H_nb[%d][%d]=%d\n", j, k, H_nb[j][k]);
		}
	} */

	for (j = 0; j < M; j++) {
		for (k = 0; k < N; k++) {
			if (H_nb[j][k] != 0) {
				nnz++;
			}
		}
	}

	fprintf(fp_params, "%d\n", nnz);

	fclose(fp_params);

	//printf("nnz=%d\n",nnz);


	BETAmn = malloc2Dunsigned_char(nnz, Q);
	ALPHAmn = malloc2Dunsigned_char(nnz, Q);


	for (i = 0; i < nnz; i++) {
		for (k = 0; k < Q; k++) {
			BETAmn[i][k] = 0;
			ALPHAmn[i][k] = 0;
		}
	}

	free(W);
	free(H_tmp);
	free(H);
	free(Hj);
	free(Hj_up);
	free(Hj_low);
	free(H_disp);
	free(H_disp_loc_orig);
	free(H_disp_loc_current);
	free(H_disp_mask_loc_orig);
	free(H_disp_mask_loc_current);
	free(one_row);
	free(col_ext_matrix);



}

void dec_init_nb_small() //not changed to short
{
	/* 	int i,j,k;
		int v;

		printf("N, M and are %d %d \n",N,M);
		printf("dc and dv are %d %d\n",dc,dv);

		row_weight = (int*)malloc(sizeof(int) * M);	//row weight table
		col_weight = (int*)malloc(sizeof(int) * N);	//colum weight table

		row_col = malloc2Dint(dc, N);
		col_row = malloc2Dint(M, dv);
		H_nb	= malloc2Dint(M,N);
		G_nb	= malloc2Dint(N-M,N);
		BETAmn	= malloc3Dunsigned_char(M,N,Q);
		ALPHAmn	= malloc3Dunsigned_char(M,N,Q);
		GAMMAn	= malloc2Dunsigned_char(Q,N);

		for(i=0;i<N;i++)
			for(j=0;j<M;j++)
				H_nb[j][i] = 0;

		H_nb[0][0] = 0;H_nb[0][1] = 0;H_nb[0][2] = 1;H_nb[0][3] = 2;H_nb[0][4] = 0;H_nb[0][5] = 3;H_nb[0][6] = 0;H_nb[0][7] = 1;
		H_nb[1][0] = 1;H_nb[1][1] = 0;H_nb[1][2] = 0;H_nb[1][3] = 1;H_nb[1][4] = 2;H_nb[1][5] = 0;H_nb[1][6] = 3;H_nb[1][7] = 0;
		H_nb[2][0] = 1;H_nb[2][1] = 1;H_nb[2][2] = 0;H_nb[2][3] = 0;H_nb[2][4] = 1;H_nb[2][5] = 2;H_nb[2][6] = 0;H_nb[2][7] = 3;
		H_nb[3][0] = 0;H_nb[3][1] = 1;H_nb[3][2] = 1;H_nb[3][3] = 0;H_nb[3][4] = 0;H_nb[3][5] = 1;H_nb[3][6] = 2;H_nb[3][7] = 0;

		for(i=0;i<(N-M);i++)
			for(j=0;j<N;j++)
				G_nb[i][j] = 0;

		for(i=0;i<M;i++)
			for(j=0;j<N;j++)
				for(k=0;k<Q;k++)
					BETAmn[i][j][k] = ALPHAmn[i][j][k] = 0.0;

		for(i=0;i<Q;i++)
			for(j=0;j<N;j++)
				GAMMAn[i][j] = 0;
		for(i = 0;i < M;i++){
			for(j = 0;j < N;j++){
				if(H_nb[i][j] != 0){
					row_weight[i]++;
				}
			}
			//fprintf(fp,"row_weight[%d] = %d\n", i , row_weight[i]);
		}

		for(j = 0;j < N;j++){
			for(i = 0;i < M;i++){
				if(H_nb[i][j] != 0){
					col_weight[j]++;
				}
			}
			//fprintf(fp,"col_weight[%d] = %d\n", j , col_weight[j]);
		}

		for(i = 0; i < M; i++){
			for(j = 0; j < row_weight[i]; j++){
				for(k = 0; k < N;k ++){
					if(H_nb[i][k] != 0){
						col_row[i][j] = k;
						//fprintf(fp,"%d ", col_row[i][j]);
						j++;
					}
				}
			}
			//fprintf(fp,"\n");
		}

		for(j = 0; j < N; j++){
			for(i = 0; i < col_weight[j]; i++){
				for(k = 0; k < M;k ++){
					if(H_nb[k][j] != 0){
						row_col[i][j] = k;
						//fprintf(fp,"%d ", row_col[i][j]);
						i++;
					}
				}
			}
			//fprintf(fp,"\n");
		}

		for(i = 0; i < N; i++){
			for(k = 1; k < col_weight[i] ; k++){
				for(j = 0; j < col_weight[i] - k; j++){
					if(row_col[j][i] > row_col[j+1][i]){
						v = row_col[j][i];
						row_col[j][i] = row_col[j+1][i];
						row_col[j+1][i] = v;
					}
				}
			}
		}




		for(i = 0; i < M; i++){
			for(j = 0; j < row_weight[i]; j++){
				for(k = 0; k < N;k ++){
					if(H_nb[i][k] != 0){
						col_row[i][j] = k;
						j++;
					}
				}
			}
		}

		printf("non-binary decoder small initilization finishes\n"); */

}


//===================================
// Generate Random Parity Check Matrix
// Refer to http://www.inference.phy.cam.ac.uk/mackay/codes/data.html
//===================================
void dec_init_nb() //not changed to short
{
	/*int i,j,k,tmp;
	char buf[BUFSIZ];
	int v;

	FILE *fp;
	FILE *fpp;
	FILE *frow_col;
	FILE *fcol_row;

	if(Q==2){
		fp = fopen("matrix/816.txt","r");
		fpp = fopen("matrix/816.H.txt","w");
		frow_col = fopen("matrix/816.row_col.txt","w");
		fcol_row = fopen("matrix/816.col_row.txt","w");
	}
	if(Q==4){
		fp = fopen("matrix/408.txt","r");
		fpp = fopen("matrix/408.H.txt","w");
		frow_col = fopen("matrix/408.row_col.txt","w");
		fcol_row = fopen("matrix/408.col_row.txt","w");
	}
	if(Q==16){
		fp = fopen("matrix/204.txt","r");
		fpp = fopen("matrix/204.H.txt","w");
		frow_col = fopen("matrix/204.row_col.txt","w");
		fcol_row = fopen("matrix/204.col_row.txt","w");
	}

	if(fp==NULL) {printf("file open error\n");}
	else printf("file open succeed\n");

	if(fgets(buf,sizeof(buf),fp)!=NULL)
	{
		//puts (buf);
	}

	sscanf(buf, "%d%d",&N,&M);
	fscanf(fp, "%d%d", &dc, &dv);
	printf("N, M and are %d %d \n",N,M);
	printf("dc and rman are %d %d\n",dc,dv);

	row_weight = (int*)malloc(sizeof(int) * M);	//row weight table
	col_weight = (int*)malloc(sizeof(int) * N);	//colum weight table

	for (i = 0; i < N; i++) {fscanf(fp, "%d", &col_weight[i]);}
	for (j = 0; j < M; j++) {fscanf(fp, "%d", &row_weight[j]);}

	row_col = malloc2Dint(dc, N);
	col_row = malloc2Dint(M, dv);
	H_nb	= malloc2Dint(M,N);
	G_nb	= malloc2Dint(N-M,N);
	BETAmn	= malloc3Dunsigned_char(M,N,Q);
	ALPHAmn	= malloc3Dunsigned_char(M,N,Q);
	GAMMAn	= malloc2Dunsigned_char(Q,N);

	for(i=0;i<N;i++)
		for(j=0;j<M;j++)
			H_nb[j][i] = 0;

	for(i=0;i<(N-M);i++)
		for(j=0;j<N;j++)
			G_nb[i][j] = 0;

	for(i=0;i<M;i++)
		for(j=0;j<N;j++)
			for(k=0;k<Q;k++)
				BETAmn[i][j][k] = ALPHAmn[i][j][k] = 0.0;

	for(i=0;i<Q;i++)
		for(j=0;j<N;j++)
			GAMMAn[i][j] = 0;


	for (i = 0; i < N; i++) {
		for (j = 0; j < col_weight[i]; j++) {
			fscanf(fp, "%d", &v);
			v--;
			row_col[j][i] = v;
		}
	}


	for(i = 0; i < N; i++){
		for(k = 1; k < col_weight[i] ; k++){
			for(j = 0; j < col_weight[i] - k; j++){
				if(row_col[j][i] > row_col[j+1][i]){
					v = row_col[j][i];
					row_col[j][i] = row_col[j+1][i];
					row_col[j+1][i] = v;
				}
			}
		}
	}

	#ifdef DEBUG
		for (i = 0; i < N; i++) {
			for (j = 0; j < col_weight[i]; j++){

				fprintf(frow_col,"%4d ", row_col[j][i]);
			}
			fprintf(frow_col, "\n");
		}
		fclose(frow_col);
	#endif


	for(i = 0; i < N; i++){
		for(j = 0; j < col_weight[i]; j++){
			v = row_col[j][i];
			tmp = 0;
			while(tmp == 0)
				tmp = rand()%Q;
			H_nb[v][i] = tmp;
		}
	}

	for(i = 0; i < M; i++){
		for(j = 0; j < row_weight[i]; j++){
			for(k = 0; k < N;k ++){
				if(H_nb[i][k] != 0){
					col_row[i][j] = k;
					j++;
				}
			}
		}
	}


	#ifdef DEBUG
		for (i = 0; i < M; i++) {
			for (j = 0; j < row_weight[i]; j++){
				fprintf(fcol_row,"%4d ", col_row[i][j]);
			}
			fprintf(fcol_row, "\n");
		}
		fclose(fcol_row);
	#endif

	#ifdef DEBUG
		for (i = 0; i < M; i++) {
			for (j = 0; j < N; j++){
				fprintf(fpp,"%4d ", H_nb[i][j]);
			}
			fprintf(fpp, "\n");
		}
		fclose(fpp);
	#endif

	fclose(fp);
	printf("non-binary decoder initialization finishes\n"); */

}

//#endif

//===================================
// H to G transform
// Using Gauss Jordan Elimination
// binary
//===================================
void h2g_nb(unsigned char **H, int n)
{
	int i, j, k, p, q, maxrow, scale;
	int tmp;
	unsigned char **H_;
	FILE *fp, *fpp;



#ifdef QC
	fp = fopen("data/qc.H'.nb.txt", "w+");
	fpp = fopen("data/qc.G'.nb.txt", "w+");
#else
	if (Q == 16)
	{
		fp = fopen("matrix/204.H'.nb.txt", "w+");
		fpp = fopen("matrix/204.G'.nb.txt", "w+");
	}
	else if (Q == 4)
	{
		fp = fopen("matrix/408.H'.nb.txt", "w+");
		fpp = fopen("matrix/408.G'.nb.txt", "w+");
	}
	else if (Q == 2)
	{
		fp = fopen("matrix/816.H'.nb.txt", "w+");
		fpp = fopen("matrix/816.G'.nb.txt", "w+");
	}
#endif

	H_ = malloc2Dunsigned_char(M, N);

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			H_[i][j] = H[i][j];
		}
	}

	for (j = 0; j < M; j++) {
		maxrow = j;
		for (i = j; i < M; i++) {	//find max entry in one colum
			if ((H_[i][j]) > (H_[maxrow][j]))
				maxrow = i;
		}

		for (k = j; k < n; k++) {	//swap the row with jth row
			tmp = H_[j][k];
			H_[j][k] = H_[maxrow][k];
			H_[maxrow][k] = tmp;
		}

		scale = inv_gf(H_[j][j]);
		//printf("H_[j][j] is %d and scale is %d\n",H_[j][j],scale);
		for (k = j; k < N; k++) {
			H_[j][k] = mul_gf(H_[j][k], scale);
		}

		for (q = 0; q < M; q++) {		//eliminate other entry within this colum
			if ((q != j) && (H_[q][j] != 0)) {
				tmp = H_[q][j];
				for (p = j; p < n; p++) {
					H_[q][p] = sub_gf(H_[q][p], mul_gf(H_[j][p], tmp));
				}
			}
		}
	}
	printf("eliminate finishes\n");

	for (i = 0; i < M; i++) {
		for (j = 0; j < n; j++) {
			fprintf(fp, "%4d ", H_[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	printf("H' written to file\n");

	for (i = 0; i < M; i++) //build identity matrix in G
		for (j = M; j < n; j++)
			if (j == (i + M)) G_nb[i][j] = 1;

	for (i = 0; i < M; i++)
		for (j = 0; j < M; j++)
			G_nb[i][j] = H_[j][i + M];

	for (i = 0; i < M; i++) {
		for (j = 0; j < n; j++) {
			fprintf(fpp, "%4d ", G_nb[i][j]);
		}
		fprintf(fpp, "\n");
	}
	fclose(fpp);



	printf("G written to file\n");




}

//===================================
// Ramdom info data generation
//===================================
void info_gen(int info_bin[])// radom number generation
{
	#ifdef DEBUG
		FILE *fpp = fopen("data/info.txt","w");
	#endif
		srand(time(0));
		int i ;
		for (i = 0 ; i <(N-M)*LOG2Q ; i++)
		{
			info_bin [i] = (rand())%2;
			//info_bin[i] = 1;
	#ifdef DEBUG
			fprintf(fpp, "%d\n",info_bin[i]);
	#endif
			printf("info_bin[%d]=%d\n", i, info_bin[i]);
		}
	#ifdef DEBUG
		fclose(fpp);
	#endif


	//======================================
	//read pre generated file

	/*FILE *file;

	printf("check 2\n");

	if (Q == 4)
		file = fopen("random_data/Q4/info.txt", "r");
	else if (Q == 8)
		file = fopen("random_data/Q8_extra/info.txt", "r");
	else if (Q == 16)
		file = fopen("random_data/Q16/info.txt", "r");
	else if (Q == 32)
		file = fopen("random_data/Q32/info.txt", "r");
	else if (Q == 64)
		file = fopen("random_data/Q64/info.txt", "r");
	else if (Q == 128)
		file = fopen("random_data/Q128/info.txt", "r");
	else if (Q == 256)
		file = fopen("random_data/Q256/info.txt", "r");

	//printf("check 2\n");


	char line[256];
	int d = 0;
	int i;

	//printf("check 2\n");

	while (fgets(line, sizeof(line), file)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		info_bin[d] = i;
		d++;
	}

	//printf("check 2\n");
	/* for (i = 0 ; i <(N-M)*LOG2Q ; i++){
		printf("info_bin[%d]=%d\n", i, info_bin[i]);
		//printf("check 2\n");
	} */

	//fclose(file);
}

//===================================
// Group bits into symbols
//===================================
void grouping(int info_bin[], int info_sym[])
{
	int i, j;
	int result, power;
#ifdef DEBUG
	FILE *fp = fopen("data/info_sym.nb.txt", "w+");
#endif
	for (i = 0; i < (N - M); i++)
	{
		result = 0;
		power = 1;
		for (j = LOG2Q - 1; j >= 0; j--)
		{
			if (info_bin[i  * LOG2Q + j] == 1) { result += power; }
			power *= 2;
		}
		info_sym[i] = result;
#ifdef DEBUG	
		fprintf(fp, "%d\n", info_sym[i]);
#endif
	}
#ifdef DEBUG
	fclose(fp);
#endif
}

//===================================
// Encoder nonbinary
//===================================
void encoder_nb(int info_sym[], unsigned char encoded_sym[])
{
	int i, j, xHmn, sum;
#ifdef DEBUG
	FILE *fp = fopen("data/encoded_sym.nb.txt", "w+");
#endif

	for (i = 0; i < N; i++) {
		sum = 0;
		for (j = 0; j < (N - M); j++) {
			xHmn = mul_gf(info_sym[j], G_nb[j][i]);
			sum = add_gf(sum, xHmn);
		}
		encoded_sym[i] = sum;
#ifdef DEBUG
		fprintf(fp, "%d\n", encoded_sym[i]);
#endif  
	}

#ifdef DEBUG
	fclose(fp);
#endif 
}

//===================================
// Ungroup symbols into bits
//===================================
void ungrouping(unsigned char info_sym[], int info_bin[])
{
	int i, j;
	int decimal, remain;
#ifdef DEBUG
	FILE *fp = fopen("data/info_bin_coded.txt", "w+");
#endif
	for (i = 0; i < N; i++)
	{
		decimal = info_sym[i];
		for (j = (LOG2Q - 1); j >= 0; j--)
		{
			remain = decimal % 2;
			decimal = decimal / 2;
			info_bin[i*LOG2Q + j] = remain;
		}
	}
#ifdef DEBUG
	for (i = 0; i < LOG2Q*N; i++) {
		fprintf(fp, "%d \n", info_bin[i]);
	}
	fclose(fp);
#endif
}

//===================================
// BPSK modulation
//===================================
void bpsk_modulation(int code[], double trans[])
{
	int i;
#ifdef DEBUG
	FILE *fp = fopen("data/trans.txt", "w");
#endif	
	for (i = 0; i < LOG2Q * N; i++) {
		if (code[i] == 1)
			trans[i] = 1;
		else
			trans[i] = -1;
#ifdef DEBUG
		fprintf(fp, "%lf\n", trans[i]);
#endif
	}
#ifdef DEBUG
	fclose(fp);
#endif
}

//===================================
// AWGN Channel
//===================================
void awgn(double *trans, double *rec, double sigma)
{
	double u1,u2,s,noise,randmum;
	int i;
	#ifdef DEBUG
		FILE *fp = fopen("data/recv.txt","w");
		printf("sigma is %f\n",sigma);
	#endif
	for (i=0; i< LOG2Q * N; i++)
	{
		do
		{
			randmum = (double)(rand())/RAND_MAX;
			u1 = randmum * 2.0 - 1.0;
			randmum = (double)(rand())/RAND_MAX;
			u2 = randmum * 2.0 - 1.0;
			s = u1 * u1 + u2 * u2;
		} while( s >= 1) ;
		noise = u1 * sqrt( (-2.0 * log(s))/s );
	#ifdef NONOISE
		rec[i] = trans[i];
	#else
		rec[i] = trans[i] + noise * sigma;
	#endif
	#ifdef DEBUG
			fprintf(fp,"%f \n",rec[i]);
	#endif
	}
	#ifdef DEBUG
		printf("awgn finishes\n");
		fclose(fp);
	#endif


	//======================================
	//read pre generated file

	/* FILE *file;

	if (Q == 4)
		file = fopen("random_data/Q4/recv.txt", "r");
	else if (Q == 8)
		file = fopen("random_data/Q8_extra/recv.txt", "r");
	else if (Q == 16)
		file = fopen("random_data/Q16/recv.txt", "r");
	else if (Q == 32)
		file = fopen("random_data/Q32/recv.txt", "r");
	else if (Q == 64)
		file = fopen("random_data/Q64/recv.txt", "r");
	else if (Q == 128)
		file = fopen("random_data/Q128/recv.txt", "r");
	else if (Q == 256)
		file = fopen("random_data/Q256/recv.txt", "r");

	char line[256];
	int d = 0;
	double i;

	while (fgets(line, sizeof(line), file)) {
		sscanf(line, "%lf", &i);
		//printf("%lf\n", i); 
		rec[d] = i;
		d++;
	}
	/* for (int x = 0 ; x <LOG2Q * N ; x++){
		printf("rec[%d]=%lf\n", x, rec[x]);
	}  */

	//fclose(file); 
}

//===================================
// Find the most likely symbols
//===================================
void hard_decision(double recv[], double sigmma, int hard[])
{
	int i;
#ifdef DEBUG
	FILE * fp = fopen("data/llr_fp.txt", "w");
#endif
	for (i = 0; i < LOG2Q * N; i++)
	{
		if (recv[i] > 0)	hard[i] = 1;
		else hard[i] = 0;
#ifdef DEBUG
		fprintf(fp, "recv[%d] = %lf, hard = %d\n", i, recv[i], hard[i]);
#endif
	}
#ifdef DEBUG
	fclose(fp);
#endif
}

//===================================
// calculate symbol LLRs
//===================================
void sym_llr(double* llr_, unsigned char** BETAmn_, unsigned char** GAMMAn_, int* hard, double* recv, double sigma)
{
	int i, j, k, l;
	int dev, remain;
	double temp;

	#ifdef DEBUG
		FILE *fp = fopen("data/symbol_llr.txt", "w");
	#endif

	// initialize channel information
	/*for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			if (H_nb[i][j] != 0) {
				for (k = 0; k < Q; k++) {
					dev = k;
					temp = 0;
					for (l = (LOG2Q - 1); l >= 0; l--) {
						remain = dev % 2;
						dev = dev / 2;

						int Log2Q = LOG2Q;
						if (remain == 1 && hard[j * Log2Q + l] == 0) {
							//BETAmn_[i][j][k] -= 2*recv[j * LOG2Q + l]/(sigma * sigm    a);
							temp -= 2 * recv[j * Log2Q + l] / (sigma * sigma);
						}
						else if (remain == 0 && hard[j*Log2Q + l] == 1) {
							//BETAmn_[i][j][k] += 2 * recv[j * LOG2Q + l] / (sigma * sigma);
							temp += 2 * recv[j * Log2Q + l] / (sigma * sigma);
						}
					}
					//GAMMAn[k][j] = BETAmn_[i][j][k];

					GAMMAn[k][j] = (unsigned char)(temp + 0.5);
					
				}
			}
		}
	}*/ 
	for (int row = 0; row < M; row++) {
		for (int z = row_ptr[row]; z < row_ptr[row+1]; z++) {
			for (k = 0; k < Q; k++) {
				dev = k;
				temp = 0;
				for (l = (LOG2Q - 1); l >= 0; l--) {
					remain = dev % 2;
					dev = dev / 2;

					int Log2Q = LOG2Q;
					if (remain == 1 && hard[col_ind[z] * Log2Q + l] == 0) {
						//BETAmn_[i][j][k] -= 2*recv[j * LOG2Q + l]/(sigma * sigm    a);
						temp -= 2 * recv[col_ind[z] * Log2Q + l] / (sigma * sigma);
					}
					else if (remain == 0 && hard[col_ind[j]*Log2Q + l] == 1) {
						//BETAmn_[i][j][k] += 2 * recv[j * LOG2Q + l] / (sigma * sigma);
						temp += 2 * recv[col_ind[z] * Log2Q + l] / (sigma * sigma);
					}
				}
				//GAMMAn[k][j] = BETAmn_[i][j][k];

				GAMMAn[k][col_ind[z]] = (unsigned char)(temp + 0.5);
				//printf("GAMMAn[%d][%d]=%d\n", k, col_ind[z], GAMMAn[k][col_ind[z]]);
			}
		}
		//printf("row=%d\n", row);

	}


	#ifdef DEBUG
		for (i = 0; i < N; i++) {
			for (j = 0; j < Q; j++) {
				fprintf(fp, "%d", GAMMAn[j][i]);
			}
		}
		fclose(fp);
	#endif
}

//===================================
// Convert to csr-csc format
//===================================
void H_to_csr_csc() {

	val = (unsigned char*)malloc(sizeof(unsigned char) * nnz);
	col_ind = (unsigned short*)malloc(sizeof(unsigned short) * nnz);
	row_ind = (unsigned short*)malloc(sizeof(unsigned short) * nnz);
	row_ptr = (unsigned short*)malloc(sizeof(unsigned short) * (M + 1));
	col_ptr = (unsigned short*)malloc(sizeof(unsigned short) * (N + 1));
	ptr_to_val = (unsigned short*)malloc(sizeof(unsigned short) * nnz);

	int z = 0;
	int row_flag = 0;
	int row_index = 0;
	int col_flag = 0;
	int col_index = 0;

	FILE *fp_val = fopen("data/val.txt", "w");
	FILE *fp_col_ind = fopen("data/col_ind.txt", "w");
	FILE *fp_row_ptr = fopen("data/row_ptr.txt", "w");
	FILE *fp_col_ptr = fopen("data/col_ptr.txt", "w");
	FILE *fp_ptr_to_val = fopen("data/ptr_to_val.txt", "w");
	FILE *fp_row_ind = fopen("data/row_ind.txt", "w");


	/* for(int j = 0; j < M; j++){
		for(int k = 0; k < N;k ++){
			if( H_nb[j][k]!=0)
			printf("H_nb[%d][%d]=%d\n", j, k, H_nb[j][k]);
		}
	} */

	for (int i = 0; i < M; i++) {
		row_flag = 0;
		for (int j = 0; j < N; j++) {
			if (H_nb[i][j] != 0) {
				if (row_flag == 0) {
					row_ptr[row_index] = z;
					fprintf(fp_row_ptr,"%d \n", row_ptr[row_index] );
					row_index++;
					row_flag = 1;
				}
				val[z] = H_nb[i][j];
				fprintf(fp_val,"%d \n", val[z] );
				col_ind[z] = j;
				fprintf(fp_col_ind,"%d \n", col_ind[z] );
				//if(i==0)
				//printf("i=%d, j=%d, col=%hu\n", i, j, val[z] );
				z++;

			}
			//printf("i=%d, j=%d\n", i, j );


		}
	}
	row_ptr[M] = nnz;
	fprintf(fp_row_ptr,"%d \n", row_ptr[M] );

	z = 0;
	int u=0;
	for (int i = 0; i < N; i++) {
		col_flag = 0;
		for (int j = 0; j < M; j++) {
			if (H_nb[j][i] != 0) {
				if (col_flag == 0) {
					col_ptr[col_index] = z;
					fprintf(fp_col_ptr,"%d \n", col_ptr[col_index] );
					col_index++;
					col_flag = 1;
				}
				row_ind[u] = j;
				fprintf(fp_row_ind, "%d\n" , row_ind[u]);
				u++;
				int p = 0;
				for (int k = 0; k < M; k++) {
					for (int l = 0; l < N; l++) {
						if (H_nb[k][l] != 0) {
							if (i == l && j == k) {
								ptr_to_val[z] = p;
								fprintf(fp_ptr_to_val,"%d \n",ptr_to_val[z] );
								break;
							}
							p++;
						}

					}
				}
				z++;
				//printf("H[%d][%d]=%hhu\n", j, i, H_nb[j][i] );
			}
		}
	}
	col_ptr[N] = nnz;
	fprintf(fp_col_ptr,"%d \n", col_ptr[N] );

	for (int i = 0; i < nnz; i++) {
		printf("row_ind[%d]=%hhu\n", i, row_ind[i] );
	}

	fclose(fp_val);
	fclose(fp_col_ind);
	fclose(fp_row_ptr);
	fclose(fp_col_ptr);
	fclose(fp_ptr_to_val);


}


//===================================
// Read CSR-CSC data format
//===================================
void read_data() {


	FILE *fp_val;
	FILE *fp_col_ind;
	FILE *fp_row_ptr;
	FILE *fp_col_ptr;
	FILE *fp_ptr_to_val;
	FILE *fp_row_ind;


	if (Q == 4){
		fp_val = fopen("random_data/Q4/val", "r");
		fp_col_ind = fopen("random_data/Q4/col_ind", "r");
		fp_row_ptr = fopen("random_data/Q4/row_ptr", "r");
		fp_col_ptr = fopen("random_data/Q4/col_ptr", "r");
		fp_ptr_to_val = fopen("random_data/Q4/ptr_to_val", "r");
		fp_row_ind = fopen("random_data/Q4/row_ind", "r");
	}


	else if (Q == 8){
		fp_val = fopen("random_data/Q8/val", "r");
		fp_col_ind = fopen("random_data/Q8/col_ind", "r");
		fp_row_ptr = fopen("random_data/Q8/row_ptr", "r");
		fp_col_ptr = fopen("random_data/Q8/col_ptr", "r");
		fp_ptr_to_val = fopen("random_data/Q8/ptr_to_val", "r");
		fp_row_ind = fopen("random_data/Q8/row_ind", "r");
	}
	else if (Q == 16){
		fp_val = fopen("random_data/Q16/val", "r");
		fp_col_ind = fopen("random_data/Q16/col_ind", "r");
		fp_row_ptr = fopen("random_data/Q16/row_ptr", "r");
		fp_col_ptr = fopen("random_data/Q16/col_ptr", "r");
		fp_ptr_to_val = fopen("random_data/Q16/ptr_to_val", "r");
		fp_row_ind = fopen("random_data/Q16/row_ind", "r");
	}
	else if (Q == 32){
		fp_val = fopen("random_data/Q32/val", "r");
		fp_col_ind = fopen("random_data/Q32/col_ind", "r");
		fp_row_ptr = fopen("random_data/Q32/row_ptr", "r");
		fp_col_ptr = fopen("random_data/Q32/col_ptr", "r");
		fp_ptr_to_val = fopen("random_data/Q32/ptr_to_val", "r");
		fp_row_ind = fopen("random_data/Q32/row_ind", "r");
	}
	else if (Q == 64){
		fp_val = fopen("random_data/Q64/val", "r");
		fp_col_ind = fopen("random_data/Q64/col_ind", "r");
		fp_row_ptr = fopen("random_data/Q64/row_ptr", "r");
		fp_col_ptr = fopen("random_data/Q64/col_ptr", "r");
		fp_ptr_to_val = fopen("random_data/Q64/ptr_to_val", "r");
		fp_row_ind = fopen("random_data/Q64/row_ind", "r");
	}
	else if (Q == 128){
		fp_val = fopen("random_data/Q128/val", "r");
		fp_col_ind = fopen("random_data/Q128/col_ind", "r");
		fp_row_ptr = fopen("random_data/Q128/row_ptr", "r");
		fp_col_ptr = fopen("random_data/Q128/col_ptr", "r");
		fp_ptr_to_val = fopen("random_data/Q128/ptr_to_val", "r");
		fp_row_ind = fopen("random_data/Q128/row_ind", "r");
	}
	else if (Q == 256){
		fp_val = fopen("random_data/Q256/val", "r");
		fp_col_ind = fopen("random_data/Q256/col_ind", "r");
		fp_row_ptr = fopen("random_data/Q256/row_ptr", "r");
		fp_col_ptr = fopen("random_data/Q256/col_ptr", "r");
		fp_ptr_to_val = fopen("random_data/Q256/ptr_to_val", "r");
		fp_row_ind = fopen("random_data/Q256/row_ind", "r");
	}


	char line[256];
	int d = 0;
	int u;
	unsigned int i;



	printf("params check\n"); 


	d=0;
	while (fgets(line, sizeof(line), fp_val)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		val[d] = i;
		d++;
	}

	printf("val check\n"); 


	d=0;
	while (fgets(line, sizeof(line), fp_col_ind)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		col_ind[d] = i;
		d++;
	}

	printf("col_ind check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_row_ptr)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		row_ptr[d] = i;
		d++;
	}

	printf("row_ptr check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_col_ptr)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		col_ptr[d] = i;
		d++;
	}

	printf("col_ptr check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_ptr_to_val)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		ptr_to_val[d] = i;
		d++;
	}

	printf("ptr_to_val check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_row_ind)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		row_ind[d] = i;
		d++;
	}

	printf("ptr_to_val check\n");

	for (i = 0; i < M; i++) {
		row_weight[i] = 3;
	}


	for (i = 0; i < N; i++) {
		col_weight[i] = 2;
	}
	printf("col_weight check\n");

	/* for (int x = 0 ; x < nnz+1 ; x++){
		printf("ptr_to_val[%d]=%d\n", x, col_ind[x]);
	} */  


	

	printf("files close\n");



	for (int z = 0; z < nnz; z++) {
		for (int k = 0; k < Q; k++) {
			BETAmn[z][k] = 0;
			ALPHAmn[z][k] = 0;
		}
		//printf("i=%d\n",z);
	}

	printf("alpha and beta init\n");

	GAMMAn = malloc2Dunsigned_char(Q, N);

	for (int z = 0; z < Q; z++){
		for (int j = 0; j < N; j++){
			GAMMAn[z][j] = 0;
		}
	}



	printf("read exit\n");

	fclose(fp_val);
	fclose(fp_col_ind);
	fclose(fp_row_ptr);
	fclose(fp_col_ptr);
	fclose(fp_ptr_to_val);

}



//===================================
// Read CSR-CSC data format
//===================================
void read_data2() {


	FILE *fp_params;
	FILE *fp_row_weight;
	FILE *fp_col_weight;
	FILE *fp_val;
	FILE *fp_col_ind;
	FILE *fp_row_ptr;
	FILE *fp_col_ptr;
	FILE *fp_ptr_to_val;
	FILE *fp_row_ind;
	FILE *fp_gnb;


	if (Q == 4){
		fp_params = fopen("random_data/Q4/params.txt", "r");
		fp_row_weight = fopen("random_data/Q4/row_weight.txt", "r");
		fp_col_weight = fopen("random_data/Q4/col_weight.txt", "r");
		fp_val = fopen("random_data/Q4/val.txt", "r");
		fp_col_ind = fopen("random_data/Q4/col_ind.txt", "r");
		fp_row_ptr = fopen("random_data/Q4/row_ptr.txt", "r");
		fp_col_ptr = fopen("random_data/Q4/col_ptr.txt", "r");
		fp_ptr_to_val = fopen("random_data/Q4/ptr_to_val.txt", "r");
		fp_row_ind = fopen("random_data/Q4/row_ind.txt", "r");
		fp_gnb = fopen("random_data/Q4/gnb.txt", "r");
	}
	else if (Q == 8){
		fp_params = fopen("random_data/Q8/params.txt", "r");
		fp_row_weight = fopen("random_data/Q8/row_weight.txt", "r");
		fp_col_weight = fopen("random_data/Q8/col_weight.txt", "r");
		fp_val = fopen("random_data/Q8/val.txt", "r");
		fp_col_ind = fopen("random_data/Q8/col_ind.txt", "r");
		fp_row_ptr = fopen("random_data/Q8/row_ptr.txt", "r");
		fp_col_ptr = fopen("random_data/Q8/col_ptr.txt", "r");
		fp_ptr_to_val = fopen("random_data/Q8/ptr_to_val.txt", "r");
		fp_row_ind = fopen("random_data/Q8/row_ind.txt", "r");
		fp_gnb = fopen("random_data/Q8/gnb.txt", "r");
	}
	else if (Q == 16){
		fp_params = fopen("random_data/Q16/params.txt", "r");
		fp_row_weight = fopen("random_data/Q16/row_weight.txt", "r");
		fp_col_weight = fopen("random_data/Q16/col_weight.txt", "r");
		fp_val = fopen("random_data/Q16/val.txt", "r");
		fp_col_ind = fopen("random_data/Q16/col_ind.txt", "r");
		fp_row_ptr = fopen("random_data/Q16/row_ptr.txt", "r");
		fp_col_ptr = fopen("random_data/Q16/col_ptr.txt", "r");
		fp_ptr_to_val = fopen("random_data/Q16/ptr_to_val.txt", "r");
		fp_row_ind = fopen("random_data/Q16/row_ind.txt", "r");
		fp_gnb = fopen("random_data/Q16/gnb.txt", "r");
	}
	else if (Q == 32){
		fp_params = fopen("random_data/Q32/params.txt", "r");
		fp_row_weight = fopen("random_data/Q32/row_weight.txt", "r");
		fp_col_weight = fopen("random_data/Q32/col_weight.txt", "r");
		fp_val = fopen("random_data/Q32/val.txt", "r");
		fp_col_ind = fopen("random_data/Q32/col_ind.txt", "r");
		fp_row_ptr = fopen("random_data/Q32/row_ptr.txt", "r");
		fp_col_ptr = fopen("random_data/Q32/col_ptr.txt", "r");
		fp_ptr_to_val = fopen("random_data/Q32/ptr_to_val.txt", "r");
		fp_row_ind = fopen("random_data/Q32/row_ind.txt", "r");
		fp_gnb = fopen("random_data/Q32/gnb.txt", "r");
	}
	else if (Q == 64){
		fp_params = fopen("random_data/Q64/params.txt", "r");
		fp_row_weight = fopen("random_data/Q64/row_weight.txt", "r");
		fp_col_weight = fopen("random_data/Q64/col_weight.txt", "r");
		fp_val = fopen("random_data/Q64/val.txt", "r");
		fp_col_ind = fopen("random_data/Q64/col_ind.txt", "r");
		fp_row_ptr = fopen("random_data/Q64/row_ptr.txt", "r");
		fp_col_ptr = fopen("random_data/Q64/col_ptr.txt", "r");
		fp_ptr_to_val = fopen("random_data/Q64/ptr_to_val.txt", "r");
		fp_row_ind = fopen("random_data/Q64/row_ind.txt", "r");
		fp_gnb = fopen("random_data/Q64/gnb.txt", "r");
	}
	else if (Q == 128){
		fp_params = fopen("random_data/Q128/params.txt", "r");
		fp_row_weight = fopen("random_data/Q128/row_weight.txt", "r");
		fp_col_weight = fopen("random_data/Q128/col_weight.txt", "r");
		fp_val = fopen("random_data/Q128/val.txt", "r");
		fp_col_ind = fopen("random_data/Q128/col_ind.txt", "r");
		fp_row_ptr = fopen("random_data/Q128/row_ptr.txt", "r");
		fp_col_ptr = fopen("random_data/Q128/col_ptr.txt", "r");
		fp_ptr_to_val = fopen("random_data/Q128/ptr_to_val.txt", "r");
		fp_row_ind = fopen("random_data/Q128/row_ind.txt", "r");
		fp_gnb = fopen("random_data/Q128/gnb.txt", "r");
	}
	else if (Q == 256){
		fp_params = fopen("random_data/Q256/params.txt", "r");
		fp_row_weight = fopen("random_data/Q256/row_weight.txt", "r");
		fp_col_weight = fopen("random_data/Q256/col_weight.txt", "r");
		fp_val = fopen("random_data/Q256/val.txt", "r");
		fp_col_ind = fopen("random_data/Q256/col_ind.txt", "r");
		fp_row_ptr = fopen("random_data/Q256/row_ptr.txt", "r");
		fp_col_ptr = fopen("random_data/Q256/col_ptr.txt", "r");
		fp_ptr_to_val = fopen("random_data/Q256/ptr_to_val.txt", "r");
		fp_row_ind = fopen("random_data/Q256/row_ind.txt", "r");
		fp_gnb = fopen("random_data/Q256/gnb.txt", "r");
	}


	char line[256];
	int d = 0;
	int u;
	unsigned int i;

	GAMMAn = malloc2Dunsigned_char(Q, N);

	for (int z = 0; z < Q; z++){
		for (int j = 0; j < N; j++){
			GAMMAn[z][j] = 0;
		}
	}

	while (fgets(line, sizeof(line), fp_params)) {
		sscanf(line, "%d", &u);
		//printf("params %d\n", u); 
		if(d==0){
			M = u;
		}
		else if(d==1){
			N = u;
		}
		else if(d==2){
			dc = u;
		}
		else if(d==3){
			dv = u;
		}
		else if(d==4){
			nnz = u;
		}
		d++;
	}

	printf("params check\n"); 

	val = (unsigned char*)malloc(sizeof(unsigned char) * nnz);
	col_ind = (unsigned short*)malloc(sizeof(unsigned short) * nnz);
	row_ind = (unsigned short*)malloc(sizeof(unsigned short) * nnz);
	row_ptr = (unsigned short*)malloc(sizeof(unsigned short) * (M + 1));
	col_ptr = (unsigned short*)malloc(sizeof(unsigned short) * (N + 1));
	ptr_to_val = (unsigned short*)malloc(sizeof(unsigned short) * nnz);
	row_weight = (unsigned short*)malloc(sizeof(unsigned short) * M);	//row weight table
	col_weight = (unsigned short*)malloc(sizeof(unsigned short) * N);
	G_nb = malloc2Dint(M, N);

	d=0;
	while (fgets(line, sizeof(line), fp_val)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		val[d] = i;
		d++;
	}

	printf("val check\n"); 


	d=0;
	while (fgets(line, sizeof(line), fp_col_ind)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		col_ind[d] = i;
		d++;
	}

	printf("col_ind check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_row_ptr)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		row_ptr[d] = i;
		d++;
	}

	printf("row_ptr check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_col_ptr)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		col_ptr[d] = i;
		d++;
	}

	printf("col_ptr check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_ptr_to_val)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		ptr_to_val[d] = i;
		d++;
	}

	printf("ptr_to_val check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_row_ind)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		row_ind[d] = i;
		d++;
	}

	printf("ptr_to_val check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_row_weight)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		row_weight[d] = i;
		d++;
	}

	printf("row_weight check\n");

	d=0;
	while (fgets(line, sizeof(line), fp_col_weight)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		col_weight[d] = i;
		d++;
	}

	printf("col_weight check\n");

	/* for (int x = 0 ; x < nnz+1 ; x++){
		printf("ptr_to_val[%d]=%d\n", x, col_ind[x]);
	} */  


	

	printf("files close\n");

	BETAmn = malloc2Dunsigned_char(nnz, Q);
	ALPHAmn = malloc2Dunsigned_char(nnz, Q);


	for (int z = 0; z < nnz; z++) {
		for (int k = 0; k < Q; k++) {
			BETAmn[z][k] = 0;
			ALPHAmn[z][k] = 0;
		}
		//printf("i=%d\n",z);
	}

	printf("alpha and beta init\n");

	int z=0;
	d=0;
	while (fgets(line, sizeof(line), fp_gnb)) {
		sscanf(line, "%d", &i);
		//printf("%d\n", i); 
		G_nb[z][d] = i;
		d++;
		if(d>=N){
			d=0;
			z++;
		}
	}



	printf("read exit\n");

	fclose(fp_params);
	fclose(fp_row_weight);
	fclose(fp_col_weight);
	fclose(fp_val);
	fclose(fp_col_ind);
	fclose(fp_row_ptr);
	fclose(fp_col_ptr);
	fclose(fp_ptr_to_val);
	fclose(fp_gnb);

}


//===================================
// Read CSR-CSC data format
//===================================
void read_llr() {

	FILE *fp_gamma;



	if (Q == 4){
		fp_gamma = fopen("random_data/Q4/symbol_llr.txt", "r");
	}
	else if (Q == 8){
		fp_gamma = fopen("random_data/Q8/symbol_llr.txt", "r");
	}
	else if (Q == 16){
		fp_gamma = fopen("random_data/Q16/symbol_llr.txt", "r");
	}
	else if (Q == 32){
		fp_gamma = fopen("random_data/Q32/symbol_llr.txt", "r");
	}
	else if (Q == 64){
		fp_gamma = fopen("random_data/Q64/symbol_llr.txt", "r");
	}
	else if (Q == 128){
		fp_gamma = fopen("random_data/Q128/symbol_llr.txt", "r");
	}
	else if (Q == 256){
		fp_gamma = fopen("random_data/Q256/symbol_llr.txt", "r");
	}


	char line[256];
	int d = 0;
	int u;
	unsigned int i;


	d=0;
	int z=0;
	while (fgets(line, sizeof(line), fp_gamma)) {
		sscanf(line, "%d", &i);
		GAMMAn[d][z] = i;
		//printf("GAMMAn[%d][%d] = %d\n", d, z, GAMMAn[d][z]); 
		d++;
		if (d>=Q){
			d = 0;
			z++;
		}
	}


	printf("read exit\n");

	fclose(fp_gamma);


}

//===================================
// Main Function
//===================================

int main(int argc, char*argv[])
{


	#if (RUN_ONCE == 1)
	#else
		int 	fer_threshold = 100;
	#endif

	double 	snr;
	double 	sigma;
	int 	iteration;
	int 	error = 0;
	int 	error_num = 0;
	int 	counter = 0;
	int 	fe = 0;


	//sigma = 1 / sqrt(2.0*rate*pow(10.0, (snr / 10.0)));

	counter = 0;
	error_num = 0;
	fe = 0;

	double total_time = 0.0;

	M = 256;
	N = 384;

	dc = 2;
	dv = 3;
	nnz=768;



	row_weight = (unsigned short*)malloc(sizeof(unsigned short) * M);	//row weight table
	col_weight = (unsigned short*)malloc(sizeof(unsigned short) * N);	//colum weight table



	H_nb = malloc2Dunsigned_char(M, N);
	G_nb = malloc2Dint(N - M, N);
	BETAmn = malloc2Dunsigned_char(nnz, Q);
	ALPHAmn = malloc2Dunsigned_char(nnz, Q);
	GAMMAn = malloc2Dunsigned_char(Q, N);
	encoded_sym = (unsigned char *)malloc(N * sizeof(unsigned char));
	
	val = (unsigned char*)malloc(sizeof(unsigned char) * nnz);
	col_ind = (unsigned short*)malloc(sizeof(unsigned short) * nnz);
	row_ind = (unsigned short*)malloc(sizeof(unsigned short) * nnz);
	row_ptr = (unsigned short*)malloc(sizeof(unsigned short) * (M + 1));
	col_ptr = (unsigned short*)malloc(sizeof(unsigned short) * (N + 1));
	ptr_to_val = (unsigned short*)malloc(sizeof(unsigned short) * nnz);


	read_data();
	printf("check\n");

	FILE *fp_times = fopen("data/times.txt", "a+");

		
	read_llr();

	error = minmax(BETAmn, ALPHAmn, GAMMAn, &iteration);

	printf(" time=%lf ms\n",((end.tv_sec-start.tv_sec)*1e3)+((end.tv_nsec-start.tv_nsec)*1e-6));
	fprintf(fp_times,"%lf\n",((end.tv_sec-start.tv_sec)*1e3)+((end.tv_nsec-start.tv_nsec)*1e-6));
	fprintf(stderr, "throughput is %f Kbps\n", counter * 310 * 5 / total_time);

	fclose(fp_times);
	printf("check\n");

	printf("SER is %f at SNR = %f\n", (double)error_num / (counter*(N - M)), snr);
	printf("FER is %f at SNR = %f\n", (double)fe / (counter), snr);
	printf("SNR = %f is finished\n", snr);
		
	


	free(val);
	free(col_ind);
	free(row_ind);
	free(row_ptr);
	free(col_ptr);
	free(ptr_to_val);
	free(row_weight);
	free(col_weight);
	free(encoded_sym); 


	return 1;
}