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


int		N;
int		M;
int		dc,dv;
int		**H_qc;
int		**G_qc;
int		**H_nb;
int		**G_nb;
int		*row_weight;
int		*col_weight;
int		**row_col;
int		**col_row;
double	***BETAmn;
double	***ALPHAmn;
double	**GAMMAn;
int		*encoded_sym;
double	**GAMMAn_post;

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
		pp[i] = p + b*i;
	}
	return pp;
}



double ***malloc3Ddouble(int a, int b, int c)
{
	int i,j;
	double*** ppp = (double ***)malloc(sizeof(double **) * M);
	if (ppp == NULL) exit(-1);
	for(i = 0;i < a;i++)
	{
		ppp[i] = (double **)malloc(sizeof(double *) * b);
		for(j = 0;j < b;j++)
			ppp[i][j] = (double *)malloc(sizeof(double) * c); 
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
	int i,j,k,x,x_index,m,n;
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
	int K,L,S,T;
	int row_w;
	int col_w;
	FILE *fp = fopen("data/qc.txt","w+");
	FILE *fp1 = fopen("data/row_col.csv", "w+");
	FILE *fp2 = fopen("data/H_nb.csv", "w+");
	FILE *fp3 = fopen("data/row_weight.csv", "w+");
	FILE *fp4 = fopen("data/col_weight.csv", "w+");
	FILE *fp5 = fopen("data/col_row.csv", "w+");

	K = 2;
	L = 2;
	S = 2;
	T = 5;

	/*K = 2;
	L = 1;
	S = 0;
	T = 1;*/

	M = (Q - 1) * T * L;
	N = (Q - 1) * T * L * K;

	dc = T - S;
	dv = K * (T - S);

	W = malloc2Dint((Q - 1),(Q - 1));
	H_tmp = malloc2Dint((Q - 1) * (Q - 1),(Q - 1) * (Q - 1));
	one_row = (int*)malloc(sizeof(int) * (Q - 1));
	col_ext_matrix = malloc2Dint((Q-1) * (Q - 1),(Q - 1));
	Hj_up = malloc2Dint(T * (Q - 1),T * (Q - 1));
	Hj_low = malloc2Dint(T * (Q - 1),T * (Q - 1));
	Hj = malloc2Dint(T * (Q - 1),T * (Q - 1));
	H_disp = malloc2Dint(L * T * (Q - 1),L * T * (Q - 1));
	H_disp_loc_orig = (int*)malloc(sizeof(int) * L);
	H_disp_loc_current = (int*)malloc(sizeof(int) * L);

	H_disp_mask_loc_orig = (int*)malloc(sizeof(int) * L * T);
	H_disp_mask_loc_current = (int*)malloc(sizeof(int) * L * T);
	H = malloc2Dint((Q - 1) * T, K * (Q - 1) * T);

	row_weight = (int*)malloc(sizeof(int) * M);	//row weight table
	col_weight = (int*)malloc(sizeof(int) * N);	//colum weight table

	row_col = malloc2Dint(dc, N); 
	col_row = malloc2Dint(M, dv);
	H_nb	= malloc2Dint(M,N);     
	G_nb	= malloc2Dint(N - M,N);
	BETAmn	= malloc3Ddouble(M,N,Q);
	ALPHAmn	= malloc3Ddouble(M,N,Q);
	GAMMAn	= malloc2Ddouble(Q,N);

	for(i = 0; i < M; i++){
		row_weight[i] = 0;
	}

	for(i = 0; i < N; i++){
		col_weight[i] = 0;
	}

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++){
			for(k = 0; k < Q; k++){
				BETAmn[i][j][k] = 0;
				ALPHAmn[i][j][k] = 0;
			}
		}
	}

	for(i = 0; i < Q; i++){
		for(j = 0; j < N; j++){
			GAMMAn[i][j] = 0;
		}
	}

	for(i = 0; i < dc; i++){
		for(j = 0; j < N; j++){
			row_col[i][j] = 0;
		}
	}

	for(i = 0; i < M; i++){
		for(j = 0; j < dv; j++){
			col_row[i][j] = 0;
		}
	}

	for(i = 0;i < (Q - 1) * T;i++){
		for(j = 0;j < K * (Q - 1) * T;j++){
			H[i][j] = 0;
		}
	}

	for(i = 0;i < M;i++){
		for(j = 0;j < N;j++){
			H_nb[i][j] = 0;
		}
	}

	for(i = 0;i < N - M;i++){
		for(j = 0;j < N;j++){
			G_nb[i][j] = 0;
		}
	}

	for(i = 0;i < (Q - 1);i++){
		for(j = 0;j < (Q - 1);j++){
			W[i][j] = 0;
		}
	}

	for(i = 0;i < (Q - 1) * (Q - 1);i++){
		for(j = 0;j < (Q - 1) * (Q - 1);j++){
			H_tmp[i][j] = 0;
		}
	}


	for(i = 0;i < T * (Q - 1);i++){
		for(j = 0;j < T * (Q - 1);j++){
			Hj_up[i][j] = Hj_low[i][j] = Hj[i][j] = 0;
		}
	}

	for(i = 0;i < L * T * (Q - 1);i++){
		for(j = 0;j < L * T * (Q - 1);j++){
			H_disp[i][j] = 0;
		}
	}

	for(i = 0; i < L; i++){
		if(i == 0){
			H_disp_loc_orig[i] = 1;
		}
		else if(i == (L - 1)){
			H_disp_loc_orig[i] = -1;
		}
		else{
			H_disp_loc_orig[i] = 0;
		}
	}

	for(i = 0; i < L * T; i++){
		if(i == 0){
			H_disp_mask_loc_orig[i] = 1;
		}
		else if (i >= (T + S + 1)){
			H_disp_mask_loc_orig[i] = 1;
		}
		else{
			H_disp_mask_loc_orig[i] = 0;
		}
		fprintf(fp,"%2d ",H_disp_mask_loc_orig[i]);
	}

	fprintf(fp,"\n\n\n\n");

	for(i = 0;i < (Q - 1);i++){
		one_row[i] = expq[i];
	}

	for(i = 0;i < (Q - 1);i++){
		for(j = 0;j < (Q - 1);j++){
			W[i][j] = one_row[((j - i) + (Q - 1)) % (Q - 1)] - 1;
			fprintf(fp,"%2d ",W[i][j]);
		}
		fprintf(fp,"\n");
	}

	fprintf(fp,"\n\n\n\n");


	for(i = 0;i < (Q - 1) * (Q - 1);i++){
		x = i;
		x_index = 0;
		while(x >= (Q - 1)){
			x -= (Q - 1);
			x_index ++;
		}
		for(j = 0;j < (Q - 1);j++){
			col_ext_matrix[i][j] = mul_gf(W[x_index][j],one_row[x]);
			fprintf(fp,"%2d ",col_ext_matrix[i][j]);
		}
		fprintf(fp,"\n");
	}

	fprintf(fp,"\n");
	fprintf(fp,"\n");
	fprintf(fp,"\n");
	fprintf(fp,"\n");

	fprintf(fp,"hello\n");

	for(i = 0;i < (Q - 1) * (Q - 1);i++){
		for(j = 0;j < (Q - 1);j++){
			k = 0;
			while(one_row[k] != col_ext_matrix[i][j]){
				k++;
			}
			H_tmp[i][j * (Q - 1) + k] =  col_ext_matrix[i][j];
		}
	}

	for(i = 0; i < (Q - 1) * (Q - 1); i++){
		for(j = 0; j < (Q - 1) * (Q - 1); j++){
			fprintf(fp,"%2d ", H_tmp[i][j]);
		}
		fprintf(fp,"\n");
	}

	fprintf(fp,"\n\n\n\n");

	for(i = 0; i < (Q - 1) * T; i++){
		for(j = 0; j < K * (Q - 1) * T; j++){
			H[i][j] = H_tmp[(Q - 1) * (Q - 1) - (Q - 1) * T + i][j];
			fprintf(fp,"%2d ", H[i][j]);
		}
		fprintf(fp,"\n");
	}

	fprintf(fp,"\n\n\n\n");

	for(k = 0; k < K; k++){
		fprintf(fp,"Hj\n");
		for(i = 0; i < T * (Q - 1); i++){
			for(j = 0; j < T * (Q - 1); j++){
				Hj[i][j] = H[i][j + (Q-1) * k * T];
				fprintf(fp,"%2d ", Hj[i][j]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n\n\n\n");

		for(i = 0; i < T * (Q - 1); i++){
			for(j = 0; j < T * (Q - 1); j++){
				if(j > (i + (Q - 1))){
					Hj_up[i][j] = Hj[i][j];
					Hj_low[i][j] = 0; 
				}
				else{
					Hj_low[i][j] = Hj[i][j];
					Hj_up[i][j] = 0;
				}
			}
		}

		fprintf(fp,"Hj_low\n");
		for(i = 0; i < T * (Q - 1); i++){
			for(j = 0; j < T * (Q - 1); j++){
				fprintf(fp,"%2d ", Hj_low[i][j]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n\n\n\n");
		fprintf(fp,"Hj_up\n");
		for(i = 0; i < T * (Q - 1); i++){
			for(j = 0; j < T * (Q - 1); j++){
				fprintf(fp,"%2d ", Hj_up[i][j]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n\n\n\n");


		fprintf(fp,"\n\n\n\n");


		for(m = 0; m < L; m++){
			for(n = 0; n < L; n++){
				H_disp_loc_current[n] = H_disp_loc_orig[((n - m) + L) % L];
				fprintf(fp,"%2d ", H_disp_loc_current[n]);
			}
			for(n = 0; n < L; n++){
				if(H_disp_loc_current[n] == 1){
					for(i = 0 + m * T * (Q - 1); i < T * (Q - 1) + m * T * (Q - 1); i++){
						for(j = 0 + n * T * (Q - 1); j < T * (Q - 1) + n * T * (Q - 1); j++){
							H_disp[i][j] = Hj_low[i - m * T * (Q - 1)][j - n * T * (Q - 1)];
						}
					}
				}
				else if(H_disp_loc_current[n] == -1){
					for(i = 0 + m * T * (Q - 1); i < T * (Q - 1) + m * T * (Q - 1); i++){
						for(j = 0 + n * T * (Q - 1); j < T * (Q - 1) + n * T * (Q - 1); j++){
							H_disp[i][j] = Hj_up[i - m * T * (Q - 1)][j - n * T * (Q - 1)];
						}
					}
				}
				else{
					for(i = 0 + m * T * (Q - 1); i < T * (Q - 1) + m * T * (Q - 1); i++){
						for(j = 0 + n * T * (Q - 1); j < T * (Q - 1) + n * T * (Q - 1); j++){
							H_disp[i][j] = 0;
						}
					}
				}
			}
		}

		for(i = 0;i < L * T * (Q - 1);i++){
			for(j = 0;j < L * T * (Q - 1);j++){
				fprintf(fp,"%2d ", H_disp[i][j]);
			}
			fprintf(fp,"\n");
		}

		for(m = 0; m < L * T; m++){
			for(n = 0; n < L * T; n++){
				H_disp_mask_loc_current[n] = H_disp_mask_loc_orig[((n - m) + L * T) % (L * T)];
				if(H_disp_mask_loc_current[n] == 0){
					for(i = m * (Q - 1); i < (m + 1) * (Q - 1);i++){
						for(j = n * (Q - 1); j < (n + 1) * (Q - 1); j++){
							H_disp[i][j] = 0;
						}
					}
				}
			}
		}

		for(i = 0; i < L * T * (Q - 1); i++){
			for(j = 0; j < L * T * (Q - 1); j++){
				H_nb[i][j + k * L * T * (Q - 1)] = H_disp[i][j];
			}
		}
	}

	fprintf(fp,"\n\n\n\n");
	fprintf(fp,"\n\n\n\n");

	fprintf(fp2, "sep=,\n");

	for(i = 0;i < M;i++){
		for(j = 0;j < N;j++){
			fprintf(fp,"%2d ", H_nb[i][j]);
			fprintf(fp2, "%d,", H_nb[i][j]);
		}
		fprintf(fp,"\n");
		fprintf(fp2, "\n");
		
	}
	fclose(fp2);

	for(i = 0;i < M;i++){
		row_w = 0;
		for(j = 0;j < N;j++){
			if(H_nb[i][j] != 0){
				row_w++;
			}
		}
		fprintf(fp,"row_w[%d] = %d\n", i , row_w);
	}

	for(j = 0;j < N;j++){
		col_w = 0;
		for(i = 0;i < M;i++){
			if(H_nb[i][j] != 0){
				col_w++;
			}
		}
		fprintf(fp,"col_w[%d] = %d\n", j , col_w);
	}

	fprintf(fp3, "sep=,\n");
	for(i = 0;i < M;i++){
		for(j = 0;j < N;j++){
			if(H_nb[i][j] != 0){
				row_weight[i]++;
			}
		}
		fprintf(fp3,"%d,\n", row_weight[i]);
	}
	fclose(fp3);

	fprintf(fp4, "sep=,\n");
	for(j = 0;j < N;j++){
		for(i = 0;i < M;i++){
			if(H_nb[i][j] != 0){
				col_weight[j]++;
			}
		}
		fprintf(fp4,"%d,\n", col_weight[j]);
	}
	fclose(fp4);

	fprintf(fp5, "sep=,\n");
	for(i = 0; i < M; i++){
		for(j = 0; j < row_weight[i]; j++){
			for(k = 0; k < N;k ++){
				if(H_nb[i][k] != 0){
					col_row[i][j] = k;
					fprintf(fp5,"%d,", col_row[i][j]);
					j++;
				}
			}
		}
		fprintf(fp5,"\n");
	}
	fclose(fp5);

	fprintf(fp1, "sep=,\n");

	for(j = 0; j < N; j++){
		for(i = 0; i < col_weight[j]; i++){
			for(k = 0; k < M;k ++){
				if(H_nb[k][j] != 0){
					row_col[i][j] = k;
					fprintf(fp,"%d ", row_col[i][j]);
					fprintf(fp1, "%d,", row_col[i][j]);
					i++;
				}
			}
		}
		fprintf(fp,"\n");
		fprintf(fp1, "\n");
	}
	fclose(fp1);


	fprintf(fp,"\n\n\n\n");

	fclose(fp);
}

void dec_init_nb_small()
{
	int i,j,k,tmp;
	int v;

	//FILE *fp;
	FILE *fpp;
	FILE *frow_col;
	//FILE *fcol_row;


	printf("N, M and are %d %d \n",N,M);
	printf("dc and dv are %d %d\n",dc,dv);

	row_weight = (int*)malloc(sizeof(int) * M);	//row weight table
	col_weight = (int*)malloc(sizeof(int) * N);	//colum weight table

	row_col = malloc2Dint(dc, N); 
	col_row = malloc2Dint(M, dv);
	H_nb	= malloc2Dint(M,N);     
	G_nb	= malloc2Dint(N-M,N);
	BETAmn	= malloc3Ddouble(M,N,Q);
	ALPHAmn	= malloc3Ddouble(M,N,Q);
	GAMMAn	= malloc2Ddouble(Q,N);

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
			GAMMAn[i][j] = 0.0;
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

	printf("non-binary decoder small initilization finishes\n");

}


//===================================
// Generate Random Parity Check Matrix
// Refer to http://www.inference.phy.cam.ac.uk/mackay/codes/data.html
//===================================
void dec_init_nb()
{
	int i,j,k,tmp;
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

	sscanf(buf, "%d%d%d",&N,&M);
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
	BETAmn	= malloc3Ddouble(M,N,Q);
	ALPHAmn	= malloc3Ddouble(M,N,Q);
	GAMMAn	= malloc2Ddouble(Q,N);

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
			GAMMAn[i][j] = 0.0;


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
	printf("non-binary decoder initialization finishes\n");

}

//#endif

//===================================
// H to G transform
// Using Gauss Jordan Elimination
// binary
//===================================
void h2g_nb(int **H, int n)
{
	int i,j,k,p,q,maxrow,scale;
	int tmp;
	int **H_;
	FILE *fp,*fpp;

#ifdef QC
	fp = fopen("data/qc.H'.nb.txt","w+");
	fpp = fopen("data/qc.G'.nb.txt","w+");
#else
	if(Q==16)
	{
		fp = fopen("matrix/204.H'.nb.txt","w+");
		fpp = fopen("matrix/204.G'.nb.txt","w+");
	}
	else if(Q==4)
	{
		fp = fopen("matrix/408.H'.nb.txt","w+");
		fpp = fopen("matrix/408.G'.nb.txt","w+");
	}
	else if(Q==2)
	{
		fp = fopen("matrix/816.H'.nb.txt","w+");
		fpp = fopen("matrix/816.G'.nb.txt","w+");
	}
#endif

	H_ = malloc2Dint(M,N);

	for(i = 0; i < M; i++){
		for(j = 0; j < N; j++){
			H_[i][j] = H[i][j];
		}
	}

	for(j = 0;j < M; j++){
		maxrow = j;
		for(i = j; i < M; i++){	//find max entry in one colum
			if((H_[i][j]) > (H_[maxrow][j])) 
				maxrow = i;
		}

		for(k = j; k < n; k++){	//swap the row with jth row
			tmp = H_[j][k];
			H_[j][k] = H_[maxrow][k];
			H_[maxrow][k] = tmp;
		}

		scale = inv_gf(H_[j][j]);
		//printf("H_[j][j] is %d and scale is %d\n",H_[j][j],scale);
		for(k = j; k< N; k++){
			H_[j][k] = mul_gf(H_[j][k],scale);
		}

		for(q = 0; q < M; q++){		//eliminate other entry within this colum
			if((q != j)&&(H_[q][j]!=0)){
				tmp = H_[q][j];
				for(p = j; p < n; p++){
					H_[q][p] = sub_gf(H_[q][p],mul_gf(H_[j][p],tmp)); 
				}
			}
		}
	}
	printf("eliminate finishes\n");

	for (i = 0; i < M; i++) {
		for (j = 0; j < n; j++){
			fprintf(fp,"%4d ", H_[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	printf("H' written to file\n");

	for(i= 0;i < M;i++) //build identity matrix in G
		for(j = M;j<n;j++)
			if(j==(i+M)) G_nb[i][j] = 1;

	for(i=0;i < M;i++)
		for(j=0;j < M;j++)
			G_nb[i][j] = H_[j][i+M];

	for (i = 0; i < M; i++) {
		for (j = 0; j < n; j++){
			fprintf(fpp,"%4d ", G_nb[i][j]);
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
	int i ;
	for (i = 0 ; i <(N-M)*LOG2Q ; i++)
	{
		info_bin [i] = (rand())%2;
		//info_bin[i] = 1;
#ifdef DEBUG
		fprintf(fpp, "%d\n",info_bin[i]);
#endif
	}
#ifdef DEBUG
	fclose(fpp);
#endif
}

//===================================
// Group bits into symbols
//===================================
void grouping(int info_bin[], int info_sym[])
{
	int i,j;
	int result,power;
#ifdef DEBUG
	FILE *fp = fopen("data/info_sym.nb.txt","w+");
#endif
	for(i = 0; i < (N-M); i++)
	{
		result = 0;
		power = 1;
		for(j = LOG2Q-1; j >= 0; j--)
		{
			if(info_bin[i  * LOG2Q + j] == 1) {result += power;}
			power *= 2;
		}
		info_sym[i] = result;
#ifdef DEBUG	
		fprintf(fp,"%d\n",info_sym[i]);
#endif
	}
#ifdef DEBUG
	fclose(fp);
#endif
}

//===================================
// Encoder nonbinary
//===================================
void encoder_nb(int info_sym[], int encoded_sym[])
{
	int i, j, xHmn, sum;
#ifdef DEBUG
	FILE *fp = fopen("data/encoded_sym.nb.txt","w+");
#endif

	for (i = 0; i < N; i++) {
		sum = 0;
		for (j = 0; j < (N-M); j++) {
			xHmn = mul_gf(info_sym[j],G_nb[j][i]);
			sum = add_gf(sum, xHmn);
		}
		encoded_sym[i] = sum;
#ifdef DEBUG
		fprintf(fp,"%d\n",encoded_sym[i]);
#endif  
	}

#ifdef DEBUG
	fclose(fp);
#endif 
}

//===================================
// Ungroup symbols into bits
//===================================
void ungrouping(int info_sym[], int info_bin[])
{
	int i,j;
	int decimal,remain;
#ifdef DEBUG
	FILE *fp = fopen("data/info_bin_coded.txt","w+");
#endif
	for(i=0;i<N;i++)
	{
		decimal = info_sym[i];
		for(j=(LOG2Q-1);j>=0;j--)
		{
			remain = decimal % 2;
			decimal = decimal / 2;
			info_bin[i*LOG2Q + j] = remain;
		}
	}
#ifdef DEBUG
	for(i=0;i<LOG2Q*N;i++){
		fprintf(fp,"%d \n", info_bin[i]);
	}
	fclose(fp);
#endif
}

//===================================
// BPSK modulation
//===================================
void bpsk_modulation (int code [], double trans [])
{
	int i ;
#ifdef DEBUG
	FILE *fp = fopen("data/trans.txt","w");
#endif	
	for (i = 0; i < LOG2Q * N; i++){
		if (code [i] == 1)
			trans [i] = 1.0 ;
		else
			trans [i] = -1.0 ;
#ifdef DEBUG
		fprintf(fp,"%f\n",trans[i]);
#endif
	}
#ifdef DEBUG
	fclose(fp);
#endif
}

//===================================
// AWGN Channel
//===================================
void awgn (double *trans, double *rec, double sigma)
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
}

//===================================
// Find the most likely symbols
//===================================
void hard_decision (double recv[], double sigma, int hard[])
{
	int i;
#ifdef DEBUG
	FILE * fp = fopen("data/llr_fp.txt", "w") ;
#endif
	for (i = 0; i < LOG2Q * N; i++)
	{
		if(recv[i] > 0)	hard[i] = 1;
		else hard[i] = 0;
#ifdef DEBUG
		fprintf(fp,"recv[%d] = %f, hard = %d\n", i, recv[i], hard[i]);
#endif
	}
#ifdef DEBUG
	fclose(fp) ;
#endif
}

//===================================
// calculate symbol LLRs
//===================================
void sym_llr(double* llr_, double*** BETAmn_, double** GAMMAn_, int* hard, double* recv, double sigma)
{
	int i,j,k,l,m;
	int dev,remain;
	double temp;

#ifdef DEBUG
	FILE *fp = fopen("data/gamma.csv", "w");
#endif

	// initialize channel information
	for(i = 0; i < M;i++){	
		for(m = 0, j = col_row[i][0]; m < row_weight[i]; m++, j = col_row[i][m]){
			for(k = 0; k < Q;k++){
				dev = k;
				temp = 0;
				for(l = (LOG2Q - 1); l >= 0; l--){
					remain = dev % 2;
					dev = dev / 2;

					int Log2Q = LOG2Q;
					if( remain == 1 && hard[j * Log2Q + l] == 0){
						//BETAmn_[i][j][k] -= 2*recv[j * LOG2Q + l]/(sigma * sigm    a);
						temp -= 2*recv[j * Log2Q + l]/(sigma * sigma);
					}
					else if( remain == 0 && hard[j*Log2Q+l]==1){
						//BETAmn_[i][j][k] += 2 * recv[j * LOG2Q + l] / (sigma * sigma);
						temp += 2*recv[j * Log2Q + l]/(sigma * sigma);
					}
				}
				//GAMMAn[k][j] = BETAmn_[i][j][k];
				GAMMAn[k][j] = temp;
				
			}
		}
	}

#ifdef DEBUG
	fprintf(fp, "sep=,\n");
	for(i = 0;i < N;i++){
		for(j = 0;j < Q;j++){
			fprintf(fp,"%lf, ",GAMMAn_[j][i]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
#endif
}

//===================================
// Main Function
//===================================

int main(int argc, char*argv[])
{
	int		*info_bin;
	int		*decoded_bit;
	int		*info_sym;
	int		*info_bin_coded;
	double 	*trans;
	double 	*recv;
	int		*hard_decision_bit;
	double 	*llr;

	double 	rate = 0.5;
	double 	snr;
	double 	sigma;
	int 	iteration;	
	int 	seed = 69012;
	int 	error = 0;
	int 	error_num = 0;
	int 	fer_threshold = 50;
	int 	counter = 0;
	int 	fe = 0;

#if (RANDOM_BITS == 1)
	srand((unsigned int)time(NULL));
#else
	srand((unsigned int)1010);
#endif

	FILE *fp = fopen("data/data_new.txt", "w+");
	//FILE *fpp = fopen("matrix/H_nb_main.txt", "w+");
	srand (seed);
	qc_gen();	// call quicy-cyclic generate function

	trans 		= (double *) malloc(LOG2Q * N * sizeof(double));
	recv  		= (double *) malloc(LOG2Q * N * sizeof(double));
	llr   		= (double *) malloc(LOG2Q * N * sizeof(double));
	encoded_sym	= (int *) malloc(N * sizeof(int));
	info_bin 	= (int *) malloc(LOG2Q * (N-M) * sizeof(int));
	decoded_bit	= (int *) malloc(LOG2Q * (N-M) * sizeof(int));
	info_sym	= (int *) malloc((N-M) * sizeof(int));
	info_bin_coded = (int *)malloc(LOG2Q * N * sizeof(int));
	hard_decision_bit = (int*)malloc(LOG2Q * N * sizeof(int));

	printf("Q is %d\n",Q);
	h2g_nb(H_nb,N);
	fprintf(fp,"%d\n",Q);

	for (snr = SNR_LOW; snr <= SNR_HIGH; snr += SNR_DELTA)
	{
#ifdef NONOISE
		sigma = 0.01;
#else
		sigma =  1/sqrt(2.0*rate*pow(10.0,(snr/10.0)));
#endif
		counter = 0;
		error_num = 0;
		fe = 0;

		double total_time = 0.0;

#if (RUN_ONCE == 1)
#else
		while(fe<fer_threshold)
#endif
		{
			counter++;
			info_gen(info_bin);
			grouping(info_bin, info_sym);
			encoder_nb(info_sym, encoded_sym);
			ungrouping(encoded_sym, info_bin_coded);
			bpsk_modulation(info_bin_coded, trans); 
			awgn(trans, recv, sigma);

			t1 = clock();
			hard_decision(recv, sigma, hard_decision_bit);
			sym_llr(llr, BETAmn, GAMMAn, hard_decision_bit, recv, sigma);
			error = minmax(BETAmn, ALPHAmn, GAMMAn, &iteration, decoded_bit);
			t2 = clock();  
			double time_diff = 1000* double(t2-t1)/CLOCKS_PER_SEC;
			total_time += time_diff; 
			fprintf(stderr, "throughput is %f Kbps\n", counter * 310 * 5/total_time);

			error_num += error;
			if(error != 0) fe++;
			printf("frame = %d frame error = %d, bit_error=%d\n",counter, fe, error);
		}

		fprintf(fp,"SER is %f at SNR = %f\n",(double)error_num/(counter*(N-M)),snr);
		fprintf(fp,"FER is %f at SNR = %f\n",(double)fe/(counter),snr);
		printf("SER is %f at SNR = %f\n",(double)error_num/(counter*(N-M)),snr);
		printf("FER is %f at SNR = %f\n",(double)fe/(counter),snr);
		printf("SNR = %f is finished\n",snr);
	}

	fclose(fp);
	return 1;
}