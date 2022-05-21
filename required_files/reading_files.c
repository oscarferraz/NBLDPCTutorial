//===================================
// Read Gamma
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

	fclose(fp_gamma);

}



//===================================
// Read H matrix
//===================================
void read_H() {

	FILE *fp_H;
	FILE *fp_row_col;
	FILE *fp_col_row;

	if (Q == 4){
		fp_H = fopen("random_data/Q4/GF4_H_col", "r");
		fp_row_col = fopen("random_data/Q4/row_col_vert", "r");
		fp_col_row = fopen("random_data/Q4/col_row_vert", "r");
	}
	else if (Q == 8){
		fp_H = fopen("random_data/Q8/GF8_H_col", "r");
		fp_row_col = fopen("random_data/Q8/row_col_vert", "r");
		fp_col_row = fopen("random_data/Q8/col_row_vert", "r");	
	}
	else if (Q == 16){
		fp_H = fopen("random_data/Q16/GF16_H_col", "r");
		fp_row_col = fopen("random_data/Q16/row_col_vert", "r");
		fp_col_row = fopen("random_data/Q16/col_row_vert", "r");	
	}
	else if (Q == 32){
		fp_H = fopen("random_data/Q32/GF32_H_col", "r");
		fp_row_col = fopen("random_data/Q32/row_col_vert", "r");
		fp_col_row = fopen("random_data/Q32/col_row_vert", "r");	
	}
	else if (Q == 64){
		fp_H = fopen("random_data/Q64/GF64_H_col", "r");
		fp_row_col = fopen("random_data/Q64/row_col_vert", "r");
		fp_col_row = fopen("random_data/Q64/col_row_vert", "r");	
	}
	else if (Q == 128){
		fp_H = fopen("random_data/Q128/GF128_H_col", "r");
		fp_row_col = fopen("random_data/Q128/row_col_vert", "r");
		fp_col_row = fopen("random_data/Q128/col_row_vert", "r");	
	}
	else if (Q == 256){
		fp_H = fopen("random_data/Q256/GF256_H_col", "r");
		fp_row_col = fopen("random_data/Q256/row_col_vert", "r");
		fp_col_row = fopen("random_data/Q256/col_row_vert", "r");	
	}


	char line[256];
	unsigned int i;
	int p=0;
	int d = 0;

	while (fgets(line, sizeof(line), fp_H)){
		//printf("d=%d, p=%d\n", d, p);
		sscanf(line, "%d", &i);
		H_nb[d][p] = i;
		p++;
		if(p>=N){
			p=0;
			d++;
		}
	}


	fclose(fp_H);

	p=0;
	d = 0;

	while (fgets(line, sizeof(line), fp_row_col)){
		sscanf(line, "%d", &i);
		row_col[d][p] = i;
		d++;
		if(d>=2){
			d=0;
			p++;
		}
	}

	fclose(fp_row_col);


	p=0;
	d = 0;

	while (fgets(line, sizeof(line), fp_col_row)){
		sscanf(line, "%d", &i);
		col_row[d][p] = i;
		p++;
		if(p>=3){
			p=0;
			d++;
		}
	}

	fclose(fp_col_row);

	for (i = 0; i < M; i++) {
		row_weight[i] = 3;
	}


	for (i = 0; i < N; i++) {
		col_weight[i] = 2;
	}

	/* for (int j = 0; j < N; j++) {
		printf("H_nb[0][j]=%d\n",H_nb[0][j]);
	} */


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
	double	*recv;
	int		*hard_decision_bit;
	double 	*llr;

	double 	rate = 0.5;
	double 	snr;
	double 	sigma;
	int 	iteration;
	//int 	seed = 69012;
	int 	error = 0;
	int 	error_num = 0;
	//int 	fer_threshold = 100;
	int 	counter = 0;
	int 	fe = 0;

	M = 256;
	N = 384;

	dc = 2;
	dv = 3;


	row_weight = (int*)malloc(sizeof(int) * M);	//row weight table
	col_weight = (int*)malloc(sizeof(int) * N);	//colum weight table


	row_col = malloc2Dint(dc, N);
	col_row = malloc2Dint(M, dv);
	H_nb = malloc2Dint(M, N);
	G_nb = malloc2Dint(N - M, N);
	BETAmn = malloc3Dunsigned_char(M, N, Q);
	ALPHAmn = malloc3Dunsigned_char(M, N, Q);
	GAMMAn = malloc2Dunsigned_char(Q, N);
	encoded_sym = (int *)malloc(N * sizeof(int));


	FILE *fp = fopen("data/data_new.txt", "w+");
	FILE *fp_times = fopen("data/times.txt", "a+");
	//FILE *fpp = fopen("matrix/H_nb_main.txt", "w+");
	//srand (seed);

	counter = 0;
	error_num = 0;
	fe = 0;

	double total_time = 0.0;

	

	read_H();
	read_llr();

	//for(int i=0; i<M;i++){
		/* for(int j=0; j<N;j++){
			for(int k=0; k<Q;k++){
				printf(" gamma=%hhu\n",GAMMAn[k][j]);
			}
		} */
	//} 

	error = minmax(BETAmn, ALPHAmn, GAMMAn, &iteration, decoded_bit);
	
	/* for(int i=0; i<M;i++){
		for(int j=0; j<N;j++){
			for(int k=0; k<Q;k++){
				printf(" Beta=%hhu\n",BETAmn[i][j][k]);
			}
		}
	} */
	printf(" time=%lf ms\n",((end.tv_sec-start.tv_sec)*1e3)+((end.tv_nsec-start.tv_nsec)*1e-6));
	fprintf(fp_times,"%lf\n",((end.tv_sec-start.tv_sec)*1e3)+((end.tv_nsec-start.tv_nsec)*1e-6));
	fprintf(stderr, "throughput is %f Kbps\n", counter * 310 * 5 / total_time);

	fclose(fp_times);
	printf("check\n");

	printf("SER is %f at SNR = %f\n", (double)error_num / (counter*(N - M)), snr);
	printf("FER is %f at SNR = %f\n", (double)fe / (counter), snr);
	printf("SNR = %f is finished\n", snr);
	
	fclose(fp);


	
	return 1;
}
