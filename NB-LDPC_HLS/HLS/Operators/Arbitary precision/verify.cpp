#include"header.h"
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <ap_int.h>
#include <sys/resource.h>
unsigned char encoded_sym[V];
using namespace std;
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
//#include"CSR4.h"
ap_uint<5>GAMMAn[4][384];

void read_llr() {

	FILE *fp_gamma;



	if (Q == 4){
		fp_gamma = fopen("/home/srinivasan/NB_LDPC/nbldpc-cpu_csr_csc/random_data/Q4/symbol_llr.txt", "r");
	}
	else if (Q == 8){
		fp_gamma = fopen("/home/srinivasan/NB_LDPC/nbldpc-cpu_csr_csc/random_data/Q8/symbol_llr.txt", "r");
	}
	else if (Q == 16){
		fp_gamma = fopen("/home/srinivasan/NB_LDPC/nbldpc-cpu_csr_csc/random_data/Q16/symbol_llr.txt", "r");
	}
	else if (Q == 32){
		fp_gamma = fopen("/home/srinivasan/NB_LDPC/nbldpc-cpu_csr_csc/random_data/Q32/symbol_llr.txt", "r");
	}
	else if (Q == 64){
		fp_gamma = fopen("/home/srinivasan/NB_LDPC/nbldpc-cpu_csr_csc/random_data/Q64/symbol_llr.txt", "r");
	}
	else if (Q == 128){
		fp_gamma = fopen("/home/srinivasan/NB_LDPC/nbldpc-cpu_csr_csc/random_data/Q128/symbol_llr.txt", "r");
	}
	else if (Q == 256){
		fp_gamma = fopen("/home/srinivasan/NB_LDPC/nbldpc-cpu_csr_csc/random_data/Q256/symbol_llr.txt", "r");
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
	//	printf("GAMMAn[%d][%d] = %d\n", d, z, GAMMAn[d][z]);
		//std::cout<<"ap_uint<5> "<<"GAMMAn"<<"["<<d<<"]"<<"["<<z<<"]"<<"="<<GAMMAn[d][z]<<";"<<"\n";
		//std::cout<<GAMMAn[d][z]<<",";
		d++;
		if (d>=Q){
			d = 0;
			z++;
		}
	}


	//printf("read exit\n");

	fclose(fp_gamma);


}





int main()
{
int error;
read_llr();
	error = minmax(GAMMAn);

return 0;

}

