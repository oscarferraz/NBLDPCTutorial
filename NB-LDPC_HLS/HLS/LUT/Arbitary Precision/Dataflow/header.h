#include<ap_int.h>


#define M 256
#define N 384
#define NNZ 768
#define MAXITERATION 10
#define X 2
#define Y 3
#define Q 16

#if Q==4
#define len 5
#define bitwidth 7

#elif Q==8
#define len 4
#define bitwidth 8

#elif Q==16
#define len 3
#define bitwidth 9
#endif


void minmax(ap_uint<bitwidth>GAMMAN[Q][N], ap_uint<bitwidth>ALPHAm[NNZ][Q]);


