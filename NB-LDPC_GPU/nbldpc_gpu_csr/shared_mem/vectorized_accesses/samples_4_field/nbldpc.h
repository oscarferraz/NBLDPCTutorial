#include "stdio.h"
#include "stdlib.h"
#include <math.h>
#include <string.h>
#include "definition.h"

//#define SMALL
#define VS
//#define NONOISE
//#define DEBUG
//#define PRINT_A_B
#define QC
//#define PRINTMSG_V
//#define PRINTMSG_P
//#define PRINTMSG
//#define SYNTHESIS
//#define TREE

#define DEBUG
#define RANDOM_BITS		0
#define RUN_ONCE		0
#define MAX_ITERATION	10


#define EARLY_STOP		1

#ifdef DEBUG
#define SNR_LOW      	2.0
#define SNR_HIGH        2.0
#define SNR_DELTA       0.5
#else
#define SNR_LOW      	2.0
#define SNR_HIGH        2.0
#define SNR_DELTA       0.1
#endif


#define add_gf(a, b) ((a)^(b))
#define sub_gf(a, b) ((a)^(b))
#define ABS(x) (x < 0 ? -(x) : (x))



#ifndef mmax
#define mmax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef mmin
#define mmin( a, b ) (((a) > (b)) ? (b) : (a))
#endif

#ifdef QC
#define Q 4
#define LOG2Q 2
#else
#ifndef LOG2Q
#define LOG2Q 2               // GF(2^2)
#endif
#define Q  (1<<LOG2Q)
#endif

#ifndef GLOBAL_H // header guards
#define GLOBAL_H

#if Q==4
const unsigned char logq[4] = { 0,0,1,2 };
const unsigned char expq[3] = { 1,2,3 };
#elif Q==8
const unsigned char logq[8] = { 0,0,1,3,2,6,4,5 };
const unsigned char expq[7] = { 1,2,4,3,6,7,5 };
#elif Q==16
const unsigned char logq[16] = { 0,0,1,4,2,8,5,10,3,14,9,7,6,13,11,12 };
const unsigned char expq[15] = { 1,2,4,8,3,6,12,11,5,10,7,14,15,13,9 };
#elif Q==32
const unsigned char logq[32] = { 0,0,1,18,2,5,19,11,3,29,6,27,20,8,12,23,4,
10,30,17,7,22,28,26,21,25,9,16,13,14,24,15 };
const unsigned char expq[31] = { 1,2,4,8,16,5,10,20,13,26,17,7,14,28,29,31,
27,19,3,6,12,24,21,15,30,25,23,11,22,9,18 };
#elif Q==64
const unsigned char logq[64] = { 0,0,1,6,2,12,7,26,3,32,13,35,8,48,27,18,4,24,
33,16,14,52,36,54,9,45,49,38,28,41,19,56,5,62,
25,11,34,31,17,47,15,23,53,51,37,44,55,40,10,
61,46,30,50,22,39,43,29,60,42,21,20,59,57,58 };
const unsigned char expq[63] = { 1,2,4,8,16,32,3,6,12,24,48,35,5,10,20,40,19,
38,15,30,60,59,53,41,17,34,7,14,28,56,51,37,
9,18,36,11,22,44,27,54,47,29,58,55,45,25,50,
39,13,26,52,43,21,42,23,46,31,62,63,61,57,49,33 };
#elif Q==128
const unsigned char logq[128] = { 0,0,1,31,2,62,32,103,3,7,63,15,33,84,104,
93, 4,124,8,121,64,79,16,115,34,11,85,38,105,46,94,51,
5,82,125,60,9,44,122,77,65,67,80,42,17,69,116,23,35,118,
12,28,86,25,39,57,106,19,47,89,95,71,52,110,6,14,83,92,126,
30,61,102,10,37,45,50,123,120,78,114,66,41,68,22,81,59,43,76,
18,88,70,109,117,27,24,56,36,49,119,113,13,91,29,101,87,108,
26,55,40,21,58,75,107,54,20,74,48,112,90,100,96,97,72,98,53,73,111,99 };
const unsigned char expq[127] = { 1,2,4,8,16,32,64,9,18,36,72,25,50,100,65,11,
22,44,88,57,114,109,83,47,94,53,106,93,51,102,69,3,6,12,24,
48,96,73,27,54,108,81,43,86,37,74,29,58,116,97,75,31,62,124,
113,107,95,55,110,85,35,70,5,10,20,40,80,41,82,45,90,61,122,
125,115,111,87,39,78,21,42,84,33,66,13,26,52,104,89,59,118,101,
67,15,30,60,120,121,123,127,119,103,71,7,14,28,56,112,105,91,63,
126,117,99,79,23,46,92,49,98,77,19,38,76,17,34,68 };
#elif Q==256
const unsigned char logq[256] = { 0,0,1,25,2,50,26,198,3,223,51,238,27,104,199,75,4,100,
224,14,52,141,239,129,28,193,105,248,200,8,76,113,5,138,101,47,225,
36,15,33,53,147,142,218,240,18,130,69,29,181,194,125,106,39,249,185,
201,154,9,120,77,228,114,166,6,191,139,98,102,221,48,253,226,152,37,
179,16,145,34,136,54,208,148,206,143,150,219,189,241,210,19,92,131,
56,70,64,30,66,182,163,195,72,126,110,107,58,40,84,250,133,186,61,202,
94,155,159,10,21,121,43,78,212,229,172,115,243,167,87,7,112,192,247,
140,128,99,13,103,74,222,237,49,197,254,24,227,165,153,119,38,184,180,
124,17,68,146,217,35,32,137,46,55,63,209,91,149,188,207,205,144,135,151,
178,220,252,190,97,242,86,211,171,20,42,93,158,132,60,57,83,71,109,65,
162,31,45,67,216,183,123,164,118,196,23,73,236,127,12,111,246,108,161,59,
82,41,157,85,170,251,96,134,177,187,204,62,90,203,89,95,176,156,169,160,
81,11,245,22,235,122,117,44,215,79,174,213,233,230,231,173,232,116,214,
244,234,168,80,88,175 };
const unsigned char expq[255] = { 1,2,4,8,16,32,64,128,29,58,116,232,205,135,19,38,76,
152,45,90,180,117,234,201,143,3,6,12,24,48,96,192,157,39,78,156,
37,74,148,53,106,212,181,119,238,193,159,35,70,140,5,10,20,40,80,
160,93,186,105,210,185,111,222,161,95,190,97,194,153,47,94,188,101,
202,137,15,30,60,120,240,253,231,211,187,107,214,177,127,254,
225,223,163,91,182,113,226,217,175,67,134,17,34,68,136,13,26,52,104,
208,189,103,206,129,31,62,124,248,237,199,147,59,118,236,197,151,51,
102,204,133,23,46,92,184,109,218,169,79,158,33,66,132,21,42,84,168,
77,154,41,82,164,85,170,73,146,57,114,228,213,183,115,230,209,191,99,
198,145,63,126,252,229,215,179,123,246,241,255,227,219,171,75,150,49,
98,196,149,55,110,220,165,87,174,65,130,25,50,100,200,141,7,14,28,56,
112,224,221,167,83,166,81,162,89,178,121,242,249,239,195,155,43,86,172,
69,138,9,18,36,72,144,61,122,244,245,247,243,251,235,203,139,11,22,44,
88,176,125,250,233,207,131,27,54,108,216,173,71,142 };
#endif


extern unsigned short	N;
extern unsigned short	M;
extern unsigned short	dc, dv;
extern int				**H_nb;
extern int				**G_nb;
extern unsigned short				*row_weight;
extern unsigned short				*col_weight;
extern int				**row_col;
extern int				**col_row;
extern unsigned char	**BETAmn;
extern unsigned char	**ALPHAmn;
extern unsigned char	**GAMMAn;
extern int				*encoded_sym;
extern unsigned char	**GAMMAn_post;

extern float milliseconds;

extern unsigned char *h_F;
extern unsigned char *h_B;

extern unsigned char	*h_gamma;
extern unsigned char	*h_alpha;
extern unsigned char	*h_beta;

extern unsigned char	val[768];
extern unsigned short	row_ptr[257];
extern unsigned short	col_ind[768];
extern unsigned short	row_ind[768];
extern unsigned short	col_ptr[385];
extern unsigned short	ptr_to_val[768];
#endif

//typedef ap_fixed<8,4,AP_RND> my_type;
unsigned char mul_gf(unsigned char  a, unsigned char  b);
unsigned char  inv_gf(unsigned char  a);
double **malloc2Ddouble(int a, int b);
unsigned char **malloc2Dunsigned_char(int a, int b);
int minmax(unsigned char*** BETAmn_, unsigned char*** ALPHAmn_, unsigned char** GAMMAn_, int* iteration, int* decoded_bit);