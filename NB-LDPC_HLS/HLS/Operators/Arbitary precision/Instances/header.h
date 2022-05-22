
#include "ap_fixed.h"
#include <ap_int.h>
#include <cmath>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>


#define add_gf(a, b) ((a)^(b))
#define sub_gf(a, b) ((a)^(b))
#define ABS(x) (x < 0 ? -(x) : (x))

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) > (b)) ? (b) : (a) )
#endif


void read_data();
#define X 2
#define Y 3
#define U 256
#define V 384
#define nnnz 768



#define Q 4
#define LOG2Q 2



extern ap_uint<5>GAMMAn[Q][V];

//void read_data1();
ap_uint<5> minmax(ap_uint<5>GAMMAn[Q][V]);
extern unsigned char encoded_sym[V];
ap_uint<3> mul_gf( ap_uint<3> a, ap_uint<3>b);



