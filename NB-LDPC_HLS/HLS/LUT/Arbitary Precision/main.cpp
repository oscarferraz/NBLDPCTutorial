#include"header.h"



void CN(ap_uint<bitwidth>ALPHAm[NNZ][Q], ap_uint<bitwidth>BETAMn[NNZ][Q])
{

int min_F;
ap_uint<bitwidth>B[Q][Y];
ap_uint<bitwidth>F[Q][Y];
int index,index_r,a,index_p,index_B,c,min_B,b,max_F,max_B,min_value,max_value;
#if Q==4
ap_uint<bitwidth-4>add[4][4] =	{{ 0,1,2,3 },
								 { 1,0,3,2 },
								 { 2,3,0,1 },
								 { 3,2,1,0 }};
ap_uint<bitwidth-4>mult[4][4] = {{ 0,0,0,0}, { 0,1,2,3 },{ 0,2,3,1 },{ 0,3,1,2 }};
ap_uint<bitwidth-4>inv[4] = { 0,1,3,2};
ap_uint<bitwidth-4>val[NNZ]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,2,1,2,1,1,1,1,3,1,3,1,1,3,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,3,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,3,1,1,2,1,1,1,1,1,1,3,3,3,3,1,1,1,2,1,1,1,1,2,1,1,1,1,3,3,1,1,1,2,1,1,1,1,1,3,2,1,1,3,1,3,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,1,3,1,1,1,1,1,1,3,3,3,1,3,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,2,1,1,1,1,2,1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,1,1,1,1,1,2,1,3,1,1,1,3,3,1,1,2,1,1,1,1,1,3,1,2,1,3,1,1,1,2,1,1,3,1,1,2,1,1,1,1,1,1,1,1,1,1,3,3,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,2,1,3,1,3,1,1,3,1,2,1,1,1,1,3,1,1,1,3,1,1,1,3,1,1,2,1,1,2,1,1,1,1,3,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,3,2,1,1,1,1,3,2,1,1,3,1,3,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,3,1,1,2,1,3,3,1,1,2,1,1,1,3,1,1,3,1,3,3,1,1,1,1,1,1,1,2,1,1,1,2,1,1,3,1,1,1,1,1,3,1,1,3,3,1,1,1,1,1,1,1,1,2,1,1,1,1,1,2,1,1,1,2,1,2,1,1,2,1,1,1,3,1,1,2,2,1,1,2,1,1,1,2,1,1,1,1,1,2,1,2,1,1,1,2,1,1,1,1,1,3,1,3,1,2,1,2,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,3,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,3,1,1,1,2,1,1,1,1,1,2,1,1,3,1,1,1,3,1,1,1,1,2,1,3,3,1,2,1,1,2,1,2,1,1,3,1,2,1,1,1,1,1,1,1,1,1,1,1,3,1,3,3,1,1,1,1,1,3,1,1,1,1,2,1,1,3,1,1,1,3,1,1,1,2,1,1,1,3,1,1,1,1,3,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,2,1,1,1,2,3,1,2,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,3,3,3,3,1,3,1,3,1,1,1,1,3,1,1,3,1,1,1,1,1,1,3,3,3,1,1,1,1,3,3,1,2,1,3,1,1,1,1,1,1,1,1,1};
ap_uint<bitwidth+len>row_ptr[M+1]={0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,234,237,240,243,246,249,252,255,258,261,264,267,270,273,276,279,282,285,288,291,294,297,300,303,306,309,312,315,318,321,324,327,330,333,336,339,342,345,348,351,354,357,360,363,366,369,372,375,378,381,384,387,390,393,396,399,402,405,408,411,414,417,420,423,426,429,432,435,438,441,444,447,450,453,456,459,462,465,468,471,474,477,480,483,486,489,492,495,498,501,504,507,510,513,516,519,522,525,528,531,534,537,540,543,546,549,552,555,558,561,564,567,570,573,576,579,582,585,588,591,594,597,600,603,606,609,612,615,618,621,624,627,630,633,636,639,642,645,648,651,654,657,660,663,666,669,672,675,678,681,684,687,690,693,696,699,702,705,708,711,714,717,720,723,726,729,732,735,738,741,744,747,750,753,756,759,762,765,768};
#elif Q==8
ap_uint<bitwidth>val[NNZ]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,2,1,6,1,1,1,1,3,1,3,1,1,3,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,7,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,3,1,1,2,1,1,1,1,1,1,3,3,3,3,1,1,1,6,1,1,1,1,2,1,1,1,1,3,3,1,1,1,2,1,1,1,1,1,3,2,1,1,7,1,7,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,1,3,1,1,1,1,1,1,3,3,3,1,3,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,7,1,2,1,1,1,1,2,1,1,1,6,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,3,3,1,7,1,1,1,1,1,6,1,3,1,1,1,3,3,1,1,2,1,1,1,1,1,3,1,2,1,3,1,1,1,6,1,1,3,1,1,2,1,1,1,1,1,1,1,1,1,1,7,7,1,1,2,1,1,1,1,1,1,6,1,1,1,1,1,1,6,1,1,1,1,1,1,6,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,6,1,3,1,3,1,1,3,1,2,1,1,1,1,3,1,1,1,7,1,1,1,7,1,1,6,1,1,2,1,1,1,1,3,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,3,6,1,1,1,1,7,2,1,1,3,1,3,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,3,1,1,6,1,3,3,1,1,6,1,1,1,3,1,1,3,1,3,3,1,1,1,1,1,1,1,2,1,1,1,2,1,1,3,1,1,1,1,1,3,1,1,3,3,1,1,1,1,1,1,1,1,2,1,1,1,1,1,2,1,1,1,2,1,6,1,1,2,1,1,1,3,1,1,2,2,1,1,6,1,1,1,2,1,1,1,1,1,2,1,6,1,1,1,6,1,1,1,1,1,3,1,7,1,6,1,6,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,3,1,1,1,7,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,7,7,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,3,1,1,1,2,1,1,1,1,1,6,1,1,3,1,1,1,3,1,1,1,1,2,1,3,3,1,2,1,1,2,1,6,1,1,7,1,6,1,1,1,1,1,1,1,1,1,1,1,7,1,3,3,1,1,1,1,1,3,1,1,1,1,2,1,1,3,1,1,1,7,1,1,1,2,1,1,1,3,1,1,1,1,7,7,1,1,1,1,7,7,1,1,1,1,1,1,1,1,1,1,6,1,1,1,6,7,1,6,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1,1,6,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,3,3,7,7,1,3,1,3,1,1,1,1,3,1,1,3,1,1,1,1,1,1,3,7,7,1,1,1,1,3,3,1,6,1,7,1,1,1,1,1,1,1,1,1};
ap_uint<bitwidth+len>row_ptr[M+1]={0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,234,237,240,243,246,249,252,255,258,261,264,267,270,273,276,279,282,285,288,291,294,297,300,303,306,309,312,315,318,321,324,327,330,333,336,339,342,345,348,351,354,357,360,363,366,369,372,375,378,381,384,387,390,393,396,399,402,405,408,411,414,417,420,423,426,429,432,435,438,441,444,447,450,453,456,459,462,465,468,471,474,477,480,483,486,489,492,495,498,501,504,507,510,513,516,519,522,525,528,531,534,537,540,543,546,549,552,555,558,561,564,567,570,573,576,579,582,585,588,591,594,597,600,603,606,609,612,615,618,621,624,627,630,633,636,639,642,645,648,651,654,657,660,663,666,669,672,675,678,681,684,687,690,693,696,699,702,705,708,711,714,717,720,723,726,729,732,735,738,741,744,747,750,753,756,759,762,765,768};
ap_uint<bitwidth-4>add[8][8] = {{ 0,1,2,3,4,5,6,7 }, { 1,0,3,2,5,4,7,6 },{ 2,3,0,1,6,7,4,5 },{ 3,2,1,0,7,6,5,4 },{ 4,5,6,7,0,1,2,3 }, { 5,4,7,6,1,0,3,2 },{ 6,7,4,5,2,3,0,1 },{ 7,6,5,4,3,2,1,0 }};
ap_uint<bitwidth-4>mult[8][8] = {{ 0,0,0,0,0,0,0,0}, { 0,1,2,3,4,5,6,7 },{ 0,2,4,6,3,1,7,5 },{ 0,3,6,5,7,4,1,2 },{ 0,4,3,7,6,2,5,1}, { 0,5,1,4,2,7,3,6 },{ 0,6,7,1,5,3,2,4 },{ 0,7,5,2,1,6,4,3 }};
ap_uint<bitwidth-4>inv[8] = { 0,1,5,6,7,2,3,4};
#elif Q==16
ap_uint<bitwidth-3>add[16][16] = {	{ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
										{ 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14 },
										{ 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13 },
										{ 3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12 },
										{ 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11 },
										{ 5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10 },
										{ 6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9 },
										{ 7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8 },
										{ 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 },
										{ 9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6 },
										{ 10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5 },
										{ 11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4 },
										{ 12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3 },
										{ 13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2 },
										{ 14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1 },
										{ 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 }};

ap_uint<bitwidth-4>mult[16][16] = {{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
										{ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 },
										{ 0,2,4,6,8,10,12,14,3,1,7,5,11,9,15,13 },
										{ 0,3,6,5,12,15,10,9,11,8,13,14,7,4,1,2 },
										{ 0,4,8,12,3,7,11,15,6,2,14,10,5,1,13,9},
										{ 0,5,10,15,7,2,13,8,14,11,4,1,9,12,3,6},
										{ 0,6,12,10,11,13,7,1,5,3,9,15,14,8,2,4},
										{ 0,7,14,9,15,8,1,6,13,10,3,4,2,5,12,11},
										{ 0,8,3,11,6,14,5,13,12,4,15,7,10,2,9,1},
										{ 0,9,1,8,2,11,3,10,4,13,5,12,6,15,7,14},
										{ 0,10,7,13,14,4,9,3,15,5,8,2,1,11,6,12},
										{ 0,11,5,14,10,1,15,4,7,12,2,9,13,6,8,3},
										{ 0,12,11,7,5,9,14,2,10,6,1,13,15,3,4,8},
										{ 0,13,9,4,1,12,8,5,2,15,11,6,3,14,10,7},
										{ 0,14,15,1,13,3,2,12,9,7,6,8,4,10,11,5},
										{ 0,15,13,2,9,6,4,11,1,14,12,3,8,7,5,10}};

	ap_uint<bitwidth-4>inv[16] = { 0,1,9,14,13,11,7,6,15,2,12,5,10,4,3,8};
	ap_uint<bitwidth+len>row_ptr[M+1]={0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,234,237,240,243,246,249,252,255,258,261,264,267,270,273,276,279,282,285,288,291,294,297,300,303,306,309,312,315,318,321,324,327,330,333,336,339,342,345,348,351,354,357,360,363,366,369,372,375,378,381,384,387,390,393,396,399,402,405,408,411,414,417,420,423,426,429,432,435,438,441,444,447,450,453,456,459,462,465,468,471,474,477,480,483,486,489,492,495,498,501,504,507,510,513,516,519,522,525,528,531,534,537,540,543,546,549,552,555,558,561,564,567,570,573,576,579,582,585,588,591,594,597,600,603,606,609,612,615,618,621,624,627,630,633,636,639,642,645,648,651,654,657,660,663,666,669,672,675,678,681,684,687,690,693,696,699,702,705,708,711,714,717,720,723,726,729,732,735,738,741,744,747,750,753,756,759,762,765,768};
	ap_uint<bitwidth+len>val[NNZ]={1,8,9,9,1,8,8,1,9,1,1,8,8,9,1,1,11,1,1,1,10,1,14,8,1,9,8,3,1,11,1,1,11,10,1,1,8,9,1,1,10,1,1,8,1,1,1,8,1,1,10,8,1,7,8,9,1,8,1,1,1,8,14,9,8,1,1,9,8,1,9,9,8,9,1,11,1,1,11,1,1,10,1,1,8,1,1,1,3,11,11,3,1,8,1,14,8,1,9,1,10,1,1,1,8,11,3,1,1,1,10,8,9,1,1,1,11,10,1,1,7,1,7,8,1,1,1,10,1,1,9,8,8,9,1,1,1,8,1,1,10,10,1,1,11,1,1,1,8,9,1,11,3,3,1,11,8,1,1,1,11,1,8,1,1,1,9,8,1,8,1,8,9,1,8,1,7,1,10,1,9,1,8,10,1,1,8,14,1,1,10,1,1,10,1,8,1,1,1,8,9,8,1,1,1,11,3,8,7,1,1,1,8,8,14,1,11,1,1,1,3,11,1,1,10,1,8,1,1,1,11,1,10,1,11,1,1,9,6,1,1,11,1,1,10,1,9,1,1,1,1,9,1,8,1,7,7,1,1,10,1,8,1,9,1,8,14,8,1,1,1,9,1,14,8,1,1,1,8,1,6,9,1,10,1,1,8,1,10,1,1,1,1,10,1,1,8,1,8,1,8,14,1,11,1,3,1,1,11,1,10,1,9,1,9,11,1,1,8,7,1,1,8,7,1,8,14,1,1,10,8,1,1,1,11,1,8,1,9,1,8,9,1,1,10,10,1,1,1,1,11,14,8,1,1,8,7,10,1,1,11,1,3,1,9,1,1,9,8,11,1,1,8,1,1,1,8,1,1,10,1,1,10,1,1,9,8,8,9,1,3,11,1,1,9,8,1,1,11,1,8,14,1,3,11,8,1,14,9,8,1,11,1,1,11,1,3,11,1,1,1,8,1,1,1,10,1,1,8,10,1,1,11,1,1,1,1,8,11,1,1,11,3,1,1,8,9,1,8,9,1,10,1,1,8,1,1,10,1,1,1,10,8,14,1,1,10,1,1,1,11,1,1,10,10,1,1,6,9,1,1,10,1,8,1,1,1,10,1,14,1,8,1,6,9,9,1,1,1,11,1,7,1,6,9,6,1,1,1,10,1,1,8,1,10,1,1,8,1,1,1,8,9,1,8,1,1,10,9,1,9,1,1,11,1,8,1,7,7,1,1,1,9,1,1,8,9,1,8,8,9,1,1,7,7,1,8,1,1,1,10,9,1,1,9,8,1,1,10,1,1,11,1,1,1,10,8,1,1,1,8,14,1,1,11,1,9,8,11,1,1,1,1,10,1,11,3,1,10,1,1,10,1,6,9,1,7,1,6,9,8,1,1,8,1,1,9,8,8,1,7,1,11,3,1,8,9,1,1,11,9,1,8,1,10,1,1,11,1,8,1,7,1,8,9,10,1,1,1,11,1,8,1,1,7,7,1,1,1,8,7,7,1,9,1,9,1,8,1,1,1,9,6,9,1,1,6,7,9,6,1,1,10,1,1,1,9,10,1,1,1,9,8,1,1,9,1,6,9,1,1,10,1,9,1,1,1,8,1,9,1,1,1,8,9,1,1,1,1,10,1,9,8,1,11,3,7,7,1,11,1,3,9,1,9,1,11,1,1,11,1,8,1,1,1,1,11,7,7,1,9,9,1,3,11,1,6,1,7,9,1,9,9,1,8,8,1,1};
	#endif





CN_label1:for (index_r =0; index_r<M; index_r++) {
	for (a= 0; a< Q; a++)
		{
		F[a][0] = ALPHAm[index_r*3][mult[inv[val[index_r*3]]][a]];
		B[a][2] = ALPHAm[row_ptr[index_r + 1] - 1][mult[inv[val[row_ptr[index_r + 1] - 1]]][a]];
		}
	index_p = ((index_r*3)+1) - (index_r*3) - 1;
	index_B = (index_r*3)+3 - ((index_r*3)+1)-1;

	for (c = 0; c < Q; c++) {
		min_F =((F[c][index_p])>(ALPHAm[((index_r*3)+1)][0]))? (((F[c][index_p]))):(ALPHAm[((index_r*3)+1)][0]);
		min_B = ((ALPHAm[row_ptr[index_r ] + index_B][0]))>(B[c][index_B + 1])?((ALPHAm[row_ptr[index_r ] + index_B][0])):(B[c][index_B + 1]);
		for (b= 1;b< Q; b++) {
			a = add[c][mult[val[((index_r*3)+1)]][b]];
			max_F =(F[a][index_p])>(ALPHAm[((index_r*3)+1)][b])?(F[a][index_p]):(ALPHAm[((index_r*3)+1)][b]) ;
			min_F = (min_F)>(max_F)?max_F:min_F;
							 }
		for (b= 0; b < Q; b++) {
			a = add[c][mult[val[row_ptr[index_r] + index_B]][b]];
			max_B=(B[a][index_B + 1])>(ALPHAm[row_ptr[index_r ] + index_B][b])?(B[a][index_B + 1]):(ALPHAm[row_ptr[index_r ] + index_B][b]);
			min_B = (min_B)>(max_B)?max_B:min_B;
							}
		F[c][((index_r*3)+1) - (index_r*3)] = min_F;
		B[c][index_B] = min_B;
							 }
	for (a= 0;a<Q; a++){
		BETAMn[index_r*3][a] = B[mult[val[index_r*3]][a]][1];
		BETAMn[(index_r*3)+2][a] =F[mult[val[(index_r*3)+2]][a]][(index_r*3+2)-row_ptr[index_r]-1];
						}

	for (c = 0; c < Q; c++) {
		min_value = (F[mult[val[((index_r*3)+1)]][c]][((index_r*3)+1)-row_ptr[index_r]-1])>(B[0][((index_r*3)+1)-row_ptr[index_r]+1]) ? (F[mult[val[((index_r*3)+1)]][c]][((index_r*3)+1)-row_ptr[index_r]-1]):B[0][((index_r*3)+1)-row_ptr[index_r]+1];
		for (b = 0; b < Q; b++) {
			a = add[b][mult[val[((index_r*3)+1)]][c]];
			max_value=(F[a][((index_r*3)+1)-row_ptr[index_r] - 1])>(B[b][((index_r*3)+1)-row_ptr[index_r] + 1])?(F[a][((index_r*3)+1)-row_ptr[index_r] - 1]):(B[b][((index_r*3)+1)-row_ptr[index_r] + 1]);
			min_value =(max_value>min_value)?min_value:max_value;
								}
		BETAMn[(index_r*3)+1][c] = min_value;
						}
}}


void VN(ap_uint<bitwidth>ALPHAm[NNZ][Q], ap_uint<bitwidth>BETAMn[NNZ][Q], ap_uint<bitwidth>GAMMAN[Q][N])
{


#if Q==4
ap_uint<bitwidth+len>ptr_to_val[NNZ]={0,381,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39,40,42,43,45,46,48,49,51,52,54,55,57,58,60,61,63,64,66,67,69,70,72,73,75,76,78,79,81,82,84,85,87,88,90,91,93,94,96,97,99,100,102,103,105,106,108,109,111,112,114,115,117,118,120,121,123,124,126,127,129,130,132,133,135,136,138,139,141,142,144,145,147,148,150,151,153,154,156,157,159,160,162,163,165,166,168,169,171,172,174,175,177,178,180,181,183,184,186,187,189,190,192,193,195,196,198,199,201,202,204,205,207,208,210,211,213,214,216,217,219,220,222,223,225,226,228,229,231,232,234,235,237,238,240,241,243,244,246,247,249,250,252,253,255,256,258,259,261,262,264,265,267,268,270,271,273,274,276,277,279,280,282,283,285,286,288,289,291,292,294,295,297,298,300,301,303,304,306,307,309,310,312,313,315,316,318,319,321,322,324,325,327,328,330,331,333,334,336,337,339,340,342,343,345,346,348,349,351,352,354,355,357,358,360,361,363,364,366,367,369,370,372,373,375,376,378,379,382,384,765,385,387,388,390,391,393,394,396,397,399,400,402,403,405,406,408,409,411,412,414,415,417,418,420,421,423,424,426,427,429,430,432,433,435,436,438,439,441,442,444,445,447,448,450,451,453,454,456,457,459,460,462,463,465,466,468,469,471,472,474,475,477,478,480,481,483,484,486,487,489,490,492,493,495,496,498,499,501,502,504,505,507,508,510,511,513,514,516,517,519,520,522,523,525,526,528,529,531,532,534,535,537,538,540,541,543,544,546,547,549,550,552,553,555,556,558,559,561,562,564,565,567,568,570,571,573,574,576,577,579,580,582,583,585,586,588,589,591,592,594,595,597,598,600,601,603,604,606,607,609,610,612,613,615,616,618,619,621,622,624,625,627,628,630,631,633,634,636,637,639,640,642,643,645,646,648,649,651,652,654,655,657,658,660,661,663,664,666,667,669,670,672,673,675,676,678,679,681,682,684,685,687,688,690,691,693,694,696,697,699,700,702,703,705,706,708,709,711,712,714,715,717,718,720,721,723,724,726,727,729,730,732,733,735,736,738,739,741,742,744,745,747,748,750,751,753,754,756,757,759,760,762,763,766,2,425,5,398,8,755,11,446,14,731,17,470,20,497,23,710,26,407,29,518,32,437,35,740,38,389,41,458,44,533,47,716,50,479,53,554,56,695,59,764,62,671,65,428,68,461,71,512,74,563,77,728,80,587,83,692,86,659,89,524,92,635,95,602,98,503,101,416,104,743,107,686,110,569,113,452,116,491,119,596,122,653,125,404,128,629,131,560,134,443,137,674,140,527,143,746,146,617,149,476,152,593,155,434,158,545,161,644,164,719,167,572,170,761,173,626,176,419,179,488,182,677,185,734,188,611,191,536,194,401,197,584,200,467,203,638,206,767,209,449,212,410,215,551,218,665,221,752,224,506,227,647,230,680,233,704,236,386,239,590,242,422,245,521,248,548,251,464,254,608,257,575,260,395,263,509,266,485,269,542,272,698,275,737,278,650,281,578,284,431,287,494,290,758,293,530,296,701,299,413,302,473,305,566,308,662,311,392,314,620,317,713,320,689,323,440,326,605,329,749,332,725,335,455,338,632,341,707,344,581,347,482,350,668,353,599,356,515,359,623,362,539,365,656,368,500,371,722,374,683,377,557,380,614,383,641};
ap_uint<bitwidth+len>col_ptr[N+1]={0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288,290,292,294,296,298,300,302,304,306,308,310,312,314,316,318,320,322,324,326,328,330,332,334,336,338,340,342,344,346,348,350,352,354,356,358,360,362,364,366,368,370,372,374,376,378,380,382,384,386,388,390,392,394,396,398,400,402,404,406,408,410,412,414,416,418,420,422,424,426,428,430,432,434,436,438,440,442,444,446,448,450,452,454,456,458,460,462,464,466,468,470,472,474,476,478,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510,512,514,516,518,520,522,524,526,528,530,532,534,536,538,540,542,544,546,548,550,552,554,556,558,560,562,564,566,568,570,572,574,576,578,580,582,584,586,588,590,592,594,596,598,600,602,604,606,608,610,612,614,616,618,620,622,624,626,628,630,632,634,636,638,640,642,644,646,648,650,652,654,656,658,660,662,664,666,668,670,672,674,676,678,680,682,684,686,688,690,692,694,696,698,700,702,704,706,708,710,712,714,716,718,720,722,724,726,728,730,732,734,736,738,740,742,744,746,748,750,752,754,756,758,760,762,764,766,768};
#elif Q==8
ap_uint<bitwidth+len>ptr_to_val[NNZ]={0,381,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39,40,42,43,45,46,48,49,51,52,54,55,57,58,60,61,63,64,66,67,69,70,72,73,75,76,78,79,81,82,84,85,87,88,90,91,93,94,96,97,99,100,102,103,105,106,108,109,111,112,114,115,117,118,120,121,123,124,126,127,129,130,132,133,135,136,138,139,141,142,144,145,147,148,150,151,153,154,156,157,159,160,162,163,165,166,168,169,171,172,174,175,177,178,180,181,183,184,186,187,189,190,192,193,195,196,198,199,201,202,204,205,207,208,210,211,213,214,216,217,219,220,222,223,225,226,228,229,231,232,234,235,237,238,240,241,243,244,246,247,249,250,252,253,255,256,258,259,261,262,264,265,267,268,270,271,273,274,276,277,279,280,282,283,285,286,288,289,291,292,294,295,297,298,300,301,303,304,306,307,309,310,312,313,315,316,318,319,321,322,324,325,327,328,330,331,333,334,336,337,339,340,342,343,345,346,348,349,351,352,354,355,357,358,360,361,363,364,366,367,369,370,372,373,375,376,378,379,382,384,765,385,387,388,390,391,393,394,396,397,399,400,402,403,405,406,408,409,411,412,414,415,417,418,420,421,423,424,426,427,429,430,432,433,435,436,438,439,441,442,444,445,447,448,450,451,453,454,456,457,459,460,462,463,465,466,468,469,471,472,474,475,477,478,480,481,483,484,486,487,489,490,492,493,495,496,498,499,501,502,504,505,507,508,510,511,513,514,516,517,519,520,522,523,525,526,528,529,531,532,534,535,537,538,540,541,543,544,546,547,549,550,552,553,555,556,558,559,561,562,564,565,567,568,570,571,573,574,576,577,579,580,582,583,585,586,588,589,591,592,594,595,597,598,600,601,603,604,606,607,609,610,612,613,615,616,618,619,621,622,624,625,627,628,630,631,633,634,636,637,639,640,642,643,645,646,648,649,651,652,654,655,657,658,660,661,663,664,666,667,669,670,672,673,675,676,678,679,681,682,684,685,687,688,690,691,693,694,696,697,699,700,702,703,705,706,708,709,711,712,714,715,717,718,720,721,723,724,726,727,729,730,732,733,735,736,738,739,741,742,744,745,747,748,750,751,753,754,756,757,759,760,762,763,766,2,425,5,398,8,755,11,446,14,731,17,470,20,497,23,710,26,407,29,518,32,437,35,740,38,389,41,458,44,533,47,716,50,479,53,554,56,695,59,764,62,671,65,428,68,461,71,512,74,563,77,728,80,587,83,692,86,659,89,524,92,635,95,602,98,503,101,416,104,743,107,686,110,569,113,452,116,491,119,596,122,653,125,404,128,629,131,560,134,443,137,674,140,527,143,746,146,617,149,476,152,593,155,434,158,545,161,644,164,719,167,572,170,761,173,626,176,419,179,488,182,677,185,734,188,611,191,536,194,401,197,584,200,467,203,638,206,767,209,449,212,410,215,551,218,665,221,752,224,506,227,647,230,680,233,704,236,386,239,590,242,422,245,521,248,548,251,464,254,608,257,575,260,395,263,509,266,485,269,542,272,698,275,737,278,650,281,578,284,431,287,494,290,758,293,530,296,701,299,413,302,473,305,566,308,662,311,392,314,620,317,713,320,689,323,440,326,605,329,749,332,725,335,455,338,632,341,707,344,581,347,482,350,668,353,599,356,515,359,623,362,539,365,656,368,500,371,722,374,683,377,557,380,614,383,641};
ap_uint<bitwidth+len>col_ptr[N+1]={0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288,290,292,294,296,298,300,302,304,306,308,310,312,314,316,318,320,322,324,326,328,330,332,334,336,338,340,342,344,346,348,350,352,354,356,358,360,362,364,366,368,370,372,374,376,378,380,382,384,386,388,390,392,394,396,398,400,402,404,406,408,410,412,414,416,418,420,422,424,426,428,430,432,434,436,438,440,442,444,446,448,450,452,454,456,458,460,462,464,466,468,470,472,474,476,478,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510,512,514,516,518,520,522,524,526,528,530,532,534,536,538,540,542,544,546,548,550,552,554,556,558,560,562,564,566,568,570,572,574,576,578,580,582,584,586,588,590,592,594,596,598,600,602,604,606,608,610,612,614,616,618,620,622,624,626,628,630,632,634,636,638,640,642,644,646,648,650,652,654,656,658,660,662,664,666,668,670,672,674,676,678,680,682,684,686,688,690,692,694,696,698,700,702,704,706,708,710,712,714,716,718,720,722,724,726,728,730,732,734,736,738,740,742,744,746,748,750,752,754,756,758,760,762,764,766,768};
#elif Q==16
ap_uint<bitwidth+len>col_ptr[N+1]={0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288,290,292,294,296,298,300,302,304,306,308,310,312,314,316,318,320,322,324,326,328,330,332,334,336,338,340,342,344,346,348,350,352,354,356,358,360,362,364,366,368,370,372,374,376,378,380,382,384,386,388,390,392,394,396,398,400,402,404,406,408,410,412,414,416,418,420,422,424,426,428,430,432,434,436,438,440,442,444,446,448,450,452,454,456,458,460,462,464,466,468,470,472,474,476,478,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510,512,514,516,518,520,522,524,526,528,530,532,534,536,538,540,542,544,546,548,550,552,554,556,558,560,562,564,566,568,570,572,574,576,578,580,582,584,586,588,590,592,594,596,598,600,602,604,606,608,610,612,614,616,618,620,622,624,626,628,630,632,634,636,638,640,642,644,646,648,650,652,654,656,658,660,662,664,666,668,670,672,674,676,678,680,682,684,686,688,690,692,694,696,698,700,702,704,706,708,710,712,714,716,718,720,722,724,726,728,730,732,734,736,738,740,742,744,746,748,750,752,754,756,758,760,762,764,766,768};
ap_uint<bitwidth+len>ptr_to_val[NNZ]={0,381,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39,40,42,43,45,46,48,49,51,52,54,55,57,58,60,61,63,64,66,67,69,70,72,73,75,76,78,79,81,82,84,85,87,88,90,91,93,94,96,97,99,100,102,103,105,106,108,109,111,112,114,115,117,118,120,121,123,124,126,127,129,130,132,133,135,136,138,139,141,142,144,145,147,148,150,151,153,154,156,157,159,160,162,163,165,166,168,169,171,172,174,175,177,178,180,181,183,184,186,187,189,190,192,193,195,196,198,199,201,202,204,205,207,208,210,211,213,214,216,217,219,220,222,223,225,226,228,229,231,232,234,235,237,238,240,241,243,244,246,247,249,250,252,253,255,256,258,259,261,262,264,265,267,268,270,271,273,274,276,277,279,280,282,283,285,286,288,289,291,292,294,295,297,298,300,301,303,304,306,307,309,310,312,313,315,316,318,319,321,322,324,325,327,328,330,331,333,334,336,337,339,340,342,343,345,346,348,349,351,352,354,355,357,358,360,361,363,364,366,367,369,370,372,373,375,376,378,379,382,384,765,385,387,388,390,391,393,394,396,397,399,400,402,403,405,406,408,409,411,412,414,415,417,418,420,421,423,424,426,427,429,430,432,433,435,436,438,439,441,442,444,445,447,448,450,451,453,454,456,457,459,460,462,463,465,466,468,469,471,472,474,475,477,478,480,481,483,484,486,487,489,490,492,493,495,496,498,499,501,502,504,505,507,508,510,511,513,514,516,517,519,520,522,523,525,526,528,529,531,532,534,535,537,538,540,541,543,544,546,547,549,550,552,553,555,556,558,559,561,562,564,565,567,568,570,571,573,574,576,577,579,580,582,583,585,586,588,589,591,592,594,595,597,598,600,601,603,604,606,607,609,610,612,613,615,616,618,619,621,622,624,625,627,628,630,631,633,634,636,637,639,640,642,643,645,646,648,649,651,652,654,655,657,658,660,661,663,664,666,667,669,670,672,673,675,676,678,679,681,682,684,685,687,688,690,691,693,694,696,697,699,700,702,703,705,706,708,709,711,712,714,715,717,718,720,721,723,724,726,727,729,730,732,733,735,736,738,739,741,742,744,745,747,748,750,751,753,754,756,757,759,760,762,763,766,2,425,5,398,8,755,11,446,14,731,17,470,20,497,23,710,26,407,29,518,32,437,35,740,38,389,41,458,44,533,47,716,50,479,53,554,56,695,59,764,62,671,65,428,68,461,71,512,74,563,77,728,80,587,83,692,86,659,89,524,92,635,95,602,98,503,101,416,104,743,107,686,110,569,113,452,116,491,119,596,122,653,125,404,128,629,131,560,134,443,137,674,140,527,143,746,146,617,149,476,152,593,155,434,158,545,161,644,164,719,167,572,170,761,173,626,176,419,179,488,182,677,185,734,188,611,191,536,194,401,197,584,200,467,203,638,206,767,209,449,212,410,215,551,218,665,221,752,224,506,227,647,230,680,233,704,236,386,239,590,242,422,245,521,248,548,251,464,254,608,257,575,260,395,263,509,266,485,269,542,272,698,275,737,278,650,281,578,284,431,287,494,290,758,293,530,296,701,299,413,302,473,305,566,308,662,311,392,314,620,317,713,320,689,323,440,326,605,329,749,332,725,335,455,338,632,341,707,344,581,347,482,350,668,353,599,356,515,359,623,362,539,365,656,368,500,371,722,374,683,377,557,380,614,383,641};
#endif

ap_uint<bitwidth>ALPHA_t[Q];

int col_v,index_v,a,temp,index_,index2,min_index;

VN_label2:for(col_v = 0; col_v < N; col_v ++){
	for(index_v=0;index_v< 2;index_v++){
		for(a = 0,min_index=0; a < Q; a++){
			temp = 0;
			for(index_=0;index_< 2;index_++){
					if(index_ != index_v){
						temp = temp + BETAMn[ptr_to_val[col_ptr[col_v]+index_]][a];
							}
						}

			 ALPHA_t[a] = temp + GAMMAN[a][col_v];}


	for(a = 1; a < Q; a++){
		if(ALPHA_t[a] < ALPHA_t[min_index]){
			min_index = a;
						}}

	for(a = 0; a < Q; a++){
		ALPHAm[ptr_to_val[col_ptr[col_v]+index_v]][a] = ALPHA_t[a] - ALPHA_t[min_index];
		}}}



}








void minmax(ap_uint<bitwidth>GAMMAN[Q][N], ap_uint<bitwidth>ALPHAm[NNZ][Q])
{



ap_uint<bitwidth>ALPHAm1[NNZ][Q];
ap_uint<bitwidth>BETAMn[NNZ][Q];

ap_uint<bitwidth>Gamalocal[Q][N];





int row,a;




for (row = 0; row < NNZ; row++) {
	for (a = 0; a < Q; a++) {
		ALPHAm1[row][a]=ALPHAm[row][a];
	}
}




for(int i=0;i<Q;i++)
{

	for (int j=0;j<N;j++)
	{
		Gamalocal[i][j]=GAMMAN[i][j];
	}
}


#pragma HLS DATAFLOW

for(int i=0;i<10;i++)
{
CN(ALPHAm1,BETAMn);
VN(ALPHAm1,BETAMn,Gamalocal);
}




for (row = 0; row < NNZ; row++) {
	for (a = 0; a < Q; a++) {
		ALPHAm[row][a]=ALPHAm1[row][a];
	}
}




}













