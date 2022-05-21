GFX_H contains the H matrix with X corresponding to the number of the Galois field, with M rows and N columns separeted by space.

GFX_H_col contains the H matrix collumn-wise for easy reading.

col_row and row_col contains the respective variables. The collumn-wise representation are in the col_row_vert and row_col_vert files.

symbol_llr.txt contains the gamma values (symbol-wise (Q) ).

The val file contains the a vector with the elements of the H matrix with zeros eliminated, row-wise.

col_ind contains the collumn indexes for the val file. The first element of col_ind designates the column to which the first element of val belongs on the H matrix.

row_ptr maps the firts elements on each row of the H matrix to the val vector. ( since the matrix is regular, the elements in row_ptr will have steps of 3).

ptr_to_val maps the the row-wise CSC to val (to eliminate the redundancy of having CSR and CSC vectors (same symbols, different ordering), we keep the CSR vector and create a vector that maps the CSC to the CSR vector).   

row_ind contains the row indexes for the ptr_to_val file. The first element of row_ind designates the row to which the first element of ptr_to_val, that in turn its linked to val, belongs on the H matrix.

col_ptr maps the firts elements on each column of the H matrix to the ptr_to_val vector. ( since the matrix is regular, the elements in col_ptr will have steps of 2)


EXAMPLE:

        |1 0 0 0 2 0|
GFX_H = |0 2 0 0 0 3|
        |0 0 3 1 0 0|

=====================================================

GFX_H_col = [1 0 0 0 2 0 0 2 0 0 0 3 0 0 3 1 0 0]

=====================================================

          |0 4|
col_row = |1 5|
          |2 3|

=====================================================

row_col = [0 1 2 2 0 1]

=====================================================

val = [1 2 2 3 3 1]

=====================================================

col_ind = [0 4 1 5 2 3]

=====================================================

row_ptr = [0 2 4 6]        

=====================================================

ptr_to_val = [0 2 4 5 1 3]

=====================================================

row_ind = [0 1 2 2 0 1]

====================================================

col_ptr = [0 1 2 3 4 5]
