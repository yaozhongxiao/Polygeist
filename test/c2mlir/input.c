
int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

void const_loop_example(float A[10][10], float B[10][10]) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            B[i][j] = A[i][j];
        }
    }
}

/*
void var_step_loop_example(float A[10][10], float B[10][10], int step) {
    for (int i = 0; i < 10; i+=step) {
        for (int j = 0; j < 10; j+=step) {
            B[i][j] = A[i][j];
        }
    }
}
*/

void var_loop_example(float A[10][10], float B[10][10], int loops) {
    int inner_loop = factorial(loops);
    for (int i = 0; i < inner_loop; i++) {
        for (int j = 0; j < loops; j++) {
            B[i][j] = A[i][j];
        }
    }
}
