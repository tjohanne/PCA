typedef struct SVD {
    float* U;
    float* S;
    float* V;
} svd_t;
void printMatrix(int m, int n, const float *A, int lda, const char *name);
svd_t perform_svd(float* A, int m, int n);
void printVector(int m, const float *A, const char *name);