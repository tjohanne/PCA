typedef struct SVD {
    float* U;
    float* S;
    float* V;
} svd_t;

svd_t perform_svd(float* A, int m, int n);