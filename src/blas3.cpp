#include <cstdint>
#include <cstdio>

#include "cblas-qmkl6.h"
#include "qmkl6-internal.hpp"


static const uint64_t qpu_sgemm_rnn_orig[] = {
#include "sgemm_rnn.qhex6"
};

static const uint64_t qpu_sgemm_rnt_orig[] = {
#include "sgemm_rnt.qhex6"
};

static const uint64_t qpu_sgemm_rtn_orig[] = {
#include "sgemm_rtn.qhex6"
};

static const uint64_t qpu_sgemm_rtt_orig[] = {
#include "sgemm_rtt.qhex6"
};


void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
        const CBLAS_TRANSPOSE transb, const int m, const int n, const int k,
        const float alpha, const float *a, const int lda, const float *b,
        const int ldb, const float beta, float *c, const int ldc)
{
    if (transa != CblasNoTrans && transa != CblasTrans) {
        fprintf(stderr, "error: transa must be NoTrans or Trans for now\n");
        XERBLA(1);
    }
    if (transb != CblasNoTrans && transb != CblasTrans) {
        fprintf(stderr, "error: transb must be NoTrans or Trans for now\n");
        XERBLA(1);
    }
    if (m <= 0) {
        fprintf(stderr, "error: m (%d) must be greater than zero\n", m);
        XERBLA(1);
    }
    if (n <= 0) {
        fprintf(stderr, "error: n (%d) must be greater than zero\n", n);
        XERBLA(1);
    }
    if (k <= 0) {
        fprintf(stderr, "error: k (%d) must be greater than zero\n", k);
        XERBLA(1);
    }
    if ((layout == CblasRowMajor) == (transa == CblasNoTrans)) {
        if (lda < k) {
            fprintf(stderr, "error: lda (%d) must not be smaller than k (%d)"
                    " for row major, no trans type\n", lda, k);
            XERBLA(1);
        }
    } else {
        if (lda < m) {
            fprintf(stderr, "error: lda (%d) must not be smaller than m (%d)"
                    " for row major, trans type\n", lda, m);
            XERBLA(1);
        }
    }
    if ((layout == CblasRowMajor) == (transb == CblasNoTrans)) {
        if (ldb < n) {
            fprintf(stderr, "error: ldb (%d) must not be smaller than n (%d)"
                    " for row major, no trans type\n", ldb, n);
            XERBLA(1);
        }
    } else {
        if (ldb < k) {
            fprintf(stderr, "error: ldb (%d) must not be smaller than k (%d)"
                    " for row major, trans type\n", ldb, k);
            XERBLA(1);
        }
    }
    if (layout == CblasRowMajor) {
        if (ldc < n) {
            fprintf(stderr, "error: ldc (%d) must not be smaller than n (%d)"
                    " for row major\n", ldc, n);
            XERBLA(1);
        }
    } else {
        if (ldc < m) {
            fprintf(stderr, "error: ldc (%d) must not be smaller than m (%d)"
                    " for column major\n", ldc, m);
            XERBLA(1);
        }
    }

    constexpr int num_qpus = 8;

    uint32_t a_handle, b_handle, c_handle;
    uint32_t a_bus, b_bus, c_bus;

    qmkl6.locate_virt((void*) a, a_handle, a_bus);
    qmkl6.locate_virt((void*) b, b_handle, b_bus);
    qmkl6.locate_virt((void*) c, c_handle, c_bus);

    qmkl6.unif[0] = (layout == CblasRowMajor) ? m : n;
    qmkl6.unif[1] = (layout == CblasRowMajor) ? n : m;
    qmkl6.unif[2] = k;
    qmkl6.unif[3] = (layout == CblasRowMajor) ? a_bus : b_bus;
    qmkl6.unif[4] = (layout == CblasRowMajor) ? b_bus : a_bus;
    qmkl6.unif[5] = c_bus;
    qmkl6.unif[6] = (layout == CblasRowMajor) ? lda : ldb;
    qmkl6.unif[7] = (layout == CblasRowMajor) ? ldb : lda;
    qmkl6.unif[8] = ldc;
    qmkl6.unif[9] = qmkl6.bit_cast <uint32_t> (alpha);
    qmkl6.unif[10] = qmkl6.bit_cast <uint32_t> (beta);

    qmkl6.execute_qpu_code(
            (layout == CblasRowMajor && transa == CblasNoTrans && transb == CblasNoTrans) ? qmkl6.qpu_sgemm_rnn_bus :
            (layout == CblasRowMajor && transa == CblasNoTrans && transb == CblasTrans)   ? qmkl6.qpu_sgemm_rnt_bus :
            (layout == CblasRowMajor && transa == CblasTrans   && transb == CblasNoTrans) ? qmkl6.qpu_sgemm_rtn_bus :
            (layout == CblasRowMajor && transa == CblasTrans   && transb == CblasTrans)   ? qmkl6.qpu_sgemm_rtt_bus :
            (layout == CblasColMajor && transa == CblasNoTrans && transb == CblasNoTrans) ? qmkl6.qpu_sgemm_rnn_bus :
            (layout == CblasColMajor && transa == CblasNoTrans && transb == CblasTrans)   ? qmkl6.qpu_sgemm_rtn_bus :
            (layout == CblasColMajor && transa == CblasTrans   && transb == CblasNoTrans) ? qmkl6.qpu_sgemm_rnt_bus :
            (layout == CblasColMajor && transa == CblasTrans   && transb == CblasTrans)   ? qmkl6.qpu_sgemm_rtt_bus : 0,
            qmkl6.unif_bus, num_qpus, 1, c_handle);

    qmkl6.wait_for_handles(qmkl6.timeout_ns, 1, c_handle);
}

void qmkl6_context::init_blas3(void)
{
    qpu_sgemm_rnn = (uint64_t*) alloc_memory(sizeof(qpu_sgemm_rnn_orig),
            qpu_sgemm_rnn_handle, qpu_sgemm_rnn_bus);
    memcpy(qpu_sgemm_rnn, qpu_sgemm_rnn_orig, sizeof(qpu_sgemm_rnn_orig));
    qpu_sgemm_rnt = (uint64_t*) alloc_memory(sizeof(qpu_sgemm_rnt_orig),
            qpu_sgemm_rnt_handle, qpu_sgemm_rnt_bus);
    memcpy(qpu_sgemm_rnt, qpu_sgemm_rnt_orig, sizeof(qpu_sgemm_rnt_orig));
    qpu_sgemm_rtn = (uint64_t*) alloc_memory(sizeof(qpu_sgemm_rtn_orig),
            qpu_sgemm_rtn_handle, qpu_sgemm_rtn_bus);
    memcpy(qpu_sgemm_rtn, qpu_sgemm_rtn_orig, sizeof(qpu_sgemm_rtn_orig));
    qpu_sgemm_rtt = (uint64_t*) alloc_memory(sizeof(qpu_sgemm_rtt_orig),
            qpu_sgemm_rtt_handle, qpu_sgemm_rtt_bus);
    memcpy(qpu_sgemm_rtt, qpu_sgemm_rtt_orig, sizeof(qpu_sgemm_rtt_orig));
}

void qmkl6_context::finalize_blas3(void)
{
    free_memory(sizeof(qpu_sgemm_rtt_orig), qpu_sgemm_rtt_handle, qpu_sgemm_rtt);
    free_memory(sizeof(qpu_sgemm_rtn_orig), qpu_sgemm_rtn_handle, qpu_sgemm_rtn);
    free_memory(sizeof(qpu_sgemm_rnt_orig), qpu_sgemm_rnt_handle, qpu_sgemm_rnt);
    free_memory(sizeof(qpu_sgemm_rnn_orig), qpu_sgemm_rnn_handle, qpu_sgemm_rnn);
}
