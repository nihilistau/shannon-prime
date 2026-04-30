/*
 * Shannon-Prime / Phase 2.3 — minimal QNN HTP runner shim.
 *
 * Public C API for loading a QNN context binary (compiled via AI Hub or
 * Qualcomm tools) and executing it as a matmul kernel inside a host
 * application. Designed to be linked into shannon-prime-llama as the
 * backend for ggml_map_custom2-style attention-matmul ops, but
 * standalone-usable as a plain C library.
 *
 * Why a shim instead of pulling in QnnSampleApp.cpp wholesale: the
 * SampleApp pulls 80+ KB of C++ across IOTensor / QnnWrapperUtils /
 * QnnSampleAppUtils plus the entire PAL/Logger machinery — useful as
 * reference but heavy to integrate. We only need the load-execute
 * primitives, and we can supply our own input/output buffers (rpcmem-
 * backed, allocated by the caller).
 *
 * Threading model: the handle is NOT thread-safe. Wrap externally if
 * you need concurrency. shannon-prime-llama uses one handle per layer
 * sequentially, so single-threaded is fine for the typical use.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3.
 * Commercial license: raydaniels@gmail.com.
 */
#ifndef SHANNON_PRIME_QNN_H
#define SHANNON_PRIME_QNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_qnn_handle sp_qnn_handle;

/* Status codes returned by every API. 0 == success. */
typedef enum {
    SP_QNN_OK                       =  0,
    SP_QNN_ERR_DLOPEN               = -1,  /* libQnnHtp.so / libQnnSystem.so not loadable */
    SP_QNN_ERR_GET_INTERFACE        = -2,  /* QnnInterface_getProviders failed */
    SP_QNN_ERR_BACKEND_CREATE       = -3,  /* QnnBackend_create failed */
    SP_QNN_ERR_DEVICE_CREATE        = -4,  /* QnnDevice_create failed */
    SP_QNN_ERR_READ_FILE            = -5,  /* couldn't read context binary */
    SP_QNN_ERR_BINARY_INFO          = -6,  /* QnnSystemContext_getBinaryInfo failed */
    SP_QNN_ERR_CONTEXT_CREATE       = -7,  /* QnnContext_createFromBinary failed */
    SP_QNN_ERR_GRAPH_RETRIEVE       = -8,  /* QnnContext_retrieveGraph failed */
    SP_QNN_ERR_TENSOR_SHAPE         = -9,  /* caller-supplied tensor doesn't match graph */
    SP_QNN_ERR_EXECUTE              = -10, /* QnnGraph_execute failed */
    SP_QNN_ERR_INVALID              = -11, /* generic invalid argument */
} sp_qnn_status;

/* Library-level init: dlopen the two QNN libs once per process.
 * Subsequent calls are no-ops.
 *
 * On Android, pass the canonical filenames; LD_LIBRARY_PATH should be
 * set to the directory containing libQnnHtp.so + libQnnSystem.so.
 *
 * NULL paths fall back to "libQnnHtp.so" + "libQnnSystem.so".
 */
sp_qnn_status sp_qnn_init(const char *backend_lib_path,
                          const char *system_lib_path);

/* Optional: clean up the dlopen'd libs at process exit. */
void sp_qnn_shutdown(void);

/* Load a QNN context binary from disk and prepare it for execution.
 * The binary must have been produced by a compile flow that targets
 * the same backend (HTP V69 in our case). On success, *out_h is a
 * fresh handle that owns the loaded context + graph + device.
 *
 * `graph_name` may be NULL to auto-pick the first graph in the binary.
 * Most AI-Hub-produced binaries have a single graph; pass the name
 * explicitly only for multi-graph contexts.
 */
sp_qnn_status sp_qnn_load_binary(const char *context_bin_path,
                                 const char *graph_name,  /* may be NULL */
                                 sp_qnn_handle **out_h);

/* Free the handle and all resources it owns (context, graph, device,
 * profile). After this returns, *h is no longer valid.
 */
void sp_qnn_destroy(sp_qnn_handle **h);

/* Query graph metadata. Returned strings + dim arrays are owned by the
 * handle; do not free. NULL out-params are skipped.
 *
 * Typical use: after sp_qnn_load_binary, call this to learn the input
 * tensor shape so the caller knows how big to make the input buffer.
 */
typedef struct {
    const char  *name;
    uint32_t     rank;
    const uint32_t *dims;       /* rank entries */
    uint32_t     bytes_per_element;
    uint32_t     dtype;          /* QNN_DATATYPE_* */
} sp_qnn_tensor_info;

sp_qnn_status sp_qnn_get_io_info(sp_qnn_handle *h,
                                 size_t *out_n_inputs,
                                 const sp_qnn_tensor_info **out_inputs,
                                 size_t *out_n_outputs,
                                 const sp_qnn_tensor_info **out_outputs);

/* One-shot execute. Caller supplies input buffers (one per declared
 * input tensor, in order) and pre-allocated output buffers. On success,
 * outputs are filled and *out_exec_us holds the wall-clock execution
 * time in microseconds (excluding tensor binding).
 *
 * The buffers do NOT have to be rpcmem-backed for correctness, but
 * rpcmem-backed buffers avoid an internal copy on aarch64 Android.
 * Pass an rpcmem-backed buffer if you have one.
 */
sp_qnn_status sp_qnn_execute(sp_qnn_handle *h,
                             const void *const *inputs,    /* n_inputs pointers */
                             const size_t *input_sizes,    /* n_inputs sizes in bytes */
                             void *const *outputs,         /* n_outputs pointers */
                             const size_t *output_sizes,   /* n_outputs sizes in bytes */
                             uint64_t *out_exec_us);       /* optional, may be NULL */

/* Convenience: bench N executions of the same input, returning min/
 * avg/max in microseconds. Useful for the per-layer matmul profiling
 * we'll do during shannon-prime-llama integration.
 */
typedef struct {
    uint32_t n_iterations;
    uint64_t min_us;
    uint64_t avg_us;
    uint64_t max_us;
} sp_qnn_bench_result;

sp_qnn_status sp_qnn_bench(sp_qnn_handle *h,
                           const void *const *inputs,
                           const size_t *input_sizes,
                           void *const *outputs,
                           const size_t *output_sizes,
                           uint32_t n_iterations,
                           sp_qnn_bench_result *out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SHANNON_PRIME_QNN_H */
