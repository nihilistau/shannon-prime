/*
 * Shannon-Prime / Phase 2.3 — minimal QNN HTP runner shim implementation.
 *
 * Status: scaffold complete. dlopen + interface-fetching is real.
 * load_binary + execute have explicit TODOs marking where the QNN
 * sequence calls go — next-session work to fill those in against
 * the API documented in PHASE_2_3_DESIGN.md.
 *
 * The scaffold compiles + links cleanly via NDK aarch64-clang against
 * the QAIRT include tree and dlopen-resolves the runtime libs that
 * we already validated on-device in Phase 2.1+2.2. So when the
 * implementation gaps close, the link surface and ABI are already
 * proven.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3.
 */
#include "sp_qnn.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* QNN headers from the QAIRT install. The build script's -I path
 * supplies $(QAIRT_ROOT)/include/QNN so these resolve. */
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

/* ------------------------------------------------------------------ */
/* Internal state                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    void *backend_dlh;             /* dlopen of libQnnHtp.so */
    void *system_dlh;              /* dlopen of libQnnSystem.so */
    QNN_INTERFACE_VER_TYPE qnn;    /* core QNN function table */
    QNN_SYSTEM_INTERFACE_VER_TYPE qsys; /* QnnSystem function table */
    int initialized;
} sp_qnn_lib_t;

static sp_qnn_lib_t g_lib = {0};

struct sp_qnn_handle {
    Qnn_BackendHandle_t backend;
    Qnn_DeviceHandle_t  device;
    Qnn_LogHandle_t     log;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t   graph;

    /* Owned mmap'd context-binary bytes — kept alive for the context
     * lifetime since QnnContext_createFromBinary may reference it. */
    void  *bin_data;
    size_t bin_size;

    /* Binary-info-derived metadata (lifetime-tied to bin_data). */
    const QnnSystemContext_BinaryInfo_t *binary_info;

    /* Cached I/O tensor info for sp_qnn_get_io_info(). Allocated. */
    sp_qnn_tensor_info *inputs;
    size_t              n_inputs;
    sp_qnn_tensor_info *outputs;
    size_t              n_outputs;
};

/* ------------------------------------------------------------------ */
/* dlopen + interface-fetch plumbing — the part that's complete.      */
/* ------------------------------------------------------------------ */

static void *dl_load(const char *path) {
    void *h = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "[sp_qnn] dlopen(%s) failed: %s\n", path, dlerror());
    }
    return h;
}

/* Pull the latest-version interface from the providers array — the
 * QNN convention is "highest minor wins for a given major you support". */
static int pick_qnn_interface(QnnInterface_t *const *providers, uint32_t n,
                              QNN_INTERFACE_VER_TYPE *out) {
    /* Major version we're building against. QNN promises ABI stability
     * within a major version. */
    const uint32_t want_major = QNN_API_VERSION_MAJOR;
    int best = -1;
    uint32_t best_minor = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const Qnn_ApiVersion_t *v = &providers[i]->apiVersion;
        if (v->coreApiVersion.major != want_major) continue;
        if ((int)i == best && v->coreApiVersion.minor < best_minor) continue;
        best = (int)i;
        best_minor = v->coreApiVersion.minor;
    }
    if (best < 0) return -1;
    *out = providers[best]->QNN_INTERFACE_VER_NAME;
    return 0;
}

static int pick_qsys_interface(QnnSystemInterface_t *const *providers, uint32_t n,
                               QNN_SYSTEM_INTERFACE_VER_TYPE *out) {
    const uint32_t want_major = QNN_SYSTEM_API_VERSION_MAJOR;
    int best = -1;
    uint32_t best_minor = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const Qnn_Version_t *v = &providers[i]->systemApiVersion;
        if (v->major != want_major) continue;
        if ((int)i == best && v->minor < best_minor) continue;
        best = (int)i;
        best_minor = v->minor;
    }
    if (best < 0) return -1;
    *out = providers[best]->QNN_SYSTEM_INTERFACE_VER_NAME;
    return 0;
}

sp_qnn_status sp_qnn_init(const char *backend_lib_path,
                          const char *system_lib_path) {
    if (g_lib.initialized) return SP_QNN_OK;

    if (!backend_lib_path) backend_lib_path = "libQnnHtp.so";
    if (!system_lib_path)  system_lib_path  = "libQnnSystem.so";

    /* 1. Open backend (HTP). */
    g_lib.backend_dlh = dl_load(backend_lib_path);
    if (!g_lib.backend_dlh) return SP_QNN_ERR_DLOPEN;

    typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(
        const QnnInterface_t ***, uint32_t *);
    QnnInterface_getProviders_t getProviders =
        (QnnInterface_getProviders_t)dlsym(g_lib.backend_dlh,
                                           "QnnInterface_getProviders");
    if (!getProviders) {
        fprintf(stderr, "[sp_qnn] dlsym(QnnInterface_getProviders) failed: %s\n", dlerror());
        return SP_QNN_ERR_GET_INTERFACE;
    }
    const QnnInterface_t **providers = NULL;
    uint32_t n = 0;
    if (getProviders(&providers, &n) != QNN_SUCCESS || n == 0) {
        return SP_QNN_ERR_GET_INTERFACE;
    }
    if (pick_qnn_interface(providers, n, &g_lib.qnn) != 0) {
        fprintf(stderr, "[sp_qnn] no compatible QNN interface (want major %u)\n",
                QNN_API_VERSION_MAJOR);
        return SP_QNN_ERR_GET_INTERFACE;
    }

    /* 2. Open System lib (for context-binary info parsing). */
    g_lib.system_dlh = dl_load(system_lib_path);
    if (!g_lib.system_dlh) return SP_QNN_ERR_DLOPEN;

    typedef Qnn_ErrorHandle_t (*QnnSystemInterface_getProviders_t)(
        const QnnSystemInterface_t ***, uint32_t *);
    QnnSystemInterface_getProviders_t sysGetProviders =
        (QnnSystemInterface_getProviders_t)dlsym(g_lib.system_dlh,
                                                  "QnnSystemInterface_getProviders");
    if (!sysGetProviders) {
        fprintf(stderr, "[sp_qnn] dlsym(QnnSystemInterface_getProviders) failed: %s\n",
                dlerror());
        return SP_QNN_ERR_GET_INTERFACE;
    }
    const QnnSystemInterface_t **sysProviders = NULL;
    uint32_t sysN = 0;
    if (sysGetProviders(&sysProviders, &sysN) != QNN_SUCCESS || sysN == 0) {
        return SP_QNN_ERR_GET_INTERFACE;
    }
    if (pick_qsys_interface(sysProviders, sysN, &g_lib.qsys) != 0) {
        return SP_QNN_ERR_GET_INTERFACE;
    }

    g_lib.initialized = 1;
    fprintf(stderr, "[sp_qnn] initialized: backend=%s system=%s\n",
            backend_lib_path, system_lib_path);
    return SP_QNN_OK;
}

void sp_qnn_shutdown(void) {
    if (g_lib.system_dlh)  { dlclose(g_lib.system_dlh);  g_lib.system_dlh = NULL; }
    if (g_lib.backend_dlh) { dlclose(g_lib.backend_dlh); g_lib.backend_dlh = NULL; }
    g_lib.initialized = 0;
}

/* ------------------------------------------------------------------ */
/* File-load helpers                                                  */
/* ------------------------------------------------------------------ */

static int read_file(const char *path, void **out_data, size_t *out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    void *buf = malloc((size_t)st.st_size);
    if (!buf) { close(fd); return -1; }
    ssize_t got = read(fd, buf, (size_t)st.st_size);
    close(fd);
    if (got != st.st_size) { free(buf); return -1; }
    *out_data = buf;
    *out_size = (size_t)st.st_size;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Load + execute — TODO sections marked, see PHASE_2_3_DESIGN.md     */
/* ------------------------------------------------------------------ */

sp_qnn_status sp_qnn_load_binary(const char *context_bin_path,
                                 const char *graph_name,
                                 sp_qnn_handle **out_h) {
    if (!g_lib.initialized || !context_bin_path || !out_h)
        return SP_QNN_ERR_INVALID;

    sp_qnn_handle *h = (sp_qnn_handle *)calloc(1, sizeof(*h));
    if (!h) return SP_QNN_ERR_INVALID;

    /* (1) Read context binary into memory. */
    if (read_file(context_bin_path, &h->bin_data, &h->bin_size) != 0) {
        fprintf(stderr, "[sp_qnn] read_file(%s) failed\n", context_bin_path);
        free(h);
        return SP_QNN_ERR_READ_FILE;
    }
    fprintf(stderr, "[sp_qnn] loaded %zu bytes from %s\n",
            h->bin_size, context_bin_path);

    /* (2) Use QnnSystemContext to extract binary info (graphs, tensors). */
    QnnSystemContext_Handle_t sysCtx = NULL;
    if (g_lib.qsys.systemContextCreate(&sysCtx) != QNN_SUCCESS) {
        free(h->bin_data); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }
    Qnn_ContextBinarySize_t info_size = 0;
    if (g_lib.qsys.systemContextGetBinaryInfo(
            sysCtx, h->bin_data, h->bin_size,
            &h->binary_info, &info_size) != QNN_SUCCESS) {
        g_lib.qsys.systemContextFree(sysCtx);
        free(h->bin_data); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }
    /* sysCtx leak is intentional — binary_info pointers reference it.
     * QnnSystemContext_free invalidates them. We free both at destroy. */
    /* TODO: store sysCtx in handle for later free. */

    /* TODO (Phase 2.3 fill-in):
     * (3) Create log handle  : g_lib.qnn.logCreate(...)
     * (4) Create backend     : g_lib.qnn.backendCreate(log, NULL, &h->backend)
     * (5) Create device      : g_lib.qnn.deviceCreate(log, NULL, &h->device)
     *      (HTP-specific QnnDevice_Config can be added for perf-mode tuning)
     * (6) Create context     : g_lib.qnn.contextCreateFromBinary(
     *                              h->backend, h->device, NULL,
     *                              h->bin_data, h->bin_size,
     *                              &h->context, NULL)
     * (7) Retrieve graph     : g_lib.qnn.graphRetrieve(
     *                              h->context, graph_name_to_use,
     *                              &h->graph)
     *      (graph_name_to_use comes from binary_info if NULL)
     * (8) Cache IO tensor info:
     *      iterate binary_info->contextBinaryInfoV2->graphs[*]->graph
     *          .graphTensorInfo[*] etc, populate h->inputs/h->outputs.
     */
    (void)graph_name;  /* TODO use */
    fprintf(stderr, "[sp_qnn] WARNING: load_binary scaffold complete, "
                    "context-create + graph-retrieve TODO. See "
                    "PHASE_2_3_DESIGN.md for the API call sequence.\n");

    *out_h = h;
    return SP_QNN_OK;
}

void sp_qnn_destroy(sp_qnn_handle **h_io) {
    if (!h_io || !*h_io) return;
    sp_qnn_handle *h = *h_io;
    /* TODO: release in reverse order:
     *   graphFree (if QNN provides one — usually no)
     *   contextFree(h->context)
     *   deviceFree(h->device)
     *   backendFree(h->backend)
     *   logFree(h->log)
     */
    free(h->inputs);
    free(h->outputs);
    free(h->bin_data);
    free(h);
    *h_io = NULL;
}

sp_qnn_status sp_qnn_get_io_info(sp_qnn_handle *h,
                                 size_t *out_n_inputs,
                                 const sp_qnn_tensor_info **out_inputs,
                                 size_t *out_n_outputs,
                                 const sp_qnn_tensor_info **out_outputs) {
    if (!h) return SP_QNN_ERR_INVALID;
    /* TODO populate from binary_info during load. */
    if (out_n_inputs)  *out_n_inputs  = h->n_inputs;
    if (out_inputs)    *out_inputs    = h->inputs;
    if (out_n_outputs) *out_n_outputs = h->n_outputs;
    if (out_outputs)   *out_outputs   = h->outputs;
    return SP_QNN_OK;
}

sp_qnn_status sp_qnn_execute(sp_qnn_handle *h,
                             const void *const *inputs,
                             const size_t *input_sizes,
                             void *const *outputs,
                             const size_t *output_sizes,
                             uint64_t *out_exec_us) {
    if (!h) return SP_QNN_ERR_INVALID;
    (void)inputs; (void)input_sizes; (void)outputs; (void)output_sizes;

    /* TODO (Phase 2.3 fill-in):
     * (1) Build Qnn_Tensor_t structs for each input/output:
     *      .v1.dataType   from cached io info
     *      .v1.dimensions from cached io info
     *      .v1.memType    = QNN_TENSORMEMTYPE_RAW
     *      .v1.clientBuf  = { .data = inputs[i], .dataSize = input_sizes[i] }
     * (2) Time the call: gettimeofday before, gettimeofday after.
     * (3) g_lib.qnn.graphExecute(h->graph, in_tensors, n_in,
     *                             out_tensors, n_out, NULL, NULL)
     * (4) Output buffers are filled by QNN; nothing else to do.
     */
    if (out_exec_us) *out_exec_us = 0;
    fprintf(stderr, "[sp_qnn] execute() TODO (Phase 2.3 fill-in).\n");
    return SP_QNN_ERR_EXECUTE;
}

sp_qnn_status sp_qnn_bench(sp_qnn_handle *h,
                           const void *const *inputs,
                           const size_t *input_sizes,
                           void *const *outputs,
                           const size_t *output_sizes,
                           uint32_t n_iter,
                           sp_qnn_bench_result *out) {
    if (!h || !out || n_iter == 0) return SP_QNN_ERR_INVALID;
    out->n_iterations = 0;
    out->min_us = UINT64_MAX;
    out->avg_us = 0;
    out->max_us = 0;
    uint64_t sum = 0;
    for (uint32_t i = 0; i < n_iter; ++i) {
        uint64_t us = 0;
        sp_qnn_status rc = sp_qnn_execute(h, inputs, input_sizes,
                                          outputs, output_sizes, &us);
        if (rc != SP_QNN_OK) return rc;
        if (us < out->min_us) out->min_us = us;
        if (us > out->max_us) out->max_us = us;
        sum += us;
        out->n_iterations++;
    }
    out->avg_us = sum / n_iter;
    return SP_QNN_OK;
}
