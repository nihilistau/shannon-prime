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
#include <stdbool.h>
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
#include "QnnDevice.h"
#include "QnnContext.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

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

    /* QnnSystemContext handle — must live for the lifetime of any
     * binary_info pointers we cached. Freed at destroy(). */
    QnnSystemContext_Handle_t sysCtx;

    /* Binary-info-derived metadata (lifetime-tied to sysCtx). */
    const QnnSystemContext_BinaryInfo_t *binary_info;

    /* Cached I/O tensor info for sp_qnn_get_io_info(). Allocated. */
    sp_qnn_tensor_info *inputs;
    size_t              n_inputs;
    sp_qnn_tensor_info *outputs;
    size_t              n_outputs;

    /* Tensor templates copied from binary_info at load — used at every
     * execute() call. We mutate clientBuf.{data,dataSize} per call. */
    Qnn_Tensor_t *in_tensors;    /* n_inputs entries */
    Qnn_Tensor_t *out_tensors;   /* n_outputs entries */

    /* HTP perf-mode state. power_cfg_id=0 means perf-mode wasn't enabled
     * (e.g., older SDK or feature not supported). At destroy we call
     * destroyPowerConfigId only if it's non-zero. */
    uint32_t power_cfg_id;
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

/* Resolve the per-graph Qnn_Tensor_t arrays from the versioned
 * binary_info struct. Returns 0 on success, -1 on unsupported version. */
static int resolve_graph_info(const QnnSystemContext_BinaryInfo_t *bi,
                              const char **out_graph_name,
                              const Qnn_Tensor_t **out_inputs,
                              uint32_t *out_n_in,
                              const Qnn_Tensor_t **out_outputs,
                              uint32_t *out_n_out) {
    /* binary_info is versioned. We pick graph 0 in either layout. */
    const QnnSystemContext_GraphInfo_t *graph0 = NULL;
    if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        if (bi->contextBinaryInfoV1.numGraphs == 0) return -1;
        graph0 = &bi->contextBinaryInfoV1.graphs[0];
    } else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        if (bi->contextBinaryInfoV2.numGraphs == 0) return -1;
        graph0 = &bi->contextBinaryInfoV2.graphs[0];
    } else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        if (bi->contextBinaryInfoV3.numGraphs == 0) return -1;
        graph0 = &bi->contextBinaryInfoV3.graphs[0];
    } else {
        fprintf(stderr, "[sp_qnn] unknown binary_info version: %d\n", (int)bi->version);
        return -1;
    }
    /* Per-graph info also versioned. */
    if (graph0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        *out_graph_name = graph0->graphInfoV1.graphName;
        *out_inputs     = graph0->graphInfoV1.graphInputs;
        *out_n_in       = graph0->graphInfoV1.numGraphInputs;
        *out_outputs    = graph0->graphInfoV1.graphOutputs;
        *out_n_out      = graph0->graphInfoV1.numGraphOutputs;
    } else if (graph0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        *out_graph_name = graph0->graphInfoV2.graphName;
        *out_inputs     = graph0->graphInfoV2.graphInputs;
        *out_n_in       = graph0->graphInfoV2.numGraphInputs;
        *out_outputs    = graph0->graphInfoV2.graphOutputs;
        *out_n_out      = graph0->graphInfoV2.numGraphOutputs;
    } else if (graph0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        *out_graph_name = graph0->graphInfoV3.graphName;
        *out_inputs     = graph0->graphInfoV3.graphInputs;
        *out_n_in       = graph0->graphInfoV3.numGraphInputs;
        *out_outputs    = graph0->graphInfoV3.graphOutputs;
        *out_n_out      = graph0->graphInfoV3.numGraphOutputs;
    } else {
        return -1;
    }
    return 0;
}

/* Convert a Qnn_DataType_t to bytes per element (rough; QNN has more types). */
static uint32_t dtype_bytes(Qnn_DataType_t dt) {
    switch (dt) {
    case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32: case QNN_DATATYPE_UFIXED_POINT_32:
        return 4;
    case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16:
        return 2;
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
        return 1;
    case QNN_DATATYPE_FLOAT_64: case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64:
        return 8;
    default:
        return 0;
    }
}

/* Cache one tensor template into our flat sp_qnn_tensor_info struct. */
static void cache_tensor_info(const Qnn_Tensor_t *t, sp_qnn_tensor_info *out) {
    if (t->version == QNN_TENSOR_VERSION_1) {
        out->name              = t->v1.name;
        out->rank              = t->v1.rank;
        out->dims              = t->v1.dimensions;
        out->dtype             = (uint32_t)t->v1.dataType;
        out->bytes_per_element = dtype_bytes(t->v1.dataType);
    } else {
        out->name              = t->v2.name;
        out->rank              = t->v2.rank;
        out->dims              = t->v2.dimensions;
        out->dtype             = (uint32_t)t->v2.dataType;
        out->bytes_per_element = dtype_bytes(t->v2.dataType);
    }
}

/* Enable HTP burst mode (DCVS off, perf governor pinned high) for the
 * given device. Match qnn-net-run's defaults. Sets h->power_cfg_id on
 * success so destroy can clean up. Best-effort: any failure is logged
 * but doesn't fail the whole load (we can still execute, just slower). */
static void htp_enable_burst_mode(sp_qnn_handle *h) {
    QnnDevice_Infrastructure_t infra_raw = NULL;
    if (g_lib.qnn.deviceGetInfrastructure(&infra_raw) != QNN_SUCCESS || !infra_raw) {
        fprintf(stderr, "[sp_qnn] perf-mode: deviceGetInfrastructure failed (skipping burst)\n");
        return;
    }
    QnnHtpDevice_Infrastructure_t *infra = (QnnHtpDevice_Infrastructure_t *)infra_raw;
    if (infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        fprintf(stderr, "[sp_qnn] perf-mode: not an HTP perf-infra (type=%d)\n",
                (int)infra->infraType);
        return;
    }
    QnnHtpDevice_PerfInfrastructure_t *perf = &infra->perfInfra;

    /* deviceId=0 / coreId=0 are the defaults; all our binaries target
     * the single on-chip HTP. */
    uint32_t cfg_id = 0;
    if (perf->createPowerConfigId(0, 0, &cfg_id) != QNN_SUCCESS || cfg_id == 0) {
        fprintf(stderr, "[sp_qnn] perf-mode: createPowerConfigId failed\n");
        return;
    }

    /* DCVS V3 burst mode: disable DCVS, lock to PERFORMANCE_MODE,
     * minimal sleep latency, voltage corners pegged to TURBO. The
     * voltage corner enum is defined in QnnHtpPerfInfrastructure.h;
     * value 7 (DCVS_VOLTAGE_VCORNER_TURBO) is the canonical pin-high. */
    QnnHtpPerfInfrastructure_PowerConfig_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    cfg.dcvsV3Config.contextId          = cfg_id;
    cfg.dcvsV3Config.setDcvsEnable      = 1;
    cfg.dcvsV3Config.dcvsEnable         = 0;  /* disable DCVS — lock high */
    cfg.dcvsV3Config.powerMode          = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    cfg.dcvsV3Config.setSleepLatency    = 1;
    cfg.dcvsV3Config.sleepLatency       = 40;  /* 40us — same as qnn-net-run */
    cfg.dcvsV3Config.setBusParams       = 1;
    cfg.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.setCoreParams      = 1;
    cfg.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO;

    const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = { &cfg, NULL };
    if (perf->setPowerConfig(cfg_id, cfgs) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] perf-mode: setPowerConfig failed\n");
        perf->destroyPowerConfigId(cfg_id);
        return;
    }
    h->power_cfg_id = cfg_id;
    fprintf(stderr, "[sp_qnn] HTP burst mode enabled (cfg_id=%u)\n", cfg_id);
}

static void htp_disable_burst_mode(sp_qnn_handle *h) {
    if (h->power_cfg_id == 0) return;
    QnnDevice_Infrastructure_t infra_raw = NULL;
    if (g_lib.qnn.deviceGetInfrastructure(&infra_raw) == QNN_SUCCESS && infra_raw) {
        QnnHtpDevice_Infrastructure_t *infra =
            (QnnHtpDevice_Infrastructure_t *)infra_raw;
        if (infra->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
            infra->perfInfra.destroyPowerConfigId(h->power_cfg_id);
        }
    }
    h->power_cfg_id = 0;
}

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

    /* (2) QnnSystemContext to extract binary info (graphs, tensors). */
    if (g_lib.qsys.systemContextCreate(&h->sysCtx) != QNN_SUCCESS) {
        free(h->bin_data); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }
    Qnn_ContextBinarySize_t info_size = 0;
    if (g_lib.qsys.systemContextGetBinaryInfo(
            h->sysCtx, h->bin_data, h->bin_size,
            &h->binary_info, &info_size) != QNN_SUCCESS) {
        g_lib.qsys.systemContextFree(h->sysCtx);
        free(h->bin_data); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }

    /* (3) Resolve the graph + tensor templates from binary_info. */
    const char *bin_graph_name = NULL;
    const Qnn_Tensor_t *bin_inputs = NULL, *bin_outputs = NULL;
    uint32_t n_in = 0, n_out = 0;
    if (resolve_graph_info(h->binary_info, &bin_graph_name,
                           &bin_inputs, &n_in,
                           &bin_outputs, &n_out) != 0) {
        fprintf(stderr, "[sp_qnn] unsupported binary_info version\n");
        g_lib.qsys.systemContextFree(h->sysCtx);
        free(h->bin_data); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }
    fprintf(stderr, "[sp_qnn] graph '%s': %u inputs / %u outputs\n",
            bin_graph_name ? bin_graph_name : "(unnamed)", n_in, n_out);

    /* (4) Create log handle (NULL callback = stderr-default). */
    if (g_lib.qnn.logCreate(NULL, QNN_LOG_LEVEL_WARN, &h->log) != QNN_SUCCESS) {
        h->log = NULL;  /* non-fatal — some backends accept NULL log */
    }

    /* (5) Create backend. */
    if (g_lib.qnn.backendCreate(h->log, NULL, &h->backend) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] backendCreate failed\n");
        if (h->log) g_lib.qnn.logFree(h->log);
        g_lib.qsys.systemContextFree(h->sysCtx);
        free(h->bin_data); free(h);
        return SP_QNN_ERR_BACKEND_CREATE;
    }

    /* (6) Create device. NULL config -> default HTP V69 picks itself up
     *     from the loaded libQnnHtp.so. Perf-mode tuning is a follow-up. */
    if (g_lib.qnn.deviceCreate(h->log, NULL, &h->device) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] deviceCreate failed\n");
        g_lib.qnn.backendFree(h->backend);
        if (h->log) g_lib.qnn.logFree(h->log);
        g_lib.qsys.systemContextFree(h->sysCtx);
        free(h->bin_data); free(h);
        return SP_QNN_ERR_DEVICE_CREATE;
    }

    /* Enable HTP burst mode before context create so the graph
     * finalize step also runs at high clocks. Best-effort. */
    htp_enable_burst_mode(h);

    /* (7) Create context FROM the binary with HIGH priority hint.
     *     contextCreateFromBinary takes a NULL-terminated array of
     *     QnnContext_Config_t* — same convention as setPowerConfig. */
    QnnContext_Config_t prio_cfg;
    memset(&prio_cfg, 0, sizeof(prio_cfg));
    prio_cfg.option   = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
    prio_cfg.priority = QNN_PRIORITY_HIGH;
    const QnnContext_Config_t *ctx_cfgs[] = { &prio_cfg, NULL };

    if (g_lib.qnn.contextCreateFromBinary(h->backend, h->device,
                                          ctx_cfgs,
                                          h->bin_data, h->bin_size,
                                          &h->context,
                                          NULL /*profile*/) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] contextCreateFromBinary failed\n");
        g_lib.qnn.deviceFree(h->device);
        g_lib.qnn.backendFree(h->backend);
        if (h->log) g_lib.qnn.logFree(h->log);
        g_lib.qsys.systemContextFree(h->sysCtx);
        free(h->bin_data); free(h);
        return SP_QNN_ERR_CONTEXT_CREATE;
    }

    /* (8) Retrieve the named graph (or first). */
    const char *gname_use = graph_name ? graph_name : bin_graph_name;
    if (g_lib.qnn.graphRetrieve(h->context, gname_use, &h->graph) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphRetrieve('%s') failed\n", gname_use);
        g_lib.qnn.contextFree(h->context, NULL);
        g_lib.qnn.deviceFree(h->device);
        g_lib.qnn.backendFree(h->backend);
        if (h->log) g_lib.qnn.logFree(h->log);
        g_lib.qsys.systemContextFree(h->sysCtx);
        free(h->bin_data); free(h);
        return SP_QNN_ERR_GRAPH_RETRIEVE;
    }

    /* (9) Cache I/O tensor templates and metadata. We allocate three
     *     parallel arrays per side: the public sp_qnn_tensor_info, and
     *     the internal Qnn_Tensor_t templates (copied from binary_info
     *     so the originals can be freed when sysCtx is freed). */
    h->n_inputs   = n_in;
    h->n_outputs  = n_out;
    h->inputs     = (sp_qnn_tensor_info *)calloc(n_in,  sizeof(*h->inputs));
    h->outputs    = (sp_qnn_tensor_info *)calloc(n_out, sizeof(*h->outputs));
    h->in_tensors = (Qnn_Tensor_t *)calloc(n_in,  sizeof(*h->in_tensors));
    h->out_tensors= (Qnn_Tensor_t *)calloc(n_out, sizeof(*h->out_tensors));
    if ((n_in > 0 && (!h->inputs || !h->in_tensors)) ||
        (n_out > 0 && (!h->outputs || !h->out_tensors))) {
        sp_qnn_destroy(&h);
        return SP_QNN_ERR_INVALID;
    }
    for (uint32_t i = 0; i < n_in;  ++i) {
        h->in_tensors[i]  = bin_inputs[i];   /* shallow copy template */
        cache_tensor_info(&bin_inputs[i],  &h->inputs[i]);
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        h->out_tensors[i] = bin_outputs[i];
        cache_tensor_info(&bin_outputs[i], &h->outputs[i]);
    }

    fprintf(stderr, "[sp_qnn] context + graph ready\n");
    *out_h = h;
    return SP_QNN_OK;
}

void sp_qnn_destroy(sp_qnn_handle **h_io) {
    if (!h_io || !*h_io) return;
    sp_qnn_handle *h = *h_io;
    /* Release in reverse-creation order. Each call is no-op-safe on
     * NULL/zero handles since we calloc'd the struct. */
    htp_disable_burst_mode(h);
    if (h->context) g_lib.qnn.contextFree(h->context, NULL);
    if (h->device)  g_lib.qnn.deviceFree(h->device);
    if (h->backend) g_lib.qnn.backendFree(h->backend);
    if (h->log)     g_lib.qnn.logFree(h->log);
    if (h->sysCtx)  g_lib.qsys.systemContextFree(h->sysCtx);

    free(h->in_tensors);
    free(h->out_tensors);
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

/* Set the (data, dataSize) on a tensor that's a copy of a binary_info
 * template. Handles version 1 vs 2 layouts. */
static void tensor_set_buf(Qnn_Tensor_t *t, void *data, size_t bytes) {
    if (t->version == QNN_TENSOR_VERSION_1) {
        t->v1.memType            = QNN_TENSORMEMTYPE_RAW;
        t->v1.clientBuf.data     = data;
        t->v1.clientBuf.dataSize = (uint32_t)bytes;
    } else {
        t->v2.memType            = QNN_TENSORMEMTYPE_RAW;
        t->v2.clientBuf.data     = data;
        t->v2.clientBuf.dataSize = (uint32_t)bytes;
    }
}

sp_qnn_status sp_qnn_execute(sp_qnn_handle *h,
                             const void *const *inputs,
                             const size_t *input_sizes,
                             void *const *outputs,
                             const size_t *output_sizes,
                             uint64_t *out_exec_us) {
    if (!h) return SP_QNN_ERR_INVALID;
    if (!inputs && h->n_inputs)  return SP_QNN_ERR_INVALID;
    if (!outputs && h->n_outputs) return SP_QNN_ERR_INVALID;

    /* Bind the supplied buffers into our cached templates. The rest of
     * the tensor metadata (name, id, dataType, dims, type APP_WRITE/READ)
     * came from binary_info via the shallow copy at load time, so we
     * only need to update the data pointer + size per call. */
    for (size_t i = 0; i < h->n_inputs; ++i) {
        tensor_set_buf(&h->in_tensors[i],  (void *)inputs[i],  input_sizes[i]);
    }
    for (size_t i = 0; i < h->n_outputs; ++i) {
        tensor_set_buf(&h->out_tensors[i], outputs[i],         output_sizes[i]);
    }

    /* Wall-clock the QNN call. */
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    Qnn_ErrorHandle_t rc = g_lib.qnn.graphExecute(
        h->graph,
        h->in_tensors,  (uint32_t)h->n_inputs,
        h->out_tensors, (uint32_t)h->n_outputs,
        NULL /*profile*/, NULL /*signal*/);
    gettimeofday(&t1, NULL);

    if (out_exec_us) {
        *out_exec_us = (uint64_t)((t1.tv_sec  - t0.tv_sec)  * 1000000) +
                       (uint64_t)((t1.tv_usec - t0.tv_usec));
    }

    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphExecute failed: 0x%lx\n", (unsigned long)rc);
        return SP_QNN_ERR_EXECUTE;
    }
    return SP_QNN_OK;
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
