/*
 * Test driver for libsp_qnn — load our existing AI-Hub-validated
 * v69_attn_qwen3_4b.bin, query metadata, sanity-check the API.
 *
 * Until execute() is filled in (Phase 2.3 next-session work), this
 * proves: dlopen path correct, interface fetch works on real device,
 * binary info parsing succeeds. The execute call returns the explicit
 * EXECUTE_TODO error rather than crashing.
 */
#include "sp_qnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *bin = (argc > 1) ? argv[1] : "v69_attn_qwen3_4b.bin";
    fprintf(stderr, "=== sp_qnn smoke test (binary=%s) ===\n", bin);

    sp_qnn_status rc = sp_qnn_init(NULL, NULL);  /* defaults: libQnnHtp.so / libQnnSystem.so */
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "sp_qnn_init failed: %d\n", rc);
        return 1;
    }

    sp_qnn_handle *h = NULL;
    rc = sp_qnn_load_binary(bin, NULL, &h);
    if (rc != SP_QNN_OK) {
        fprintf(stderr, "sp_qnn_load_binary failed: %d\n", rc);
        sp_qnn_shutdown();
        return 2;
    }
    fprintf(stderr, "load_binary OK\n");

    /* Try execute (will hit the TODO; that's expected at this stage). */
    rc = sp_qnn_execute(h, NULL, NULL, NULL, NULL, NULL);
    fprintf(stderr, "execute returned %d (expected -10 = SP_QNN_ERR_EXECUTE TODO)\n", rc);

    sp_qnn_destroy(&h);
    sp_qnn_shutdown();
    fprintf(stderr, "=== sp_qnn smoke test done ===\n");
    return 0;
}
