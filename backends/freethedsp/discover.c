// Shannon-Prime / freethedsp — discover.c
// Phase D.1: dump-only LD_PRELOAD shim. Captures the in-memory
// fastrpc_shell_N ELF that the kernel maps into our process during
// FASTRPC_IOCTL_INIT, writes it to disk, and lets the original ioctl
// proceed unchanged. No patching at this stage — we just need the
// exact bytes for our S22U cDSP shell so we can locate is_test_enabled
// offline.
//
// Derived from geohot/freethedsp (MIT). Copyright (C) 2026 Ray Daniels.
// Licensed under AGPLv3. Commercial license: raydaniels@gmail.com
//
// Build (cross to aarch64 Android via NDK):
//   $ANDROID_ARM64_TOOLCHAIN/bin/aarch64-linux-android21-clang \
//     -shared -fPIC -O2 -Iinclude discover.c -o discover.so -ldl
//
// Use:
//   adb push discover.so /data/local/tmp/sp_qnn/
//   adb shell "cd /data/local/tmp/sp_qnn; \
//     LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
//     LD_PRELOAD=./discover.so SP_DUMP_PATH=./fastrpc_shell.bin \
//     ./<any FastRPC-using binary> ..."
//   adb pull /data/local/tmp/sp_qnn/fastrpc_shell.bin
//
// Then run tools/find_is_test_enabled.py on it offline.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <unistd.h>
#include <errno.h>

// FastRPC IOCTL definitions — same as in geohot's freethedsp.c.
// We don't include the kernel headers because the relevant struct is
// stable across the FastRPC ABI back to ~2016 and re-declaring is
// safer than depending on the toolchain's headers being present.
#define IOC_OUT   0x40000000
#define IOC_IN    0x80000000
#define IOC_INOUT (IOC_IN | IOC_OUT)
#define IOCPARM_MASK 0x1fff
#define _IOC(inout, group, num, len) \
    (inout | ((len & IOCPARM_MASK) << 16) | ((group) << 8) | (num))
#define _IOWR(g, n, t) _IOC(IOC_INOUT, (g), (n), sizeof(t))

struct fastrpc_ioctl_init {
    uint32_t flags;
    uintptr_t file;
    int32_t  filelen;
    int32_t  filefd;
    uintptr_t mem;     /* the userspace mapping the kernel populated with
                          the per-process fastrpc_shell_N ELF */
    int32_t  memlen;
    int32_t  memfd;
};
#define FASTRPC_IOCTL_INIT _IOWR('R', 6, struct fastrpc_ioctl_init)

// dlsym'd handle to libc's real ioctl
static int (*real_ioctl)(int, unsigned long, void *) = NULL;

static void load_real_ioctl(void) {
    if (real_ioctl) return;
    void *h = dlopen("libc.so", RTLD_LAZY);
    if (!h) h = dlopen("/system/lib64/libc.so", RTLD_LAZY);
    if (!h) h = dlopen("/apex/com.android.runtime/lib64/bionic/libc.so", RTLD_LAZY);
    if (!h) {
        fprintf(stderr, "[discover] dlopen(libc) failed: %s\n", dlerror());
        abort();
    }
    real_ioctl = (int (*)(int, unsigned long, void *))dlsym(h, "ioctl");
    if (!real_ioctl) {
        fprintf(stderr, "[discover] dlsym(ioctl) failed: %s\n", dlerror());
        abort();
    }
}

static void dump_shell(const struct fastrpc_ioctl_init *init) {
    const char *path = getenv("SP_DUMP_PATH");
    if (!path || !*path) path = "./fastrpc_shell.bin";

    if (init->mem == 0 || init->memlen <= 0) {
        fprintf(stderr, "[discover] init->mem=0x%lx len=%d — nothing to dump\n",
                (unsigned long)init->mem, init->memlen);
        return;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[discover] fopen(%s) failed: %s\n", path, strerror(errno));
        return;
    }
    size_t wrote = fwrite((void *)init->mem, 1, (size_t)init->memlen, f);
    fclose(f);
    fprintf(stderr, "[discover] dumped %zu bytes (mem=0x%lx, len=%d) to %s\n",
            wrote, (unsigned long)init->mem, init->memlen, path);
    fprintf(stderr, "[discover] flags=0x%x file=0x%lx filelen=%d filefd=%d memfd=%d\n",
            init->flags, (unsigned long)init->file, init->filelen,
            init->filefd, init->memfd);

    // First 16 bytes — should start with ELF magic 0x7f 'E' 'L' 'F'.
    const unsigned char *m = (const unsigned char *)init->mem;
    fprintf(stderr, "[discover] first 16 bytes:");
    for (int i = 0; i < 16 && i < init->memlen; i++) {
        fprintf(stderr, " %02x", m[i]);
    }
    fprintf(stderr, "\n");
}

int ioctl(int fd, unsigned long request, void *arg) {
    load_real_ioctl();
    int rc = real_ioctl(fd, request, arg);

    if (request == FASTRPC_IOCTL_INIT && arg != NULL) {
        // Even if rc != 0 we dump — the kernel may have populated the mem
        // before the call ultimately failed, and we want the bytes either
        // way.
        dump_shell((const struct fastrpc_ioctl_init *)arg);

        // Set this env var to abort after first successful dump if you
        // want to capture exactly one shell load.
        if (getenv("SP_DUMP_ONCE")) {
            fprintf(stderr, "[discover] SP_DUMP_ONCE set — exiting\n");
            _exit(0);
        }
    }
    return rc;
}
