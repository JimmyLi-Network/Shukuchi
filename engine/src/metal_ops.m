#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "metal_ops.h"

struct metal_ops_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> q4k_pipeline;
    id<MTLComputePipelineState> q5k_pipeline;
    id<MTLComputePipelineState> q6k_pipeline;
};

static uint64_t g_metal_calls = 0;
static uint64_t g_cpu_calls = 0;

static const char *load_shader_source(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (len <= 0) {
        fclose(fp);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        fclose(fp);
        free(buf);
        return NULL;
    }
    fclose(fp);
    buf[len] = '\0';
    if (out_len) {
        *out_len = (size_t)len;
    }
    return buf;
}

void *metal_ops_init(const char *shader_path) {
    struct metal_ops_ctx *ctx = (struct metal_ops_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        return NULL;
    }
    @autoreleasepool {
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "[Metal] ERROR: MTLCreateSystemDefaultDevice failed\n");
            free(ctx);
            return NULL;
        }
        fprintf(stderr, "[Metal] Device: %s\n", [[ctx->device name] UTF8String]);
        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "[Metal] ERROR: failed to create command queue\n");
            free(ctx);
            return NULL;
        }
        size_t len = 0;
        const char *src = load_shader_source(shader_path, &len);
        if (!src) {
            fprintf(stderr, "[Metal] ERROR: failed to read shader source: %s\n", shader_path);
            free(ctx);
            return NULL;
        }
        NSString *source = [[NSString alloc] initWithBytes:src length:len encoding:NSUTF8StringEncoding];
        free((void *)src);
        NSError *err = nil;
        ctx->library = [ctx->device newLibraryWithSource:source options:nil error:&err];
        if (!ctx->library) {
            fprintf(stderr, "[Metal] ERROR: failed to compile source: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            free(ctx);
            return NULL;
        }
        id<MTLFunction> fn = [ctx->library newFunctionWithName:@"matmul_q4k_vec"];
        if (!fn) {
            fprintf(stderr, "[Metal] ERROR: function 'matmul_q4k_vec' not found\n");
            free(ctx);
            return NULL;
        }
        ctx->q4k_pipeline = [ctx->device newComputePipelineStateWithFunction:fn error:&err];
        if (!ctx->q4k_pipeline) {
            fprintf(stderr, "[Metal] ERROR: failed to create pipeline: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            free(ctx);
            return NULL;
        }
        id<MTLFunction> fn5 = [ctx->library newFunctionWithName:@"matmul_q5k_vec"];
        if (!fn5) {
            fprintf(stderr, "[Metal] ERROR: function 'matmul_q5k_vec' not found\n");
            free(ctx);
            return NULL;
        }
        ctx->q5k_pipeline = [ctx->device newComputePipelineStateWithFunction:fn5 error:&err];
        if (!ctx->q5k_pipeline) {
            fprintf(stderr, "[Metal] ERROR: failed to create pipeline q5k: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            free(ctx);
            return NULL;
        }
        id<MTLFunction> fn6 = [ctx->library newFunctionWithName:@"matmul_q6k_vec"];
        if (!fn6) {
            fprintf(stderr, "[Metal] ERROR: function 'matmul_q6k_vec' not found\n");
            free(ctx);
            return NULL;
        }
        ctx->q6k_pipeline = [ctx->device newComputePipelineStateWithFunction:fn6 error:&err];
        if (!ctx->q6k_pipeline) {
            fprintf(stderr, "[Metal] ERROR: failed to create pipeline q6k: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            free(ctx);
            return NULL;
        }
    }
    return ctx;
}

void metal_ops_shutdown(void *opaque) {
    struct metal_ops_ctx *ctx = (struct metal_ops_ctx *)opaque;
    if (!ctx) {
        return;
    }
    @autoreleasepool {
        ctx->q4k_pipeline = nil;
        ctx->q5k_pipeline = nil;
        ctx->q6k_pipeline = nil;
        ctx->library = nil;
        ctx->queue = nil;
        ctx->device = nil;
    }
    free(ctx);
}

int metal_matmul_q4k_vec(void *opaque,
                         const void *weights,
                         const float *x,
                         float *y,
                         uint32_t m,
                         uint32_t k) {
    struct metal_ops_ctx *ctx = (struct metal_ops_ctx *)opaque;
    if (!ctx || !weights || !x || !y || m == 0 || k == 0) {
        g_cpu_calls++;
        return -1;
    }
    @autoreleasepool {
        size_t weights_size = (size_t)m * (size_t)(k / 256) * 144;
        size_t x_size = (size_t)k * sizeof(float);
        size_t y_size = (size_t)m * sizeof(float);

        id<MTLBuffer> wbuf = [ctx->device newBufferWithBytesNoCopy:(void *)weights
                                                          length:weights_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!wbuf) {
            wbuf = [ctx->device newBufferWithBytes:weights
                                            length:weights_size
                                           options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> xbuf = [ctx->device newBufferWithBytesNoCopy:(void *)x
                                                          length:x_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!xbuf) {
            xbuf = [ctx->device newBufferWithBytes:x
                                            length:x_size
                                           options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> ybuf = [ctx->device newBufferWithBytesNoCopy:(void *)y
                                                          length:y_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!ybuf) {
            ybuf = [ctx->device newBufferWithLength:y_size
                                            options:MTLResourceStorageModeShared];
        }
        if (!wbuf || !xbuf || !ybuf) {
            g_cpu_calls++;
            return -1;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->q4k_pipeline];
        [enc setBuffer:wbuf offset:0 atIndex:0];
        [enc setBuffer:xbuf offset:0 atIndex:1];
        [enc setBuffer:ybuf offset:0 atIndex:2];
        [enc setBytes:&k length:sizeof(k) atIndex:3];

        MTLSize grid = MTLSizeMake(m, 1, 1);
        NSUInteger tg = ctx->q4k_pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > 256) {
            tg = 256;
        }
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (ybuf.contents != (void *)y) {
            memcpy(y, ybuf.contents, y_size);
        }
    }
    g_metal_calls++;
    return 0;
}

int metal_matmul_q5k_vec(void *opaque,
                         const void *weights,
                         const float *x,
                         float *y,
                         uint32_t m,
                         uint32_t k) {
    struct metal_ops_ctx *ctx = (struct metal_ops_ctx *)opaque;
    if (!ctx || !weights || !x || !y || m == 0 || k == 0) {
        g_cpu_calls++;
        return -1;
    }
    @autoreleasepool {
        size_t weights_size = (size_t)m * (size_t)(k / 256) * 176;
        size_t x_size = (size_t)k * sizeof(float);
        size_t y_size = (size_t)m * sizeof(float);

        id<MTLBuffer> wbuf = [ctx->device newBufferWithBytesNoCopy:(void *)weights
                                                          length:weights_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!wbuf) {
            wbuf = [ctx->device newBufferWithBytes:weights
                                            length:weights_size
                                           options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> xbuf = [ctx->device newBufferWithBytesNoCopy:(void *)x
                                                          length:x_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!xbuf) {
            xbuf = [ctx->device newBufferWithBytes:x
                                            length:x_size
                                           options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> ybuf = [ctx->device newBufferWithBytesNoCopy:(void *)y
                                                          length:y_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!ybuf) {
            ybuf = [ctx->device newBufferWithLength:y_size
                                            options:MTLResourceStorageModeShared];
        }
        if (!wbuf || !xbuf || !ybuf) {
            g_cpu_calls++;
            return -1;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->q5k_pipeline];
        [enc setBuffer:wbuf offset:0 atIndex:0];
        [enc setBuffer:xbuf offset:0 atIndex:1];
        [enc setBuffer:ybuf offset:0 atIndex:2];
        [enc setBytes:&k length:sizeof(k) atIndex:3];

        MTLSize grid = MTLSizeMake(m, 1, 1);
        NSUInteger tg = ctx->q5k_pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > 256) {
            tg = 256;
        }
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (ybuf.contents != (void *)y) {
            memcpy(y, ybuf.contents, y_size);
        }
    }
    g_metal_calls++;
    return 0;
}

int metal_matmul_q6k_vec(void *opaque,
                         const void *weights,
                         const float *x,
                         float *y,
                         uint32_t m,
                         uint32_t k) {
    struct metal_ops_ctx *ctx = (struct metal_ops_ctx *)opaque;
    if (!ctx || !weights || !x || !y || m == 0 || k == 0) {
        g_cpu_calls++;
        return -1;
    }
    @autoreleasepool {
        size_t weights_size = (size_t)m * (size_t)(k / 256) * 210;
        size_t x_size = (size_t)k * sizeof(float);
        size_t y_size = (size_t)m * sizeof(float);

        id<MTLBuffer> wbuf = [ctx->device newBufferWithBytesNoCopy:(void *)weights
                                                          length:weights_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!wbuf) {
            wbuf = [ctx->device newBufferWithBytes:weights
                                            length:weights_size
                                           options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> xbuf = [ctx->device newBufferWithBytesNoCopy:(void *)x
                                                          length:x_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!xbuf) {
            xbuf = [ctx->device newBufferWithBytes:x
                                            length:x_size
                                           options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> ybuf = [ctx->device newBufferWithBytesNoCopy:(void *)y
                                                          length:y_size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!ybuf) {
            ybuf = [ctx->device newBufferWithLength:y_size
                                            options:MTLResourceStorageModeShared];
        }
        if (!wbuf || !xbuf || !ybuf) {
            g_cpu_calls++;
            return -1;
        }

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->q6k_pipeline];
        [enc setBuffer:wbuf offset:0 atIndex:0];
        [enc setBuffer:xbuf offset:0 atIndex:1];
        [enc setBuffer:ybuf offset:0 atIndex:2];
        [enc setBytes:&k length:sizeof(k) atIndex:3];

        MTLSize grid = MTLSizeMake(m, 1, 1);
        NSUInteger tg = ctx->q6k_pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > 256) {
            tg = 256;
        }
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (ybuf.contents != (void *)y) {
            memcpy(y, ybuf.contents, y_size);
        }
    }
    g_metal_calls++;
    return 0;
}

int metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev ? 1 : 0;
    }
}

void metal_ops_report(void) {
    fprintf(stderr, "metal_calls=%llu cpu_calls=%llu\n",
            (unsigned long long)g_metal_calls,
            (unsigned long long)g_cpu_calls);
}
