#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "Metal unavailable\n");
            return 1;
        }
        fprintf(stderr, "Metal device: %s\n", [[dev name] UTF8String]);
    }
    return 0;
}
