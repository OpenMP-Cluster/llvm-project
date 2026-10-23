#define _GNU_SOURCE
#include <cpuid.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

static int mask_width(unsigned count) {
    int width = 0;
    while ((1U << width) < count)
        ++width;
    return width;
}

int main(void) {
    cpu_set_t available;
    if (sched_getaffinity(0, sizeof(available), &available) != 0) {
        perror("sched_getaffinity");
        return 1;
    }

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &available))
            continue;

        cpu_set_t target;
        CPU_ZERO(&target);
        CPU_SET(cpu, &target);

        if (sched_setaffinity(0, sizeof(target), &target) != 0) {
            perror("sched_setaffinity");
            return 1;
        }

        unsigned eax, ebx, ecx, edx;

        __cpuid_count(1, 0, eax, ebx, ecx, edx);
        unsigned apic = (ebx >> 24) & 0xff;
        unsigned threads = (ebx >> 16) & 0xff;
        if (!threads)
            threads = 1;

        __cpuid_count(4, 0, eax, ebx, ecx, edx);
        unsigned cores = ((eax >> 26) & 0x3f) + 1;

        int widthCT = mask_width(threads);
        int widthC = mask_width(cores);
        int widthT = widthCT - widthC;

        if (widthT < 0) {
            fprintf(stderr, "CPU %d: invalid CPUID widths\n", cpu);
            return 1;
        }

        unsigned coreId = (apic >> widthT) & ((1U << widthC) - 1);
        unsigned threadId = apic & ((1U << widthT) - 1);

        printf("CPU=%d APIC=%u threads/pkg=%u cores/pkg=%u "
               "widthCT=%d widthC=%d widthT=%d coreId=%u threadId=%u\n",
               cpu, apic, threads, cores,
               widthCT, widthC, widthT, coreId, threadId);
    }

    return 0;
}
