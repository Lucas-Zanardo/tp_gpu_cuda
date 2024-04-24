//
// Created by louvtt on 24-4-24.
//

#ifndef TP_GPU_PROFILE_SECTION_H
#define TP_GPU_PROFILE_SECTION_H

#include "defs.h"
#include <string>
#include "CUDA_common.h"

class ProfileSection {
private:
    std::string m_name;
    double m_start;
    bool m_cuda_sync;
public:
    ProfileSection(const std::string &name, bool cuda_sync = false)
            : m_start(my_gettimeofday()),
              m_name(name),
              m_cuda_sync(cuda_sync) {}

    ~ProfileSection() {
        if (m_cuda_sync) cudaDeviceSynchronize();
        double end = my_gettimeofday();
        printf("Section '%s' took %.5f s\n", m_name.c_str(), end - m_start);
    }
};

#ifdef PROFILING
#define profile_scope(name) ProfileSection __section_profile(name)
#define profile_cuda_scope(name) ProfileSection __section_profile(name, true)
#else
#define profile_scope(name)
#define profile_cuda_scope(name)
#endif

#endif //TP_GPU_PROFILE_SECTION_H