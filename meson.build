project('DFTTest', 'cpp',
  default_options: ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++17'],
  meson_version: '>=0.48.0',
  version: '7'
)

cxx = meson.get_compiler('cpp')

sources = [
  'DFTTest/DFTTest.cpp',
  'DFTTest/DFTTest.h'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)

fftw3f_dep = dependency('fftw3f')

deps = [vapoursynth_dep, fftw3f_dep]

test_fftwf_threads = '''
#include <fftw3.h>
int main() {
    fftwf_init_threads();
    return 0;
}
'''

# Try finding fftw3f_threads library
fftw3f_threads_dep = cxx.find_library('fftw3f_threads', required: false)

# Check if it's macOS and library wasn't found
if not fftw3f_threads_dep.found() and host_machine.system() == 'darwin'
  message('fftw3f_threads not found in default locations, trying Homebrew directory.')
  fftw3f_threads_dep = cxx.find_library('fftw3f_threads', dirs: ['/opt/homebrew/lib'])
endif

if fftw3f_threads_dep.found()
  message('fftw3f_threads library found.')
else
  error('fftw3f_threads library not found.')
endif

deps += fftw3f_threads_dep

libs = []

if host_machine.cpu_family().startswith('x86')
  message('x86')
  add_project_arguments('-fno-math-errno', '-fno-trapping-math', '-DDFTTEST_X86', '-mfpmath=sse', '-msse2', language: 'cpp')

  sources += [
    'DFTTest/DFTTest_SSE2.cpp',
    'DFTTest/VCL2/instrset.h',
    'DFTTest/VCL2/instrset_detect.cpp',
    'DFTTest/VCL2/vector_convert.h',
    'DFTTest/VCL2/vectorclass.h',
    'DFTTest/VCL2/vectorf128.h',
    'DFTTest/VCL2/vectorf256.h',
    'DFTTest/VCL2/vectorf256e.h',
    'DFTTest/VCL2/vectorf512.h',
    'DFTTest/VCL2/vectorf512e.h',
    'DFTTest/VCL2/vectori128.h',
    'DFTTest/VCL2/vectori256.h',
    'DFTTest/VCL2/vectori256e.h',
    'DFTTest/VCL2/vectori512.h',
    'DFTTest/VCL2/vectori512e.h',
    'DFTTest/VCL2/vectori512s.h',
    'DFTTest/VCL2/vectori512se.h',
    'DFTTest/VCL2/vectormath_common.h',
    'DFTTest/VCL2/vectormath_exp.h',
    'DFTTest/VCL2/vectormath_hyp.h',
    'DFTTest/VCL2/vectormath_lib.h',
    'DFTTest/VCL2/vectormath_trig.h'
  ]

  libs += static_library('avx2', 'DFTTest/DFTTest_AVX2.cpp',
    dependencies: deps,
    cpp_args: ['-mavx2', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )

  libs += static_library('avx512', 'DFTTest/DFTTest_AVX512.cpp',
    dependencies: deps,
    cpp_args: ['-mavx512f', '-mavx512vl', '-mavx512bw', '-mavx512dq', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )
elif host_machine.cpu_family().startswith('arm') or host_machine.cpu_family() == 'aarch64'
  message('arm')
  add_project_arguments('-Ofast', '-ftree-vectorize', '-fopenmp', '-fno-math-errno', '-fno-trapping-math', language: 'cpp')
  
  sources += [
    'DFTTest/DFTTest_SSE2.cpp',
    'DFTTest/VCL2/instrset.h',
    'DFTTest/VCL2/sse2neon.h',
    'DFTTest/VCL2/instrset_detect.cpp',
    'DFTTest/VCL2/vector_convert.h',
    'DFTTest/VCL2/vectorclass.h',
    'DFTTest/VCL2/vectorf128.h',
    'DFTTest/VCL2/vectorf256.h',
    'DFTTest/VCL2/vectorf256e.h',
    'DFTTest/VCL2/vectorf512.h',
    'DFTTest/VCL2/vectorf512e.h',
    'DFTTest/VCL2/vectori128.h',
    'DFTTest/VCL2/vectori256.h',
    'DFTTest/VCL2/vectori256e.h',
    'DFTTest/VCL2/vectori512.h',
    'DFTTest/VCL2/vectori512e.h',
    'DFTTest/VCL2/vectori512s.h',
    'DFTTest/VCL2/vectori512se.h',
    'DFTTest/VCL2/vectormath_common.h',
    'DFTTest/VCL2/vectormath_exp.h',
    'DFTTest/VCL2/vectormath_hyp.h',
    'DFTTest/VCL2/vectormath_lib.h',
    'DFTTest/VCL2/vectormath_trig.h'
  ]
else
  message('unknown')
endif

shared_module('dfttest', sources,
  dependencies: deps,
  link_with: libs,
  install: true,
  install_dir: join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
)