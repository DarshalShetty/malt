#lang racket

(require ffi/cvector
         ffi/unsafe
         opencl/c
         "0-vectors.rkt"
         "../../impl-loader.rkt"
         "../../utils.rkt")

;; TODO: Implement MNIST as an example along with iris and morse

(define local-work-size (make-parameter #f))

(define context
  (let ([context #f])
    (λ ()
      (or context
          (begin
            (set! context (clCreateContext #f (cvector->vector (devices))))
            context)))))
(define command-queue
  (let ([command-queue #f])
    (λ ()
      (or command-queue
          (begin
            (set! command-queue (clCreateCommandQueue (context) (device) '()))
            command-queue)))))

(define old-exit-handler (exit-handler))
(exit-handler
 (λ (v)
   (when (command-queue)
     (clReleaseCommandQueue (command-queue)))
   (when (context)
     (clReleaseContext (context)))
   (old-exit-handler v)))

(define platform
  (let ([platform #f])
    (lambda ()
      (or platform
          (begin
            (set! platform (cvector-ref (clGetPlatformIDs:vector) 0))
            platform)))))
(define devices
  (let ([devices #f])
    (lambda ()
      (or devices
          (begin
            (set! devices (clGetDeviceIDs:vector (platform) (opencl-device-type)))
            devices)))))
(define device
  (let ([device #f])
    (lambda ()
      (or device
          (begin
            (set! device (cvector-ref (devices) 0))
            device)))))

(define (cvector->vector cv)
  (build-vector (cvector-length cv)
                (curry cvector-ref cv)))

;; callback function to be used for debugging clBuildProgram if we expose that
;; parameter in the opencl/c library source code.
(define print-cl-build-log
  (λ (program _)
    (when (debug-kernel?)
      (printf "Program Source:~n~a~n"
              (clGetProgramInfo:generic program 'CL_PROGRAM_SOURCE))
      (printf "Build status:~a~n"
              (clGetProgramBuildInfo:generic program (device)
                                             'CL_PROGRAM_BUILD_STATUS))
      (printf "Build log:~a~n"
              (clGetProgramBuildInfo:generic program (device)
                                             'CL_PROGRAM_BUILD_LOG)))))

(define (run-prim1-ρ! kernel-code ker-name
                      v0 off0 size0 stride0
                      v-out size-out stride-out)
  (when (debug-kernel?)
    (printf "Number of GPU threads: ~a~n" (/ size-out stride-out))
    (printf "Input size: ~a~n" size0)
    (printf "Output size: ~a~n" size-out))
  (let* ([buf0 #f]
         [buf-out #f]
         [program #f]
         [kernel #f]
         [event #f])
    (dynamic-wind
     (λ ()
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf-out (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                     (* (ctype-sizeof _cl_float)
                                        size-out)
                                     #f))
       (set! program (clCreateProgramWithSource (context)
                                                (make-vector
                                                 1
                                                 (string->bytes/utf-8
                                                  kernel-code))))
       (clBuildProgram^ program (vector (device)) (make-bytes 0) print-cl-build-log #f)
       (set! kernel (clCreateKernel program (string->bytes/utf-8 ker-name)))
       (clSetKernelArg:_cl_mem kernel 0 buf0)
       (clSetKernelArg:_cl_int kernel 1 stride0)
       (clSetKernelArg:_cl_mem kernel 2 buf-out)
       (clSetKernelArg:_cl_int kernel 3 stride-out))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 (/ size-out stride-out))
                                           (if (local-work-size) (make-vector 1 (local-work-size)) (make-vector 0))
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-out 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size-out)
                                        (vec->cpointer v-out) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-out
         (clReleaseMemObject buf-out))
       (when buf0
         (clReleaseMemObject buf0))))))

(define (run-prim1-∇! kernel-code ker-name g0
                      v0 off0 size0 stride0
                      vz offz size-z stride-z)
  (when (debug-kernel?)
    (printf "Number of GPU threads: ~a~n" (/ size-z stride-z))
    (printf "Input size: ~a~n" size0)
    (printf "Output size: ~a~n" size-z))
  (let* ([buf0 #f]
         [buf-z #f]
         [buf-g #f]
         [program #f]
         [kernel #f]
         [event #f])
    (dynamic-wind
     (λ ()
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf-z (clCreateBuffer (context)
                                   '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                   (* (ctype-sizeof _cl_float)
                                      size-z)
                                   (vref-cpointer vz offz)))
       (set! buf-g (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                   (* (ctype-sizeof _cl_float)
                                      size0)
                                   #f))
       (set! program (clCreateProgramWithSource (context)
                                                (make-vector
                                                 1
                                                 (string->bytes/utf-8
                                                  kernel-code))))
       (clBuildProgram^ program (vector (device)) (make-bytes 0) print-cl-build-log #f)
       (set! kernel (clCreateKernel program (string->bytes/utf-8 ker-name)))
       (clSetKernelArg:_cl_mem kernel 0 buf-g)
       (clSetKernelArg:_cl_mem kernel 1 buf0)
       (clSetKernelArg:_cl_int kernel 2 stride0)
       (clSetKernelArg:_cl_mem kernel 3 buf-z)
       (clSetKernelArg:_cl_int kernel 4 stride-z))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 (/ size-z stride-z))
                                           (if (local-work-size) (make-vector 1 (local-work-size)) (make-vector 0))
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size0)
                                        (vec->cpointer g0) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-g
         (clReleaseMemObject buf-g))
       (when buf-z
         (clReleaseMemObject buf-z))
       (when buf0
         (clReleaseMemObject buf0))))))

(define (run-prim2-ρ! kernel-code ker-name
                      v0 off0 size0 stride0
                      v1 off1 size1 stride1
                      v-out size-out stride-out)
  (when (debug-kernel?)
    (printf "Number of GPU threads: ~a~n" (/ size-out stride-out))
    (printf "Input 0 size: ~a~n" size0)
    (printf "Input 1 size: ~a~n" size1)
    (printf "Output size: ~a~n" size-out))
  (let* ([buf0 #f]
         [buf1 #f]
         [buf-out #f]
         [program #f]
         [kernel #f]
         [event #f])
    (dynamic-wind
     (λ ()
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf1 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size1)
                                  (vref-cpointer v1 off1)))
       (set! buf-out (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                     (* (ctype-sizeof _cl_float)
                                        size-out)
                                     #f))
       (set! program (clCreateProgramWithSource
                      (context)
                      (make-vector
                       1
                       (string->bytes/utf-8 kernel-code))))
       (clBuildProgram^ program (vector (device)) (make-bytes 0) print-cl-build-log #f)
       (set! kernel (clCreateKernel program (string->bytes/utf-8 ker-name)))
       (clSetKernelArg:_cl_mem kernel 0 buf0)
       (clSetKernelArg:_cl_int kernel 1 stride0)
       (clSetKernelArg:_cl_mem kernel 2 buf1)
       (clSetKernelArg:_cl_int kernel 3 stride1)
       (clSetKernelArg:_cl_mem kernel 4 buf-out)
       (clSetKernelArg:_cl_int kernel 5 stride-out))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 (/ size-out stride-out))
                                           (if (local-work-size) (make-vector 1 (local-work-size)) (make-vector 0))
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-out 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size-out)
                                        (vec->cpointer v-out) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-out
         (clReleaseMemObject buf-out))
       (when buf1
         (clReleaseMemObject buf1))
       (when buf0
         (clReleaseMemObject buf0))))))

(define (run-prim2-∇! kernel-code ker-name g0 g1
                      v0 off0 size0 stride0
                      v1 off1 size1 stride1
                      vz offz size-z stride-z)
  (when (debug-kernel?)
    (printf "Number of GPU threads: ~a~n" (max (/ size0 stride0)
                                             (/ size1 stride1)))
    (printf "Input 0 size: ~a~n" size0)
    (printf "Input 1 size: ~a~n" size1)
    (printf "Output size: ~a~n" size-z))
  (let* ([global-work-size (max (/ size0 stride0)
                                (/ size1 stride1))]
         [buf0 #f]
         [buf1 #f]
         [buf-z #f]
         [buf-g0 #f]
         [buf-g1 #f]
         [program #f]
         [kernel #f]
         [event #f])
    (dynamic-wind
     (λ ()
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf1 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size1)
                                  (vref-cpointer v1 off1)))
       (set! buf-z (clCreateBuffer (context)
                                   '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                   (* (ctype-sizeof _cl_float)
                                      size-z)
                                   (vref-cpointer vz offz)))
       (set! buf-g0 (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                    (* (ctype-sizeof _cl_float)
                                       size0)
                                    #f))
       (set! buf-g1 (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                    (* (ctype-sizeof _cl_float)
                                       size1)
                                    #f))
       (set! program (clCreateProgramWithSource
                      (context)
                      (make-vector 1 (string->bytes/utf-8 kernel-code))))
       (clBuildProgram^ program (vector (device)) (make-bytes 0) print-cl-build-log #f)
       (set! kernel (clCreateKernel program (string->bytes/utf-8 ker-name)))
       (clSetKernelArg:_cl_mem kernel 0 buf-g0)
       (clSetKernelArg:_cl_mem kernel 1 buf-g1)
       (clSetKernelArg:_cl_mem kernel 2 buf0)
       (clSetKernelArg:_cl_int kernel 3 stride0)
       (clSetKernelArg:_cl_int kernel 4 size0)
       (clSetKernelArg:_cl_mem kernel 5 buf1)
       (clSetKernelArg:_cl_int kernel 6 stride1)
       (clSetKernelArg:_cl_int kernel 7 size1)
       (clSetKernelArg:_cl_mem kernel 8 buf-z)
       (clSetKernelArg:_cl_int kernel 9 stride-z))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 global-work-size)
                                           (if (local-work-size) (make-vector 1 (local-work-size)) (make-vector 0))
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g0 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size0)
                                        (vec->cpointer g0) (vector event)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g1 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size1)
                                        (vec->cpointer g1) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-g1
         (clReleaseMemObject buf-g1))
       (when buf-g0
         (clReleaseMemObject buf-g0))
       (when buf-z
         (clReleaseMemObject buf-z))
       (when buf1
         (clReleaseMemObject buf1))
       (when buf0
         (clReleaseMemObject buf0))))))

(include "test/test-2-acc-runtime.rkt")

(provide run-prim1-ρ! run-prim1-∇! run-prim2-ρ! run-prim2-∇! local-work-size)
