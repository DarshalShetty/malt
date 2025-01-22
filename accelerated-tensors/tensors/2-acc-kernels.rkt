#lang racket

(require "ext2-strides.rkt" string-interpolation)
(require "../kernel-name-gen.rkt")

(define (binary-expr rator rand1 rand2)
  (string-append "(" rand1 " " rator " " rand2 ")"))

(define idx-exprs
  (λ (strides i0 i1)
    (λ (out-i)
      (for/fold ([i0 (number->string i0)]
                 [i1 (number->string i1)]
                 [x out-i] #:result (values i0 i1))
                ([stride (strides-strides strides)])
        (let ((stride-out (number->string (vector-ref stride 0)))
              (stride0 (number->string (vector-ref stride 1)))
              (stride1 (number->string (vector-ref stride 2))))
          (let ((idx (binary-expr "/" x stride-out))
                (next-x (binary-expr "%" x stride-out)))
            (values (binary-expr "+" i0 (binary-expr "*" idx stride0))
                    (binary-expr "+" i1 (binary-expr "*" idx stride1))
                    next-x)))))))

(define idx-exprs-inv
  (λ (strides i-out repeats0 repeats1 s-out)
    (λ (i0-var-str i1-var-str i-rep-var-str)
      (let ((gen-expr
             (λ (i-in-var-str stride-i repeats)
               (for/fold ([i-out (number->string i-out)]
                          [dividend-rep i-rep-var-str]
                          [predivisor-rep repeats]
                          [x i-in-var-str] #:result i-out)
                         ([desc-out s-out] ;; s-out == (append descents-out sf-out)
                          [stride (strides-strides strides)]) ;; (len strides) == (len descents-out)
                 (let ((stride-out (vector-ref stride 0))
                       (stride-in (vector-ref stride stride-i)))
                   (cond
                     ((zero? stride-in)
                      (let* ((divisor-rep (quotient predivisor-rep desc-out))
                             (divisor-rep-str (number->string divisor-rep))
                             (scaling (binary-expr "/" dividend-rep divisor-rep-str))
                             (next-dividend (binary-expr "%"
                                                         dividend-rep
                                                         divisor-rep-str)))
                        (values (binary-expr "+" i-out
                                             (binary-expr "*"
                                                          scaling
                                                          (number->string
                                                           stride-out)))
                                next-dividend
                                divisor-rep
                                x)))
                     (else
                      (let ((stride-in-str (number->string stride-in)))
                        (let ((idx (binary-expr "/" x stride-in-str))
                              (next-x (binary-expr "%" x stride-in-str)))
                          (values (binary-expr "+" i-out
                                               (binary-expr "*" idx
                                                            (number->string
                                                             stride-out)))
                                  dividend-rep
                                  predivisor-rep
                                  next-x))))))))))
        (values (gen-expr i0-var-str 1 repeats0)
                (gen-expr i1-var-str 2 repeats1))))))

(define (ext1-ρ-kernel/name prim1-ρ-f prim-sig)
  (let ((kernel-name (ext1-kernel-name prim-sig)))
    (values
#<<EOF
__kernel void @{kernel-name} (__global float* v0,
                      int stride0,
                      __global float* v_out,
                      int stride_out)
{

    int i_out = get_global_id(0) * stride_out;
    // offset is handled by the platform API
    int i0 = (i_out / stride_out) * stride0;

@{(prim1-ρ-f "v0" "i0" "stride0" "v_out" "i_out" "stride_out")}

}
EOF
   kernel-name)))

(define (ext1-∇-kernel/name prim1-∇-f prim-sig)
  (let ((kernel-name (ext1-kernel-name prim-sig)))
    (values
#<<EOF
__kernel void @{kernel-name} (__global float* g0,
                      __global float* v0,
                      int stride0,
                      __global float* vz,
                      int stridez)
{

    int iz = get_global_id(0) * stridez;
    // offset is handled by the platform API
    int i0 = 0 + (iz / stridez) * stride0;

@{(prim1-∇-f "g0" "v0" "i0" "stride0"
                  "vz" "iz" "stridez")}
}
EOF
     kernel-name)))

(define ext2-ρ-kernel/name
  (let ((cache (make-hash)))
    (λ (prim2-ρ-f prim-sig strides)
      (let ((kernel-name (ext2-ρ-kernel-name prim-sig (strides-signature strides))))
        (cond
          ((hash-has-key? cache kernel-name)
           (values (hash-ref cache kernel-name) kernel-name))
          (else
           (let*-values (((generate-idxs) (idx-exprs strides 0 0))
                         ((i0-expr i1-expr) (generate-idxs "i_out"))
                         ((kernel-code)
#<<EOF
__kernel void @{kernel-name} (__global float* v0,
                      int stride0,
                      __global float* v1,
                      int stride1,
                      __global float* v_out,
                      int stride_out)
{
    int i_out = get_global_id(0) * stride_out;
    int i0 = @{i0-expr};
    int i1 = @{i1-expr};

@{(prim2-ρ-f "v0" "i0" "stride0"
             "v1" "i1" "stride1"
             "v_out" "i_out" "stride_out")}
}
EOF
                          ))
             (hash-set! cache kernel-name kernel-code)
             (values
              kernel-code
              kernel-name))))))))

(define calc-repeats
  (λ (s0 s1 r0 r1 s-out r-out)
    (let ((size-rep0 (apply * (drop-right s0 r0)))
          (size-rep1 (apply * (drop-right s1 r1)))
          (size-rep-out (apply * (drop-right s-out r-out))))
      (values (/ size-rep-out size-rep0)
              (/ size-rep-out size-rep1)))))

(define ext2-∇-kernel/name
  (let ((cache (make-hash)))
    (λ (prim2-∇-f prim-sig strides
                  s0 s1 r0 r1 s-out r-out)
      (let ((kernel-name (ext2-∇-kernel-name prim-sig (strides-signature strides)
                                             s0 s1 r0 r1 s-out r-out)))
        (cond
          ((hash-has-key? cache kernel-name)
           (values (hash-ref cache kernel-name) kernel-name))
          (else
           (let*-values (((prim-effect0 prim-effect1) (prim2-∇-f "g"
                                                                 "v0" "i0" "stride0"
                                                                 "v1" "i1" "stride1"
                                                                 "vz" "iz" "stride_z"))
                         ((repeats0 repeats1) (calc-repeats s0 s1 r0 r1 s-out r-out))
                         ((generate-idxs) (idx-exprs strides 0 0))
                         ((generate-idxs-inv) (idx-exprs-inv strides 0
                                                             repeats0 repeats1 s-out))
                         ((i0-expr i1-expr) (generate-idxs "iz"))
                         ((iz-expr0 iz-expr1) (generate-idxs-inv "i0" "i1" "i_rep"))
                         ((kernel-code)
#<<EOF
__kernel void @{kernel-name} (__global float* g0,
                      __global float* g1,
                      __global float* v0,
                      int stride0,
                      int size0,
                      __global float* v1,
                      int stride1,
                      int size1,
                      __global float* vz,
                      int stride_z)
{
    int g_id = get_global_id(0);
    int i0_g = g_id * stride0;
    int i1_g = g_id * stride1;
    __global float *g;
    int i0, i1, iz;

    if (i0_g < size0) {
        g = g0;
        i0 = i0_g;
        for(int i_rep=0; i_rep<@{repeats0}; i_rep++) {
            iz = @{iz-expr0};
            i1 = @{i1-expr};

@{prim-effect0}
        }
    }

    if (i1_g < size1) {
        g = g1;
        i1 = i1_g;
        for(int i_rep=0; i_rep<@{repeats1}; i_rep++) {
            iz = @{iz-expr1};
            i0 = @{i0-expr};

@{prim-effect1}
        }
    }
}
EOF
                         ))

             (hash-set! cache kernel-name kernel-code)
             (values
              kernel-code
              kernel-name))))))))

(provide ext1-ρ-kernel/name ext1-∇-kernel/name ext2-ρ-kernel/name ext2-∇-kernel/name)
