#lang racket
(require "0-vectors.ss")
(require "1-flats.ss")

(define functional->preallocated-1-ρ
  (λ (f base-shape out-shape)
    (λ (v0 i0 stride0 v-out i-out stride-out)
      (set-prealloc-ρ! v-out i-out out-shape
        (f (arg-value base-shape v0 i0))))))

(define functional->preallocated-1-∇
  (λ (f base-shape out-shape)
    (λ (g0 v0 i0 stride0 vz iz stride-z)
      (let ((z (arg-value out-shape vz iz))
            (a (arg-value base-shape v0 i0)))
        (set-prealloc-∇! g0 i0 base-shape (f a z))))))

(define set-prealloc-ρ!
  (λ (v-out i-out out-shape a)
    (cond
      ((null? out-shape) (vset! v-out i-out a))
      (else
       (v-copy-flat! v-out i-out a)))))

(define set-prealloc-∇!
  (λ (v-out i-out out-shape a)
    (cond
      ((null? out-shape) (vset! v-out i-out (+ (vref v-out i-out) a)))
      (else
       (v-add-flat! v-out i-out a)))))

(define arg-value
  (λ (v-shape v i)
    (cond
      ((null? v-shape) (vref v i))
      (else
       (error 'ρ-functional-non-scalar-in
              (string-append "Functional primitives can only accept scalars,"
                             " so try defining a preallocated primitive"
                             " instead. Input shape found: ~a")
              v-shape)
       #;(flat v-shape v i)))))

(define v-copy-flat!
  (λ (vg ig a)
    ;; copy elements from a to vg
    (let ((va (flat-store a))
          (a-offset (flat-offset a))
          (a-stride (size-of (flat-shape a))))
      (for ([i (in-range 0 a-stride)])
        (vset! vg (+ ig i)
               (vref va (+ a-offset i)))))))

(define v-add-flat!
  (λ (vg ig a)
    ;; copy elements to a to vg while adding them to vg
    (let ((va (flat-store a))
          (a-offset (flat-offset a))
          (a-stride (size-of (flat-shape a))))
      (for ([i (in-range 0 a-stride)])
        (vset! vg (+ ig i)
               (+ (vref vg (+ ig i))
                  (vref va (+ a-offset i))))))))

(define functional->preallocated-2-ρ
  (λ (f t-shape u-shape out-shape)
    (λ (v0 i0 stride0 v1 i1 stride1 v-out i-out stride-out)
      (set-prealloc-ρ! v-out i-out out-shape
        (f (arg-value t-shape v0 i0)
           (arg-value u-shape v1 i1))))))

(define functional->preallocated-2-∇
  (λ (f t-shape u-shape out-shape)
    (λ (g0 g1 v0 i0 stride0 v1 i1 stride1 vz iz stride-z)
      (let ((z (arg-value out-shape vz iz))
            (a (arg-value t-shape v0 i0))
            (b (arg-value u-shape v1 i1)))
        (let-values (((da db) (f a b z)))
          (set-prealloc-∇! g0 i0 t-shape da)
          (set-prealloc-∇! g1 i1 u-shape db))))))

(provide functional->preallocated-1-ρ functional->preallocated-1-∇
         functional->preallocated-2-ρ functional->preallocated-2-∇)
