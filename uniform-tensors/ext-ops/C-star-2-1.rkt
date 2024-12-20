#lang racket

(require "../tensors/0-vectors.rkt")
(require (only-in "../tensors.rkt" ext2-ρ))
(require "../autodiff.rkt")

(define *-2-1-base-ρ
  (λ (v0 i0 stride0
      v1 i1 stride1
      v-out i-out stride-out)
    (for ([i (in-range 0 stride-out)])
      (vset! v-out (+ i-out i)
        (* (vref v0 (+ i0 i))
           (vref v1 (+ i1 (modulo i stride1))))))))

(define *-2-1-base-∇
  (λ (g0 g1 v0 i0 stride0
      v1 i1 stride1
      vz iz stride-z)
    (for ([i (in-range 0 stride-z)])
      (let ((a (vref v0 (+ i0 i)))
            (b (vref v1 (+ i1 (modulo i stride1))))
            (z (vref vz (+ iz i))))
        (vset! g0 (+ i0 i)
          (+ (vref g0 (+ i0 i)) (* z b)))
        (vset! g1 (+ i1 (modulo  i stride1))
          (+ (vref g1 (+ i1 (modulo i stride1))) (* z a)))))))

(define *-2-1-shape
  (λ (s t)
    s))

(define *-2-1
  (prim2 *-2-1-base-ρ *-2-1-base-∇ *-2-1-shape))

(define d*-2-1
  (ext2 *-2-1 2 1))

(define *-2-1-ρ
  (ext2-ρ *-2-1-base-ρ 2 1 *-2-1-shape))

(include "test/test-C-star-2-1.rkt")

(provide *-2-1-ρ d*-2-1)
