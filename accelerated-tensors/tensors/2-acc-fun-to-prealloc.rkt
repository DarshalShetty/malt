#lang racket

(require string-interpolation)

(define functional->preallocated-1-ρ-acc
  (λ (f-acc base-shape out-shape)
    (unless (and (null? base-shape) (null? out-shape))
      (error 'ρ1-functional-non-scalar-acc
             (string-append "Accelerated functional primitives can only accept and"
                            " return scalars, so try defining a"
                            " preallocated primitive instead."
                            " Input and output shape found: ~a ~a")
             base-shape out-shape))
    (λ (v0 i0 stride0 v-out i-out stride-out)
      (let ((a "@{v0}[@{i0}]"))
#<<EOF
    @{v-out}[@{i-out}] = (@{(f-acc a)});
EOF
        ))))

(define functional->preallocated-1-∇-acc
  (λ (f-acc base-shape out-shape)
    (unless (and (null? base-shape) (null? out-shape))
      (error '∇1-functional-non-scalar-acc
             (string-append "Accelerated functional primitives can only accept and"
                            " return scalars, so try defining a"
                            " preallocated primitive instead."
                            " Input and output shape found: ~a ~a")
              base-shape out-shape))
    (λ (g0 v0 i0 stride0 vz iz stride-z)
      (let ((z "@{vz}[@{iz}]")
            (a "@{v0}[@{i0}]"))
#<<EOF
    @{g0}[@{i0}] += (@{(f-acc a z)});
EOF
        ))))

(define functional->preallocated-2-ρ-acc
  (λ (f-acc t-shape u-shape out-shape)
    (unless (and (null? t-shape) (null? u-shape) (null? out-shape))
      (error 'ρ2-functional-non-scalar-acc
             (string-append "Accelerated functional primitives can only accept and"
                            " return scalars, so try defining a"
                            " preallocated primitive instead."
                            " Input 1, input 2 and output shape found: ~a ~a ~a")
              t-shape u-shape out-shape))
    (λ (v0 i0 stride0 v1 i1 stride1 v-out i-out stride-out)
      (let ((a "@{v0}[@{i0}]")
            (b "@{v1}[@{i1}]"))
#<<EOF
    @{v-out}[@{i-out}] = (@{(f-acc a b)});
EOF
        ))))

(define functional->preallocated-2-∇-acc
  (λ (f-acc t-shape u-shape out-shape)
    (unless (and (null? t-shape) (null? u-shape) (null? out-shape))
      (error '∇2-functional-non-scalar-acc
             (string-append "Accelerated functional primitives can only accept and"
                            " return scalars, so try defining a"
                            " preallocated primitive instead."
                            " Input 1, input 2 and output shape found: ~a ~a ~a")
              t-shape u-shape out-shape))
    (λ (g v0 i0 stride0 v1 i1 stride1 vz iz stride-z)
      (let ((z "@{vz}[@{iz}]")
            (a "@{v0}[@{i0}]")
            (b "@{v1}[@{i1}]"))
        (let-values (((da db) (f-acc a b z)))
          (values
#<<EOF
    @{g}[@{i0}] += (@{da});
EOF

#<<EOF
    @{g}[@{i1}] += (@{db});
EOF
           ))))))

(provide functional->preallocated-1-ρ-acc
         functional->preallocated-1-∇-acc
         functional->preallocated-2-ρ-acc
         functional->preallocated-2-∇-acc)
