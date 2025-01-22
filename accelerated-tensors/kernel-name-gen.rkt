#lang racket

(require file/xxhash32)
(require (only-in "../impl-loader.rkt" max-kernel-name-len))

;; This file encapsulates all the logic needed to generate OpenCL kernel names.
;; The OpenCL runtime gives an error if the kernel name is greater than a
;; certain length. While developing this implementation on a 2017 Macbook pro
;; the limit seemed to be 15 characters. Since kernel name generation logic is
;; in this file, we can raise errors when the kernel name goes above the limit.

(define ensure-kernel-name-len
  (λ (kernel-name)
    (unless (<= (string-length kernel-name) (max-kernel-name-len))
      (error 'ensure-kernel-name-len "Kernel name '~a' exceeded ~a characters. You can try increasing the max-kernel-name-len config parameter, but that might read to OpenCL build errors claiming that the kernel name isn't defined." kernel-name (max-kernel-name-len)))
    kernel-name))


;; Generate fresh signatures for unary primitives.
(define prim1-gensig
  (let ([id -1])
    (λ ()
      (set! id (add1 id))
      (string-append "p1" (~r id #:base 16)))))

;; Generate fresh signatures for binary primitives.
(define prim2-gensig
  (let ([id -1])
    (λ ()
      (set! id (add1 id))
      (string-append "p2" (~r id #:base 16)))))

;; Generate signatures for the forward (ρ) and backward (∇) components of unary
;; or binary extended primitives from the base primitive's signature.
(define ext-gensig
  (λ (prim-sig)
    (values (string-append "r" prim-sig)
            (string-append "n" prim-sig))))


;; Generate fresh primitive signatures for ext1-ρ when signature of base
;; primitive isn't available.
(define ext1-ρ-gensig
  (let ([id -1])
    (λ ()
      (set! id (add1 id))
      (string-append "re1" (~r id #:base 16)))))


;; Generate fresh primitive signatures for ext1-∇ when signature of base
;; primitive isn't available.
(define ext1-∇-gensig
  (let ([id -1])
    (λ ()
      (set! id (add1 id))
      (string-append "ne1" (~r id #:base 16)))))

;; Generate fresh primitive signatures for ext2-ρ when signature of base
;; primitive isn't available.
(define ext2-ρ-gensig
  (let ([id -1])
    (λ ()
      (set! id (add1 id))
      (string-append "re2" (~r id #:base 16)))))


;; Generate fresh primitive signatures for ext2-∇ when signature of base
;; primitive isn't available.
(define ext2-∇-gensig
  (let ([id -1])
    (λ ()
      (set! id (add1 id))
      (string-append "ne2" (~r id #:base 16)))))

;; The following functions generate OpenCL kernel names from primitive
;; signatures (and sometimes other parameters)

;; TODO: Use a longer hash function to generate a 128-bit hash (with first
;; nibble set to "a") out of the current kernel names so that the kernel length
;; is guaranteed to be within out limits. We'll worry about collisions later.

(define bytes->xxh32-digest
  (λ bs
    (xxh32-reset! xxh32-ctx 0)
    (for ((b bs))
      (xxh32-update! xxh32-ctx b))
    (xxh32-digest xxh32-ctx)))

(define ext1-kernel-name
  (λ (prim-sig)
    #;
    (let ((prim-digest (bytes->xxh32-digest (string->bytes/utf-8 prim-sig))))
      (cond
        ((or (string-prefix? prim-sig "re1")
             (string-prefix? prim-sig "rp1"))
         (format "r~a" (~r prim-digest #:base 16)))
        ((or (string-prefix? prim-sig "ne1")
             (string-prefix? prim-sig "np1"))
         (format "n~a" (~r prim-digest #:base 16)))
        (else
         (error 'ext1-kernel-name
                "Primitive signature should begin with 're1', 'rp1', 'ne1' or 'np1'. Found ~a"
                prim-sig))))
    (ensure-kernel-name-len prim-sig)))

(define ext2-ρ-kernel-name
  (λ (prim-sig strides-sig)
    #;
    (cond
      ((or (string-prefix? prim-sig "re2")
           (string-prefix? prim-sig "rp2"))
       (let ((prim-digest (bytes->xxh32-digest
                           (string->bytes/utf-8 prim-sig)
                           (string->bytes/utf-8 strides-sig))))
         (format "R~a" (~r prim-digest #:base 16))))
      (else
         (error 'ext2-ρ-kernel-name
                "Primitive signature should begin with 're2' or 'rp2'. Found ~a"
                prim-sig)))
    (ensure-kernel-name-len (format "~a~a" prim-sig strides-sig))))

(define xxh32-ctx (make-xxh32))

#;
(define ext2-∇-kernel-name
  (λ (prim-sig strides-sig s0 s1 r0 r1 s-out r-out)
    (cond
      ((or (string-prefix? prim-sig "ne2")
           (string-prefix? prim-sig "np2"))
       (let ((prim-digest (bytes->xxh32-digest
                           (string->bytes/utf-8 prim-sig)
                           (string->bytes/utf-8 strides-sig)
                           (apply bytes-append
                                  (map (λ (x)
                                         (integer->integer-bytes x 4 #f))
                                       s0))
                           (apply bytes-append
                                  (map (λ (x)
                                         (integer->integer-bytes x 4 #f))
                                       s1))
                           (integer->integer-bytes r0 1 #f)
                           (integer->integer-bytes r1 1 #f)
                           (apply bytes-append
                                  (map (λ (x)
                                         (integer->integer-bytes x 4 #f))
                                       s-out))
                           (integer->integer-bytes r-out 1 #f))))
         (format "N~a" (~r prim-digest #:base 16))))
      (else
       (error 'ext2-ρ-kernel-name
              "Primitive signature should begin with 'ne2' or 'np2'. Found ~a"
              prim-sig)))))
(define (ext2-∇-kernel-name prim-sig strides-sig
                            s0 s1 r0 r1 s-out r-out)
  (let ((params-hash (bytes->xxh32-digest
                      (string->bytes/utf-8 strides-sig)
                      (apply bytes-append
                             (map (λ (x)
                                    (integer->integer-bytes x 4 #f))
                                  s0))
                      (apply bytes-append
                             (map (λ (x)
                                    (integer->integer-bytes x 4 #f))
                                  s1))
                      (integer->integer-bytes r0 1 #f)
                      (integer->integer-bytes r1 1 #f)
                      (apply bytes-append
                             (map (λ (x)
                                    (integer->integer-bytes x 4 #f))
                                  s-out))
                      (integer->integer-bytes r-out 1 #f))))
    (ensure-kernel-name-len (format "~a~a" prim-sig params-hash))))

(provide prim1-gensig prim2-gensig
         ext-gensig
         ext1-ρ-gensig ext1-∇-gensig
         ext2-ρ-gensig ext2-∇-gensig
         ext1-kernel-name ext2-ρ-kernel-name ext2-∇-kernel-name)
