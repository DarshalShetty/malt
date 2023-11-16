#lang racket

(require "c0-ast.rkt")
(require (only-in "c2-interpreter.rkt" interp-tensor interp-racket))
(require "../../flat-tensors/ext-impl.rkt")
(require (prefix-in flat: "../../flat-tensors/tensors.rkt"))
(require rackunit)
(require file/xxhash32)

(struct counter-data (binding-name
                      ref-count)
  #:transparent)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compiler Passes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;TODO: later eds and gs passes should not be needed because the tcomp AST nodes
;;should have a signature and dss field which will be populated at the time of
;;their instantiation. Then we just access those fields from the AST node rather
;;than computing them. Use a global data segment that has flat tensors used by
;;all tcomp nodes in our program.

;;Extracts the data segment which is a vector that contains
;;  * scalars (arguments to tref)
;;  * flat tensors
;;  * flat tensor¹ of indices that will be the arguments to trefs
;;  * the symbol 'uncalculated as an initial placeholder for the output of
;;    tcomp-ext2-∇ which will be later replaced by the flat tensor output
;;
;; TODO: Remove all uses of build-refs because we longer have tcomp-build-tensor nodes
(define extract-data-segment
  (λ (t)
    (let-values (((t^ data-segment-stack build-refs) (eds-expr t '() (hasheq))))
      ;; convert data segment stack to data segment array
      (values t^
              (list->vector (reverse data-segment-stack))
              build-refs))))

;; Checks if a member equivalent to v exists in dss using equiv? and based on
;; that returns the dss index where v was inserted and the new dss with
;; insertion as values
;; TODO: Reconsider performance impact of this function
(define insert-unless-exists
  (λ (v dss equiv?)
    (cond
      ((member v dss equiv?)
       => (λ (res/rest)
            (values (length (cdr res/rest)) dss)))
      (else (values (length dss) (cons v dss))))))

(define eds-expr
  (λ (t dss build-refs)
    (match t
      (s #:when (number? s)
        (values s dss build-refs))
      (ft
       #:when (flat? ft)
       (let-values (((idx dss^) (insert-unless-exists ft dss eq?)))
         (values (tcomp-ds-ref idx) dss^ build-refs)))
      ((tpromise tc s)
       (let-values (((tc^ dss^ build-refs^) (eds-expr tc dss build-refs)))
         (cond
           ((number? tc^) (values tc^ dss^ build-refs^))
           (else (values (tpromise tc^ s) dss^ build-refs^)))))
      ((tcomp) (eds-tcomp t dss build-refs)))))

(define eds-tcomp
  (λ (tc dss build-refs)
    (match tc
      [(tcomp-list->tensor lst)
       (let-values (((ts dss^ build-refs^)
                     (for/fold ((ts '())
                                (dss^ dss)
                                (build-refs^ build-refs))
                               ((l lst))
                       (let-values (((t dss^^ build-refs^^)
                                     (eds-expr l dss^ build-refs^)))
                         (values (cons t ts) dss^^ build-refs^^)))))
         (values (tcomp-list->tensor (reverse ts)) dss^ build-refs^))]

      [(tcomp-tref tp i)
       (let-values (((t dss^ build-refs^) (eds-expr tp dss build-refs)))
         (let-values (((idx dss^^) (insert-unless-exists i dss^ eqv?)))
           (values (tcomp-tref t (tcomp-ds-ref idx)) dss^^ build-refs^)))]
      [(tcomp-trefs tp i-list)
       (let-values (((t dss^ build-refs^) (eds-expr tp dss build-refs)))
         (let-values (((idx dss^^)
                       ;; Comparison by flat:tensor-equal? is okay because
                       ;; members of b are integers (not reals) and their
                       ;; equality is checked without a tolerance.
                       ;; TODO: Reconsider performance impact of flat:list->tensor.
                       ;; Maybe memoize it.
                       (insert-unless-exists (flat:list->tensor i-list)
                                             dss^
                                             flat:tensor-equal?)))
           (values (tcomp-trefs t (tcomp-ds-ref idx)) dss^^ build-refs^)))]
      [(tcomp-ext2-∇ fᵈ signature r0 r1 shape-fn tp-t0 tp-t1 tp-z out0 out1 i)
       (let-values (((t0 dss^ build-refs^) (eds-expr tp-t0 dss build-refs)))
         (let-values (((t1 dss^^ build-refs^^) (eds-expr tp-t1 dss^ build-refs^)))
           (let-values (((z dss^^^ build-refs^^^) (eds-expr tp-z dss^^ build-refs^^)))
             (values (tcomp-ext2-∇ fᵈ signature r0 r1 shape-fn t0 t1 z
                                   (length dss^^^)
                                   (add1 (length dss^^^)) i)
                     (cons out1 (cons out0 dss^^^))
                     build-refs^^^))))]
      [(tcomp-ext1-∇ tp zp f signature m shape-fn)
       (let-values (((tp^ dss^ build-refs^) (eds-expr tp dss build-refs)))
         (let-values (((zp^ dss^^ build-refs^^) (eds-expr zp dss^ build-refs^)))
           (values (tcomp-ext1-∇ tp^ zp^ f signature m shape-fn)
                   dss^^
                   build-refs^^)))]
      [(tcomp-ext2-ρ-scalar f signature tp-t tp-u)
       (let-values (((t dss^ build-refs^) (eds-expr tp-t dss build-refs)))
         (let-values (((u dss^^ build-refs^^) (eds-expr tp-u dss^ build-refs^)))
           (values (tcomp-ext2-ρ-scalar f signature t u) dss^^ build-refs^^)))]
      [(tcomp-ext2-ρ tp-t tp-u f signature m n shape-fn)
       (let-values (((t dss^ build-refs^) (eds-expr tp-t dss build-refs)))
         (let-values (((u dss^^ build-refs^^) (eds-expr tp-u dss^ build-refs^)))
           (values (tcomp-ext2-ρ t u f signature m n shape-fn) dss^^ build-refs^^)))]
      [(tcomp-ext1-ρ-scalar f signature tp)
       (let-values (((tp^ dss^ build-refs^) (eds-expr tp dss build-refs)))
           (values (tcomp-ext1-ρ-scalar f signature tp^) dss^ build-refs^))]
      [(tcomp-ext1-ρ f signature m shape-fn tp)
       (let-values (((tp^ dss^ build-refs^) (eds-expr tp dss build-refs)))
           (values (tcomp-ext1-ρ f signature m shape-fn tp^) dss^ build-refs^))]
      [(tcomp-reshape s tp)
       (let-values (((tp^ dss^ build-refs^) (eds-expr tp dss build-refs)))
           (values (tcomp-reshape s tp^) dss^ build-refs^))])))

(define hash-signatures?
  (make-parameter #f))
;;TODO: Optimize sign by replacing it with this commented function
#;
(define sign
  (let ((xxh32-ctx (make-xxh32)))
    (λ ss
      (cond
        ((hash-signatures?)
         (xxh32-reset! xxh32-ctx 0)
         (xxh32-update! xxh32-ctx (apply bytes-append ss))
         (xxh32-digest xxh32-ctx))
        (else (format "~a" ss))))))
(define sign
  (let ((xxh32-ctx (make-xxh32)))
    (λ (s)
      (cond
        ((hash-signatures?)
         (xxh32-reset! xxh32-ctx 0)
         (xxh32-update! xxh32-ctx (string->bytes/utf-8 s))
         (format "~a" (xxh32-digest xxh32-ctx)))
        (else (format "~a" s))))))

(define generate-signature
  (λ (t)
    (gs-expr t)))

(define gs-expr
  (λ (t)
    (match t
      (s #:when (number? s)
        (sign (format "s~a" s)))
      ((tpromise tc _) (gs-expr tc))
      ((tcomp) (gs-tcomp t)))))

(define gs-tcomp
  (λ (tc)
    (match tc
      [(tcomp-list->tensor lst)
       (let ((list-sig
              (for/fold ((sig ""))
                        ((l lst)
                         (i (in-naturals 0)))
                (string-append sig (gs-expr l)))))
         (sign (format "l>t~a" list-sig)))]

      [(tcomp-tref tp i)
       (sign
        (format "tr~a~a" (gs-expr tp ) (gs-expr i )))]
      [(tcomp-trefs tp b)
       (sign
        (format "trs~a~a" (gs-expr tp ) (gs-expr b )))]
      [(tcomp-ext2-∇ fᵈ signature r0 r1 shape-fn tp-t0 tp-t1 tp-z out0 out1 i)
       (sign
        (format "e2∇~a~a_~a~a~a~a~a~a~a~a"
                signature r0 r1 shape-fn
                (gs-expr tp-t0 )
                (gs-expr tp-t1 )
                (gs-expr tp-z )
                out0 out1
                i))]
      [(tcomp-ext1-∇ tp zp f signature m shape-fn)
       (sign
        (format "e1∇~a~a~a~a~a"
                signature m shape-fn
                (gs-expr tp )
                (gs-expr zp )))]
      [(tcomp-ext2-ρ-scalar f signature tp-t tp-u)
       (sign
        (format "e2ρs~a~a~a"
                signature
                (gs-expr tp-t )
                (gs-expr tp-u )))]
      [(tcomp-ext2-ρ tp-t tp-u f signature m n shape-fn)
       (sign
        (format "e2ρ~a~a_~a~a~a~a"
                signature m n shape-fn
                (gs-expr tp-t )
                (gs-expr tp-u )))]
      [(tcomp-ext1-ρ-scalar f signature tp)
       (sign
        (format "e1ρs~a~a" signature (gs-expr tp )))]
      [(tcomp-ext1-ρ f signature m shape-fn tp)
       (sign
        (format "e1ρ~a~a~a~a"
                signature m shape-fn
                (gs-expr tp )))]
      [(tcomp-reshape s tp)
       (sign
        (format "r~a~a"
                s (gs-expr tp ) ))]
      [(tcomp-ds-ref index)
       (sign
        (format "dsr~a" index ))])))

;; Count references so that the tcomp AST nodes that refer to the same memory
;; location i.e. common AST nodes get extracted by let-binding them in the
;; compiled output racket code.
(define count-references
  (λ (t)
    (let-values (((counter uid) (count-references-expr t (hasheq) 0)))
      counter)))

(define count-references-expr
  (λ (t counter uid)
    (match t
      ((tpromise tc _)
       (count-references-expr tc counter uid))
      ((tcomp) (count-references-tcomp t counter uid))
      (_ (values counter uid)))))

(define count-references-tcomp
  (λ (tc counter uid)
    (match-let (((counter-data tc-binding-name tc-ref-count)
                 (hash-ref counter tc
                           (λ ()
                             (let-values (((st _) (struct-info tc)))
                               (let-values (((tcomp-name _0 _1 _2 _3 _4 _5 _6)
                                             (struct-type-info st)))
                                 (counter-data (string->symbol
                                                (format "~a_~a" tcomp-name uid))
                                               0)))))))
      (let* ((new-count (add1 tc-ref-count))
             (counter^ (hash-set counter tc
                                 (counter-data tc-binding-name
                                               new-count)))
             (uid^ (add1 uid)))
        (cond
          ((> new-count 1)
           ;; No need to increase reference count of children if parent occurs
           ;; more than once. This helps avoid creating extra tcom-var later in
           ;; ecs if child tcomp occurs only once within the parent tcomp, but
           ;; the parent tcomp itself occurs more than once.
           (values counter^ uid^))
          (else
           (match tc
             [(tcomp-list->tensor lst)
              (for/fold
               ((counter^^ counter^)
                (uid^^ uid^))
               ((l lst))
                (count-references-expr l counter^^ uid^^))]
             [(tcomp-tref tp i)
              (count-references-expr tp counter^ uid^)]
             [(tcomp-trefs tp b)
              (count-references-expr tp counter^ uid^)]
             [(tcomp-ext2-∇ fᵈ sign r0 r1 shape-fn tp-t0 tp-t1 tp-z out0 out1 i)
              (let*-values (((counter-1 uid-1) (count-references-expr tp-t0 counter^ uid^))
                            ((counter-2 uid-2) (count-references-expr tp-z counter-1 uid-1)))
                (count-references-expr tp-t1 counter-2 uid-2))]
             [(tcomp-ext1-∇ tp zp f sign m shape-fn)
              (let-values (((counter-1 uid-1) (count-references-expr tp counter^ uid^)))
                (count-references-expr zp counter-1 uid-1))]
             [(tcomp-ext2-ρ-scalar f sign tp-t tp-u)
              (let-values (((counter-1 uid-1) (count-references-expr tp-t counter^ uid^)))
                (count-references-expr tp-u counter-1 uid-1))]
             [(tcomp-ext2-ρ tp-t tp-u f sign m n shape-fn)
              (let-values (((counter-1 uid-1) (count-references-expr tp-t counter^ uid^)))
                (count-references-expr tp-u counter-1 uid-1))]
             [(tcomp-ext1-ρ-scalar f sign tp)
              (count-references-expr tp counter^ uid^)]
             [(tcomp-ext1-ρ f sign m shape-fn tp)
              (count-references-expr tp counter^ uid^)]
             [(tcomp-reshape s tp)
              (count-references-expr tp counter^ uid^)]
             [(tcomp-ds-ref index) (values counter^ uid^)]
             ;;need these cases for testing compiler invariant
             [(tcomp-let lhs rhs body)
              (let-values (((counter-1 uid-1) (count-references-expr rhs counter^ uid^)))
                (count-references-expr body counter-1 uid-1))]
             [(tcomp-var name) (values counter^ uid^)])))))))

(define extract-common-subexpressions
  (λ (t counter)
    (let-values (((instrs bindings)
                  (run-compiler-ecs (ecs-expr t counter) '())))
      (for/fold ((body instrs))
                ((binding bindings))
        (tcomp-let (car binding) (cdr binding) body)))))

(define ecs-expr
  (λ (tc counter)
    (match tc
      [(tpromise tc s)
       (->ecs
        (ecs-expr tc counter)
        (λ (instrs)
          (inj-ecs-val (tpromise instrs s))))]
      [tc #:when (number? tc)
          (inj-ecs-val tc)]
      [(tcomp) (ecs-tcomp tc counter)])))

(define ecs-tcomp
  (λ (tc counter)
    (let ((tc-counter-data
           (hash-ref counter tc
                     (λ ()
                       (counter-data (gensym 'illegal) 0)))))
      (match tc
        [(tcomp-list->tensor lst)
         (let ((instrs-list-compiler
                (for/foldr
                  ((list-compiler (inj-ecs-val '())))
                  ((arg lst))
                  (->ecs
                   (ecs-expr arg counter)
                   (λ (instrs)
                     (->ecs
                      list-compiler
                      (λ (instrs-list)
                        (inj-ecs-val (cons instrs instrs-list)))))))))
           (->ecs
            instrs-list-compiler
            (λ (instrs-list)
              (inj-ecs-tcomp (tcomp-list->tensor instrs-list) tc-counter-data))))]
        [(tcomp-tref tp i)
         (->ecs
          (ecs-expr tp counter)
          (λ (instrs)
            (inj-ecs-tcomp (tcomp-tref instrs i) tc-counter-data)))]
        [(tcomp-trefs tp b)
         (->ecs
          (ecs-expr tp counter)
          (λ (instrs)
            (inj-ecs-tcomp (tcomp-trefs instrs b) tc-counter-data)))]
        [(tcomp-ext2-∇ fᵈ sign r0 r1 shape-fn tp-t0 tp-t1 tp-z out0 out1 i)
         (->ecs
          (ecs-expr tp-t0 counter)
          (λ (t0-instrs)
            (->ecs
             (ecs-expr tp-t1 counter)
             (λ (t1-instrs)
               (->ecs
                (ecs-expr tp-z counter)
                (λ (z-instrs)
                  (inj-ecs-tcomp
                   (tcomp-ext2-∇ fᵈ sign r0 r1 shape-fn
                                 t0-instrs t1-instrs z-instrs
                                 out0 out1 i)
                   tc-counter-data)))))))]
        [(tcomp-ext1-∇ tp zp f sign m shape-fn)
         (->ecs
          (ecs-expr tp counter)
          (λ (t-instrs)
            (->ecs
             (ecs-expr zp counter)
             (λ (z-instrs)
               (inj-ecs-tcomp
                (tcomp-ext1-∇ t-instrs z-instrs f sign m shape-fn)
                tc-counter-data)))))]
        [(tcomp-ext2-ρ-scalar f sign tp-t tp-u)
         (->ecs
          (ecs-expr tp-t counter)
          (λ (t-instrs)
            (->ecs
             (ecs-expr tp-u counter)
             (λ (u-instrs)
               (inj-ecs-tcomp
                (tcomp-ext2-ρ-scalar f sign t-instrs u-instrs)
                tc-counter-data)))))]
        [(tcomp-ext2-ρ tp-t tp-u f sign m n shape-fn)
         (->ecs
          (ecs-expr tp-t counter)
          (λ (t-instrs)
            (->ecs
             (ecs-expr tp-u counter)
             (λ (u-instrs)
               (inj-ecs-tcomp
                (tcomp-ext2-ρ t-instrs u-instrs f sign m n shape-fn)
                tc-counter-data)))))]
        [(tcomp-ext1-ρ-scalar f sign tp)
         (->ecs
          (ecs-expr tp counter)
          (λ (instrs)
            (inj-ecs-tcomp (tcomp-ext1-ρ-scalar f sign instrs) tc-counter-data)))]
        [(tcomp-ext1-ρ f sign m shape-fn tp)
         (->ecs
          (ecs-expr tp counter)
          (λ (instrs)
            (inj-ecs-tcomp (tcomp-ext1-ρ f sign m shape-fn instrs) tc-counter-data)))]
        [(tcomp-reshape s tp)
         (->ecs
          (ecs-expr tp counter)
          (λ (instrs)
            (inj-ecs-tcomp (tcomp-reshape s instrs) tc-counter-data)))]
        [(tcomp-ds-ref index) (inj-ecs-tcomp tc tc-counter-data)]))))

(struct CompilerECS (run-compiler) #:transparent)

(define run-compiler-ecs
  (λ (c bindings)
    ((CompilerECS-run-compiler c) bindings)))

(define inj-ecs-val
  (λ (v)
    (CompilerECS (λ (bindings) (values v bindings)))))

(define inj-ecs-tcomp
  (λ (instrs cd)
    (match-let (((counter-data binding-var ref-count) cd))
      (CompilerECS (λ (bindings)
                  (cond
                    ((<= ref-count 1)
                     (values instrs bindings))
                    ((assv binding-var bindings)
                     (values (tcomp-var binding-var) bindings))
                    (else
                     (values (tcomp-var binding-var)
                             (extend-bindings binding-var instrs bindings)))))))))

(define ->ecs
  (λ (c f)
    (CompilerECS
     (λ (bindings)
       (let-values (((instrs bindings^) (run-compiler-ecs c bindings)))
         (run-compiler-ecs (f instrs) bindings^))))))

(define extend-env
  (λ (k v env)
    `((,k . ,v) . ,env)))

(define extend-bindings extend-env)

(define exists-in-env?
  (λ (ft env)
    (match env
      ('() #f)
      (`((,k . ,v) . ,_) #:when (eq? ft v) k)
      (`(,_ . ,rest-env) (exists-in-env? ft rest-env)))))

(define generate-racket
  (λ (t build-refs)
    (gr-expr t build-refs)))

(define gr-expr
  (λ (t build-refs)
    (match t
      [(tpromise tc _) (gr-expr tc build-refs)]
      [v #:when (number? v) v]
      [(tcomp) (gr-tcomp t build-refs)])))

(define gr-tcomp
  (λ (tc build-refs)
    (match tc
      [(tcomp-list->tensor lst)
       (let ((instrs-list (map (λ (t) (gr-expr t build-refs)) lst)))
         `(flat:list->tensor (list ,@instrs-list)))]
      [(tcomp-tref tp i)
       (let ((instrs (gr-expr tp build-refs))
             (i-instrs (gr-expr i build-refs)))
         `(flat:tref ,instrs ,i-instrs))]
      [(tcomp-trefs tp b)
       (let ((instrs (gr-expr tp build-refs))
             (b-instrs (gr-expr b build-refs)))
         `(rt:trefs ,instrs ,b-instrs))]
      [(tcomp-ext2-∇ fᵈ sign r0 r1 shape-fn tp-t0 tp-t1 tp-z out0 out1 i)
       (let ((t0-instrs (gr-expr tp-t0 build-refs))
             (t1-instrs (gr-expr tp-t1 build-refs))
             (z-instrs (gr-expr tp-z build-refs)))
         (let ((b (if (zero? i) out0 out1)))
           `(let* ([b ,b]
                   [v (data-segment-ref b)])
              (cond
                ((eqv? v 'uncalculated)
                 (ext2-∇-forcer ,fᵈ ,r0 ,r1 ,shape-fn
                                ,t0-instrs ,t1-instrs
                                ,z-instrs ,out0 ,out1)
                 (data-segment-ref b))
                (else v)))))]
      [(tcomp-ext1-∇ tp zp f sign m shape-fn)
       (let ((t-instrs (gr-expr tp build-refs))
             (z-instrs (gr-expr zp build-refs)))
         `(scalarize
           (flat-ext1-∇ ,f ,m ,shape-fn
                        (ensure-flat ,t-instrs)
                        (ensure-flat ,z-instrs))))]
      [(tcomp-ext2-ρ-scalar f sign tp-t tp-u)
       (let ((t-instrs (gr-expr tp-t build-refs))
             (u-instrs (gr-expr tp-u build-refs)))
         `(,f ,t-instrs ,u-instrs))]
      [(tcomp-ext2-ρ tp-t tp-u f sign m n shape-fn)
       (let ((t-instrs (gr-expr tp-t build-refs))
             (u-instrs (gr-expr tp-u build-refs)))
         `(scalarize
           (flat-ext2-ρ ,f ,m ,n ,shape-fn
                        (ensure-flat ,t-instrs)
                        (ensure-flat ,u-instrs))))]
      [(tcomp-ext1-ρ-scalar f sign tp)
       (let ((instrs (gr-expr tp build-refs)))
         `(,f ,instrs))]
      [(tcomp-ext1-ρ f sign m shape-fn tp)
       (let ((instrs (gr-expr tp build-refs)))
         `(scalarize
           (flat-ext1-ρ ,f ,m ,shape-fn
                        (ensure-flat ,instrs))))]
      [(tcomp-reshape s tp)
       (let ((instrs (gr-expr tp build-refs)))
         `(flat ',s
                (flat-store ,instrs)
                (flat-offset ,instrs)))]
      [(tcomp-let lhs rhs body)
       (let ((rhs-instrs (gr-expr rhs build-refs))
             (body-instrs (gr-expr body build-refs)))
         `(let ((,lhs ,rhs-instrs))
            ,body-instrs))]
      [(tcomp-var name) name]
      [(tcomp-ds-ref index) `(data-segment-ref ,index)])))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Composing Compiler Passes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define print-compiler? (make-parameter #f))
(define display-compiler-trace
  (λ (title value)
    (when (or (equal? (print-compiler?) #t)
              (and (list? (print-compiler?)) (member title (print-compiler?))))
      (printf "~a:~n" title)
      (displayln "--------------")
      (pretty-print value)
      (displayln ""))))
(define cache
  (make-parameter (make-hash)))
(define compile-tensor
  (λ (t)
    (display-compiler-trace 'Source-Tensor t)
    (let-values (((eds-instrs ds build-refs) (extract-data-segment t)))
      (display-compiler-trace 'Extract-Data-Segment-data ds)
      (display-compiler-trace 'Extract-Data-Segment-instructions eds-instrs)
      (let ((signature (generate-signature eds-instrs)))
        (display-compiler-trace 'Generate-Signature signature)
        (cond
          ((hash-has-key? (cache) signature)
           (let ((compiled (hash-ref (cache) signature)))
             (display-compiler-trace 'Cache-Hit signature)
             (values compiled ds)))
          (else
           (let ((counter (count-references eds-instrs)))
             (display-compiler-trace 'Count-References counter)
             (let ((extracted (extract-common-subexpressions eds-instrs counter)))
               (display-compiler-trace 'Extract-Common-Subexpressions extracted)
               (let ((rkt (generate-racket extracted build-refs)))
                 (display-compiler-trace 'Generate-Racket rkt)
                 (hash-set! (cache) signature rkt)
                 (values rkt ds))))))))))

;;TODO: update this for new compiler passes
(define compile-tensor/checks
  (λ (t)
    (let-values (((eds-instrs ds build-refs) (extract-data-segment t)))
      (flat:check-tensor-equal? (interp-tensor t) (interp-tensor eds-instrs))
      (let ((counter (count-references t)))
        (let ((extracted (extract-common-subexpressions t counter)))
          (flat:check-tensor-equal? (interp-tensor t) (interp-tensor extracted))
          (for/list ((cd (hash-values (count-references extracted))))
            (check-equal? (counter-data-ref-count cd) 1))
          (let-values (((rkt env) (generate-racket extracted build-refs)))
            (flat:check-tensor-equal? (interp-tensor extracted)
                                      (interp-racket rkt env))
            (values rkt env)))))))

(define get-compiled
  (λ (t)
    (let-values (((instrs env)
                  (compile-tensor t)))
      `(parameterize ((data-segment ,env))
         ,instrs))))

(include "test/test-c3-compiler.rkt")
(provide get-compiled compile-tensor compile-tensor/checks print-compiler?
         (rename-out (cache compiler-cache)))