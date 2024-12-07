(module+ test
  (require rackunit)
  (define v (init-shape (list 1000 4)))
  (define mean-v (abs (/ (sum (sum v)) 4000)))
  (define variance-v (- (/ (sum (sum (* v v))) 4000) (* mean-v mean-v)))
  (check-true (< mean-v 0.05))
  (check-true (and (>= variance-v 0.4)
                   (<= variance-v 0.6)))

  ;; Here variance will be 2/8 = 0.25
  (define r (init-shape (list 1000 4 2)))
  (define mean-r (abs (/ (sum (sum (sum r))) 8000)))
  (define variance-r (- (/ (sum (sum (sum (* r r)))) 8000) (* mean-r mean-r)))

  (check-true (< mean-r 0.05))
  (check-true (and (>= variance-r 0.22)
                   (<= variance-r 0.28))))
