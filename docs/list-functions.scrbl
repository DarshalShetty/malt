#lang scribble/manual

@defmodule*[(malt malt/base malt/base-no-duals malt/base-no-overrides malt/base-no-duals-no-overrides malt/learner malt/flat-tensors malt/nested-tensors)]

@title{List functions}

The list data type is identical to the one used in racket, but
@racket[malt] provides functions to access members of the list
in the presence of automatic differentiation.

@defproc[(ref [lst (listof any?)] [i scalar?]) any?]{
  Returns the @racket[i]'th member of @racket[lst].
}

@defproc[(refr [lst (listof any?)] [n scalar?]) (listof any?)]{
  Returns the remaining members in @racket[lst] after removing
  the first @racket[n] members. This is equivalent to @racket[drop].
}

@defproc[(len [lst (listof any?)]) scalar?]{
  Returns the number of members in @racket[lst].
}
