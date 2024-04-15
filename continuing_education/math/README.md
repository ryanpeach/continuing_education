# Logseq flash cards
* What are the equalities of logarithms? #card
    * They turn multiplication into addition: $\log(a \cdot b) = \log(a) + \log(b)$
    * They turn division into subtraction: $\log(a / b) = \log(a) - \log(b)$
    * They turn exponentiation into multiplication: $\log(a^b) = b \cdot \log(a)$
* Why would you use a negative log probability in a loss function? #card
    * It's infinity at 0 and 0 at 1, which means at high confidence in something you get a low loss approaching 0, and at low confidence you get a high loss approaching infinity. It gives a strong gradient signal to the network to update its parameters.
