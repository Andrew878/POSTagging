from nltk import FreqDist, WittenBellProbDist

emitted = [('N', 'apple'), ('N', 'apple'), ('N', 'banana')]
smoothed = {}
tags = set([t for (t,_) in emitted])
for tag in tags:
    words = [w for (t,w) in emitted if t == tag]
    smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
print('prob of N -> apple is',smoothed['N'].prob('apple'))
print('prob of N -> banana is',smoothed['N'].prob('banana'))
print('prob of N -> peach is',smoothed['N'].prob('peach'))