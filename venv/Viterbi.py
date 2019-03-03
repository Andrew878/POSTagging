from nltk import FreqDist, WittenBellProbDist


def viterbi_path(sent, freq_dist_tag_single_SMOOTH, freq_dist_tagWordPair_SMOOTH, freq_dist_tagBigram_SMOOTH,
                 freq_dist_tag_single):
    prob_start = 1.0
    word_list = [w for (w, _) in sent]
    start = '<s>'

    # create path prob matrix. the  '-2' is to remove the start and end symbols

    viterbi = [[]] * (len(word_list))

    set(sent)

    i = 0
    for state in freq_dist_tag_single.keys():
        to_from_tag_pair = (start, state)
        print("to_from_tag_pair =", to_from_tag_pair)

        word_given_tag_pair = (word_list[1], state)
        print("word_given_tag_pair = ", word_given_tag_pair)

        a = freq_dist_tagBigram_SMOOTH.prob(to_from_pair) * prob_start
        b = freq_dist_tagWordPair_SMOOTH(word_given_tag_pair)

        viterbi[i][1] = a * b

        print(viterbi[i][1])
