from nltk import FreqDist, WittenBellProbDist
from math import log, exp


def viterbi_path(sent, freq_dist_tag_single, freq_dist_tag_single_SMOOTH, freq_dist_tagWordPair_SMOOTH,
                 freq_dist_tagBigram_SMOOTH,
                 freq_dist_tagWordPairWithStartEnd_SMOOTH):

    if(len(sent))


    start = '<s>'
    end = '</s>'

    prob_start = 1.0
    word_list = [w for (w, _) in sent]

    # create a set of states without start or end states
    non_beg_or_end_states = set(freq_dist_tag_single.keys())
    non_beg_or_end_states.remove(start)
    non_beg_or_end_states.remove(end)

    viterbi = [['' for x in range(len(non_beg_or_end_states))] for y in range(len(word_list))]
    back_pointer = [['' for x in range(len(non_beg_or_end_states) + 1)] for y in range(len(word_list))]

    print(viterbi)

    state_num = 0
    for state in non_beg_or_end_states:
        to_from_tag_pair = (start, state)
        print("to_from_tag_pair =", to_from_tag_pair)

        word_given_tag_pair = (word_list[1], state)
        print("word_given_tag_pair = ", word_given_tag_pair)

        a = log(freq_dist_tagBigram_SMOOTH.prob(to_from_tag_pair)) + log(prob_start)
        b = log(freq_dist_tagWordPair_SMOOTH.prob(word_given_tag_pair))

        if (state == start):
            viterbi[0][state_num] = log(1)

        viterbi[1][state_num] = a + b
        back_pointer[1][state_num] = start
        state_num += 1

    print(viterbi[:][:])
    print(back_pointer[:][:])

    word_num = 1
    for word in word_list[1:]:

        state_num = 0
        for state in non_beg_or_end_states:
            viterbi[word_num][state_num] = viterbi[word_num-1][state_num]

    return ()


def best_path_given_back_pointers(back_pointer):
    while (len(back_pointer) > 0):
        be
