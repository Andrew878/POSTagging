def return_max_prob_and_best_path(exisiting_viterbi, current_state, current_word, non_beg_or_end_states,
                                  previous_word_index,
                                  freq_dist_tagWordPair_SMOOTH,
                                  freq_dist_tagBigram_SMOOTH):
    maximum_prob = -10 ** 10
    maximum_origin = 0
    current_tag_word_pair = (current_word, current_state)

    for previous_state_index in range(len(non_beg_or_end_states)):

        tag_pair_to_test = (non_beg_or_end_states[previous_state_index], current_state)

        print(current_tag_word_pair)
        print(tag_pair_to_test)

        prob_est = exisiting_viterbi[previous_word_index][previous_state_index] + log(
            freq_dist_tagBigram_SMOOTH.prob(tag_pair_to_test))

        print("prob est is", prob_est)
        print("max prob is", maximum_prob)

        if (maximum_prob < prob_est):
            maximum_prob = prob_est + log(freq_dist_tagWordPair_SMOOTH.prob(current_tag_word_pair))
            maximum_origin = non_beg_or_end_states[previous_state_index]

        print("max prob is", maximum_prob)


    print("returning max prob and max origin ", maximum_prob, maximum_origin)

    return maximum_prob, maximum_origin


def viterbi_path(sent, freq_dist_tag_single, freq_dist_tag_single_SMOOTH, freq_dist_tagWordPair_SMOOTH,
                 freq_dist_tagBigram_SMOOTH,
                 freq_dist_tagWordPairWithStartEnd_SMOOTH):
    start = '<s>'

    end = '</s>'
    prob_start = 1.0

    word_list = [w for (w, _) in sent]
    # create a list of states without start or end states
    non_beg_or_end_states = list(freq_dist_tag_single.keys())
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
    word_num = 2



    for word in word_list[2:]:

        state_num = 0
        for state in non_beg_or_end_states:

            viterbi[word_num][state_num]
            back_pointer[word_num][state_num] = return_max_prob_and_best_path(viterbi, state,
                                                                              word, non_beg_or_end_states,
                                                                              word_num-1,
                                                                              freq_dist_tagWordPair_SMOOTH,
                                                                              freq_dist_tagBigram_SMOOTH)


        return ()
from nltk import FreqDist, WittenBellProbDist


from math import log, exp
