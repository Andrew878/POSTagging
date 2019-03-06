from math import log, exp
from nltk import FreqDist, WittenBellProbDist


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
        print("previous vit at location = ", previous_word_index, previous_state_index,
              exisiting_viterbi[previous_word_index][previous_state_index])

        prob_est = exisiting_viterbi[previous_word_index][previous_state_index] + log(
            freq_dist_tagBigram_SMOOTH.prob(tag_pair_to_test))

        print("prob est is", prob_est)
        print("max prob is", maximum_prob)

        if (maximum_prob < prob_est):
            maximum_prob = prob_est
            maximum_origin = non_beg_or_end_states[previous_state_index]

        print("max prob is", maximum_prob)

    print(
        "returning max prob and max origin ",
        maximum_prob + log(freq_dist_tagWordPair_SMOOTH.prob(current_tag_word_pair)),
        maximum_origin)

    return maximum_prob + log(freq_dist_tagWordPair_SMOOTH.prob(current_tag_word_pair)), maximum_origin;


def final_step_return_max_prob_and_best_path(exisiting_viterbi, current_state, non_beg_or_end_states,
                                             previous_word_index,
                                             freq_dist_tagBigram_SMOOTH):
    maximum_prob = -10 ** 10
    maximum_origin = 0

    for previous_state_index in range(len(non_beg_or_end_states)):

        tag_pair_to_test = (non_beg_or_end_states[previous_state_index], current_state)

        print(tag_pair_to_test)
        print("previous vit at location = ", previous_word_index, previous_state_index,
              exisiting_viterbi[previous_word_index][previous_state_index])

        prob_est = exisiting_viterbi[previous_word_index][previous_state_index] + log(
            freq_dist_tagBigram_SMOOTH.prob(tag_pair_to_test))

        print("prob est is", prob_est)
        print("max prob is", maximum_prob)

        if (maximum_prob < prob_est):
            maximum_prob = prob_est
            maximum_origin = non_beg_or_end_states[previous_state_index]

        print("max prob is", maximum_prob)

    print("returning max prob and max origin ", maximum_prob, maximum_origin)

    return maximum_prob, maximum_origin;


def construct_final_tag(back_pointer, best_last_tag_est, non_beg_or_end_states, tag_to_index_dict, word_list):
    best_tags_array = []
    # best_tags_array.append(best_last_tag_est)
    index = tag_to_index_dict[best_last_tag_est]

    for i in range(len(back_pointer)-1, 1, -1):
        print("i", i)
        print("index", index)
        current_tag = back_pointer[i][index]
        best_tags_array.append(current_tag)
        index = tag_to_index_dict[current_tag]
        print(best_tags_array)
        print("word", word_list[i])

    best_tags_array.reverse()

    print("FINAL")
    print(word_list)
    print(best_tags_array)

    return best_tags_array


def viterbi_path(word_list, freq_dist_tag_single, freq_dist_tag_single_SMOOTH, freq_dist_tagWordPair_SMOOTH,
                 freq_dist_tagBigram_SMOOTH,
                 freq_dist_tagWordPairWithStartEnd_SMOOTH):
    start = '<s>'

    end = '</s>'
    prob_start = 1.0

    # word_list = [w for (w, _) in sent]
    # create a list of states without start or end states
    non_beg_or_end_states = list(freq_dist_tag_single.keys())
    non_beg_or_end_states.remove(start)
    tag_to_index_dict = dict.fromkeys(non_beg_or_end_states)
    non_beg_or_end_states.remove(end)

    viterbi = [['' for x in range(len(non_beg_or_end_states))] for y in range(len(word_list))]

    back_pointer = [['' for x in range(len(non_beg_or_end_states))] for y in range(len(word_list))]

    print(viterbi)
    state_num = 0

    for state in non_beg_or_end_states:

        tag_to_index_dict[state] = state_num
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
            viterbi[word_num][state_num], back_pointer[word_num][state_num] = return_max_prob_and_best_path(viterbi,
                                                                                                            state,
                                                                                                            word,
                                                                                                            non_beg_or_end_states,
                                                                                                            word_num - 1,
                                                                                                            freq_dist_tagWordPair_SMOOTH,
                                                                                                            freq_dist_tagBigram_SMOOTH)
            state_num += 1
            print("viterbi")
            print(viterbi)

            print("backpoint")
            print(back_pointer)
        word_num += 1

    max_last_prob, best_last_tag_est = final_step_return_max_prob_and_best_path(viterbi, end, non_beg_or_end_states,
                                                                                word_num - 1,
                                                                                freq_dist_tagBigram_SMOOTH)

    print("best_last_tag_est ", best_last_tag_est)

    final_best_tag = construct_final_tag(back_pointer, best_last_tag_est, non_beg_or_end_states, tag_to_index_dict, word_list)

    print(final_best_tag)

    return (final_best_tag)
