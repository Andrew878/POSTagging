from math import log, exp
from nltk import FreqDist, WittenBellProbDist


class BeamSearch:

    def __init__(self, beam_width):
        self.beam_width = beam_width


    def return_max_prob_and_best_path(self, exisiting_viterbi, current_state, current_word, non_beg_or_end_states,
                                      previous_word_index,
                                      smoothed_tag_tag, smoothed_word_tag):
        maximum_prob = -10 ** 10
        maximum_origin = 0
        current_tag_word_pair = (current_word, current_state)

        # print("BEGINNING BEAM SEARCH -----------")
        beam_index_order = self.find_top_k_tags(previous_word_index, exisiting_viterbi)

        for previous_state_index in beam_index_order:

            tag_pair_to_test = (non_beg_or_end_states[previous_state_index], current_state)

            print("current_tag_word_pair", current_tag_word_pair)
            print("tag_pair_to_test", tag_pair_to_test)
            print("previous vit at location = ", previous_word_index, previous_state_index,
                  exisiting_viterbi[previous_word_index][previous_state_index])
            print("tag transition prob = ",
                  log(smoothed_tag_tag[non_beg_or_end_states[previous_state_index]].prob(current_state)))

            prob_est = exisiting_viterbi[previous_word_index][previous_state_index] + log(
                smoothed_tag_tag[non_beg_or_end_states[previous_state_index]].prob(current_state))

            print("prob est is", prob_est)
            print("max prob is", maximum_prob)

            if (maximum_prob < prob_est):
                maximum_prob = prob_est
                maximum_origin = non_beg_or_end_states[previous_state_index]

            print("max prob is", maximum_prob)

        print(
            "returning max prob and max origin ",
            maximum_prob + log(smoothed_word_tag[current_state].prob(current_word)),
            maximum_origin)

        return maximum_prob + log(smoothed_word_tag[current_state].prob(current_word)), maximum_origin;

    def find_top_k_tags(self, previous_word_index, exisiting_viterbi):

        """" User: bro-grammer
        https://stackoverflow.com/questions/36459969/python-convert-list-to-dictionary-with-indexes"""

        list_of_v = [v for v in exisiting_viterbi[previous_word_index]]
        value_to_index_dict = dict(map(reversed, enumerate(list_of_v)))

        top_state_index_references = []
        # print(list_of_v)
        list_of_v.sort(reverse=True)
        print(list_of_v)
        print(value_to_index_dict)
        # print("range(self.beam_width)",range(self.beam_width))
        # print("(self.beam_width)",(self.beam_width))

        for state in range(self.beam_width):
            # print("state:",state)
            # print("list_of_v[state]",list_of_v[state])
            # print("value_to_index_dict[list_of_v[state]]",value_to_index_dict[list_of_v[state]])
            top_state_index_references.append(value_to_index_dict[list_of_v[state]])

        print(top_state_index_references)

        return top_state_index_references

    def final_step_return_max_prob_and_best_path(self, exisiting_viterbi, current_state, non_beg_or_end_states,
                                                 previous_word_index,
                                                 smoothed_tag_tag):
        maximum_prob = -10 ** 10
        maximum_origin = 0

        beam_index_order = self.find_top_k_tags(previous_word_index, exisiting_viterbi)

        for previous_state_index in beam_index_order:

            tag_pair_to_test = (non_beg_or_end_states[previous_state_index], current_state)

            # print(tag_pair_to_test)
            # print("previous vit at location = ", previous_word_index, previous_state_index,
            #       exisiting_viterbi[previous_word_index][previous_state_index])

            prob_est = exisiting_viterbi[previous_word_index][previous_state_index] + log(
                smoothed_tag_tag[non_beg_or_end_states[previous_state_index]].prob(current_state))

            print("tag_pair_to_test, final stage", tag_pair_to_test)
            print("previous vit at location, final stage = ", previous_word_index, previous_state_index,
                  exisiting_viterbi[previous_word_index][previous_state_index])
            print("tag transition prob, final stage = ", log(
                smoothed_tag_tag[non_beg_or_end_states[previous_state_index]].prob(current_state)))

            if (maximum_prob < prob_est):
                maximum_prob = prob_est
                maximum_origin = non_beg_or_end_states[previous_state_index]

            # print("max prob is", maximum_prob)

        print("returning max prob and max origin, final stage ", maximum_prob, maximum_origin)

        return maximum_prob, maximum_origin;

    def construct_final_tag(self, back_pointer, best_last_tag_est, non_beg_or_end_states, tag_to_index_dict, word_list):
        best_tags_array = []
        best_tags_array.append(best_last_tag_est)
        index = tag_to_index_dict[best_last_tag_est]

        for i in range(len(back_pointer) - 2, 1, -1):
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

    def viterbi_path(self, word_list, freq_dist_tag_single, smoothed_tag_tag, smoothed_word_tag):
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
        print(word_list)

        for state in non_beg_or_end_states:

            tag_to_index_dict[state] = state_num
            to_from_tag_pair = (start, state)
            # print("to_from_tag_pair =", to_from_tag_pair)

            word_given_tag_pair = (word_list[1], state)
            # print("word_given_tag_pair = ", word_given_tag_pair)

            a = log(smoothed_tag_tag[start].prob(state))
            b = log(smoothed_word_tag[state].prob(word_list[1]))

            if (state == start):
                viterbi[0][state_num] = log(1)

            viterbi[1][state_num] = a + b
            back_pointer[1][state_num] = start
            state_num += 1

        print(viterbi[:][:])
        print(back_pointer[:][:])

        word_num = 2
        for word in word_list[2:-1]:

            state_num = 0
            for state in non_beg_or_end_states:
                print("word investigated", word)
                print(word_list[2:-1])
                viterbi[word_num][state_num], back_pointer[word_num][state_num] = self.return_max_prob_and_best_path(
                    viterbi,
                    state,
                    word,
                    non_beg_or_end_states,
                    word_num - 1, smoothed_tag_tag, smoothed_word_tag)
                state_num += 1
                # print("viterbi")
                # print(viterbi)
                #
                # print("backpoint")
                # print(back_pointer)
            word_num += 1

        max_last_prob, best_last_tag_est = self.final_step_return_max_prob_and_best_path(viterbi, end,
                                                                                         non_beg_or_end_states,
                                                                                         word_num - 1,
                                                                                         smoothed_tag_tag)

        # print("best_last_tag_est ", best_last_tag_est)

        final_best_tag = self.construct_final_tag(back_pointer, best_last_tag_est, non_beg_or_end_states,
                                                  tag_to_index_dict, word_list)

        print(final_best_tag)

        return (final_best_tag)
