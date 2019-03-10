from math import log, exp
from nltk import FreqDist, WittenBellProbDist

"""A Beam search object that contains methods to tag a single, unlabeled sentence. It uses log probabilities to avoid underflow"""

class BeamSearch:

    """The object requires beam length and smoothed probability distributions to be established"""

    def __init__(self, freq_dist_tag_single, smoothed_tag_tag, smoothed_word_tag, beam_width):
        self.freq_dist_tag_single = freq_dist_tag_single
        self.smoothed_tag_tag = smoothed_tag_tag
        self.smoothed_word_tag = smoothed_word_tag
        self.beam_width = beam_width
        self.start = '<s>'
        self.end = '</s>'

        # create a list of states that we can later cycle through to create our vit. matrix
        self.non_beg_or_end_states = list(freq_dist_tag_single.keys())
        self.non_beg_or_end_states.remove(self.start)
        self.tag_to_index_dict = dict.fromkeys(self.non_beg_or_end_states)
        self.non_beg_or_end_states.remove(self.end)


    def viterbi_path(self, word_list):

        """This is the main method that drives the entire class, and ultimately returns a estimated list of labels."""

        # initialise arrays
        viterbi = [['' for x in range(len(self.non_beg_or_end_states))] for y in range(len(word_list))]
        back_pointer = [['' for x in range(len(self.non_beg_or_end_states))] for y in range(len(word_list))]

        self.intialise_first_values_of_array(back_pointer, viterbi, word_list)

        # cycle through each word
        word_num = 2
        for word in word_list[2:-1]:

            # cycle through each word for each state
            state_num = 0
            for state in self.non_beg_or_end_states:

                # update the Viterbi and Backpointer matrices using prior path probabilities
                viterbi[word_num][state_num], back_pointer[word_num][state_num] = self.return_max_prob_and_best_path(
                    viterbi,
                    state,
                    word,
                    word_num - 1)
                state_num += 1

            word_num += 1

        # perform final token step
        max_last_prob, best_last_tag_est = self.final_step_return_max_prob_and_best_path(viterbi,
                                                                                         word_num - 1)
        # using backpointer matrix, construct tag sequence estimate
        final_best_tag = self.construct_final_tag(back_pointer, best_last_tag_est, word_list)

        return final_best_tag


    def intialise_first_values_of_array(self, back_pointer, viterbi, word_list):

        """Initialisation step of matrices."""

        # cycle through each state for the first word
        state_num = 0
        for state in self.non_beg_or_end_states:
            self.tag_to_index_dict[state] = state_num

            # logs used to avoid underflow
            a = log(self.smoothed_tag_tag[self.start].prob(state))
            b = log(self.smoothed_word_tag[state].prob(word_list[1]))

            viterbi[1][state_num] = a + b
            back_pointer[1][state_num] = self.start
            state_num += 1

    def return_max_prob_and_best_path(self, exisiting_viterbi, current_state, current_word,
                                      previous_word_index, ):
        # initialise maximum values
        maximum_prob = -10 ** 10
        maximum_origin = 0

        # find the 'best' order of states to investigate according to our beam length.
        beam_index_order = self.find_top_k_tags(previous_word_index, exisiting_viterbi)

        # cycle through previous states, and find the max probability
        for previous_state_index in beam_index_order:


            state_transition_prob_log = log(self.smoothed_tag_tag[self.non_beg_or_end_states[previous_state_index]].prob(current_state))

            prob_est = exisiting_viterbi[previous_word_index][previous_state_index] + state_transition_prob_log

            # check if max probability, and update if appropriate
            if (maximum_prob < prob_est):
                maximum_prob = prob_est
                maximum_origin = self.non_beg_or_end_states[previous_state_index]


        # after checking all, return the max prob plus the emission probability. Also return appropriate back pointer
        return maximum_prob + log(self.smoothed_word_tag[current_state].prob(current_word)), maximum_origin;

    def find_top_k_tags(self, previous_word_index, exisiting_viterbi):

        """" Find the ordered, highest K tags to check.
        Borrowed the second line from User: bro-grammer on
        https://stackoverflow.com/questions/36459969/python-convert-list-to-dictionary-with-indexes Accessed on: 08/03/2019"""

        # create new state list from a viterbi matrix column
        list_of_v = [v for v in exisiting_viterbi[previous_word_index]]

        # create dict to help us remember original index order
        value_to_index_dict = dict(map(reversed, enumerate(list_of_v)))

        top_state_index_references = []

        # order vit column
        list_of_v.sort(reverse=True)

        # add the top K values
        for state in range(self.beam_width):
            top_state_index_references.append(value_to_index_dict[list_of_v[state]])

        # return list of highest vit indices
        return top_state_index_references


    def final_step_return_max_prob_and_best_path(self, exisiting_viterbi, previous_word_index):

        """" Termination step """

        # initialise maximum values
        maximum_prob = -10 ** 10
        maximum_origin = 0

        # again, find order of highest states from vit matrix
        beam_index_order = self.find_top_k_tags(previous_word_index, exisiting_viterbi)

        # cycle through states. Behaves very similar to 'return_max_prob_and_best_path' method
        for previous_state_index in beam_index_order:

            prob_est = exisiting_viterbi[previous_word_index][previous_state_index] + log(
                self.smoothed_tag_tag[self.non_beg_or_end_states[previous_state_index]].prob(self.end))

            if (maximum_prob < prob_est):
                maximum_prob = prob_est
                maximum_origin = self.non_beg_or_end_states[previous_state_index]

        return maximum_prob, maximum_origin;

    def construct_final_tag(self, back_pointer, best_last_tag_est, word_list):

        """" construct tags from backpointer matrix """

        best_tags_array = []
        best_tags_array.append(best_last_tag_est)
        index = self.tag_to_index_dict[best_last_tag_est]

        # ignoring the start/end symbols, add tags to list
        for i in range(len(back_pointer) - 2, 1, -1):
            current_tag = back_pointer[i][index]
            best_tags_array.append(current_tag)
            index = self.tag_to_index_dict[current_tag]

        # reverse order and return
        best_tags_array.reverse()
        return best_tags_array
