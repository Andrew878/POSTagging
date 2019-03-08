from math import log, exp
from nltk import FreqDist, WittenBellProbDist


class BeamSearch:

    def __init__(self, beam_width, state_transition_prob, emission_prob, list_of_states_no_end_start):
        self.beam_width = beam_width
        self.state_transition_prob = state_transition_prob
        self.emission_prob = emission_prob
        self.list_of_states = list_of_states_no_end_start
        self.start = '<s>'
        self.end = '<s>'
        self.T_length = None
        self.N_length = len(list_of_states_no_end_start)
        self.viterbi = None
        self.back_stop = None

    def find_highest_path(self, words_with_start_stop):
        self.T_length = len(words_with_start_stop)
        self.viterbi = [[0] * self.N_length] * self.T_length
        self.back_stop = [[0] * self.N_length] * self.T_length

        print(words_with_start_stop)

        word = words_with_start_stop[1]
        for state_index in range(0, self.N_length):
            state_to = self.list_of_states[state_index]
            state_from = self.start

            print(self.state_transition_prob[state_from].prob(state_to))

            print((word, state_from))
            print((state_to, state_from))
            self.viterbi[1][state_index] = log(self.state_transition_prob.prob((state_from,state_to)))   +log(self.emission_prob.prob((word,state_to)))
            self.back_stop[1][state_index] = self.start

        print(self.viterbi)
        print(self.back_stop)

