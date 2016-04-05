from tpl.language.lexical_structure import WordEmbeddingDict

from theano import config
import numpy as np

class EmbeddingFeaturizer(object):

	TEST_WORD_EMBEDDING_FILE = '../lib/lexicon/google_word_vector/GoogleNews-vectors-negative300_test.txt'
	WORD_EMBEDDING_FILE = '../lib/lexicon/google_word_vector/GoogleNews-vectors-negative300.txt'
	OOV_VALUE = 0

	def __init__(self, word_embedding_file=WORD_EMBEDDING_FILE):
		self.word_embedding_dict = WordEmbeddingDict(word_embedding_file)

	def create_arg_matrix(self, arg_tokens):
		num_tokens = len(arg_tokens)
		sentence_matrix = np.zeros((num_tokens, self.word_embedding_dict.num_units)).astype(config.floatX)
		for i in xrange(num_tokens):
			if arg_tokens[i] in self.word_embedding_dict:
				sentence_matrix[i, :] = self.word_embedding_dict[arg_tokens[i]]
			else:
				sentence_matrix[i, :] = EmbeddingFeaturizer.OOV_VALUE
		return sentence_matrix
	
	def additive_args(self, relation_list):
		num_relations = len(relation_list)
		num_units = self.word_embedding_dict.num_units
		
		arg1_matrix = np.array([self.create_arg_matrix(x.arg_tokens(1)).sum(0) 
			for x in relation_list])
		arg2_matrix = np.array([self.create_arg_matrix(x.arg_tokens(2)).sum(0) 
			for x in relation_list])
		return [arg1_matrix, arg2_matrix]

	def mean_args(self, relation_list):	
		num_relations = len(relation_list)
		num_units = self.word_embedding_dict.num_units
		
		arg1_matrix = np.array([self.create_arg_matrix(x.arg_tokens(1)).mean(0) 
			for x in relation_list])
		arg2_matrix = np.array([self.create_arg_matrix(x.arg_tokens(2)).mean(0) 
			for x in relation_list])
		return [arg1_matrix, arg2_matrix]

	def max_args(self, relation_list):
		num_relations = len(relation_list)
		num_units = self.word_embedding_dict.num_units
		
		arg1_matrix = np.array([self.create_arg_matrix(x.arg_tokens(1)).max(0) 
			for x in relation_list])
		arg2_matrix = np.array([self.create_arg_matrix(x.arg_tokens(2)).max(0) 
			for x in relation_list])
		return [arg1_matrix, arg2_matrix]

	def top_args(self, relation_list):
		num_relations = len(relation_list)
		num_units = self.word_embedding_dict.num_units
		
		arg1_matrix = np.array([self.create_arg_matrix(x.arg_tokens(1))[-1,:]
			for x in relation_list])
		arg2_matrix = np.array([self.create_arg_matrix(x.arg_tokens(2))[-1,:]
			for x in relation_list])
		return [arg1_matrix, arg2_matrix]

def cdssm_feature(relation_list):
	num_relations = len(relation_list)
	num_units = len(relation_list[0].relation_dict['Arg1']['CDSSMTarget']) * 2
	
	arg1_matrix1 = np.array([x.relation_dict['Arg1']['CDSSMTarget'] for x in relation_list]).astype(config.floatX)
	arg1_matrix2 = np.array([x.relation_dict['Arg1']['CDSSMSource'] for x in relation_list]).astype(config.floatX)
	arg1_matrix = np.hstack((arg1_matrix1, arg1_matrix2))

	arg2_matrix1 = np.array([x.relation_dict['Arg2']['CDSSMTarget'] for x in relation_list]).astype(config.floatX)
	arg2_matrix2 = np.array([x.relation_dict['Arg2']['CDSSMSource'] for x in relation_list]).astype(config.floatX)
	arg2_matrix = np.hstack((arg2_matrix1, arg2_matrix2))
	return [arg1_matrix, arg2_matrix]

def cached_features(relation_list, feature_name):
	num_relations = len(relation_list)
	num_units = len(relation_list[0].relation_dict[feature_name]) 
	return [np.array([x.relation_dict[feature_name] for x in relation_list]).astype(config.floatX)]

def cached_argwise_features(relation_list, feature_name):
	num_relations = len(relation_list)
	num_units = len(relation_list[0].relation_dict['Arg1'][feature_name]) 
	arg1_matrix = np.array([x.relation_dict['Arg1'][feature_name] for x in relation_list])
	arg2_matrix = np.array([x.relation_dict['Arg2'][feature_name] for x in relation_list])
	return [arg1_matrix, arg2_matrix]

