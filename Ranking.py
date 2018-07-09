import numpy as np
import gensim
from gensim import corpora
import math

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

k = 3 		        #number of topics set arbitrarily to 3 

def clean(doc):
    list_of_docs = []
    for document in doc:
	list_of_words = []
    	preprocess = gensim.utils.simple_preprocess(document.lower()) 
    	for word in preprocess:
		if word not in stop_words:
			list_of_words.append(word)
	list_of_docs.append(list_of_words)
    return list_of_docs

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    KL = 0
    for i in range(len(b)):
	if b[i]!=0 and a[i]!=0:
		KL += a[i]*(math.log(a[i]/b[i], 10))
	else:
		continue
    return KL


def IR(values1,values2):
	
	IR = KL(values1, [(values2[i]+values1[i])/2 for i in range(len(values1))])+ KL(values1, [(values2[i]+values1[i])/2 for i in range(len(values1))])
	expIR = 10**(-IR)
	return expIR

def get_vector(text, dictionary, Lda):
	vector = [[] for i in range(k)]
	for i in range(k):
		vector[i] = [0 for p in range(len(dictionary.keys()))]
		for word in text:
			if word not in stop_words:
				if word in dictionary.values():
					vector[i][dictionary.values().index(word)] = Lda.get_topics()[i][dictionary.values().index(word)]
	return vector
	
		

def find_best_passage(passages,query):
	
	preprocessed = clean(passages)
	dictionary = corpora.Dictionary(preprocessed)
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in preprocessed]
	Lda = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=k, id2word = dictionary, passes=50)
	query = gensim.utils.simple_preprocess(query)
	query_vector = 	get_vector(query, dictionary, Lda)
	passage_vectors = [[] for i in range(len(passages))]
	passage_sum2 = [0 for i in range(len(passages))]
	passage_sum2 = [IR([x[1] for x in Lda.get_document_topics(dictionary.doc2bow(preprocessed[i]) , minimum_probability=None, minimum_phi_value=None, per_word_topics=False)],[x[1] for x in Lda.get_document_topics(dictionary.doc2bow(query) , minimum_probability=None, minimum_phi_value=None, per_word_topics=False)]) for i in range(len(passages))]
	passage_sum1 = [0 for i in range(len(passages))]
	for i in range(len(passages)):
		passage_vectors[i] = get_vector(preprocessed[i], dictionary, Lda)
	for i in range(len(passages)):
		for j in range(k):
			passage_sum1[i] += IR(passage_vectors[i][j], query_vector[j])/k
	passage_sum = [0 for i in range(len(passages))]
	for i in range(len(passages)):
		passage_sum[i] = passage_sum1[i] * passage_sum2[i]
	print '*************'
	print passage_sum 

def main():
	query = "Doctor are sugar"
	doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
	doc2 = "My father spends a lot of time driving my sister around to dance practice."
	doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
	doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
	doc5 = "Health experts say that Sugar is not good for your lifestyle."
	
	# compile passages
	passages = [doc1, doc2, doc3, doc4, doc5]
	find_best_passage(passages, query)

main()
	
	
	
	
	






