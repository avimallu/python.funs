class fastfuzzy:
    
    def __init__(self, n_gram, n_jobs, source_strings, compare_strings):
        # "True" Class variables
        self.n_gram  = n_gram
        self.n_jobs  = n_jobs
        self.source  = source_strings
        self.compare = compare_strings
        
        # Input word variables
        self.source_tfidf_matrix  = self.vectorize_strings(self.source)
        self.compare_tfidf_matrix = self.vectorize_strings(self.compare)
        self.create_index(self.source_tfidf_matrix)
    
    def create_ngrams(self, strings):
        """Takes an input string, cleans it and converts to ngrams."""
        string = str(strings)
        string = string.lower()
        # Convert to simple ASCII code (best for most use-cases)
        string = string.encode("ascii", errors="ignore").decode()
        # Remove punctuation from text.
        chars_to_remove = [")","(",".","|","[","]","{","}","'","-"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)
        # Replace multiple spaces with a single space
        string = re.sub(' +',' ',string).strip()
        # Add padding to string values
        string = ' ' + string + ' '
        # Combine the ngrams back together
        ngrams = zip(*[string[i:] for i in range(self.n_gram)])
        return [''.join(ngram) for ngram in ngrams]
    
    def vectorize_strings(self, strings, typ="source"):
        if typ == "source":
            self.vectorizer = TfidfVectorizer(analyzer=self.create_ngrams)
            return self.vectorizer.fit_transform(strings)
        else:
            return self.vectorizer.transform(strings)

    def create_index(self, tf_idf_matrix):
        index = nmslib.init(
            method='simple_invindx', space='negdotprod_sparse_fast',
            data_type=nmslib.DataType.SPARSE_VECTOR)
        index.addDataPointBatch(tf_idf_matrix)
        index.createIndex(print_progress=True)
        self.index = index
    
    def query(self, k, return_df =True):
        query_matrix = self.index.knnQueryBatch(self.compare_tfidf_matrix, k=k, num_threads=self.n_jobs)
        compared_string, matched_strings, matched_conf = ([], [], [])
        
        # For each string in compare_strings, get available matches by decreasing confidence
        for i in range(len(query_matrix)):
            compared_string.append(np.repeat(i, len(query_matrix[i][0])))
            matched_strings.append(query_matrix[i][0])
            matched_conf.append(query_matrix[i][1])
        
        compared_string = np.hstack(compared_string)
        matched_strings = np.hstack(matched_strings)
        confidence      = np.hstack(matched_conf)

        if return_df:
            return (pd.DataFrame({
                "compared_string" : [self.compare[x] for x in compared_string],
                "matched_string"  : [self.source[x]  for x in matched_strings],
                "confidence"      : confidence})
                    .loc[lambda x: x.matched_string != x.compared_string])
        else:
            return sparse.coo_matrix((confidence, (compared_string, matched_strings))).tocsr()
