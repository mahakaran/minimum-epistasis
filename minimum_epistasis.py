import numpy as np
import itertools
import scipy
from scipy import sparse
from  tqdm.notebook import tqdm
from itertools import chain, combinations, product
from scipy.special import comb

"""
This is a Python implementation of the minimum epistasis interpolation method for sequence-fitness prediction by Zhou & McCandlish (2020)
(https://doi.org/10.1038/s41467-020-15512-5). There is an accompanying Jupyter Notebook that goes through the theory underlying the code here
(along with presenting the code).
"""


def hamming_circle(s, n, alphabet):
    """Generate strings over alphabet whose Hamming distance from s is
    exactly n. 

    (Function taken direct from StackExchange -- https://codereview.stackexchange.com/questions/88912/create-a-list-of-all-strings-within-hamming-distance-of-a-reference-string-with)

    >>> sorted(hamming_circle('abc', 0, 'abc'))
    ['abc']
    >>> sorted(hamming_circle('abc', 1, 'abc'))
    ['aac', 'aba', 'abb', 'acc', 'bbc', 'cbc']
    >>> sorted(hamming_circle('aaa', 2, 'ab'))
    ['abb', 'bab', 'bba']

    """
    for positions in combinations(range(len(s)), n):
        for replacements in product(range(len(alphabet) - 1), repeat=n):
            cousin = list(s)
            for p, r in zip(positions, replacements):
                if cousin[p] == alphabet[r]:
                    cousin[p] = alphabet[-1]
                else:
                    cousin[p] = alphabet[r]
            yield ''.join(cousin)




def all_genotypes(N, AAs):
    """Generates all possible genotypes of length N over alphabet AAs ."""
    return np.array(list(itertools.product(AAs, repeat=N)))

def custom_neighbors(sequence, sequence_space, d):
    """Search algorithm for finding sequences in sequence_space that are exactly Hamming distance d from sequence.
    This is a possibly a slow implementation -- it might be possible to obtain speed-ups."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming_distance(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==d]



def get_graph(sequenceSpace, AAs, a, l):
    """Get adjacency and degree matrices for a sequence space. This creates a Hamming graph by connecting all sequences 
    in sequence space that are 1 Hamming distance apart. Returns a sparse adjacency and degree matrix (which can be used
    for downstream applications e.g. Laplacian construction).

    sequenceSpace:      iterable of sequences
    returns:			tuple(adjacency matrix, degree matrix), where each matrix is a scipy sparse matrix """
  
    seq_space  = [''.join(list(i)) for i in sequenceSpace]
    members    = set(seq_space)
    nodes      = {x:y for x,y in zip(seq_space, range(len(seq_space)))}
    connect    = sparse.lil_matrix((len(seq_space), len(seq_space)), dtype='int8') 
    
    for ind in tqdm(range(len(sequenceSpace))):        
        seq = sequenceSpace[ind]     

        for neighbor in hamming_circle(seq, 1,AAs): 
            connect[ind,nodes[neighbor]]=1 

        degree_matrix = (l*(a-1))*sparse.eye(len(seq_space)) #this definition comes from Zhou & McCandlish 2020, pp. 11
    return connect, degree_matrix



def get_C_matrix(adjacency, degree, a, l): 
    """Gets the cost matrix C outlined in https://doi.org/10.1038/s41467-020-15512-5. C is obtained from the graph
    laplacian as outlined on page 11. NOTE: depending on the dataset, the matrices here can get very large, and
    can take up a lot of memory. There are ways to get around this, and this is a future direction for the code, 
    but at the moment, ensure you have enough RAM (~40 GB for a dataset of ~160,000 sequences). 

    adjacency:			scipy sparse CSC matrix 
    degree: 			scipy sparse CSC matrix
    a:					alphabet size (int)
    l:					positions(int)

    returns: 			cost matrix (scipy sparse CSC matrix)
    """    
    laplacian = degree - adjacency
    laplacian_sq = laplacian.dot(laplacian)
    s = comb(l,2)*(comb(a,2)**2)*(a**2)
    C = (1/(2*s))*(laplacian_sq-a*laplacian)

    return C



def minimum_epistasis_solve(C, f_in_sample, index, large=False): 
    """Performs minimum epistasis interpolation based in-sample fitness values and the graph cost matrix C.

    C:					cost matrix (scipy sparse CSC matrix)
    f_in_sample:		vector of in-sample fitness values (numpy array)
    index:				index denoting shift from in-sample to out-of-sample sequences in sequence space vector (i.e. dataset) (int)
    large:				if False, C_uu will be converted to numpy array before calculating inverse. If you expect C_uu to be very large, 
    					choose True, as this will perform inverse operation on sparse matrix (which is slower, but has lower memory usage)
    """
    #partition the cost matrix
    C_ll = C[:index,:index]
    C_lu = C[:index,index:]
    C_ul = C[index:,:index]
    C_uu = C[index:,index:]
    
    print('Computing C_uu inverse...')
    
    #calculate the inverse of C_uu
    if large: 
        C_uu_inv = scipy.sparse.linalg.inv(C_uu)
    else: 
        C_uu_inv = scipy.linalg.inv(C_uu.toarray()) #takes up more memory but is a lot faster. Choose if you have 
                                                    # enough memory
    print('C_uu inverse computed.')
    

    f_in_sample = f_in_sample.reshape(index,1) 

    pre_predict = C_ul.astype('float16').dot(f_in_sample.astype('float16'))
    
    if large: 
        predict = -1*C_uu_inv.dot(pre_predict)
    else: 
        predict = -1*np.dot(C_uu_inv, pre_predict)

    return predict   




