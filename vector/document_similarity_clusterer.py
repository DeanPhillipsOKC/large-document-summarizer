import numpy as np
import faiss # type: ignore

class DocumentSimilarityClusterer:

    @staticmethod
    def cluster_documents(docs, embeddings):
        vectors = embeddings
        array = np.array(vectors).astype("float32")
        
        num_chunks = len(docs)
        num_clusters = num_chunks / 2

        dimension = array.shape[1]
        kmeans = faiss.Kmeans(dimension, num_clusters, niter=20, verbose=True)
        kmeans.train(array)
        centroids = kmeans.centroids
        index = faiss.IndexFlatL2(dimension)
        index.add(array)

        D, I = index.search(centroids, 1)

        sorted_array = np.sort(I, axis=0)
        sorted_array=sorted_array.flatten()
        return [docs[i] for i in sorted_array]