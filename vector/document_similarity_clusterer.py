import numpy as np
import faiss  # type: ignore

class DocumentSimilarityClusterer:
    """
    A class that clusters documents based on their vector embeddings using K-Means and FAISS for efficient similarity search.
    """

    @staticmethod
    def _convert_embeddings_to_faiss_compatible_matrix(embeddings):
        return np.array(embeddings).astype("float32")

    @staticmethod
    def _get_kmeans_clustering_model(embedding_dimension, num_clusters):
        return faiss.Kmeans(d=embedding_dimension, k=num_clusters, niter=20, verbose=True)

    @staticmethod
    def _get_euclidean_distance_search_index(embedding_matrix, embedding_dimension):
        index = faiss.IndexFlatL2(embedding_dimension)        
        index.add(embedding_matrix)

        return index

    @staticmethod
    def _sort_documents_by_cluster_assignment(docs, model, embedding_matrix, embedding_dimension):
        centroids = model.centroids

        search_index = DocumentSimilarityClusterer._get_euclidean_distance_search_index(embedding_matrix, embedding_dimension)

        _, nearest_docs_indices = search_index.search(centroids, 1)

        # Sort the document indices by cluster assignment
        sorted_indices = np.sort(nearest_docs_indices, axis=0).flatten()

        # Return the documents in the sorted cluster order
        return [docs[i] for i in sorted_indices]

    @staticmethod
    def cluster_documents(docs, embeddings, num_cluster_multiplier = 0.5):
        """
        Clusters documents based on their embeddings using K-Means algorithm and returns documents in sorted cluster order.

        Parameters:
        docs (list): A list of documents to be clustered.
        embeddings (list of list): A list of vector embeddings corresponding to the documents.
        num_cluster_multiplier (float): A float between 0 and 1 that determines the number of clusters as a fraction of the number of documents.

        Returns:
        list: A list of documents sorted by their cluster assignment.
        """
        
        embedding_matrix = DocumentSimilarityClusterer._convert_embeddings_to_faiss_compatible_matrix(embeddings)
        
        num_documents = len(docs)
        num_clusters = int(num_documents * num_cluster_multiplier)

        # Get the dimension of the embeddings (assuming all embeddings are of the same dimension)
        embedding_dimension = embedding_matrix.shape[1]

        kmeans_model = DocumentSimilarityClusterer._get_kmeans_clustering_model(embedding_dimension, num_clusters)
        
        kmeans_model.train(embedding_matrix)
        
        return DocumentSimilarityClusterer._sort_documents_by_cluster_assignment(docs, kmeans_model, embedding_matrix, embedding_dimension)
        
        
        
        
        
        
