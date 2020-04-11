import h5py

embeddingDatasetName = 'embedding'

def persistEmbedding(path, embedding):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(embeddingDatasetName, data=embedding)
    h5f.close()

def getEmbeddingFromDisk(path):
    h5f = h5py.File(path, 'r')
    embedding = h5f[embeddingDatasetName][:]
    h5f.close()

    return embedding