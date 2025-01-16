import lancedb
import pandas as pd
import pyarrow as pa

from ..base.module import BaseANN

class Lance(BaseANN):
    def __init__(self, metric, index_type):
        self._metric_type = {'angular': 'cosine', 'euclidean': 'L2'}[metric]
        self.db = lancedb.connect(uri='.')
        self.index_type = index_type
        self.name = type(self).__name__

    def fit(self, X):
        dim = len(X[0])
        self.id_field = pa.field('id', pa.int32())
        vector_field = pa.field('vector', pa.list_(pa.float32(), list_size=dim))
        schema = pa.schema([self.id_field, vector_field])
        self.table = self.db.create_table(
            'test_lance',
            data = pa.Table.from_pandas(
                df = pd.DataFrame(
                    data = list(enumerate(X)),
                    columns = schema.names,
                    copy = False,
                    ),
                ),
            schema = schema,
            )
        print(f'Table {self.table.name} created.')
        print(f'Creating index...')
        self.table.create_index(
            metric = self._metric_type,
            vector_column_name = vector_field.name,
            index_type = self.index_type,
#            num_partitions = num_partitions,
#            num_sub_vectors = num_sub_vectors,
            )
        print(f'Index creation done.')

    def query(self, v, n):
        return self.table.search(v).limit(n).to_arrow().column(self.id_field.name).to_pylist()

    @staticmethod
    def done():
        return

class LanceIVF(Lance):
    def __init__(self, metric, args):
        super().__init__(metric, 'IVF_' + args['quantization'])

class LanceHNSW(Lance):
    def __init__(self, metric, args):
        super().__init__(metric, 'IVF_HNSW_' + args['quantization'])
