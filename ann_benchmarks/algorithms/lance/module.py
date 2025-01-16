import lancedb
import pandas as pd
import pyarrow as pa

from ..base.module import BaseANN

class Lance(BaseANN):
    @staticmethod
    def vector_column_name():
        return 'vector'
    @staticmethod
    def table_name():
        return 'test_lance'
    @staticmethod
    def directory():
        return '.'#data/my-lance'
    def __init__(self, metric, index_type):
        self._metric_type = {'angular': 'cosine', 'euclidean': 'L2'}[metric]
        self.db = lancedb.connect(uri=self.directory())
        self.index_type = index_type

    def create_table(self, dim):
        schema = pa.schema([
            pa.field('id', pa.string()),
            pa.field(self.vector_column_name(), pa.list_(pa.float32(), list_size=dim)),
        ])
        self.table = self.db.create_table(self.table_name(), schema=schema)
        print(f'Table {self.table.name} created.')

    def insert(self, X):
        self.table.add(X)
        print(f'Data loading done.')

    def create_index(self):
        print(f'Creating index...')
        self.table.create_index(
            metric=self._metric_type,
            vector_column_name=self.vector_column_name(),
            index_type=self.index_type,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
        print(f'Index creation done.')

    def make_batches(X):
        batch_size = 1000
        for i in range(i, len(X), batch_size):
            yield pa.RecordBatch.from_arrays()
           

    def fit(self, X):
        print(type(X))
        print(X.dtype)
        print(X.ndim)
        dim = len(X[0])
        print(dim)
        print(len(X))
        #x = X.tolist();print(len(x))
        #schema = pa.schema([
            #pa.field('id', pa.string()),
        #    pa.field(self.vector_column_name(), pa.list_(pa.float32(), list_size=dim)),
        #])
            
        #x = pd.DataFrame(X, columns=[self.vector_column_name()] 
        #x = pa.Table.from_pandas(df=X, schema=schema, preserve_index=True);print(len(x))
        #x = pa.Table.from_arrays(pa.array(X), schema=schema);print(len(x))
        x = list(enumerate(X))
        print(len(x))
        print(len(x[0]))
        print(len(x[0][1]))
        schema = pa.schema([
            pa.field('id', pa.int32()),
            pa.field(self.vector_column_name(), pa.list_(pa.float32(), list_size=dim)),
        ])
        #df = pd.DataFrame(enumerate(X))
        df = pd.DataFrame(x, columns=schema.names, copy=False)
        print(len(df))
        print(df.columns)
        print(df.index)
        print(df.dtypes)
        print(df.info)
            #names=['id', self.vector_column_name()])
        #print(type(enumerate(X)))
        #print(type(list(enumerate(X))[0]))

        #x = pa.Table.from_arrays(arrays=[enumerate(X)], schema=schema)
        #self.table = self.db.create_table(self.table_name(), data=X, schema=schema)
        #self.table = self.db.create_table(self.table_name(), data=list(enumerate(X)), schema=schema)
        #self.table = self.db.create_table(self.table_name(), data=x)
        self.table = self.db.create_table(self.table_name(), data=pa.Table.from_pandas(df=df), schema=schema)
        #self.table = self.db.create_table(self.table_name(), data=X.tolist())
        #self.table = self.db.create_table(self.table_name(), data=make_batches(), schema=schema)
        print(f'Table {self.table.name} created.')
        #self.create_table()
        #self.insert(X)
        self.create_index()

    def query(self, v, n):
        return

    @staticmethod
    def done():
        return

class LanceIVF(Lance):
    def __init__(self, metric, args):
        super().__init__(metric, 'IVF_' + args['quantization'])

class LanceHNSW(Lance):
    def __init__(self, metric, args):
        super().__init__(metric, 'IVF_HNSW_' + args['quantization'])
