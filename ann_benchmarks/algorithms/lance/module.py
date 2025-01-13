import lancedb
import pyarrow as pa

from ..base.module import BaseANN

class Lance(BaseANN):
    def vector_column_name():
        return 'vector'
    def table_name():
        return 'test_lance'
    def directory():
        return 'data/lance0'
    def __init__(self, metric, dim, args, index_type, quantization_type):
        self._metric_type = {'angular': 'cosine', 'euclidean': 'L2'}[metric]
        self._dim = dim
        self.db = lancedb.connect(uri=directory)
        self.index_type = index_type

    def create_table(self):
        schema = pa.schema([
            pa.field('id', pa.string()),
            pa.field(self.vector_column_name(), pa.list_(pa.float32(), list_size=self._dim)),
        ])
        self.table = self.db.create_table(self.table_name, schema=schema)
        print(f'Table {tablename} created.')

    def insert(self):
        for f in parquet_filelist:
            df = pd.read_parquet(f, engine='pyarrow')
            table.add(df)
            print(f'Loaded data from file {os.path.basename(f)}.')

        print(f'Data loading done.')

    def create_index(self):
        print(f'Creating index...')

        self.table.create_index(
            metric=self._metric_type
            vector_column_name=vector_column_name,
            index_type=self.index_type,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors
        )

        print(f'Index creation done.')

    def fit(self, X):
        self.create_table()
        self.insert()
        self.create_index()

    def query(self, v, n):
        return

    def done():
        return

class LanceIVF(Lance):
    def __init__(self, metric, dim, args):
        super().__init__(metric, dim, 'IVF_' + args.['quantization_type'])

class LanceHNSW(Lance):
    def __init__(self, metric, dim, args):
        super().__init__(metric, dim, 'IVF_HNSW_' + args.['quantization_type'])
