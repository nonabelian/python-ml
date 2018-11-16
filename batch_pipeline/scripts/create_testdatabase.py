import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from sqlalchemy.sql import select


if __name__ == '__main__':

    source_engine = create_engine('sqlite:///../data/iris.db')
    Session = sessionmaker(bind=source_engine)
    source_session = Session()

    source_meta = MetaData(bind=source_engine)
    source_meta.reflect(source_engine)
    print('Tables: ', source_meta.tables.keys())

    dest_engine = create_engine('sqlite:///../tests/data/test_iris.db')
    Session = sessionmaker(bind=dest_engine)
    dest_session = Session()

    for table in source_meta.sorted_tables:
        table.metadata.create_all(dest_engine)

        Base = declarative_base()
        class NewRecord(Base):
            __table__ = table

        columns = table.columns.keys()

        # Straight-up copying table data up to 5 rows
        # print('Copying table: ', table.name)
        # for i, record in enumerate(source_session.query(table).all()):
        #     data = dict(
        #         [(str(column), getattr(record, column))
        #          for column in columns]
        #     )
        #     dest_session.merge(NewRecord(**data))
        #     if i > 5:
        #         break

    # And finalize
    dest_session.commit()

    dest_session.close()
    source_session.close()

    # Or we could individually append data from pandas, mocking up the data
    for table in source_meta.sorted_tables:
        query = select([table]).limit(5)

        # Get the data
        df = pd.read_sql(query, source_engine)

        # Set for mocking
        df['date'] = np.random.choice([0, 1], size=5)

        df.to_sql(table.name, dest_engine, if_exists='append', index=False)
