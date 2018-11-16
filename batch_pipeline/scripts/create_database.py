import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Numeric


if __name__ == '__main__':

    data = load_iris()
    engine = create_engine('sqlite:///../data/iris.db')

    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    df['date'] = 0

    df_append = df.copy()
    df_append['date'] = 1

    df = df.append(df_append)
    df = df.reset_index(drop=True).reset_index()

    for cl in range(len(data['target_names'])):
        meta = MetaData()
        table_pop = Table(data['target_names'][cl],
                          meta,
                          Column(df.columns[0], Integer, primary_key=True),
                          Column(df.columns[1], Numeric),
                          Column(df.columns[2], Numeric),
                          Column(df.columns[3], Numeric),
                          Column(df.columns[4], Numeric),
                          Column(df.columns[5], Integer),
                          Column(df.columns[6], Integer)
                         )
        meta.create_all(engine)

        df.to_sql(data['target_names'][cl],
                  engine,
                  if_exists='append',
                  index=False)

