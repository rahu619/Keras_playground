import psycopg2
import pandas.io.sql as psql
class DbContext:
    conn = None

    def __init__(self, config_dict):
        db_dict = config_dict.get('DATABASE', {})
        self.conn = psycopg2.connect("host='{}' port={} dbname='{}' user={} password={}"
                       .format(db_dict['HOST'], db_dict['PORT'], db_dict['DB'], db_dict['USER'], db_dict['PASSWORD']))
        
    def get_claims(self):
        return self.create_dataframe("SELECT claim_text FROM public.claims ORDER BY id")

    def get_dependencies(self):
        """The dependency result will be in rugged format"""
        return self.create_dataframe("""SELECT dependency from public.patent_claim_dependencies AS pd 
                                   INNER JOIN claims as c
                                   ON pd.claim_id = c.id
                                   ORDER BY pd.claim_id
                                """)


    def create_dataframe(self, sql_query):
        table = psql.read_sql_query(sql_query, self.conn)
        return table

    def close_connection(self):
        self.conn = None