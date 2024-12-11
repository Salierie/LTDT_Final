from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
        self.session = self.driver.session()

    def close(self):
        self.session.close()

    def query(self, query, parameters=None):
        result = self.session.run(query, parameters or {})
        return result