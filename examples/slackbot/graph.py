import marvin
from neo4j import GraphDatabase
from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    type: str
    properties: dict

    def __hash__(self) -> int:
        return hash(self.name)


class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    def create_entity(self, entity_name: str, entity_type: str, properties: dict):
        with self.driver.session() as session:
            query = f"CREATE (e:{entity_type} {{name: '{entity_name}', "
            query += ", ".join([f"{k}: '{v}'" for k, v in properties.items()])
            query += "}})"

            session.run(query)  # type: ignore

    def query_entity(self, entity_name: str):
        with self.driver.session() as session:
            query = f"MATCH (e {{name: '{entity_name}'}}) RETURN e"
            result = session.run(query)  # type: ignore
            return result.single()


neo4j_conn = Neo4jConnection(uri="bolt://neo4j:7687", user="neo4j", pwd="testtest")


def extract_and_store_entities(text: str) -> dict[str, str]:
    for entity in marvin.extract(text, target=Entity):
        neo4j_conn.create_entity(entity.name, entity.type, entity.properties)

    return {"status": "Entities added to knowledge graph"}
