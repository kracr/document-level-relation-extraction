from wikimapper import WikiMapper
from wikimapper import WikiMapper

from SPARQLWrapper import SPARQLWrapper,JSON


import requests

import requests


def remove(string):
    return string.replace(" ", "")


def find_property_label(property_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={property_id}&format=json&props=labels"
    response = requests.get(url)
    data = response.json()

    # Check if the property ID exists in the response
    if property_id in data["entities"]:
        entity = data["entities"][property_id]
        if "labels" in entity:
            labels = entity["labels"]
            if "en" in labels:
                return labels["en"]["value"]

    # Return None if the property ID or label is not found
    return property_id

def sparql_query(entity_id1,entity_id2):
    # Create a SPARQLWrapper object and set the endpoint
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Define the SPARQL query
    query = f"""
    SELECT ?property WHERE {{
      wd:{entity_id1} ?property wd:{entity_id2}.
    }}
    LIMIT 1
    """

    # Set the query and response format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the SPARQL query and retrieve the results
    results = sparql.query().convert()

    # Check if there are any results
    if 'results' in results and 'bindings' in results['results']:
        bindings = results['results']['bindings']
        if len(bindings) > 0 and 'property' in bindings[0]:
            property_id = bindings[0]['property']['value'].rsplit('/', 1)[-1]
            return find_property_label(property_id)

    # If no property exists, return None
    return None


def onehop(entity_id1,entity_id2):
    # Create a SPARQLWrapper object and set the endpoint
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Define the SPARQL query
    query = f"""
    SELECT ?property  ?s ?p1  WHERE {{
      wd:{entity_id1} ?property  ?s. ?s  ?p1 wd:{entity_id2}.
    }}
    LIMIT 1
    """

    # Set the query and response format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the SPARQL query and retrieve the results
    results = sparql.query().convert()

    # Check if there are any results
    if 'results' in results and 'bindings' in results['results']:
        bindings = results['results']['bindings']
        if len(bindings) > 0 and 'property' in bindings[0]:
            property_id = bindings[0]['property']['value'].rsplit('/', 1)[-1]
            p1 = bindings[0]['p1']['value'].rsplit('/', 1)[-1]
            s = bindings[0]['s']['value'].rsplit('/', 1)[-1]
            return find_property_label(property_id),s,p1

    # If no property exists, return None
    return None



def find_onehoprelations(entity1, entity2):
    # Set up the SPARQL endpoint
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    query = """
    SELECT DISTINCT ?relation1 ?relation2 ?z
    WHERE {{
    wd:{entity1_id} ?relation1   ?z.
    ?z  ?relation2 wd:{entity2_id}
    }} LIMIT 1
    """.format(entity1_id=entity1, entity2_id=entity2)

    #print(query)
    sparql.setQuery(query)

    # Send the SPARQL query and retrieve the results
    results = sparql.query().convert()

    # Extract the relation labels from the query results
    relation_labels1 = [result["relation1"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    relation_labels2 = [result["relation2"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    z = [result["z"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]

    relation_labels1=str(relation_labels1).strip("[]''")
    relation_labels2=str(relation_labels2).strip("[]''")
    z=str(z).strip("[]''")
    #z=mapper.title_to_id(z)
    #print(find_property_label(relation_labels1),find_property_label(relation_labels2),z)


    #relation_labels = relation_labels.rsplit('/', 1)[-1]
    #pred_labels=[result["pred"]["value"] for result in results["results"]["bindings"]]
    #print(i["sub"],i["obj"])

    #try:
    # relation_labels=find_relations(i["sub"],i["obj"])

    # #print("The relations between  are:,",i["sub"],i["obj"],relation_labels)
    #except:
    # pass
    return relation_labels1,relation_labels2,z


def find_twohoprelations(entity1, entity2):
    # Set up the SPARQL endpoint
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    query = """
    SELECT DISTINCT ?relation1 ?z ?relation2 ?e ?relation3
    WHERE {{
    wd:{entity1_id} ?relation1   ?z.
    ?z  ?relation2   ?e.
    ?e   ?relation3  wd:{entity2_id}
    }} LIMIT 1
    """.format(entity1_id=entity1, entity2_id=entity2)

    #print(query)
    sparql.setQuery(query)

    # Send the SPARQL query and retrieve the results
    results = sparql.query().convert()

    # Extract the relation labels from the query results
    relation_labels1 = [result["relation1"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    relation_labels2 = [result["relation2"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    z = [result["z"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    e = [result["e"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    relation_labels3 = [result["relation3"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    if relation_labels1 is not None and z is not None and relation_labels2 is not None and e is not None and relation_labels3 is not None:
     relation_labels1=str(relation_labels1).strip("[]''")
     relation_labels2=str(relation_labels2).strip("[]''")
     z=str(z).strip("[]''")
     e=str(e).strip("[]''")
     relation_labels3=str(relation_labels3).strip("[]''")    
 
     #print(relation_labels1,z,relation_labels2,e,relation_labels3)


     #relation_labels = relation_labels.rsplit('/', 1)[-1]
     #pred_labels=[result["pred"]["value"] for result in results["results"]["bindings"]]
     #print(i["sub"],i["obj"])

     #try:
     # relation_labels=find_relations(i["sub"],i["obj"])

     # #print("The relations between  are:,",i["sub"],i["obj"],relation_labels)
     #except:
     # pass
     return relation_labels1,z,relation_labels2,e,relation_labels3
    




def find_threehoprelations(entity1, entity2):
    # Set up the SPARQL endpoint
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    query = """
    SELECT DISTINCT ?relation1 ?z ?relation2 ?e ?relation3 ?f ?relation4
    WHERE {{
    wd:{entity1_id} ?relation1   ?z.
    ?z  ?relation2   ?e.
    ?e   ?relation3   ?f.
    ?f   ?relation4 wd:{entity2_id}.
    }} LIMIT 1
    """.format(entity1_id=entity1, entity2_id=entity2)

    #print(query)
    sparql.setQuery(query)

    # Send the SPARQL query and retrieve the results
    results = sparql.query().convert()

    # Extract the relation labels from the query results
    relation_labels1 = [result["relation1"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    relation_labels2 = [result["relation2"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    relation_labels3 = [result["relation3"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    relation_labels4 = [result["relation3"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]

    z = [result["z"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    e = [result["e"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]
    f = [result["f"]["value"].rsplit('/', 1)[-1] for result in results["results"]["bindings"]]

    if relation_labels1 is not None and z is not None and relation_labels2 is not None and e is not None and relation_labels3 is not None and f is not None:
     relation_labels1=str(relation_labels1).strip("[]''")
     relation_labels2=str(relation_labels2).strip("[]''")
     relation_labels3=str(relation_labels3).strip("[]''")
     relation_labels4=str(relation_labels4).strip("[]''")

     z=str(z).strip("[]''")
     e=str(e).strip("[]''")
     f=str(f).strip("[]''")

     #print(relation_labels1,z,relation_labels2,e,relation_labels3,f,relation_labels4)

    return relation_labels1,z, relation_labels2,e, relation_labels3,f,relation_labels4


with open("entities","r") as file:
    for line in file:
         en1,en2=line.split("\t")

         mapper = WikiMapper("index_enwiki-latest.db")

         wikidata_id1 = mapper.title_to_id(str(en1))
         #print(wikidata_id1) 
         wikidata_id2 = mapper.title_to_id1(str(en2))
         #print(title,en1,prel,en2,ac_rel)
         print(wikidata_id1,wikidata_id2)

         #print(wikidata_id2) 
         if(wikidata_id1 is not None and wikidata_id2 is not None):

              property_id = sparql_query(wikidata_id1,wikidata_id2)
              if property_id:
                   
                   
                   with open("onehop.txt","a+") as g:          
                    g.write(str(remove(en1))+"\t"+str(remove(property_id))+"\t"+str(remove(en2))+"\n")
                    #print(f"One for entity {wikidata_id1} {wikidata_id2} is:",property_id)
              else:
                         
                    #print(f"No property  exists for entity {wikidata_id1}{wikidata_id2}.")
                      relation_labels1,relation_labels2,z=find_onehoprelations(wikidata_id1,wikidata_id2)
                      if(relation_labels1 and relation_labels2 and z):
                        
                         with open("twohop.txt","a+") as g:

                           g.write(str(en1)+"\t"+str(relation_labels)+"\t"+str(z)+"\n"+str(z)+"\t"+str(relation_labels2)+"\t"+str(en2)+"\n")
                         print("two hop relations between",en1,"and",en2,"is:",relation_labels1,relation_labels2)
                      else: 
                       
                      
                       relation_labels1,z,relation_labels2,e,relation_labels3=find_twohoprelations(wikidata_id1,wikidata_id2)
                       if(relation_labels1 and z and relation_labels2 and e and relation_labels3):
                           
                         with open("threehop.txt","a+") as g:
                           g.write(str(en1)+"\t"+str(find_property_label(relation_labels1))+"\t"+str(z)+"\n"+str(z)+"\t"+str(find_property_label(relation_labels2))+"\t"+str(e)+"\n"+str(e)+"\t"+str(find_property_label(relation_labels3))+"\t"+str(en2)+"\n")
                       else:
                         relation_labels1,z,relation_labels2,e,relation_labels3,f,relation_labels4=find_threehoprelations(wikidata_id1,wikidata_id2)
                         if(relation_labels1 and z and relation_labels2 and e and relation_labels3 and f and relation_labels4):
                           with open("fourhop.txt","a+") as g:
                            g.write("FOURHOP:"+"\t"+str((mapper.id_to_titles(en1)))+"\t"+str((find_property_label(relation_labels1)))+"\t"+str(mapper.id_to_titles((z)))+"\n"+str((mapper.id_to_titles(z)))+"\t"+str((find_property_label(relation_labels2)))+"\t"+str((e))+"\n"+str((mapper.id_to_titles(e)))+"\t"+str((find_property_label(relation_labels3)))+"\t"+str((mapper.id_to_titles(f)))+"\n"+str((mapper.id_to_titles(f)))+"\t"+str((find_property_label(relation_labels4)))+"\t"+str((mapper.id_to_titles(en2)))+"\n")
                         #print("The four hop relations between",en1,"and",en2,"is:",relation_labels1,relation_labels2,relation_labels3,relation_labels4)
                    




