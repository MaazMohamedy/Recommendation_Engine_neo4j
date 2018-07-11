import pandas as pd
import math
import numpy as np
import matplotlib as plt
from py2neo import Graph, Node, Relationship 
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import cProfile, pstats, io
import time
import profile
# from neo4j.v1 import GraphDatabase

graph = Graph(ip_addr = 'http://localhost:7474/browser/', username = 'neo4j', password = 'PracticeServer*')

def createDataFrame():
	# Loading data on the users
	user = pd.read_csv('ml-100k/u.user', sep='|', header=None, names=['id','age','gender','occupation','zip code'])
	n_u = user.shape[0]

	# Loading data on the genres
	genre = pd.read_csv('ml-100k/u.genre', sep='|', header=None, names=['name', 'id'])
	n_g = genre.shape[0]

	# Loading item-related data
	# Format : id | title | release date | | IMDb url | "genres"
	# where "genres" is a vector of size n_g : genres[i]=1 if the movie belongs to genre i
	movie_col = ['id', 'title','release date', 'useless', 'IMDb url']
	movie_col = movie_col + genre['id'].tolist()
	movie = pd.read_csv('ml-100k/u.item', encoding='cp1252', sep='|', header=None, names=movie_col)
	movie = movie.fillna('unknown')
	n_m = movie.shape[0]

	# Loading ratings
	rating_col = ['user_id', 'item_id','rating', 'timestamp']
	rating = pd.read_csv('ml-100k/u.data', sep='\t' ,header=None, names=rating_col)
	n_r = rating.shape[0]

	return

def buildGraph():
	tx = graph.begin()
	### Create User nodes
	graph.run("CREATE CONSTRAINT ON (u:USER) ASSERT u.id IS UNIQUE")
	for u in user['id']:
		id = str(u)
		graph.run("CREATE (:USER {id:"+id+"})")


	### Create Genre nodes, each one being identified by its id
	graph.run("CREATE CONSTRAINT ON (g:GENRE) ASSERT g.id IS UNIQUE")
	graph.run("CREATE CONSTRAINT ON (g:GENRE) ASSERT g.name IS UNIQUE")

	for g,row in genre.iterrows():
		genreName = str(row.iloc[0])
		id = row.iloc[1]
		graph.run("MERGE (g:GENRE {name:{name_}, id:{id_}}) RETURN g",name_ = genreName, id_ = str(id))


	graph.run("CREATE CONSTRAINT ON (m:MOVIE) ASSERT m.id IS UNIQUE")

	statement1 = "MERGE (m:MOVIE {id:{ID}, title:{TITLE}, url:{URL}}) RETURN m"

	statement2 = ("MATCH (t:GENRE{id:{D}}) "
	              "MATCH (a:MOVIE{id:{A}, title:{B}, url:{C}}) MERGE (a)-[r:Is_genre]->(t) RETURN r")


	for m,row in movie.iterrows():
	 	### Create "Movie" node, identified by id
		graph.run(statement1, {"ID": row.loc['id'], "TITLE":row.loc['title'], "URL":row.loc['IMDb url']})

		# is_genre : vector of size n_g, is_genre[i]=True if Movie m belongs to Genre i
		is_genre = row.iloc[-19:]==1
		itsGenres = genre[is_genre].axes[0].values

		# Looping over Genres g which satisfy the condition : is_genre[i]=True
		for g in itsGenres:
			# find node corresponding to genre g, and create relation between g and m
			graph.run(statement2, {"A": row.loc['id'], "B": row.loc['title'], "C": row.loc['IMDb url'], "D": str(g)})

	#ENDFOR

	# create rating relationship
	statement3 = ("MATCH (u:USER{id:{A}}) MATCH (m:MOVIE{id:{C}}) MERGE (u)-[r:Has_rated {rating:{B}}]->(m) RETURN r")

	# print( rating.loc[rating.user_id == 276, 'user_id'].count() )

	for r,row in rating.iterrows() :
		# Retrieve "User" and "Movie" nodes, and create relationship with the corresponding rating as property
	 	graph.run(statement3, {"A": int(row.loc['user_id']), "C": int(row.loc['item_id']), "B": int(row.loc['rating'])})

	return


def buildSimilarityMatrix():

	#### Ideas for Optimization:
	# if you come across a duplicate, for ex: (m1,m2) and (m2,m1) then look up this value and 

	WatchedBoth = ("MATCH (U:USER) WHERE (U:USER)-[:Has_rated]->(:MOVIE{id:{A}}) "
				"AND (U:USER)-[:Has_rated]->(:MOVIE{id:{B}})  RETURN U.id")

	findRating = "MATCH (USER {id:{user_id}})-[r:Has_rated]->(MOVIE{id:{movie_id}}) RETURN r.rating"

	numMovies = graph.evaluate("MATCH (m:MOVIE) RETURN COUNT(m)")

	m1_ratings = []
	m2_ratings = []
	angle_in_degrees = 0
	Row = []
	matrix = []

	pr = cProfile.Profile()


	for m1 in range(1,3):#numMovies+1):
		pr.enable()
		for m2 in range(1, numMovies+1):

			if m2 < m1:
				angle_in_degrees = matrix[m2-1][m1-1]

			# on diagonal
			if m1 == m2: angle_in_degrees = 0

			else:
				# Find 'users' who've watched both m1 and m2
				users = graph.run(WatchedBoth, {"A": m1, "B":m2}).data()#[0]['U.id']

				if len(users) == 0:
					angle_in_degrees = 90

				else:

					# create arrays of m1's and m2's ratings
					for u in users:
						m1rating = graph.evaluate(findRating, {"user_id": u['U.id'], "movie_id": m1} )
						m1_ratings.append(m1rating)

						m2rating = graph.evaluate(findRating, {"user_id": u['U.id'], "movie_id": m2} )
						m2_ratings.append(m2rating)


					# create vector v1 andv2
					v1 = np.array(m1_ratings).reshape(1,-1)
					v2 = np.array(m2_ratings).reshape(1,-1)

					# calculate cosine similarity
					similarity = cosine_similarity(v1, v2)
					similarity = np.clip(similarity, -1, 1)
					angle_in_radians = math.acos(similarity)
					angle_in_degrees = math.degrees(angle_in_radians)


					m1_ratings = []
					m2_ratings = []
					users = []
			
			Row.append(angle_in_degrees)
		pr.disable()
		pr.print_stats()
		matrix.append(Row)
		Row =[]

	
	df = pd.DataFrame(matrix)
	
	print(df)
		
	return

def main():

	pr = cProfile.Profile()
	pr.enable()

	# Create data frames from dsv files
	createDataFrame()

	# Build graph of 3 nodes (user, genre, movie) and two edges (Has_rated, Is_genre)
	buildGraph()

	# Build our Item-Item matrix of similarities between each item
	buildSimilarityMatrix() 

	pr.disable()
	pr.print_stats()


if __name__ == "__main__":
    main()

