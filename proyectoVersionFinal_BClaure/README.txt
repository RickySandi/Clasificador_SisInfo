(Para ejecutar directamente con el clasificador y diccionario ya creado en la carpeta "Clasificador")
1.Entrar a la carpeta Clasificador y levantar el servidor con el comando "python serverClasificador.py"
2.Ejecutar el ETL "clasificarComentarios.ktr" que  generara un archivo "comentariosClasificados.txt"
3.Tambien puede ejecutar todo el notebook en orden descendente de "PruebasRest.ipynb" para 100 ejemplos aleatorios con respuesta y porcentaje de acierto
(Para crear nuevo clasificador y nuevo vocabulario)
1.Borrar de la carpeta clasificador el archivo "tree_v4.pkl" y "VocabularioProblema.pkl"
2.Entrar al notebook "Clasificador version final.ipynb" hacer correr todo el notebook en orden descendente
3.Entrar a la carpeta Clasificador y levantar el servidor con el comando "python serverClasificador.py"
4.Tambien puede ejecutar todo el notebook en orden descendente de "PruebasRest.ipynb" para 100 ejemplos aleatorios con respuesta y porcentaje acierto
5.Ejecutar el ETL "clasificarComentarios.ktr" que  generara un archivo "comentariosClasificados.txt"

Notas: La carpeta "versionesDePrueba" contiene los notebooks con algunos de los intentos que hicimos.
	Pruebas.csv contiene 5 comentarios aleatorios.
	PruebasRespuestas.txt contiene las respuestas de los 5 comentarios sin utilizar el clasificador.