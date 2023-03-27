# Code refactor - from notebooks to productize pipelines

This working example contains two Jupyter notebooks with data preprocessing and
training routines. The main goal of the fefactoring I did was to convert those notebooks
into pipelines that can be scheduled by using any scheduling tool (Airflow for example).
The approach I follow was to create two small pipelines (one for data processing and the other
for model training) that can be integrated as two tasks of a single DAG that can be used
for scheduled retraining. Some technical notes about the implementation:
1. In both notebooks is very clear the steps done by the Data Science team to create the model.
Because of that, I did a deep dive into the notebooks and extracted the main functionalities,
and wrote them as functions. For each one of those functions, I put in place the corresponding
unit tests.
2. Since I want pipelines independent from the data source (in this case, I have as a data source
a CSV file, but in a production environment the source can be an S3 bucket or a BigQuery database),
I used it as input for the two task sources and sink callables. These callables contain perform
some operations (in this case, write or read CSV/Parquet/JSON files) to load or save data. The
main goal of this implementation is to make it easy to exchange sources/sinks (for example, in the case
of a migration from one storage service to another, for this model the migrations are as easy as writing a new
function for the new storage service).
3. I made some small changes to the code made by the DS team in order to improve some functionalities. For
example, in the has_amenity() function, I used a different approach to find the amenities in the string
because the DS team implementation is sensible to the uppercase-lowercase differences. I also saved
the processed data as a parquet file to persist also the data schema.

# API implementation

Now I have a model trained, is time to create an API to consume it. I used Flask to develop it.
This API is able to transform the input payload into a Numpy array that can be used as input by the
trained model and return the outcome with the desired format. To do this I developed a few functions.
All these functions have the proper unit testing.

# Dockerize the solution

Now I have a working API, it's time to dockerize it. First of all, I created a requirements.txt by doing
a *pip freeze* of the virtual environment I created for this task. After that, I wrote the dockerfile
that performs the following tasks:
1. Installs Python (3.7; the version I used in my virtual environment), pip, and setuptools. 
2. I set up my directory, and I copied there the source code, the model artifact, and the requirements.txt.
3. Installs the libraries in the requirements.txt.
4. Runs the API script.

# Code refactor - from notebooks to productize pipelines

This working example contains two Jupyter notebooks with data preprocessing and
training routines. The main goal of the refactor I did was to convert those notebooks
into pipelines that can be scheduled by using any scheduling tool (Airflow for example).
The approach I follow was to create two small pipelines (one for data processing and the other
for model training) that can be integrated as two tasks of a single DAG that can be used
for scheduled retraining. Some technical notes about the implementation:
1. In both notebooks is very clear the steps done by the Data Science team to create the model.
Because of that, I did a deep dive into the notebooks and extracted the main functionalities, 
and wrote them as functions. For each one of those functions, I put in place the corresponding
unit tests.
2. Since I want pipelines independent from the data source (in this case, I have as a data source
a CSV file, but in a production environment the source can be an S3 bucket or a BigQuery database),
I used it as input for the two task sources and sink callables. These callables contain perform
some operations (in this case, write or read CSV/Parquet/JSON files) to load or save data. The
main goal of this implementation is to make it easy to exchange sources/sinks (for example, in the case
of a migration from one storage service to another, for this model the migrations are as easy as writing a new
function for the new storage service).
3. I made some small changes to the code made by the DS team in order to improve some functionalities. For
example, in the has_amenity() function, I used a different approach to find the amenities in the string
because the DS team implementation is sensible to the uppercase-lowercase differences. I also saved
the processed data as a parquet file to persist also the data schema.

# API implementation

Now I have a model trained, is time to create an API to consume it. I used Flask to develop it.
This API is able to transform the input payload into a Numpy array that can be used as input by the
trained model and return the outcome with the desired format. To do this I developed a few functions.
All these functions have the proper unit testing.

# Dockerize the solution

Now I have a working API, it's time to dockerize it. First of all, I created a requirements.txt by doing
a *pip freeze* of the virtual environment I created for this task. After that, I wrote the dockerfile
that performs the following tasks:
1. Installs Python (3.7; the version I used in my virtual environment), pip, and setuptools. 
2. Set up the working directory, and I copied there the source code, the model artifact,
and the requirements.txt.
3. Installs the libraries in the requirements.txt.
4. Runs the API script.
